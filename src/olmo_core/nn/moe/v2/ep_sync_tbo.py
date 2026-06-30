"""
Synchronous (count-synchronized) expert-parallel forward with two-batch overlap (TBO).

``combined_forward_ep_tbo`` runs two micro-batches through the synchronous expert-parallel pipeline
with their stages interleaved so one batch's expert-parallel all-to-all overlaps the other's
attention/router compute. It shares the per-batch permute/dispatch/combine machinery with
:mod:`~olmo_core.nn.moe.v2.ep_sync_1d`. Batch 0 ("TBO-0") is finished within the call; batch 1
("TBO-1") is left mid-flight and returned in a :class:`SyncedTboPendingContext` for the next block
(or the model's final TBO step) to complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.distributed as dist

from olmo_core.ops import moe as ops

from ...moe.utils import async_copy_to_cpu
from ..utils import moe_permute_no_compile, moe_unpermute_no_compile
from ._nvtx import annotate
from .ep_sync_1d import checkpointed_permute_routed_experts_unpermute_1d
from .routed_experts import requires_host_side_split_sizes
from .tbo_state import SyncedTboPendingContext

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def combined_forward_ep_tbo(
    block: MoEFusedV2TransformerBlock,
    x0: torch.Tensor,
    x1_ctx: object,
    x1_is_fresh: bool,
    *,
    loss_div_factor: Optional[torch.Tensor | float] = None,
    **kwargs,
) -> tuple[torch.Tensor, object]:
    """Synchronous expert-parallel forward with two-batch overlap (see module docstring)."""
    self = block

    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None

    B, S, D = x0.shape

    # rename "x" to avoid confusion
    block_inp = x0
    del x0

    with torch.no_grad():
        # Construct the expert indices for the permuted tokens.
        global_x_local_expert_indices_0 = torch.remainder(
            torch.arange(
                self.routed_experts.num_experts,
                dtype=torch.int32,
                device=block_inp.device,
            ),
            self.num_local_routed_experts,
        )  # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts

    with annotate("tbo_1", "tbo"):
        if x1_is_fresh:
            local_x1, local_x_handle1 = None, None
            last_block = None
        else:
            if not isinstance(x1_ctx, SyncedTboPendingContext):
                raise RuntimeError(
                    "Expected synced TBO context from previous block, " f"got type={type(x1_ctx)}"
                )
            global_x1 = x1_ctx.global_x
            send_counts1 = x1_ctx.send_counts
            recv_counts1 = x1_ctx.recv_counts
            last_block = x1_ctx.last_block

            assert last_block.routed_experts_router is not None
            # finish reverse all2all and other ops for x1
            with annotate("reverse_all_to_all", "comm"):
                global_x1 = cast(torch.Tensor, global_x1)
                global_x1, local_x1, local_x_handle1 = ops.all_to_all_async(
                    global_x1,
                    send_counts1,
                    recv_counts1,
                    group=last_block.ep_pg,
                )

    with annotate("tbo_0", "tbo"):
        # attention
        # + attention norm
        # + residual connection
        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)
        moe_inp = self._prepare_moe_input(attn_res_out)
        # routed expert router
        (
            local_x_global_routed_expert_weights,  # (B, S, top_k)
            local_x_global_routed_expert_indices,  # (B, S, top_k)
            local_batch_size_per_global_routed_expert,  # (num_experts, )
            routed_expert_router_aux_loss_info,
        ) = self.routed_experts_router(
            moe_inp, False, loss_div_factor=loss_div_factor  # scalar
        )

        attn_res_out = self._attach_routed_aux_loss(
            attn_res_out,
            routed_expert_router_aux_loss_info,
        )

        # 1. Communicate the number of tokens that will be sent to each device
        with annotate("token_count_all_to_all", "comm"):
            with torch.no_grad():
                # Pass token count information to the device on which the
                # target expert resides.
                global_batch_size_per_local_expert = torch.empty_like(
                    local_batch_size_per_global_routed_expert,
                )
                global_batch_size_handle = dist.all_to_all_single(
                    global_batch_size_per_local_expert,  # Gathered concatenated output tensor.
                    local_batch_size_per_global_routed_expert,  # Input tensor to scatter.
                    group=self.ep_pg,
                    async_op=True,
                )

                assert global_batch_size_handle is not None  # because of async

        if self.shared_experts_router:
            # shared expert router
            (
                local_x_global_shared_expert_weights,  # (B, S, E_shared)
                _,
                _,
                _,
            ) = self.shared_experts_router(
                moe_inp,
                True,  # only need scores for shared experts
                loss_div_factor=loss_div_factor,  # scalar
            )
        else:
            local_x_global_shared_expert_weights = None

        # forward shared experts
        if self.shared_experts is not None:
            shared_out = self.shared_experts.forward(moe_inp)
            with annotate("merge_shared", "experts"):
                if self.shared_experts_router:
                    assert local_x_global_shared_expert_weights is not None
                    # weighted sum of the shared experts by router weights
                    # local_x_global_shared_expert_weights -> (B, S, E_shared)
                    # shared_out -> (E_shared, B, S, D)
                    _, _, E_s = local_x_global_shared_expert_weights.shape
                    mixed_shared_out = (
                        torch.bmm(
                            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(
                                B * S, 1, E_s
                            ),  # (BS, 1, E),
                            shared_out.permute(1, 2, 0, 3)
                            .contiguous()
                            .view(B * S, E_s, D),  # (BS, E, D)
                        )
                        .squeeze(1)
                        .view(B, S, D)
                    )
                else:
                    mixed_shared_out = shared_out.squeeze(0)
        else:
            mixed_shared_out = None

        in_shape = moe_inp.size()

        moe_inp = moe_inp.view(-1, in_shape[-1])  # (B*S, D)

        # 3. Configure the sizes for grouped GEMM

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with annotate("sync_token_count", "comm"):
            with torch.no_grad():
                global_batch_size_handle.wait()

                # Reshape to (ep_world_size, num_local_routed_experts).
                local_batch_size_per_global_routed_expert = (
                    local_batch_size_per_global_routed_expert.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                )
                global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
                # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                parallel_batch_size_per_local_expert = global_batch_size_per_local_expert.sum(
                    dim=0,
                    dtype=torch.long,
                )

                send_counts_gpu = local_batch_size_per_global_routed_expert.sum(dim=-1)
                recv_counts_gpu = global_batch_size_per_local_expert.sum(dim=-1)
                send_counts_cpu, _, dtoh_event_send = async_copy_to_cpu(
                    send_counts_gpu, event=self._dtoh_event_send
                )
                recv_counts_cpu, _, dtoh_event_recv = async_copy_to_cpu(
                    recv_counts_gpu, event=self._dtoh_event_recv
                )
                parallel_batch_size_per_local_expert_cpu, _, dtoh_event = async_copy_to_cpu(
                    parallel_batch_size_per_local_expert, event=self._dtoh_event
                )

        # 2. permute local tokens to be ready for all-to-all communication

        with annotate("permute_local_tokens", "comm"):
            routing_map = local_x_global_routed_expert_indices.view(
                -1, self.routed_experts_router.top_k
            ).int()
            num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k  # dropless
            hidden_shape_before_permute = moe_inp.shape
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                inp=moe_inp,
                routing_map=routing_map,
                num_out_tokens=num_out_tokens,
                map_type="index",
            )

        with torch.no_grad():
            assert dtoh_event_send
            assert dtoh_event_recv
            assert dtoh_event
            dtoh_event.synchronize()
            send_counts = send_counts_cpu.tolist()  # tensor to list
            recv_counts = recv_counts_cpu.tolist()  # tensor to list
            tokens_received = sum(recv_counts)

        with annotate("all2all", "comm"):
            permutated_local_x, global_x, global_x_handle = ops.all_to_all_async(
                permutated_local_x,
                recv_counts,
                send_counts,
                group=self.ep_pg,
            )

        with torch.no_grad():
            # this specifiyes for the received global tokens, which local expert they belong to
            global_x_local_expert_indices = torch.repeat_interleave(
                global_x_local_expert_indices_0,
                global_batch_size_per_local_expert.flatten(),
                output_size=tokens_received,
            )  # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts

    with annotate("tbo_1", "tbo"):
        if x1_is_fresh:
            x1_ctx = cast(dict, x1_ctx)
            x1 = x1_ctx["x1"]
            assert x1.shape == (B, S, D)
            block_inp1 = x1
            del x1
        else:
            assert isinstance(x1_ctx, SyncedTboPendingContext)
            reversed_local_x_permutation_mapping1 = x1_ctx.reversed_local_x_permutation_mapping
            local_x_global_routed_expert_weights1 = x1_ctx.local_x_global_routed_expert_weights
            hidden_shape_before_permute1 = x1_ctx.hidden_shape_before_permute
            in_shape1 = x1_ctx.in_shape
            mixed_shared_out1 = x1_ctx.mixed_shared_out
            attn_res_out1 = x1_ctx.attn_res_out

            assert last_block is not None
            assert local_x_handle1 is not None
            assert local_x1 is not None
            assert last_block.routed_experts_router is not None

            local_x1 = ops.all_to_all_wait(global_x1, local_x1, local_x_handle1)

            # 9. Unpermute the (local) tokens returned by all-to-all communication
            with annotate("unpermute_merge_local_tokens", "comm"):
                local_x1 = moe_unpermute_no_compile(
                    inp=local_x1,
                    row_id_map=reversed_local_x_permutation_mapping1,
                    merging_probs=local_x_global_routed_expert_weights1.view(
                        -1, last_block.routed_experts_router.top_k
                    ),
                    restore_shape=hidden_shape_before_permute1,
                    map_type="index",
                )

            local_x1 = local_x1.view(in_shape1)

            mlp_out1 = last_block._merge_routed_and_shared(local_x1, mixed_shared_out1)

            block_inp1 = last_block._res_norm_mlp(attn_res_out1, mlp_out1)

        # x1 last step done

        # attention
        # + attention norm
        # + residual connection
        attn_res_out1 = self._checkpointed_res_norm_attn(block_inp1, **kwargs)
        moe_inp1 = self._prepare_moe_input(attn_res_out1)

        # routed expert router
        (
            local_x_global_routed_expert_weights1,  # (B, S, top_k)
            local_x_global_routed_expert_indices1,  # (B, S, top_k)
            local_batch_size_per_global_routed_expert1,  # (num_experts, )
            routed_expert_router_aux_loss_info1,
        ) = self.routed_experts_router(
            moe_inp1, False, loss_div_factor=loss_div_factor  # scalar
        )

        attn_res_out1 = self._attach_routed_aux_loss(
            attn_res_out1,
            routed_expert_router_aux_loss_info1,
        )

        with annotate("token_count_all_to_all", "comm"):
            with torch.no_grad():
                # Pass token count information to the device on which the
                # target expert resides.
                global_batch_size_per_local_expert1 = torch.empty_like(
                    local_batch_size_per_global_routed_expert1,
                )
                global_batch_size_handle1 = dist.all_to_all_single(
                    global_batch_size_per_local_expert1,  # Gathered concatenated output tensor.
                    local_batch_size_per_global_routed_expert1,  # Input tensor to scatter.
                    group=self.ep_pg,
                    async_op=True,
                )

                assert global_batch_size_handle1 is not None  # because of async

        if self.shared_experts_router:
            # shared expert router
            (
                local_x_global_shared_expert_weights1,  # (B, S, E_shared)
                _,
                _,
                _,
            ) = self.shared_experts_router(
                moe_inp1,
                True,  # only need scores for shared experts
                loss_div_factor=loss_div_factor,  # scalar
            )
        else:
            local_x_global_shared_expert_weights1 = None

        if self.shared_experts is not None:
            shared_out1 = self.shared_experts.forward(moe_inp1)

            with annotate("merge_shared", "experts"):
                if self.shared_experts_router:
                    assert local_x_global_shared_expert_weights1 is not None
                    # weighted sum of the shared experts by router weights
                    # local_x_global_shared_expert_weights -> (B, S, E_shared)
                    # shared_out -> (E_shared, B, S, D)
                    _, _, E_s1 = local_x_global_shared_expert_weights1.shape
                    mixed_shared_out1 = (
                        torch.bmm(
                            local_x_global_shared_expert_weights1.to(shared_out1.dtype).reshape(
                                B * S, 1, E_s1
                            ),  # (BS, 1, E),
                            shared_out1.permute(1, 2, 0, 3)
                            .contiguous()
                            .view(B * S, E_s1, D),  # (BS, E, D)
                        )
                        .squeeze(1)
                        .view(B, S, D)
                    )
                else:
                    mixed_shared_out1 = shared_out1.squeeze(0)
        else:
            mixed_shared_out1 = None

        in_shape1 = moe_inp1.size()

        moe_inp1 = moe_inp1.view(-1, in_shape1[-1])  # (B*S, D)

        # 3. Configure the sizes for grouped GEMM

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with annotate("sync_token_count", "comm"):
            with torch.no_grad():
                global_batch_size_handle1.wait()

                # Reshape to (ep_world_size, num_local_routed_experts).
                local_batch_size_per_global_routed_expert1 = (
                    local_batch_size_per_global_routed_expert1.view(
                        self.ep_world_size, self.num_local_routed_experts
                    )
                )
                global_batch_size_per_local_expert1 = global_batch_size_per_local_expert1.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
                # Calculate the bins boundaries from the token counts. # [EP, num_local_routed_experts] -> [num_local_routed_experts,]
                parallel_batch_size_per_local_expert1 = global_batch_size_per_local_expert1.sum(
                    dim=0,
                    dtype=torch.long,
                )

                send_counts_gpu1 = local_batch_size_per_global_routed_expert1.sum(dim=-1)
                recv_counts_gpu1 = global_batch_size_per_local_expert1.sum(dim=-1)
                send_counts_cpu1, _, dtoh_event_send1 = async_copy_to_cpu(
                    send_counts_gpu1, event=self._dtoh_event_send1
                )
                recv_counts_cpu1, _, dtoh_event_recv1 = async_copy_to_cpu(
                    recv_counts_gpu1, event=self._dtoh_event_recv1
                )
                parallel_batch_size_per_local_expert_cpu1, _, dtoh_event1 = async_copy_to_cpu(
                    parallel_batch_size_per_local_expert1, event=self._dtoh_event1
                )

        # 2. permute local tokens to be ready for all-to-all communication

        with annotate("permute_local_tokens", "comm"):
            routing_map1 = local_x_global_routed_expert_indices1.view(
                -1, self.routed_experts_router.top_k
            ).int()
            num_out_tokens1 = routing_map1.size(0) * self.routed_experts_router.top_k  # dropless
            hidden_shape_before_permute1 = moe_inp1.shape
            permutated_local_x1, reversed_local_x_permutation_mapping1 = moe_permute_no_compile(
                inp=moe_inp1,
                routing_map=routing_map1,
                num_out_tokens=num_out_tokens1,
                map_type="index",
            )

        with torch.no_grad():
            # assert dtoh_event_send1
            # assert dtoh_event_recv1
            assert dtoh_event1
            dtoh_event1.synchronize()
            send_counts1 = send_counts_cpu1.tolist()  # tensor to list
            recv_counts1 = recv_counts_cpu1.tolist()  # tensor to list
            tokens_received1 = sum(recv_counts1)

    with annotate("tbo_0", "tbo"):
        global_x = ops.all_to_all_wait(permutated_local_x, global_x, global_x_handle)

    with annotate("tbo_1", "tbo"):
        with annotate("all2all", "comm"):
            permutated_local_x1, global_x1, global_x_handle1 = ops.all_to_all_async(
                permutated_local_x1,
                recv_counts1,
                send_counts1,
                group=self.ep_pg,
            )

        with torch.no_grad():
            # this specifiyes for the received global tokens, which local expert they belong to
            global_x_local_expert_indices1 = torch.repeat_interleave(
                global_x_local_expert_indices_0,
                global_batch_size_per_local_expert1.flatten(),
                output_size=tokens_received1,
            )  # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts

    with annotate("tbo_0", "tbo"):
        global_x = checkpointed_permute_routed_experts_unpermute_1d(
            self,
            global_x,
            global_x_local_expert_indices,
            (
                parallel_batch_size_per_local_expert_cpu
                if requires_host_side_split_sizes()
                else parallel_batch_size_per_local_expert
            ),
        )

    with annotate("tbo_1", "tbo"):
        global_x1 = ops.all_to_all_wait(permutated_local_x1, global_x1, global_x_handle1)

    with annotate("tbo_0", "tbo"):
        # 8. reverse_all_to_all

        with annotate("reverse_all_to_all", "comm"):
            global_x = cast(torch.Tensor, global_x)
            global_x, local_x, local_x_handle = ops.all_to_all_async(
                global_x,
                send_counts,
                recv_counts,
                group=self.ep_pg,
            )

    with annotate("tbo_1", "tbo"):
        global_x1 = checkpointed_permute_routed_experts_unpermute_1d(
            self,
            global_x1,
            global_x_local_expert_indices1,
            (
                parallel_batch_size_per_local_expert_cpu1
                if requires_host_side_split_sizes()
                else parallel_batch_size_per_local_expert1
            ),
        )

    with annotate("tbo_0", "tbo"):
        local_x = ops.all_to_all_wait(global_x, local_x, local_x_handle)

        # 9. Unpermute the (local) tokens returned by all-to-all communication
        with annotate("unpermute_merge_local_tokens", "comm"):
            local_x = moe_unpermute_no_compile(
                inp=local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=local_x_global_routed_expert_weights.view(
                    -1, self.routed_experts_router.top_k
                ),
                restore_shape=hidden_shape_before_permute,
                map_type="index",
            )

        local_x = local_x.view(in_shape)

        mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)

        final_out = self._res_norm_mlp(attn_res_out, mlp_out)

    with annotate("tbo_1", "tbo"):
        x1_ctx = SyncedTboPendingContext(
            global_x=global_x1,
            send_counts=send_counts1,
            recv_counts=recv_counts1,
            reversed_local_x_permutation_mapping=reversed_local_x_permutation_mapping1,
            local_x_global_routed_expert_weights=local_x_global_routed_expert_weights1,
            hidden_shape_before_permute=hidden_shape_before_permute1,
            in_shape=in_shape1,
            mixed_shared_out=mixed_shared_out1,
            attn_res_out=attn_res_out1,
            last_block=self,
        )

    return (
        final_out,
        x1_ctx,
    )
