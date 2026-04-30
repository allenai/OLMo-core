from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, cast

import nvtx
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from olmo_core.distributed.utils import get_rank
from olmo_core.ops import moe as ops

from ...moe.utils import async_copy_to_cpu, wait_stream_no_compile
from ..utils import moe_permute_no_compile, moe_unpermute_no_compile
from .routed_experts import requires_host_side_split_sizes

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def routed_experts_unpermute_1d(
    block: MoEFusedV2TransformerBlock,
    global_x: torch.Tensor,
    global_x_local_expert_indices: torch.Tensor,
    parallel_batch_size_per_local_expert_cpu,
    hidden_shape_before_permute2: torch.Size,
    reversed_global_x_permutation_mapping: Optional[torch.Tensor],
) -> torch.Tensor:
    del global_x_local_expert_indices
    self = block
    assert self.routed_experts is not None

    global_x = self.routed_experts(global_x, parallel_batch_size_per_local_expert_cpu)

    with nvtx.annotate("Unpermute global tokens", color="green"):
        if self.routed_experts.num_local_experts == 1:
            return global_x
        return moe_unpermute_no_compile(
            inp=global_x,
            row_id_map=reversed_global_x_permutation_mapping,
            merging_probs=None,
            restore_shape=hidden_shape_before_permute2,
            map_type="index",
        )


def checkpointed_permute_routed_experts_unpermute_1d(
    block: MoEFusedV2TransformerBlock,
    global_x: torch.Tensor,
    global_x_local_expert_indices: torch.Tensor,
    parallel_batch_size_per_local_expert_cpu,
) -> torch.Tensor:
    self = block
    assert self.routed_experts is not None

    # The initial permute does not save input for backward, so only the
    # routed-expert/unpermute section is optionally checkpointed.
    with nvtx.annotate("Permute global tokens for MLP", color="green"):
        routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
        num_out_tokens2 = routing_map2.size(0)
        hidden_shape_before_permute2 = global_x.shape
        if self.routed_experts.num_local_experts == 1:
            reversed_global_x_permutation_mapping = None
        else:
            global_x, reversed_global_x_permutation_mapping = moe_permute_no_compile(
                inp=global_x,
                routing_map=routing_map2,
                num_out_tokens=num_out_tokens2,
                map_type="index",
            )

    if self.checkpoint_permute_moe_unpermute:
        out = checkpoint(
            routed_experts_unpermute_1d,
            self,
            global_x,
            global_x_local_expert_indices,
            parallel_batch_size_per_local_expert_cpu,
            hidden_shape_before_permute2,
            reversed_global_x_permutation_mapping,
            use_reentrant=False,
        )
        return cast(torch.Tensor, out)
    return routed_experts_unpermute_1d(
        self,
        global_x,
        global_x_local_expert_indices,
        parallel_batch_size_per_local_expert_cpu,
        hidden_shape_before_permute2,
        reversed_global_x_permutation_mapping,
    )


def combined_forward_ep_1d(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward function with synced expert parallelism."""
    self = block
    assert self.routed_experts_router is not None
    assert self.ep_enabled == True
    assert self.num_local_routed_experts is not None

    B, S, D = x.shape

    block_inp = x
    del x

    attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)

    kwargs.pop("max_doc_len", None)
    kwargs.pop("cu_doc_lens", None)
    moe_inp = self._prepare_moe_input(attn_res_out)

    (
        local_x_global_routed_expert_weights,
        local_x_global_routed_expert_indices,
        local_batch_size_per_global_routed_expert,
        routed_expert_router_aux_loss_info,
    ) = self.routed_experts_router(
        moe_inp,
        False,
        loss_div_factor=loss_div_factor,
    )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    with nvtx.annotate("Token count all_to_all", color="green"):
        with torch.no_grad():
            global_batch_size_per_local_expert = torch.empty_like(
                local_batch_size_per_global_routed_expert,
            )
            global_batch_size_handle = dist.all_to_all_single(
                global_batch_size_per_local_expert,
                local_batch_size_per_global_routed_expert,
                group=self.ep_pg,
                async_op=True,
            )
            assert global_batch_size_handle is not None

    with torch.cuda.stream(self.get_dense_stream()):
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights,
                _,
                _,
                _,
            ) = self.shared_experts_router(
                moe_inp,
                True,
                loss_div_factor=loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None

    in_shape = moe_inp.size()

    moe_inp = moe_inp.view(-1, in_shape[-1])

    with nvtx.annotate("Sync token count", color="green"):
        with torch.no_grad():
            global_batch_size_handle.wait()

            local_batch_size_per_global_routed_expert = (
                local_batch_size_per_global_routed_expert.view(
                    self.ep_world_size, self.num_local_routed_experts
                )
            )
            global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                self.ep_world_size, self.num_local_routed_experts
            )
            parallel_batch_size_per_local_expert = global_batch_size_per_local_expert.sum(
                dim=0,
                dtype=torch.long,
            )

            send_counts_gpu = local_batch_size_per_global_routed_expert.sum(dim=-1)
            recv_counts_gpu = global_batch_size_per_local_expert.sum(dim=-1)
            send_counts_cpu, copy_stream, dtoh_event_send = async_copy_to_cpu(
                send_counts_gpu, event=self._dtoh_event_send
            )
            recv_counts_cpu, copy_stream, dtoh_event_recv = async_copy_to_cpu(
                recv_counts_gpu, event=self._dtoh_event_recv
            )
            parallel_batch_size_per_local_expert_cpu, copy_stream, dtoh_event = async_copy_to_cpu(
                parallel_batch_size_per_local_expert,
                event=self._dtoh_event,
            )

    with torch.no_grad():
        global_x_local_expert_indices = torch.remainder(
            torch.arange(
                self.routed_experts_router.num_experts,
                dtype=torch.int32,
                device=moe_inp.device,
            ),
            self.num_local_routed_experts,
        )

    with nvtx.annotate("Permute local tokens", color="green"):
        routing_map = local_x_global_routed_expert_indices.view(
            -1, self.routed_experts_router.top_k
        ).int()
        num_out_tokens = routing_map.size(0) * self.routed_experts_router.top_k
        hidden_shape_before_permute = moe_inp.shape
        permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
            inp=moe_inp,
            routing_map=routing_map,
            num_out_tokens=num_out_tokens,
            map_type="index",
        )

    if self.shared_experts is not None:
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    with torch.no_grad():
        assert dtoh_event_send
        assert dtoh_event_recv
        assert dtoh_event

        dtoh_event.synchronize()
        send_counts = send_counts_cpu.tolist()
        recv_counts = recv_counts_cpu.tolist()
        tokens_received = sum(recv_counts)

    if tokens_received == 0:
        print(
            f"[Warning] (grad={torch.is_grad_enabled()}) block {self.block_idx} "
            f"EP rank {get_rank(self.ep_pg)} has 0 tokens received in all2all: "
            f"send_counts={send_counts} recv_counts={recv_counts}"
        )

    with nvtx.annotate("all2all", color="green"):
        permutated_local_x, global_x, global_x_handle = ops.all_to_all_async(
            permutated_local_x,
            recv_counts,
            send_counts,
            group=self.ep_pg,
        )

    with torch.no_grad():
        global_x_local_expert_indices = torch.repeat_interleave(
            global_x_local_expert_indices,
            global_batch_size_per_local_expert.flatten(),
            output_size=tokens_received,
        )

    global_x = ops.all_to_all_wait(permutated_local_x, global_x, global_x_handle)

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

    before_rev_all2all_event = torch.cuda.current_stream().record_event(
        event=self._before_rev_all2all_event
    )
    with nvtx.annotate("reverse_all_to_all", color="green"):
        global_x = cast(torch.Tensor, global_x)

        global_x, local_x, local_x_handle = ops.all_to_all_async(
            global_x,
            send_counts,
            recv_counts,
            group=self.ep_pg,
        )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None

        self.get_dense_stream().wait_event(before_rev_all2all_event)
        with nvtx.annotate("merge_shared", color="purple"):
            with torch.cuda.stream(self.get_dense_stream()):
                shared_out = self.shared_experts.forward2(
                    shared_out_up, shared_out_gate, attn_res_out.shape
                )
                if self.shared_experts_router:
                    assert local_x_global_shared_expert_weights is not None
                    _, _, E_s = local_x_global_shared_expert_weights.shape
                    mixed_shared_out = (
                        torch.bmm(
                            local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(
                                B * S, 1, E_s
                            ),
                            shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
                        )
                        .squeeze(1)
                        .view(B, S, D)
                    )
                else:
                    mixed_shared_out = shared_out.squeeze(0)
    else:
        mixed_shared_out = None

    local_x = ops.all_to_all_wait(global_x, local_x, local_x_handle)

    with nvtx.annotate("Unpermute-Merge local tokens", color="green"):
        if self.checkpoint_second_unpermute:
            local_x = checkpoint(
                moe_unpermute_no_compile,
                inp=local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=local_x_global_routed_expert_weights.view(
                    -1, self.routed_experts_router.top_k
                ),
                restore_shape=hidden_shape_before_permute,
                map_type="index",
                use_reentrant=False,
            )
            local_x = cast(torch.Tensor, local_x)
        else:
            local_x = moe_unpermute_no_compile(
                inp=local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=local_x_global_routed_expert_weights.view(
                    -1, self.routed_experts_router.top_k
                ),
                restore_shape=hidden_shape_before_permute,
                map_type="index",
            )
            local_x = cast(torch.Tensor, local_x)

    local_x = local_x.view(in_shape)

    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)

    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
    )
