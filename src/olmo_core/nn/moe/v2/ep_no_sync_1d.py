from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import torch

from olmo_core._nvtx import nvtx

from ...moe.utils import wait_stream_no_compile
from ..utils import (
    build_chunk_te_routing_map,
    moe_chunk_reorder_no_compile,
    moe_permute_1d_fused_drop_no_compile,
)
from .comm import _CombineVDevAutograd, _DispatchVDevAutograd
from .ep_backend import finish_moe_forward, prepare_and_route_moe
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
)
from .ep_no_sync_common import (
    build_keep_reorder,
    padded_local_expert_splits_for_capacity,
    restore_drop_unpermute_1d,
    sync_tail_drop_allowed_splits_single_a2a,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


@dataclass(frozen=True)
class _NoSync1DCapacityPlan:
    requested_splits: torch.Tensor
    allowed_splits: torch.Tensor
    recv_splits_by_src_local: torch.Tensor
    local_reorder_indices: torch.Tensor
    local_inverse_reorder_indices: torch.Tensor
    packed_keep_mask: torch.Tensor
    num_kept: torch.Tensor
    num_out_tokens: int
    rank_capacity: int


def _apply_token_capacity_and_drop(
    block: OLMoDDPTransformerBlock,
    local_batch_size_per_global_routed_expert: torch.Tensor,
    *,
    num_out_tokens: int,
) -> _NoSync1DCapacityPlan:
    """Build the deterministic tail-drop and local reorder plan for no-sync 1D EP."""

    with torch.no_grad(), nvtx.annotate("ConfigCapacity", color="green"):
        requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
        rank_capacity = compute_ep_no_sync_rank_capacity(block, num_out_tokens)
        allowed_splits, recv_splits_by_src_local, drop_token_count = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            sync_tail_drop_allowed_splits_single_a2a(
                block,
                requested_splits,
                rank_capacity=rank_capacity,
            ),
        )
        (
            local_reorder_indices,
            local_inverse_reorder_indices,
            packed_keep_mask,
        ) = build_keep_reorder(
            requested_splits=requested_splits,
            keep_splits=allowed_splits,
            num_out_tokens=num_out_tokens,
        )
        num_kept = allowed_splits.sum(dtype=torch.long)
        block._ep_no_sync_last_debug = {
            "num_dropped": drop_token_count.detach(),
            "rank_capacity": torch.tensor(
                rank_capacity,
                device=requested_splits.device,
                dtype=torch.long,
            ),
            "received_tokens_after_drop": recv_splits_by_src_local.sum(
                dtype=torch.long
            ).detach(),
            "allowed_splits": allowed_splits.detach(),
            "local_kept_tokens": num_kept.detach(),
            "combined_tokens": num_kept.detach(),
            "zero_rows_after_local_unpermute": (
                torch.tensor(
                    num_out_tokens,
                    device=requested_splits.device,
                    dtype=torch.long,
                )
                - num_kept
            ).detach(),
        }

    return _NoSync1DCapacityPlan(
        requested_splits=requested_splits,
        allowed_splits=allowed_splits,
        recv_splits_by_src_local=recv_splits_by_src_local,
        local_reorder_indices=local_reorder_indices,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        num_kept=num_kept,
        num_out_tokens=num_out_tokens,
        rank_capacity=rank_capacity,
    )


def _get_buffers(block: OLMoDDPTransformerBlock, plan: _NoSync1DCapacityPlan, x: torch.Tensor):
    """Acquire no-sync 1D buffers sized from the capacity plan and routed width."""

    return get_ep_no_sync_buffers(
        block,
        dispatch_in_cap=plan.num_out_tokens,
        dispatch_out_cap=plan.rank_capacity,
        combine_in_cap=plan.rank_capacity,
        combine_out_cap=plan.num_out_tokens,
        d_model=x.shape[-1],
        dtype=x.dtype,
        device=x.device,
    )


def combined_forward_ep_no_sync_1d(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Legacy 1D EP no-sync forward using symmetric-memory all_to_all_vdev ops.

    This path is kept primarily because the current TBO implementation still
    shares its 1D machinery. Row-wise no-sync is the production no-sync path.
    """
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert use_torch_grouped_mm(), "EP no-sync implementation requires torch.grouped_mm support"
    assert (
        not requires_host_side_split_sizes()
    ), "EP no-sync implementation does not support host-side split size communication"
    group_name = get_ep_no_sync_group_name(self)
    B, S, D = x.shape
    prepared, routing = prepare_and_route_moe(
        self,
        x,
        loss_div_factor=loss_div_factor,
        forward_kwargs=kwargs,
    )
    attn_res_out = prepared.attention_residual
    moe_inp = prepared.model_input
    local_x_global_routed_expert_weights = routing.expert_weights
    local_x_global_routed_expert_indices = routing.expert_indices
    local_batch_size_per_global_routed_expert = routing.tokens_per_expert

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

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

    capacity = _apply_token_capacity_and_drop(
        self,
        local_batch_size_per_global_routed_expert,
        num_out_tokens=local_x_global_routed_expert_indices.numel(),
    )
    buffers = _get_buffers(self, capacity, moe_inp)

    requested_splits = capacity.requested_splits
    allowed_splits = capacity.allowed_splits
    recv_splits_by_src_local = capacity.recv_splits_by_src_local
    local_reorder_indices = capacity.local_reorder_indices
    local_inverse_reorder_indices = capacity.local_inverse_reorder_indices
    packed_keep_mask = capacity.packed_keep_mask
    num_kept = capacity.num_kept
    num_out_tokens = capacity.num_out_tokens
    rank_capacity = capacity.rank_capacity

    routing_map = local_x_global_routed_expert_indices.view(
        -1, self.routed_experts_router.top_k
    ).int()
    hidden_shape_before_permute = moe_inp.shape

    with torch.no_grad():
        padded_batch_size_per_local_expert = padded_local_expert_splits_for_capacity(
            recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )

    with nvtx.annotate("Permute local tokens", color="green"):
        (
            permutated_local_x,
            reversed_local_x_permutation_mapping,
        ) = moe_permute_1d_fused_drop_no_compile(
            inp=moe_inp,
            routing_map=routing_map,
            num_out_tokens=num_out_tokens,
            reorder_indices=local_reorder_indices,
            inverse_reorder_indices=local_inverse_reorder_indices,
            requested_splits=requested_splits,
            keep_splits=allowed_splits,
            out=buffers.dispatch_in.detach(),
            map_type="index",
        )

    with torch.no_grad():
        send_rank_splits = allowed_splits.view(
            self.ep_world_size, self.num_local_routed_experts
        ).sum(dim=-1, dtype=torch.long)

    if self.shared_experts is not None:
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    dispatch_out, dispatch_rank_splits_offsets = _DispatchVDevAutograd.apply(
        permutated_local_x,
        send_rank_splits,
        buffers.dispatch_in,
        buffers.dispatch_in_rank_splits,
        buffers.dispatch_out,
        buffers.dispatch_rank_splits_offsets,
        buffers.dispatch_tmp_rank_splits_offsets,
        group_name,
        self.ep_pg,
    )

    dispatch_rank_major = dispatch_out

    with nvtx.annotate("Permute global tokens", color="green"):
        if self.routed_experts.num_local_experts == 1:
            dispatch_rank_major = dispatch_rank_major.clone()
            global_chunk_row_id_map = None
        else:
            with torch.no_grad():
                global_chunk_routing_map = build_chunk_te_routing_map(
                    recv_splits_by_src_local,
                    rows=dispatch_rank_major.shape[0],
                )
            dispatch_rank_major, global_chunk_row_id_map = moe_chunk_reorder_no_compile(
                dispatch_rank_major,
                routing_map=global_chunk_routing_map,
                num_out_tokens=dispatch_rank_major.shape[0],
                backward_grad_input_buffer=buffers.dispatch_out.detach(),
            )

    dispatch_rank_major = self.routed_experts(
        dispatch_rank_major,
        padded_batch_size_per_local_expert,
    )

    with nvtx.annotate("Unpermute global tokens", color="green"):
        if self.routed_experts.num_local_experts == 1:
            global_x_rank_major = dispatch_rank_major
        else:
            assert global_chunk_row_id_map is not None
            global_x_rank_major = moe_chunk_reorder_no_compile(
                inp=dispatch_rank_major,
                row_id_map=global_chunk_row_id_map,
                out=buffers.combine_in.detach(),
            )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    combine_out, _combine_rank_splits_offsets = _CombineVDevAutograd.apply(
        global_x_rank_major,
        dispatch_rank_splits_offsets[0],
        buffers.combine_in,
        buffers.combine_in_rank_splits,
        buffers.combine_out,
        buffers.combine_rank_splits_offsets,
        buffers.combine_tmp_rank_splits_offsets,
        group_name,
        self.ep_pg,
    )

    with nvtx.annotate("Unpermute-Merge local tokens", color="green"):
        combine_out_for_unpermute = (
            combine_out.clone() if buffers.combine_out_is_shared else combine_out
        )
        local_x = restore_drop_unpermute_1d(
            self,
            combine_out=combine_out_for_unpermute,
            local_inverse_reorder_indices=local_inverse_reorder_indices,
            packed_keep_mask=packed_keep_mask,
            num_kept=num_kept,
            reversed_local_x_permutation_mapping=reversed_local_x_permutation_mapping,
            local_x_global_routed_expert_weights=local_x_global_routed_expert_weights,
            hidden_shape_before_permute=hidden_shape_before_permute,
            row_id_map_is_packed=True,
            backward_grad_input_buffer=buffers.combine_out.detach(),
        )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None

        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts.forward2(
                shared_out_up, shared_out_gate, attn_res_out.shape
            )
            mixed_shared_out = self._mix_shared_out(
                shared_out,
                local_x_global_shared_expert_weights,
                attn_res_out.shape,
            )
    else:
        mixed_shared_out = None

    local_x = local_x.view(in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)

    return finish_moe_forward(self, prepared, mlp_out, routing)
