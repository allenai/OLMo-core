from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast

import nvtx
import torch

from ...moe.utils import (
    record_stream_event_no_compile,
    wait_event_no_compile,
    wait_stream_no_compile,
)
from ..utils import (
    build_chunk_te_routing_map,
    moe_chunk_reorder_no_compile,
    moe_permute_1d_fused_drop_no_compile,
)
from .comm import _CombineVDevAutograd, _DispatchVDevAutograd
from .ep_no_sync_state import (
    _NoSyncStageAState,
    _NoSyncStageDState,
    _NoSyncTboPendingContext,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


def ep_no_sync_stage_a(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    lane_id: int,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> _NoSyncStageAState:
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert use_torch_grouped_mm() == True, "EP no-sync implementation requires torch.grouped_mm support"
    assert not requires_host_side_split_sizes(), "EP no-sync implementation does not support host-side split size communication"
    if self.ep_no_sync_use_2d_all_to_all:
        raise RuntimeError(
            "ep_no_sync_use_2d_all_to_all=True is no longer supported: "
            "the 2D all_to_all path was removed due to correctness/performance issues."
        )
    if self.ep_no_sync_use_rowwise_all_to_all:
        raise RuntimeError(
            "ep_no_sync_use_rowwise_all_to_all=True is only implemented for "
            "combined_forward_ep_no_sync() (non-TBO path) right now."
        )

    group_name = self._get_ep_no_sync_group_name()
    slot_idx = self._ep_no_sync_slot_for_lane(lane_id)
    B, S, D = x.shape
    block_inp = x
    del x

    attn_kwargs = dict(kwargs)
    with nvtx.annotate("A-AttnRouter", color="purple"):
        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **attn_kwargs)
        moe_inp = self._prepare_moe_input(attn_res_out)
        (
            local_x_global_routed_expert_weights,
            local_x_global_routed_expert_indices,
            local_batch_size_per_global_routed_expert,
            routed_expert_router_aux_loss_info,
        ) = self.router_forward(
            router=self.routed_experts_router,
            local_x=moe_inp,
            scores_only=False,
            loss_div_factor=loss_div_factor,
        )

    mixed_shared_out: Optional[torch.Tensor]
    shared_done_event: Optional[torch.cuda.Event] = None
    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )
    dense_stream = self.get_dense_stream()
    with torch.cuda.stream(dense_stream):
        if self.shared_experts_router:
            (
                local_x_global_shared_expert_weights,
                _,
                _,
                _,
            ) = self.router_forward(
                router=self.shared_experts_router,
                local_x=moe_inp,
                scores_only=True,
                loss_div_factor=loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None

        if self.shared_experts is not None:
            shared_out = self.shared_experts(moe_inp)
            if self.shared_experts_router:
                assert local_x_global_shared_expert_weights is not None
                _, _, E_s = local_x_global_shared_expert_weights.shape
                mixed_shared_out = torch.bmm(
                    local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
                    shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
                ).squeeze(1).view(B, S, D)
            else:
                mixed_shared_out = shared_out.squeeze(0)
            shared_done_event = record_stream_event_no_compile(dense_stream)
        else:
            mixed_shared_out = None

    in_shape = moe_inp.size()
    moe_inp = moe_inp.view(-1, in_shape[-1])
    hidden_shape_before_permute = moe_inp.shape
    num_out_tokens = local_x_global_routed_expert_indices.numel()

    with torch.no_grad():
        with nvtx.annotate("A-ConfigCapacity", color="green"):
            requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
            rank_capacity = self._compute_ep_no_sync_rank_capacity(num_out_tokens)
            allowed_splits, recv_splits_by_src_local, _drop_token_cnt = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                self._sync_tail_drop_allowed_splits_single_a2a(
                    requested_splits,
                    rank_capacity=rank_capacity,
                ),
            )
            local_reorder_indices, local_inverse_reorder_indices, packed_keep_mask = self._build_keep_reorder(
                requested_splits=requested_splits,
                keep_splits=allowed_splits,
                num_out_tokens=num_out_tokens,
            )
            num_kept = allowed_splits.sum(dtype=torch.long)

            dispatch_in_cap = num_out_tokens
            dispatch_out_cap = rank_capacity
            combine_in_cap = rank_capacity
            combine_out_cap = num_out_tokens

    buffers = self._get_ep_no_sync_buffers(
        dispatch_in_cap=dispatch_in_cap,
        dispatch_out_cap=dispatch_out_cap,
        combine_in_cap=combine_in_cap,
        combine_out_cap=combine_out_cap,
        d_model=moe_inp.shape[-1],
        dtype=moe_inp.dtype,
        device=moe_inp.device,
        slot_idx=slot_idx,
    )

    with nvtx.annotate("A-PermuteLocal", color="green"):
        routing_map = local_x_global_routed_expert_indices.view(
            -1, self.routed_experts_router.top_k
        ).int()
        permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_1d_fused_drop_no_compile(
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

    return _NoSyncStageAState(
        lane_id=lane_id,
        slot_idx=slot_idx,
        group_name=group_name,
        in_shape=in_shape,
        hidden_shape_before_permute=hidden_shape_before_permute,
        B=B,
        S=S,
        D=D,
        attn_res_out=attn_res_out,
        mixed_shared_out=mixed_shared_out,
        shared_done_event=shared_done_event,
        local_x_global_routed_expert_weights=local_x_global_routed_expert_weights,
        routed_expert_router_aux_loss_info=routed_expert_router_aux_loss_info,
        requested_splits=requested_splits,
        allowed_splits=allowed_splits,
        recv_splits_by_src_local=recv_splits_by_src_local,
        local_inverse_reorder_indices=local_inverse_reorder_indices,
        packed_keep_mask=packed_keep_mask,
        num_kept=num_kept,
        num_out_tokens=num_out_tokens,
        send_rank_splits=send_rank_splits,
        buffers=buffers,
        permutated_local_x=permutated_local_x,
        reversed_local_x_permutation_mapping=reversed_local_x_permutation_mapping,
    )


def ep_no_sync_stage_d_launch(block: MoEFusedV2TransformerBlock, a_state: _NoSyncStageAState) -> _NoSyncStageDState:
    self = block
    comm_stream = self.get_ep_no_sync_comm_stream()
    wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())

    with torch.cuda.stream(comm_stream):
        dispatch_out, dispatch_rank_splits_offsets = _DispatchVDevAutograd.apply(
            a_state.permutated_local_x,
            a_state.send_rank_splits,
            a_state.buffers.dispatch_in,
            a_state.buffers.dispatch_in_rank_splits,
            a_state.buffers.dispatch_out,
            a_state.buffers.dispatch_rank_splits_offsets,
            a_state.buffers.dispatch_tmp_rank_splits_offsets,
            a_state.group_name,
            self.ep_pg,
        )
        dispatch_done_event = record_stream_event_no_compile(comm_stream)

    return _NoSyncStageDState(
        lane_id=a_state.lane_id,
        a_state=a_state,
        dispatch_out=dispatch_out,
        dispatch_rank_splits_offsets=dispatch_rank_splits_offsets,
        dispatch_done_event=dispatch_done_event,
    )


def ep_no_sync_stage_e(block: MoEFusedV2TransformerBlock, d_state: _NoSyncStageDState) -> _NoSyncTboPendingContext:
    self = block
    assert self.routed_experts is not None
    wait_event_no_compile(torch.cuda.current_stream(), d_state.dispatch_done_event)

    a_state = d_state.a_state
    buffers = a_state.buffers
    dispatch_rank_major = d_state.dispatch_out

    with torch.no_grad():
        padded_batch_size_per_local_expert = a_state.recv_splits_by_src_local.sum(
            dim=0,
            dtype=torch.long,
        )

    with nvtx.annotate("E-PermuteGlobal", color="green"):
        if self.routed_experts.num_local_experts == 1:
            dispatch_rank_major = dispatch_rank_major.clone()
            global_chunk_row_id_map = None
        else:
            with torch.no_grad():
                global_chunk_routing_map = build_chunk_te_routing_map(
                    a_state.recv_splits_by_src_local,
                    rows=dispatch_rank_major.shape[0],
                )
            dispatch_rank_major, global_chunk_row_id_map = moe_chunk_reorder_no_compile(
                inp=dispatch_rank_major,
                routing_map=global_chunk_routing_map,
                num_out_tokens=dispatch_rank_major.shape[0],
                backward_grad_input_buffer=buffers.dispatch_out.detach(),
            )

    with nvtx.annotate("E-RoutedExperts", color="green"):
        dispatch_rank_major = self.routed_experts(
            dispatch_rank_major,
            padded_batch_size_per_local_expert,
        )

    with nvtx.annotate("E-UnpermuteGlobal", color="green"):
        if self.routed_experts.num_local_experts == 1:
            global_x_rank_major = dispatch_rank_major
        else:
            assert global_chunk_row_id_map is not None
            global_x_rank_major = moe_chunk_reorder_no_compile(
                inp=dispatch_rank_major,
                row_id_map=global_chunk_row_id_map,
                out=buffers.combine_in.detach(),
            )

    return _NoSyncTboPendingContext(
        block=self,
        lane_id=d_state.lane_id,
        a_state=a_state,
        dispatch_rank_splits_offsets=d_state.dispatch_rank_splits_offsets,
        global_x_rank_major=global_x_rank_major,
    )


def ep_no_sync_stage_c_launch(
    block: MoEFusedV2TransformerBlock,
    pending_ctx: _NoSyncTboPendingContext,
) -> _NoSyncTboPendingContext:
    del block
    block = pending_ctx.block
    a_state = pending_ctx.a_state
    buffers = a_state.buffers
    comm_stream = block.get_ep_no_sync_comm_stream()
    wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())

    with torch.cuda.stream(comm_stream):
        combine_out, _combine_rank_splits_offsets = _CombineVDevAutograd.apply(
            pending_ctx.global_x_rank_major,
            pending_ctx.dispatch_rank_splits_offsets[0],
            buffers.combine_in,
            buffers.combine_in_rank_splits,
            buffers.combine_out,
            buffers.combine_rank_splits_offsets,
            buffers.combine_tmp_rank_splits_offsets,
            a_state.group_name,
            block.ep_pg,
        )
        combine_done_event = record_stream_event_no_compile(comm_stream)

    pending_ctx.combine_out = combine_out
    pending_ctx.combine_done_event = combine_done_event
    return pending_ctx


def ep_no_sync_stage_tail(block: MoEFusedV2TransformerBlock, pending_ctx: _NoSyncTboPendingContext) -> torch.Tensor:
    self = block
    if pending_ctx.combine_done_event is not None:
        wait_event_no_compile(torch.cuda.current_stream(), pending_ctx.combine_done_event)

    a_state = pending_ctx.a_state
    buffers = a_state.buffers
    if a_state.shared_done_event is not None:
        wait_event_no_compile(torch.cuda.current_stream(), a_state.shared_done_event)

    combine_out = pending_ctx.combine_out if pending_ctx.combine_out is not None else buffers.combine_out
    with nvtx.annotate("Tail-UnpermuteMerge", color="green"):
        combine_out_for_unpermute = combine_out.clone() if buffers.combine_out_is_shared else combine_out
        local_x = self._restore_drop_unpermute_1d(
            combine_out=combine_out_for_unpermute,
            local_inverse_reorder_indices=a_state.local_inverse_reorder_indices,
            packed_keep_mask=a_state.packed_keep_mask,
            num_kept=a_state.num_kept,
            reversed_local_x_permutation_mapping=a_state.reversed_local_x_permutation_mapping,
            local_x_global_routed_expert_weights=a_state.local_x_global_routed_expert_weights,
            hidden_shape_before_permute=a_state.hidden_shape_before_permute,
            row_id_map_is_packed=True,
            backward_grad_input_buffer=buffers.combine_out.detach(),
        )

    local_x = local_x.view(a_state.in_shape)
    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())

    if self.shared_experts is not None:
        assert a_state.mixed_shared_out is not None
        mlp_out = local_x + a_state.mixed_shared_out
    else:
        mlp_out = local_x

    final_out = self._res_norm_mlp(a_state.attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        a_state.routed_expert_router_aux_loss_info,
    )


def combined_forward_ep_no_sync_tbo(
    block: MoEFusedV2TransformerBlock,
    x0: torch.Tensor,
    x1_ctx: object,
    x1_is_fresh: bool,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, _NoSyncTboPendingContext]:
    self = block
    if x1_is_fresh:
        pending_prev = None
    else:
        if not isinstance(x1_ctx, _NoSyncTboPendingContext):
            raise RuntimeError(
                "Expected no-sync TBO context from previous block, "
                f"got type={type(x1_ctx)}"
            )
        pending_prev = x1_ctx

    with nvtx.annotate("TBO-1", color="orange"):
        if pending_prev is not None:
            pending_prev = pending_prev.block._ep_no_sync_stage_c_launch(pending_prev)
    with nvtx.annotate("TBO-0", color="purple"):
        a0 = self._ep_no_sync_stage_a(
            x0,
            lane_id=0,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    with nvtx.annotate("TBO-1", color="orange"):
        if x1_is_fresh:
            fresh_ctx = cast(Dict[str, torch.Tensor], x1_ctx)
            block_inp1 = fresh_ctx["x1"]
        else:
            assert pending_prev is not None
            block_inp1 = pending_prev.block._ep_no_sync_stage_tail(pending_prev)

    with nvtx.annotate("TBO-0", color="purple"):
        d0 = self._ep_no_sync_stage_d_launch(a0)
    with nvtx.annotate("TBO-1", color="orange"):
        a1 = self._ep_no_sync_stage_a(
            block_inp1,
            lane_id=1,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    with nvtx.annotate("TBO-1", color="orange"):
        d1 = self._ep_no_sync_stage_d_launch(a1)
    with nvtx.annotate("TBO-0", color="purple"):
        pending0_pre_c = self._ep_no_sync_stage_e(d0)

    with nvtx.annotate("TBO-0", color="purple"):
        pending0_post_c = self._ep_no_sync_stage_c_launch(pending0_pre_c)
    with nvtx.annotate("TBO-1", color="orange"):
        pending1_pre_c = self._ep_no_sync_stage_e(d1)

    with nvtx.annotate("TBO-0", color="purple"):
        final_out = self._ep_no_sync_stage_tail(pending0_post_c)

    return final_out, pending1_pre_c
