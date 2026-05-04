from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import nvtx
import torch

from olmo_core.utils import get_or_init_stream

from ...moe.utils import (
    record_stream_event_no_compile,
    wait_event_no_compile,
    wait_stream_no_compile,
)
from .comm import _DispatchRowwiseAutograd, _RowwiseCombineWeightedAutograd
from .ep_no_sync_common import sync_tail_drop_allowed_splits_single_a2a
from .ep_no_sync_buffers import (
    _NoSyncSymmBuffers,
    compute_ep_no_sync_rank_capacity,
    ep_no_sync_slot_for_lane,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
    use_ep_no_sync_rowwise_symm_combine_gather,
    use_ep_no_sync_rowwise_symm_combine_out,
    use_ep_no_sync_rowwise_symm_dispatch_in,
)
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


@dataclass
class _NoSyncRowwiseStageAState:
    lane_id: int
    slot_idx: int
    group_name: str
    in_shape: torch.Size
    B: int
    S: int
    D: int
    attn_res_out: torch.Tensor
    shared_out_up: Optional[torch.Tensor]
    shared_out_gate: Optional[torch.Tensor]
    local_x_global_shared_expert_weights: Optional[torch.Tensor]
    local_x_global_routed_expert_weights: torch.Tensor
    routed_expert_router_aux_loss_info: Optional[Tuple[object, ...]]
    padded_batch_size_per_local_expert: torch.Tensor
    dst_ranks: torch.Tensor
    dst_rows: torch.Tensor
    rowwise_nblocks: int
    use_symm_dispatch_in: bool
    use_symm_combine_out: bool
    use_symm_combine_gather: bool
    buffers: _NoSyncSymmBuffers
    moe_inp: torch.Tensor


@dataclass
class _NoSyncRowwiseStageDState:
    lane_id: int
    a_state: _NoSyncRowwiseStageAState
    dispatch_out: torch.Tensor
    dispatch_done_event: torch.cuda.Event


@dataclass
class _NoSyncRowwiseTboPendingContext:
    block: "MoEFusedV2TransformerBlock"
    lane_id: int
    a_state: _NoSyncRowwiseStageAState
    global_x_rank_major: torch.Tensor
    combine_out: Optional[torch.Tensor] = None
    combine_done_event: Optional[torch.cuda.Event] = None


def _check_rowwise_tbo_supported(block: "MoEFusedV2TransformerBlock") -> None:
    if not block.ep_no_sync_use_rowwise_all_to_all:
        raise RuntimeError("Rowwise no-sync TBO requires ep_no_sync_use_rowwise_all_to_all=True")
    if block.rowwise_fp8 is not None and block.rowwise_fp8.enabled:
        raise NotImplementedError(
            "Rowwise FP8 is not implemented for no-sync TBO yet. "
            "It needs per-lane symmetric q/scale buffers before two batches can be in flight."
        )


def ep_no_sync_rowwise_tbo_stage_a(
    block: "MoEFusedV2TransformerBlock",
    x: torch.Tensor,
    *,
    lane_id: int,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> _NoSyncRowwiseStageAState:
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    assert use_torch_grouped_mm() == True, "EP no-sync implementation requires torch.grouped_mm support"
    assert not requires_host_side_split_sizes(), "EP no-sync implementation does not support host-side split size communication"
    _check_rowwise_tbo_supported(self)
    if self.ep_no_sync_use_2d_all_to_all:
        raise RuntimeError(
            "ep_no_sync_use_2d_all_to_all=True is no longer supported: "
            "the 2D all_to_all path was removed due to correctness/performance issues."
        )

    group_name = get_ep_no_sync_group_name(self)
    slot_idx = ep_no_sync_slot_for_lane(self, lane_id)
    B, S, D = x.shape
    block_inp = x
    del x

    with nvtx.annotate("RowwiseTBO-A-AttnRouter", color="purple"):
        attn_res_out = self._checkpointed_res_norm_attn(block_inp, **kwargs)
        moe_inp_3d = self._prepare_moe_input(attn_res_out)
        (
            local_x_global_routed_expert_weights,
            local_x_global_routed_expert_indices,
            local_batch_size_per_global_routed_expert,
            routed_expert_router_aux_loss_info,
        ) = self.routed_experts_router(
            moe_inp_3d,
            False,
            loss_div_factor=loss_div_factor,
        )

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
                moe_inp_3d,
                True,
                loss_div_factor=loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None

    in_shape = moe_inp_3d.size()
    moe_inp = moe_inp_3d.view(-1, in_shape[-1])
    del moe_inp_3d

    num_out_tokens = local_x_global_routed_expert_indices.numel()
    num_input_tokens = moe_inp.shape[0]
    top_k = self.routed_experts_router.top_k
    use_symm_dispatch_in = use_ep_no_sync_rowwise_symm_dispatch_in(self)
    use_symm_combine_out = use_ep_no_sync_rowwise_symm_combine_out(self)
    use_symm_combine_gather = use_ep_no_sync_rowwise_symm_combine_gather(self)
    with torch.no_grad():
        with nvtx.annotate("RowwiseTBO-A-ConfigCapacity", color="green"):
            requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
            rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
            (
                allowed_splits,
                recv_splits_by_src_local,
                drop_token_cnt,
                keep_from_src_dest_local,
            ) = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                sync_tail_drop_allowed_splits_single_a2a(
                    self,
                    requested_splits,
                    rank_capacity=rank_capacity,
                    return_keep_matrix=True,
                ),
            )
            dispatch_in_cap = num_out_tokens
            dispatch_out_cap = rank_capacity
            combine_in_cap = rank_capacity
            combine_out_cap = num_input_tokens
            accumulate_ep_no_sync_rowwise_metrics(
                self,
                drop_token_cnt=drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    buffers = get_ep_no_sync_buffers(
        self,
        dispatch_in_cap=dispatch_in_cap,
        dispatch_out_cap=dispatch_out_cap,
        combine_in_cap=combine_in_cap,
        combine_out_cap=combine_out_cap,
        d_model=moe_inp.shape[-1],
        dtype=moe_inp.dtype,
        device=moe_inp.device,
        slot_idx=slot_idx,
        need_dispatch_in=use_symm_dispatch_in,
        need_dispatch_meta=False,
        need_combine_meta=False,
        need_combine_out=use_symm_combine_out,
        need_combine_gather=use_symm_combine_gather,
        combine_gather_cap=num_input_tokens,
        combine_gather_top_k=top_k,
    )

    routing_map = local_x_global_routed_expert_indices.view(
        -1, top_k
    ).int()

    with torch.no_grad():
        padded_batch_size_per_local_expert = recv_splits_by_src_local.sum(
            dim=0,
            dtype=torch.long,
        )
        dst_ranks, dst_rows = build_rowwise_route_maps(
            self,
            routing_map=routing_map,
            allowed_splits=allowed_splits,
            keep_from_src_dest_local=keep_from_src_dest_local,
        )
        rowwise_nblocks = self.ep_no_sync_rowwise_nblocks

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    return _NoSyncRowwiseStageAState(
        lane_id=lane_id,
        slot_idx=slot_idx,
        group_name=group_name,
        in_shape=in_shape,
        B=B,
        S=S,
        D=D,
        attn_res_out=attn_res_out,
        shared_out_up=shared_out_up,
        shared_out_gate=shared_out_gate,
        local_x_global_shared_expert_weights=local_x_global_shared_expert_weights,
        local_x_global_routed_expert_weights=local_x_global_routed_expert_weights,
        routed_expert_router_aux_loss_info=routed_expert_router_aux_loss_info,
        padded_batch_size_per_local_expert=padded_batch_size_per_local_expert,
        dst_ranks=dst_ranks,
        dst_rows=dst_rows,
        rowwise_nblocks=rowwise_nblocks,
        use_symm_dispatch_in=use_symm_dispatch_in,
        use_symm_combine_out=use_symm_combine_out,
        use_symm_combine_gather=use_symm_combine_gather,
        buffers=buffers,
        moe_inp=moe_inp,
    )


def ep_no_sync_rowwise_tbo_stage_d_launch(
    block: "MoEFusedV2TransformerBlock",
    a_state: _NoSyncRowwiseStageAState,
) -> _NoSyncRowwiseStageDState:
    self = block
    comm_stream = get_or_init_stream(id=f"ep_no_sync_rowwise_comm_block_{self.block_idx}", priority=0)
    wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())

    with torch.cuda.stream(comm_stream):
        dispatch_out = _DispatchRowwiseAutograd.apply(
            a_state.moe_inp,
            a_state.buffers.dispatch_in if a_state.use_symm_dispatch_in else None,
            a_state.dst_ranks,
            a_state.dst_rows,
            a_state.buffers.dispatch_out,
            a_state.group_name,
            self.ep_pg,
            a_state.rowwise_nblocks,
        )
        dispatch_done_event = record_stream_event_no_compile(comm_stream)

    return _NoSyncRowwiseStageDState(
        lane_id=a_state.lane_id,
        a_state=a_state,
        dispatch_out=dispatch_out,
        dispatch_done_event=dispatch_done_event,
    )


def ep_no_sync_rowwise_tbo_stage_e(
    block: "MoEFusedV2TransformerBlock",
    d_state: _NoSyncRowwiseStageDState,
) -> _NoSyncRowwiseTboPendingContext:
    self = block
    assert self.routed_experts is not None
    wait_event_no_compile(torch.cuda.current_stream(), d_state.dispatch_done_event)

    a_state = d_state.a_state
    global_x_rank_major = self.routed_experts(
        d_state.dispatch_out,
        a_state.padded_batch_size_per_local_expert,
        down_proj_out=a_state.buffers.combine_in.detach(),
        up_proj_input_grad_out=a_state.buffers.dispatch_out.detach(),
    )

    return _NoSyncRowwiseTboPendingContext(
        block=self,
        lane_id=d_state.lane_id,
        a_state=a_state,
        global_x_rank_major=global_x_rank_major,
    )


def ep_no_sync_rowwise_tbo_stage_c_launch(
    block: "MoEFusedV2TransformerBlock",
    pending_ctx: _NoSyncRowwiseTboPendingContext,
) -> _NoSyncRowwiseTboPendingContext:
    del block
    block = pending_ctx.block
    assert block.routed_experts_router is not None
    a_state = pending_ctx.a_state
    comm_stream = get_or_init_stream(id=f"ep_no_sync_rowwise_comm_block_{block.block_idx}", priority=0)
    wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())

    with torch.cuda.stream(comm_stream):
        route_probs = a_state.local_x_global_routed_expert_weights.view(
            -1,
            block.routed_experts_router.top_k,
        )
        combine_out = _RowwiseCombineWeightedAutograd.apply(
            pending_ctx.global_x_rank_major,
            a_state.buffers.combine_in,
            a_state.buffers.combine_out if a_state.use_symm_combine_out else None,
            a_state.buffers.combine_gather if a_state.use_symm_combine_gather else None,
            a_state.dst_ranks,
            a_state.dst_rows,
            route_probs,
            a_state.group_name,
            block.ep_pg,
            a_state.rowwise_nblocks,
        )
        combine_done_event = record_stream_event_no_compile(comm_stream)

    pending_ctx.combine_out = combine_out
    pending_ctx.combine_done_event = combine_done_event
    return pending_ctx


def ep_no_sync_rowwise_tbo_stage_tail(
    block: "MoEFusedV2TransformerBlock",
    pending_ctx: _NoSyncRowwiseTboPendingContext,
) -> torch.Tensor:
    self = block
    a_state = pending_ctx.a_state
    if pending_ctx.combine_done_event is not None:
        wait_event_no_compile(torch.cuda.current_stream(), pending_ctx.combine_done_event)
    if pending_ctx.combine_out is None:
        raise RuntimeError("Rowwise TBO tail called before combine was launched")

    local_x = pending_ctx.combine_out.view(a_state.in_shape)

    if self.shared_experts is not None:
        assert a_state.shared_out_up is not None
        assert a_state.shared_out_gate is not None
        wait_stream_no_compile(
            this_stream=self.get_dense_stream(),
            other_stream=torch.cuda.current_stream(),
        )
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts.forward2(
                a_state.shared_out_up,
                a_state.shared_out_gate,
                a_state.attn_res_out.shape,
            )
            if self.shared_experts_router:
                assert a_state.local_x_global_shared_expert_weights is not None
                _, _, E_s = a_state.local_x_global_shared_expert_weights.shape
                mixed_shared_out = torch.bmm(
                    a_state.local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(
                        a_state.B * a_state.S,
                        1,
                        E_s,
                    ),
                    shared_out.permute(1, 2, 0, 3).contiguous().view(
                        a_state.B * a_state.S,
                        E_s,
                        a_state.D,
                    ),
                ).squeeze(1).view(a_state.B, a_state.S, a_state.D)
            else:
                mixed_shared_out = shared_out.squeeze(0)
    else:
        mixed_shared_out = None

    wait_stream_no_compile(torch.cuda.current_stream(), self.get_dense_stream())
    mlp_out = self._merge_routed_and_shared(local_x, mixed_shared_out)
    final_out = self._res_norm_mlp(a_state.attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        a_state.routed_expert_router_aux_loss_info,
    )


def combined_forward_ep_no_sync_tbo_rowwise(
    block: "MoEFusedV2TransformerBlock",
    x0: torch.Tensor,
    x1_ctx: object,
    x1_is_fresh: bool,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, _NoSyncRowwiseTboPendingContext]:
    self = block
    if x1_is_fresh:
        pending_prev = None
    else:
        if not isinstance(x1_ctx, _NoSyncRowwiseTboPendingContext):
            raise RuntimeError(
                "Expected rowwise no-sync TBO context from previous block, "
                f"got type={type(x1_ctx)}"
            )
        pending_prev = x1_ctx

    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        if pending_prev is not None:
            pending_prev = ep_no_sync_rowwise_tbo_stage_c_launch(
                pending_prev.block,
                pending_prev,
            )
    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        a0 = ep_no_sync_rowwise_tbo_stage_a(
            self,
            x0,
            lane_id=0,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        if x1_is_fresh:
            fresh_ctx = cast(Dict[str, torch.Tensor], x1_ctx)
            block_inp1 = fresh_ctx["x1"]
        else:
            assert pending_prev is not None
            block_inp1 = ep_no_sync_rowwise_tbo_stage_tail(
                pending_prev.block,
                pending_prev,
            )

    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        d0 = ep_no_sync_rowwise_tbo_stage_d_launch(self, a0)
    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        a1 = ep_no_sync_rowwise_tbo_stage_a(
            self,
            block_inp1,
            lane_id=1,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )

    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        d1 = ep_no_sync_rowwise_tbo_stage_d_launch(self, a1)
    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        pending0_pre_c = ep_no_sync_rowwise_tbo_stage_e(self, d0)

    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        pending0_post_c = ep_no_sync_rowwise_tbo_stage_c_launch(self, pending0_pre_c)
    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        pending1_pre_c = ep_no_sync_rowwise_tbo_stage_e(self, d1)

    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        final_out = ep_no_sync_rowwise_tbo_stage_tail(self, pending0_post_c)

    return final_out, pending1_pre_c
