from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist

from olmo_core.utils import get_or_init_stream

from ...moe.utils import (
    record_stream_event_no_compile,
    wait_event_no_compile,
    wait_stream_no_compile,
)
from .comm import _DispatchRowwiseAutograd, _RowwiseCombineWeightedAutograd
from .checkpointing import is_activation_checkpointing
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


def _tbo_debug_enabled() -> bool:
    if os.getenv("OLMO_TBO_DEBUG_PRINT", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    ranks = os.getenv("OLMO_TBO_DEBUG_RANKS")
    if not ranks or not dist.is_available() or not dist.is_initialized():
        return True
    rank = str(dist.get_rank())
    return rank in {part.strip() for part in ranks.split(",") if part.strip()}


def _tbo_debug_sync_enabled() -> bool:
    return os.getenv("OLMO_TBO_DEBUG_SYNC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tbo_rank_tag() -> str:
    if not dist.is_available() or not dist.is_initialized():
        return "rank=? local_rank=?"
    return f"rank={dist.get_rank()} local_rank={os.getenv('LOCAL_RANK', '?')}"


def _tbo_tensor_desc(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    return f"{name}=tensor"


def _tbo_debug_print(block: "MoEFusedV2TransformerBlock", label: str, **tensors: Optional[torch.Tensor]) -> None:
    if not _tbo_debug_enabled():
        return
    parts = [
        "[OLMO_TBO_DEBUG]",
        _tbo_rank_tag(),
        f"block={block.block_idx}",
        label,
    ]
    parts.extend(_tbo_tensor_desc(name, tensor) for name, tensor in tensors.items())
    print(" | ".join(str(part) for part in parts), flush=True)


def _tbo_debug_sync(block: "MoEFusedV2TransformerBlock", label: str, device: torch.device) -> None:
    if not _tbo_debug_sync_enabled():
        return
    # _tbo_debug_print(block, f"sync-enter {label}")
    torch.cuda.synchronize(device)
    # _tbo_debug_print(block, f"sync-exit {label}")


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
    loss_div_factor: Optional[Union[torch.Tensor, float]]
    mixed_shared_out: Optional[torch.Tensor]
    shared_done_event: Optional[torch.cuda.Event]
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
    # _tbo_debug_print(self, f"A{lane_id}:enter slot={slot_idx} group={group_name}", x=x)
    block_inp = x
    del x

    with nvtx.annotate("RowwiseTBO-A-AttnRouter", color="purple"):
        # _tbo_debug_print(self, f"A{lane_id}:attn-router-enter", block_inp=block_inp)
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
        # _tbo_debug_print(
        #     self,
        #     f"A{lane_id}:attn-router-exit",
        #     attn_res_out=attn_res_out,
        #     moe_inp_3d=moe_inp_3d,
        #     routed_indices=local_x_global_routed_expert_indices,
        #     routed_weights=local_x_global_routed_expert_weights,
        #     requested_splits=local_batch_size_per_global_routed_expert,
        # )
        # _tbo_debug_sync(self, f"A{lane_id}:attn-router", moe_inp_3d.device)

    mixed_shared_out: Optional[torch.Tensor] = None
    shared_done_event: Optional[torch.cuda.Event] = None

    in_shape = moe_inp_3d.size()
    moe_inp = moe_inp_3d.view(-1, in_shape[-1])
    del moe_inp_3d

    num_out_tokens = local_x_global_routed_expert_indices.numel()
    num_input_tokens = moe_inp.shape[0]
    top_k = self.routed_experts_router.top_k
    activation_checkpointing = is_activation_checkpointing()
    use_symm_dispatch_in = use_ep_no_sync_rowwise_symm_dispatch_in(self)
    use_symm_combine_out = (
        (not activation_checkpointing)
        and use_ep_no_sync_rowwise_symm_combine_out(self)
    )
    use_symm_combine_gather = (
        (not activation_checkpointing)
        and use_ep_no_sync_rowwise_symm_combine_gather(self)
    )
    lease_lifetime_buffers = torch.is_grad_enabled() and not activation_checkpointing
    with torch.no_grad():
        with nvtx.annotate("RowwiseTBO-A-ConfigCapacity", color="green"):
            requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
            rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
            # _tbo_debug_print(
            #     self,
            #     f"A{lane_id}:capacity-sync-enter rank_capacity={rank_capacity} num_out_tokens={num_out_tokens}",
            #     requested_splits=requested_splits,
            # )
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
            # _tbo_debug_print(
            #     self,
            #     f"A{lane_id}:capacity-sync-exit rank_capacity={rank_capacity}",
            #     allowed_splits=allowed_splits,
            #     recv_splits=recv_splits_by_src_local,
            #     keep_matrix=keep_from_src_dest_local,
            # )
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

    # _tbo_debug_print(
    #     self,
    #     (
    #         f"A{lane_id}:buffers-enter slot={slot_idx} dispatch_in_cap={dispatch_in_cap} "
    #         f"dispatch_out_cap={dispatch_out_cap} combine_in_cap={combine_in_cap} "
    #         f"combine_out_cap={combine_out_cap} symm_dispatch_in={use_symm_dispatch_in} "
    #         f"symm_combine_out={use_symm_combine_out} symm_gather={use_symm_combine_gather}"
    #     ),
    #     moe_inp=moe_inp,
    # )
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
        lease_dispatch_out=lease_lifetime_buffers,
        lease_combine_out=use_symm_combine_out and lease_lifetime_buffers,
        lease_combine_gather=use_symm_combine_gather and lease_lifetime_buffers,
    )
    # _tbo_debug_print(
    #     self,
    #     f"A{lane_id}:buffers-exit slot={slot_idx}",
    #     dispatch_in=buffers.dispatch_in,
    #     dispatch_out=buffers.dispatch_out,
    #     combine_in=buffers.combine_in,
    #     combine_out=buffers.combine_out,
    #     combine_gather=buffers.combine_gather,
    # )

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
        # _tbo_debug_print(
        #     self,
        #     f"A{lane_id}:route-maps-exit nblocks={rowwise_nblocks}",
        #     routing_map=routing_map,
        #     dst_ranks=dst_ranks,
        #     dst_rows=dst_rows,
        #     padded_batch_size=padded_batch_size_per_local_expert,
        # )

    # _tbo_debug_print(self, f"A{lane_id}:exit", moe_inp=moe_inp)
    return _NoSyncRowwiseStageAState(
        lane_id=lane_id,
        slot_idx=slot_idx,
        group_name=group_name,
        in_shape=in_shape,
        B=B,
        S=S,
        D=D,
        attn_res_out=attn_res_out,
        loss_div_factor=loss_div_factor,
        mixed_shared_out=mixed_shared_out,
        shared_done_event=shared_done_event,
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
    comm_stream = get_or_init_stream(id=f"ep_no_sync_rowwise_comm_{a_state.group_name}", priority=-10)
    # _tbo_debug_print(self, f"D{a_state.lane_id}:wait-stream-enter", moe_inp=a_state.moe_inp)
    wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())
    # _tbo_debug_print(self, f"D{a_state.lane_id}:wait-stream-exit")

    with torch.cuda.stream(comm_stream):
        # _tbo_debug_print(
        #     self,
        #     f"D{a_state.lane_id}:dispatch-enter",
        #     moe_inp=a_state.moe_inp,
        #     dispatch_in=(a_state.buffers.dispatch_in if a_state.use_symm_dispatch_in else None),
        #     dst_ranks=a_state.dst_ranks,
        #     dst_rows=a_state.dst_rows,
        #     dispatch_out=a_state.buffers.dispatch_out,
        # )
        source_input_aliases_symm_input = False
        grad_out_aliases_symm_out = True
        dispatch_out = _DispatchRowwiseAutograd.apply(
            a_state.moe_inp,
            a_state.buffers.dispatch_in if getattr(a_state, "use_symm_dispatch_in", False) else None,
            a_state.dst_ranks,
            a_state.dst_rows,
            a_state.buffers.dispatch_out,
            getattr(a_state.buffers, "dispatch_out_lease", None),
            a_state.group_name,
            self.ep_pg,
            a_state.rowwise_nblocks,
            source_input_aliases_symm_input,
            grad_out_aliases_symm_out,
            True,
            True,
        )
        # _tbo_debug_print(self, f"D{a_state.lane_id}:dispatch-exit", dispatch_out=dispatch_out)
        # _tbo_debug_sync(self, f"D{a_state.lane_id}:dispatch", dispatch_out.device)
        dispatch_done_event = record_stream_event_no_compile(comm_stream)
        # _tbo_debug_print(self, f"D{a_state.lane_id}:event-recorded")

    return _NoSyncRowwiseStageDState(
        lane_id=a_state.lane_id,
        a_state=a_state,
        dispatch_out=dispatch_out,
        dispatch_done_event=dispatch_done_event,
    )


def ep_no_sync_rowwise_tbo_stage_shared_launch(
    block: "MoEFusedV2TransformerBlock",
    a_state: _NoSyncRowwiseStageAState,
) -> None:
    self = block
    if self.shared_experts is None or a_state.shared_done_event is not None:
        return

    dense_stream = self.get_dense_stream()
    # _tbo_debug_print(self, f"E{a_state.lane_id}:dense-wait-enter")
    wait_stream_no_compile(
        this_stream=dense_stream,
        other_stream=torch.cuda.current_stream(),
    )
    # _tbo_debug_print(self, f"E{a_state.lane_id}:dense-wait-exit")

    B = a_state.B
    S = a_state.S
    D = a_state.D
    moe_inp_3d = a_state.moe_inp.view(a_state.in_shape)
    with torch.cuda.stream(dense_stream):
        if self.shared_experts_router:
            # _tbo_debug_print(self, f"E{a_state.lane_id}:shared-router-enter", moe_inp_3d=moe_inp_3d)
            (
                local_x_global_shared_expert_weights,
                _,
                _,
                _,
            ) = self.shared_experts_router(
                moe_inp_3d,
                True,
                loss_div_factor=a_state.loss_div_factor,
            )
        else:
            local_x_global_shared_expert_weights = None
        # _tbo_debug_print(
        #     self,
        #     f"E{a_state.lane_id}:shared-router-exit",
        #     shared_weights=local_x_global_shared_expert_weights,
        # )
        # _tbo_debug_print(self, f"E{a_state.lane_id}:shared-forward-enter", moe_inp_3d=moe_inp_3d)
        shared_out = self.shared_experts(moe_inp_3d)
        if self.shared_experts_router:
            assert local_x_global_shared_expert_weights is not None
            _, _, E_s = local_x_global_shared_expert_weights.shape
            mixed_shared_out = torch.bmm(
                local_x_global_shared_expert_weights.to(shared_out.dtype).reshape(B * S, 1, E_s),
                shared_out.permute(1, 2, 0, 3).contiguous().view(B * S, E_s, D),
            ).squeeze(1).view(B, S, D)
        else:
            mixed_shared_out = shared_out.squeeze(0)
        # _tbo_debug_print(self, f"E{a_state.lane_id}:shared-forward-exit", mixed_shared_out=mixed_shared_out)
        a_state.mixed_shared_out = mixed_shared_out
        a_state.shared_done_event = record_stream_event_no_compile(dense_stream)


def ep_no_sync_rowwise_tbo_stage_e(
    block: "MoEFusedV2TransformerBlock",
    d_state: _NoSyncRowwiseStageDState,
) -> _NoSyncRowwiseTboPendingContext:
    self = block
    assert self.routed_experts is not None
    ep_no_sync_rowwise_tbo_stage_shared_launch(self, d_state.a_state)
    # _tbo_debug_print(self, f"E{d_state.lane_id}:wait-dispatch-enter", dispatch_out=d_state.dispatch_out)
    wait_event_no_compile(torch.cuda.current_stream(), d_state.dispatch_done_event)
    # _tbo_debug_print(self, f"E{d_state.lane_id}:wait-dispatch-exit")

    a_state = d_state.a_state
    # _tbo_debug_print(
    #     self,
    #     f"E{d_state.lane_id}:experts-enter",
    #     dispatch_out=d_state.dispatch_out,
    #     combine_in=a_state.buffers.combine_in,
    #     dispatch_out_buffer=a_state.buffers.dispatch_out,
    #     padded_batch_size=a_state.padded_batch_size_per_local_expert,
    # )
    global_x_rank_major = self.routed_experts(
        d_state.dispatch_out,
        a_state.padded_batch_size_per_local_expert,
        down_proj_out=a_state.buffers.combine_in.detach(),
        up_proj_input_grad_out=a_state.buffers.dispatch_out.detach(),
    )
    # _tbo_debug_print(self, f"E{d_state.lane_id}:experts-exit", global_x_rank_major=global_x_rank_major)
    # _tbo_debug_sync(self, f"E{d_state.lane_id}:experts", global_x_rank_major.device)

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
    comm_stream = get_or_init_stream(id=f"ep_no_sync_rowwise_comm_{a_state.group_name}", priority=-10)
    # _tbo_debug_print(block, f"C{pending_ctx.lane_id}:wait-stream-enter", global_x_rank_major=pending_ctx.global_x_rank_major)
    wait_stream_no_compile(this_stream=comm_stream, other_stream=torch.cuda.current_stream())
    # _tbo_debug_print(block, f"C{pending_ctx.lane_id}:wait-stream-exit")

    with torch.cuda.stream(comm_stream):
        route_probs = a_state.local_x_global_routed_expert_weights.view(
            -1,
            block.routed_experts_router.top_k,
        )
        # _tbo_debug_print(
        #     block,
        #     f"C{pending_ctx.lane_id}:combine-enter",
        #     global_x_rank_major=pending_ctx.global_x_rank_major,
        #     combine_in=a_state.buffers.combine_in,
        #     combine_out=(a_state.buffers.combine_out if a_state.use_symm_combine_out else None),
        #     combine_gather=(a_state.buffers.combine_gather if a_state.use_symm_combine_gather else None),
        #     dst_ranks=a_state.dst_ranks,
        #     dst_rows=a_state.dst_rows,
        #     route_probs=route_probs,
        # )
        expert_out_aliases_symm_expert_out = True
        combine_out = _RowwiseCombineWeightedAutograd.apply(
            pending_ctx.global_x_rank_major,
            a_state.buffers.combine_in,
            a_state.buffers.combine_out if a_state.use_symm_combine_out else None,
            (
                a_state.buffers.combine_out_lease
                if a_state.use_symm_combine_out
                else None
            ),
            a_state.buffers.combine_gather if a_state.use_symm_combine_gather else None,
            (
                a_state.buffers.combine_gather_lease
                if a_state.use_symm_combine_gather
                else None
            ),
            a_state.dst_ranks,
            a_state.dst_rows,
            route_probs,
            a_state.group_name,
            block.ep_pg,
            a_state.rowwise_nblocks,
            expert_out_aliases_symm_expert_out,
            True,
            True,
        )
        # _tbo_debug_print(block, f"C{pending_ctx.lane_id}:combine-exit", combine_out=combine_out)
        # _tbo_debug_sync(block, f"C{pending_ctx.lane_id}:combine", combine_out.device)
        combine_done_event = record_stream_event_no_compile(comm_stream)
        # _tbo_debug_print(block, f"C{pending_ctx.lane_id}:event-recorded")

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
        # _tbo_debug_print(self, f"T{pending_ctx.lane_id}:wait-combine-enter", combine_out=pending_ctx.combine_out)
        wait_event_no_compile(torch.cuda.current_stream(), pending_ctx.combine_done_event)
        # _tbo_debug_print(self, f"T{pending_ctx.lane_id}:wait-combine-exit")
    if pending_ctx.combine_out is None:
        raise RuntimeError("Rowwise TBO tail called before combine was launched")

    local_x = pending_ctx.combine_out.view(a_state.in_shape)
    # _tbo_debug_print(self, f"T{pending_ctx.lane_id}:tail-enter", local_x=local_x)

    if a_state.shared_done_event is not None:
        # _tbo_debug_print(self, f"T{pending_ctx.lane_id}:wait-shared-enter", mixed_shared_out=a_state.mixed_shared_out)
        wait_event_no_compile(torch.cuda.current_stream(), a_state.shared_done_event)
        # _tbo_debug_print(self, f"T{pending_ctx.lane_id}:wait-shared-exit")

    mlp_out = self._merge_routed_and_shared(local_x, a_state.mixed_shared_out)
    final_out = self._res_norm_mlp(a_state.attn_res_out, mlp_out)
    # _tbo_debug_print(self, f"T{pending_ctx.lane_id}:tail-exit", final_out=final_out)
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
            # _tbo_debug_print(self, "combined:prev-C-launch-enter")
            pending_prev = ep_no_sync_rowwise_tbo_stage_c_launch(
                pending_prev.block,
                pending_prev,
            )
            # _tbo_debug_print(self, "combined:prev-C-launch-exit")
    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        # _tbo_debug_print(self, f"combined:A0-enter x1_is_fresh={x1_is_fresh}", x0=x0)
        a0 = ep_no_sync_rowwise_tbo_stage_a(
            self,
            x0,
            lane_id=0,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )
        # _tbo_debug_print(self, "combined:A0-exit")

    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        if x1_is_fresh:
            fresh_ctx = cast(Dict[str, torch.Tensor], x1_ctx)
            block_inp1 = fresh_ctx["x1"]
            # _tbo_debug_print(self, "combined:x1-fresh", block_inp1=block_inp1)
        else:
            assert pending_prev is not None
            # _tbo_debug_print(pending_prev.block, "combined:prev-tail-enter")
            block_inp1 = ep_no_sync_rowwise_tbo_stage_tail(
                pending_prev.block,
                pending_prev,
            )
            # _tbo_debug_print(pending_prev.block, "combined:prev-tail-exit", block_inp1=block_inp1)

    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        # _tbo_debug_print(self, "combined:D0-enter")
        d0 = ep_no_sync_rowwise_tbo_stage_d_launch(self, a0)
        # _tbo_debug_print(self, "combined:D0-exit")
    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        # _tbo_debug_print(self, "combined:A1-enter", block_inp1=block_inp1)
        a1 = ep_no_sync_rowwise_tbo_stage_a(
            self,
            block_inp1,
            lane_id=1,
            loss_div_factor=loss_div_factor,
            **kwargs,
        )
        # _tbo_debug_print(self, "combined:A1-exit")

    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        # _tbo_debug_print(self, "combined:D1-enter")
        d1 = ep_no_sync_rowwise_tbo_stage_d_launch(self, a1)
        # _tbo_debug_print(self, "combined:D1-exit")
    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        # _tbo_debug_print(self, "combined:E0-enter")
        pending0_pre_c = ep_no_sync_rowwise_tbo_stage_e(self, d0)
        # _tbo_debug_print(self, "combined:E0-exit")

    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        # _tbo_debug_print(self, "combined:C0-enter")
        pending0_post_c = ep_no_sync_rowwise_tbo_stage_c_launch(self, pending0_pre_c)
        # _tbo_debug_print(self, "combined:C0-exit")
    with nvtx.annotate("RowwiseTBO-1", color="orange"):
        # _tbo_debug_print(self, "combined:E1-enter")
        pending1_pre_c = ep_no_sync_rowwise_tbo_stage_e(self, d1)
        # _tbo_debug_print(self, "combined:E1-exit")

    with nvtx.annotate("RowwiseTBO-0", color="purple"):
        # _tbo_debug_print(self, "combined:T0-enter")
        final_out = ep_no_sync_rowwise_tbo_stage_tail(self, pending0_post_c)
        # _tbo_debug_print(self, "combined:T0-exit", final_out=final_out)

    return final_out, pending1_pre_c
