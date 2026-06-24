from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist

from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.utils import get_or_init_stream

from ...moe.utils import (
    record_stream_event_no_compile,
    wait_event_no_compile,
    wait_stream_no_compile,
)
from .comm import _DispatchRowwiseAutograd
from .ep_config import ExpertParallelPath
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity,
    get_cached_ep_no_sync_buffers,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
    get_or_init_ep_no_sync_symm_tensor,
)
from .ep_no_sync_common import sync_tail_drop_allowed_splits_single_a2a
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
)
from .routed_experts import requires_host_side_split_sizes, use_torch_grouped_mm

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


_ROWWISE_WAVE_EXPERIMENTAL_WARNING_EMITTED = False


def _warn_rowwise_wave_experimental() -> None:
    global _ROWWISE_WAVE_EXPERIMENTAL_WARNING_EMITTED
    if _ROWWISE_WAVE_EXPERIMENTAL_WARNING_EMITTED:
        return
    warnings.warn(
        "combined_forward_ep_no_sync_rowwise_wave is experimental: it runs "
        "expert-major waves on top of the rowwise NVSHMEM transport "
        "and prioritizes correctness/measurement over launch efficiency.",
        RuntimeWarning,
        stacklevel=2,
    )
    _ROWWISE_WAVE_EXPERIMENTAL_WARNING_EMITTED = True


def _rowwise_wave_groups(num_local_experts: int, num_waves: int) -> Tuple[Tuple[int, int], ...]:
    if num_local_experts <= 0:
        raise RuntimeError(f"num_local_experts must be > 0, got {num_local_experts}")
    if num_waves <= 0:
        raise RuntimeError(f"rowwise_wave_num_waves must be > 0, got {num_waves}")
    experts_per_wave = max(1, math.ceil(num_local_experts / num_waves))
    groups = []
    for start in range(0, num_local_experts, experts_per_wave):
        groups.append((start, min(start + experts_per_wave, num_local_experts)))
    return tuple(groups)


def _wave_local_expert_mask(
    *,
    num_local_experts: int,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    local_ids = torch.arange(num_local_experts, device=device, dtype=torch.long)
    return (local_ids >= start) & (local_ids < end)


def _wave_global_expert_mask(
    *,
    ep_world_size: int,
    num_local_experts: int,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    local_mask = _wave_local_expert_mask(
        num_local_experts=num_local_experts,
        start=start,
        end=end,
        device=device,
    )
    return local_mask.repeat(ep_world_size)


class _RowwiseGatherSlotsAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        symm_expert_out: torch.Tensor,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
        expert_out_aliases_symm_expert_out: bool,
        pre_barrier: bool,
        post_barrier: bool,
    ) -> torch.Tensor:
        if expert_out.ndim != 2 or symm_expert_out.ndim != 2:
            raise RuntimeError("expert_out/symm_expert_out must be rank-2 [R, D]")
        if expert_out.shape != symm_expert_out.shape:
            raise RuntimeError(
                "expert_out/symm_expert_out shape mismatch: "
                f"{tuple(expert_out.shape)} vs {tuple(symm_expert_out.shape)}"
            )
        if src_ranks.ndim != 2 or src_rows.ndim != 2:
            raise RuntimeError(
                "src_ranks/src_rows must be rank-2 [N, K], "
                f"got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
            )
        if src_ranks.shape != src_rows.shape:
            raise RuntimeError("src_ranks/src_rows shape mismatch")
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        src_ranks_i64 = src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        src_rows_i64 = src_rows if src_rows.dtype == torch.long else src_rows.to(dtype=torch.long)
        if not src_ranks_i64.is_contiguous():
            src_ranks_i64 = src_ranks_i64.contiguous()
        if not src_rows_i64.is_contiguous():
            src_rows_i64 = src_rows_i64.contiguous()

        if not expert_out_aliases_symm_expert_out:
            symm_expert_out.copy_(expert_out)

        num_tokens, top_k = src_ranks_i64.shape
        hidden = symm_expert_out.shape[1]
        gathered_routes = torch.zeros(
            (num_tokens, top_k, hidden),
            device=symm_expert_out.device,
            dtype=symm_expert_out.dtype,
        )
        flat_ranks = src_ranks_i64.reshape(-1, 1).contiguous()
        flat_rows = src_rows_i64.reshape(-1, 1).contiguous()
        flat_gathered_routes = gathered_routes.view(num_tokens * top_k, hidden)

        symm_mem_vdev2d_kernels.rowwise_gather_get(
            symm_expert_out,
            flat_gathered_routes,
            flat_ranks,
            flat_rows,
            group_name,
            nblocks=nblocks,
            pre_barrier=pre_barrier,
            post_barrier=post_barrier,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.symm_expert_out = symm_expert_out
        ctx.save_for_backward(src_ranks_i64, src_rows_i64)
        return gathered_routes

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        src_ranks, src_rows = ctx.saved_tensors
        if grad_out.ndim != 3:
            raise RuntimeError(f"gather slot grad must be rank-3 [N, K, D], got {tuple(grad_out.shape)}")
        if grad_out.shape[:2] != src_ranks.shape:
            raise RuntimeError(
                "gather slot grad route shape mismatch: "
                f"{tuple(grad_out.shape[:2])} vs {tuple(src_ranks.shape)}"
            )
        if grad_out.shape[2] != ctx.symm_expert_out.shape[1]:
            raise RuntimeError(
                "gather slot grad hidden dim mismatch: "
                f"{grad_out.shape[2]} vs {ctx.symm_expert_out.shape[1]}"
            )

        grad_expert_out = None
        if ctx.needs_input_grad[0]:
            grad_slots = grad_out.contiguous().view(src_ranks.numel(), grad_out.shape[2])
            flat_ranks = src_ranks.reshape(-1, 1).contiguous()
            flat_rows = src_rows.reshape(-1, 1).contiguous()
            symm_mem_vdev2d_kernels.rowwise_dispatch_put(
                grad_slots,
                ctx.symm_expert_out,
                flat_ranks,
                flat_rows,
                ctx.group_name,
                nblocks=ctx.nblocks,
            )
            grad_expert_out = ctx.symm_expert_out

        ctx.symm_expert_out = None
        ctx.group = None
        return grad_expert_out, None, None, None, None, None, None, None, None, None


def combined_forward_ep_no_sync_rowwise_wave(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    activation_checkpointing: Optional[bool] = None,
    accumulate_routed_aux_loss_metrics: Optional[bool] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward with EP no-sync using expert-major rowwise waves."""
    _warn_rowwise_wave_experimental()
    self = block
    assert self.routed_experts is not None
    assert self.routed_experts_router is not None
    assert self.ep_enabled
    assert self.num_local_routed_experts is not None
    if self.ep.path != ExpertParallelPath.rowwise_wave:
        raise RuntimeError(
            "combined_forward_ep_no_sync_rowwise_wave requires "
            f"path={ExpertParallelPath.rowwise_wave!r}"
        )
    if self.ep.rowwise_wave_mode != "expert":
        raise RuntimeError(
            "combined_forward_ep_no_sync_rowwise_wave currently supports only "
            f"rowwise_wave_mode='expert', got {self.ep.rowwise_wave_mode!r}"
        )
    if activation_checkpointing:
        raise NotImplementedError("rowwise_wave does not support activation checkpointing yet")
    if (
        self.rowwise_fp8 is not None
        and self.rowwise_fp8.enabled
        and x.device.type == "cuda"
    ):
        raise NotImplementedError("rowwise_wave does not support rowwise FP8 yet")
    assert use_torch_grouped_mm(), "rowwise_wave requires torch.grouped_mm support"
    assert not requires_host_side_split_sizes(), "rowwise_wave does not support host-side split size communication"

    group_name = get_ep_no_sync_group_name(self)
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

    num_out_tokens = local_x_global_routed_expert_indices.numel()
    num_input_tokens = moe_inp.shape[0]
    top_k = self.routed_experts_router.top_k
    rank_capacity = compute_ep_no_sync_rank_capacity(self, num_out_tokens)
    # The compact/event-staged rowwise-wave path is currently forward-only:
    # it uses row-offset grouped GEMM and range-aware combine PUT over shared
    # scratch buffers. The grad-enabled path below keeps the stable-slot
    # semantics but uses normal autograd-capable rowwise dispatch/combine.
    use_compact_forward_path = not torch.is_grad_enabled()
    with torch.no_grad():
        requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
        (
            allowed_splits,
            recv_splits_by_src_local,
            _drop_token_cnt,
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
        accumulate_ep_no_sync_rowwise_metrics(
            self,
            drop_token_cnt=_drop_token_cnt,
            num_out_tokens=num_out_tokens,
            recv_splits_by_src_local=recv_splits_by_src_local,
            rank_capacity=rank_capacity,
        )

    buffers = get_cached_ep_no_sync_buffers(
        self,
        dispatch_in_cap=num_out_tokens,
        dispatch_out_cap=rank_capacity,
        combine_in_cap=rank_capacity,
        combine_out_cap=num_input_tokens,
        d_model=moe_inp.shape[-1],
        dtype=moe_inp.dtype,
        device=moe_inp.device,
        need_dispatch_in=False,
        need_dispatch_meta=False,
        need_dispatch_out=True,
        need_combine_in=True,
        need_combine_meta=False,
        need_combine_out=False,
        need_combine_gather=use_compact_forward_path,
        combine_gather_cap=num_input_tokens if use_compact_forward_path else 0,
        combine_gather_top_k=top_k if use_compact_forward_path else 0,
    )
    if buffers is None:
        buffers = get_ep_no_sync_buffers(
            self,
            dispatch_in_cap=num_out_tokens,
            dispatch_out_cap=rank_capacity,
            combine_in_cap=rank_capacity,
            combine_out_cap=num_input_tokens,
            d_model=moe_inp.shape[-1],
            dtype=moe_inp.dtype,
            device=moe_inp.device,
            need_dispatch_in=False,
            need_dispatch_meta=False,
            need_dispatch_out=True,
            need_combine_in=True,
            need_combine_meta=False,
            need_combine_out=False,
            need_combine_gather=use_compact_forward_path,
            combine_gather_cap=num_input_tokens if use_compact_forward_path else 0,
            combine_gather_top_k=top_k if use_compact_forward_path else 0,
        )

    routing_map = local_x_global_routed_expert_indices.view(-1, top_k).int()
    route_probs = local_x_global_routed_expert_weights.view(-1, top_k)
    rowwise_nblocks = self.ep.rowwise_nblocks

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    if use_compact_forward_path:
        route_out = buffers.combine_gather
    else:
        route_out = torch.zeros(
            (num_input_tokens, top_k, moe_inp.shape[-1]),
            device=moe_inp.device,
            dtype=moe_inp.dtype,
        )
    wave_groups = _rowwise_wave_groups(
        self.num_local_routed_experts,
        self.ep.rowwise_wave_num_waves,
    )

    if use_compact_forward_path:
        with torch.no_grad():
            batch_size_per_local_expert_full = recv_splits_by_src_local.sum(
                dim=0,
                dtype=torch.long,
            )
            local_expert_base_rows_full = (
                torch.cumsum(batch_size_per_local_expert_full, dim=0)
                - batch_size_per_local_expert_full
            )
            dst_ranks_full, dst_rows_full = build_rowwise_route_maps(
                self,
                routing_map=routing_map,
                allowed_splits=allowed_splits,
                keep_from_src_dest_local=keep_from_src_dest_local,
            )
            compact_route_records, compact_wave_offsets = (
                symm_mem_vdev2d_kernels.rowwise_build_compact_route_records(
                    dst_ranks_full,
                    dst_rows_full,
                    routing_map,
                    num_local_experts=self.num_local_routed_experts,
                    num_waves=len(wave_groups),
                    nblocks=rowwise_nblocks,
                )
            )
            inverse_route_meta = get_or_init_ep_no_sync_symm_tensor(
                self,
                name="rowwise_wave_inverse_route_meta",
                shape=(rank_capacity, 2),
                dtype=torch.long,
                device=moe_inp.device,
            )
            symm_mem_vdev2d_kernels.rowwise_inverse_route_meta_put_compact(
                inverse_route_meta,
                compact_route_records,
                compact_wave_offsets,
                src_rank=dist.get_rank(self.ep_pg),
                group_name=group_name,
                nblocks=rowwise_nblocks,
                pre_barrier=False,
                post_barrier=True,
            )
    else:
        batch_size_per_local_expert_full = None
        local_expert_base_rows_full = None
        dst_ranks_full = None
        dst_rows_full = None
        compact_route_records = None
        compact_wave_offsets = None
        inverse_route_meta = None

    if use_compact_forward_path:
        assert batch_size_per_local_expert_full is not None
        assert local_expert_base_rows_full is not None
        assert inverse_route_meta is not None
        assert compact_route_records is not None
        assert compact_wave_offsets is not None

        num_waves = len(wave_groups)
        dispatch_stream = get_or_init_stream(
            id=f"rowwise_wave_dispatch_{group_name}",
            priority=-10,
        )
        compute_stream = torch.cuda.current_stream()
        combine_stream = get_or_init_stream(
            id=f"rowwise_wave_combine_{group_name}",
            priority=-10,
        )
        wait_stream_no_compile(dispatch_stream, compute_stream)
        wait_stream_no_compile(combine_stream, compute_stream)

        wave_row_ranges: list[tuple[torch.Tensor, torch.Tensor]] = []
        dispatch_done_events: list[torch.cuda.Event] = []
        compute_done_events: list[torch.cuda.Event] = []
        for wave_idx, (start, end) in enumerate(wave_groups):
            wave_label = f"rowwise_wave/wave_{wave_idx}/experts_{int(start)}_{int(end)}"
            wave_row_start = local_expert_base_rows_full[int(start)]
            wave_num_rows = batch_size_per_local_expert_full[int(start) : int(end)].sum()
            wave_row_ranges.append((wave_row_start, wave_num_rows))

            with torch.cuda.stream(dispatch_stream):
                with nvtx.annotate(f"{wave_label}/dispatch", color="green"):
                    symm_mem_vdev2d_kernels.rowwise_dispatch_put_compact(
                        moe_inp,
                        buffers.dispatch_out,
                        compact_route_records,
                        compact_wave_offsets,
                        wave_idx,
                        group_name,
                        nblocks=rowwise_nblocks,
                        pre_barrier=False,
                        post_barrier=True,
                    )
                dispatch_done_events.append(record_stream_event_no_compile(dispatch_stream))

            wait_event_no_compile(compute_stream, dispatch_done_events[wave_idx])
            with torch.cuda.stream(compute_stream):
                with nvtx.annotate(wave_label, color="orange"):
                    with nvtx.annotate(f"{wave_label}/experts", color="purple"):
                        self.routed_experts.forward_row_offset(
                            buffers.dispatch_out,
                            batch_size_per_local_expert_full,
                            expert_start=start,
                            expert_end=end,
                            row_start=wave_row_start,
                            down_proj_out=buffers.combine_in.detach(),
                        )
                compute_done_event = record_stream_event_no_compile(compute_stream)
                compute_done_events.append(compute_done_event)

        dispatch_done_event = record_stream_event_no_compile(dispatch_stream)
        wait_event_no_compile(combine_stream, dispatch_done_event)
        for wave_idx, (start, end) in enumerate(wave_groups):
            wave_label = f"rowwise_wave/wave_{wave_idx}/experts_{int(start)}_{int(end)}"
            wave_row_start, wave_num_rows = wave_row_ranges[wave_idx]
            compute_done_event = compute_done_events[wave_idx]
            wait_event_no_compile(combine_stream, compute_done_event)
            with torch.cuda.stream(combine_stream):
                with nvtx.annotate(f"{wave_label}/combine_put", color="red"):
                    symm_mem_vdev2d_kernels.rowwise_combine_put(
                        buffers.combine_in,
                        route_out,
                        inverse_route_meta,
                        wave_row_start,
                        wave_num_rows,
                        group_name,
                        nblocks=rowwise_nblocks,
                        pre_barrier=False,
                        post_barrier=True,
                    )
        wait_stream_no_compile(compute_stream, combine_stream)
    else:
        for wave_idx, (start, end) in enumerate(wave_groups):
            wave_label = f"rowwise_wave/wave_{wave_idx}/experts_{int(start)}_{int(end)}"
            with nvtx.annotate(wave_label, color="orange"):
                with nvtx.annotate(f"{wave_label}/route", color="blue"):
                    with torch.no_grad():
                        local_mask = _wave_local_expert_mask(
                            num_local_experts=self.num_local_routed_experts,
                            start=start,
                            end=end,
                            device=keep_from_src_dest_local.device,
                        )
                        global_mask = _wave_global_expert_mask(
                            ep_world_size=self.ep_world_size,
                            num_local_experts=self.num_local_routed_experts,
                            start=start,
                            end=end,
                            device=allowed_splits.device,
                        )
                        wave_allowed_splits = torch.where(
                            global_mask,
                            allowed_splits,
                            torch.zeros_like(allowed_splits),
                        )
                        wave_keep_from_src_dest_local = torch.where(
                            local_mask.view(1, 1, -1),
                            keep_from_src_dest_local,
                            torch.zeros_like(keep_from_src_dest_local),
                        )
                        wave_recv_splits_by_src_local = torch.where(
                            local_mask.view(1, -1),
                            recv_splits_by_src_local,
                            torch.zeros_like(recv_splits_by_src_local),
                        )
                        batch_size_per_local_expert = wave_recv_splits_by_src_local.sum(
                            dim=0,
                            dtype=torch.long,
                        )
                        dst_ranks, dst_rows = build_rowwise_route_maps(
                            self,
                            routing_map=routing_map,
                            allowed_splits=wave_allowed_splits,
                            keep_from_src_dest_local=wave_keep_from_src_dest_local,
                        )

                with nvtx.annotate(f"{wave_label}/dispatch", color="green"):
                    dispatch_rank_major = _DispatchRowwiseAutograd.apply(
                        moe_inp,
                        None,
                        dst_ranks,
                        dst_rows,
                        buffers.dispatch_out,
                        None,
                        group_name,
                        self.ep_pg,
                        rowwise_nblocks,
                        False,
                        False,
                        True,
                        False,
                    )
                    dispatch_rank_major_for_experts = dispatch_rank_major.clone()
                    down_proj_out = buffers.combine_in.detach()

                with nvtx.annotate(f"{wave_label}/experts", color="purple"):
                    expert_out = self.routed_experts(
                        dispatch_rank_major_for_experts,
                        batch_size_per_local_expert,
                        down_proj_out=down_proj_out,
                        up_proj_input_grad_out=None,
                    )

                with nvtx.annotate(f"{wave_label}/combine_get", color="red"):
                    expert_out_for_gather = expert_out
                    expert_out_aliases_symm_expert_out = (
                        expert_out.data_ptr() == buffers.combine_in.data_ptr()
                        and expert_out.storage_offset() == buffers.combine_in.storage_offset()
                        and tuple(expert_out.shape) == tuple(buffers.combine_in.shape)
                    )
                    wave_route_out = _RowwiseGatherSlotsAutograd.apply(
                        expert_out_for_gather,
                        buffers.combine_in,
                        dst_ranks,
                        dst_rows,
                        group_name,
                        self.ep_pg,
                        rowwise_nblocks,
                        expert_out_aliases_symm_expert_out,
                        True,
                        False,
                    )
                    route_out = route_out + wave_route_out

    if use_compact_forward_path:
        local_x = torch.empty(
            (num_input_tokens, moe_inp.shape[-1]),
            device=moe_inp.device,
            dtype=moe_inp.dtype,
        )
        route_probs_for_reduce = route_probs
        assert dst_ranks_full is not None
        route_probs_for_reduce = torch.where(
            dst_ranks_full >= 0,
            route_probs_for_reduce,
            torch.zeros_like(route_probs_for_reduce),
        )
        if not route_probs_for_reduce.is_contiguous():
            route_probs_for_reduce = route_probs_for_reduce.contiguous()
        symm_mem_vdev2d_kernels.rowwise_reduce_gathered_routes(
            route_out,
            route_probs_for_reduce,
            local_x,
        )
    else:
        local_x = torch.sum(
            route_out * route_probs.to(dtype=route_out.dtype).unsqueeze(-1),
            dim=1,
        )

    wait_stream_no_compile(
        this_stream=self.get_dense_stream(),
        other_stream=torch.cuda.current_stream(),
    )

    if self.shared_experts is not None:
        assert shared_out_up is not None
        assert shared_out_gate is not None

        with torch.cuda.stream(self.get_dense_stream()):
            shared_out = self.shared_experts.forward2(shared_out_up, shared_out_gate, attn_res_out.shape)
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

    final_out = self._res_norm_mlp(attn_res_out, mlp_out)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
        accumulate_metrics=accumulate_routed_aux_loss_metrics,
    )
