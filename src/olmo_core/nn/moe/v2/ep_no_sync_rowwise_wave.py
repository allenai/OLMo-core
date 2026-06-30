from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import nvtx
import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.kernels import grouped_mm_row_offset
from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.kernels.swiglu import swiglu_backward_valid_prefix, swiglu_valid_prefix
from olmo_core.utils import get_or_init_stream

from ...moe.utils import (
    record_stream_event_no_compile,
    wait_event_no_compile,
    wait_stream_no_compile,
)
from .checkpointing import get_rowwise_checkpoint_state
from .comm import _DispatchRowwiseAutograd, _RowwiseCombineWeightedAutograd
from .ep_config import ExpertParallelPath
from .ep_no_sync_buffers import (
    compute_ep_no_sync_rank_capacity,
    get_cached_ep_no_sync_buffers,
    get_ep_no_sync_buffers,
    get_ep_no_sync_group_name,
    get_or_init_ep_no_sync_symm_tensor,
    use_ep_no_sync_rowwise_symm_dispatch_in,
)
from .ep_no_sync_common import (
    rowwise_stage_debug_print,
    rowwise_stage_debug_sync,
    sync_tail_drop_allowed_splits_single_a2a,
)
from .ep_no_sync_rowwise_helpers import (
    accumulate_ep_no_sync_rowwise_metrics,
    build_rowwise_route_maps,
    should_accumulate_ep_no_sync_rowwise_metrics,
)
from .routed_experts import (
    ExpertActivation,
    requires_host_side_split_sizes,
    use_torch_grouped_mm,
)

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


_ROWWISE_WAVE_EXPERIMENTAL_WARNING_EMITTED = False


def _use_rowwise_wave_global_route_meta() -> bool:
    raw = os.getenv("OLMO_MOE_ROWWISE_WAVE_GLOBAL_ROUTE_META", "1").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "OLMO_MOE_ROWWISE_WAVE_GLOBAL_ROUTE_META must be one of "
        "0|1|true|false|yes|no|on|off, got "
        f"{raw!r}"
    )


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


def _rowwise_wave_grouped_wgrad(
    grad_out: torch.Tensor,
    mat_a: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    return F.grouped_mm(mat_a.transpose(-2, -1), grad_out, offs=offs)


class _RowwiseWaveDispatchExpertsCombineAutograd(torch.autograd.Function):
    """Fused BF16 rowwise-wave routed-expert region.

    The fused node keeps the whole MoE transport/MLP/combine region behind
    one autograd boundary so forward and backward can manage wave streams
    explicitly.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        probs: torch.Tensor,
        w_up_gate: torch.Tensor,
        w_down: torch.Tensor,
        dispatch_input: torch.Tensor,
        dispatch_out: torch.Tensor,
        combine_in: torch.Tensor,
        gathered_routes: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        compact_route_records: torch.Tensor,
        compact_wave_offsets: torch.Tensor,
        inverse_route_meta: torch.Tensor,
        batch_size_per_local_expert: torch.Tensor,
        local_expert_base_rows: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
        wave_groups: Tuple[Tuple[int, int], ...],
        recompute_linear1: bool,
        recompute_act: bool,
        dispatch_out_lease,
        gathered_routes_lease,
    ) -> torch.Tensor:
        if source_input.ndim != 2:
            raise RuntimeError(
                f"rowwise-wave expects source_input [N, D], got {tuple(source_input.shape)}"
            )
        if dispatch_input.ndim != 2 or dispatch_input.shape != source_input.shape:
            raise RuntimeError(
                "dispatch_input must be [N, D] matching source_input: "
                f"{tuple(dispatch_input.shape)} vs {tuple(source_input.shape)}"
            )
        if probs.ndim != 2:
            raise RuntimeError(f"probs must be [N, K], got {tuple(probs.shape)}")
        if dst_ranks.shape != probs.shape or dst_rows.shape != probs.shape:
            raise RuntimeError(
                "dst_ranks/dst_rows/probs shape mismatch: "
                f"{tuple(dst_ranks.shape)}, {tuple(dst_rows.shape)}, {tuple(probs.shape)}"
            )
        if dispatch_out.ndim != 2 or dispatch_out.shape[1] != source_input.shape[1]:
            raise RuntimeError(
                "dispatch_out must be [C, D] with D matching source_input: "
                f"{tuple(dispatch_out.shape)} vs {tuple(source_input.shape)}"
            )
        if combine_in.shape != dispatch_out.shape:
            raise RuntimeError(
                f"combine_in must match dispatch_out shape, got {tuple(combine_in.shape)} "
                f"vs {tuple(dispatch_out.shape)}"
            )
        if gathered_routes.ndim != 3 or gathered_routes.shape[:2] != probs.shape:
            raise RuntimeError(
                "gathered_routes must be [N, K, D] matching probs: "
                f"{tuple(gathered_routes.shape)} vs {tuple(probs.shape)}"
            )
        if gathered_routes.shape[2] != source_input.shape[1]:
            raise RuntimeError(
                "gathered_routes hidden dim must match source_input: "
                f"{gathered_routes.shape[2]} vs {source_input.shape[1]}"
            )
        if w_up_gate.ndim != 3 or w_down.ndim != 3:
            raise RuntimeError("w_up_gate and w_down must be grouped expert weights")
        if w_up_gate.shape[0] != w_down.shape[0]:
            raise RuntimeError("w_up_gate and w_down expert counts must match")
        if w_up_gate.shape[2] != source_input.shape[1] or w_down.shape[2] != source_input.shape[1]:
            raise RuntimeError("expert weight d_model mismatch with source_input")
        if w_up_gate.shape[1] != 2 * w_down.shape[1]:
            raise RuntimeError("rowwise-wave fused node currently supports SwiGLU expert weights only")
        if batch_size_per_local_expert.ndim != 1:
            raise RuntimeError("batch_size_per_local_expert must be rank-1")
        if local_expert_base_rows.ndim != 1:
            raise RuntimeError("local_expert_base_rows must be rank-1")
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")
        recompute_linear1 = bool(recompute_linear1)
        recompute_act = bool(recompute_act)

        source_input_contig = source_input if source_input.is_contiguous() else source_input.contiguous()
        dispatch_input_contig = dispatch_input if dispatch_input.is_contiguous() else dispatch_input.contiguous()
        probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
        if not probs_f32.is_contiguous():
            probs_f32 = probs_f32.contiguous()
        dst_ranks_i64 = dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        dst_rows_i64 = dst_rows if dst_rows.dtype == torch.long else dst_rows.to(dtype=torch.long)
        if not dst_ranks_i64.is_contiguous():
            dst_ranks_i64 = dst_ranks_i64.contiguous()
        if not dst_rows_i64.is_contiguous():
            dst_rows_i64 = dst_rows_i64.contiguous()
        batch_sizes_i32 = batch_size_per_local_expert.to(dtype=torch.int32)
        if not batch_sizes_i32.is_contiguous():
            batch_sizes_i32 = batch_sizes_i32.contiguous()
        batch_sizes_i64 = batch_size_per_local_expert.to(dtype=torch.long)
        if not batch_sizes_i64.is_contiguous():
            batch_sizes_i64 = batch_sizes_i64.contiguous()
        local_bases_i64 = local_expert_base_rows.to(dtype=torch.long)
        if not local_bases_i64.is_contiguous():
            local_bases_i64 = local_bases_i64.contiguous()

        num_waves = len(wave_groups)
        if compact_wave_offsets.numel() != num_waves + 1:
            raise RuntimeError(
                "compact_wave_offsets must have one more element than wave_groups: "
                f"{compact_wave_offsets.numel()} vs {num_waves}"
            )

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

        dispatch_done_events: list[torch.cuda.Event] = []
        compute_done_events: list[torch.cuda.Event] = []
        wave_row_ranges: list[tuple[torch.Tensor, torch.Tensor]] = []
        needs_backward = any(ctx.needs_input_grad[:4])
        save_linear1 = needs_backward and not recompute_linear1
        save_act = needs_backward and not recompute_act and ctx.needs_input_grad[3]
        saved_up_gate = (
            torch.empty(
                (dispatch_out.shape[0], w_up_gate.shape[1]),
                device=dispatch_out.device,
                dtype=dispatch_out.dtype,
            )
            if save_linear1
            else source_input_contig.new_empty(0)
        )
        saved_h = (
            torch.empty(
                (dispatch_out.shape[0], w_down.shape[1]),
                device=dispatch_out.device,
                dtype=dispatch_out.dtype,
            )
            if save_act
            else source_input_contig.new_empty(0)
        )
        rowwise_stage_debug_print(
            "rowwise_wave:fused-forward-enter",
            rows=source_input.shape[0],
            d_model=source_input.shape[1],
            waves=num_waves,
            nblocks=nblocks,
        )
        for wave_idx, (start, end) in enumerate(wave_groups):
            wave_label = f"rowwise_wave/wave_{wave_idx}/experts_{int(start)}_{int(end)}"
            wave_row_start = local_bases_i64[int(start)]
            wave_num_rows = batch_sizes_i64[int(start) : int(end)].sum()
            wave_row_ranges.append((wave_row_start, wave_num_rows))

            with torch.cuda.stream(dispatch_stream):
                with nvtx.annotate(f"{wave_label}/dispatch", color="green"):
                    rowwise_stage_debug_print(
                        "rowwise_wave:dispatch-compact-enter",
                        wave=wave_idx,
                        start=int(start),
                        end=int(end),
                    )
                    symm_mem_vdev2d_kernels.rowwise_dispatch_put_compact(
                        dispatch_input_contig,
                        dispatch_out,
                        compact_route_records,
                        compact_wave_offsets,
                        wave_idx,
                        group_name,
                        nblocks=int(nblocks),
                        pre_barrier=False,
                        post_barrier=True,
                    )
                    rowwise_stage_debug_print(
                        "rowwise_wave:dispatch-compact-exit",
                        wave=wave_idx,
                    )
                    rowwise_stage_debug_sync(
                        f"rowwise_wave:dispatch-compact:{wave_idx}",
                        dispatch_out.device,
                    )
                dispatch_done_events.append(record_stream_event_no_compile(dispatch_stream))

            wait_event_no_compile(compute_stream, dispatch_done_events[wave_idx])
            with torch.cuda.stream(compute_stream):
                with nvtx.annotate(wave_label, color="orange"):
                    with nvtx.annotate(f"{wave_label}/experts", color="purple"):
                        rowwise_stage_debug_print(
                            "rowwise_wave:experts-enter",
                            wave=wave_idx,
                            start=int(start),
                            end=int(end),
                        )
                        batch_size_window = batch_sizes_i32[int(start) : int(end)]
                        up_gate = (
                            saved_up_gate
                            if save_linear1
                            else torch.empty(
                                (dispatch_out.shape[0], w_up_gate.shape[1]),
                                device=dispatch_out.device,
                                dtype=dispatch_out.dtype,
                            )
                        )
                        grouped_mm_row_offset(
                            dispatch_out,
                            w_up_gate[int(start) : int(end)].transpose(1, 2),
                            batch_size_window,
                            row_start=wave_row_start,
                            out=up_gate,
                        )
                        h = swiglu_valid_prefix(
                            up_gate,
                            wave_num_rows,
                            start=wave_row_start,
                            out=saved_h if save_act else None,
                        )
                        grouped_mm_row_offset(
                            h,
                            w_down[int(start) : int(end)],
                            batch_size_window,
                            row_start=wave_row_start,
                            out=combine_in,
                        )
                        rowwise_stage_debug_print(
                            "rowwise_wave:experts-exit",
                            wave=wave_idx,
                        )
                        rowwise_stage_debug_sync(
                            f"rowwise_wave:experts:{wave_idx}",
                            combine_in.device,
                        )
                compute_done_events.append(record_stream_event_no_compile(compute_stream))

        dispatch_done_event = record_stream_event_no_compile(dispatch_stream)
        wait_event_no_compile(combine_stream, dispatch_done_event)
        for wave_idx, (start, end) in enumerate(wave_groups):
            wave_label = f"rowwise_wave/wave_{wave_idx}/experts_{int(start)}_{int(end)}"
            wave_row_start, wave_num_rows = wave_row_ranges[wave_idx]
            wait_event_no_compile(combine_stream, compute_done_events[wave_idx])
            with torch.cuda.stream(combine_stream):
                with nvtx.annotate(f"{wave_label}/combine_put", color="red"):
                    rowwise_stage_debug_print(
                        "rowwise_wave:combine-put-enter",
                        wave=wave_idx,
                        start=int(start),
                        end=int(end),
                    )
                    symm_mem_vdev2d_kernels.rowwise_combine_put(
                        combine_in,
                        gathered_routes,
                        inverse_route_meta,
                        wave_row_start,
                        wave_num_rows,
                        group_name,
                        nblocks=int(nblocks),
                        pre_barrier=False,
                        post_barrier=True,
                    )
                    rowwise_stage_debug_print(
                        "rowwise_wave:combine-put-exit",
                        wave=wave_idx,
                    )
                    rowwise_stage_debug_sync(
                        f"rowwise_wave:combine-put:{wave_idx}",
                        gathered_routes.device,
                    )
        wait_stream_no_compile(compute_stream, combine_stream)

        out = torch.empty(
            (source_input.shape[0], source_input.shape[1]),
            device=source_input.device,
            dtype=source_input.dtype,
        )
        rowwise_stage_debug_print("rowwise_wave:reduce-enter")
        symm_mem_vdev2d_kernels.rowwise_reduce_gathered_routes(
            gathered_routes,
            probs_f32,
            out,
            route_ranks=dst_ranks_i64,
        )
        rowwise_stage_debug_print("rowwise_wave:reduce-exit")
        rowwise_stage_debug_sync("rowwise_wave:reduce", out.device)

        if needs_backward:
            ctx.group = group
            ctx.group_name = group_name
            ctx.nblocks = int(nblocks)
            ctx.recompute_linear1 = recompute_linear1
            ctx.recompute_act = recompute_act
            ctx.probs_input_dtype = probs.dtype
            ctx.source_input_shape = tuple(source_input.shape)
            ctx.dispatch_out = dispatch_out
            ctx.combine_in = combine_in
            ctx.gathered_routes = gathered_routes
            ctx.dispatch_out_lease = dispatch_out_lease
            ctx.gathered_routes_lease = gathered_routes_lease
            ctx.wave_groups = tuple((int(start), int(end)) for start, end in wave_groups)
            ctx.save_for_backward(
                probs_f32,
                dst_ranks_i64,
                dst_rows_i64,
                compact_route_records,
                compact_wave_offsets,
                inverse_route_meta,
                batch_sizes_i32,
                batch_sizes_i64,
                local_bases_i64,
                w_up_gate,
                w_down,
                saved_up_gate,
                saved_h,
            )
        else:
            if dispatch_out_lease is not None:
                dispatch_out_lease.release()
            if gathered_routes_lease is not None:
                gathered_routes_lease.release()
        rowwise_stage_debug_print("rowwise_wave:fused-forward-exit")
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (
            probs,
            dst_ranks,
            _dst_rows,
            compact_route_records,
            compact_wave_offsets,
            inverse_route_meta,
            batch_sizes_i32,
            batch_sizes_i64,
            local_bases_i64,
            w_up_gate,
            w_down,
            saved_up_gate,
            saved_h,
        ) = ctx.saved_tensors
        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        dispatch_out = ctx.dispatch_out
        combine_in = ctx.combine_in
        gathered_routes = ctx.gathered_routes
        offs = torch.cumsum(batch_sizes_i32, dim=0, dtype=torch.int32)

        grad_probs = None
        if ctx.needs_input_grad[1]:
            with nvtx.annotate("rowwise_wave/backward/grad_probs", color="blue"):
                grad_out_for_probs = grad_out_contig
                if grad_out_for_probs.dtype != gathered_routes.dtype:
                    grad_out_for_probs = grad_out_for_probs.to(dtype=gathered_routes.dtype)
                grad_probs = torch.bmm(
                    gathered_routes,
                    grad_out_for_probs.unsqueeze(-1),
                ).squeeze(-1)
                grad_probs = torch.where(dst_ranks >= 0, grad_probs, torch.zeros_like(grad_probs))
                if grad_probs.dtype != ctx.probs_input_dtype:
                    grad_probs = grad_probs.to(dtype=ctx.probs_input_dtype)

        need_expert_backward = (
            ctx.needs_input_grad[0]
            or ctx.needs_input_grad[2]
            or ctx.needs_input_grad[3]
        )
        grad_source = None
        grad_w_up_gate = None
        grad_w_down = None
        if need_expert_backward:
            need_grad_h = ctx.needs_input_grad[0] or ctx.needs_input_grad[2]
            need_h_for_wgrad = ctx.needs_input_grad[3]
            have_saved_up_gate = saved_up_gate.numel() != 0
            have_saved_h = saved_h.numel() != 0
            need_up_gate = need_grad_h or (need_h_for_wgrad and not have_saved_h)
            up_gate = (
                saved_up_gate
                if have_saved_up_gate
                else torch.empty(
                    (dispatch_out.shape[0], w_up_gate.shape[1]),
                    device=dispatch_out.device,
                    dtype=dispatch_out.dtype,
                )
                if need_up_gate
                else None
            )
            h = (
                saved_h
                if have_saved_h
                else torch.empty(
                    (dispatch_out.shape[0], w_down.shape[1]),
                    device=dispatch_out.device,
                    dtype=dispatch_out.dtype,
                )
                if need_h_for_wgrad
                else None
            )
            grad_h = None
            if need_grad_h:
                grad_h = torch.empty(
                    (dispatch_out.shape[0], w_down.shape[1]),
                    device=dispatch_out.device,
                    dtype=dispatch_out.dtype,
                )
            grad_up_gate = None
            if need_grad_h:
                grad_up_gate = torch.empty(
                    (dispatch_out.shape[0], w_up_gate.shape[1]),
                    device=dispatch_out.device,
                    dtype=dispatch_out.dtype,
                )
            grad_dispatch = torch.empty_like(dispatch_out) if ctx.needs_input_grad[0] else None

            compute_stream = torch.cuda.current_stream()
            combine_grad_stream = get_or_init_stream(
                id=f"rowwise_wave_backward_combine_grad_{ctx.group_name}",
                priority=-10,
            )
            dispatch_grad_stream = (
                get_or_init_stream(
                    id=f"rowwise_wave_backward_dispatch_grad_{ctx.group_name}",
                    priority=-10,
                )
                if grad_dispatch is not None
                else None
            )
            wait_stream_no_compile(combine_grad_stream, compute_stream)
            if dispatch_grad_stream is not None:
                wait_stream_no_compile(dispatch_grad_stream, compute_stream)
            rowwise_stage_debug_print(
                "rowwise_wave:fused-backward-enter",
                waves=len(ctx.wave_groups),
                nblocks=ctx.nblocks,
            )

            wave_infos = []
            combine_grad_done_events: list[torch.cuda.Event] = []
            for wave_idx, (start, end) in enumerate(ctx.wave_groups):
                wave_label = f"rowwise_wave/backward/wave_{wave_idx}/experts_{start}_{end}"
                wave_row_start = local_bases_i64[start]
                wave_num_rows = batch_sizes_i64[start:end].sum()
                batch_size_window = batch_sizes_i32[start:end]
                wave_infos.append(
                    (
                        wave_idx,
                        start,
                        end,
                        wave_label,
                        wave_row_start,
                        wave_num_rows,
                        batch_size_window,
                    )
                )

                with torch.cuda.stream(combine_grad_stream):
                    with nvtx.annotate(f"{wave_label}/combine_grad_put", color="red"):
                        rowwise_stage_debug_print(
                            "rowwise_wave:backward-combine-grad-put-enter",
                            wave=wave_idx,
                            start=start,
                            end=end,
                        )
                        symm_mem_vdev2d_kernels.rowwise_dispatch_put_compact_weighted(
                            grad_out_contig,
                            combine_in,
                            compact_route_records,
                            compact_wave_offsets,
                            wave_idx,
                            probs,
                            ctx.group_name,
                            nblocks=ctx.nblocks,
                            pre_barrier=False,
                            post_barrier=True,
                        )
                        rowwise_stage_debug_print(
                            "rowwise_wave:backward-combine-grad-put-exit",
                            wave=wave_idx,
                        )
                        rowwise_stage_debug_sync(
                            f"rowwise_wave:backward-combine-grad-put:{wave_idx}",
                            combine_in.device,
                        )
                    combine_grad_done_events.append(record_stream_event_no_compile(combine_grad_stream))

            compute_done_events: list[torch.cuda.Event] = []
            for (
                wave_idx,
                start,
                end,
                wave_label,
                wave_row_start,
                wave_num_rows,
                batch_size_window,
            ) in wave_infos:
                wait_event_no_compile(compute_stream, combine_grad_done_events[wave_idx])
                with torch.cuda.stream(compute_stream):
                    with nvtx.annotate(wave_label, color="orange"):
                        rowwise_stage_debug_print(
                            "rowwise_wave:backward-experts-enter",
                            wave=wave_idx,
                            start=start,
                            end=end,
                        )
                        if need_up_gate and not have_saved_up_gate:
                            assert up_gate is not None
                            with nvtx.annotate(f"{wave_label}/recompute_linear1", color="purple"):
                                grouped_mm_row_offset(
                                    dispatch_out,
                                    w_up_gate[start:end].transpose(1, 2),
                                    batch_size_window,
                                    row_start=wave_row_start,
                                    out=up_gate,
                                )
                        if need_h_for_wgrad and not have_saved_h:
                            assert up_gate is not None
                            assert h is not None
                            with nvtx.annotate(f"{wave_label}/recompute_act", color="yellow"):
                                swiglu_valid_prefix(
                                    up_gate,
                                    wave_num_rows,
                                    start=wave_row_start,
                                    out=h,
                                )

                        if grad_h is not None:
                            with nvtx.annotate(f"{wave_label}/dgrad_down_linear2", color="blue"):
                                grouped_mm_row_offset(
                                    combine_in,
                                    w_down[start:end].transpose(1, 2),
                                    batch_size_window,
                                    row_start=wave_row_start,
                                    out=grad_h,
                                )
                            assert up_gate is not None
                            assert grad_up_gate is not None
                            with nvtx.annotate(f"{wave_label}/swiglu_backward", color="yellow"):
                                swiglu_backward_valid_prefix(
                                    up_gate,
                                    grad_h,
                                    wave_num_rows,
                                    start=wave_row_start,
                                    out=grad_up_gate,
                                )
                            if grad_dispatch is not None:
                                with nvtx.annotate(f"{wave_label}/dgrad_up_gate_linear1", color="blue"):
                                    grouped_mm_row_offset(
                                        grad_up_gate,
                                        w_up_gate[start:end],
                                        batch_size_window,
                                        row_start=wave_row_start,
                                        out=grad_dispatch,
                                    )
                        rowwise_stage_debug_print(
                            "rowwise_wave:backward-experts-exit",
                            wave=wave_idx,
                        )
                        rowwise_stage_debug_sync(
                            f"rowwise_wave:backward-experts:{wave_idx}",
                            dispatch_out.device,
                        )
                    if dispatch_grad_stream is not None:
                        compute_done_events.append(record_stream_event_no_compile(compute_stream))

            if dispatch_grad_stream is not None:
                # The two backward comm stages both use NVSHMEM barriers inside
                # the transport kernels. Keep those collective epochs ordered
                # across streams/ranks: compute may overlap combine-grad PUTs,
                # but dispatch-grad PUTs must not interleave with later
                # combine-grad barrier epochs.
                wait_event_no_compile(dispatch_grad_stream, combine_grad_done_events[-1])
                for (
                    wave_idx,
                    _start,
                    _end,
                    wave_label,
                    wave_row_start,
                    wave_num_rows,
                    _batch_size_window,
                ) in wave_infos:
                    wait_event_no_compile(dispatch_grad_stream, compute_done_events[wave_idx])
                    with torch.cuda.stream(dispatch_grad_stream):
                        with nvtx.annotate(f"{wave_label}/dispatch_grad_put", color="green"):
                            rowwise_stage_debug_print(
                                "rowwise_wave:backward-dispatch-grad-put-enter",
                                wave=wave_idx,
                            )
                            symm_mem_vdev2d_kernels.rowwise_combine_put(
                                grad_dispatch,
                                gathered_routes,
                                inverse_route_meta,
                                wave_row_start,
                                wave_num_rows,
                                ctx.group_name,
                                nblocks=ctx.nblocks,
                                pre_barrier=False,
                                post_barrier=True,
                            )
                            rowwise_stage_debug_print(
                                "rowwise_wave:backward-dispatch-grad-put-exit",
                                wave=wave_idx,
                            )
                            rowwise_stage_debug_sync(
                                f"rowwise_wave:backward-dispatch-grad-put:{wave_idx}",
                                gathered_routes.device,
                            )

            if ctx.needs_input_grad[3]:
                assert h is not None
                with nvtx.annotate("rowwise_wave/backward/wgrad_down", color="purple"):
                    grad_w_down = _rowwise_wave_grouped_wgrad(
                        combine_in,
                        h,
                        offs,
                    )

            if need_grad_h:
                assert grad_up_gate is not None
                if ctx.needs_input_grad[2]:
                    with nvtx.annotate("rowwise_wave/backward/wgrad_up_gate", color="purple"):
                        grad_w_up_gate = _rowwise_wave_grouped_wgrad(
                            grad_up_gate,
                            dispatch_out,
                            offs,
                        ).transpose(1, 2).contiguous()

                if ctx.needs_input_grad[0]:
                    grad_source = torch.empty(
                        ctx.source_input_shape,
                        device=grad_out.device,
                        dtype=grad_out.dtype,
                    )
                    assert dispatch_grad_stream is not None
                    wait_stream_no_compile(compute_stream, dispatch_grad_stream)
                    with nvtx.annotate("rowwise_wave/backward/reduce_grad_source", color="green"):
                        rowwise_stage_debug_print("rowwise_wave:backward-reduce-grad-source-enter")
                        symm_mem_vdev2d_kernels.rowwise_reduce_gathered_routes_unweighted(
                            gathered_routes,
                            grad_source,
                            route_ranks=dst_ranks,
                        )
                        rowwise_stage_debug_print("rowwise_wave:backward-reduce-grad-source-exit")
                        rowwise_stage_debug_sync(
                            "rowwise_wave:backward-reduce-grad-source",
                            grad_source.device,
                        )

        if ctx.dispatch_out_lease is not None:
            ctx.dispatch_out_lease.release()
        if ctx.gathered_routes_lease is not None:
            ctx.gathered_routes_lease.release()
        ctx.dispatch_out_lease = None
        ctx.gathered_routes_lease = None
        ctx.dispatch_out = None
        ctx.combine_in = None
        ctx.gathered_routes = None
        ctx.group = None
        return (
            grad_source,
            grad_probs,
            grad_w_up_gate,
            grad_w_down,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    if activation_checkpointing is None:
        activation_checkpointing, accumulate_routed_aux_loss_metrics = (
            get_rowwise_checkpoint_state()
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
    rowwise_stage_debug_print(
        "rowwise_wave:enter",
        block=self.block_idx,
        shape=tuple(x.shape),
        group=group_name,
    )

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
    rowwise_stage_debug_print(
        "rowwise_wave:router-exit",
        block=self.block_idx,
        numel=int(local_x_global_routed_expert_indices.numel()),
        top_k=self.routed_experts_router.top_k,
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
    use_fused_wave_node = moe_inp.dtype == torch.bfloat16
    use_symm_dispatch_in = use_ep_no_sync_rowwise_symm_dispatch_in(self)
    lease_dispatch_out = torch.is_grad_enabled()
    lease_combine_gather = torch.is_grad_enabled() and use_fused_wave_node
    need_combine_gather = use_fused_wave_node
    with torch.no_grad():
        requested_splits = local_batch_size_per_global_routed_expert.to(dtype=torch.long)
        rowwise_stage_debug_print(
            "rowwise_wave:sync-tail-enter",
            block=self.block_idx,
            rank_capacity=rank_capacity,
            num_out_tokens=num_out_tokens,
            use_symm_dispatch_in=use_symm_dispatch_in,
        )
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
        rowwise_stage_debug_print(
            "rowwise_wave:sync-tail-exit",
            block=self.block_idx,
            rank_capacity=rank_capacity,
        )
        if should_accumulate_ep_no_sync_rowwise_metrics(accumulate_routed_aux_loss_metrics):
            accumulate_ep_no_sync_rowwise_metrics(
                self,
                drop_token_cnt=_drop_token_cnt,
                num_out_tokens=num_out_tokens,
                recv_splits_by_src_local=recv_splits_by_src_local,
                rank_capacity=rank_capacity,
            )

    buffers = None
    if not lease_dispatch_out and not lease_combine_gather:
        rowwise_stage_debug_print(
            "rowwise_wave:get-cached-buffers-enter",
            block=self.block_idx,
            dispatch_out_cap=rank_capacity,
            need_dispatch_in=use_symm_dispatch_in,
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
            need_dispatch_in=use_symm_dispatch_in,
            need_dispatch_meta=False,
            need_dispatch_out=True,
            need_combine_in=True,
            need_combine_meta=False,
            need_combine_out=False,
            need_combine_gather=need_combine_gather,
            combine_gather_cap=num_input_tokens if need_combine_gather else 0,
            combine_gather_top_k=top_k if need_combine_gather else 0,
        )
        rowwise_stage_debug_print(
            "rowwise_wave:get-cached-buffers-exit",
            block=self.block_idx,
            hit=buffers is not None,
        )
    if buffers is None:
        rowwise_stage_debug_print(
            "rowwise_wave:get-buffers-enter",
            block=self.block_idx,
            dispatch_out_cap=rank_capacity,
            need_dispatch_in=use_symm_dispatch_in,
            need_combine_gather=need_combine_gather,
        )
        buffers = get_ep_no_sync_buffers(
            self,
            dispatch_in_cap=num_out_tokens,
            dispatch_out_cap=rank_capacity,
            combine_in_cap=rank_capacity,
            combine_out_cap=num_input_tokens,
            d_model=moe_inp.shape[-1],
            dtype=moe_inp.dtype,
            device=moe_inp.device,
            need_dispatch_in=use_symm_dispatch_in,
            need_dispatch_meta=False,
            need_dispatch_out=True,
            need_combine_in=True,
            need_combine_meta=False,
            need_combine_out=False,
            need_combine_gather=need_combine_gather,
            combine_gather_cap=num_input_tokens if need_combine_gather else 0,
            combine_gather_top_k=top_k if need_combine_gather else 0,
            lease_dispatch_out=lease_dispatch_out,
            lease_combine_gather=lease_combine_gather,
        )
        rowwise_stage_debug_print("rowwise_wave:get-buffers-exit", block=self.block_idx)

    dispatch_input = moe_inp
    if use_symm_dispatch_in:
        rowwise_stage_debug_print("rowwise_wave:stage-dispatch-input-enter", block=self.block_idx)
        dispatch_input = buffers.dispatch_in.narrow(0, 0, moe_inp.shape[0])
        if not dispatch_input.is_contiguous():
            raise RuntimeError("rowwise_wave symmetric dispatch staging view must be contiguous")
        with torch.no_grad():
            dispatch_input.copy_(moe_inp)
        rowwise_stage_debug_print("rowwise_wave:stage-dispatch-input-exit", block=self.block_idx)
        rowwise_stage_debug_sync("rowwise_wave:stage-dispatch-input", dispatch_input.device)

    routing_map = local_x_global_routed_expert_indices.view(-1, top_k).int()
    route_probs = local_x_global_routed_expert_weights.view(-1, top_k)
    rowwise_nblocks = self.ep.rowwise_nblocks

    if self.shared_experts is not None:
        with torch.cuda.stream(self.get_dense_stream()):
            shared_out_up, shared_out_gate = self.shared_experts.forward1(moe_inp.view(B, S, D))
    else:
        shared_out_up, shared_out_gate = None, None

    if use_fused_wave_node:
        if self.routed_experts.activation != ExpertActivation.swiglu:
            raise RuntimeError("rowwise_wave fused autograd currently requires swiglu routed experts")
        if self.routed_experts.b_up_gate is not None or self.routed_experts.b_down is not None:
            raise RuntimeError("rowwise_wave fused autograd does not support expert biases yet")
        if self.routed_experts.rowwise_fp8 is not None and self.routed_experts.rowwise_fp8.enabled:
            raise RuntimeError("rowwise_wave fused autograd does not support rowwise FP8 experts yet")

    route_out = buffers.combine_gather if use_fused_wave_node else None
    wave_groups = _rowwise_wave_groups(
        self.num_local_routed_experts,
        self.ep.rowwise_wave_num_waves,
    )

    with torch.no_grad():
        rowwise_stage_debug_print("rowwise_wave:build-route-enter", block=self.block_idx)
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
        rowwise_stage_debug_print("rowwise_wave:build-route-exit", block=self.block_idx)
        rowwise_stage_debug_print("rowwise_wave:compact-route-enter", block=self.block_idx)
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
        rowwise_stage_debug_print("rowwise_wave:compact-route-exit", block=self.block_idx)
        rowwise_stage_debug_sync("rowwise_wave:compact-route", moe_inp.device)
        rowwise_stage_debug_print("rowwise_wave:inverse-meta-alloc-enter", block=self.block_idx)
        inverse_route_meta = get_or_init_ep_no_sync_symm_tensor(
            self,
            name="rowwise_wave_inverse_route_meta",
            shape=(rank_capacity, 2),
            dtype=torch.long,
            device=moe_inp.device,
        )
        local_ep_rank = dist.get_rank(self.ep_pg)
        if _use_rowwise_wave_global_route_meta():
            ep_world_size = dist.get_world_size(self.ep_pg)
            rowwise_stage_debug_print(
                "rowwise_wave:inverse-meta-global-sync-enter",
                block=self.block_idx,
                world_size=ep_world_size,
            )
            with nvtx.annotate("rowwise_wave/global_route_records_all_gather", color="blue"):
                compact_route_records_flat = compact_route_records.reshape(-1)
                global_route_records_flat = torch.empty(
                    ep_world_size * compact_route_records_flat.numel(),
                    device=compact_route_records.device,
                    dtype=compact_route_records.dtype,
                )
                dist.all_gather_into_tensor(
                    global_route_records_flat,
                    compact_route_records_flat,
                    group=self.ep_pg,
                )
                global_route_records = global_route_records_flat.view(
                    ep_world_size,
                    *compact_route_records.shape,
                )

                compact_wave_offsets_flat = compact_wave_offsets.reshape(-1)
                global_wave_offsets_flat = torch.empty(
                    ep_world_size * compact_wave_offsets_flat.numel(),
                    device=compact_wave_offsets.device,
                    dtype=compact_wave_offsets.dtype,
                )
                dist.all_gather_into_tensor(
                    global_wave_offsets_flat,
                    compact_wave_offsets_flat,
                    group=self.ep_pg,
                )
                global_wave_offsets = global_wave_offsets_flat.view(
                    ep_world_size,
                    compact_wave_offsets.numel(),
                )
            rowwise_stage_debug_print(
                "rowwise_wave:inverse-meta-local-build-enter",
                block=self.block_idx,
            )
            with nvtx.annotate("rowwise_wave/build_inverse_route_meta_local", color="blue"):
                symm_mem_vdev2d_kernels.rowwise_build_inverse_route_meta_from_global_records(
                    inverse_route_meta,
                    global_route_records,
                    global_wave_offsets,
                    local_rank=local_ep_rank,
                    nblocks=rowwise_nblocks,
                )
            rowwise_stage_debug_print(
                "rowwise_wave:inverse-meta-local-build-exit",
                block=self.block_idx,
            )
            rowwise_stage_debug_sync("rowwise_wave:inverse-meta-local-build", moe_inp.device)
        else:
            rowwise_stage_debug_print("rowwise_wave:inverse-meta-put-enter", block=self.block_idx)
            symm_mem_vdev2d_kernels.rowwise_inverse_route_meta_put_compact(
                inverse_route_meta,
                compact_route_records,
                compact_wave_offsets,
                src_rank=local_ep_rank,
                group_name=group_name,
                nblocks=rowwise_nblocks,
                pre_barrier=False,
                post_barrier=True,
                scalar_put=use_symm_dispatch_in,
            )
            rowwise_stage_debug_print("rowwise_wave:inverse-meta-put-exit", block=self.block_idx)
            rowwise_stage_debug_sync("rowwise_wave:inverse-meta-put", moe_inp.device)

    if use_fused_wave_node:
        assert route_out is not None
        local_x = _RowwiseWaveDispatchExpertsCombineAutograd.apply(
            moe_inp,
            route_probs,
            self.routed_experts.w_up_gate,
            self.routed_experts.w_down,
            dispatch_input,
            buffers.dispatch_out,
            buffers.combine_in,
            route_out,
            dst_ranks_full,
            dst_rows_full,
            compact_route_records,
            compact_wave_offsets,
            inverse_route_meta,
            batch_size_per_local_expert_full,
            local_expert_base_rows_full,
            group_name,
            self.ep_pg,
            rowwise_nblocks,
            wave_groups,
            self.ep.rowwise_wave_recompute_linear1,
            self.ep.rowwise_wave_recompute_act,
            buffers.dispatch_out_lease,
            buffers.combine_gather_lease,
        )
    else:
        dispatch_rank_major = _DispatchRowwiseAutograd.apply(
            moe_inp,
            buffers.dispatch_in if use_symm_dispatch_in else None,
            dst_ranks_full,
            dst_rows_full,
            buffers.dispatch_out,
            buffers.dispatch_out_lease,
            group_name,
            self.ep_pg,
            rowwise_nblocks,
            False,
            True,
            True,
            False,
        )
        expert_out = self.routed_experts(
            dispatch_rank_major,
            batch_size_per_local_expert_full,
            down_proj_out=buffers.combine_in.detach(),
            up_proj_input_grad_out=buffers.dispatch_out.detach(),
        )
        expert_out_aliases_symm_expert_out = (
            expert_out.data_ptr() == buffers.combine_in.data_ptr()
            and expert_out.storage_offset() == buffers.combine_in.storage_offset()
            and tuple(expert_out.shape) == tuple(buffers.combine_in.shape)
        )
        local_x = _RowwiseCombineWeightedAutograd.apply(
            expert_out,
            buffers.combine_in,
            None,
            None,
            None,
            None,
            dst_ranks_full,
            dst_rows_full,
            route_probs,
            group_name,
            self.ep_pg,
            rowwise_nblocks,
            expert_out_aliases_symm_expert_out,
            True,
            False,
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
    rowwise_stage_debug_print("rowwise_wave:exit", block=self.block_idx)
    return self._attach_routed_aux_loss(
        final_out,
        routed_expert_router_aux_loss_info,
        accumulate_metrics=accumulate_routed_aux_loss_metrics,
    )
