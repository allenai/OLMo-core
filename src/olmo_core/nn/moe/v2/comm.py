import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.kernels import symm_mem_vdev2d as symm_mem_vdev2d_kernels
from olmo_core.kernels.mxfp8_tensor import OlmoMXFP8Tensor
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    dot_gathered_rows_mxfp8_with_grad,
    quantize_grouped_2d_to_mxfp8_blocked_fused,
    quantize_row_halves_to_mxfp8,
    quantize_rows_to_mxfp8,
    swiglu_quantize_rows_to_mxfp8,
)
from olmo_core.kernels.scaled_grouped_mm import (
    ScaledGroupedMMPrequantizedLHS,
    _forward_scaled_grouped_mm_mxfp8_prequantized_rhs,
)


def _rowwise_debug_enabled() -> bool:
    if os.getenv("OLMO_ROWWISE_DEBUG_PRINT", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    ranks = os.getenv("OLMO_ROWWISE_DEBUG_RANKS") or os.getenv("OLMO_TBO_DEBUG_RANKS")
    if not ranks or not dist.is_available() or not dist.is_initialized():
        return True
    rank = str(dist.get_rank())
    return rank in {part.strip() for part in ranks.split(",") if part.strip()}


def _rowwise_debug_sync_enabled() -> bool:
    return os.getenv("OLMO_ROWWISE_DEBUG_SYNC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rowwise_rank_tag() -> str:
    if not dist.is_available() or not dist.is_initialized():
        return "rank=? local_rank=?"
    return f"rank={dist.get_rank()} local_rank={os.getenv('LOCAL_RANK', '?')}"


def _rowwise_tensor_desc(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    return f"{name}=tensor"


def _rowwise_debug_print(label: str, phase: str, group_name: str, **tensors: Optional[torch.Tensor]) -> None:
    if not _rowwise_debug_enabled():
        return
    parts = [
        "[OLMO_ROWWISE_DEBUG]",
        _rowwise_rank_tag(),
        f"{phase} {label}",
        f"group={group_name}",
    ]
    parts.extend(_rowwise_tensor_desc(name, tensor) for name, tensor in tensors.items())
    print(" | ".join(str(part) for part in parts), flush=True)


def _rowwise_debug_sync(label: str, device: torch.device) -> None:
    if not _rowwise_debug_sync_enabled():
        return
    if _rowwise_debug_enabled():
        print(
            f"[OLMO_ROWWISE_DEBUG] {_rowwise_rank_tag()} sync {label} device={device}",
            flush=True,
        )
    torch.cuda.synchronize(device)


def _logical_rank2_tensor(shape: Tuple[int, int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # FP8 rowwise comm exposes a high-precision-shaped autograd edge, but the
    # actual payload lives in q/scales. Use one scalar of storage instead of a
    # full capacity-sized mirror buffer.
    return torch.empty((), dtype=dtype, device=device).expand(shape)


def _rowwise_fp8_prequantized_lhs(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    *,
    shape: Tuple[int, int],
    scales_are_blocked: bool = False,
) -> ScaledGroupedMMPrequantizedLHS:
    return ScaledGroupedMMPrequantizedLHS(
        mat_a_q=qdata,
        scale_a=scales,
        mat_a_shape=shape,
        scales_are_blocked=scales_are_blocked,
    )


def _rowwise_fp8_swiglu_backward_impl(up_gate: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    hidden = up_gate.shape[-1] // 2
    up = up_gate[:, :hidden]
    gate = up_gate[:, hidden:]

    gate_f32 = gate.to(torch.float32)
    grad_h_f32 = grad_h.to(torch.float32)
    up_f32 = up.to(torch.float32)

    sig = torch.sigmoid(gate_f32)
    silu_gate = gate_f32 * sig
    dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))

    grad_up = grad_h_f32 * silu_gate
    grad_gate = grad_h_f32 * up_f32 * dsilu
    return torch.cat((grad_up, grad_gate), dim=-1).to(dtype=up_gate.dtype)


_rowwise_fp8_swiglu_backward_compiled = torch.compile(
    _rowwise_fp8_swiglu_backward_impl,
    fullgraph=True,
    dynamic=False,
)


def _rowwise_fp8_swiglu_backward(up_gate: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    # The fused autograd backward itself is a Python custom-autograd callback.
    # Keep the large SwiGLU elementwise chain compiled so the fp32 casts and
    # multiplies do not reappear as many eager aten ops in the backward profile.
    if (
        up_gate.is_cuda
        and grad_h.is_cuda
        and os.getenv("OLMO_MXFP8_COMPILE_SWIGLU_BWD", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        return _rowwise_fp8_swiglu_backward_compiled(up_gate, grad_h)
    return _rowwise_fp8_swiglu_backward_impl(up_gate, grad_h)


def _rowwise_fp8_swiglu_forward_impl(up_gate: torch.Tensor) -> torch.Tensor:
    hidden = up_gate.shape[-1] // 2
    up = up_gate[:, :hidden]
    gate = up_gate[:, hidden:]
    return up * F.silu(gate)


_rowwise_fp8_swiglu_forward_compiled = torch.compile(
    _rowwise_fp8_swiglu_forward_impl,
    fullgraph=True,
    dynamic=False,
)


def _rowwise_fp8_swiglu_forward(up_gate: torch.Tensor) -> torch.Tensor:
    if up_gate.is_cuda:
        return _rowwise_fp8_swiglu_forward_compiled(up_gate)
    return _rowwise_fp8_swiglu_forward_impl(up_gate)


def _rowwise_fp8_debug_up_gate_q_enabled() -> bool:
    return os.getenv("OLMO_ROWWISE_FP8_DEBUG_UP_GATE_Q", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rowwise_fp8_accumulate_wgrad_sink(
    sink,
    grad_anchor: torch.Tensor,
    *,
    transpose_last2: bool,
    squeeze_first_dim: bool,
) -> None:
    if sink is None:
        return
    sink_grad = (
        grad_anchor.transpose(-2, -1).contiguous()
        if transpose_last2
        else grad_anchor
    )
    if squeeze_first_dim:
        if sink_grad.shape[0] != 1:
            raise RuntimeError(
                "wgrad_sink_squeeze_first_dim expects first dim size 1, "
                f"got {tuple(sink_grad.shape)}"
            )
        sink_grad = sink_grad.squeeze(0).contiguous()
    sink.accumulate_wgrad(sink_grad)


def _rowwise_fp8_grouped_wgrad(
    grad_out: torch.Tensor,
    mat_a: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    return F.grouped_mm(
        grad_out.transpose(-2, -1),
        mat_a,
        offs=offs,
    ).transpose(-2, -1)


class _RowwiseFP8DispatchExpertsCombineAutograd(torch.autograd.Function):
    """Fused rowwise FP8 routed-expert region.

    This intentionally owns the whole dispatch -> experts -> combine chain so
    backward can control when shared FP8 comm buffers are read and overwritten.
    The old separate dispatch / grouped-mm / combine autograd path remains as a
    fallback; the default fused mode trades extra SwiGLU compute for lower
    activation memory by saving the SwiGLU input in FP8 and recomputing bf16 h.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        offs: torch.Tensor,
        probs: torch.Tensor,
        dispatch_out_q: torch.Tensor,
        dispatch_out_scales: torch.Tensor,
        combine_in_q: torch.Tensor,
        combine_in_scales: torch.Tensor,
        up_gate_anchor: torch.Tensor,
        down_anchor: torch.Tensor,
        up_gate_prequant,
        up_gate_prequant_t,
        down_prequant,
        down_prequant_t,
        dispatch_out_lease,
        block_size: int,
        use_fast_accum: bool,
        recompute_swiglu: bool,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
        up_wgrad_sink,
        up_wgrad_sink_transpose_last2: bool,
        up_wgrad_sink_squeeze_first_dim: bool,
        down_wgrad_sink,
        down_wgrad_sink_transpose_last2: bool,
        down_wgrad_sink_squeeze_first_dim: bool,
    ) -> torch.Tensor:
        if source_input.ndim != 2:
            raise RuntimeError(
                f"fused rowwise FP8 expects source_input [N, D], got {tuple(source_input.shape)}"
            )
        if dst_ranks.shape != dst_rows.shape:
            raise RuntimeError("dst_ranks/dst_rows must have identical shapes")
        if dst_ranks.ndim != 2 or dst_ranks.shape[0] != source_input.shape[0]:
            raise RuntimeError(
                "dst_ranks/dst_rows must be [N, K] and match source_input first dim"
            )
        if offs.ndim != 1:
            raise RuntimeError(f"offs must be rank-1, got {tuple(offs.shape)}")
        if probs.shape != dst_ranks.shape:
            raise RuntimeError(
                f"probs shape mismatch with dst_ranks/dst_rows: {tuple(probs.shape)} vs {tuple(dst_ranks.shape)}"
            )
        if block_size <= 0 or source_input.shape[1] % int(block_size) != 0:
            raise RuntimeError(
                f"Invalid block_size={block_size} for hidden dim {source_input.shape[1]}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        source_input_contig = source_input if source_input.is_contiguous() else source_input.contiguous()
        dst_ranks_i64 = dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        dst_rows_i64 = dst_rows if dst_rows.dtype == torch.long else dst_rows.to(dtype=torch.long)
        if not dst_ranks_i64.is_contiguous():
            dst_ranks_i64 = dst_ranks_i64.contiguous()
        if not dst_rows_i64.is_contiguous():
            dst_rows_i64 = dst_rows_i64.contiguous()
        probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
        if not probs_f32.is_contiguous():
            probs_f32 = probs_f32.contiguous()
        offs_i32 = offs if offs.dtype == torch.int32 else offs.to(dtype=torch.int32)
        if offs_i32.device != source_input.device:
            offs_i32 = offs_i32.to(device=source_input.device)
        if not offs_i32.is_contiguous():
            offs_i32 = offs_i32.contiguous()

        # Forward starts at bf16 and immediately writes the dispatch payload in
        # MXFP8 rowwise comm format. The high-precision logical tensor below is
        # only an autograd/API placeholder; grouped mm reads the q/scales via
        # the prequantized_lhs object.
        symm_mem_vdev2d_kernels.rowwise_dispatch_put_scaled(
            source_input_contig,
            dispatch_out_q,
            dispatch_out_scales,
            dst_ranks_i64,
            dst_rows_i64,
            group_name,
            block_size=int(block_size),
            nblocks=int(nblocks),
        )

        dispatch_shape = tuple(dispatch_out_q.shape)
        dispatch_logical = _logical_rank2_tensor(
            dispatch_shape,
            dtype=source_input.dtype,
            device=source_input.device,
        )
        dispatch_prequantized_lhs = _rowwise_fp8_prequantized_lhs(
            dispatch_out_q,
            dispatch_out_scales,
            shape=dispatch_shape,
        )
        up_gate = _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
            dispatch_logical,
            up_gate_prequant,
            offs_i32,
            use_fast_accum=bool(use_fast_accum),
            prequantized_lhs=dispatch_prequantized_lhs,
        )

        if bool(recompute_swiglu):
            # Save-Q immediately after the up/gate projection produces up_gate.
            # This keeps the optional activation-memory trade close to the
            # producer; doing it after down/combine made the large read of
            # up_gate much less cache-friendly in profiles.
            # debug_up_gate_q = _rowwise_fp8_debug_up_gate_q_enabled()
            # if debug_up_gate_q:
            #     print(
            #         "[OLMO_ROWWISE_FP8_DEBUG_UP_GATE_Q]",
            #         _rowwise_rank_tag(),
            #         f"shape={tuple(up_gate.shape)}",
            #         f"stride={up_gate.stride()}",
            #         f"contiguous={up_gate.is_contiguous()}",
            #         f"dtype={up_gate.dtype}",
            #         f"device={up_gate.device}",
            #         flush=True,
            #     )
            up_gate_saved, up_gate_scales_saved = quantize_row_halves_to_mxfp8(
                up_gate,
                block_size=int(block_size),
            )
            # if debug_up_gate_q:
            #     print(
            #         "[OLMO_ROWWISE_FP8_DEBUG_UP_GATE_Q]",
            #         _rowwise_rank_tag(),
            #         f"q_shape={tuple(up_gate_saved.shape)}",
            #         f"scales_shape={tuple(up_gate_scales_saved.shape)}",
            #         flush=True,
            #     )
        else:
            up_gate_saved = up_gate
            up_gate_scales_saved = torch.empty((0,), device=up_gate.device, dtype=torch.float8_e8m0fnu)

        # Forward still needs the quantized SwiGLU output for down projection.
        # When recompute_swiglu is enabled, h_q/h_scales are deliberately not
        # saved for backward; down wgrad will recompute bf16 h from up_gate.
        h, h_q, h_scales = swiglu_quantize_rows_to_mxfp8(
            up_gate,
            block_size=int(block_size),
        )
        h_shape = tuple(h.shape)
        h_prequantized_lhs = _rowwise_fp8_prequantized_lhs(
            h_q,
            h_scales,
            shape=h_shape,
        )
        expert_out = _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
            h,
            down_prequant,
            offs_i32,
            use_fast_accum=bool(use_fast_accum),
            prequantized_lhs=h_prequantized_lhs,
        )

        expert_out_contig = expert_out if expert_out.is_contiguous() else expert_out.contiguous()
        quantize_rows_to_mxfp8(
            expert_out_contig,
            block_size=int(block_size),
            out=combine_in_q,
            scales_out=combine_in_scales,
        )

        combine_out = torch.empty(
            (dst_ranks_i64.shape[0], expert_out.shape[1]),
            device=expert_out.device,
            dtype=expert_out.dtype,
        )
        need_grad_probs = ctx.needs_input_grad[4]
        if need_grad_probs:
            gathered_q_saved = torch.empty(
                (dst_ranks_i64.shape[0], dst_ranks_i64.shape[1], combine_in_q.shape[1]),
                device=combine_in_q.device,
                dtype=combine_in_q.dtype,
            )
            gathered_scales_saved = torch.empty(
                (dst_ranks_i64.shape[0], dst_ranks_i64.shape[1], combine_in_scales.shape[1]),
                device=combine_in_scales.device,
                dtype=combine_in_scales.dtype,
            )
        else:
            gathered_q_saved = torch.empty(
                (0, 0, combine_in_q.shape[1]),
                device=combine_in_q.device,
                dtype=combine_in_q.dtype,
            )
            gathered_scales_saved = torch.empty(
                (0, 0, combine_in_scales.shape[1]),
                device=combine_in_scales.device,
                dtype=combine_in_scales.dtype,
            )

        # Combine ends the fused node back in bf16. If probs need grad, the
        # kernel also gathers the routed FP8 rows used to form grad_probs in
        # backward; otherwise we avoid saving that large [N, top_k, D] payload.
        symm_mem_vdev2d_kernels.rowwise_combine_get_scaled(
            combine_in_q,
            combine_in_scales,
            combine_out,
            dst_ranks_i64,
            dst_rows_i64,
            group_name,
            probs=probs_f32,
            block_size=int(block_size),
            nblocks=int(nblocks),
            gathered_q_out=(gathered_q_saved if need_grad_probs else None),
            gathered_scales_out=(gathered_scales_saved if need_grad_probs else None),
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.block_size = int(block_size)
        ctx.use_fast_accum = bool(use_fast_accum)
        ctx.recompute_swiglu = bool(recompute_swiglu)
        ctx.probs_input_dtype = probs.dtype
        ctx.source_input_dtype = source_input.dtype
        ctx.source_input_shape = tuple(source_input.shape)
        ctx.dispatch_shape = dispatch_shape
        ctx.h_shape = h_shape
        ctx.expert_out_shape = tuple(expert_out.shape)
        ctx.up_gate_prequant_t = up_gate_prequant_t
        ctx.down_prequant_t = down_prequant_t
        ctx.dispatch_out_lease = dispatch_out_lease
        ctx.combine_in_q = combine_in_q
        ctx.combine_in_scales = combine_in_scales
        ctx.up_wgrad_sink = up_wgrad_sink
        ctx.up_wgrad_sink_transpose_last2 = bool(up_wgrad_sink_transpose_last2)
        ctx.up_wgrad_sink_squeeze_first_dim = bool(up_wgrad_sink_squeeze_first_dim)
        ctx.down_wgrad_sink = down_wgrad_sink
        ctx.down_wgrad_sink_transpose_last2 = bool(down_wgrad_sink_transpose_last2)
        ctx.down_wgrad_sink_squeeze_first_dim = bool(down_wgrad_sink_squeeze_first_dim)
        if ctx.recompute_swiglu:
            h_q_saved = torch.empty((0,), device=h_q.device, dtype=h_q.dtype)
            h_scales_saved = torch.empty((0,), device=h_scales.device, dtype=h_scales.dtype)
        else:
            h_q_saved = h_q
            h_scales_saved = h_scales
        # Saved tensor order mirrors the destructive-buffer ordering in
        # backward: read h/dispatch qdata before those shared buffers are reused
        # for gradients.
        ctx.save_for_backward(
            dst_ranks_i64,
            dst_rows_i64,
            probs_f32,
            gathered_q_saved,
            gathered_scales_saved,
            up_gate_saved,
            up_gate_scales_saved,
            dispatch_out_q,
            dispatch_out_scales,
            h_q_saved,
            h_scales_saved,
            offs_i32,
        )
        return combine_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (
            dst_ranks,
            dst_rows,
            probs,
            gathered_q_saved,
            gathered_scales_saved,
            up_gate_saved,
            up_gate_scales_saved,
            dispatch_out_q,
            dispatch_out_scales,
            h_q,
            h_scales,
            offs,
        ) = ctx.saved_tensors
        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        grad_probs = None
        valid_mask = (dst_ranks >= 0) & (dst_rows >= 0) & (dst_rows < ctx.expert_out_shape[0])
        if ctx.needs_input_grad[4]:
            grad_probs = dot_gathered_rows_mxfp8_with_grad(
                gathered_q_saved,
                gathered_scales_saved,
                grad_out_contig,
                valid_mask=valid_mask,
                block_size=ctx.block_size,
                out_dtype=ctx.probs_input_dtype,
            )

        # Combine backward first weights each route by its router probability,
        # then dispatches the weighted token grads into combine_in_q/scales.
        # This materialized weighted_flat is the main remaining eager overhead;
        # replacing it with a fused weighted rowwise FP8 dispatch kernel is the
        # next obvious performance cleanup.
        num_rows, top_k = dst_ranks.shape
        hidden = grad_out_contig.shape[1]
        weighted_flat = (
            grad_out_contig.unsqueeze(1)
            * probs.to(dtype=grad_out_contig.dtype).unsqueeze(-1)
        ).reshape(num_rows * top_k, hidden)
        if not weighted_flat.is_contiguous():
            weighted_flat = weighted_flat.contiguous()
        flat_ranks = dst_ranks.reshape(-1, 1).contiguous()
        flat_rows = dst_rows.reshape(-1, 1).contiguous()
        symm_mem_vdev2d_kernels.rowwise_dispatch_put_scaled(
            weighted_flat,
            ctx.combine_in_q,
            ctx.combine_in_scales,
            flat_ranks,
            flat_rows,
            ctx.group_name,
            block_size=ctx.block_size,
            nblocks=ctx.nblocks,
        )

        # Down dgrad consumes the just-written combine_in_q/scales. Do this
        # before dequantizing the same buffer for down wgrad so later changes
        # can safely reuse storage without changing dependency order.
        grad_expert_logical = _logical_rank2_tensor(
            ctx.expert_out_shape,
            dtype=ctx.source_input_dtype,
            device=grad_out.device,
        )
        grad_expert_prequantized_lhs = _rowwise_fp8_prequantized_lhs(
            ctx.combine_in_q,
            ctx.combine_in_scales,
            shape=ctx.expert_out_shape,
        )
        grad_h = _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
            grad_expert_logical,
            ctx.down_prequant_t,
            offs,
            use_fast_accum=True,
            prequantized_lhs=grad_expert_prequantized_lhs,
        )

        # Wgrad paths need bf16 operands. In recompute mode we restore the
        # saved FP8 SwiGLU input, then recompute bf16 h instead of saving the
        # forward h_q/h_scales activation.
        if ctx.recompute_swiglu:
            up_gate = dequantize_rows_from_mxfp8(
                up_gate_saved,
                up_gate_scales_saved,
                block_size=ctx.block_size,
                out_dtype=ctx.source_input_dtype,
            )
            h = _rowwise_fp8_swiglu_forward(up_gate)
        else:
            up_gate = up_gate_saved
            h = dequantize_rows_from_mxfp8(
                h_q,
                h_scales,
                block_size=ctx.block_size,
                out_dtype=grad_h.dtype,
            )
        grad_expert_hp = dequantize_rows_from_mxfp8(
            ctx.combine_in_q,
            ctx.combine_in_scales,
            block_size=ctx.block_size,
            out_dtype=h.dtype,
        )
        grad_down_anchor = _rowwise_fp8_grouped_wgrad(grad_expert_hp, h, offs)
        _rowwise_fp8_accumulate_wgrad_sink(
            ctx.down_wgrad_sink,
            grad_down_anchor,
            transpose_last2=ctx.down_wgrad_sink_transpose_last2,
            squeeze_first_dim=ctx.down_wgrad_sink_squeeze_first_dim,
        )
        if not ctx.needs_input_grad[10]:
            grad_down_anchor = None

        # Up/gate dgrad still needs an FP8 LHS, but wgrad can use the bf16
        # SwiGLU gradient directly. The old split autograd path returned an
        # OlmoMXFP8Tensor here and dequantized it for wgrad; the fused node can
        # avoid that extra Q->DQ round trip.
        grad_up_gate = _rowwise_fp8_swiglu_backward(up_gate, grad_h)
        grad_up_gate_q, grad_up_gate_scales_blocked = quantize_grouped_2d_to_mxfp8_blocked_fused(
            grad_up_gate,
            offs,
            block_size=ctx.block_size,
            zero_unwritten_tail=False,
        )
        grad_up_gate_logical = _logical_rank2_tensor(
            tuple(grad_up_gate.shape),
            dtype=grad_up_gate.dtype,
            device=grad_up_gate.device,
        )
        grad_up_gate_prequantized_lhs = _rowwise_fp8_prequantized_lhs(
            grad_up_gate_q,
            grad_up_gate_scales_blocked,
            shape=tuple(grad_up_gate.shape),
            scales_are_blocked=True,
        )
        grad_dispatch = _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
            grad_up_gate_logical,
            ctx.up_gate_prequant_t,
            offs,
            use_fast_accum=True,
            prequantized_lhs=grad_up_gate_prequantized_lhs,
        )

        # Read dispatch_out_q/scales for up/gate wgrad before overwriting them
        # with grad_dispatch below. This is the main buffer-lifetime invariant
        # that motivated fusing the region into one autograd node.
        dispatch_hp = dequantize_rows_from_mxfp8(
            dispatch_out_q,
            dispatch_out_scales,
            block_size=ctx.block_size,
            out_dtype=grad_up_gate.dtype,
        )
        grad_up_anchor = _rowwise_fp8_grouped_wgrad(grad_up_gate, dispatch_hp, offs)
        _rowwise_fp8_accumulate_wgrad_sink(
            ctx.up_wgrad_sink,
            grad_up_anchor,
            transpose_last2=ctx.up_wgrad_sink_transpose_last2,
            squeeze_first_dim=ctx.up_wgrad_sink_squeeze_first_dim,
        )
        if not ctx.needs_input_grad[9]:
            grad_up_anchor = None

        # Final dispatch backward: reuse the dispatch FP8 buffer for the
        # quantized grad_dispatch payload, then combine-get back to bf16 source
        # grad. The lifetime lease can be released once this is complete.
        grad_dispatch_contig = grad_dispatch if grad_dispatch.is_contiguous() else grad_dispatch.contiguous()
        quantize_rows_to_mxfp8(
            grad_dispatch_contig,
            block_size=ctx.block_size,
            out=dispatch_out_q,
            scales_out=dispatch_out_scales,
        )
        grad_source = torch.empty(
            ctx.source_input_shape,
            device=grad_out.device,
            dtype=ctx.source_input_dtype,
        )
        symm_mem_vdev2d_kernels.rowwise_combine_get_scaled(
            dispatch_out_q,
            dispatch_out_scales,
            grad_source,
            dst_ranks,
            dst_rows,
            ctx.group_name,
            block_size=ctx.block_size,
            nblocks=ctx.nblocks,
        )

        if ctx.dispatch_out_lease is not None:
            ctx.dispatch_out_lease.release()
        ctx.dispatch_out_lease = None
        ctx.group = None
        return (
            grad_source,
            None,
            None,
            None,
            grad_probs,
            None,
            None,
            None,
            None,
            grad_up_anchor,
            grad_down_anchor,
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


class _RowwiseCombineWeightedAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        symm_expert_out: torch.Tensor,
        symm_combine_out: Optional[torch.Tensor],
        symm_combine_out_lease,
        symm_gathered_routes: Optional[torch.Tensor],
        symm_gathered_routes_lease,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        probs: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
        expert_out_aliases_symm_expert_out: bool,
        pre_barrier: bool,
        post_barrier: bool,
    ) -> torch.Tensor:
        if expert_out.ndim != 2 or symm_expert_out.ndim != 2:
            raise RuntimeError("expert_out/symm_expert_out must be rank-2 [R, D]")
        if expert_out.shape[0] != symm_expert_out.shape[0] or expert_out.shape[1] != symm_expert_out.shape[1]:
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
        if probs.shape != src_ranks.shape:
            raise RuntimeError(
                "probs shape mismatch with src_ranks/src_rows: "
                f"{tuple(probs.shape)} vs {tuple(src_ranks.shape)}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        src_ranks_i64 = src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        src_rows_i64 = src_rows if src_rows.dtype == torch.long else src_rows.to(dtype=torch.long)
        if not src_ranks_i64.is_contiguous():
            src_ranks_i64 = src_ranks_i64.contiguous()
        if not src_rows_i64.is_contiguous():
            src_rows_i64 = src_rows_i64.contiguous()

        probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
        if not probs_f32.is_contiguous():
            probs_f32 = probs_f32.contiguous()

        if not expert_out_aliases_symm_expert_out:
            symm_expert_out.copy_(expert_out)

        combine_out_shape = (src_ranks_i64.shape[0], symm_expert_out.shape[1])
        if symm_combine_out is not None:
            if (
                symm_combine_out.ndim != 2
                or symm_combine_out.shape[0] < combine_out_shape[0]
                or symm_combine_out.shape[1] != combine_out_shape[1]
            ):
                raise RuntimeError(
                    "symm_combine_out must be [C, D] with C >= combine rows and matching hidden dim: "
                    f"{tuple(symm_combine_out.shape)} vs {combine_out_shape}"
                )
            if symm_combine_out.dtype != symm_expert_out.dtype:
                raise RuntimeError(
                    "symm_combine_out dtype must match symm_expert_out dtype: "
                    f"{symm_combine_out.dtype} vs {symm_expert_out.dtype}"
                )
            combine_out = symm_combine_out.narrow(0, 0, combine_out_shape[0])
            if not combine_out.is_contiguous():
                raise RuntimeError("symm_combine_out staging view must be contiguous")
        else:
            combine_out = torch.empty(
                combine_out_shape,
                device=symm_expert_out.device,
                dtype=symm_expert_out.dtype,
            )

        need_grad_probs = ctx.needs_input_grad[8]
        gathered_shape = (
            src_ranks_i64.shape[0],
            src_ranks_i64.shape[1],
            symm_expert_out.shape[1],
        )
        symm_gathered_routes_view = None
        if symm_gathered_routes is not None:
            if (
                symm_gathered_routes.ndim != 3
                or symm_gathered_routes.shape[0] < gathered_shape[0]
                or symm_gathered_routes.shape[1] != gathered_shape[1]
                or symm_gathered_routes.shape[2] != gathered_shape[2]
            ):
                raise RuntimeError(
                    "symm_gathered_routes must be [C, K, D] with C >= combine rows: "
                    f"{tuple(symm_gathered_routes.shape)} vs {gathered_shape}"
                )
            if symm_gathered_routes.dtype != symm_expert_out.dtype:
                raise RuntimeError(
                    "symm_gathered_routes dtype must match symm_expert_out dtype: "
                    f"{symm_gathered_routes.dtype} vs {symm_expert_out.dtype}"
                )
            symm_gathered_routes_view = symm_gathered_routes.narrow(0, 0, gathered_shape[0])
            if not symm_gathered_routes_view.is_contiguous():
                raise RuntimeError("symm_gathered_routes staging view must be contiguous")
        if symm_gathered_routes_view is not None:
            gathered_routes_for_kernel = symm_gathered_routes_view
            gathered_routes = (
                gathered_routes_for_kernel
                if need_grad_probs
                else torch.empty(
                    (0, 0, symm_expert_out.shape[1]),
                    device=symm_expert_out.device,
                    dtype=symm_expert_out.dtype,
                )
            )
        elif need_grad_probs:
            gathered_routes_for_kernel = torch.empty(
                gathered_shape,
                device=symm_expert_out.device,
                dtype=symm_expert_out.dtype,
            )
            gathered_routes = gathered_routes_for_kernel
        else:
            gathered_routes_for_kernel = None
            gathered_routes = torch.empty(
                (0, 0, symm_expert_out.shape[1]),
                device=symm_expert_out.device,
                dtype=symm_expert_out.dtype,
            )

        # _rowwise_debug_print(
        #     "rowwise_combine_forward_get",
        #     "enter",
        #     group_name,
        #     expert_out=symm_expert_out,
        #     out=combine_out,
        #     src_ranks=src_ranks_i64,
        #     src_rows=src_rows_i64,
        #     probs=probs_f32,
        #     gathered_out=gathered_routes_for_kernel,
        # )
        symm_mem_vdev2d_kernels.rowwise_combine_get(
            symm_expert_out,
            combine_out,
            src_ranks_i64,
            src_rows_i64,
            group_name,
            probs=probs_f32,
            nblocks=nblocks,
            gathered_out=gathered_routes_for_kernel,
            pre_barrier=pre_barrier,
            post_barrier=post_barrier,
        )
        # _rowwise_debug_sync("rowwise_combine_forward_get", symm_expert_out.device)
        # _rowwise_debug_print(
        #     "rowwise_combine_forward_get",
        #     "exit",
        #     group_name,
        #     expert_out=symm_expert_out,
        #     out=combine_out,
        #     gathered_out=gathered_routes_for_kernel,
        # )
        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.probs_input_dtype = probs.dtype
        ctx.symm_expert_out = symm_expert_out
        ctx.symm_combine_out_lease = symm_combine_out_lease
        ctx.symm_gathered_routes = symm_gathered_routes_view
        ctx.symm_gathered_routes_lease = symm_gathered_routes_lease
        ctx.save_for_backward(src_ranks_i64, src_rows_i64, probs_f32, gathered_routes)
        return combine_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        src_ranks, src_rows, probs, gathered_routes = ctx.saved_tensors
        if grad_out.shape[0] != src_ranks.shape[0]:
            raise RuntimeError(
                "rowwise combine backward grad rows must match src_ranks rows: "
                f"{grad_out.shape[0]} vs {src_ranks.shape[0]}"
            )
        if grad_out.shape[1] != ctx.symm_expert_out.shape[1]:
            raise RuntimeError(
                "rowwise combine backward grad hidden dim must match symm_expert_out hidden dim: "
                f"{grad_out.shape[1]} vs {ctx.symm_expert_out.shape[1]}"
            )
        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        grad_probs = None
        if ctx.needs_input_grad[8]:
            grad_out_for_probs = grad_out_contig
            if grad_out_for_probs.dtype != gathered_routes.dtype:
                grad_out_for_probs = grad_out_for_probs.to(dtype=gathered_routes.dtype)
            grad_probs = torch.bmm(
                gathered_routes,
                grad_out_for_probs.unsqueeze(-1),
            ).squeeze(-1)
            if grad_probs.dtype != ctx.probs_input_dtype:
                grad_probs = grad_probs.to(dtype=ctx.probs_input_dtype)

        grad_expert_out = None
        if ctx.needs_input_grad[0]:
            symm_grad_expert_out = ctx.symm_expert_out
            # rowwise_dispatch_put only writes rows referenced by valid (src_ranks, src_rows).
            # symm_grad_expert_out is a reused shared buffer, so untouched rows can contain stale data.
            # routed_experts must use batch_size_per_expert = recv_splits_by_src_local.sum(dim=0),
            # so grouped-mm backward/wgrad consumes only the valid prefix/segments, not tail capacity rows.
            # Route rows are built densely for kept routes, so consumed rows should be fully overwritten.
            # symm_grad_expert_out.zero_()  <----- Likely not necessary
            dispatch_source = grad_out_contig

            symm_weighted_routes = ctx.symm_gathered_routes
            if symm_weighted_routes is not None:
                num_rows, top_k = src_ranks.shape
                hidden = grad_out_contig.shape[1]
                if tuple(symm_weighted_routes.shape) != (num_rows, top_k, hidden):
                    raise RuntimeError(
                        "symm_gathered_routes view shape must match combine backward weighted routes: "
                        f"{tuple(symm_weighted_routes.shape)} vs {(num_rows, top_k, hidden)}"
                    )
                torch.mul(
                    grad_out_contig.unsqueeze(1),
                    probs.to(dtype=grad_out_contig.dtype).unsqueeze(-1),
                    out=symm_weighted_routes,
                )
                dispatch_source = symm_weighted_routes.view(num_rows * top_k, hidden)
                flat_ranks = src_ranks.reshape(-1, 1).contiguous()
                flat_rows = src_rows.reshape(-1, 1).contiguous()

                # _rowwise_debug_print(
                #     "rowwise_combine_backward_dispatch_put_unweighted",
                #     "enter",
                #     ctx.group_name,
                #     input=dispatch_source,
                #     out=symm_grad_expert_out,
                #     dst_ranks=flat_ranks,
                #     dst_rows=flat_rows,
                # )
                symm_mem_vdev2d_kernels.rowwise_dispatch_put(
                    dispatch_source,
                    symm_grad_expert_out,
                    flat_ranks,
                    flat_rows,
                    ctx.group_name,
                    nblocks=ctx.nblocks,
                )
                # _rowwise_debug_sync("rowwise_combine_backward_dispatch_put_unweighted", symm_grad_expert_out.device)
                # _rowwise_debug_print(
                #     "rowwise_combine_backward_dispatch_put_unweighted",
                #     "exit",
                #     ctx.group_name,
                #     input=dispatch_source,
                #     out=symm_grad_expert_out,
                # )
            else:
                # _rowwise_debug_print(
                #     "rowwise_combine_backward_dispatch_put",
                #     "enter",
                #     ctx.group_name,
                #     input=dispatch_source,
                #     out=symm_grad_expert_out,
                #     dst_ranks=src_ranks,
                #     dst_rows=src_rows,
                #     probs=probs,
                # )
                symm_mem_vdev2d_kernels.rowwise_dispatch_put(
                    dispatch_source,
                    symm_grad_expert_out,
                    src_ranks,
                    src_rows,
                    ctx.group_name,
                    probs=probs,
                    nblocks=ctx.nblocks,
                )
                # _rowwise_debug_sync("rowwise_combine_backward_dispatch_put", symm_grad_expert_out.device)
                # _rowwise_debug_print(
                #     "rowwise_combine_backward_dispatch_put",
                #     "exit",
                #     ctx.group_name,
                #     input=dispatch_source,
                #     out=symm_grad_expert_out,
                # )
            grad_expert_out = symm_grad_expert_out

        ctx.symm_expert_out = None
        ctx.symm_gathered_routes = None
        if ctx.symm_combine_out_lease is not None:
            ctx.symm_combine_out_lease.release()
        ctx.symm_combine_out_lease = None
        if ctx.symm_gathered_routes_lease is not None:
            ctx.symm_gathered_routes_lease.release()
        ctx.symm_gathered_routes_lease = None
        ctx.group = None
        return (
            grad_expert_out,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_probs,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _DispatchRowwiseAutograd(torch.autograd.Function):
    @staticmethod
    # @torch.compiler.disable
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        symm_input: Optional[torch.Tensor],
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_lease,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
        source_input_aliases_symm_input: bool,
        grad_out_aliases_symm_out: bool,
        get_pre_barrier: bool,
        get_post_barrier: bool,
    ) -> torch.Tensor:
        if source_input.ndim != 2:
            raise RuntimeError(
                f"rowwise dispatch expects source_input [N, D], got {tuple(source_input.shape)}"
            )
        if dst_ranks.shape != dst_rows.shape:
            raise RuntimeError("dst_ranks/dst_rows must have identical shapes")
        if dst_ranks.ndim != 2 or dst_ranks.shape[0] != source_input.shape[0]:
            raise RuntimeError(
                "dst_ranks/dst_rows must be [N, K] and match source_input first dim"
            )
        if symm_out.ndim != 2 or symm_out.shape[1] != source_input.shape[1]:
            raise RuntimeError(
                "symm_out must be [C, D] with matching hidden dim to source_input"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        source_input_contig = source_input
        if not source_input_contig.is_contiguous():
            source_input_contig = source_input_contig.contiguous()

        dst_ranks_i64 = dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        dst_rows_i64 = dst_rows if dst_rows.dtype == torch.long else dst_rows.to(dtype=torch.long)
        if not dst_ranks_i64.is_contiguous():
            dst_ranks_i64 = dst_ranks_i64.contiguous()
        if not dst_rows_i64.is_contiguous():
            dst_rows_i64 = dst_rows_i64.contiguous()

        dispatch_source = source_input_contig
        if symm_input is not None:
            if (
                symm_input.ndim != 2
                or symm_input.shape[0] < source_input.shape[0]
                or symm_input.shape[1] != source_input.shape[1]
            ):
                raise RuntimeError(
                    "symm_input must be [C, D] with C >= source_input rows and matching hidden dim: "
                    f"{tuple(symm_input.shape)} vs {tuple(source_input.shape)}"
                )
            if symm_input.dtype != source_input.dtype:
                raise RuntimeError(
                    f"symm_input dtype must match source_input dtype: {symm_input.dtype} vs {source_input.dtype}"
                )

            symm_input_view = symm_input.narrow(0, 0, source_input.shape[0])
            if not symm_input_view.is_contiguous():
                raise RuntimeError("symm_input staging view must be contiguous")

            if not source_input_aliases_symm_input:
                symm_input_view.copy_(source_input_contig)
            dispatch_source = symm_input_view

        # _rowwise_debug_print(
        #     "rowwise_dispatch_forward_put",
        #     "enter",
        #     group_name,
        #     input=dispatch_source,
        #     out=symm_out,
        #     dst_ranks=dst_ranks_i64,
        #     dst_rows=dst_rows_i64,
        # )
        symm_mem_vdev2d_kernels.rowwise_dispatch_put(
            dispatch_source,
            symm_out,
            dst_ranks_i64,
            dst_rows_i64,
            group_name,
            nblocks=nblocks,
        )
        # _rowwise_debug_sync("rowwise_dispatch_forward_put", symm_out.device)
        # _rowwise_debug_print(
        #     "rowwise_dispatch_forward_put",
        #     "exit",
        #     group_name,
        #     input=dispatch_source,
        #     out=symm_out,
        # )

        ctx.group_name = group_name
        ctx.group = group
        ctx.nblocks = int(nblocks)
        ctx.get_pre_barrier = bool(get_pre_barrier)
        ctx.get_post_barrier = bool(get_post_barrier)
        ctx.grad_out_aliases_symm_out = bool(grad_out_aliases_symm_out)
        ctx.symm_input = symm_input
        ctx.symm_out = symm_out
        ctx.symm_out_lease = symm_out_lease
        ctx.save_for_backward(dst_ranks_i64, dst_rows_i64)
        return symm_out

    @staticmethod
    # @torch.compiler.disable
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        dst_ranks, dst_rows = ctx.saved_tensors
        symm_grad_out = ctx.symm_out
        if not ctx.grad_out_aliases_symm_out:
            symm_grad_out.copy_(grad_out)

        grad_input = torch.empty(
            (dst_ranks.shape[0], symm_grad_out.shape[1]),
            device=symm_grad_out.device,
            dtype=symm_grad_out.dtype,
        )
        gathered_grad_out = None
        symm_input = ctx.symm_input
        if symm_input is not None:
            gathered_rows = dst_ranks.numel()
            if (
                symm_input.ndim != 2
                or symm_input.shape[0] < gathered_rows
                or symm_input.shape[1] != symm_grad_out.shape[1]
            ):
                raise RuntimeError(
                    "symm_input must have enough rows to serve as dispatch backward gather scratch: "
                    f"need ({gathered_rows}, {symm_grad_out.shape[1]}), got {tuple(symm_input.shape)}"
                )
            gathered_grad_out = symm_input.narrow(0, 0, gathered_rows).view(
                dst_ranks.shape[0],
                dst_ranks.shape[1],
                symm_grad_out.shape[1],
            )
            if not gathered_grad_out.is_contiguous():
                raise RuntimeError("dispatch backward gather scratch must be contiguous")
        # _rowwise_debug_print(
        #     "rowwise_dispatch_backward_combine_get",
        #     "enter",
        #     ctx.group_name,
        #     expert_out=symm_grad_out,
        #     out=grad_input,
        #     src_ranks=dst_ranks,
        #     src_rows=dst_rows,
        #     gathered_out=gathered_grad_out,
        # )
        symm_mem_vdev2d_kernels.rowwise_combine_get(
            symm_grad_out,
            grad_input,
            dst_ranks,
            dst_rows,
            ctx.group_name,
            nblocks=ctx.nblocks,
            gathered_out=gathered_grad_out,
            pre_barrier=ctx.get_pre_barrier,
            post_barrier=ctx.get_post_barrier,
        )
        # _rowwise_debug_sync("rowwise_dispatch_backward_combine_get", symm_grad_out.device)
        # _rowwise_debug_print(
        #     "rowwise_dispatch_backward_combine_get",
        #     "exit",
        #     ctx.group_name,
        #     expert_out=symm_grad_out,
        #     out=grad_input,
        #     gathered_out=gathered_grad_out,
        # )
        ctx.symm_input = None
        ctx.symm_out = None
        if ctx.symm_out_lease is not None:
            ctx.symm_out_lease.release()
        ctx.symm_out_lease = None
        ctx.group = None
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None


class _DispatchRowwiseFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        dst_ranks: torch.Tensor,
        dst_rows: torch.Tensor,
        symm_out_q: torch.Tensor,
        symm_out_scales: torch.Tensor,
        symm_out_lease,
        block_size: int,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        if source_input.ndim != 2:
            raise RuntimeError(
                f"rowwise FP8 dispatch expects source_input [N, D], got {tuple(source_input.shape)}"
            )
        if dst_ranks.shape != dst_rows.shape:
            raise RuntimeError("dst_ranks/dst_rows must have identical shapes")
        if dst_ranks.ndim != 2 or dst_ranks.shape[0] != source_input.shape[0]:
            raise RuntimeError(
                "dst_ranks/dst_rows must be [N, K] and match source_input first dim"
            )
        if symm_out_q.ndim != 2 or symm_out_q.shape[1] != source_input.shape[1]:
            raise RuntimeError("symm_out_q must be [C, D] with D matching source_input")
        if symm_out_q.device != source_input.device:
            raise RuntimeError(
                f"symm_out_q device mismatch: {symm_out_q.device} vs {source_input.device}"
            )
        if symm_out_scales.device != source_input.device:
            raise RuntimeError(
                "symm_out_scales device mismatch: "
                f"{symm_out_scales.device} vs {source_input.device}"
            )
        if symm_out_q.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"symm_out_q must be float8_e4m3fn, got {symm_out_q.dtype}"
            )
        if block_size <= 0 or symm_out_q.shape[1] % block_size != 0:
            raise RuntimeError(
                f"Invalid block_size={block_size} for hidden dim {symm_out_q.shape[1]}"
            )
        expected_scales_shape = (symm_out_q.shape[0], symm_out_q.shape[1] // block_size)
        if tuple(symm_out_scales.shape) != expected_scales_shape:
            raise RuntimeError(
                "symm_out_scales shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(symm_out_scales.shape)}"
            )
        if symm_out_scales.dtype != torch.float8_e8m0fnu:
            raise RuntimeError(
                f"symm_out_scales must be float8_e8m0fnu, got {symm_out_scales.dtype}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        source_input_contig = source_input if source_input.is_contiguous() else source_input.contiguous()
        dst_ranks_i64 = dst_ranks if dst_ranks.dtype == torch.long else dst_ranks.to(dtype=torch.long)
        dst_rows_i64 = dst_rows if dst_rows.dtype == torch.long else dst_rows.to(dtype=torch.long)
        if not dst_ranks_i64.is_contiguous():
            dst_ranks_i64 = dst_ranks_i64.contiguous()
        if not dst_rows_i64.is_contiguous():
            dst_rows_i64 = dst_rows_i64.contiguous()

        # _rowwise_debug_print(
        #     "rowwise_fp8_dispatch_forward_put_scaled",
        #     "enter",
        #     group_name,
        #     input=source_input_contig,
        #     out_q=symm_out_q,
        #     out_scales=symm_out_scales,
        #     dst_ranks=dst_ranks_i64,
        #     dst_rows=dst_rows_i64,
        # )
        symm_mem_vdev2d_kernels.rowwise_dispatch_put_scaled(
            source_input_contig,
            symm_out_q,
            symm_out_scales,
            dst_ranks_i64,
            dst_rows_i64,
            group_name,
            block_size=int(block_size),
            nblocks=nblocks,
        )
        # _rowwise_debug_sync("rowwise_fp8_dispatch_forward_put_scaled", symm_out_q.device)
        # _rowwise_debug_print(
        #     "rowwise_fp8_dispatch_forward_put_scaled",
        #     "exit",
        #     group_name,
        #     input=source_input_contig,
        #     out_q=symm_out_q,
        #     out_scales=symm_out_scales,
        # )
        # Keep dispatch payload fully FP8 through expert compute.

        ctx.group_name = group_name
        ctx.group = group
        ctx.nblocks = int(nblocks)
        ctx.block_size = int(block_size)
        ctx.logical_out_shape = tuple(symm_out_q.shape)
        ctx.logical_out_dtype = source_input.dtype
        ctx.logical_out_device = source_input.device
        ctx.symm_out_q = symm_out_q
        ctx.symm_out_scales = symm_out_scales
        ctx.symm_out_lease = symm_out_lease
        ctx.save_for_backward(dst_ranks_i64, dst_rows_i64)
        return _logical_rank2_tensor(
            ctx.logical_out_shape,
            dtype=ctx.logical_out_dtype,
            device=ctx.logical_out_device,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        dst_ranks, dst_rows = ctx.saved_tensors
        grad_out_hp = grad_out
        if grad_out_hp.dtype != ctx.logical_out_dtype:
            grad_out_hp = grad_out_hp.to(dtype=ctx.logical_out_dtype)
        if not grad_out_hp.is_contiguous():
            grad_out_hp = grad_out_hp.contiguous()

        quantize_rows_to_mxfp8(
            grad_out_hp,
            block_size=ctx.block_size,
            out=ctx.symm_out_q,
            scales_out=ctx.symm_out_scales,
        )

        grad_input = torch.empty(
            (dst_ranks.shape[0], grad_out_hp.shape[1]),
            device=grad_out_hp.device,
            dtype=ctx.logical_out_dtype,
        )
        # _rowwise_debug_print(
        #     "rowwise_fp8_dispatch_backward_combine_get_scaled",
        #     "enter",
        #     ctx.group_name,
        #     expert_out_q=ctx.symm_out_q,
        #     expert_out_scales=ctx.symm_out_scales,
        #     out=grad_input,
        #     src_ranks=dst_ranks,
        #     src_rows=dst_rows,
        # )
        symm_mem_vdev2d_kernels.rowwise_combine_get_scaled(
            ctx.symm_out_q,
            ctx.symm_out_scales,
            grad_input,
            dst_ranks,
            dst_rows,
            ctx.group_name,
            block_size=ctx.block_size,
            nblocks=ctx.nblocks,
        )
        # _rowwise_debug_sync("rowwise_fp8_dispatch_backward_combine_get_scaled", ctx.symm_out_q.device)
        # _rowwise_debug_print(
        #     "rowwise_fp8_dispatch_backward_combine_get_scaled",
        #     "exit",
        #     ctx.group_name,
        #     expert_out_q=ctx.symm_out_q,
        #     expert_out_scales=ctx.symm_out_scales,
        #     out=grad_input,
        # )
        if ctx.symm_out_lease is not None:
            ctx.symm_out_lease.release()
        ctx.symm_out_lease = None
        ctx.group = None
        return grad_input, None, None, None, None, None, None, None, None, None


class _RowwiseCombineWeightedFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        src_ranks: torch.Tensor,
        src_rows: torch.Tensor,
        probs: torch.Tensor,
        symm_expert_out_q: torch.Tensor,
        symm_expert_out_scales: torch.Tensor,
        block_size: int,
        group_name: str,
        group: dist.ProcessGroup,
        nblocks: int,
    ) -> torch.Tensor:
        if expert_out.ndim != 2:
            raise RuntimeError("expert_out must be rank-2 [R, D]")
        if src_ranks.ndim != 2 or src_rows.ndim != 2:
            raise RuntimeError(
                "src_ranks/src_rows must be rank-2 [N, K], "
                f"got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
            )
        if src_ranks.shape != src_rows.shape:
            raise RuntimeError("src_ranks/src_rows shape mismatch")
        if probs.shape != src_ranks.shape:
            raise RuntimeError(
                f"probs shape mismatch with src_ranks/src_rows: {tuple(probs.shape)} vs {tuple(src_ranks.shape)}"
            )
        if block_size <= 0 or expert_out.shape[1] % block_size != 0:
            raise RuntimeError(
                f"Invalid block_size={block_size} for hidden dim {expert_out.shape[1]}"
            )
        if tuple(symm_expert_out_q.shape) != tuple(expert_out.shape):
            raise RuntimeError(
                "symm_expert_out_q shape mismatch: "
                f"expected {tuple(expert_out.shape)}, got {tuple(symm_expert_out_q.shape)}"
            )
        if symm_expert_out_q.device != expert_out.device:
            raise RuntimeError(
                "symm_expert_out_q device mismatch: "
                f"{symm_expert_out_q.device} vs {expert_out.device}"
            )
        if symm_expert_out_scales.device != expert_out.device:
            raise RuntimeError(
                "symm_expert_out_scales device mismatch: "
                f"{symm_expert_out_scales.device} vs {expert_out.device}"
            )
        if symm_expert_out_q.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"symm_expert_out_q must be float8_e4m3fn, got {symm_expert_out_q.dtype}"
            )
        expected_scales_shape = (
            expert_out.shape[0],
            expert_out.shape[1] // block_size,
        )
        if tuple(symm_expert_out_scales.shape) != expected_scales_shape:
            raise RuntimeError(
                "symm_expert_out_scales shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(symm_expert_out_scales.shape)}"
            )
        if symm_expert_out_scales.dtype != torch.float8_e8m0fnu:
            raise RuntimeError(
                f"symm_expert_out_scales must be float8_e8m0fnu, got {symm_expert_out_scales.dtype}"
            )
        if nblocks < 0:
            raise RuntimeError(f"nblocks must be >= 0 (got {nblocks})")

        src_ranks_i64 = src_ranks if src_ranks.dtype == torch.long else src_ranks.to(dtype=torch.long)
        src_rows_i64 = src_rows if src_rows.dtype == torch.long else src_rows.to(dtype=torch.long)
        if not src_ranks_i64.is_contiguous():
            src_ranks_i64 = src_ranks_i64.contiguous()
        if not src_rows_i64.is_contiguous():
            src_rows_i64 = src_rows_i64.contiguous()

        probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
        if not probs_f32.is_contiguous():
            probs_f32 = probs_f32.contiguous()

        expert_out_contig = expert_out if expert_out.is_contiguous() else expert_out.contiguous()
        quantize_rows_to_mxfp8(
            expert_out_contig,
            block_size=int(block_size),
            out=symm_expert_out_q,
            scales_out=symm_expert_out_scales,
        )

        combine_out = torch.empty(
            (src_ranks_i64.shape[0], expert_out.shape[1]),
            device=expert_out.device,
            dtype=expert_out.dtype,
        )

        need_grad_probs = ctx.needs_input_grad[3]
        if need_grad_probs:
            gathered_q_saved = torch.empty(
                (src_ranks_i64.shape[0], src_ranks_i64.shape[1], symm_expert_out_q.shape[1]),
                device=symm_expert_out_q.device,
                dtype=symm_expert_out_q.dtype,
            )
            gathered_scales_saved = torch.empty(
                (
                    src_ranks_i64.shape[0],
                    src_ranks_i64.shape[1],
                    symm_expert_out_scales.shape[1],
                ),
                device=symm_expert_out_scales.device,
                dtype=symm_expert_out_scales.dtype,
            )
        else:
            gathered_q_saved = torch.empty(
                (0, 0, symm_expert_out_q.shape[1]),
                device=symm_expert_out_q.device,
                dtype=symm_expert_out_q.dtype,
            )
            gathered_scales_saved = torch.empty(
                (0, 0, symm_expert_out_scales.shape[1]),
                device=symm_expert_out_scales.device,
                dtype=symm_expert_out_scales.dtype,
            )

        # _rowwise_debug_print(
        #     "rowwise_fp8_combine_forward_get_scaled",
        #     "enter",
        #     group_name,
        #     expert_out_q=symm_expert_out_q,
        #     expert_out_scales=symm_expert_out_scales,
        #     out=combine_out,
        #     src_ranks=src_ranks_i64,
        #     src_rows=src_rows_i64,
        #     probs=probs_f32,
        #     gathered_q_out=(gathered_q_saved if need_grad_probs else None),
        #     gathered_scales_out=(gathered_scales_saved if need_grad_probs else None),
        # )
        symm_mem_vdev2d_kernels.rowwise_combine_get_scaled(
            symm_expert_out_q,
            symm_expert_out_scales,
            combine_out,
            src_ranks_i64,
            src_rows_i64,
            group_name,
            probs=probs_f32,
            block_size=int(block_size),
            nblocks=nblocks,
            gathered_q_out=(gathered_q_saved if need_grad_probs else None),
            gathered_scales_out=(gathered_scales_saved if need_grad_probs else None),
        )
        # _rowwise_debug_sync("rowwise_fp8_combine_forward_get_scaled", symm_expert_out_q.device)
        # _rowwise_debug_print(
        #     "rowwise_fp8_combine_forward_get_scaled",
        #     "exit",
        #     group_name,
        #     expert_out_q=symm_expert_out_q,
        #     expert_out_scales=symm_expert_out_scales,
        #     out=combine_out,
        # )

        ctx.group = group
        ctx.group_name = group_name
        ctx.nblocks = int(nblocks)
        ctx.block_size = int(block_size)
        ctx.probs_input_dtype = probs.dtype
        ctx.expert_out_shape = tuple(expert_out.shape)
        ctx.expert_out_dtype = expert_out.dtype
        ctx.symm_expert_out_q = symm_expert_out_q
        ctx.symm_expert_out_scales = symm_expert_out_scales
        ctx.save_for_backward(src_ranks_i64, src_rows_i64, probs_f32, gathered_q_saved, gathered_scales_saved)
        return combine_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        src_ranks, src_rows, probs, gathered_q_saved, gathered_scales_saved = ctx.saved_tensors
        if grad_out.shape[0] != src_ranks.shape[0]:
            raise RuntimeError(
                "rowwise combine FP8 backward grad rows must match src_ranks rows: "
                f"{grad_out.shape[0]} vs {src_ranks.shape[0]}"
            )
        if grad_out.shape[1] != ctx.expert_out_shape[1]:
            raise RuntimeError(
                "rowwise combine FP8 backward grad hidden dim must match symm_expert_out hidden dim: "
                f"{grad_out.shape[1]} vs {ctx.expert_out_shape[1]}"
            )
        grad_out_contig = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        grad_probs = None
        valid_mask = (src_ranks >= 0) & (src_rows >= 0) & (src_rows < ctx.expert_out_shape[0])
        if ctx.needs_input_grad[3]:
            grad_probs = dot_gathered_rows_mxfp8_with_grad(
                gathered_q_saved,
                gathered_scales_saved,
                grad_out_contig,
                valid_mask=valid_mask,
                block_size=ctx.block_size,
                out_dtype=ctx.probs_input_dtype,
            )

        grad_expert_out = None
        if ctx.needs_input_grad[0]:
            num_rows, top_k = src_ranks.shape
            hidden = grad_out_contig.shape[1]
            weighted_flat = (
                grad_out_contig.unsqueeze(1)
                * probs.to(dtype=grad_out_contig.dtype).unsqueeze(-1)
            ).reshape(num_rows * top_k, hidden)
            if not weighted_flat.is_contiguous():
                weighted_flat = weighted_flat.contiguous()

            flat_ranks = src_ranks.reshape(-1, 1).contiguous()
            flat_rows = src_rows.reshape(-1, 1).contiguous()
            # _rowwise_debug_print(
            #     "rowwise_fp8_combine_backward_dispatch_put_scaled",
            #     "enter",
            #     ctx.group_name,
            #     input=weighted_flat,
            #     out_q=ctx.symm_expert_out_q,
            #     out_scales=ctx.symm_expert_out_scales,
            #     dst_ranks=flat_ranks,
            #     dst_rows=flat_rows,
            # )
            symm_mem_vdev2d_kernels.rowwise_dispatch_put_scaled(
                weighted_flat,
                ctx.symm_expert_out_q,
                ctx.symm_expert_out_scales,
                flat_ranks,
                flat_rows,
                ctx.group_name,
                block_size=ctx.block_size,
                nblocks=ctx.nblocks,
            )
            # _rowwise_debug_sync("rowwise_fp8_combine_backward_dispatch_put_scaled", ctx.symm_expert_out_q.device)
            # _rowwise_debug_print(
            #     "rowwise_fp8_combine_backward_dispatch_put_scaled",
            #     "exit",
            #     ctx.group_name,
            #     input=weighted_flat,
            #     out_q=ctx.symm_expert_out_q,
            #     out_scales=ctx.symm_expert_out_scales,
            # )
            grad_expert_out = OlmoMXFP8Tensor.from_qdata_scales(
                ctx.symm_expert_out_q,
                ctx.symm_expert_out_scales,
                block_size=ctx.block_size,
                orig_dtype=ctx.expert_out_dtype,
            )

        return grad_expert_out, None, None, grad_probs, None, None, None, None, None, None


class _DispatchVDevAutograd(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable # this is required if dispatch_out is shared across layers, no idea why.
    def forward(  # type: ignore[override]
        ctx,
        source_input: torch.Tensor,
        in_rank_splits: torch.Tensor,
        symm_input: torch.Tensor,
        symm_in_rank_splits: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_rank_splits_offsets: torch.Tensor,
        symm_tmp_rank_splits_offsets: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_rows = source_input.shape[0]
        if source_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"dispatch input rows ({source_rows}) must equal symmetric dispatch input capacity ({symm_input.shape[0]})"
            )

        input_aliases_symm_input = (
            source_input.untyped_storage().data_ptr() == symm_input.untyped_storage().data_ptr()
            and source_input.storage_offset() == symm_input.storage_offset()
            and tuple(source_input.shape) == tuple(symm_input.shape)
            and tuple(source_input.stride()) == tuple(symm_input.stride())
        )
        if not input_aliases_symm_input:
            raise RuntimeError("Not Expected: dispatch source_input should alias symm_input buffer to avoid extra copy")
            symm_input.copy_(source_input)
        if in_rank_splits.dtype != torch.int64:
            symm_in_rank_splits.copy_(in_rank_splits.to(dtype=torch.int64))
        else:
            symm_in_rank_splits.copy_(in_rank_splits)

        work = dist.barrier(
            group=group,
            async_op=True,
            device_ids=[symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_input,
            symm_out,
            symm_in_rank_splits,
            symm_out_rank_splits_offsets,
            group_name,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.source_rows = source_input.shape[0]
        ctx.symm_input = symm_input
        ctx.symm_out = symm_out
        ctx.symm_out_rank_splits_offsets = symm_out_rank_splits_offsets
        ctx.symm_tmp_rank_splits_offsets = symm_tmp_rank_splits_offsets
        out_rank_splits_offsets = symm_out_rank_splits_offsets.clone()
        # Keep metadata directly on ctx to avoid saved_tensors lifetime issues
        # under compiled autograd.
        ctx.forward_out_rank_splits_offsets = out_rank_splits_offsets
        ctx.mark_non_differentiable(out_rank_splits_offsets)
        return symm_out, out_rank_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_rank_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_rank_splits_offsets
        forward_out_rank_splits_offsets = ctx.forward_out_rank_splits_offsets

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"dispatch backward grad rows ({grad_out.shape[0]}) must equal symmetric dispatch grad input capacity ({symm_grad_out.shape[0]})"
            )
        
        symm_grad_out_aliases_grad_out = (
            grad_out.untyped_storage().data_ptr() == symm_grad_out.untyped_storage().data_ptr()
            and grad_out.storage_offset() == symm_grad_out.storage_offset()
            and tuple(grad_out.shape) == tuple(symm_grad_out.shape)
            and tuple(grad_out.stride()) == tuple(symm_grad_out.stride())
        )
        if not symm_grad_out_aliases_grad_out:
            raise RuntimeError("Not Expected: dispatch backward grad_out should alias symm_grad_out buffer to avoid extra copy")


        grad_symm_input = ctx.symm_input
        # Ensure any rows not written by vdev (e.g. dropped-token tail capacity)
        # stay zero without doing a defrag/gather pass.
        grad_symm_input.zero_()
        symm_forward_out_rank_splits_offsets = ctx.symm_out_rank_splits_offsets
        symm_forward_out_rank_splits_offsets.copy_(forward_out_rank_splits_offsets)
        grad_input_rank_splits_offsets = ctx.symm_tmp_rank_splits_offsets

        work = dist.barrier(
            group=ctx.group,
            async_op=True,
            device_ids=[ctx.symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_grad_out,
            grad_symm_input,
            symm_forward_out_rank_splits_offsets[0],
            grad_input_rank_splits_offsets,
            ctx.group_name,
        )

        # 1D vdev layout is contiguous for this path; return directly.
        if grad_symm_input.shape[0] != ctx.source_rows:
            raise RuntimeError(
                f"dispatch backward produced {grad_symm_input.shape[0]} rows, expected {ctx.source_rows}"
            )
        # grad_source_input = grad_symm_input.clone() # no need to copy
        grad_source_input = grad_symm_input
        return grad_source_input, None, None, None, None, None, None, None, None


class _CombineVDevAutograd(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        in_rank_splits: torch.Tensor,
        symm_input: torch.Tensor,
        symm_in_rank_splits: torch.Tensor,
        symm_out: torch.Tensor,
        symm_out_rank_splits_offsets: torch.Tensor,
        symm_tmp_rank_splits_offsets: torch.Tensor,
        group_name: str,
        group: dist.ProcessGroup,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_rows = input.shape[0]
        if input_rows != symm_input.shape[0]:
            raise RuntimeError(
                f"combine input rows ({input_rows}) must equal symmetric combine input capacity ({symm_input.shape[0]})"
            )

        input_aliases_symm_input = (
            input.untyped_storage().data_ptr() == symm_input.untyped_storage().data_ptr()
            and input.storage_offset() == symm_input.storage_offset()
            and tuple(input.shape) == tuple(symm_input.shape)
            and tuple(input.stride()) == tuple(symm_input.stride())
        )
        if not input_aliases_symm_input:
            symm_input.copy_(input)
        if in_rank_splits.dtype != torch.int64:
            symm_in_rank_splits.copy_(in_rank_splits.to(dtype=torch.int64))
        else:
            symm_in_rank_splits.copy_(in_rank_splits)

        work = dist.barrier(
            group=group,
            async_op=True,
            device_ids=[symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_input,
            symm_out,
            symm_in_rank_splits,
            symm_out_rank_splits_offsets,
            group_name,
        )

        ctx.group = group
        ctx.group_name = group_name
        ctx.input_rows = input_rows
        ctx.symm_input = symm_input
        ctx.symm_in_rank_splits = symm_in_rank_splits
        ctx.symm_out = symm_out
        ctx.symm_out_rank_splits_offsets = symm_out_rank_splits_offsets
        ctx.symm_tmp_rank_splits_offsets = symm_tmp_rank_splits_offsets
        out_rank_splits_offsets = symm_out_rank_splits_offsets.clone()
        # Keep metadata directly on ctx to avoid saved_tensors lifetime issues
        # under compiled autograd.
        ctx.forward_out_rank_splits_offsets = out_rank_splits_offsets
        ctx.mark_non_differentiable(out_rank_splits_offsets)

        # the user of the output is going to be unpermute kernel, which will save combine_out for backward.
        # we need to ensure combine_out will not be overwritten, by either:
        # (1) return a new tensor if the combine_out buffer is shared
        # out = torch.empty_like(symm_out)
        # out.copy_(symm_out)
        # or 
        # (2) return the symm_out buffer directly if it's not shared
        out = symm_out
        return out, out_rank_splits_offsets

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_out_rank_splits_offsets: torch.Tensor):  # type: ignore[override]
        del grad_out_rank_splits_offsets
        forward_out_rank_splits_offsets = ctx.forward_out_rank_splits_offsets

        symm_grad_out = ctx.symm_out
        if grad_out.shape[0] != symm_grad_out.shape[0]:
            raise RuntimeError(
                f"combine backward grad rows ({grad_out.shape[0]}) must equal symmetric combine grad input capacity ({symm_grad_out.shape[0]})"
            )
        symm_grad_out_aliases_grad_out = (
            grad_out.untyped_storage().data_ptr() == symm_grad_out.untyped_storage().data_ptr()
            and grad_out.storage_offset() == symm_grad_out.storage_offset()
            and tuple(grad_out.shape) == tuple(symm_grad_out.shape)
            and tuple(grad_out.stride()) == tuple(symm_grad_out.stride())
        )
        if not symm_grad_out_aliases_grad_out:
            # raise RuntimeError("Not Expected: combine backward grad_out should alias symm_grad_out buffer to avoid extra copy")
            # Shared-combine_out mode may route grad through clone() and lose aliasing.
            # Copy into the symmetric buffer in that case.
            symm_grad_out.copy_(grad_out)

        symm_grad_input = ctx.symm_input
        # Do not clear the whole capacity buffer here. Properly routed
        # downstream operations only consume rows described by split metadata,
        # so unwritten tail rows are ignored and zero-fill would add bandwidth
        # cost on the 1D path.
        symm_forward_out_rank_splits = ctx.symm_in_rank_splits
        symm_forward_out_rank_splits.copy_(forward_out_rank_splits_offsets[0])
        grad_input_rank_splits_offsets = ctx.symm_tmp_rank_splits_offsets

        work = dist.barrier(
            group=ctx.group,
            async_op=True,
            device_ids=[ctx.symm_input.device.index],
        )
        assert work is not None
        work.block_current_stream()

        torch.ops.symm_mem.all_to_all_vdev(
            symm_grad_out,
            symm_grad_input,
            symm_forward_out_rank_splits,
            grad_input_rank_splits_offsets,
            ctx.group_name,
        )

        grad_input = symm_grad_input
        # grad_input = torch.empty_like(symm_grad_input)
        # grad_input.copy_(symm_grad_input)
        return grad_input, None, None, None, None, None, None, None, None, None
