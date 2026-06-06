from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import nvtx
except ImportError:
    from olmo_core._nvtx import nvtx

from ..doc_utils import beta_feature
from .mxfp8_tensor import OlmoMXFP8Tensor
from .mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    grouped_scales_to_mxfp8_blocked,
    quantize_grouped_weight_3d_to_mxfp8_blocked,
    quantize_rows_to_mxfp8,
)

try:
    _MXFP8_RECIPE_BLOCKWISE_1X32 = [int(F.ScalingType.BlockWise1x32)]
    _MXFP8_SWIZZLE = (
        [int(F.SwizzleType.NO_SWIZZLE)]
        if torch.version.hip
        else [int(F.SwizzleType.SWIZZLE_32_4_4)]
    )
except AttributeError:
    # Fallback values for torch builds where enum symbols are not exposed.
    _MXFP8_RECIPE_BLOCKWISE_1X32 = [3]
    _MXFP8_SWIZZLE = [0] if torch.version.hip else [1]


@dataclass(frozen=True)
class ScaledGroupedMMPrequantizedLHS:
    mat_a_q: Tensor
    scale_a: Tensor
    mat_a_shape: Tuple[int, ...]  # logically (M, K)
    scales_are_blocked: bool = False


@dataclass(frozen=True)
class ScaledGroupedMMPrequantizedRHS:
    mat_b_q: Tensor
    scale_b: Tensor
    mat_b_shape: Tuple[int, ...]  # logically (E, N, K)
    mat_b_version: int = -1


def _tensor_version(x: Tensor) -> int:
    try:
        return int(x._version)
    except Exception:
        return -1


def _prequantized_lhs_tensors_for_backward(
    prequantized_lhs: ScaledGroupedMMPrequantizedLHS,
) -> tuple[Tensor, Tensor]:
    return prequantized_lhs.mat_a_q, prequantized_lhs.scale_a


def _to_col_major_last2(x: Tensor) -> Tensor:
    # Keep shape while forcing column-major storage for the trailing 2 dims.
    # _scaled_grouped_mm_v2 requires mat_b to be transposed (K,N with strides [1, K]).
    if x.stride(-2) == 1 and x.stride(-1) == x.shape[-2]:
        return x
    return x.transpose(-2, -1).contiguous().transpose(-2, -1)


@torch.no_grad()
def prequantize_scaled_grouped_mm_rhs(
    mat_b: Tensor,
    *,
    check_mat_b_version: bool = True,
) -> ScaledGroupedMMPrequantizedRHS:
    with nvtx.annotate("prequantize_scaled_grouped_mm_rhs", color="red"):
        if mat_b.ndim != 3:
            raise ValueError(
                "prequantize_scaled_grouped_mm_rhs expects mat_b rank-3 [G,K,N], "
                f"got {tuple(mat_b.shape)}"
            )
        mat_b_q, scale_b = quantize_grouped_weight_3d_to_mxfp8_blocked(mat_b)
        mat_b_q = _to_col_major_last2(mat_b_q)
        return ScaledGroupedMMPrequantizedRHS(
            mat_b_q=mat_b_q,
            scale_b=scale_b,
            mat_b_shape=tuple(mat_b.shape),
            mat_b_version=_tensor_version(mat_b) if check_mat_b_version else -1,
        )


@torch.compiler.disable
def _scaled_grouped_mm_v2_cuda(
    mat_a_q: Tensor,
    mat_b_q: Tensor,
    scale_a_blocked: Tensor,
    scale_b_blocked: Tensor,
    *,
    offs: Tensor,
    use_fast_accum: bool,
) -> Tensor:
    return torch.ops.aten._scaled_grouped_mm_v2.default(
        mat_a_q,
        mat_b_q,
        [scale_a_blocked],
        _MXFP8_RECIPE_BLOCKWISE_1X32,
        _MXFP8_SWIZZLE,
        [scale_b_blocked],
        _MXFP8_RECIPE_BLOCKWISE_1X32,
        _MXFP8_SWIZZLE,
        offs,
        None,
        torch.bfloat16,
        [],
        use_fast_accum,
    )


def _forward_scaled_grouped_mm_mxfp8(
    mat_a: Tensor,
    mat_b: Tensor,
    offs: Tensor,
    *,
    use_fast_accum: bool,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
    prequantized_rhs: Optional[ScaledGroupedMMPrequantizedRHS] = None,
) -> Tensor:
    if mat_a.ndim != 2 or mat_b.ndim != 3:
        raise ValueError(
            "scaled_grouped_mm_q currently supports 2D-3D grouped mm only, "
            f"got mat_a={tuple(mat_a.shape)} mat_b={tuple(mat_b.shape)}"
        )

    if offs.dtype != torch.int32:
        offs = offs.to(dtype=torch.int32)

    if not mat_a.is_cuda or not mat_b.is_cuda:
        raise RuntimeError("scaled_grouped_mm_q requires CUDA tensors")

    scale_a_blocked: Optional[Tensor] = None
    scale_b_blocked: Optional[Tensor] = None

    if prequantized_lhs is not None:
        if tuple(prequantized_lhs.mat_a_shape) != tuple(mat_a.shape):
            raise ValueError(
                "prequantized_lhs shape mismatch: "
                f"expected {tuple(mat_a.shape)}, got {tuple(prequantized_lhs.mat_a_shape)}"
            )
        if prequantized_lhs.mat_a_q.device != mat_a.device:
            raise ValueError(
                "prequantized_lhs device mismatch: "
                f"q_device={prequantized_lhs.mat_a_q.device} mat_a_device={mat_a.device}"
            )
        mat_a_q = prequantized_lhs.mat_a_q
        if prequantized_lhs.scales_are_blocked:
            scale_a_blocked = prequantized_lhs.scale_a
        else:
            scale_a_blocked = grouped_scales_to_mxfp8_blocked(
                prequantized_lhs.scale_a,
                offs,
                zero_unwritten_tail=False,
            )
    else:
        mat_a_q, scale_a_unblocked = quantize_rows_to_mxfp8(mat_a, block_size=32)
        scale_a_blocked = grouped_scales_to_mxfp8_blocked(
            scale_a_unblocked,
            offs,
            zero_unwritten_tail=False,
        )

    use_cached_rhs = (
        prequantized_rhs is not None
        and tuple(prequantized_rhs.mat_b_shape) == tuple(mat_b.shape)
        and prequantized_rhs.mat_b_q.device == mat_b.device
        and (
            prequantized_rhs.mat_b_version < 0
            or prequantized_rhs.mat_b_version == _tensor_version(mat_b)
        )
    )
    if use_cached_rhs:
        assert prequantized_rhs is not None
        mat_b_q = prequantized_rhs.mat_b_q
        scale_b_blocked = prequantized_rhs.scale_b
        mat_b_q = _to_col_major_last2(mat_b_q)
    else:
        with nvtx.annotate("scaled_grouped_mm_q_rhs_cache_miss_quantize", color="red"):
            mat_b_q, scale_b_blocked = quantize_grouped_weight_3d_to_mxfp8_blocked(mat_b)
            mat_b_q = _to_col_major_last2(mat_b_q)

    if scale_a_blocked is None:
        raise RuntimeError("aten::_scaled_grouped_mm_v2 path requires blocked lhs scales")
    if scale_b_blocked is None:
        raise RuntimeError("aten::_scaled_grouped_mm_v2 path requires blocked rhs scales")

    return _scaled_grouped_mm_v2_cuda(
        mat_a_q,
        mat_b_q,
        scale_a_blocked,
        scale_b_blocked,
        offs=offs,
        use_fast_accum=use_fast_accum,
    )


def _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
    mat_a: Tensor,
    prequantized_rhs: ScaledGroupedMMPrequantizedRHS,
    offs: Tensor,
    *,
    use_fast_accum: bool,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
) -> Tensor:
    if mat_a.ndim != 2:
        raise ValueError(
            "scaled_grouped_mm_q_fp8_weight currently supports rank-2 LHS only, "
            f"got mat_a={tuple(mat_a.shape)}"
        )

    mat_b_shape = tuple(prequantized_rhs.mat_b_shape)
    if len(mat_b_shape) != 3:
        raise ValueError(
            "prequantized_rhs must describe rank-3 [G,K,N] RHS, " f"got mat_b_shape={mat_b_shape}"
        )

    if offs.dtype != torch.int32:
        offs = offs.to(dtype=torch.int32)

    if not mat_a.is_cuda or not prequantized_rhs.mat_b_q.is_cuda:
        raise RuntimeError("scaled_grouped_mm_q_fp8_weight requires CUDA tensors")
    if prequantized_rhs.mat_b_q.device != mat_a.device:
        raise RuntimeError(
            "prequantized_rhs device mismatch: "
            f"rhs_device={prequantized_rhs.mat_b_q.device} mat_a_device={mat_a.device}"
        )

    scale_a_blocked: Optional[Tensor] = None
    if prequantized_lhs is not None:
        if tuple(prequantized_lhs.mat_a_shape) != tuple(mat_a.shape):
            raise ValueError(
                "prequantized_lhs shape mismatch: "
                f"expected {tuple(mat_a.shape)}, got {tuple(prequantized_lhs.mat_a_shape)}"
            )
        if prequantized_lhs.mat_a_q.device != mat_a.device:
            raise ValueError(
                "prequantized_lhs device mismatch: "
                f"q_device={prequantized_lhs.mat_a_q.device} mat_a_device={mat_a.device}"
            )
        mat_a_q = prequantized_lhs.mat_a_q
        if prequantized_lhs.scales_are_blocked:
            scale_a_blocked = prequantized_lhs.scale_a
        else:
            scale_a_blocked = grouped_scales_to_mxfp8_blocked(
                prequantized_lhs.scale_a,
                offs,
                zero_unwritten_tail=False,
            )
    else:
        mat_a_q, scale_a_unblocked = quantize_rows_to_mxfp8(mat_a, block_size=32)
        scale_a_blocked = grouped_scales_to_mxfp8_blocked(
            scale_a_unblocked,
            offs,
            zero_unwritten_tail=False,
        )

    mat_b_q = _to_col_major_last2(prequantized_rhs.mat_b_q)
    scale_b_blocked = prequantized_rhs.scale_b

    if scale_a_blocked is None:
        raise RuntimeError("aten::_scaled_grouped_mm_v2 path requires blocked lhs scales")

    return _scaled_grouped_mm_v2_cuda(
        mat_a_q,
        mat_b_q,
        scale_a_blocked,
        scale_b_blocked,
        offs=offs,
        use_fast_accum=use_fast_accum,
    )


class _ScaledGroupedMMQFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mat_a: Tensor,
        mat_b: Tensor,
        offs: Optional[Tensor],
        input_grad_out: Optional[Tensor],
        use_fast_accum: bool,
        prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS],
        prequantized_rhs: Optional[ScaledGroupedMMPrequantizedRHS],
        prequantized_rhs_for_dgrad: Optional[ScaledGroupedMMPrequantizedRHS],
    ) -> Tensor:
        if offs is None:
            raise ValueError("offs is required for scaled_grouped_mm_q")

        if input_grad_out is not None:
            if input_grad_out.shape != mat_a.shape:
                raise ValueError(
                    f"input_grad_out shape mismatch: expected {tuple(mat_a.shape)}, got {tuple(input_grad_out.shape)}"
                )
            if input_grad_out.dtype != mat_a.dtype:
                raise ValueError(
                    f"input_grad_out dtype mismatch: expected {mat_a.dtype}, got {input_grad_out.dtype}"
                )
        if prequantized_rhs_for_dgrad is not None:
            mat_b_t_shape = tuple(mat_b.transpose(-2, -1).shape)
            if tuple(prequantized_rhs_for_dgrad.mat_b_shape) != mat_b_t_shape:
                raise ValueError(
                    "prequantized_rhs_for_dgrad shape mismatch: "
                    f"expected {mat_b_t_shape}, got {tuple(prequantized_rhs_for_dgrad.mat_b_shape)}"
                )
            if prequantized_rhs_for_dgrad.mat_b_q.device != mat_b.device:
                raise ValueError(
                    "prequantized_rhs_for_dgrad device mismatch: "
                    f"expected {mat_b.device}, got {prequantized_rhs_for_dgrad.mat_b_q.device}"
                )

        if mat_a.numel() == 0:
            result = torch.empty(
                (mat_a.shape[0], mat_b.shape[-1]),
                device=mat_a.device,
                dtype=torch.bfloat16,
            )
            ctx.save_for_backward(mat_a, mat_b, offs)
            ctx._saved_mat_a_from_prequantized_lhs = False
            ctx._mat_a_empty = True
            ctx.input_grad_out = input_grad_out
            ctx.prequantized_rhs_for_dgrad = prequantized_rhs_for_dgrad
            return result

        result = _forward_scaled_grouped_mm_mxfp8(
            mat_a,
            mat_b,
            offs,
            use_fast_accum=use_fast_accum,
            prequantized_lhs=prequantized_lhs,
            prequantized_rhs=prequantized_rhs,
        )

        # In rowwise FP8 paths we can skip saving bf16 mat_a and reconstruct it
        # from saved prequantized (q, scale) during backward to avoid dispatch DQ.
        use_saved_prequantized_lhs = (
            prequantized_lhs is not None and not prequantized_lhs.scales_are_blocked
        )
        if use_saved_prequantized_lhs:
            assert prequantized_lhs is not None
            mat_a_q, scale_a = _prequantized_lhs_tensors_for_backward(prequantized_lhs)
            ctx.save_for_backward(
                mat_b,
                offs,
                mat_a_q,
                scale_a,
            )
            ctx._saved_mat_a_from_prequantized_lhs = True
        else:
            ctx.save_for_backward(mat_a, mat_b, offs)
            ctx._saved_mat_a_from_prequantized_lhs = False
        ctx.input_grad_out = input_grad_out
        ctx.prequantized_rhs_for_dgrad = prequantized_rhs_for_dgrad
        ctx._mat_a_empty = False
        return result

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        mat_a: Optional[Tensor] = None
        saved_mat_a_q: Optional[Tensor] = None
        saved_mat_a_scale: Optional[Tensor] = None
        grad_out_compute = grad_out
        grad_out_mxfp8 = grad_out if isinstance(grad_out, OlmoMXFP8Tensor) else None
        if grad_out_mxfp8 is not None:
            grad_out_prequantized_lhs = grad_out_mxfp8.as_scaled_grouped_mm_prequantized_lhs()
        else:
            grad_out_prequantized_lhs = None
        if grad_out_compute.dtype == torch.float8_e4m3fn:
            grad_out_compute = grad_out_compute.to(dtype=torch.bfloat16)
        if bool(getattr(ctx, "_saved_mat_a_from_prequantized_lhs", False)):
            mat_b, offs, mat_a_q, mat_a_scale = ctx.saved_tensors
            saved_mat_a_q = mat_a_q
            saved_mat_a_scale = mat_a_scale
        else:
            mat_a, mat_b, offs = ctx.saved_tensors

        grad_a = None
        grad_b = None

        if bool(getattr(ctx, "_mat_a_empty", False)):
            if ctx.needs_input_grad[0]:
                assert mat_a is not None
                grad_a = torch.empty_like(mat_a)
                if ctx.input_grad_out is not None:
                    ctx.input_grad_out.copy_(grad_a)
                    grad_a = ctx.input_grad_out
            if ctx.needs_input_grad[1]:
                grad_b = torch.zeros_like(mat_b)
            return grad_a, grad_b, None, None, None, None, None, None

        if ctx.needs_input_grad[0]:
            if grad_out_compute.is_cuda and mat_b.is_cuda:
                grad_a = _forward_scaled_grouped_mm_mxfp8(
                    grad_out_compute,
                    mat_b.transpose(-2, -1),
                    offs,
                    use_fast_accum=True,
                    prequantized_lhs=grad_out_prequantized_lhs,
                    prequantized_rhs=ctx.prequantized_rhs_for_dgrad,
                )
                if ctx.input_grad_out is not None:
                    ctx.input_grad_out.copy_(grad_a)
                    grad_a = ctx.input_grad_out
            else:
                grad_a = F.grouped_mm(grad_out_compute, mat_b.transpose(-2, -1), offs=offs)
                if ctx.input_grad_out is not None:
                    ctx.input_grad_out.copy_(grad_a)
                    grad_a = ctx.input_grad_out

        if ctx.needs_input_grad[1]:
            if grad_b is None:
                if mat_a is None:
                    if saved_mat_a_q is None or saved_mat_a_scale is None:
                        raise RuntimeError(
                            "scaled_grouped_mm_q backward expected mat_a when grad_b is required"
                        )
                    mat_a = dequantize_rows_from_mxfp8(
                        saved_mat_a_q,
                        saved_mat_a_scale,
                        block_size=32,
                        out_dtype=grad_out_compute.dtype,
                    )
                if grad_out_mxfp8 is not None:
                    grad_out_compute = grad_out_mxfp8.dequantize(out_dtype=mat_a.dtype)
                grad_b = F.grouped_mm(
                    grad_out_compute.transpose(-2, -1),
                    mat_a,
                    offs=offs,
                ).transpose(-2, -1)

        return grad_a, grad_b, None, None, None, None, None, None


class _ScaledGroupedMMQFP8WeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mat_a: Tensor,
        grad_anchor: Tensor,
        offs: Optional[Tensor],
        input_grad_out: Optional[Tensor],
        use_fast_accum: bool,
        prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS],
        prequantized_rhs: ScaledGroupedMMPrequantizedRHS,
        prequantized_rhs_for_dgrad: ScaledGroupedMMPrequantizedRHS,
        wgrad_sink: Optional[Any],
        wgrad_sink_transpose_last2: bool,
        wgrad_sink_squeeze_first_dim: bool,
    ) -> Tensor:
        if offs is None:
            raise ValueError("offs is required for scaled_grouped_mm_q_fp8_weight")
        if grad_anchor.ndim != 3:
            raise ValueError(
                "scaled_grouped_mm_q_fp8_weight expects a rank-3 gradient anchor, "
                f"got {tuple(grad_anchor.shape)}"
            )
        if tuple(prequantized_rhs.mat_b_shape) != tuple(grad_anchor.shape):
            raise ValueError(
                "prequantized_rhs shape mismatch: "
                f"expected {tuple(grad_anchor.shape)}, got {tuple(prequantized_rhs.mat_b_shape)}"
            )
        grad_anchor_t_shape = tuple(grad_anchor.transpose(-2, -1).shape)
        if tuple(prequantized_rhs_for_dgrad.mat_b_shape) != grad_anchor_t_shape:
            raise ValueError(
                "prequantized_rhs_for_dgrad shape mismatch: "
                f"expected {grad_anchor_t_shape}, got {tuple(prequantized_rhs_for_dgrad.mat_b_shape)}"
            )

        if input_grad_out is not None:
            if input_grad_out.shape != mat_a.shape:
                raise ValueError(
                    f"input_grad_out shape mismatch: expected {tuple(mat_a.shape)}, got {tuple(input_grad_out.shape)}"
                )
            if input_grad_out.dtype != mat_a.dtype:
                raise ValueError(
                    f"input_grad_out dtype mismatch: expected {mat_a.dtype}, got {input_grad_out.dtype}"
                )

        ctx.grad_anchor_shape = tuple(grad_anchor.shape)
        ctx.grad_anchor_dtype = grad_anchor.dtype
        ctx.grad_anchor_device = grad_anchor.device
        ctx.input_grad_out = input_grad_out
        ctx.prequantized_rhs_for_dgrad = prequantized_rhs_for_dgrad
        ctx.wgrad_sink = wgrad_sink
        ctx.wgrad_sink_transpose_last2 = bool(wgrad_sink_transpose_last2)
        ctx.wgrad_sink_squeeze_first_dim = bool(wgrad_sink_squeeze_first_dim)

        if mat_a.numel() == 0:
            result = torch.empty(
                (mat_a.shape[0], grad_anchor.shape[-1]),
                device=mat_a.device,
                dtype=torch.bfloat16,
            )
            ctx.save_for_backward(mat_a, offs)
            ctx._saved_mat_a_from_prequantized_lhs = False
            ctx._mat_a_empty = True
            return result

        result = _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
            mat_a,
            prequantized_rhs,
            offs,
            use_fast_accum=use_fast_accum,
            prequantized_lhs=prequantized_lhs,
        )

        use_saved_prequantized_lhs = (
            prequantized_lhs is not None and not prequantized_lhs.scales_are_blocked
        )
        if use_saved_prequantized_lhs:
            assert prequantized_lhs is not None
            mat_a_q, scale_a = _prequantized_lhs_tensors_for_backward(prequantized_lhs)
            ctx.save_for_backward(
                offs,
                mat_a_q,
                scale_a,
            )
            ctx._saved_mat_a_from_prequantized_lhs = True
        else:
            ctx.save_for_backward(mat_a, offs)
            ctx._saved_mat_a_from_prequantized_lhs = False
        ctx._mat_a_empty = False
        return result

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        mat_a: Optional[Tensor] = None
        saved_mat_a_q: Optional[Tensor] = None
        saved_mat_a_scale: Optional[Tensor] = None
        grad_out_compute = grad_out
        grad_out_mxfp8 = grad_out if isinstance(grad_out, OlmoMXFP8Tensor) else None
        if grad_out_mxfp8 is not None:
            grad_out_prequantized_lhs = grad_out_mxfp8.as_scaled_grouped_mm_prequantized_lhs()
        else:
            grad_out_prequantized_lhs = None
        if grad_out_compute.dtype == torch.float8_e4m3fn:
            grad_out_compute = grad_out_compute.to(dtype=torch.bfloat16)

        if bool(getattr(ctx, "_saved_mat_a_from_prequantized_lhs", False)):
            offs, mat_a_q, mat_a_scale = ctx.saved_tensors
            saved_mat_a_q = mat_a_q
            saved_mat_a_scale = mat_a_scale
        else:
            mat_a, offs = ctx.saved_tensors

        grad_a = None
        grad_anchor = None

        if bool(getattr(ctx, "_mat_a_empty", False)):
            if ctx.needs_input_grad[0]:
                assert mat_a is not None
                grad_a = torch.empty_like(mat_a)
                if ctx.input_grad_out is not None:
                    ctx.input_grad_out.copy_(grad_a)
                    grad_a = ctx.input_grad_out
            if ctx.needs_input_grad[1] or ctx.wgrad_sink is not None:
                grad_anchor = torch.zeros(
                    ctx.grad_anchor_shape,
                    device=ctx.grad_anchor_device,
                    dtype=ctx.grad_anchor_dtype,
                )
                if ctx.wgrad_sink is not None:
                    sink_grad = (
                        grad_anchor.transpose(-2, -1).contiguous()
                        if ctx.wgrad_sink_transpose_last2
                        else grad_anchor
                    )
                    if ctx.wgrad_sink_squeeze_first_dim:
                        if sink_grad.shape[0] != 1:
                            raise RuntimeError(
                                "wgrad_sink_squeeze_first_dim expects first dim size 1, "
                                f"got {tuple(sink_grad.shape)}"
                            )
                        sink_grad = sink_grad.squeeze(0).contiguous()
                    ctx.wgrad_sink.accumulate_wgrad(sink_grad)
            if not ctx.needs_input_grad[1]:
                grad_anchor = None
            return grad_a, grad_anchor, None, None, None, None, None, None, None, None, None

        if ctx.needs_input_grad[0]:
            if grad_out_compute.is_cuda and ctx.prequantized_rhs_for_dgrad.mat_b_q.is_cuda:
                grad_a = _forward_scaled_grouped_mm_mxfp8_prequantized_rhs(
                    grad_out_compute,
                    ctx.prequantized_rhs_for_dgrad,
                    offs,
                    use_fast_accum=True,
                    prequantized_lhs=grad_out_prequantized_lhs,
                )
                if ctx.input_grad_out is not None:
                    ctx.input_grad_out.copy_(grad_a)
                    grad_a = ctx.input_grad_out
            else:
                raise RuntimeError("scaled_grouped_mm_q_fp8_weight dgrad requires CUDA cached RHS")

        need_wgrad = ctx.needs_input_grad[1] or ctx.wgrad_sink is not None
        if need_wgrad:
            if mat_a is None:
                if saved_mat_a_q is None or saved_mat_a_scale is None:
                    raise RuntimeError(
                        "scaled_grouped_mm_q_fp8_weight backward expected mat_a when wgrad is required"
                    )
                mat_a = dequantize_rows_from_mxfp8(
                    saved_mat_a_q,
                    saved_mat_a_scale,
                    block_size=32,
                    out_dtype=grad_out_compute.dtype,
                )
            if grad_out_mxfp8 is not None:
                grad_out_compute = grad_out_mxfp8.dequantize(out_dtype=mat_a.dtype)
            grad_anchor = F.grouped_mm(
                grad_out_compute.transpose(-2, -1),
                mat_a,
                offs=offs,
            ).transpose(-2, -1)
            if ctx.wgrad_sink is not None:
                sink_grad = (
                    grad_anchor.transpose(-2, -1).contiguous()
                    if ctx.wgrad_sink_transpose_last2
                    else grad_anchor
                )
                if ctx.wgrad_sink_squeeze_first_dim:
                    if sink_grad.shape[0] != 1:
                        raise RuntimeError(
                            "wgrad_sink_squeeze_first_dim expects first dim size 1, "
                            f"got {tuple(sink_grad.shape)}"
                        )
                    sink_grad = sink_grad.squeeze(0).contiguous()
                ctx.wgrad_sink.accumulate_wgrad(sink_grad)
            if not ctx.needs_input_grad[1]:
                grad_anchor = None

        return grad_a, grad_anchor, None, None, None, None, None, None, None, None, None


@beta_feature
def scaled_grouped_mm_q(
    mat_a: Tensor,
    mat_b: Tensor,
    *,
    offs: Tensor,
    input_grad_out: Optional[Tensor] = None,
    use_fast_accum: bool = True,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
    prequantized_rhs: Optional[ScaledGroupedMMPrequantizedRHS] = None,
    prequantized_rhs_for_dgrad: Optional[ScaledGroupedMMPrequantizedRHS] = None,
) -> Tensor:
    """
    MXFP8 grouped mm wrapper used by MoE routed experts.

    Forward uses aten::_scaled_grouped_mm_v2 and returns bf16 output.
    Backward uses the same bf16 grouped MXFP8 path for dgrad on CUDA and grouped_mm for wgrad.

    Supports 2D-3D grouped mm with `offs` and optional explicit buffers:
    - input_grad_out: grad(mat_a) destination in backward
    - prequantized_lhs: optional pre-quantized LHS q/scales to bypass LHS quantization
    - prequantized_rhs: optional pre-quantized RHS q/scales to bypass RHS quantization
    - prequantized_rhs_for_dgrad: optional pre-quantized transposed RHS used by backward dgrad
    """
    return _ScaledGroupedMMQFunction.apply(
        mat_a,
        mat_b,
        offs,
        input_grad_out,
        use_fast_accum,
        prequantized_lhs,
        prequantized_rhs,
        prequantized_rhs_for_dgrad,
    )


@beta_feature
def scaled_grouped_mm_q_fp8_weight(
    mat_a: Tensor,
    grad_anchor: Tensor,
    *,
    offs: Tensor,
    input_grad_out: Optional[Tensor] = None,
    use_fast_accum: bool = True,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
    prequantized_rhs: ScaledGroupedMMPrequantizedRHS,
    prequantized_rhs_for_dgrad: ScaledGroupedMMPrequantizedRHS,
    wgrad_sink: Optional[Any] = None,
    wgrad_sink_transpose_last2: bool = False,
    wgrad_sink_squeeze_first_dim: bool = False,
) -> Tensor:
    """
    MXFP8 grouped mm for a prequantized expert weight.

    This transitional API uses only the cached MXFP8 RHS tensors for forward
    and dgrad, while `grad_anchor` receives the normal high-precision wgrad.
    If `wgrad_sink` is provided, the same wgrad is also accumulated there as a
    side channel for the staged FP8-only-params refactor. That keeps today's
    DDP/optimizer behavior intact while decoupling kernel execution from the
    bf16 model weight storage.
    """
    return _ScaledGroupedMMQFP8WeightFunction.apply(
        mat_a,
        grad_anchor,
        offs,
        input_grad_out,
        use_fast_accum,
        prequantized_lhs,
        prequantized_rhs,
        prequantized_rhs_for_dgrad,
        wgrad_sink,
        bool(wgrad_sink_transpose_last2),
        bool(wgrad_sink_squeeze_first_dim),
    )


__all__ = [
    "scaled_grouped_mm_q",
    "scaled_grouped_mm_q_fp8_weight",
    "prequantize_scaled_grouped_mm_rhs",
    "ScaledGroupedMMPrequantizedLHS",
    "ScaledGroupedMMPrequantizedRHS",
]
