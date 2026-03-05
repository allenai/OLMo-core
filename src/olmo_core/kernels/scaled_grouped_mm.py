from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
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
    mat_a_shape: Tuple[int, int]
    scales_are_blocked: bool = False


@dataclass(frozen=True)
class ScaledGroupedMMPrequantizedRHS:
    mat_b_q: Tensor
    scale_b: Tensor
    mat_b_shape: Tuple[int, int, int]
    mat_b_version: int = -1


def _tensor_version(x: Tensor) -> int:
    try:
        return int(x._version)
    except Exception:
        return -1


def _to_col_major_last2(x: Tensor) -> Tensor:
    # Keep shape while forcing column-major storage for the trailing 2 dims.
    # _scaled_grouped_mm_v2 requires mat_b to be transposed (K,N with strides [1, K]).
    if x.stride(-2) == 1 and x.stride(-1) == x.shape[-2]:
        return x
    return x.transpose(-2, -1).contiguous().transpose(-2, -1)


@torch.no_grad()
def prequantize_scaled_grouped_mm_rhs(mat_b: Tensor) -> ScaledGroupedMMPrequantizedRHS:
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
        mat_b_version=_tensor_version(mat_b),
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
            scale_a_blocked = grouped_scales_to_mxfp8_blocked(prequantized_lhs.scale_a, offs)
    else:
        mat_a_q, scale_a_unblocked = quantize_rows_to_mxfp8(mat_a, block_size=32)
        scale_a_blocked = grouped_scales_to_mxfp8_blocked(scale_a_unblocked, offs)

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
    ) -> Tensor:
        if offs is None:
            raise ValueError("offs is required for scaled_grouped_mm_q")
        if mat_a.numel() == 0:
            return torch.empty(
                (mat_a.shape[0], mat_b.shape[-1]),
                device=mat_a.device,
                dtype=torch.bfloat16,
            )

        result = _forward_scaled_grouped_mm_mxfp8(
            mat_a,
            mat_b,
            offs,
            use_fast_accum=use_fast_accum,
            prequantized_lhs=prequantized_lhs,
            prequantized_rhs=prequantized_rhs,
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

        # In rowwise FP8 paths we can skip saving bf16 mat_a and reconstruct it
        # from saved prequantized (q, scale) during backward to avoid dispatch DQ.
        use_saved_prequantized_lhs = (
            prequantized_lhs is not None
            and not prequantized_lhs.scales_are_blocked
        )
        if use_saved_prequantized_lhs:
            ctx.save_for_backward(mat_b, offs, prequantized_lhs.mat_a_q, prequantized_lhs.scale_a)
            ctx._saved_mat_a_from_prequantized_lhs = True
        else:
            ctx.save_for_backward(mat_a, mat_b, offs)
            ctx._saved_mat_a_from_prequantized_lhs = False
        ctx.input_grad_out = input_grad_out
        return result

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        mat_a: Optional[Tensor] = None
        grad_out_compute = grad_out
        if grad_out_compute.dtype == torch.float8_e4m3fn:
            grad_out_compute = grad_out_compute.to(dtype=torch.bfloat16)
        if bool(getattr(ctx, "_saved_mat_a_from_prequantized_lhs", False)):
            mat_b, offs, mat_a_q, mat_a_scale = ctx.saved_tensors
            if ctx.needs_input_grad[1]:
                mat_a = dequantize_rows_from_mxfp8(
                    mat_a_q,
                    mat_a_scale,
                    block_size=32,
                    out_dtype=grad_out_compute.dtype,
                )
        else:
            mat_a, mat_b, offs = ctx.saved_tensors

        grad_a = None
        grad_b = None

        if ctx.needs_input_grad[0]:
            if grad_out_compute.is_cuda and mat_b.is_cuda:
                grad_a = _forward_scaled_grouped_mm_mxfp8(
                    grad_out_compute,
                    mat_b.transpose(-2, -1),
                    offs,
                    use_fast_accum=True,
                    prequantized_lhs=None,
                    prequantized_rhs=None,
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
            if mat_a is None:
                raise RuntimeError("scaled_grouped_mm_q backward expected mat_a when grad_b is required")
            grad_b = F.grouped_mm(
                grad_out_compute.transpose(-2, -1),
                mat_a,
                offs=offs,
            ).transpose(-2, -1)

        return grad_a, grad_b, None, None, None, None, None


def scaled_grouped_mm_q(
    mat_a: Tensor,
    mat_b: Tensor,
    *,
    offs: Tensor,
    input_grad_out: Optional[Tensor] = None,
    use_fast_accum: bool = True,
    prequantized_lhs: Optional[ScaledGroupedMMPrequantizedLHS] = None,
    prequantized_rhs: Optional[ScaledGroupedMMPrequantizedRHS] = None,
) -> Tensor:
    """
    MXFP8 grouped mm wrapper used by MoE routed experts.

    Forward uses aten::_scaled_grouped_mm_v2 and returns bf16 output.
    Backward uses the same bf16 grouped MXFP8 path for dgrad on CUDA and grouped_mm for wgrad.

    Supports 2D-3D grouped mm with `offs` and optional explicit buffers:
    - input_grad_out: grad(mat_a) destination in backward
    - prequantized_lhs: optional pre-quantized LHS q/scales to bypass LHS quantization
    - prequantized_rhs: optional pre-quantized RHS q/scales to bypass RHS quantization
    """
    return _ScaledGroupedMMQFunction.apply(
        mat_a,
        mat_b,
        offs,
        input_grad_out,
        use_fast_accum,
        prequantized_lhs,
        prequantized_rhs,
    )


__all__ = [
    "scaled_grouped_mm_q",
    "prequantize_scaled_grouped_mm_rhs",
    "ScaledGroupedMMPrequantizedLHS",
    "ScaledGroupedMMPrequantizedRHS",
]
