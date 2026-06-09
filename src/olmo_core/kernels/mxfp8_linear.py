from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .mxfp8_utils import quantize_rows_to_mxfp8, to_blocked

try:
    _MXFP8_RECIPE_BLOCKWISE_1X32 = F.ScalingType.BlockWise1x32
    _MXFP8_SWIZZLE = (
        F.SwizzleType.NO_SWIZZLE
        if torch.version.hip
        else F.SwizzleType.SWIZZLE_32_4_4
    )
except AttributeError:
    _MXFP8_RECIPE_BLOCKWISE_1X32 = 3
    _MXFP8_SWIZZLE = 0 if torch.version.hip else 1


@dataclass(frozen=True)
class ScaledMMPrequantizedRHS:
    mat_b_q: Tensor
    scale_b: Tensor
    mat_b_shape: Tuple[int, int]
    mat_b_version: int = -1


def _tensor_version(x: Tensor) -> int:
    try:
        return int(x._version)
    except Exception:
        return -1


def _scales_to_mxfp8_blocked(scales: Tensor) -> Tensor:
    return to_blocked(scales).view(torch.float8_e8m0fnu)


def _check_scaled_mm_shapes(mat_a: Tensor, mat_b: Tensor) -> None:
    if mat_a.ndim != 2 or mat_b.ndim != 2:
        raise ValueError(
            "scaled_mm_mxfp8 expects rank-2 operands, "
            f"got mat_a={tuple(mat_a.shape)} mat_b={tuple(mat_b.shape)}"
        )
    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError(
            "scaled_mm_mxfp8 shape mismatch: "
            f"mat_a={tuple(mat_a.shape)} mat_b={tuple(mat_b.shape)}"
        )
    if mat_a.shape[1] % 32 != 0:
        raise ValueError(
            "MXFP8 scaled_mm reduction dim must be divisible by 32, "
            f"got mat_a={tuple(mat_a.shape)} mat_b={tuple(mat_b.shape)}"
        )
    if mat_b.shape[1] % 16 != 0:
        raise ValueError(
            "MXFP8 scaled_mm output dim must be divisible by 16, "
            f"got mat_a={tuple(mat_a.shape)} mat_b={tuple(mat_b.shape)}"
        )


def _record_debug_saved_activation(tensor: Tensor, name: str) -> None:
    if not os.getenv("OLMO_EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG"):
        return

    try:
        from olmo_core.nn.moe.v2.activation_debug import record_named_saved_activation

        record_named_saved_activation(tensor, name)
    except Exception:
        pass


@torch.no_grad()
def prequantize_scaled_mm_rhs(
    mat_b: Tensor,
    *,
    check_mat_b_version: bool = True,
) -> ScaledMMPrequantizedRHS:
    """
    Quantize a dense RHS operand [K, N] for MXFP8 scaled_mm.

    The RHS is quantized as rows of [N, K], then exposed back as [K, N] so
    mat_b_q has the column-major layout expected by scaled_mm.
    """
    if mat_b.ndim != 2:
        raise ValueError(f"Expected mat_b rank-2 [K, N], got {tuple(mat_b.shape)}")
    if mat_b.shape[0] % 32 != 0:
        raise ValueError(
            f"mat_b K dim must be divisible by 32, got shape={tuple(mat_b.shape)}"
        )
    if mat_b.shape[1] % 16 != 0:
        raise ValueError(
            f"mat_b N dim must be divisible by 16, got shape={tuple(mat_b.shape)}"
        )

    mat_b_nk = mat_b.transpose(0, 1)
    mat_b_q_nk, scale_b_nk = quantize_rows_to_mxfp8(mat_b_nk, block_size=32)
    return ScaledMMPrequantizedRHS(
        mat_b_q=mat_b_q_nk.transpose(0, 1),
        scale_b=_scales_to_mxfp8_blocked(scale_b_nk),
        mat_b_shape=tuple(mat_b.shape),
        mat_b_version=_tensor_version(mat_b) if check_mat_b_version else -1,
    )


@torch.compiler.disable
def _scaled_mm_cuda(
    mat_a_q: Tensor,
    mat_b_q: Tensor,
    scale_a_blocked: Tensor,
    scale_b_blocked: Tensor,
) -> Tensor:
    if hasattr(F, "scaled_mm"):
        return F.scaled_mm(
            mat_a_q,
            mat_b_q,
            scale_a_blocked,
            _MXFP8_RECIPE_BLOCKWISE_1X32,
            scale_b_blocked,
            _MXFP8_RECIPE_BLOCKWISE_1X32,
            swizzle_a=_MXFP8_SWIZZLE,
            swizzle_b=_MXFP8_SWIZZLE,
            output_dtype=torch.bfloat16,
        )

    return torch._scaled_mm(  # type: ignore[attr-defined]
        mat_a_q,
        mat_b_q,
        scale_a_blocked,
        scale_b_blocked,
        out_dtype=torch.bfloat16,
    )


def _forward_scaled_mm_mxfp8(
    mat_a: Tensor,
    mat_b: Tensor,
    *,
    prequantized_rhs: Optional[ScaledMMPrequantizedRHS] = None,
) -> Tensor:
    _check_scaled_mm_shapes(mat_a, mat_b)

    if mat_a.shape[0] == 0:
        return torch.empty(
            (0, mat_b.shape[1]),
            device=mat_a.device,
            dtype=torch.bfloat16,
        )

    if not mat_a.is_cuda or not mat_b.is_cuda:
        raise RuntimeError("scaled_mm_mxfp8 requires CUDA tensors")
    if mat_a.device != mat_b.device:
        raise RuntimeError(
            f"scaled_mm_mxfp8 device mismatch: mat_a={mat_a.device} mat_b={mat_b.device}"
        )

    mat_a_q, scale_a = quantize_rows_to_mxfp8(mat_a, block_size=32)
    scale_a_blocked = _scales_to_mxfp8_blocked(scale_a)

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
    else:
        rhs = prequantize_scaled_mm_rhs(mat_b)
        mat_b_q = rhs.mat_b_q
        scale_b_blocked = rhs.scale_b

    return _scaled_mm_cuda(
        mat_a_q,
        mat_b_q,
        scale_a_blocked,
        scale_b_blocked,
    )


def _scaled_mm_mxfp8_prequantized_rhs(
    mat_a: Tensor,
    prequantized_rhs: ScaledMMPrequantizedRHS,
) -> Tensor:
    mat_b_shape = prequantized_rhs.mat_b_shape
    if mat_a.ndim != 2:
        raise ValueError(f"scaled_mm_mxfp8 expects rank-2 mat_a, got {tuple(mat_a.shape)}")
    if mat_a.shape[1] != mat_b_shape[0]:
        raise ValueError(
            "scaled_mm_mxfp8 shape mismatch: "
            f"mat_a={tuple(mat_a.shape)} mat_b={mat_b_shape}"
        )
    if mat_a.shape[1] % 32 != 0:
        raise ValueError(
            "MXFP8 scaled_mm reduction dim must be divisible by 32, "
            f"got mat_a={tuple(mat_a.shape)}"
        )
    if mat_b_shape[1] % 16 != 0:
        raise ValueError(
            "MXFP8 scaled_mm output dim must be divisible by 16, "
            f"got mat_b={mat_b_shape}"
        )
    if mat_a.shape[0] == 0:
        return torch.empty(
            (0, mat_b_shape[1]),
            device=mat_a.device,
            dtype=torch.bfloat16,
        )
    if not mat_a.is_cuda:
        raise RuntimeError("scaled_mm_mxfp8 requires CUDA tensors")
    if prequantized_rhs.mat_b_q.device != mat_a.device:
        raise RuntimeError(
            "scaled_mm_mxfp8 device mismatch: "
            f"mat_a={mat_a.device} mat_b={prequantized_rhs.mat_b_q.device}"
        )

    mat_a_q, scale_a = quantize_rows_to_mxfp8(mat_a, block_size=32)
    scale_a_blocked = _scales_to_mxfp8_blocked(scale_a)

    return _scaled_mm_cuda(
        mat_a_q,
        prequantized_rhs.mat_b_q,
        scale_a_blocked,
        prequantized_rhs.scale_b,
    )


def _wgrad_weight_mxfp8(
    grad_out: Tensor,
    prequantized_mat_a_as_rhs: ScaledMMPrequantizedRHS,
) -> Tensor:
    mat_a_shape = prequantized_mat_a_as_rhs.mat_b_shape
    if grad_out.ndim != 2:
        raise ValueError(
            "MXFP8 linear wgrad expects rank-2 operands, "
            f"got grad_out={tuple(grad_out.shape)}"
        )
    if grad_out.shape[0] != mat_a_shape[0]:
        raise ValueError(
            "MXFP8 linear wgrad batch mismatch: "
            f"mat_a={mat_a_shape} grad_out={tuple(grad_out.shape)}"
        )
    if grad_out.shape[0] == 0:
        return torch.zeros(
            (grad_out.shape[1], mat_a_shape[1]),
            device=grad_out.device,
            dtype=torch.bfloat16,
        )
    assert grad_out.shape[0] % 32 == 0, (
        "MXFP8 linear wgrad reduction dim must be divisible by 32, "
        f"got mat_a={mat_a_shape} grad_out={tuple(grad_out.shape)}"
    )

    grad_out_t_q, grad_out_t_scale = quantize_rows_to_mxfp8(
        grad_out.transpose(0, 1),
        block_size=32,
    )
    return _scaled_mm_cuda(
        grad_out_t_q,
        prequantized_mat_a_as_rhs.mat_b_q,
        _scales_to_mxfp8_blocked(grad_out_t_scale),
        prequantized_mat_a_as_rhs.scale_b,
    )


def _wgrad_weight_mxfp8_from_bf16_input(
    grad_out: Tensor,
    mat_a: Tensor,
) -> Tensor:
    if mat_a.dtype != torch.bfloat16:
        raise ValueError(f"MXFP8 linear wgrad expected bf16 mat_a, got {mat_a.dtype}")
    return _wgrad_weight_mxfp8(
        grad_out,
        prequantize_scaled_mm_rhs(mat_a, check_mat_b_version=False),
    )


class _ScaledMMMXFP8FP8WeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mat_a: Tensor,
        prequantized_rhs: ScaledMMPrequantizedRHS,
        prequantized_rhs_for_dgrad: ScaledMMPrequantizedRHS,
        wgrad_sink: Optional[Any],
        save_wgrad_input_as_mxfp8: bool,
    ) -> Tensor:
        assert mat_a.dtype == torch.bfloat16, (
            "MXFP8 linear currently expects bf16 activations, "
            f"got mat_a.dtype={mat_a.dtype}"
        )
        grad_anchor_t_shape = (
            prequantized_rhs.mat_b_shape[1],
            prequantized_rhs.mat_b_shape[0],
        )
        if tuple(prequantized_rhs_for_dgrad.mat_b_shape) != grad_anchor_t_shape:
            raise ValueError(
                "prequantized_rhs_for_dgrad shape mismatch: "
                f"expected {grad_anchor_t_shape}, got {tuple(prequantized_rhs_for_dgrad.mat_b_shape)}"
            )

        prequantized_mat_a_as_rhs = None
        if wgrad_sink is not None:
            if save_wgrad_input_as_mxfp8:
                # Forward quantizes mat_a as an LHS. Wgrad uses mat_a as the RHS
                # of grad_out.T @ mat_a, which needs the transposed RHS layout.
                prequantized_mat_a_as_rhs = prequantize_scaled_mm_rhs(
                    mat_a,
                    check_mat_b_version=False,
                )
                # _record_debug_saved_activation(
                #     prequantized_mat_a_as_rhs.mat_b_q,
                #     "scaled_mm_mxfp8_fp8_weight.wgrad_input_as_rhs.mxfp8_qdata",
                # )
                # _record_debug_saved_activation(
                #     prequantized_mat_a_as_rhs.scale_b,
                #     "scaled_mm_mxfp8_fp8_weight.wgrad_input_as_rhs.mxfp8_scales",
                # )
                ctx.save_for_backward()
            else:
                ctx.save_for_backward(mat_a)
        else:
            ctx.save_for_backward()

        result = _scaled_mm_mxfp8_prequantized_rhs(
            mat_a,
            prequantized_rhs=prequantized_rhs,
        )

        ctx.mat_a_dtype = mat_a.dtype
        ctx.prequantized_mat_a_as_rhs = prequantized_mat_a_as_rhs
        ctx.prequantized_rhs_for_dgrad = prequantized_rhs_for_dgrad
        ctx.wgrad_sink = wgrad_sink
        ctx.save_wgrad_input_as_mxfp8 = bool(save_wgrad_input_as_mxfp8)
        return result

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        grad_out_compute = grad_out
        if grad_out_compute.dtype == torch.float8_e4m3fn:
            grad_out_compute = grad_out_compute.to(dtype=torch.bfloat16)
        assert ctx.mat_a_dtype == torch.bfloat16, (
            "MXFP8 linear backward expects saved activations to be bf16, "
            f"got mat_a.dtype={ctx.mat_a_dtype}"
        )
        assert grad_out_compute.dtype == torch.bfloat16, (
            "MXFP8 linear backward expects bf16 output gradients, "
            f"got grad_out.dtype={grad_out_compute.dtype}"
        )

        grad_a = None

        if ctx.needs_input_grad[0]:
            grad_a = _scaled_mm_mxfp8_prequantized_rhs(
                grad_out_compute,
                prequantized_rhs=ctx.prequantized_rhs_for_dgrad,
            )
            assert grad_a.dtype == ctx.mat_a_dtype, (
                "MXFP8 linear dgrad should preserve bf16 activation dtype, "
                f"got grad_a.dtype={grad_a.dtype} mat_a.dtype={ctx.mat_a_dtype}"
            )

        if ctx.wgrad_sink is not None:
            if ctx.save_wgrad_input_as_mxfp8:
                grad_weight = _wgrad_weight_mxfp8(
                    grad_out_compute,
                    ctx.prequantized_mat_a_as_rhs,
                )
            else:
                (mat_a,) = ctx.saved_tensors
                grad_weight = _wgrad_weight_mxfp8_from_bf16_input(
                    grad_out_compute,
                    mat_a,
                )
            if ctx.wgrad_sink is not None:
                ctx.wgrad_sink.accumulate_wgrad(grad_weight)

        return grad_a, None, None, None, None


def scaled_mm_mxfp8_fp8_weight(
    mat_a: Tensor,
    *,
    prequantized_rhs: ScaledMMPrequantizedRHS,
    prequantized_rhs_for_dgrad: ScaledMMPrequantizedRHS,
    wgrad_sink: Optional[Any] = None,
    save_wgrad_input_as_mxfp8: bool = True,
) -> Tensor:
    """
    Dense MXFP8 mm for a prequantized linear weight.

    Weight gradients are routed to `wgrad_sink`; the module weight is only an
    initialization/checkpoint/cache anchor.
    """
    return _ScaledMMMXFP8FP8WeightFunction.apply(
        mat_a,
        prequantized_rhs,
        prequantized_rhs_for_dgrad,
        wgrad_sink,
        save_wgrad_input_as_mxfp8,
    )


__all__ = [
    "ScaledMMPrequantizedRHS",
    "prequantize_scaled_mm_rhs",
    "scaled_mm_mxfp8_fp8_weight",
]
