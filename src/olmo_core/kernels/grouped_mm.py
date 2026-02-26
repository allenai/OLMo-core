from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .cuda_extension_utils import load_cuda_extension


_CUDA_EXTENSION = None
_CUDA_EXTENSION_ATTEMPTED = False
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ATTEMPTED
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ATTEMPTED and _CUDA_EXTENSION is None:
        raise RuntimeError("CUDA grouped_mm extension is unavailable") from _CUDA_EXTENSION_ERROR

    _CUDA_EXTENSION_ATTEMPTED = True
    try:
        this_dir = Path(__file__).resolve().parent
        cpp_src = this_dir / "cuda" / "grouped_mm_out.cpp"
        _CUDA_EXTENSION = load_cuda_extension(
            base_name="olmo_grouped_mm_out_ext",
            sources=[cpp_src],
            extra_cflags=["-O3"],
            verbose_env_names=("OLMO_GROUPED_MM_VERBOSE", "OLMO_MOE_CUDA_EXT_VERBOSE"),
            force_rebuild_env_names=(
                "OLMO_GROUPED_MM_FORCE_REBUILD",
                "OLMO_MOE_CUDA_EXT_FORCE_REBUILD",
            ),
            stale_lock_timeout_env_names=("OLMO_MOE_EXT_STALE_LOCK_TIMEOUT_SEC",),
            with_arch_suffix=True,
        )
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(f"Failed to build/load CUDA grouped_mm extension: {e}") from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def _grouped_mm_out_cuda(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    out: torch.Tensor,
    offs: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.grouped_mm_out_cuda(mat_a, mat_b, out, offs, bias)


def _expected_grouped_mm_output_shape(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: Optional[torch.Tensor],
) -> tuple[int, ...]:
    a_is_2d = mat_a.ndim == 2
    b_is_2d = mat_b.ndim == 2

    if a_is_2d:
        if b_is_2d:
            if offs is None:
                raise ValueError("offs is required for 2D/2D grouped_mm")
            return (int(offs.shape[0]), int(mat_a.shape[0]), int(mat_b.shape[1]))
        if offs is None:
            raise ValueError("offs is required for 2D/3D grouped_mm")
        return (int(mat_a.shape[0]), int(mat_b.shape[-1]))

    if b_is_2d:
        if offs is None:
            raise ValueError("offs is required for 3D/2D grouped_mm")
        return (int(mat_a.shape[1]), int(mat_b.shape[1]))
    return (int(mat_a.shape[0]), int(mat_a.shape[1]), int(mat_b.shape[-1]))


def _check_out_buffer(
    *,
    out: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> None:
    if not out.is_cuda:
        raise ValueError("out must be CUDA")
    if out.device != mat_a.device:
        raise ValueError(f"out/mat_a device mismatch: {out.device} vs {mat_a.device}")
    if mat_b.device != mat_a.device:
        raise ValueError(f"mat_b/mat_a device mismatch: {mat_b.device} vs {mat_a.device}")

    expected_dtype = out_dtype if out_dtype is not None else mat_a.dtype
    if out.dtype != expected_dtype:
        raise ValueError(f"out dtype mismatch: expected {expected_dtype}, got {out.dtype}")

    expected_shape = _expected_grouped_mm_output_shape(mat_a, mat_b, offs)
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out shape mismatch: expected {expected_shape}, got {tuple(out.shape)}")


def _check_input_grad_out_buffer(
    *,
    input_grad_out: torch.Tensor,
    mat_a: torch.Tensor,
) -> None:
    if not input_grad_out.is_cuda:
        raise ValueError("input_grad_out must be CUDA")
    if input_grad_out.device != mat_a.device:
        raise ValueError(
            f"input_grad_out/mat_a device mismatch: {input_grad_out.device} vs {mat_a.device}"
        )
    if input_grad_out.dtype != mat_a.dtype:
        raise ValueError(
            f"input_grad_out/mat_a dtype mismatch: {input_grad_out.dtype} vs {mat_a.dtype}"
        )
    if tuple(input_grad_out.shape) != tuple(mat_a.shape):
        raise ValueError(
            "input_grad_out shape mismatch: "
            f"expected {tuple(mat_a.shape)}, got {tuple(input_grad_out.shape)}"
        )


def _is_col_major_last2(x: torch.Tensor) -> bool:
    if x.ndim < 2:
        return False
    strides = x.stride()
    sizes = x.shape
    return strides[-2] == 1 and strides[-1] == sizes[-2]


def _grouped_mm_mat1_backward(
    grad: torch.Tensor,
    mat2: torch.Tensor,
    offs: Optional[torch.Tensor],
    *,
    mat1_was_col_major: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    if mat1_was_col_major:
        if out is None:
            return F.grouped_mm(mat2, grad.transpose(-2, -1), offs=offs).transpose(-2, -1)
        out_t = out.transpose(-2, -1)
        _grouped_mm_out_cuda(mat2, grad.transpose(-2, -1), out=out_t, offs=offs, bias=None)
        return out

    if out is None:
        return F.grouped_mm(grad, mat2.transpose(-2, -1), offs=offs)
    return _grouped_mm_out_cuda(grad, mat2.transpose(-2, -1), out=out, offs=offs, bias=None)


def _grouped_mm_mat2_backward(
    grad: torch.Tensor,
    mat1: torch.Tensor,
    offs: Optional[torch.Tensor],
    *,
    mat2_was_col_major: bool,
) -> torch.Tensor:
    if mat2_was_col_major:
        return F.grouped_mm(grad.transpose(-2, -1), mat1, offs=offs).transpose(-2, -1)
    return F.grouped_mm(mat1.transpose(-2, -1), grad, offs=offs)


class _GroupedMMWithBuffersFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        offs: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        out_dtype: Optional[torch.dtype],
        out: Optional[torch.Tensor],
        input_grad_out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if out is not None:
            _check_out_buffer(out=out, mat_a=mat_a, mat_b=mat_b, offs=offs, out_dtype=out_dtype)
            result = _grouped_mm_out_cuda(mat_a, mat_b, out=out, offs=offs, bias=bias)
            if result is out:
                ctx.mark_dirty(result)
        else:
            result = F.grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)

        if input_grad_out is not None:
            _check_input_grad_out_buffer(input_grad_out=input_grad_out, mat_a=mat_a)

        ctx.mat_a = mat_a if ctx.needs_input_grad[1] else None
        ctx.mat_b = mat_b if ctx.needs_input_grad[0] else None
        ctx.offs = offs
        ctx.mat_a_was_col_major = _is_col_major_last2(mat_a)
        ctx.mat_b_was_col_major = _is_col_major_last2(mat_b)
        ctx.input_grad_out = input_grad_out
        return result

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        grad_mat_a = None
        grad_mat_b = None
        offs = ctx.offs

        if ctx.needs_input_grad[1]:
            if ctx.mat_a is None:
                raise RuntimeError("Missing saved mat_a for grouped_mm backward")
            grad_mat_b = _grouped_mm_mat2_backward(
                grad_out,
                ctx.mat_a,
                offs,
                mat2_was_col_major=ctx.mat_b_was_col_major,
            )

        # Compute grad(mat_a) after grad(mat_b), because input_grad_out can alias mat_a storage.
        if ctx.needs_input_grad[0]:
            if ctx.mat_b is None:
                raise RuntimeError("Missing saved mat_b for grouped_mm backward")
            grad_mat_a = _grouped_mm_mat1_backward(
                grad_out,
                ctx.mat_b,
                offs,
                mat1_was_col_major=ctx.mat_a_was_col_major,
                out=ctx.input_grad_out,
            )
            if ctx.input_grad_out is not None and grad_mat_a is not ctx.input_grad_out:
                raise RuntimeError(
                    "Expected grad(mat_a) to be written into the provided input_grad_out buffer"
                )

        return grad_mat_a, grad_mat_b, None, None, None, None, None


def grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    input_grad_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    grouped_mm(mat_a, mat_b, *, offs=None, bias=None, out_dtype=None, out=None, input_grad_out=None)

    Equivalent to torch.nn.functional.grouped_mm with two extra buffers:
    - `out`: if provided, forward writes the result in-place into this tensor and returns it.
    - `input_grad_out`: if provided, backward writes grad(mat_a) into this buffer and returns it.
    """
    if out is not None or input_grad_out is not None:
        if not mat_a.is_cuda or not mat_b.is_cuda:
            raise ValueError("grouped_mm with explicit buffers requires CUDA tensors")
    return _GroupedMMWithBuffersFunction.apply(
        mat_a,
        mat_b,
        offs,
        bias,
        out_dtype,
        out,
        input_grad_out,
    )


__all__ = ["grouped_mm"]
