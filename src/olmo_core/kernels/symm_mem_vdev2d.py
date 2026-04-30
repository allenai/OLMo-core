from __future__ import annotations

import importlib
import os
from typing import Optional

import nvtx
import torch

from .mxfp8_utils import quantize_rows_to_mxfp8, reduce_gathered_rows_from_mxfp8

_EXTENSION_MODULE_NAME = "olmo_core.kernels._symm_mem_vdev2d_ext_gpu"
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    auto_build = os.getenv("OLMO_SYMM_VDEV2D_AUTO_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    build_backend = os.getenv("OLMO_SYMM_VDEV2D_BUILD_BACKEND", "cmake")
    if auto_build:
        try:
            from .build_symm_mem_vdev2d_ext import build_extension

            build_extension(
                inplace=True,
                verbose=False,
                force=False,
                backend=build_backend,
            )
        except Exception as e:
            _CUDA_EXTENSION_ERROR = e
            raise RuntimeError(f"Failed to auto-build CUDA symm_mem_vdev2d extension: {e}") from e

    try:
        _CUDA_EXTENSION = importlib.import_module(_EXTENSION_MODULE_NAME)
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(
            "GPU-side symm_mem_vdev2d extension is unavailable. Build it first with:\n"
            "  python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace --backend cmake\n"
            "Or set OLMO_SYMM_VDEV2D_AUTO_BUILD=1 to build automatically at import time."
        ) from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def all_to_all_vdev_2d_nblocks(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    *,
    major_align: int = 1,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.all_to_all_vdev_2d_nblocks(
        input,
        out,
        in_splits,
        out_splits_offsets,
        group_name,
        major_align,
        nblocks,
    )


@torch.compiler.disable
def all_to_all_vdev_2d_offset_nblocks(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.all_to_all_vdev_2d_offset_nblocks(
        input,
        out,
        in_splits_offsets,
        out_splits_offsets,
        group_name,
        nblocks,
    )


@torch.compiler.disable
def rowwise_dispatch_put(
    input: torch.Tensor,
    out: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_dispatch_put(
        input,
        out,
        dst_ranks,
        dst_rows,
        probs,
        group_name,
        nblocks,
    )


@torch.compiler.disable
def rowwise_dispatch_put_scaled(
    input_hp: torch.Tensor,
    out_q: torch.Tensor,
    out_scales: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    block_size: int = 32,
    nblocks: int = 0,
) -> None:
    # Optional debug safety init for stale-capacity issues; off by default to
    # avoid full-buffer memset overhead in the fp8 hot path.
    if os.getenv("OLMO_ROWWISE_FP8_DISPATCH_INIT_OUT", "0") == "1":
        out_q.zero_()
        out_scales.fill_(1.0)

    qdata, scales = quantize_rows_to_mxfp8(input_hp, block_size=block_size)
    rowwise_dispatch_put(
        qdata,
        out_q,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks=nblocks,
    )
    rowwise_dispatch_put(
        scales,
        out_scales,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks=nblocks,
    )


@torch.compiler.disable
def rowwise_combine_get(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_combine_get(
        expert_out,
        out,
        src_ranks,
        src_rows,
        probs,
        group_name,
        nblocks,
        gathered_out,
    )


@nvtx.annotate("rowwise_combine_get_scaled")
def rowwise_combine_get_scaled(
    expert_out_q: torch.Tensor,
    expert_out_scales: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    block_size: int = 32,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
    gathered_q_out: Optional[torch.Tensor] = None,
    gathered_scales_out: Optional[torch.Tensor] = None,
) -> None:
    if src_ranks.ndim != 2 or src_rows.ndim != 2:
        raise ValueError(
            f"src_ranks/src_rows must be [N,K], got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
        )
    n, k = src_ranks.shape
    if src_rows.shape != src_ranks.shape:
        raise ValueError("src_ranks/src_rows shape mismatch")

    valid_rows = (src_rows >= 0) & (src_rows < expert_out_q.shape[0])
    valid = (src_ranks >= 0) & valid_rows
    safe_ranks = torch.where(valid, src_ranks, torch.full_like(src_ranks, -1))
    safe_rows = torch.where(valid, src_rows, torch.full_like(src_rows, -1))

    flat_ranks = safe_ranks.reshape(-1, 1).contiguous()
    flat_rows = safe_rows.reshape(-1, 1).contiguous()

    expected_q_shape = (n, k, expert_out_q.shape[1])
    if gathered_q_out is not None:
        if tuple(gathered_q_out.shape) != expected_q_shape:
            raise ValueError(
                f"gathered_q_out shape mismatch: expected {expected_q_shape}, got {tuple(gathered_q_out.shape)}"
            )
        if gathered_q_out.dtype != expert_out_q.dtype:
            raise ValueError(
                f"gathered_q_out dtype mismatch: expected {expert_out_q.dtype}, got {gathered_q_out.dtype}"
            )
        if gathered_q_out.device != out.device:
            raise ValueError(
                f"gathered_q_out device mismatch: expected {out.device}, got {gathered_q_out.device}"
            )
        if not gathered_q_out.is_contiguous():
            raise ValueError("gathered_q_out must be contiguous")
        gathered_q_3d = gathered_q_out
        gathered_q = gathered_q_3d.view(n * k, -1)
    else:
        gathered_q = torch.empty(
            (n * k, expert_out_q.shape[1]),
            device=out.device,
            dtype=expert_out_q.dtype,
        )
        gathered_q_3d = gathered_q.view(n, k, -1)

    rowwise_gather_get(
        expert_out_q,
        gathered_q,
        flat_ranks,
        flat_rows,
        group_name,
        nblocks=nblocks,
    )

    expected_scales_shape = (n, k, expert_out_scales.shape[1])
    if gathered_scales_out is not None:
        if tuple(gathered_scales_out.shape) != expected_scales_shape:
            raise ValueError(
                "gathered_scales_out shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(gathered_scales_out.shape)}"
            )
        if gathered_scales_out.dtype != expert_out_scales.dtype:
            raise ValueError(
                "gathered_scales_out dtype mismatch: "
                f"expected {expert_out_scales.dtype}, got {gathered_scales_out.dtype}"
            )
        if gathered_scales_out.device != out.device:
            raise ValueError(
                "gathered_scales_out device mismatch: "
                f"expected {out.device}, got {gathered_scales_out.device}"
            )
        if not gathered_scales_out.is_contiguous():
            raise ValueError("gathered_scales_out must be contiguous")
        gathered_scales_3d = gathered_scales_out
        gathered_scales = gathered_scales_3d.view(n * k, -1)
    else:
        gathered_scales = torch.empty(
            (n * k, expert_out_scales.shape[1]),
            device=out.device,
            dtype=expert_out_scales.dtype,
        )
        gathered_scales_3d = gathered_scales.view(n, k, -1)

    rowwise_gather_get(
        expert_out_scales,
        gathered_scales,
        flat_ranks,
        flat_rows,
        group_name,
        nblocks=nblocks,
    )

    reduce_gathered_rows_from_mxfp8(
        gathered_q_3d,
        gathered_scales_3d,
        out,
        probs=probs,
        valid_mask=valid,
        block_size=block_size,
        gathered_out=gathered_out,
    )


@torch.compiler.disable
def rowwise_combine_get_fused(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_combine_get_fused(
        expert_out,
        out,
        src_ranks,
        src_rows,
        probs,
        group_name,
        nblocks,
    )


@torch.compiler.disable
def rowwise_gather_get(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_gather_get(
        expert_out,
        out,
        src_ranks,
        src_rows,
        group_name,
        nblocks,
    )
