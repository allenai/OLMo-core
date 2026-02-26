from __future__ import annotations

import importlib
import os
from typing import Optional

import torch

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
    nblocks: int = 0,
) -> None:
    ext = _load_cuda_extension()
    ext.rowwise_dispatch_put(
        input,
        out,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks,
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
