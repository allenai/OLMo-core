from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from .cuda_extension_utils import load_cuda_extension


_CUDA_EXTENSION = None
_CUDA_EXTENSION_ATTEMPTED = False
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _cutlass_include_paths() -> list[Path]:
    roots = []
    env_root = os.getenv("OLMO_CUTLASS_ROOT") or os.getenv("CUTLASS_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path("/workspace/ref/pytorch/third_party/cutlass"))

    for root in roots:
        include = root / "include"
        util_include = root / "tools" / "util" / "include"
        if (include / "cutlass" / "cutlass.h").is_file() and (
            include / "cute" / "tensor.hpp"
        ).is_file():
            out = [include]
            if util_include.is_dir():
                out.append(util_include)
            return out
    raise RuntimeError(
        "Could not locate CUTLASS headers. Set OLMO_CUTLASS_ROOT or initialize "
        "/workspace/ref/pytorch/third_party/cutlass."
    )


def _arch_list() -> list[str]:
    raw = os.getenv("OLMO_GROUPED_MM_ROW_OFFSET_ARCH_LIST")
    if raw is None:
        raw = os.getenv("OLMO_MOE_CUTLASS_ARCH_LIST")
    if raw is None:
        raw = "9.0a;10.0a;10.3a"

    archs = [part.strip() for part in raw.replace(",", ";").split(";") if part.strip()]
    if not archs:
        raise RuntimeError("OLMO_GROUPED_MM_ROW_OFFSET_ARCH_LIST cannot be empty")
    supported = {"9.0a", "10.0a", "10.3a"}
    unknown = [arch for arch in archs if arch not in supported]
    if unknown:
        raise RuntimeError(
            "Unsupported grouped_mm row-offset arch list entries "
            f"{unknown}; supported entries are {sorted(supported)}"
        )
    return archs


def _arch_cuda_flags(archs: list[str]) -> list[str]:
    flags = []
    for arch in archs:
        num = arch.replace(".", "")
        flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
    return flags


def _arch_suffix(archs: list[str]) -> str:
    return "_".join(f"sm{arch.replace('.', '')}" for arch in archs)


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ATTEMPTED
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ATTEMPTED and _CUDA_EXTENSION is None:
        raise RuntimeError("CUDA grouped_mm row-offset extension is unavailable") from _CUDA_EXTENSION_ERROR

    _CUDA_EXTENSION_ATTEMPTED = True
    try:
        this_dir = Path(__file__).resolve().parent
        cu_src = this_dir / "cuda" / "grouped_mm_row_offset.cu"
        archs = _arch_list()
        base_name = f"olmo_grouped_mm_row_offset_ext_{_arch_suffix(archs)}"
        extra_cuda_cflags = [
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            *_arch_cuda_flags(archs),
        ]
        _CUDA_EXTENSION = load_cuda_extension(
            base_name=base_name,
            sources=[cu_src],
            extra_cflags=["-O3"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=_cutlass_include_paths(),
            verbose_env_names=(
                "OLMO_GROUPED_MM_ROW_OFFSET_VERBOSE",
                "OLMO_GROUPED_MM_VERBOSE",
                "OLMO_MOE_CUDA_EXT_VERBOSE",
            ),
            force_rebuild_env_names=(
                "OLMO_GROUPED_MM_ROW_OFFSET_FORCE_REBUILD",
                "OLMO_GROUPED_MM_FORCE_REBUILD",
                "OLMO_MOE_CUDA_EXT_FORCE_REBUILD",
            ),
            stale_lock_timeout_env_names=("OLMO_MOE_EXT_STALE_LOCK_TIMEOUT_SEC",),
            with_arch_suffix=False,
        )
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(f"Failed to build/load CUDA grouped_mm row-offset extension: {e}") from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def grouped_mm_out_row_offset_cuda(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    *,
    out: torch.Tensor,
    offs: torch.Tensor,
    row_start: torch.Tensor,
) -> torch.Tensor:
    ext = _load_cuda_extension()
    return ext.grouped_mm_out_row_offset_cuda(mat_a, mat_b, out, offs, row_start)


def grouped_mm_row_offset(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    batch_sizes: torch.Tensor,
    *,
    row_start: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if mat_a.ndim != 2:
        raise ValueError(f"mat_a must be rank-2 [M, K], got {tuple(mat_a.shape)}")
    if mat_b.ndim != 3:
        raise ValueError(f"mat_b must be rank-3 [G, K, N], got {tuple(mat_b.shape)}")
    if out.ndim != 2:
        raise ValueError(f"out must be rank-2 [M, N], got {tuple(out.shape)}")
    if batch_sizes.ndim != 1:
        raise ValueError(f"batch_sizes must be rank-1 [G], got {tuple(batch_sizes.shape)}")
    if batch_sizes.shape[0] != mat_b.shape[0]:
        raise ValueError(
            f"batch_sizes length must match mat_b groups: {batch_sizes.shape[0]} vs {mat_b.shape[0]}"
        )
    if row_start.numel() != 1:
        raise ValueError(f"row_start must be a scalar tensor, got {tuple(row_start.shape)}")
    if not mat_a.is_cuda or not mat_b.is_cuda or not out.is_cuda:
        raise ValueError("grouped_mm_row_offset requires CUDA mat_a, mat_b, and out")
    if batch_sizes.device != mat_a.device:
        raise ValueError(f"batch_sizes device must match mat_a: {batch_sizes.device} vs {mat_a.device}")
    if row_start.device != mat_a.device:
        raise ValueError(f"row_start device must match mat_a: {row_start.device} vs {mat_a.device}")
    if mat_a.dtype != torch.bfloat16 or mat_b.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("grouped_mm_row_offset currently supports only bf16 tensors")
    if row_start.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(f"row_start must be int32/int64, got {row_start.dtype}")

    offs = torch.cumsum(batch_sizes.to(dtype=torch.int32), dim=0, dtype=torch.int32)
    return grouped_mm_out_row_offset_cuda(
        mat_a,
        mat_b,
        out=out,
        offs=offs,
        row_start=row_start,
    )


__all__ = ["grouped_mm_row_offset", "grouped_mm_out_row_offset_cuda"]
