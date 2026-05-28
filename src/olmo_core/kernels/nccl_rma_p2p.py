from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from .cuda_extension_utils import load_cuda_extension

_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _find_nccl_paths() -> tuple[Path, Path]:
    include_candidates: list[Path] = []
    lib_candidates: list[Path] = []

    nccl_include = os.getenv("NCCL_INCLUDE_DIR")
    if nccl_include:
        include_candidates.append(Path(nccl_include))

    nccl_lib = os.getenv("NCCL_LIB_DIR")
    if nccl_lib:
        lib_candidates.append(Path(nccl_lib))

    nccl_home = os.getenv("NCCL_HOME")
    if nccl_home:
        home = Path(nccl_home)
        include_candidates.append(home / "include")
        lib_candidates.append(home / "lib")
        lib_candidates.append(home / "lib64")

    for base in site.getsitepackages() + [site.getusersitepackages()] + sys.path:
        if not base:
            continue
        root = Path(base) / "nvidia" / "nccl"
        include_candidates.append(root / "include")
        lib_candidates.append(root / "lib")

    include_dir = next((path for path in include_candidates if (path / "nccl.h").is_file()), None)
    if include_dir is None:
        raise RuntimeError("Could not locate NCCL include dir. Set NCCL_INCLUDE_DIR or NCCL_HOME.")

    lib_dir = next((path for path in lib_candidates if (path / "libnccl.so.2").is_file()), None)
    if lib_dir is None:
        raise RuntimeError("Could not locate NCCL lib dir. Set NCCL_LIB_DIR or NCCL_HOME.")

    return include_dir, lib_dir


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    this_dir = Path(__file__).resolve().parent
    include_dir, lib_dir = _find_nccl_paths()
    try:
        _CUDA_EXTENSION = load_cuda_extension(
            base_name="nccl_rma_p2p_ext",
            sources=[this_dir / "cuda" / "nccl_rma_p2p.cpp"],
            extra_cflags=["-O3"],
            extra_include_paths=[include_dir],
            extra_ldflags=[
                f"-L{lib_dir}",
                "-lnccl",
                f"-Wl,-rpath,{lib_dir}",
            ],
            verbose_env_names=("OLMO_NCCL_RMA_EXT_VERBOSE",),
            force_rebuild_env_names=("OLMO_NCCL_RMA_EXT_FORCE_REBUILD",),
            with_arch_suffix=False,
            with_cuda=True,
        )
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(f"Failed to load NCCL RMA P2P extension: {e}") from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def get_unique_id() -> bytes:
    return _load_cuda_extension().get_unique_id()


@torch.compiler.disable
def runtime_version() -> int:
    return int(_load_cuda_extension().runtime_version())


@torch.compiler.disable
def init(unique_id: bytes, *, rank: int, world_size: int, device: int) -> int:
    return int(_load_cuda_extension().init(unique_id, rank, world_size, device))


@torch.compiler.disable
def alloc_window(
    context_id: int,
    sizes: Sequence[int],
    *,
    dtype: str = "float32",
    win_flags: int | None = None,
) -> tuple[int, torch.Tensor]:
    ext = _load_cuda_extension()
    if win_flags is None:
        return ext.alloc_window(context_id, list(sizes), dtype)
    return ext.alloc_window(context_id, list(sizes), dtype, win_flags)


@torch.compiler.disable
def put_signal(
    context_id: int,
    src: torch.Tensor,
    *,
    peer: int,
    window_id: int,
    peer_window_offset_bytes: int = 0,
) -> None:
    _load_cuda_extension().put_signal(
        context_id,
        src,
        peer,
        window_id,
        peer_window_offset_bytes,
    )


@torch.compiler.disable
def wait_signal(context_id: int, *, peer: int, op_count: int) -> None:
    _load_cuda_extension().wait_signal(context_id, peer, op_count)


@torch.compiler.disable
def free_window(context_id: int, window_id: int) -> None:
    _load_cuda_extension().free_window(context_id, window_id)


@torch.compiler.disable
def destroy(context_id: int) -> None:
    _load_cuda_extension().destroy(context_id)
