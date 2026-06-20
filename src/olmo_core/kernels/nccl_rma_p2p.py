from __future__ import annotations

import os
import re
import site
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from .cuda_extension_utils import load_cuda_extension

_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None
_MIN_NCCL_RMA_VERSION = 22900


def _nccl_header_version(include_dir: Path) -> Optional[int]:
    header = include_dir / "nccl.h"
    try:
        text = header.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    match = re.search(r"^\s*#\s*define\s+NCCL_VERSION_CODE\s+(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        return None
    return int(match.group(1))


def _nccl_lib_file(lib_dir: Path) -> Optional[Path]:
    for name in ("libnccl.so.2", "libnccl.so"):
        path = lib_dir / name
        if path.is_file():
            return path
    return None


def _candidate_nccl_paths() -> list[tuple[Path, list[Path], str]]:
    candidates: list[tuple[Path, list[Path], str]] = []

    nccl_include = os.getenv("NCCL_INCLUDE_DIR")
    nccl_lib = os.getenv("NCCL_LIB_DIR")
    if nccl_include or nccl_lib:
        include_dir = Path(nccl_include) if nccl_include else Path("/usr/include")
        lib_dirs = [Path(nccl_lib)] if nccl_lib else []
        if not lib_dirs:
            lib_dirs.extend(
                [
                    include_dir.parent / "lib",
                    include_dir.parent / "lib64",
                    Path("/usr/lib/x86_64-linux-gnu"),
                    Path("/usr/local/cuda/lib64"),
                ]
            )
        candidates.append((include_dir, lib_dirs, "NCCL_INCLUDE_DIR/NCCL_LIB_DIR"))

    nccl_home = os.getenv("NCCL_HOME")
    if nccl_home:
        home = Path(nccl_home)
        candidates.append((home / "include", [home / "lib", home / "lib64"], "NCCL_HOME"))

    for base in site.getsitepackages() + [site.getusersitepackages()] + sys.path:
        if not base:
            continue
        root = Path(base) / "nvidia" / "nccl"
        candidates.append((root / "include", [root / "lib"], str(root)))

    candidates.extend(
        [
            (
                Path("/usr/local/cuda/include"),
                [Path("/usr/local/cuda/lib64"), Path("/usr/local/cuda/lib")],
                "/usr/local/cuda",
            ),
            (
                Path("/usr/include"),
                [Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib64"), Path("/usr/lib")],
                "/usr",
            ),
        ]
    )

    return candidates


def _find_nccl_paths(required_version: int = _MIN_NCCL_RMA_VERSION) -> tuple[Path, Path, Path]:
    skipped: list[str] = []

    for include_dir, lib_dirs, source in _candidate_nccl_paths():
        header = include_dir / "nccl.h"
        if not header.is_file():
            skipped.append(f"{source}: missing {header}")
            continue

        header_version = _nccl_header_version(include_dir)
        if header_version is None:
            skipped.append(f"{source}: could not parse NCCL_VERSION_CODE from {header}")
            continue
        if header_version < required_version:
            skipped.append(
                f"{source}: NCCL header version {header_version} is older than {required_version}"
            )
            continue

        for lib_dir in lib_dirs:
            lib_file = _nccl_lib_file(lib_dir)
            if lib_file is not None:
                return include_dir, lib_dir, lib_file

        skipped.append(
            f"{source}: header {header} is usable but no libnccl.so.2/libnccl.so was found "
            f"in {[str(path) for path in lib_dirs]}"
        )

    detail = "\n  - ".join(skipped) if skipped else "no NCCL candidates were inspected"
    raise RuntimeError(
        "NCCL RMA P2P requires NCCL headers and runtime >= "
        f"{required_version}. Could not find a usable NCCL install.\n"
        "Checked candidates:\n"
        f"  - {detail}\n"
        "Set NCCL_HOME, or set NCCL_INCLUDE_DIR and NCCL_LIB_DIR, to a NCCL 2.29+ install."
    )


def _nccl_link_flag(lib_file: Path) -> str:
    if lib_file.name == "libnccl.so.2":
        return "-l:libnccl.so.2"
    return "-lnccl"


def availability_error() -> Optional[str]:
    try:
        _find_nccl_paths()
    except Exception as e:
        return str(e)
    return None


def is_available() -> bool:
    return availability_error() is None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    this_dir = Path(__file__).resolve().parent
    include_dir, lib_dir, lib_file = _find_nccl_paths()
    try:
        _CUDA_EXTENSION = load_cuda_extension(
            base_name="nccl_rma_p2p_ext",
            sources=[this_dir / "cuda" / "nccl_rma_p2p.cpp"],
            extra_cflags=["-O3"],
            extra_include_paths=[include_dir],
            extra_ldflags=[
                f"-L{lib_dir}",
                _nccl_link_flag(lib_file),
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
def signal(context_id: int, *, peer: int) -> None:
    _load_cuda_extension().signal(context_id, peer)


@torch.compiler.disable
def free_window(context_id: int, window_id: int) -> None:
    _load_cuda_extension().free_window(context_id, window_id)


@torch.compiler.disable
def destroy(context_id: int) -> None:
    _load_cuda_extension().destroy(context_id)
