from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Sequence

import torch

from .cuda_extension_utils import LazyCudaExtension


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


def _nccl_build_kwargs() -> dict:
    # Resolved at load time (not import) so the module stays import-safe without NCCL.
    include_dir, lib_dir = _find_nccl_paths()
    return {
        "extra_include_paths": [include_dir],
        "extra_ldflags": [f"-L{lib_dir}", "-lnccl", f"-Wl,-rpath,{lib_dir}"],
    }


_EXTENSION = LazyCudaExtension(
    name="NCCL RMA P2P",
    base_name="nccl_rma_p2p_ext",
    sources=("nccl_rma_p2p.cpp",),
    extra_cflags=["-O3"],
    with_cuda=True,
    with_arch_suffix=False,
    verbose_env_names=("OLMO_NCCL_RMA_EXT_VERBOSE",),
    force_rebuild_env_names=("OLMO_NCCL_RMA_EXT_FORCE_REBUILD",),
    dynamic_build_kwargs=_nccl_build_kwargs,
)


def _load_cuda_extension():
    return _EXTENSION.load()


@torch.compiler.disable
def get_unique_id() -> bytes:
    """
    Create an NCCL unique ID (``ncclGetUniqueId``).

    Call on rank 0 and broadcast the result to all ranks before :func:`init`; it
    seeds the standalone NCCL communicator the RMA transport uses.
    """
    return _load_cuda_extension().get_unique_id()


@torch.compiler.disable
def runtime_version() -> int:
    """Return the NCCL runtime version the extension was built/loaded against."""
    return int(_load_cuda_extension().runtime_version())


@torch.compiler.disable
def init(unique_id: bytes, *, rank: int, world_size: int, device: int) -> int:
    """
    Initialize an RMA context and return its handle.

    Builds a dedicated NCCL communicator (``ncclCommInitRank``) from the broadcast
    ``unique_id`` — independent of any torch process group.

    :param unique_id: The id from :func:`get_unique_id`, broadcast to all ranks.
    :param rank: This process's rank within the communicator.
    :param world_size: The communicator size.
    :param device: The local CUDA device index to bind.

    :returns: An opaque context id passed to the other functions here.
    """
    return int(_load_cuda_extension().init(unique_id, rank, world_size, device))


@torch.compiler.disable
def alloc_window(
    context_id: int,
    sizes: Sequence[int],
    *,
    dtype: str = "float32",
    win_flags: int | None = None,
) -> tuple[int, torch.Tensor]:
    """
    Allocate a symmetric RMA window that peers can put into.

    :param context_id: The context from :func:`init`.
    :param sizes: The window shape.
    :param dtype: The window element dtype (as a string, e.g. ``"float32"``).
    :param win_flags: Optional backend-specific window flags.

    :returns: A tuple ``(window_id, tensor)`` of the window handle and a tensor view
        of this rank's local window buffer.
    """
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
    """
    One-sided put of ``src`` into ``peer``'s window, then signal completion.

    :param context_id: The context from :func:`init`.
    :param src: The local source tensor to write to the peer.
    :param peer: The destination rank.
    :param window_id: The target window (from :func:`alloc_window`).
    :param peer_window_offset_bytes: Byte offset into the peer's window to write at.
    """
    _load_cuda_extension().put_signal(
        context_id,
        src,
        peer,
        window_id,
        peer_window_offset_bytes,
    )


@torch.compiler.disable
def wait_signal(context_id: int, *, peer: int, op_count: int) -> None:
    """
    Block until ``op_count`` signals from ``peer`` have arrived.

    Signal counts are per peer, so each receiver waits for the running count of puts
    it expects from that sending peer.
    """
    _load_cuda_extension().wait_signal(context_id, peer, op_count)


@torch.compiler.disable
def free_window(context_id: int, window_id: int) -> None:
    """Free a window previously returned by :func:`alloc_window`."""
    _load_cuda_extension().free_window(context_id, window_id)


@torch.compiler.disable
def destroy(context_id: int) -> None:
    """Tear down an RMA context and its NCCL communicator."""
    _load_cuda_extension().destroy(context_id)


__all__ = [
    "get_unique_id",
    "runtime_version",
    "init",
    "alloc_window",
    "put_signal",
    "wait_signal",
    "free_window",
    "destroy",
]
