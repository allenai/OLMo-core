"""
Helpers for just-in-time building and loading CUDA/C++ extensions via
:func:`torch.utils.cpp_extension.load`.

Adds cross-process build-lock handling (so concurrent ranks don't clobber each other's
builds) and arch/ABI-tagged build directories (so a cached build isn't reused across an
incompatible torch/CUDA/arch toolchain).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

log = logging.getLogger(__name__)

PathLikeStr = Union[str, os.PathLike[str]]


def _env_bool(names: Sequence[str], default: bool = False) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _env_float(names: Sequence[str], default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return default


def cuda_arch_tag() -> str:
    """
    Return a short tag for the current CUDA compute capability (e.g. ``"sm90"``), or ``"cpu"``
    if CUDA isn't available. Used to keep build directories distinct across GPU architectures.
    """
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"sm{major}{minor}"


def torch_extension_abi_tag() -> str:
    """
    Return a tag encoding the torch version, CUDA version, and C++11 ABI flag, so a cached
    extension build is only reused under a compatible toolchain.
    """
    torch_version = re.sub(r"[^0-9A-Za-z]+", "_", torch.__version__).strip("_")
    cxx11_abi = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False))
    cuda_version = re.sub(r"[^0-9A-Za-z]+", "_", str(torch.version.cuda or "cpu")).strip("_")
    return f"torch{torch_version}_cu{cuda_version}_cxxabi{cxx11_abi}"


def _lock_file_is_open_by_another_process(lock_path: Path) -> Optional[bool]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None

    try:
        lock_real = lock_path.resolve()
    except FileNotFoundError:
        return False

    current_pid = os.getpid()
    saw_any_pid = False
    for proc_entry in proc_root.iterdir():
        if not proc_entry.name.isdigit():
            continue
        saw_any_pid = True
        pid = int(proc_entry.name)
        if pid == current_pid:
            continue
        fd_dir = proc_entry / "fd"
        if not fd_dir.is_dir():
            continue
        try:
            for fd_entry in fd_dir.iterdir():
                try:
                    fd_target = Path(os.readlink(fd_entry))
                except OSError:
                    continue
                if fd_target == lock_real:
                    return True
        except OSError:
            continue

    if not saw_any_pid:
        return None
    return False


def maybe_remove_stale_build_lock(build_directory: PathLikeStr, *, timeout_seconds: float) -> None:
    """
    Remove a leftover ``lock`` file in a cpp-extension build directory if it is no longer held by
    any process. When ``/proc`` is available the lock's open file descriptors are checked directly;
    otherwise a leftover lock older than ``timeout_seconds`` is treated as stale. A nonexistent or
    still-held lock is left untouched.
    """
    lock_path = Path(build_directory) / "lock"
    if not lock_path.exists():
        return

    lock_is_open = _lock_file_is_open_by_another_process(lock_path)
    if lock_is_open:
        return

    # If /proc probing is unavailable, fall back to mtime age.
    if lock_is_open is None:
        if timeout_seconds <= 0:
            return
        try:
            lock_age_s = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if lock_age_s < timeout_seconds:
            return

    # If lock file is closed by all processes, it is stale and safe to remove.
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        log.warning("Failed to remove stale extension lock %s: %s", lock_path, e)


def _force_rebuild_build_directory(build_directory: str, *, enabled: bool) -> None:
    if not enabled:
        return

    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            shutil.rmtree(build_directory, ignore_errors=True)
            os.makedirs(build_directory, exist_ok=True)
        dist.barrier()
        return

    shutil.rmtree(build_directory, ignore_errors=True)
    os.makedirs(build_directory, exist_ok=True)


def _is_transient_missing_lock_error(exc: BaseException, build_directory: str) -> bool:
    if not isinstance(exc, (FileNotFoundError, OSError)):
        return False
    if getattr(exc, "errno", None) != 2:
        return False

    path = getattr(exc, "filename", None)
    if not isinstance(path, str):
        path = str(exc)

    lock_path = os.path.join(build_directory, "lock")
    return "lock" in path and (build_directory in path or lock_path in path)


def load_cuda_extension(
    *,
    base_name: str,
    sources: Sequence[PathLikeStr],
    extra_cflags: Optional[Sequence[str]] = None,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[PathLikeStr]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    with_cuda: Optional[bool] = None,
    verbose_env_names: Sequence[str] = (),
    force_rebuild_env_names: Sequence[str] = (),
    stale_lock_timeout_env_names: Sequence[str] = ("OLMO_MOE_EXT_STALE_LOCK_TIMEOUT_SEC",),
    stale_lock_timeout_default_seconds: float = 600.0,
    with_arch_suffix: bool = True,
):
    """
    JIT-build (if needed) and load a CUDA/C++ extension.

    Wraps :func:`torch.utils.cpp_extension.load` with an arch/ABI-tagged build directory,
    stale-lock cleanup, and retries on transient missing-lock errors that can occur when
    multiple ranks build concurrently. ``torch`` is imported lazily here, so importing this
    module never triggers a compile.

    :param base_name: Base name for the extension; an ABI tag (and arch tag, unless
        ``with_arch_suffix=False``) is appended to form the build/module name.
    :param sources: Source files to compile.
    :param verbose_env_names: Env var names that, if truthy, enable verbose build output.
    :param force_rebuild_env_names: Env var names that, if truthy, wipe the build directory first.
    :param stale_lock_timeout_env_names: Env var names overriding the stale-lock age threshold.

    :returns: The loaded extension module.
    """
    from torch.utils.cpp_extension import _get_build_directory, load

    ext_name_parts = [base_name, torch_extension_abi_tag()]
    if with_arch_suffix:
        ext_name_parts.append(cuda_arch_tag())
    ext_name = "_".join(ext_name_parts)
    build_directory = _get_build_directory(ext_name, verbose=False)

    stale_lock_timeout_s = max(
        _env_float(stale_lock_timeout_env_names, stale_lock_timeout_default_seconds),
        0.0,
    )
    force_rebuild = _env_bool(force_rebuild_env_names, default=False)
    _force_rebuild_build_directory(build_directory, enabled=force_rebuild)
    verbose = _env_bool(verbose_env_names, default=False)
    max_retries = max(int(_env_float(("OLMO_MOE_EXT_LOAD_RETRIES",), 8.0)), 1)
    retry_sleep_s = max(_env_float(("OLMO_MOE_EXT_LOAD_RETRY_SLEEP_SEC",), 0.1), 0.0)

    for attempt in range(max_retries):
        # Ensure build directory exists before cpp_extension.load() creates/open lock file.
        os.makedirs(build_directory, exist_ok=True)
        maybe_remove_stale_build_lock(build_directory, timeout_seconds=stale_lock_timeout_s)
        try:
            return load(
                name=ext_name,
                sources=[str(path) for path in sources],
                extra_cflags=list(extra_cflags or []),
                extra_cuda_cflags=list(extra_cuda_cflags or []),
                extra_include_paths=[str(path) for path in (extra_include_paths or [])],
                extra_ldflags=list(extra_ldflags or []),
                build_directory=build_directory,
                verbose=verbose,
                with_cuda=with_cuda,
            )
        except Exception as e:
            is_last_attempt = attempt + 1 >= max_retries
            if is_last_attempt or not _is_transient_missing_lock_error(e, build_directory):
                raise
            if retry_sleep_s > 0:
                time.sleep(retry_sleep_s * (attempt + 1))

    raise RuntimeError("Failed to load CUDA extension after retries")
