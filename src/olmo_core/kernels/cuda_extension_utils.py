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
from typing import Any, Optional, Sequence, Union

import torch

log = logging.getLogger(__name__)

__all__ = ["load_cuda_extension", "LazyCudaExtension"]

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


def _env_int(names: Sequence[str], default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _cuda_arch_tag() -> str:
    """
    Return a short tag for the current CUDA compute capability (e.g. ``"sm90"``), or ``"cpu"``
    if CUDA isn't available. Used to keep build directories distinct across GPU architectures.
    """
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"sm{major}{minor}"


def _torch_extension_abi_tag() -> str:
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


def _maybe_remove_stale_build_lock(build_directory: PathLikeStr, *, timeout_seconds: float) -> None:
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

    from olmo_core.distributed.utils import get_fs_local_rank

    # Wipe once per filesystem: get_fs_local_rank() is global rank 0 on a shared FS
    # (OLMO_SHARED_FS=1), per-node local rank 0 on node-local FS, and 0 when not
    # distributed -- so each distinct build cache is wiped exactly once (the default JIT
    # cache under TORCH_EXTENSIONS_DIR is typically node-local).
    if get_fs_local_rank() == 0:
        shutil.rmtree(build_directory, ignore_errors=True)
        os.makedirs(build_directory, exist_ok=True)

    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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
) -> Any:
    """
    JIT-build (if needed) and load a CUDA/C++ extension.

    Wraps :func:`torch.utils.cpp_extension.load` with an arch/ABI-tagged build directory,
    stale-lock cleanup, and retries on transient missing-lock errors that can occur when
    multiple ranks build concurrently. ``torch.utils.cpp_extension`` is imported lazily inside
    this function, so importing this module never triggers an extension build.

    :param base_name: Base name for the extension; an ABI tag (and arch tag, unless
        ``with_arch_suffix=False``) is appended to form the build/module name.
    :param sources: Source files to compile.
    :param verbose_env_names: Env var names that, if truthy, enable verbose build output.
    :param force_rebuild_env_names: Env var names that, if truthy, wipe the build directory first.
    :param stale_lock_timeout_env_names: Env var names overriding the stale-lock age threshold.

    :returns: The loaded extension module.
    """
    from torch.utils.cpp_extension import _get_build_directory, load

    ext_name_parts = [base_name, _torch_extension_abi_tag()]
    if with_arch_suffix:
        ext_name_parts.append(_cuda_arch_tag())
    ext_name = "_".join(ext_name_parts)
    build_directory = _get_build_directory(ext_name, verbose=False)

    stale_lock_timeout_s = max(
        _env_float(stale_lock_timeout_env_names, stale_lock_timeout_default_seconds),
        0.0,
    )
    force_rebuild = _env_bool(force_rebuild_env_names, default=False)
    _force_rebuild_build_directory(build_directory, enabled=force_rebuild)
    verbose = _env_bool(verbose_env_names, default=False)
    max_retries = max(_env_int(("OLMO_MOE_EXT_LOAD_RETRIES",), 8), 1)
    retry_sleep_s = max(_env_float(("OLMO_MOE_EXT_LOAD_RETRY_SLEEP_SEC",), 0.1), 0.0)

    for attempt in range(max_retries):
        # Ensure build directory exists before cpp_extension.load() creates/open lock file.
        os.makedirs(build_directory, exist_ok=True)
        _maybe_remove_stale_build_lock(build_directory, timeout_seconds=stale_lock_timeout_s)
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


class LazyCudaExtension:
    """
    Lazily JIT-build and cache a single CUDA/C++ extension.

    Wraps :func:`load_cuda_extension` with memoization and clear error reporting:
    the extension is built on the first :meth:`load` call and cached; if that build
    fails, the original error is cached and re-raised (chained) on every later call
    so the build isn't retried in a loop. Source file names are resolved relative to
    the ``cuda/`` directory of this package.

    :param name: Human-readable extension name used in error messages.
    :param base_name: Base name passed to :func:`load_cuda_extension`.
    :param sources: Source file names, relative to this package's ``cuda/`` directory.
    :param extra_cflags: Extra host-compiler flags.
    :param extra_cuda_cflags: Extra ``nvcc`` flags; omit for C++-only extensions.
    :param verbose_env_names: See :func:`load_cuda_extension`.
    :param force_rebuild_env_names: See :func:`load_cuda_extension`.
    :param stale_lock_timeout_env_names: See :func:`load_cuda_extension`.
    :param with_arch_suffix: See :func:`load_cuda_extension`.
    """

    def __init__(
        self,
        *,
        name: str,
        base_name: str,
        sources: Sequence[str],
        extra_cflags: Optional[Sequence[str]] = None,
        extra_cuda_cflags: Optional[Sequence[str]] = None,
        verbose_env_names: Sequence[str] = (),
        force_rebuild_env_names: Sequence[str] = (),
        stale_lock_timeout_env_names: Sequence[str] = ("OLMO_MOE_EXT_STALE_LOCK_TIMEOUT_SEC",),
        with_arch_suffix: bool = True,
    ) -> None:
        self.name = name
        self.base_name = base_name
        self.sources = tuple(sources)
        self.extra_cflags = list(extra_cflags) if extra_cflags is not None else None
        self.extra_cuda_cflags = list(extra_cuda_cflags) if extra_cuda_cflags is not None else None
        self.verbose_env_names = tuple(verbose_env_names)
        self.force_rebuild_env_names = tuple(force_rebuild_env_names)
        self.stale_lock_timeout_env_names = tuple(stale_lock_timeout_env_names)
        self.with_arch_suffix = with_arch_suffix
        self._extension: Any = None
        self._attempted = False
        self._error: Optional[Exception] = None

    def load(self) -> Any:
        """
        Return the loaded extension module, building it on first call.

        :raises RuntimeError: If the extension cannot be built or loaded.
        """
        if self._extension is not None:
            return self._extension
        if self._attempted:
            raise RuntimeError(f"CUDA {self.name} extension is unavailable") from self._error

        self._attempted = True
        cuda_dir = Path(__file__).resolve().parent / "cuda"
        try:
            self._extension = load_cuda_extension(
                base_name=self.base_name,
                sources=[cuda_dir / src for src in self.sources],
                extra_cflags=self.extra_cflags,
                extra_cuda_cflags=self.extra_cuda_cflags,
                verbose_env_names=self.verbose_env_names,
                force_rebuild_env_names=self.force_rebuild_env_names,
                stale_lock_timeout_env_names=self.stale_lock_timeout_env_names,
                with_arch_suffix=self.with_arch_suffix,
            )
        except Exception as e:
            self._error = e
            raise RuntimeError(f"Failed to build/load CUDA {self.name} extension: {e}") from e

        return self._extension
