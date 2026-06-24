from __future__ import annotations

import os
import re
import site
import sys
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _find_nvshmem_paths() -> tuple[Path, Path, Path, Path]:
    include_candidates: list[Path] = []
    lib_candidates: list[Path] = []

    nvshmem_include = os.getenv("NVSHMEM_INCLUDE_DIR")
    if nvshmem_include:
        include_candidates.append(Path(nvshmem_include))

    nvshmem_lib = os.getenv("NVSHMEM_LIB_DIR")
    if nvshmem_lib:
        lib_candidates.append(Path(nvshmem_lib))

    nvshmem_home = os.getenv("NVSHMEM_HOME")
    if nvshmem_home:
        home = Path(nvshmem_home)
        include_candidates.append(home / "include")
        lib_candidates.append(home / "lib")

    for base in (site.getsitepackages() + [site.getusersitepackages()] + sys.path):
        if not base:
            continue
        root = Path(base) / "nvidia" / "nvshmem"
        include_candidates.append(root / "include")
        lib_candidates.append(root / "lib")

    include_dir: Path | None = None
    for candidate in include_candidates:
        if (candidate / "nvshmem.h").is_file():
            include_dir = candidate
            break
    if include_dir is None:
        raise RuntimeError(
            "Could not locate NVSHMEM include dir (nvshmem.h). "
            "Set NVSHMEM_INCLUDE_DIR or NVSHMEM_HOME."
        )

    lib_dir: Path | None = None
    for candidate in lib_candidates:
        if (candidate / "libnvshmem_device.a").is_file() and (
            candidate / "libnvshmem_host.so.3"
        ).is_file():
            lib_dir = candidate
            break
    if lib_dir is None:
        raise RuntimeError(
            "Could not locate NVSHMEM library dir "
            "(libnvshmem_device.a + libnvshmem_host.so.3). "
            "Set NVSHMEM_LIB_DIR or NVSHMEM_HOME."
        )

    host_so = lib_dir / "libnvshmem_host.so.3"
    device_a = lib_dir / "libnvshmem_device.a"
    return include_dir, lib_dir, host_so, device_a


def _normalize_cuda_arch_for_cmake(arch: str) -> str | None:
    arch = arch.strip()
    if not arch:
        return None

    arch = arch.removesuffix("+PTX")
    arch = arch.removesuffix("+ptx")
    arch = arch.removeprefix("compute_")
    arch = arch.removeprefix("sm_")
    arch = arch.replace(".", "")

    if re.fullmatch(r"\d+[a-z]?", arch):
        return arch
    return None


def _cmake_cuda_architectures(torch_cuda_arch_list: str) -> str:
    archs: list[str] = []
    for arch in re.split(r"[;,\s]+", torch_cuda_arch_list):
        normalized = _normalize_cuda_arch_for_cmake(arch)
        if normalized is not None and normalized not in archs:
            archs.append(normalized)
    if not archs:
        raise RuntimeError(
            f"Could not parse TORCH_CUDA_ARCH_LIST={torch_cuda_arch_list!r} "
            "into CMake CUDA architectures."
        )
    return ";".join(archs)


def _torch_cuda_arch_list_from_cmake_architectures(cuda_architectures: str) -> str:
    torch_archs: list[str] = []
    for arch in cuda_architectures.split(";"):
        normalized = _normalize_cuda_arch_for_cmake(arch)
        if normalized is None:
            continue
        match = re.fullmatch(r"(\d+)(\d)([a-z]?)", normalized)
        if match is None:
            continue
        major, minor, suffix = match.groups()
        torch_arch = f"{major}.{minor}{suffix}"
        if torch_arch not in torch_archs:
            torch_archs.append(torch_arch)
    return " ".join(torch_archs)


def _infer_cmake_cuda_architectures(torch_module) -> str | None:
    explicit = os.getenv("CMAKE_CUDA_ARCHITECTURES") or os.getenv("CUDAARCHS")
    if explicit:
        return explicit

    torch_cuda_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST")
    if torch_cuda_arch_list:
        return _cmake_cuda_architectures(torch_cuda_arch_list)

    if not torch_module.cuda.is_available():
        return None

    archs: list[str] = []
    for device_idx in range(torch_module.cuda.device_count()):
        major, minor = torch_module.cuda.get_device_capability(device_idx)
        arch = f"{major}{minor}"
        if major == 10 and _env_bool(
            "OLMO_SYMM_VDEV2D_BLACKWELL_FEATURE_ARCH",
            True,
        ):
            arch = f"{arch}a"
        if arch not in archs:
            archs.append(arch)
    return ";".join(archs) if archs else None
