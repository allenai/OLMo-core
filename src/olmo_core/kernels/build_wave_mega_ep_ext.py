from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from .cuda_build_utils import (
    _env_bool,
    _find_nvshmem_paths,
    _infer_cmake_cuda_architectures,
    _torch_cuda_arch_list_from_cmake_architectures,
)


def _build_extension_cmake(*, inplace: bool, verbose: bool, force: bool) -> None:
    import torch

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[2]
    cuda_dir = this_dir / "cuda"
    build_dir = repo_root / "build" / "wave_mega_ep_cmake"
    output_dir = this_dir if inplace else (repo_root / "build")
    output_dir.mkdir(parents=True, exist_ok=True)

    include_dir, lib_dir, host_so, device_a = _find_nvshmem_paths()

    if force:
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Failed to resolve Python extension suffix (EXT_SUFFIX).")

    torch_cmake_prefix = torch.utils.cmake_prefix_path
    glibcxx_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    cuda_architectures = _infer_cmake_cuda_architectures(torch)

    cmake_args = [
        "cmake",
        "-S",
        str(cuda_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}",
        f"-DNVSHMEM_INCLUDE_DIR={include_dir}",
        f"-DNVSHMEM_LIB_DIR={lib_dir}",
        f"-DNVSHMEM_HOST_SO={host_so}",
        f"-DNVSHMEM_DEVICE_A={device_a}",
        f"-DPY_EXT_SUFFIX={ext_suffix}",
        f"-DGLIBCXX_USE_CXX11_ABI={glibcxx_abi}",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
    ]
    if cuda_architectures:
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])

    cmake_env = os.environ.copy()
    if cuda_architectures and "TORCH_CUDA_ARCH_LIST" not in cmake_env:
        torch_arch_list = _torch_cuda_arch_list_from_cmake_architectures(cuda_architectures)
        if torch_arch_list:
            cmake_env["TORCH_CUDA_ARCH_LIST"] = torch_arch_list

    build_args = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "_wave_mega_ep_ext_gpu",
    ]
    if verbose:
        build_args.append("--verbose")
    jobs = os.cpu_count()
    if jobs and jobs > 0:
        build_args.extend(["--parallel", str(jobs)])

    subprocess.run(cmake_args, check=True, env=cmake_env)
    subprocess.run(build_args, check=True, env=cmake_env)

    so_name = f"_wave_mega_ep_ext_gpu{ext_suffix}"
    target_path = output_dir / so_name
    if not target_path.is_file():
        candidates = list(build_dir.glob(f"**/{so_name}"))
        if not candidates:
            raise RuntimeError(f"CMake build succeeded but {so_name} was not found.")
        shutil.copy2(max(candidates, key=lambda p: p.stat().st_mtime), target_path)


def build_extension(*, inplace: bool, verbose: bool, force: bool) -> None:
    _build_extension_cmake(inplace=inplace, verbose=verbose, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OLMo wave/MegaMoE EP CUDA extension."
    )
    parser.add_argument("--inplace", action="store_true", help="Build extension in-place.")
    parser.add_argument("--verbose", action="store_true", help="Verbose build output.")
    parser.add_argument("--force", action="store_true", help="Clean build directory before build.")
    args = parser.parse_args()

    inplace = args.inplace or _env_bool("OLMO_WAVE_MEGA_EP_BUILD_INPLACE", default=True)
    verbose = args.verbose or _env_bool("OLMO_WAVE_MEGA_EP_BUILD_VERBOSE", default=False)
    force = args.force or _env_bool("OLMO_WAVE_MEGA_EP_BUILD_FORCE", default=False)
    build_extension(inplace=inplace, verbose=verbose, force=force)


if __name__ == "__main__":
    main()
