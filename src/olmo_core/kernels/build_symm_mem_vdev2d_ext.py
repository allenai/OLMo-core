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


def _build_extension_setuptools(*, inplace: bool, verbose: bool, force: bool) -> None:
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[2]
    cuda_dir = this_dir / "cuda"
    cpp_src = cuda_dir / "olmo_symm_mem_bindings.cpp"
    cu_src = cuda_dir / "olmo_symm_mem_kernels.cu"

    include_dir, lib_dir, host_so, device_a = _find_nvshmem_paths()

    if force:
        build_dir = repo_root / "build"
        shutil.rmtree(build_dir, ignore_errors=True)

    ext = CUDAExtension(
        name="olmo_core.kernels._symm_mem_vdev2d_ext_gpu",
        sources=[str(cpp_src), str(cu_src)],
        include_dirs=[str(include_dir)],
        library_dirs=[str(lib_dir), "/usr/local/cuda/lib64"],
        dlink=True,
        dlink_libraries=["nvshmem_device"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-rdc=true",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
        extra_link_args=[
            "-Wl,--no-as-needed",
            str(host_so),
            "-Wl,--as-needed",
            str(device_a),
            "-lcudadevrt",
            f"-Wl,-rpath,{lib_dir}",
        ],
    )

    script_args = ["build_ext"]
    if inplace:
        script_args.append("--inplace")
    if verbose:
        script_args.append("-v")

    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        setup(
            name="olmo-symm-mem-ext",
            ext_modules=[ext],
            cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
            script_args=script_args,
        )
    finally:
        os.chdir(cwd)


def _build_extension_cmake(*, inplace: bool, verbose: bool, force: bool) -> None:
    import torch

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[2]
    cuda_dir = this_dir / "cuda"
    build_dir = repo_root / "build" / "olmo_symm_mem_cmake"
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
        "_symm_mem_vdev2d_ext_gpu",
    ]
    if verbose:
        build_args.append("--verbose")
    jobs = os.cpu_count()
    if jobs and jobs > 0:
        build_args.extend(["--parallel", str(jobs)])

    subprocess.run(cmake_args, check=True, env=cmake_env)
    subprocess.run(build_args, check=True, env=cmake_env)

    so_name = f"_symm_mem_vdev2d_ext_gpu{ext_suffix}"
    target_path = output_dir / so_name
    if not target_path.is_file():
        candidates = list(build_dir.glob(f"**/{so_name}"))
        if not candidates:
            raise RuntimeError(f"CMake build succeeded but {so_name} was not found.")
        shutil.copy2(max(candidates, key=lambda p: p.stat().st_mtime), target_path)


def build_extension(
    *, inplace: bool, verbose: bool, force: bool, backend: str = "cmake"
) -> None:
    backend_norm = backend.strip().lower()
    if backend_norm == "cmake":
        _build_extension_cmake(inplace=inplace, verbose=verbose, force=force)
        return
    if backend_norm == "setuptools":
        _build_extension_setuptools(inplace=inplace, verbose=verbose, force=force)
        return
    raise ValueError(
        f"Unsupported backend={backend!r}. Expected one of: cmake, setuptools."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the GPU-side OLMo NVSHMEM symmetric-memory extension with CUDA device-link."
    )
    parser.add_argument("--inplace", action="store_true", help="Build extension in-place.")
    parser.add_argument("--verbose", action="store_true", help="Verbose build output.")
    parser.add_argument("--force", action="store_true", help="Clean build directory before build.")
    parser.add_argument(
        "--backend",
        choices=["cmake", "setuptools"],
        default=os.getenv("OLMO_SYMM_VDEV2D_BUILD_BACKEND", "cmake"),
        help="Build backend (default: cmake).",
    )
    args = parser.parse_args()

    inplace = args.inplace or _env_bool("OLMO_SYMM_VDEV2D_BUILD_INPLACE", default=True)
    verbose = args.verbose or _env_bool("OLMO_SYMM_VDEV2D_BUILD_VERBOSE", default=False)
    force = args.force or _env_bool("OLMO_SYMM_VDEV2D_BUILD_FORCE", default=False)
    build_extension(
        inplace=inplace,
        verbose=verbose,
        force=force,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
