from __future__ import annotations

import argparse
import os
import shutil
import site
import subprocess
import sys
import sysconfig
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

    for base in site.getsitepackages() + [site.getusersitepackages()] + sys.path:
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
            "Could not locate NVSHMEM library dir (libnvshmem_device.a + libnvshmem_host.so.3). "
            "Set NVSHMEM_LIB_DIR or NVSHMEM_HOME."
        )

    host_so = lib_dir / "libnvshmem_host.so.3"
    device_a = lib_dir / "libnvshmem_device.a"
    return include_dir, lib_dir, host_so, device_a


def _build_extension_setuptools(*, inplace: bool, verbose: bool, force: bool) -> None:
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[2]
    cuda_dir = this_dir / "cuda"
    cpp_src = cuda_dir / "symm_mem_vdev2d.cpp"
    cu_src = cuda_dir / "symm_mem_vdev2d_kernel.cu"

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
            name="olmo-symm-mem-vdev2d-ext",
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
    build_dir = repo_root / "build" / "symm_mem_vdev2d_cmake"
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
    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])

    build_args = [
        "cmake",
        "--build",
        str(build_dir),
    ]
    if verbose:
        build_args.append("--verbose")
    jobs = os.cpu_count()
    if jobs and jobs > 0:
        build_args.extend(["--parallel", str(jobs)])

    subprocess.run(cmake_args, check=True)
    subprocess.run(build_args, check=True)

    so_name = f"_symm_mem_vdev2d_ext_gpu{ext_suffix}"
    target_path = output_dir / so_name
    if not target_path.is_file():
        candidates = list(build_dir.glob(f"**/{so_name}"))
        if not candidates:
            raise RuntimeError(f"CMake build succeeded but {so_name} was not found.")
        shutil.copy2(max(candidates, key=lambda p: p.stat().st_mtime), target_path)


def build_extension(*, inplace: bool, verbose: bool, force: bool, backend: str = "cmake") -> None:
    backend_norm = backend.strip().lower()
    if backend_norm == "cmake":
        _build_extension_cmake(inplace=inplace, verbose=verbose, force=force)
        return
    if backend_norm == "setuptools":
        _build_extension_setuptools(inplace=inplace, verbose=verbose, force=force)
        return
    raise ValueError(f"Unsupported backend={backend!r}. Expected one of: cmake, setuptools.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GPU-side NVSHMEM 2D all_to_all extension with CUDA device-link."
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
