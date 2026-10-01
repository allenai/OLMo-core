from types import SimpleNamespace

import pytest

from olmo_core.kernels.cuda_build_utils import (
    _extension_source_directory,
    _infer_cmake_cuda_architectures,
)


@pytest.fixture
def arch_env(monkeypatch):
    for name in (
        "CMAKE_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "TORCH_CUDA_ARCH_LIST",
        "OLMO_SYMM_VDEV2D_BLACKWELL_FEATURE_ARCH",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "capability,expected", [((9, 0), "90"), ((10, 0), "100a"), ((10, 3), "103a")]
)
def test_detect_cuda_architecture(arch_env, capability, expected):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_capability=lambda _: capability,
        )
    )
    assert _infer_cmake_cuda_architectures(torch) == expected


def test_no_gpu_requires_explicit_architecture(arch_env):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="Set TORCH_CUDA_ARCH_LIST explicitly"):
        _infer_cmake_cuda_architectures(torch)
    arch_env.setenv("TORCH_CUDA_ARCH_LIST", "9.0;10.0a;10.3a")
    assert _infer_cmake_cuda_architectures(torch) == "90;100a;103a"
    arch_env.setenv("CMAKE_CUDA_ARCHITECTURES", "100a")
    assert _infer_cmake_cuda_architectures(torch) == "100a"


def test_inplace_prefers_checkout_over_installed_package(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    kernels = checkout / "src" / "olmo_core" / "kernels"
    (kernels / "cuda").mkdir(parents=True)
    (kernels / "cuda" / "olmo_symm_mem_kernels.cu").touch()
    (checkout / "pyproject.toml").touch()
    installed = tmp_path / "site-packages" / "olmo_core" / "kernels"
    module = str(installed / "build_symm_mem_vdev2d_ext.py")
    monkeypatch.chdir(checkout / "src")
    assert _extension_source_directory(module, inplace=True) == kernels
    assert _extension_source_directory(module, inplace=False) == installed
    monkeypatch.chdir(tmp_path)
    assert _extension_source_directory(module, inplace=True) == installed
