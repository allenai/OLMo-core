from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.kernels import build_symm_mem_vdev2d_ext
from olmo_core.kernels.cuda_build_utils import (
    _cmake_cuda_architectures,
    _infer_cmake_cuda_architectures,
    _torch_cuda_arch_list_from_cmake_architectures,
)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("8.0;9.0+PTX 10.0a", "80;90;100a"),
        ("sm_90,compute_100a,9.0", "90;100a"),
        (" 10.3;10.3 ", "103"),
    ],
)
def test_cmake_cuda_architectures(source, expected):
    assert _cmake_cuda_architectures(source) == expected


def test_cmake_cuda_architectures_rejects_empty_architecture_list():
    with pytest.raises(RuntimeError, match="Could not parse TORCH_CUDA_ARCH_LIST"):
        _cmake_cuda_architectures("invalid")


def test_torch_cuda_architectures_round_trip():
    architectures = "80;90;100a;103"
    torch_architectures = _torch_cuda_arch_list_from_cmake_architectures(architectures)
    assert torch_architectures == "8.0 9.0 10.0a 10.3"
    assert _cmake_cuda_architectures(torch_architectures) == architectures


def test_infer_architectures_prefers_explicit_settings(monkeypatch):
    monkeypatch.setenv("CMAKE_CUDA_ARCHITECTURES", "90")
    monkeypatch.setenv("CUDAARCHS", "100a")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "10.3")
    torch_module = SimpleNamespace(cuda=Mock())

    assert _infer_cmake_cuda_architectures(torch_module) == "90"
    monkeypatch.delenv("CMAKE_CUDA_ARCHITECTURES")
    assert _infer_cmake_cuda_architectures(torch_module) == "100a"
    monkeypatch.delenv("CUDAARCHS")
    assert _infer_cmake_cuda_architectures(torch_module) == "103"
    torch_module.cuda.is_available.assert_not_called()


@pytest.mark.parametrize("feature_arch,expected", [("0", "90;100"), ("1", "90;100a")])
def test_infer_architectures_from_devices(monkeypatch, feature_arch, expected):
    for name in ("CMAKE_CUDA_ARCHITECTURES", "CUDAARCHS", "TORCH_CUDA_ARCH_LIST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLMO_SYMM_VDEV2D_BLACKWELL_FEATURE_ARCH", feature_arch)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 3,
        get_device_capability=lambda index: [(9, 0), (10, 0), (10, 0)][index],
    )
    assert _infer_cmake_cuda_architectures(SimpleNamespace(cuda=cuda)) == expected


@pytest.mark.parametrize("backend", ["cmake", "setuptools"])
def test_extension_build_selects_backend(monkeypatch, backend):
    selected = Mock()
    monkeypatch.setattr(build_symm_mem_vdev2d_ext, f"_build_extension_{backend}", selected)
    build_symm_mem_vdev2d_ext.build_extension(
        inplace=True, verbose=False, force=False, backend=f" {backend.upper()} "
    )
    selected.assert_called_once_with(inplace=True, verbose=False, force=False)


def test_extension_build_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        build_symm_mem_vdev2d_ext.build_extension(
            inplace=True, verbose=False, force=False, backend="unknown"
        )
