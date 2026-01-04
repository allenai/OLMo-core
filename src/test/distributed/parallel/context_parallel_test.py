import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.parallel.context_parallel import all_to_all_cp2hp, all_to_all_hp2cp
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device


def _test_cp2hp_scatter_dim2():
    """Test cp2hp with scatter_dim=2: [T/CP, B, H] -> [T, B, H/CP]"""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    # Each rank has a chunk of the sequence: [T/CP, B, H]
    T, B, H = 8, 2, 4
    t_local = T // world_size

    # Create input where each rank's data is identifiable by rank value
    input_tensor = torch.full((t_local, B, H), float(rank), device=device, dtype=torch.float32)

    output = all_to_all_cp2hp(input_tensor, group, scatter_dim=2)

    # Output shape should be [T, B, H/CP]
    assert output.shape == (
        T,
        B,
        H // world_size,
    ), f"Expected shape {(T, B, H // world_size)}, got {output.shape}"


def _test_cp2hp_scatter_dim1():
    """Test cp2hp with scatter_dim=1: [T/CP, H, D] -> [T, H/CP, D]"""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, H, D = 8, 4, 16
    t_local = T // world_size

    input_tensor = torch.full((t_local, H, D), float(rank), device=device, dtype=torch.float32)

    output = all_to_all_cp2hp(input_tensor, group, scatter_dim=1)

    # Output shape should be [T, H/CP, D]
    assert output.shape == (
        T,
        H // world_size,
        D,
    ), f"Expected shape {(T, H // world_size, D)}, got {output.shape}"


def _test_hp2cp_gather_dim2():
    """Test hp2cp with gather_dim=2: [T, B, H/CP] -> [T/CP, B, H]"""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, B, H = 8, 2, 4
    h_local = H // world_size

    input_tensor = torch.full((T, B, h_local), float(rank), device=device, dtype=torch.float32)

    output = all_to_all_hp2cp(input_tensor, group, gather_dim=2)

    # Output shape should be [T/CP, B, H]
    assert output.shape == (
        T // world_size,
        B,
        H,
    ), f"Expected shape {(T // world_size, B, H)}, got {output.shape}"


def _test_hp2cp_gather_dim1():
    """Test hp2cp with gather_dim=1: [T, H/CP, D] -> [T/CP, H, D]"""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, H, D = 8, 4, 16
    h_local = H // world_size

    input_tensor = torch.full((T, h_local, D), float(rank), device=device, dtype=torch.float32)

    output = all_to_all_hp2cp(input_tensor, group, gather_dim=1)

    # Output shape should be [T/CP, H, D]
    assert output.shape == (
        T // world_size,
        H,
        D,
    ), f"Expected shape {(T // world_size, H, D)}, got {output.shape}"


def _test_roundtrip_dim2():
    """Test that cp2hp -> hp2cp roundtrips correctly for dim=2."""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, B, H = 8, 2, 4
    t_local = T // world_size

    # Start with CP layout: each rank has [T/CP, B, H]
    original = torch.randn(t_local, B, H, device=device, dtype=torch.float32)

    # CP -> HP -> CP should give back the original
    hp = all_to_all_cp2hp(original, group, scatter_dim=2)
    recovered = all_to_all_hp2cp(hp, group, gather_dim=2)

    assert torch.allclose(original, recovered), (
        f"Rank {rank}: roundtrip failed, max diff = {(original - recovered).abs().max()}"
    )


def _test_roundtrip_dim1():
    """Test that cp2hp -> hp2cp roundtrips correctly for dim=1."""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, H, D = 8, 4, 16
    t_local = T // world_size

    # Start with CP layout: each rank has [T/CP, H, D]
    original = torch.randn(t_local, H, D, device=device, dtype=torch.float32)

    # CP -> HP -> CP should give back the original
    hp = all_to_all_cp2hp(original, group, scatter_dim=1)
    recovered = all_to_all_hp2cp(hp, group, gather_dim=1)

    assert torch.allclose(original, recovered), (
        f"Rank {rank}: roundtrip failed, max diff = {(original - recovered).abs().max()}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cp2hp_scatter_dim2(backend: str):
    run_distributed_test(
        _test_cp2hp_scatter_dim2, backend=backend, start_method="spawn", world_size=2
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cp2hp_scatter_dim1(backend: str):
    run_distributed_test(
        _test_cp2hp_scatter_dim1, backend=backend, start_method="spawn", world_size=2
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_hp2cp_gather_dim2(backend: str):
    run_distributed_test(
        _test_hp2cp_gather_dim2, backend=backend, start_method="spawn", world_size=2
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_hp2cp_gather_dim1(backend: str):
    run_distributed_test(
        _test_hp2cp_gather_dim1, backend=backend, start_method="spawn", world_size=2
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_roundtrip_dim2(backend: str):
    run_distributed_test(_test_roundtrip_dim2, backend=backend, start_method="spawn", world_size=2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_roundtrip_dim1(backend: str):
    run_distributed_test(_test_roundtrip_dim1, backend=backend, start_method="spawn", world_size=2)
