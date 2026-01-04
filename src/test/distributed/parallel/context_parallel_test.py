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

    # Verify output values: each rank should get full sequence [T] but only a chunk of H
    # Rank 0 gets first H/CP elements of H, Rank 1 gets next H/CP elements of H
    # Each rank's sequence chunk should be preserved: first t_local elements from rank 0, next t_local from rank 1
    h_local = H // world_size
    expected_first_chunk = torch.full(
        (t_local, B, h_local), 0.0, device=device, dtype=torch.float32
    )
    expected_second_chunk = torch.full(
        (t_local, B, h_local), 1.0, device=device, dtype=torch.float32
    )
    expected = torch.cat([expected_first_chunk, expected_second_chunk], dim=0)

    assert torch.allclose(output, expected), (
        f"Rank {rank}: Output values don't match expected. "
        f"Max diff = {(output - expected).abs().max()}, "
        f"Expected first {t_local} timesteps from rank 0, next {t_local} from rank 1"
    )


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

    # Verify output values: each rank should get full sequence [T] but only a chunk of H
    # Rank 0 gets first H/CP elements of H, Rank 1 gets next H/CP elements of H
    # Each rank's sequence chunk should be preserved: first t_local elements from rank 0, next t_local from rank 1
    h_local = H // world_size
    expected_first_chunk = torch.full(
        (t_local, h_local, D), 0.0, device=device, dtype=torch.float32
    )
    expected_second_chunk = torch.full(
        (t_local, h_local, D), 1.0, device=device, dtype=torch.float32
    )
    expected = torch.cat([expected_first_chunk, expected_second_chunk], dim=0)

    assert torch.allclose(output, expected), (
        f"Rank {rank}: Output values don't match expected. "
        f"Max diff = {(output - expected).abs().max()}, "
        f"Expected first {t_local} timesteps from rank 0, next {t_local} from rank 1"
    )


def _test_hp2cp_gather_dim2():
    """Test hp2cp with gather_dim=2: [T, B, H/CP] -> [T/CP, B, H]"""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, B, H = 8, 2, 4
    h_local = H // world_size
    t_local = T // world_size

    input_tensor = torch.full((T, B, h_local), float(rank), device=device, dtype=torch.float32)

    output = all_to_all_hp2cp(input_tensor, group, gather_dim=2)

    # Output shape should be [T/CP, B, H]
    assert output.shape == (
        T // world_size,
        B,
        H,
    ), f"Expected shape {(T // world_size, B, H)}, got {output.shape}"

    # Verify output values: each rank should get a chunk of sequence [T/CP] but full H
    # Rank 0 gets first T/CP elements of T, Rank 1 gets next T/CP elements of T
    # Each rank's H chunk should be gathered: first h_local elements from rank 0, next h_local from rank 1
    # So rank 0 output should have: first t_local timesteps, with H values [0.0, 0.0, 1.0, 1.0]
    # And rank 1 output should have: next t_local timesteps, with H values [0.0, 0.0, 1.0, 1.0]
    expected_h_chunk_0 = torch.full((t_local, B, h_local), 0.0, device=device, dtype=torch.float32)
    expected_h_chunk_1 = torch.full((t_local, B, h_local), 1.0, device=device, dtype=torch.float32)
    expected = torch.cat([expected_h_chunk_0, expected_h_chunk_1], dim=2)

    assert torch.allclose(output, expected), (
        f"Rank {rank}: Output values don't match expected. "
        f"Max diff = {(output - expected).abs().max()}, "
        f"Expected {t_local} timesteps with H values gathered from all ranks"
    )


def _test_hp2cp_gather_dim1():
    """Test hp2cp with gather_dim=1: [T, H/CP, D] -> [T/CP, H, D]"""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    T, H, D = 8, 4, 16
    h_local = H // world_size
    t_local = T // world_size

    input_tensor = torch.full((T, h_local, D), float(rank), device=device, dtype=torch.float32)

    output = all_to_all_hp2cp(input_tensor, group, gather_dim=1)

    # Output shape should be [T/CP, H, D]
    assert output.shape == (
        T // world_size,
        H,
        D,
    ), f"Expected shape {(T // world_size, H, D)}, got {output.shape}"

    # Verify output values: each rank should get a chunk of sequence [T/CP] but full H
    # Rank 0 gets first T/CP elements of T, Rank 1 gets next T/CP elements of T
    # Each rank's H chunk should be gathered: first h_local elements from rank 0, next h_local from rank 1
    # So rank 0 output should have: first t_local timesteps, with H values [0.0, 0.0, 1.0, 1.0]
    # And rank 1 output should have: next t_local timesteps, with H values [0.0, 0.0, 1.0, 1.0]
    expected_h_chunk_0 = torch.full((t_local, h_local, D), 0.0, device=device, dtype=torch.float32)
    expected_h_chunk_1 = torch.full((t_local, h_local, D), 1.0, device=device, dtype=torch.float32)
    expected = torch.cat([expected_h_chunk_0, expected_h_chunk_1], dim=1)

    assert torch.allclose(output, expected), (
        f"Rank {rank}: Output values don't match expected. "
        f"Max diff = {(output - expected).abs().max()}, "
        f"Expected {t_local} timesteps with H values gathered from all ranks"
    )


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
@pytest.mark.parametrize("dim", [1, 2])
def test_cp2hp_scatter(backend: str, dim: int):
    func = _test_cp2hp_scatter_dim2 if dim == 2 else _test_cp2hp_scatter_dim1
    run_distributed_test(func, backend=backend, start_method="spawn", world_size=2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2])
def test_hp2cp_gather(backend: str, dim: int):
    func = _test_hp2cp_gather_dim2 if dim == 2 else _test_hp2cp_gather_dim1
    run_distributed_test(func, backend=backend, start_method="spawn", world_size=2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2])
def test_roundtrip(backend: str, dim: int):
    func = _test_roundtrip_dim2 if dim == 2 else _test_roundtrip_dim1
    run_distributed_test(func, backend=backend, start_method="spawn", world_size=2)
