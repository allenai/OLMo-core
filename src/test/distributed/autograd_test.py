import pytest
import torch
import torch.distributed as dist

from olmo_core.distributed.autograd import all_to_all
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device


def _all_to_all_equal_split():
    """Test all_to_all with equal splits (basic all2all) - forward and backward."""
    device = get_default_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group()
    assert group is not None

    # Create input tensor where each rank has [rank, rank, ...]
    # Shape: (world_size, 2) - each rank sends 2 elements to each other rank
    input_tensor = torch.full(
        (world_size, 2), float(rank), device=device, dtype=torch.float32, requires_grad=True
    )

    # Test forward pass
    output = all_to_all(group, input_tensor)

    # After all-to-all, each rank should have elements from all ranks
    # Row i should contain [i, i] (received from rank i)
    expected = torch.tensor(
        [[float(i)] * 2 for i in range(world_size)], device=device, dtype=torch.float32
    )
    assert torch.allclose(output, expected), f"Rank {rank}: expected {expected}, got {output}"

    # Test backward pass
    loss = output.sum()
    loss.backward()

    expected_grad = torch.ones_like(input_tensor)
    assert input_tensor.grad is not None, f"Rank {rank}: gradient is None"
    assert torch.allclose(input_tensor.grad, expected_grad), (
        f"Rank {rank}: expected grad {expected_grad}, got {input_tensor.grad}"
    )


def _all_to_all_unequal_split():
    """Test all_to_all with unequal splits (all2all-v) - forward and backward."""
    device = get_default_device()
    rank = dist.get_rank()
    group = dist.new_group()
    assert group is not None

    # Each rank sends different amounts to each other rank
    if rank == 0:
        input_split_sizes = [1, 2]  # send 1 to rank 0, 2 to rank 1
        output_split_sizes = [1, 2]  # receive 1 from rank 0, 2 from rank 1
        input_tensor = torch.tensor(
            [0.0, 1.0, 1.0], device=device, dtype=torch.float32, requires_grad=True
        )
    else:
        input_split_sizes = [2, 1]  # send 2 to rank 0, 1 to rank 1
        output_split_sizes = [2, 1]  # receive 2 from rank 0, 1 from rank 1
        input_tensor = torch.tensor(
            [10.0, 10.0, 11.0], device=device, dtype=torch.float32, requires_grad=True
        )

    # Test forward pass
    output = all_to_all(
        group,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )
    expected_size = sum(output_split_sizes)
    assert output.shape[0] == expected_size, (
        f"Rank {rank}: expected size {expected_size}, got {output.shape[0]}"
    )

    if rank == 0:
        expected = torch.tensor([0.0, 10.0, 10.0], device=device, dtype=torch.float32)
    else:
        expected = torch.tensor([1.0, 1.0, 11.0], device=device, dtype=torch.float32)
    assert torch.allclose(output, expected), f"Rank {rank}: expected {expected}, got {output}"

    # Test backward pass
    loss = output.sum()
    loss.backward()

    expected_grad = torch.ones_like(input_tensor)
    assert input_tensor.grad is not None, f"Rank {rank}: gradient is None"
    assert torch.allclose(input_tensor.grad, expected_grad), (
        f"Rank {rank}: expected grad {expected_grad}, got {input_tensor.grad}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_to_all_equal_split(backend: str):
    run_distributed_test(
        _all_to_all_equal_split, backend=backend, start_method="spawn", world_size=2
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_to_all_unequal_split(backend: str):
    run_distributed_test(
        _all_to_all_unequal_split, backend=backend, start_method="spawn", world_size=2
    )
