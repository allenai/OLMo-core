import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.nn.parallel import MultiGroupDistributedDataParallel
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import seed_all


class SimpleModel(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _device_for_backend() -> torch.device:
    if dist.get_backend() == "nccl":
        device = torch.device(f"cuda:{dist.get_rank()}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _reference_grads(model: nn.Module, world_size: int):
    """Manually all-reduce-average each parameter's local grad as the DDP reference."""
    grads = []
    for p in model.parameters():
        assert p.grad is not None
        g = p.grad.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= world_size
        grads.append(g)
    return grads


def _run_grad_parity(d_in: int, d_hidden: int, d_out: int):
    device = _device_for_backend()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Identical init across ranks, so init_sync isn't needed.
    seed_all(0)
    model = SimpleModel(d_in, d_hidden, d_out).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(model, init_sync=False)

    # Distinct per-rank batch (data parallelism).
    torch.manual_seed(100 + rank)
    x = torch.randn(4, d_in, device=device)
    y = torch.randn(4, d_out, device=device)

    ((ddp(x) - y) ** 2).mean().backward()
    ddp.finalize_grad_reduce()

    ((reference(x) - y) ** 2).mean().backward()
    expected = _reference_grads(reference, world_size)

    for (name, p), g_ref in zip(ddp.module.named_parameters(), expected):
        assert p.grad is not None, f"missing grad for {name}"
        torch.testing.assert_close(p.grad, g_ref, rtol=1e-5, atol=1e-6)


def _run_no_sync_accumulation(d_in: int, d_hidden: int, d_out: int):
    device = _device_for_backend()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = SimpleModel(d_in, d_hidden, d_out).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(model, init_sync=False)

    torch.manual_seed(100 + rank)
    xa = torch.randn(4, d_in, device=device)
    ya = torch.randn(4, d_out, device=device)
    xb = torch.randn(4, d_in, device=device)
    yb = torch.randn(4, d_out, device=device)

    # First micro-batch accumulates without syncing; the second (synced) triggers the reduce.
    with ddp.no_sync():
        ((ddp(xa) - ya) ** 2).mean().backward()
    ((ddp(xb) - yb) ** 2).mean().backward()
    ddp.finalize_grad_reduce()

    # Reference: accumulate both micro-batch grads locally, then all-reduce-average.
    ((reference(xa) - ya) ** 2).mean().backward()
    ((reference(xb) - yb) ** 2).mean().backward()
    expected = _reference_grads(reference, world_size)

    for (name, p), g_ref in zip(ddp.module.named_parameters(), expected):
        assert p.grad is not None, f"missing grad for {name}"
        torch.testing.assert_close(p.grad, g_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_grad_parity(backend):
    run_distributed_test(
        _run_grad_parity,
        backend=backend,
        func_args=(16, 32, 8),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_sync_accumulation(backend):
    run_distributed_test(
        _run_no_sync_accumulation,
        backend=backend,
        func_args=(16, 32, 8),
    )
