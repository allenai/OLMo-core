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


class SelectiveModel(nn.Module):
    """Two independent branches; only one is used per forward (an imbalanced-routing stand-in)."""

    def __init__(self, d: int):
        super().__init__()
        self.fc_a = nn.Linear(d, d)
        self.fc_b = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, use_a: bool = True) -> torch.Tensor:
        return self.fc_a(x) if use_a else self.fc_b(x)


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


def _run_multi_group_grad_parity(d_in: int, d_hidden: int, d_out: int):
    device = _device_for_backend()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # fc1 reduces over the full group; fc2 reduces only over the rank's own singleton group (the
    # expert-parallel case where an expert lives on a single rank). So fc1 grads are averaged across
    # all ranks while fc2 grads stay per-rank — verifying params route to, and reduce only within,
    # their assigned group. All ranks must create every group (new_group is collective).
    solo_groups = [dist.new_group([r]) for r in range(world_size)]
    my_solo_group = solo_groups[rank]

    # Identical init across all ranks; distinct per-rank data.
    seed_all(0)
    model = SimpleModel(d_in, d_hidden, d_out).to(device)
    reference = copy.deepcopy(model)

    def param_process_group_fn(name: str, param: torch.nn.Parameter):
        # None -> the default (full) process group.
        return None if name.startswith("fc1") else my_solo_group

    ddp = MultiGroupDistributedDataParallel(
        model, init_sync=False, param_process_group_fn=param_process_group_fn
    )

    torch.manual_seed(100 + rank)
    x = torch.randn(4, d_in, device=device)
    y = torch.randn(4, d_out, device=device)
    ((ddp(x) - y) ** 2).mean().backward()
    ddp.finalize_grad_reduce()

    # Reference: fc1 averaged over the world; fc2 left as the rank's local grad (its singleton group
    # reduce is a no-op), so fc2 differs across ranks while fc1 matches.
    ((reference(x) - y) ** 2).mean().backward()
    for name, p in reference.named_parameters():
        assert p.grad is not None
        if name.startswith("fc1"):
            g = p.grad.detach().clone()
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= world_size
            p.grad = g

    for (name, p), (_, p_ref) in zip(ddp.module.named_parameters(), reference.named_parameters()):
        assert p.grad is not None, f"missing grad for {name}"
        torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-5, atol=1e-6)


def _run_fp32_grad_accumulation(d_in: int, d_hidden: int, d_out: int):
    device = _device_for_backend()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = SimpleModel(d_in, d_hidden, d_out).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(
        model, init_sync=False, accumulate_grads_in_fp32=True, reduce_grads_in_fp32=True
    )

    torch.manual_seed(100 + rank)
    x = torch.randn(4, d_in, device=device)
    y = torch.randn(4, d_out, device=device)
    ((ddp(x) - y) ** 2).mean().backward()
    ddp.finalize_grad_reduce()

    ((reference(x) - y) ** 2).mean().backward()
    expected = _reference_grads(reference, world_size)

    # In fp32-accumulate mode the reduced gradient lives in the fp32 buffer `_main_grad_fp32`, and
    # `.grad` is consumed/left as None.
    for (name, p), g_ref in zip(ddp.module.named_parameters(), expected):
        assert p.grad is None, f"expected .grad to be None in fp32-accum mode for {name}"
        main_grad = getattr(p, "_main_grad_fp32", None)
        assert main_grad is not None, f"missing _main_grad_fp32 for {name}"
        assert main_grad.dtype == torch.float32
        torch.testing.assert_close(main_grad, g_ref.to(torch.float32), rtol=1e-5, atol=1e-6)


def _run_no_sync_skipped_param_grad_preserved(d: int):
    device = _device_for_backend()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = SelectiveModel(d).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(model, init_sync=False)

    torch.manual_seed(100 + rank)
    x_a = torch.randn(4, d, device=device)
    x_b = torch.randn(4, d, device=device)

    # fc_a receives a grad only in the unsynced accumulation micro-batch; fc_b only in the final
    # synced one. fc_a is therefore never marked ready in the synced pass — its accumulated grad
    # must survive finalize rather than be zeroed.
    with ddp.no_sync():
        ddp(x_a, use_a=True).pow(2).mean().backward()
    ddp(x_b, use_a=False).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    reference(x_a, use_a=True).pow(2).mean().backward()
    reference(x_b, use_a=False).pow(2).mean().backward()
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
def test_multi_group_grad_parity(backend):
    run_distributed_test(
        _run_multi_group_grad_parity,
        backend=backend,
        func_args=(16, 32, 8),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp32_grad_accumulation(backend):
    run_distributed_test(
        _run_fp32_grad_accumulation,
        backend=backend,
        func_args=(16, 32, 8),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_sync_skipped_param_grad_preserved(backend):
    run_distributed_test(
        _run_no_sync_skipped_param_grad_preserved,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_sync_accumulation(backend):
    run_distributed_test(
        _run_no_sync_accumulation,
        backend=backend,
        func_args=(16, 32, 8),
    )
