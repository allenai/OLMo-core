import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device, seed_all


class SelectiveModel(nn.Module):
    """Two independent branches; only one is used per forward."""

    def __init__(self, d: int):
        super().__init__()
        self.fc_a = nn.Linear(d, d)
        self.fc_b = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, use_a: bool = True) -> torch.Tensor:
        return self.fc_a(x) if use_a else self.fc_b(x)


class IgnoredParamModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.keep = nn.Linear(d, d)
        self.ignore = nn.Linear(d, d)
        self._ddp_params_and_buffers_to_ignore = {"ignore.weight", "ignore.bias"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keep(x) + self.ignore(x)


def _reference_grads(model: nn.Module, world_size: int) -> list[torch.Tensor]:
    grads = []
    for p in model.parameters():
        assert p.grad is not None
        g = p.grad.detach().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= world_size
        grads.append(g)
    return grads


def _run_no_sync_skipped_param_grad_preserved(d: int):
    device = get_default_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    seed_all(0)
    model = SelectiveModel(d).to(device)
    reference = copy.deepcopy(model)
    ddp = MultiGroupDistributedDataParallel(model, init_sync=False)

    torch.manual_seed(100 + rank)
    x_a = torch.randn(4, d, device=device)
    x_b = torch.randn(4, d, device=device)

    # fc_a receives a grad only in the unsynced accumulation micro-batch; fc_b only
    # in the final synced one. fc_a must survive finalize rather than be zeroed.
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


def _run_fp32_hooks_skip_ignored_params(d: int):
    device = get_default_device()

    seed_all(0)
    model = IgnoredParamModel(d).to(device)
    ddp = MultiGroupDistributedDataParallel(
        model,
        init_sync=False,
        accumulate_grads_in_fp32=True,
        reduce_grads_in_fp32=True,
    )

    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(4, d, device=device)
    ddp(x).pow(2).mean().backward()
    ddp.finalize_grad_reduce()

    assert getattr(ddp.module.keep.weight, "_main_grad_fp32", None) is not None
    assert getattr(ddp.module.ignore.weight, "_main_grad_fp32", None) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_sync_skipped_param_grad_preserved(backend: str):
    run_distributed_test(
        _run_no_sync_skipped_param_grad_preserved,
        backend=backend,
        func_args=(16,),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fp32_hooks_skip_ignored_params(backend: str):
    run_distributed_test(
        _run_fp32_hooks_skip_ignored_params,
        backend=backend,
        func_args=(16,),
    )
