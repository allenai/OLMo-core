"""CPU regression for parameter ownership across saved-tensor unpack/recomputation.

Only the GEMMs are mocked; the real custom autograd function and checkpoint
machinery execute. CUDA/distributed numerical parity is tested separately.
"""

from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from olmo_core.ops import rounded_wgrad


@pytest.mark.parametrize("mode", ["plain", "unpack", "recompute"])
@pytest.mark.parametrize("transpose", [False, True])
def test_rounded_wgrad_preserves_parameter_owner(monkeypatch, mode, transpose):
    def grouped(x, weight, *, offs):
        starts = [0, *offs.tolist()]
        return torch.cat([x[a:b] @ w for a, b, w in zip(starts, starts[1:], weight)])

    def accumulate(a, b, output, cumulative):
        for i, (start, end) in enumerate(zip(cumulative[:-1], cumulative[1:])):
            output[i].add_(a[:, start:end] @ b[:, start:end].T)

    monkeypatch.setattr(torch.nn.functional, "grouped_mm", grouped)
    monkeypatch.setattr(rounded_wgrad, "rounded_wgrad_add", accumulate)
    torch.manual_seed(318)
    source = torch.randn(2, 3, 4, dtype=torch.float64)
    weight = torch.nn.Parameter(source.transpose(1, 2).contiguous() if transpose else source)
    reference = torch.nn.Parameter(weight.detach().clone())
    destination = torch.zeros_like(weight)
    events = []

    def begin(owner):
        assert owner is weight
        assert owner.is_leaf and owner.grad is None
        events.append("begin")
        return destination, lambda: events.append("done")

    weight._olmo_profile_begin_external_grad = begin
    counts = torch.tensor([2, 3], dtype=torch.int32)
    for _ in range(3):
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()

        def forward(x):
            return rounded_wgrad.rounded_weight_gmm(x, weight, counts, transpose)

        hooks = (
            torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t)
            if mode == "unpack"
            else nullcontext()
        )
        with hooks:
            y = checkpoint(forward, x, use_reentrant=False) if mode == "recompute" else forward(x)
            loss = y.square().sum()
        loss.backward()
        ref_y = grouped(
            ref_x, reference.transpose(1, 2) if transpose else reference, offs=counts.cumsum(0)
        )
        ref_y.square().sum().backward()
        torch.testing.assert_close(y, ref_y, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
        torch.testing.assert_close(destination, reference.grad, rtol=0, atol=0)
        assert weight.grad is None
    assert events == ["begin", "done"] * 3
