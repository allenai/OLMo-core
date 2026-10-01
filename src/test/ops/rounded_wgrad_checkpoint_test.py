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


@pytest.mark.parametrize("recompute", [False, True])
def test_rounded_wgrad_rejects_outstanding_forwards(monkeypatch, tmp_path, recompute):
    import torch.distributed as dist

    from olmo_core.nn.parallel import MultiGroupDistributedDataParallel

    def grouped(x, weight, *, offs):
        starts = [0, *offs.tolist()]
        return torch.cat([x[a:b] @ w for a, b, w in zip(starts, starts[1:], weight)])

    def accumulate(a, b, output, cumulative):
        for i, (start, end) in enumerate(zip(cumulative[:-1], cumulative[1:])):
            output[i].add_(a[:, start:end] @ b[:, start:end].T)

    monkeypatch.setattr(torch.nn.functional, "grouped_mm", grouped)
    monkeypatch.setattr(rounded_wgrad, "rounded_wgrad_add", accumulate)

    class Experts(torch.nn.Module):
        _profile_rounded_wgrad = True

        def __init__(self):
            super().__init__()
            self.w_up_gate = torch.nn.Parameter(torch.randn(2, 3, 3))
            self.w_down = torch.nn.Parameter(torch.randn(2, 3, 3))

        def forward(self, x):
            counts = torch.tensor([2, 3], dtype=torch.int32)

            def layers(x):
                x = rounded_wgrad.rounded_weight_gmm(x, self.w_up_gate, counts, False)
                return rounded_wgrad.rounded_weight_gmm(x, self.w_down, counts, False)

            return checkpoint(layers, x, use_reentrant=False) if recompute else layers(x)

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    try:
        ddp = MultiGroupDistributedDataParallel(
            Experts(), accumulate_grads_in_fp32=True, reduce_grads_in_fp32=True
        )
        x = torch.randn(5, 3, requires_grad=True)
        # No-grad evaluation must not consume a training forward's epoch.
        with torch.no_grad():
            ddp(x)
        with ddp.no_sync():
            output = ddp(x)
            with pytest.raises(RuntimeError, match="multiple outstanding forwards"):
                ddp(x)
            output.sum().backward()
        # Sequential microbatch accumulation and checkpoint recomputation remain valid.
        ddp(x).sum().backward()
        ddp.finalize_grad_reduce()
        assert all(torch.isfinite(p._main_grad_fp32).all() for p in ddp.parameters())
    finally:
        dist.destroy_process_group()
