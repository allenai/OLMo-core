import copy
import json
from itertools import product
from typing import Optional

import pytest
import torch
import torch.nn as nn

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.optim import AdamWConfig, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.optim.adamw import SkipStepAdamW, foreach_adamw_step
from olmo_core.testing import DEVICES, requires_gpu
from olmo_core.utils import cuda_sync_debug_mode


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


def test_adamw_config_to_optim():
    config = AdamWConfig()

    model = MyModel()
    optim = config.build(model)

    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 1

    assert config.merge(["lr=1e-1"]).lr == 0.1


def test_adamw_config_to_optim_with_group_overrides():
    config = AdamWConfig(
        group_overrides=[OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.0))]
    )

    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0
    assert len(optim.param_groups[0]["params"]) == 1
    assert len(optim.param_groups[1]["params"]) == len(list(model.parameters())) - 1

    assert config.merge(["lr=1e-1"]).lr == 0.1

    for group in optim.param_groups:
        assert "initial_lr" in group


@pytest.mark.parametrize("device", DEVICES)
def test_adamw(device: torch.device, tmp_path):
    config = AdamWConfig()
    model = MyModel().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    # Save and then restore a checkpoint, and make sure fixed fields reset.
    for group in optim.param_groups:
        group["initial_lr"] = 1e-8
    save_model_and_optim_state(tmp_path, model, optim)
    load_model_and_optim_state(tmp_path, model, optim)
    for group in optim.param_groups:
        assert group["initial_lr"] == config.lr


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [None, DType.bfloat16])
def test_skip_step_adamw(device: torch.device, dtype: Optional[DType]):
    if dtype == DType.bfloat16 and device.type == "cpu":
        pytest.skip("bfloat16 dtype requires cuda")

    config = SkipStepAdamWConfig(dtype=dtype)
    model = MyModel().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [None, DType.bfloat16])
def test_skip_step_adamw_foreach(device: torch.device, dtype: Optional[DType]):
    if dtype == DType.bfloat16 and device.type == "cpu":
        pytest.skip("bfloat16 dtype requires cuda")

    config = SkipStepAdamWConfig(dtype=dtype, foreach=True)
    model = MyModel().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()


_LRS = [1e-3, 5e-4]
_WDS = [0.0, 1e-2]
_BETAS = [(0.9, 0.999), (0.7, 0.95)]


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("bad_value", [1e3, 1e20, float("inf"), float("nan")])
@pytest.mark.parametrize("accepted_steps", [0, 1, 8])
@pytest.mark.parametrize("eps", [0.0, 1e-8])
def test_skip_step_adamw_preserves_state_and_recovers(foreach, bad_value, accepted_steps, eps):
    p = nn.Parameter(torch.ones(3))
    optim = SkipStepAdamW([p], foreach=foreach, rolling_interval_length=8, eps=eps)
    # The reference sees only accepted updates, so it must remain identical after
    # the skipped update and after the next accepted update, including bias correction.
    ref_p = nn.Parameter(p.detach().clone())
    ref = torch.optim.AdamW([ref_p], lr=1e-3, weight_decay=1e-2, foreach=False, eps=eps)
    for _ in range(accepted_steps):
        optim.latest_loss = torch.tensor(1.0)
        optim.latest_grad_norm = torch.tensor(1.0)
        p.grad = torch.ones_like(p)
        ref_p.grad = torch.ones_like(ref_p)
        optim.step()
        ref.step()
    state_before = copy.deepcopy(optim.state[p])
    if not state_before:
        # Lazy initialization on a fresh skipped step may allocate zero buffers.
        state_before = dict(
            step=torch.zeros(()), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p)
        )
    param_before = p.detach().clone()
    # Finite outlier detection intentionally waits for warmup. Nonfinite metrics
    # must be rejected immediately, even on the very first sample.
    if accepted_steps < 4 and torch.isfinite(torch.tensor(bad_value)):
        optim.latest_loss = torch.tensor(float("inf"))
    else:
        optim.latest_loss = torch.tensor(bad_value)
    optim.latest_grad_norm = torch.tensor(bad_value)
    p.grad = torch.full_like(p, bad_value)
    grad_before = p.grad.clone()
    optim.step()
    assert optim.step_skipped.item() == 1.0
    torch.testing.assert_close(p, param_before, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, grad_before, rtol=0, atol=0, equal_nan=True)
    for key, value in state_before.items():
        torch.testing.assert_close(optim.state[p][key], value, rtol=0, atol=0)
    optim.latest_loss = torch.tensor(1.0)
    optim.latest_grad_norm = torch.tensor(1.0)
    p.grad = torch.ones_like(p)
    ref_p.grad = torch.ones_like(ref_p)
    optim.step()
    ref.step()
    assert optim.step_skipped.item() == 0.0
    torch.testing.assert_close(p, ref_p)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(optim.state[p][key], ref.state[ref_p][key])


@pytest.mark.parametrize("step_factor", [0.0, 1.0])
def test_foreach_adamw_temporary_memory(tmp_path, step_factor):
    """Bound live scratch storage without depending on a particular foreach implementation."""
    from torch.profiler import ProfilerActivity, profile

    params = [torch.ones(16384) for _ in range(8)]
    grads = [torch.full_like(p, 1e20 if step_factor == 0 else 1.0) for p in params]
    exp_avgs = [torch.ones_like(p) for p in params]
    exp_avg_sqs = [torch.ones_like(p) for p in params]
    steps = [torch.tensor(1.0) for _ in params]
    factor = torch.tensor(step_factor)
    gradient_bytes = sum(g.numel() * g.element_size() for g in grads)
    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True
    ) as prof:
        foreach_adamw_step(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            steps,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            step_factor=factor,
        )
    timeline = tmp_path / "memory.json"
    prof.export_memory_timeline(str(timeline), device="cpu")
    _, sizes = json.loads(timeline.read_text())
    # The first row contains pre-existing inputs/state. Allow two full scratch sets
    # (denominators + updates) and small scalar/list bookkeeping, but no retained
    # masked-gradient set. This catches the previous implementation on accepted steps.
    totals = [sum(row) for row in sizes]
    peak_extra = max(totals) - totals[0]
    assert peak_extra <= 2 * gradient_bytes + 4096
    assert all(torch.isfinite(p).all() for p in params)


@requires_gpu
@pytest.mark.parametrize("step_factor", [0.0, 1.0])
def test_foreach_adamw_cuda_temporary_memory(step_factor):
    params = [torch.ones(65536, device="cuda") for _ in range(8)]
    grads = [torch.full_like(p, 1e20 if step_factor == 0 else 1.0) for p in params]
    exp_avgs = [torch.ones_like(p) for p in params]
    exp_avg_sqs = [torch.ones_like(p) for p in params]
    steps = [torch.tensor(1.0, device="cuda") for _ in params]
    factor = torch.tensor(step_factor, device="cuda")
    gradient_bytes = sum(g.numel() * g.element_size() for g in grads)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with cuda_sync_debug_mode(debug_mode="error"):
        foreach_adamw_step(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            steps,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            step_factor=factor,
        )
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - allocated
    # Allow small CUDA allocation-rounding/scalar overhead, but not another shard.
    assert peak_extra <= 2 * gradient_bytes + 128 * 1024


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lr, wd, betas", list(product(_LRS, _WDS, _BETAS)))
def test_adamw_equivalence(
    device: torch.device,
    lr: float,
    wd: float,
    betas: tuple[float, float],
):
    """Ensure standard SkipStepAdamW matches torch's AdamW."""

    torch.manual_seed(123)

    model1 = MyModel().to(device)
    model2 = copy.deepcopy(model1).to(device)
    model3 = copy.deepcopy(model1).to(device)

    group_overrides = [OptimGroupOverride(params=["wte.*"], opts={"weight_decay": 0.0})]

    cfg_common = dict(
        lr=lr,
        betas=betas,
        weight_decay=wd,
        group_overrides=group_overrides,
    )

    optim1 = AdamWConfig(foreach=False, **cfg_common).build(model1)  # type: ignore[arg-type]
    optim2 = SkipStepAdamWConfig(foreach=True, **cfg_common).build(model2)  # type: ignore[arg-type]
    optim3 = SkipStepAdamWConfig(foreach=False, **cfg_common).build(model3)  # type: ignore[arg-type]

    # Training loop
    for step_idx in range(5):
        inp = torch.randint(0, 128, (4, 8), device=device)

        with cuda_sync_debug_mode(debug_mode="error"):
            for optim, model in [(optim1, model1), (optim2, model2), (optim3, model3)]:
                optim.zero_grad(set_to_none=True)
                loss = model(inp).sum()
                loss.backward()
                optim.step()

        # Compare parameters
        atol = 1e-5
        rtol = 1e-7
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=atol, rtol=rtol)
        for p1, p3 in zip(model1.parameters(), model3.parameters()):
            assert torch.allclose(p1, p3, atol=atol, rtol=rtol)
        # Compare optimizer state
        for t1, t2 in zip(model1.parameters(), model2.parameters()):
            st1, st2 = optim1.state[t1], optim2.state[t2]
            for key in ("exp_avg", "exp_avg_sq", "step"):
                v1, v2 = st1[key], st2[key]
                assert torch.allclose(v1, v2, atol=atol, rtol=rtol)
                if key == "step":
                    assert v1 == step_idx + 1  # step should equal current iteration + 1
        for t1, t3 in zip(model1.parameters(), model3.parameters()):
            st1, st3 = optim1.state[t1], optim3.state[t3]
            for key in ("exp_avg", "exp_avg_sq", "step"):
                v1, v3 = st1[key], st3[key]
                assert torch.allclose(v1, v3, atol=atol, rtol=rtol)
                if key == "step":
                    assert v1 == step_idx + 1  # step should equal current iteration + 1
