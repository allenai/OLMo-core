import copy
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
from olmo_core.testing import DEVICES
from olmo_core.utils import cuda_sync_debug_mode, seed_all


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bnb_style", [False, True], ids=["standard", "bnb_style"])
@pytest.mark.parametrize(
    "betas", [(0.9, 0.999), (0.9, 0.95)], ids=["beta_0.9_0.999", "beta_0.9_0.95"]
)
@pytest.mark.parametrize("wd", [0.0, 1e-2], ids=["no_wd", "wd"])
@pytest.mark.parametrize("lr", [1e-3, 5e-4], ids=["lr_1e-3", "lr_5e-4"])
def test_adamw_equivalence(
    device: torch.device,
    lr: float,
    wd: float,
    betas: tuple[float, float],
    bnb_style: bool,
):
    """Ensure SkipStepAdamW matches torch's AdamW."""
    seed_all(0)

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
    optim2 = SkipStepAdamWConfig(foreach=True, bnb_style=bnb_style, **cfg_common).build(model2)  # type: ignore[arg-type]
    optim3 = SkipStepAdamWConfig(foreach=False, bnb_style=bnb_style, **cfg_common).build(model3)  # type: ignore[arg-type]
    num_steps = 5

    # Training loop
    for step_idx in range(num_steps):
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
