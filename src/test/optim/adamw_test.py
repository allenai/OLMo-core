from test.utils import DEVICES

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.optim import AdamWConfig, OptimGroupOverride, SkipStepAdamWConfig


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.out = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.wte(x))


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
def test_skip_step_adamw(device: torch.device):
    config = SkipStepAdamWConfig()
    model = MyModel().train().to(device)
    optim = config.build(model)

    for group in optim.param_groups:
        assert "initial_lr" in group

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()
