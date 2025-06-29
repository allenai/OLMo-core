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
from olmo_core.optim import (
    AdamWConfig,
    OptimGroupOverride,
    SkipStepAdamWConfig,
    SkipStepAdamWV2Config,
)
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.testing import DEVICES
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


_LRS = [0.0, 1e-3, 5e-4]
_WDS = [0.0, 1e-2]
_BETAS = [(0.9, 0.999), (0.7, 0.95)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [None, DType.bfloat16])
@pytest.mark.parametrize("lr, wd, betas", list(product(_LRS, _WDS, _BETAS)))
@pytest.mark.parametrize("trigger_skip", [False, True])
@pytest.mark.parametrize(
    "use_skip_step_baseline",
    [pytest.param(True, id="skip-step-base"), pytest.param(False, id="adamw-base")],
)
def test_skipstep_adamw_equivalence(
    device: torch.device,
    dtype: Optional[DType],
    lr: float,
    wd: float,
    betas: tuple[float, float],
    trigger_skip: bool,
    use_skip_step_baseline: bool,
):
    """Compare full optimiser state across a variety of settings."""

    if dtype == DType.bfloat16 and device.type == "cpu":
        pytest.skip("bfloat16 dtype requires cuda")

    torch.manual_seed(123)

    model1 = MyModel().to(device)
    model2 = copy.deepcopy(model1).to(device)

    # parameter-group override: treat embedding separately
    group_overrides = [OptimGroupOverride(params=["wte.*"], opts={"weight_decay": 0.0})]

    cfg_common = dict(
        lr=lr,
        betas=betas,
        weight_decay=wd,
        group_overrides=group_overrides,
    )

    if use_skip_step_baseline:
        rolling_interval_length = 2
        optim1 = SkipStepAdamWConfig(**cfg_common, dtype=dtype, rolling_interval_length=rolling_interval_length, sigma_factor=1).build(model1)  # type: ignore[arg-type]
    else:
        rolling_interval_length = 16
        optim1 = AdamWConfig(**cfg_common, foreach=False, fused=False).build(model1)  # type: ignore[arg-type]
    optim2 = SkipStepAdamWV2Config(**cfg_common, foreach=False, fused=False, dtype=dtype, rolling_interval_length=rolling_interval_length, sigma_factor=1).build(model2)  # type: ignore[arg-type]

    # training loop
    for step_idx in range(5):
        huge = torch.tensor(1e9, device=device)
        inp = torch.randint(0, 128, (4, 8), device=device)

        with cuda_sync_debug_mode(2):
            optim1.zero_grad(set_to_none=True)
            optim2.zero_grad(set_to_none=True)

            loss1 = model1(inp).sum()
            loss2 = model2(inp).sum()

            # Inject an outlier loss on the 3rd step to trigger skip logic.
            if trigger_skip and step_idx == 2:
                optim1.latest_loss = huge.detach()
                optim2.latest_loss = huge.detach()
            else:
                optim1.latest_loss = loss1.detach()
                optim2.latest_loss = loss2.detach()

            loss1.backward()
            loss2.backward()

            optim1.step()
            optim2.step()

        if use_skip_step_baseline:
            # If we requested a skip, ensure both optimisers agree on whether it happened.
            assert torch.equal(optim1.step_skipped, optim2.step_skipped)
            if trigger_skip and step_idx == 2:
                assert torch.equal(optim1.step_skipped, torch.tensor(True))

        # compare parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)

        # compare optimizer state
        for t1, t2 in zip(model1.parameters(), model2.parameters()):
            st1, st2 = optim1.state[t1], optim2.state[t2]
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st1:  # all keys should exist but be defensive
                    v1, v2 = st1[key], st2[key]
                    torch.testing.assert_close(
                        v1,
                        v2,
                        atol=1e-6,
                        rtol=1e-5,
                        msg=lambda msg: f"Step {step_idx}, Key {key}, msg: {msg}",
                    )
