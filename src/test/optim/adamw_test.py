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
from olmo_core.nn.parametrization import MupScalingStrategy, WidthHyperParam
from olmo_core.optim import AdamWConfig, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.testing import DEVICES
from olmo_core.utils import cuda_sync_debug_mode


class MyModel(nn.Module):
    def __init__(self, bias: bool = True):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.fc1 = nn.Linear(16, 32, bias=bias)
        self.fc2 = nn.Linear(32, 16, bias=bias)

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


@pytest.mark.parametrize("optim_config_cls", [AdamWConfig, SkipStepAdamWConfig])
def test_adamw_parametrization_unchanged_weight_decay(optim_config_cls):
    from olmo_core.nn.parametrization import (
        ParametrizationConfig,
        ParametrizationOptimizerType,
    )

    lr = 1e-3

    optim_config = optim_config_cls(
        group_overrides=[
            OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.2)),
            OptimGroupOverride(params=["fc1.*"], opts=dict(weight_decay=0.5)),
        ],
        weight_decay=0.1,
        lr=lr,
    )
    parametrization_optimizer_type = optim_config.parametrization_optimizer_type()
    assert parametrization_optimizer_type == ParametrizationOptimizerType.adam_coupled_wd

    parametrization_config = ParametrizationConfig(
        optimizer=parametrization_optimizer_type,
        width_scalings={
            WidthHyperParam.d_model: 2,
            WidthHyperParam.hidden_size: 3,
            WidthHyperParam.head_dim: 2,
        },
        scaling_strategy=MupScalingStrategy.constant_inputs,
    )
    assert parametrization_config.optimizer.coupled_weight_decay

    model = MyModel(bias=False)
    model.parametrizations = {
        "wte.weight": parametrization_config.build(None, None),
        "fc1.weight": parametrization_config.build(
            {WidthHyperParam.d_model}, {WidthHyperParam.hidden_size}
        ),
        "fc2.weight": parametrization_config.build({WidthHyperParam.hidden_size}, None),
    }
    optim = optim_config.build(model)

    expected_weight_decays = {
        "wte.weight": 0.2 * lr,
        "fc1.weight": 0.5 * lr,
        "fc2.weight": 0.1 * lr,
    }

    param_to_name = {param: name for name, param in model.named_parameters()}

    for group in optim.param_groups:
        for p in group["params"]:
            param_name = param_to_name[p]
            assert param_name in expected_weight_decays, param_name
            expected_weight_decay = expected_weight_decays[param_name]
            # lr should be scaled by Parametrization except for wte.weight.
            assert group["lr"] != lr or param_name == "wte.weight", param_name
            # Overall weight decay should be unaffected by Parametrization scaling.
            assert group["weight_decay"] * group["lr"] == expected_weight_decay, param_name
