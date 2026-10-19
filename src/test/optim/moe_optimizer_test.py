import pytest
import torch.nn as nn

from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig
from olmo_core.optim.mup import MuPConfig


def test_build_groups_applies_overrides():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        group_overrides=[OptimGroupOverride(params=["*bias*"], opts={"weight_decay": 0.0})],
    )
    groups = config.build_groups([model])
    assert isinstance(groups, list)

    # Every parameter lands in exactly one group.
    grouped = [p for g in groups for p in g["named_params"].values()]
    assert len(grouped) == len(list(model.parameters()))

    # The bias parameters land in a group carrying the override option.
    bias_ids = {id(p) for n, p in model.named_parameters() if "bias" in n}
    override_group = next(g for g in groups if g.get("weight_decay") == 0.0)
    assert {id(p) for p in override_group["named_params"].values()} == bias_ids


def test_build_groups_applies_mup_with_wrapper_prefix():
    class Wrapper(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    model = Wrapper(nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)))
    config = OLMoDDPOptimizerConfig(
        lr=1e-3,
        weight_decay=0.1,
        mup=MuPConfig(base_width=8),
        group_overrides=[OptimGroupOverride(params=["*bias*"], opts={"weight_decay": 0.0})],
    )
    groups = config.build_groups([model])

    by_name = {name: g for g in groups for name in g["named_params"]}
    assert set(by_name) == {name for name, _ in model.named_parameters()}
    assert by_name["module.0.weight"]["lr"] == pytest.approx(1e-3)
    assert by_name["module.1.weight"]["lr"] == pytest.approx(0.5e-3)
    for name in ("module.0.bias", "module.1.bias"):
        assert by_name[name]["lr"] == pytest.approx(1e-3)
        assert by_name[name]["weight_decay"] == 0.0
    # The default group's splits still come first, and leave weight decay to the optimizer default.
    assert "weight_decay" not in groups[0]
    assert "weight_decay" not in by_name["module.1.weight"]
