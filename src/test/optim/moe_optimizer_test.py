import pytest
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig


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


def test_partitioned_groups_validate_overrides_before_splitting():
    model = nn.ModuleDict(
        {
            "dense": nn.Linear(4, 4),
            "expert": nn.Linear(4, 4),
        }
    )
    model["expert"]._ep_sharded = True
    config = OLMoDDPOptimizerConfig(
        group_overrides=[
            OptimGroupOverride(params=["dense.*"], opts={"lr": 1e-4}),
            OptimGroupOverride(params=["expert.*"], opts={"lr": 2e-4}),
        ]
    )

    groups = config._build_partitioned_groups([model], strict=True)

    dense_group = next(group for group in groups if group.get("lr") == 1e-4)
    expert_group = next(group for group in groups if group.get("lr") == 2e-4)
    assert dense_group["pg"] == "dp"
    assert set(dense_group["named_params"]) == {"dense.weight", "dense.bias"}
    assert expert_group["pg"] == "ep_dp"
    assert set(expert_group["named_params"]) == {"expert.weight", "expert.bias"}


def test_partitioned_groups_reject_glob_that_matches_neither_partition():
    model = nn.Linear(4, 4)
    config = OLMoDDPOptimizerConfig(
        group_overrides=[OptimGroupOverride(params=["missing.*"], opts={"lr": 1e-4})]
    )

    with pytest.raises(OLMoConfigurationError, match="does not match any parameters"):
        config._build_partitioned_groups([model], strict=True)
