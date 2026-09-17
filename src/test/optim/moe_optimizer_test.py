import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim.config import OptimGroupOverride
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer, OLMoDDPOptimizerConfig


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


def test_clip_grad_by_scheduler_group_scales_groups_independently():
    language = nn.Parameter(torch.ones(1))
    connector = nn.Parameter(torch.ones(1))
    language_grad = torch.tensor([3.0, 4.0])
    connector_grad = torch.tensor([0.5])

    optim = OLMoDDPOptimizer.__new__(OLMoDDPOptimizer)
    optim.param_groups = [
        {"named_params": {"lm.weight": language}, "pg": "dp"},
        {
            "named_params": {"connector.weight": connector},
            "pg": "dp",
            "scheduler_name": "connector",
        },
    ]
    optim.main_grad = {
        "lm.weight": language_grad,
        "connector.weight": connector_grad,
    }
    optim.max_grad_norm = 1.0
    optim.clip_grad_norm_by_scheduler_group = True
    optim.check_nan_inf_grad = False
    optim._partition_main_grads = lambda names=None: (
        [optim.main_grad[name] for name in (names or optim.main_grad)],
        [],
        [],
        [],
    )
    optim._compute_total_grad_norm = lambda grads, *_: torch.linalg.vector_norm(
        torch.cat(grads), ord=2
    )
    optim._maybe_debug_nan_inf_grad_norm = lambda *_: None

    total_norm = optim._clip_grad()

    torch.testing.assert_close(total_norm, torch.sqrt(torch.tensor(25.25)))
    torch.testing.assert_close(language_grad, torch.tensor([0.6, 0.8]), rtol=0, atol=2e-7)
    torch.testing.assert_close(connector_grad, torch.tensor([0.5]))
    assert set(optim.latest_clip_group_grad_norms) == {"<default>", "connector"}
    torch.testing.assert_close(optim.latest_clip_group_grad_norms["<default>"], torch.tensor(5.0))
    torch.testing.assert_close(optim.latest_clip_group_coefficients["connector"], torch.tensor(1.0))


def test_clip_grad_combines_dp_and_ep_fragments_with_one_scheduler_name():
    dense = nn.Parameter(torch.ones(1))
    expert = nn.Parameter(torch.ones(1))
    optim = OLMoDDPOptimizer.__new__(OLMoDDPOptimizer)
    optim.param_groups = [
        {
            "named_params": {"lm.dense": dense},
            "pg": "dp",
            "scheduler_name": "language",
        },
        {
            "named_params": {"lm.expert": expert},
            "pg": "ep_dp",
            "scheduler_name": "language",
        },
    ]
    optim.main_grad = {
        "lm.dense": torch.tensor([3.0]),
        "lm.expert": torch.tensor([4.0]),
    }
    optim.max_grad_norm = 1.0
    optim.clip_grad_norm_by_scheduler_group = True
    optim.check_nan_inf_grad = False

    def partition(names=None):
        selected = set(names or optim.main_grad)
        return (
            [optim.main_grad["lm.dense"]] if "lm.dense" in selected else [],
            [],
            [optim.main_grad["lm.expert"]] if "lm.expert" in selected else [],
            [],
        )

    optim._partition_main_grads = partition
    optim._compute_total_grad_norm = lambda *groups: torch.linalg.vector_norm(
        torch.cat([grad for group in groups for grad in group]), ord=2
    )
    optim._maybe_debug_nan_inf_grad_norm = lambda *_: None

    total_norm = optim._clip_grad()

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    assert list(optim.latest_clip_group_grad_norms) == ["language"]
    torch.testing.assert_close(optim.main_grad["lm.dense"], torch.tensor([0.6]), atol=2e-7, rtol=0)
    torch.testing.assert_close(optim.main_grad["lm.expert"], torch.tensor([0.8]), atol=2e-7, rtol=0)


def test_clip_grad_global_path_retains_one_coefficient():
    first = nn.Parameter(torch.ones(1))
    second = nn.Parameter(torch.ones(1))
    optim = OLMoDDPOptimizer.__new__(OLMoDDPOptimizer)
    optim.param_groups = [
        {"named_params": {"first": first}, "pg": "dp"},
        {"named_params": {"second": second}, "pg": "dp", "scheduler_name": "other"},
    ]
    optim.main_grad = {"first": torch.tensor([3.0, 4.0]), "second": torch.tensor([0.5])}
    optim.max_grad_norm = 1.0
    optim.clip_grad_norm_by_scheduler_group = False
    optim.check_nan_inf_grad = False
    optim._partition_main_grads = lambda names=None: (
        [optim.main_grad[name] for name in (names or optim.main_grad)],
        [],
        [],
        [],
    )
    optim._compute_total_grad_norm = lambda grads, *_: torch.linalg.vector_norm(
        torch.cat(grads), ord=2
    )
    optim._maybe_debug_nan_inf_grad_norm = lambda *_: None

    total_norm = optim._clip_grad()

    expected_coefficient = 1.0 / (total_norm + 1e-6)
    torch.testing.assert_close(total_norm, torch.sqrt(torch.tensor(25.25)))
    torch.testing.assert_close(
        optim.main_grad["first"], torch.tensor([3.0, 4.0]) * expected_coefficient
    )
    torch.testing.assert_close(
        optim.main_grad["second"], torch.tensor([0.5]) * expected_coefficient
    )
    assert not optim.latest_clip_group_grad_norms
    assert not optim.latest_clip_group_coefficients
