"""Protect topology, exact output names and the bounded medium comparison."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2] / "examples/olmo_ddp"
_SPEC = importlib.util.spec_from_file_location(
    "medium_plan", _ROOT / "olmoe3_medium_profile_plan.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("gpus,ga", [(64, 16), (128, 8)])
def test_topology(gpus, ga):
    topology = _MODULE.MediumProfileTopology(gpus)
    assert topology.accumulation == ga
    hosts = [f"node{rank // 8}" for rank in range(gpus)]
    groups = [list(range(rank, rank + 8)) for rank in range(0, gpus, 8)]
    topology.validate_rank_groups(groups, hosts)
    groups[0][0], groups[1][0] = groups[1][0], groups[0][0]
    with pytest.raises(ValueError, match="node-local"):
        topology.validate_rank_groups(groups, hosts)


@pytest.mark.parametrize(
    "overrides",
    [
        {"gpus": 32},
        {"gpus": 256},
        {"gpus": 64, "microbatch": 4},
        {"gpus": 64, "ep": 4},
        {"gpus": 64, "batch_tokens": 8388608},
    ],
)
def test_reject_unapproved_axes(overrides):
    with pytest.raises(ValueError):
        _MODULE.MediumProfileTopology(**overrides)


def test_names_and_capture_boundaries():
    timing = _MODULE.named_medium_passes("medium-g64")
    assert [arm for _, arm, _ in timing] == ["baseline", "optimized", "optimized", "baseline"]
    assert timing[2] == ("medium-g64-repeat2-optimized", "optimized", "timing")
    all_passes = _MODULE.named_medium_passes("medium-g128", capture=True)
    assert len(all_passes) == 8
    assert all_passes[4] == ("medium-g128-baseline", "baseline", "nsys")
    result_names = [f"{name}-{mode}" for name, _, mode in all_passes]
    assert len(result_names) == len(set(result_names))
    assert "medium-g128-baseline-nsys" in result_names
    assert "medium-g128-repeat1-baseline-nsys" not in result_names


def test_only_baseline_or_qualified_flags():
    spec = importlib.util.spec_from_file_location(
        "integration_policy", _ROOT / "olmoe3_integration_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for arm in ("reference", "optimized"):
        settings = module.integration_policy(arm, module.QUALIFIED_POLICY)
        assert settings["communication"] == "none"
        assert settings["flags"]["OLMO_PROFILE_LB_COUNT_BATCHED"] == "0"
        assert settings["flags"]["OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS"] == "0"
        assert settings["reduce_scatter"] == (arm == "optimized")
