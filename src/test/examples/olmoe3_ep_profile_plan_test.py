"""CPU-only checks that the EP harness cannot silently change batch/topology."""

import importlib.util
import sys
from pathlib import Path

import pytest

path = Path(__file__).parents[2] / "examples/olmo_ddp/olmoe3_ep_profile_plan.py"
spec = importlib.util.spec_from_file_location("ep_plan_test_module", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
EPProfileTopology = module.EPProfileTopology


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_fixed_batch_and_node_local_mesh(ep):
    plan = EPProfileTopology.from_test_label(f"ep{ep}-mb4")
    assert plan.accumulation == 8
    assert plan.expert_dp * ep == 64
    plan.validate_rank_groups(
        [list(range(start, start + ep)) for start in range(0, 64, ep)],
        {rank: rank // 8 for rank in range(64)},
    )
    assert EPProfileTopology.from_test_label(f"ep{ep}-mb8").accumulation == 4


@pytest.mark.parametrize("label", ["ep16-mb4", "ep3-mb4", "ep4-mb16", "ep4", "ep4-mb4-extra"])
def test_reject_unapproved_shapes(label):
    with pytest.raises(ValueError):
        EPProfileTopology.from_test_label(label)


def test_existing_profiles_unchanged_and_cross_node_ep_rejected():
    assert EPProfileTopology.from_test_label("deferred-lb") == EPProfileTopology()
    groups = [list(range(start, start + 4)) for start in range(0, 64, 4)]
    groups[0][0], groups[2][0] = groups[2][0], groups[0][0]
    with pytest.raises(ValueError, match="one node"):
        EPProfileTopology(ep=4).validate_rank_groups(groups, {r: r // 8 for r in range(64)})
