import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_probe_module():
    path = Path(__file__).parents[2] / "scripts" / "eval" / "s002_routing_probe.py"
    name = "_s002_routing_probe_test_target"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()


def test_modality_masks_partition_valid_tokens():
    input_ids = torch.tensor([[1, 99, 7, 8, 0]])
    token_type_ids = torch.tensor([[0, 1, 1, 0, 0]])
    valid = torch.tensor([[True, True, True, True, False]])

    masks = probe._modality_masks(
        input_ids,
        token_type_ids,
        valid,
        image_patch_token_id=99,
    )

    assert masks["text"].tolist() == [[True, False, False, True, False]]
    assert masks["image_patch"].tolist() == [[False, True, False, False, False]]
    assert masks["image_structural"].tolist() == [[False, False, True, False, False]]
    assert torch.equal(
        masks["text"] | masks["image_patch"] | masks["image_structural"],
        masks["all"],
    )


def test_modality_masks_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        probe._modality_masks(
            torch.ones(1, 2),
            torch.ones(1, 3),
            torch.ones(1, 2, dtype=torch.bool),
            image_patch_token_id=99,
        )


def test_capacity_stats_uses_physical_shape_and_valid_route_denominator():
    # Two destination ranks own two experts each. Rank 0 receives 14 routes and rank 1
    # receives 6. Physical capacity is ceil(1.25 * 10) = 13, so exactly one route drops.
    stats = probe._capacity_stats(
        torch.tensor([8, 6, 3, 3]),
        ep_world_size=2,
        physical_routes_per_source=10,
        capacity_factor=1.25,
        global_valid_routes=20,
    )

    assert stats["rank_capacity"] == 13
    assert stats["destination_route_counts"] == [14, 6]
    assert stats["dropped_routes"] == 1
    assert stats["global_drop_rate"] == pytest.approx(0.05)
    assert stats["destination_utilization"] == pytest.approx([1.0, 6 / 13])
    assert stats["max_destination_utilization"] == pytest.approx(1.0)
    assert stats["requested_destination_pressure"] == pytest.approx([14 / 13, 6 / 13])
    assert stats["max_requested_destination_pressure"] == pytest.approx(14 / 13)


def test_capacity_stats_rejects_nondivisible_expert_partition():
    with pytest.raises(ValueError, match="divisible"):
        probe._capacity_stats(
            torch.ones(5),
            ep_world_size=2,
            physical_routes_per_source=10,
            capacity_factor=1.0,
            global_valid_routes=20,
        )


@pytest.mark.parametrize(
    ("target", "expected"),
    [(None, (6001, 1)), (6019, (6019, 19))],
)
def test_resolve_replay_step(target, expected):
    assert probe._resolve_replay_step(6000, target) == expected


def test_resolve_replay_step_rejects_saved_or_earlier_step():
    with pytest.raises(ValueError, match="must be later"):
        probe._resolve_replay_step(6000, 6000)
