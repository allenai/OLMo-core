from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def health_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval"
        / "vision_alignment_ssmax_bridge_health.py"
    )
    spec = importlib.util.spec_from_file_location("ssmax_bridge_health_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unpacked_health_accumulation_matches_online_source_telemetry(health_module) -> None:
    stats = health_module._empty_stats()
    batch = {
        "source_names": ["pixmo_caption", "pixmo_transcript"],
        "router_token_mask": torch.tensor([[True, True, True, False], [True, True, False, False]]),
        "labels": torch.tensor([[1, -100, 2, -100], [3, 4, -100, -100]]),
        "loss_masks": torch.tensor([[1.0, 2.0, 0.5, 9.0], [0.25, 0.75, 9.0, 9.0]]),
    }

    health_module._accumulate_batch(stats, batch)

    assert stats["pixmo_caption"] == {
        "examples": 1.0,
        "tokens": 3.0,
        "positive_tokens": 2.0,
        "loss_weight": 3.5,
        "active_loss_weight": 1.5,
    }
    assert stats["pixmo_transcript"] == {
        "examples": 1.0,
        "tokens": 2.0,
        "positive_tokens": 2.0,
        "loss_weight": 1.0,
        "active_loss_weight": 1.0,
    }


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        (
            {
                "router_token_mask": torch.tensor([[True]]),
                "labels": torch.tensor([[1]]),
                "loss_masks": torch.tensor([[1.0]]),
            },
            "omits.*source_names",
        ),
        (
            {
                "source_names": ["unknown"],
                "router_token_mask": torch.tensor([[True]]),
                "labels": torch.tensor([[1]]),
                "loss_masks": torch.tensor([[1.0]]),
            },
            "unknown source",
        ),
        (
            {
                "source_names": [],
                "router_token_mask": torch.tensor([[True]]),
                "labels": torch.tensor([[1]]),
                "loss_masks": torch.tensor([[1.0]]),
            },
            "does not match batch rows",
        ),
    ],
)
def test_unpacked_health_accumulation_rejects_malformed_metadata(
    health_module, batch, match: str
) -> None:
    with pytest.raises(health_module.SSMaxBridgeEvidenceError, match=match):
        health_module._accumulate_batch(health_module._empty_stats(), batch)


def test_health_jsonable_accepts_finite_state_and_rejects_nonfinite(health_module) -> None:
    assert health_module._jsonable({"cursor": (1, torch.tensor(2)), "ratio": 0.5}) == {
        "cursor": [1, 2],
        "ratio": 0.5,
    }
    with pytest.raises(health_module.SSMaxBridgeEvidenceError, match="unsupported value"):
        health_module._jsonable({"ratio": float("nan")})


@pytest.mark.parametrize(("step", "epoch"), [(0, None), (100, 1)])
def test_health_cursor_accepts_only_canonical_epoch_for_step(
    health_module, step: int, epoch: int | None
) -> None:
    state = {
        "global_step": step,
        "world_size": 16,
        "data_loader": {"batches_processed": step, "epoch": epoch},
    }

    saved, validated_epoch = health_module._validate_trainer_cursor(
        state, step=step, world_size=16, rank=3
    )

    assert saved == state["data_loader"]
    assert validated_epoch is epoch


@pytest.mark.parametrize(
    ("step", "epoch"),
    [
        (0, 0),
        (0, False),
        (0, -1),
        (0, 1),
        (100, None),
        (100, 0),
        (100, False),
        (100, -1),
    ],
)
def test_health_cursor_rejects_noncanonical_epoch(
    health_module, step: int, epoch: int | None
) -> None:
    state = {
        "global_step": step,
        "world_size": 16,
        "data_loader": {"batches_processed": step, "epoch": epoch},
    }

    with pytest.raises(health_module.SSMaxBridgeEvidenceError, match="rank3.*invalid epoch"):
        health_module._validate_trainer_cursor(state, step=step, world_size=16, rank=3)


def test_trainer_rank_state_inventory_is_contiguous_and_step_preserving(
    tmp_path: Path, health_module
) -> None:
    checkpoint = tmp_path / "step7"
    train = checkpoint / "train"
    train.mkdir(parents=True)
    for rank in range(2):
        torch.save(
            {
                "global_step": 7,
                "world_size": 2,
                "data_loader": {"batches_processed": 7, "epoch": 1},
            },
            train / f"rank{rank}.pt",
        )

    states, inventory = health_module._trainer_rank_states(checkpoint)
    assert [state["global_step"] for state in states] == [7, 7]
    assert [item["rank"] for item in inventory] == [0, 1]

    (train / "rank0.pt").unlink()
    with pytest.raises(health_module.SSMaxBridgeEvidenceError, match="not contiguous"):
        health_module._trainer_rank_states(checkpoint)
