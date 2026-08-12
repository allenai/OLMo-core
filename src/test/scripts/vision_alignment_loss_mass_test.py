from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def loss_mass_module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "eval" / "vision_alignment_loss_mass.py"
    )
    spec = importlib.util.spec_from_file_location("vision_alignment_loss_mass_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accumulate_batch_matches_source_telemetry(loss_mass_module) -> None:
    stats = loss_mass_module._empty_stats()
    batch = {
        "pack_source_names": [["pixmo_caption", "pixmo_transcript"]],
        "example_ids": torch.tensor([[0, 0, 1, 1, -1]]),
        "router_token_mask": torch.tensor([[True, True, True, True, False]]),
        "labels": torch.tensor([[10, -100, 20, 21, -100]]),
        "loss_masks": torch.tensor([[1.0, 2.0, 0.5, 0.0, 9.0]]),
    }

    loss_mass_module._accumulate_batch(stats, batch)

    assert stats["pixmo_caption"] == {
        "examples": 1.0,
        "tokens": 2.0,
        "positive_tokens": 1.0,
        "loss_weight": 3.0,
        "active_loss_weight": 1.0,
    }
    assert stats["pixmo_transcript"] == {
        "examples": 1.0,
        "tokens": 2.0,
        "positive_tokens": 1.0,
        "loss_weight": 0.5,
        "active_loss_weight": 0.5,
    }


def test_accumulate_batch_rejects_unknown_source(loss_mass_module) -> None:
    stats = loss_mass_module._empty_stats()
    batch = {
        "pack_source_names": [["unknown"]],
        "example_ids": torch.tensor([[0]]),
        "router_token_mask": torch.tensor([[True]]),
        "labels": torch.tensor([[1]]),
        "loss_masks": torch.tensor([[1.0]]),
    }
    with pytest.raises(ValueError, match="unknown source"):
        loss_mass_module._accumulate_batch(stats, batch)


def test_jsonable_normalizes_tuple_and_scalar_tensor(loss_mass_module) -> None:
    assert loss_mass_module._jsonable({"ref": (1, 2, 3), "count": torch.tensor(4)}) == {
        "ref": [1, 2, 3],
        "count": 4,
    }


def test_decode_checkpoint_config_migrates_only_historical_bridge(loss_mass_module) -> None:
    decoded = []
    recipe = SimpleNamespace(
        ExperimentConfig=SimpleNamespace(from_dict=lambda value: decoded.append(value) or value)
    )
    raw = {
        "phase": "bridge",
        "vision_alignment": {"phase": "bridge", "recipe_version": 1},
    }

    result = loss_mass_module._decode_checkpoint_config(recipe, raw)

    assert "perception_trainability_arm" not in raw
    assert result["perception_trainability_arm"] == "treatment"
    assert decoded == [result]

    with pytest.raises(
        loss_mass_module.PromotionValidationError,
        match="Only a historical bridge config",
    ):
        loss_mass_module._decode_checkpoint_config(
            recipe,
            {"phase": "perception", "vision_alignment": {"phase": "perception"}},
        )
