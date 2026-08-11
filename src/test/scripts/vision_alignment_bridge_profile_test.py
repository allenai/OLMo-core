"""Contracts for the low-LR Vision Alignment bridge refinement profile."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import yaml

from olmo_core.train.callbacks import CheckpointerCallback, EvaluatorCallback

_REPO_ROOT = Path(__file__).parents[3]
_PROFILE = (
    _REPO_ROOT / "configs" / "vision_moe" / "vision_alignment" / "bridge" / "real_bridge_v1.yaml"
)


def _load_profile():
    profile = yaml.safe_load(_PROFILE.read_text())
    assert isinstance(profile, dict)
    return profile


def _parsed_overrides(profile):
    parsed = {}
    for override in profile["overrides"]:
        key, separator, raw_value = override.partition("=")
        assert separator
        parsed[key.removeprefix("--")] = yaml.safe_load(raw_value)
    return parsed


def test_real_bridge_profile_is_fresh_holmes_only_and_fully_pinned():
    raw = _PROFILE.read_text()
    profile = _load_profile()
    overrides = _parsed_overrides(profile)

    assert "__PIN_" not in raw
    assert profile["name"] == "vision-alignment-bridge-real-v1"
    assert profile["phase"] == "bridge"
    assert profile["launch"] == {
        "cluster": "ai2/holmes",
        "workspace": "ai2/molmofication",
        "budget": "ai2/oe-other",
        "num_nodes": 2,
        "num_gpus": 8,
        "priority": "urgent",
        "min_runtime": "8h",
    }
    assert "hostnames" not in profile["launch"]
    assert str(overrides["data.pixmo_cap_path"]).endswith("/pixmo-cap-content-disjoint-v1/dataset")
    assert str(overrides["data.source_audit_path"]).endswith("/bridge-source-audit-v1.json")
    assert len(str(overrides["data.source_audit_fingerprint"])) == 64
    assert len(str(overrides["evaluation.validation_manifest_sha256"])) == 64


def test_real_bridge_profile_preserves_early_states_and_production_horizon(monkeypatch):
    overrides = _parsed_overrides(_load_profile())

    assert overrides["trainer.max_duration.value"] == 500
    assert overrides["evaluation.interval"] == 100
    assert overrides["evaluation.examples_per_source"] == 512
    assert overrides["evaluation.eval_on_startup"] is True
    assert overrides["evaluation.eval_on_finish"] is True
    assert overrides["trainer.callbacks.checkpointer.save_interval"] is None
    assert overrides["trainer.callbacks.checkpointer.ephemeral_save_interval"] == 50
    assert overrides["trainer.callbacks.checkpointer.fixed_steps"] == [
        100,
        200,
        250,
        300,
        400,
        500,
    ]
    assert overrides["trainer.callbacks.checkpointer.max_checkpoints"] == 7

    trainer = SimpleNamespace(global_step=0, block_ephemeral_checkpoints=False)
    checkpointer = CheckpointerCallback(
        save_interval=None,
        ephemeral_save_interval=50,
        fixed_steps=[100, 200, 250, 300, 400, 500],
        max_checkpoints=7,
        pre_train_checkpoint=True,
        save_async=False,
    )
    cast(Any, checkpointer).trainer = trainer
    checkpointer._latest_checkpoint_step = 0
    saved_steps = []

    def record_checkpoint(*, save_async=None, ephemeral=False):
        del save_async
        saved_steps.append((trainer.global_step, ephemeral))
        checkpointer._latest_checkpoint_step = trainer.global_step
        return f"step{trainer.global_step}"

    monkeypatch.setattr(checkpointer, "_save_checkpoint", record_checkpoint)
    monkeypatch.setattr(checkpointer, "_await_last_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpointer, "_remove_old_checkpoints", lambda: None)
    monkeypatch.setattr(checkpointer, "_trim_checkpoints", lambda: None)
    for step in range(1, 501):
        trainer.global_step = step
        checkpointer.post_train_batch()
    checkpointer.post_train()

    assert checkpointer.pre_train_checkpoint is True
    assert [step for step in saved_steps if not step[1]] == [
        (100, False),
        (200, False),
        (250, False),
        (300, False),
        (400, False),
        (500, False),
    ]
    assert [step for step in saved_steps if step[1]] == [
        (50, True),
        (150, True),
        (350, True),
        (450, True),
    ]

    evaluation = EvaluatorCallback(
        evaluators=[],
        eval_interval=overrides["evaluation.interval"],
        eval_on_startup=overrides["evaluation.eval_on_startup"],
        eval_on_finish=overrides["evaluation.eval_on_finish"],
    )
    cast(Any, evaluation).trainer = trainer
    evaluated_steps = []
    monkeypatch.setattr(
        evaluation,
        "perform_eval",
        lambda *args, **kwargs: evaluated_steps.append(trainer.global_step),
    )
    trainer.global_step = 0
    evaluation.pre_train()
    for step in range(1, 501):
        trainer.global_step = step
        evaluation.post_step()
    evaluation.post_train()

    # The generic callback intentionally runs the required finish evaluation even when the last
    # interval lands on the terminal step.
    assert evaluated_steps == [0, 100, 200, 300, 400, 500, 500]
