"""Contracts for the first real-data Vision Alignment bridge canary template."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import yaml

from olmo_core.train.callbacks import CheckpointerCallback, EvaluatorCallback

_REPO_ROOT = Path(__file__).parents[3]
_PROFILE_TEMPLATE = (
    _REPO_ROOT
    / "configs"
    / "vision_moe"
    / "vision_alignment"
    / "bridge"
    / "real_canary_v1.yaml.template"
)
_MATERIALIZED_PROFILE = _PROFILE_TEMPLATE.with_suffix("")


def _load_launcher():
    path = _REPO_ROOT / "src" / "scripts" / "train" / "Vision-Alignment.py"
    spec = importlib.util.spec_from_file_location("vision_alignment_canary_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_profile():
    profile = yaml.safe_load(_PROFILE_TEMPLATE.read_text())
    assert isinstance(profile, dict)
    return profile


def _parsed_overrides(profile):
    parsed = {}
    for override in profile["overrides"]:
        key, separator, raw_value = override.partition("=")
        assert separator
        parsed[key.removeprefix("--")] = yaml.safe_load(raw_value)
    return parsed


def test_real_canary_is_a_nonlaunchable_holmes_only_artifact_template():
    profile = _load_profile()
    overrides = _parsed_overrides(profile)

    assert _PROFILE_TEMPLATE.name.endswith(".yaml.template")
    assert profile["name"] == "vision-alignment-bridge-real-canary-v1"
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
    assert "data.allow_unpinned_synthetic_smoke" not in overrides

    artifact_fields = {
        "data.pixmo_cap_path",
        "data.source_audit_path",
        "data.source_audit_fingerprint",
        "data.mixture.mean_loss_weight.pixmo_caption",
        "data.mixture.mean_loss_weight.pixmo_transcript",
        "evaluation.validation_manifest_path",
        "evaluation.validation_manifest_sha256",
    }
    assert artifact_fields <= set(overrides)
    assert all(str(overrides[field]).startswith("__PIN_") for field in artifact_fields)


def test_real_canary_inherits_the_locked_bridge_trainability_and_loss_mass_policy():
    launcher = _load_launcher()
    profile = _load_profile()
    launcher._validate_override_surface(["--phase=bridge", *profile["overrides"]])
    policy = launcher._PHASE_POLICIES[launcher.VisionAlignmentPhase(profile["phase"])]

    assert policy.sequence_length == 2560
    assert policy.connector_lr == 2e-4
    assert policy.connector_warmup == 100
    assert policy.vision_lr == 0.0
    assert policy.lm_lr == 0.0
    assert policy.freeze_params == (
        "vision.*",
        "lm.embedding_norm.*",
        "lm.blocks.*",
        "lm.lm_head.*",
    )
    assert launcher.VisionAlignmentMixtureConfig(phase="bridge").resolved_targets() == {
        "pixmo_caption": 0.70,
        "pixmo_transcript": 0.30,
    }


def test_real_canary_profile_expresses_the_requested_eval_and_checkpoint_cadence(monkeypatch):
    overrides = _parsed_overrides(_load_profile())

    assert overrides["trainer.max_duration.value"] == 250
    assert overrides["evaluation.interval"] == 100
    assert overrides["evaluation.examples_per_source"] == 512
    assert overrides["evaluation.eval_on_startup"] is True
    assert overrides["evaluation.eval_on_finish"] is True
    assert overrides["trainer.callbacks.checkpointer.save_interval"] is None
    assert overrides["trainer.callbacks.checkpointer.ephemeral_save_interval"] is None
    assert overrides["trainer.callbacks.checkpointer.fixed_steps"] == [100, 200]
    assert overrides["trainer.callbacks.checkpointer.max_checkpoints"] == 4

    trainer = SimpleNamespace(global_step=0, block_ephemeral_checkpoints=False)
    checkpointer = CheckpointerCallback(
        save_interval=None,
        ephemeral_save_interval=None,
        fixed_steps=[100, 200],
        max_checkpoints=4,
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
    for step in range(1, 251):
        trainer.global_step = step
        checkpointer.post_train_batch()
    checkpointer.post_train()

    # The launcher enables the permanent step-0 pre-train checkpoint. Fixed steps add 100/200,
    # and CheckpointerCallback.post_train() adds the final permanent step 250.
    assert checkpointer.pre_train_checkpoint is True
    assert saved_steps == [(100, False), (200, False), (250, False)]

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
    for step in range(1, 251):
        trainer.global_step = step
        evaluation.post_step()
    evaluation.post_train()

    assert evaluated_steps == [0, 100, 200, 250]


def test_real_canary_template_does_not_claim_unreviewed_artifact_values():
    raw = _PROFILE_TEMPLATE.read_text()

    assert "__PIN_FILTERED_PIXMO_CAP_ARROW_ROOT__" in raw
    assert "__PIN_BRIDGE_SOURCE_AUDIT_JSON__" in raw
    assert "__PIN_64_HEX_SOURCE_AUDIT_FINGERPRINT__" in raw
    assert "__PIN_VALIDATION_MANIFEST_V3_JSON__" in raw
    assert "__PIN_64_HEX_VALIDATION_MANIFEST_SHA256__" in raw
    assert "/path/to/" not in raw
    assert "synthetic" not in raw.lower()


def test_materialized_real_canary_is_fully_pinned_and_holmes_only():
    raw = _MATERIALIZED_PROFILE.read_text()
    profile = yaml.safe_load(raw)
    assert isinstance(profile, dict)
    overrides = _parsed_overrides(profile)

    assert "__PIN_" not in raw
    assert profile["name"] == "vision-alignment-bridge-real-canary-v1"
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
    assert float(overrides["data.mixture.mean_loss_weight.pixmo_caption"]) > 0
    assert float(overrides["data.mixture.mean_loss_weight.pixmo_transcript"]) > 0
