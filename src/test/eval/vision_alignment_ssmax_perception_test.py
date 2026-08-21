from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_core.eval import vision_alignment_ssmax_perception as perception
from olmo_core.train.callbacks import SSMaxHealthLedgerCallback

_RECEIPT_TIME = "2026-08-20T00:00:00+00:00"
_REPORT_TIME = "2026-08-20T01:00:00+00:00"
_HASH = "a" * 64
_TRAINER_STATE_SIZE = 123


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return perception.artifact_reference(path)


def _trainer_state_sha(arm: str) -> str:
    return ("b" if arm == perception.CONTROL_ARM else "c") * 64


def _trainer_inventory_sha(arm: str) -> str:
    return perception.canonical_sha256(
        [
            {
                "rank": 0,
                "path": "train/rank0.pt",
                "size": _TRAINER_STATE_SIZE,
                "sha256": _trainer_state_sha(arm),
            }
        ]
    )


def _checkpoint(tmp_path: Path, arm: str, step: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str((tmp_path / arm / f"step{step}").resolve()),
        "global_step": step,
        "config_sha256": "1" * 64,
        "marker_sha256": "2" * 64,
        "dcp_metadata_sha256": "3" * 64,
        "state_file_count": 2,
        "state_file_inventory_sha256": "4" * 64,
        "trainer_state_count": 1,
        "trainer_state_inventory_sha256": _trainer_inventory_sha(arm),
    }
    payload["identity_sha256"] = perception.canonical_sha256(payload)
    return payload


def _pair_config(
    *,
    arm: str,
    run_name: str,
    profile: Path,
    profile_sha: str,
    output: Path,
) -> dict[str, Any]:
    control = arm == perception.CONTROL_ARM
    return {
        "model_variant": "ssmax_head_qknorm",
        "phase": "perception",
        "perception_trainability_arm": arm,
        "required_run_name": run_name,
        "reviewed_profile_path": str(profile),
        "reviewed_profile_sha256": profile_sha,
        "expected_launch_command": f"launch {run_name}",
        "artifacts": {"tokenizer_id": "tok", "tokenizer_revision": "rev"},
        "vision_alignment": {
            "model_variant": "ssmax_head_qknorm",
            "phase": "perception",
            "lineage_id": run_name,
            "data_contract_sha256": "6" * 64,
            "trainable_contract_sha256": ("7" if control else "8") * 64,
        },
        "initialization": {
            "expected_parent_phase": "bridge",
            "checkpoint": "/bridge/step500",
            "parent_gate_sha256": "9" * 64,
        },
        "data": {"pack_sequences": False, "allow_unpinned_synthetic_smoke": False},
        "launch": {
            "name": f"{run_name}-1234abcd",
            "cmd": [
                "src/scripts/train/Vision-Alignment.py",
                "train",
                run_name,
                f"--profile={profile}",
            ],
            "description": f"Reviewed causal arm {arm}",
            "num_nodes": 2,
            "num_gpus": 8,
            "workspace": "ai2/scaling-ladders",
            "clusters": ["ai2/holmes"],
            "budget": "ai2/oe-other",
            "priority": "urgent",
            "min_runtime": "8h",
            "beaker_image": "ai2/olmo-core:pinned",
            "preemptible": True,
            "env_vars": [{"name": "PYTHONPATH", "value": "/gantry-runtime/src"}],
            "git": {
                "repo": "allenai/OLMo-core",
                "repo_url": "https://github.com/allenai/OLMo-core",
                "ref": "a" * 40,
            },
        },
        "train_module": {
            "freeze_params": ["lm.*"] + (["vision.*"] if control else []),
            "optim": {
                "group_overrides": [
                    {"params": ["*vision.*"], "opts": {"lr": 0.0 if control else 2e-6}}
                ]
            },
        },
        "trainer": {
            "save_folder": str(output),
            "max_duration": {"value": 4000, "unit": "steps"},
            "callbacks": {
                "checkpointer": {
                    "pre_train_checkpoint": True,
                    "fixed_steps": [500, 1000, 2000, 3000, 4000],
                },
                "ssmax_health_ledger": {
                    "model_variant": "ssmax_head_qknorm",
                    "phase": "perception",
                    "run_name": run_name,
                    "enabled": True,
                },
                "wandb": {"name": run_name},
            },
        },
    }


def test_saved_pair_allows_only_derived_vision_intervention(tmp_path: Path) -> None:
    recipe = tmp_path / "repo" / "src" / "scripts" / "train" / "Vision-Alignment.py"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("# recipe\n")
    profiles = {}
    configs = {}
    arms = {}
    for arm in perception.ARMS:
        profile = tmp_path / "repo" / "configs" / f"{arm}.yaml"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text(f"# {arm}\n")
        reference = perception.artifact_reference(profile)
        run_name = f"pair-{arm}"
        profiles[arm] = reference
        arms[arm] = {
            "run_name": run_name,
            "checkpoint_root": str(tmp_path / arm),
            "training_profile": str(profile),
        }
        configs[arm] = _pair_config(
            arm=arm,
            run_name=run_name,
            profile=profile,
            profile_sha=reference["sha256"],
            output=tmp_path / arm,
        )
    spec = {
        "model_variant": "ssmax_head_qknorm",
        "arms": arms,
        "topology": {
            "num_nodes": 2,
            "gpus_per_node": 8,
            "world_size": 16,
            "data_parallel": "hsdp",
        },
    }
    summaries = perception.validate_saved_config_pair(
        configs, spec=spec, profile_references=profiles, recipe_path=recipe
    )
    assert summaries[perception.TREATMENT_ARM]["vision_lr"] == 2e-6
    assert "vision.*" in summaries[perception.CONTROL_ARM]["freeze_params"]

    changed = copy.deepcopy(configs)
    changed[perception.TREATMENT_ARM]["data"]["seed"] = 17
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="differ outside"):
        perception.validate_saved_config_pair(
            changed, spec=spec, profile_references=profiles, recipe_path=recipe
        )

    changed = copy.deepcopy(configs)
    changed[perception.TREATMENT_ARM]["launch"]["priority"] = "normal"
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="launch priority differs"):
        perception.validate_saved_config_pair(
            changed, spec=spec, profile_references=profiles, recipe_path=recipe
        )

    changed = copy.deepcopy(configs)
    changed[perception.TREATMENT_ARM]["launch"]["beaker_image"] = "ai2/olmo-core:drifted"
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="differ outside"):
        perception.validate_saved_config_pair(
            changed, spec=spec, profile_references=profiles, recipe_path=recipe
        )


def _surface(sha: str, *, changed: bool = False) -> dict[str, Any]:
    return {
        "protocol": "logical-tensor-comparison-sha256-v1",
        "tensor_count": 1,
        "reference_inventory_sha256": sha,
        "candidate_inventory_sha256": "f" * 64 if changed else sha,
        "mismatch_count": int(changed),
    }


def _rows(*, correct: float, gap: float) -> list[dict[str, Any]]:
    output = []
    for index in range(2):
        correct_windows = {window: correct + index * 0.001 for window in perception.WINDOWS}
        gap_windows = {window: gap for window in perception.WINDOWS}
        output.append(
            {
                "pairing_position": index,
                "recipient_index": index,
                "donor_index": 1 - index,
                "response_tokens": 40,
                "correct_ce": correct_windows,
                "wrong_ce": {
                    window: correct_windows[window] + gap for window in perception.WINDOWS
                },
                "ce_gap_wrong_minus_correct": gap_windows,
            }
        )
    return output


def _manifest_fixture(
    tmp_path: Path, *, version: int = perception.SCHEMA_VERSION
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator\n")
    health_producer = tmp_path / "health-producer.py"
    health_producer.write_text("# health producer\n")
    recipe = tmp_path / "Vision-Alignment.py"
    recipe.write_text("# recipe\n")
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("{}\n")
    attention_probe = tmp_path / "attention-probe.json"
    attention_probe.write_text("{}\n")
    pairings = {}
    for source in perception.SOURCES:
        pairings[source] = _write_json(
            tmp_path / "pairings" / f"{source}.json",
            {"pairs": [{"recipient": 0, "donor": 1}, {"recipient": 1, "donor": 0}]},
        )
    manifest: dict[str, Any] = {
        "format": perception.MANIFEST_FORMAT,
        "version": version,
        "pair_id": "ssmax-perception-test-pair",
        "model_variant": "ssmax_head_qknorm",
        "git": {
            "repo": "allenai/OLMo-core",
            "repo_url": "https://github.com/allenai/OLMo-core",
            "ref": "b" * 40,
        },
        "producers": {
            perception.EVALUATION_PRODUCER: {
                "repo_relative_path": perception.PRODUCER_RELATIVE_PATHS[
                    perception.EVALUATION_PRODUCER
                ],
                "sha256": perception.sha256_file(evaluator),
                "git_ref": "b" * 40,
            },
            perception.HEALTH_PRODUCER: {
                "repo_relative_path": perception.PRODUCER_RELATIVE_PATHS[
                    perception.HEALTH_PRODUCER
                ],
                "sha256": perception.sha256_file(health_producer),
                "git_ref": "b" * 40,
            },
        },
        "recipe": perception.artifact_reference(recipe),
        "text_sentinel": perception.artifact_reference(sentinel),
        "attention_probe": perception.artifact_reference(attention_probe),
        "pairings": pairings,
        "evaluation": {"examples_per_source": 2, "bootstrap_seed": 7, "bootstrap_samples": 128},
        "topology": {"world_size": 1},
        "policy": perception._locked_promotion_policy(version),
        "loss_mass_targets": {source: 1 / len(perception.SOURCES) for source in perception.SOURCES},
        "arms": {
            arm: {
                "run_name": f"run-{arm}",
                "data_contract_sha256": "c" * 64,
                "trainable_contract_sha256": ("d" if arm == perception.TREATMENT_ARM else "e") * 64,
                "checkpoints": {
                    str(step): _checkpoint(tmp_path, arm, step)
                    for step in perception.REQUIRED_STEPS
                },
            }
            for arm in perception.ARMS
        },
        "content_sha256": _HASH,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest_path, manifest, dict(manifest["producers"][perception.EVALUATION_PRODUCER])


def _text_result(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_sha256": manifest["text_sentinel"]["sha256"],
        "input_sha256": "1" * 64,
        "labels_sha256": "2" * 64,
        "token_count": 3,
        "logits_sha256": "3" * 64,
        "ce_sha256": "4" * 64,
    }


def _evaluation_receipt(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    evaluator: Mapping[str, str],
    *,
    arm: str,
    step: int,
) -> dict[str, Any]:
    control = arm == perception.CONTROL_ARM
    gap = {
        (perception.CONTROL_ARM, 0): 0.1,
        (perception.CONTROL_ARM, 3000): 0.15,
        (perception.CONTROL_ARM, 4000): 0.2,
        (perception.TREATMENT_ARM, 0): 0.1,
        (perception.TREATMENT_ARM, 3000): 1.0,
        (perception.TREATMENT_ARM, 4000): 0.9,
    }[(arm, step)]
    correct = 2.02 if arm == perception.TREATMENT_ARM and step == 4000 else 2.0
    full_sha = "5" * 64 if step == 0 else f"{step % 10}" * 64
    state = {
        "full_model": {
            "protocol": "logical-model-tensor-inventory-sha256-v1",
            "tensor_count": 3,
            "inventory_sha256": full_sha,
        },
        "frozen_lm": _surface("6" * 64),
        "non_image_embedding_rows": _surface("7" * 64),
        "vision": _surface(
            "8" * 64,
            changed=not control and step > 0,
        ),
    }
    payload: dict[str, Any] = {
        "format": perception.EVALUATION_RECEIPT_FORMAT,
        "version": manifest["version"],
        "status": "passed",
        "created_at": _RECEIPT_TIME,
        "manifest": perception.manifest_reference(manifest_path, manifest),
        "pair_id": manifest["pair_id"],
        "model_variant": manifest["model_variant"],
        "arm": arm,
        "step": step,
        "checkpoint": manifest["arms"][arm]["checkpoints"][str(step)],
        "strict_generic_dcp_load": {
            "complete": True,
            "strict": True,
            "load_completed": True,
            "checkpoint_key_count": 4,
            "model_tensor_count": 3,
            "model_parameter_tensor_count": 2,
            "model_buffer_tensor_count": 1,
            "model_keyset_sha256": "9" * 64,
            "model_inventory_sha256": "a" * 64,
        },
        "state": state,
        "text_sentinel": _text_result(manifest),
        "attention_diagnostics": {
            "format": "ssmax_attention_diagnostics",
            "version": 1,
            "checkpoint": manifest["arms"][arm]["checkpoints"][str(step)],
            "protocol": {"manifest_sha256": manifest["attention_probe"]["sha256"]},
            "report_sha256": "f" * 64,
        },
        "pairings": manifest["pairings"],
        "results": {
            source: {
                "pairing_sha256": manifest["pairings"][source]["sha256"],
                "examples": 2,
                "per_example": _rows(correct=correct, gap=gap),
            }
            for source in perception.SOURCES
        },
        "evaluator": dict(evaluator),
    }
    load = payload["strict_generic_dcp_load"]
    load["sha256"] = perception.canonical_sha256(load)
    payload["content_sha256"] = perception.canonical_sha256(payload)
    return payload


def _health_receipt(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    evidence: Mapping[str, str],
    *,
    arm: str,
    step: int,
    skip_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    callback = SSMaxHealthLedgerCallback(
        model_variant=manifest["model_variant"],
        phase="perception",
        run_name=manifest["arms"][arm]["run_name"],
    )
    trainer = SimpleNamespace(
        global_step=0,
        data_loader=SimpleNamespace(state_dict=lambda: {"total_data_errors": 0}),
        train_module=SimpleNamespace(optim=SimpleNamespace(rolling_interval_length=128)),
    )
    callback.trainer = trainer
    callback.post_checkpoint_loaded("bridge-parent")
    for global_step in range(1, step + 1):
        trainer.global_step = global_step
        skipped = global_step in skip_steps
        guard_active = global_step >= 64
        callback.log_metrics(
            global_step,
            {
                "train/CE loss": 2.0,
                "optim/total grad norm": 2.0 if skipped else 1.0,
                "optim/step skipped": float(skipped),
                "optim/guard active": float(guard_active),
                "optim/guard loss within": 1.0,
                "optim/guard gradient within": float(not skipped),
            },
        )
    ledger = callback.state_dict()
    guard_ok = True
    if manifest["version"] == perception.PERCEPTION_V2_SCHEMA_VERSION:
        guard_ok = perception.summarize_optimizer_guard_trajectory(
            ledger, policy=manifest["policy"], step=step
        )["passed"]
    weight = 0.0 if step == 0 else 1.0
    payload: dict[str, Any] = {
        "format": perception.HEALTH_RECEIPT_FORMAT,
        "version": manifest["version"],
        "status": "passed" if guard_ok else "failed",
        "created_at": _RECEIPT_TIME,
        "manifest": perception.manifest_reference(manifest_path, manifest),
        "pair_id": manifest["pair_id"],
        "model_variant": manifest["model_variant"],
        "arm": arm,
        "step": step,
        "checkpoint": manifest["arms"][arm]["checkpoints"][str(step)],
        "rank_states": [
            {
                "rank": 0,
                "global_step": step,
                "batches_processed": step,
                "data_loader_state_sha256": f"{step % 10}" * 64,
                "trainer_state_sha256": _trainer_state_sha(arm),
                "trainer_state_size_bytes": _TRAINER_STATE_SIZE,
                "health_ledger": ledger,
            }
        ],
        "sources": {
            source: {
                "examples": step,
                "tokens": step,
                "positive_tokens": step,
                "loss_weight": weight,
                "active_loss_weight": weight,
                "target_loss_mass": manifest["loss_mass_targets"][source],
            }
            for source in perception.SOURCES
        },
        "run_counters": {
            "data_errors": 0,
            "optimizer_guard_skips": len([item for item in skip_steps if item <= step]),
            "nonfinite_losses": 0,
            "nonfinite_gradients": 0,
        },
        "evidence": {
            "recipe": dict(manifest["recipe"]),
            "producer": dict(manifest["producers"][perception.HEALTH_PRODUCER]),
        },
    }
    payload["content_sha256"] = perception.canonical_sha256(payload)
    return payload


def _guard_ledger(
    skip_observations: Mapping[int, tuple[float, float]],
    *,
    step: int = 4000,
    resume_at: int | None = None,
    live_decisions: Mapping[int, tuple[bool, bool, bool]] | None = None,
) -> Mapping[str, Any]:
    callback = SSMaxHealthLedgerCallback(
        model_variant="ssmax_head_qknorm",
        phase="perception",
        run_name="v2-guard-boundary-test",
    )
    trainer = SimpleNamespace(
        global_step=0,
        data_loader=SimpleNamespace(state_dict=lambda: {"total_data_errors": 0}),
        train_module=SimpleNamespace(optim=SimpleNamespace(rolling_interval_length=128)),
    )
    callback.trainer = trainer
    callback.post_checkpoint_loaded("bridge-parent")
    history_reset_step = 0
    for global_step in range(1, step + 1):
        if resume_at is not None and global_step == resume_at + 1:
            resumed = SSMaxHealthLedgerCallback(
                model_variant="ssmax_head_qknorm",
                phase="perception",
                run_name="v2-guard-boundary-test",
            )
            resumed.trainer = trainer
            resumed.load_state_dict(callback.state_dict())
            resumed.post_checkpoint_loaded(f"step{resume_at}")
            callback = resumed
            history_reset_step = resume_at
        trainer.global_step = global_step
        loss, grad_norm = skip_observations.get(global_step, (2.0, 1.0))
        guard_active = global_step - history_reset_step >= 64
        loss_within_guard = loss <= 2.0
        gradient_within_guard = grad_norm <= 1.0
        if live_decisions is not None and global_step in live_decisions:
            guard_active, loss_within_guard, gradient_within_guard = live_decisions[global_step]
        callback.log_metrics(
            global_step,
            {
                "train/CE loss": loss,
                "optim/total grad norm": grad_norm,
                "optim/step skipped": float(global_step in skip_observations),
                "optim/guard active": float(guard_active),
                "optim/guard loss within": float(loss_within_guard),
                "optim/guard gradient within": float(gradient_within_guard),
            },
        )
    return callback.state_dict()


def _legacy_v2_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    legacy = copy.deepcopy(dict(ledger))
    legacy["version"] = 2
    legacy.pop("optimizer_guard_history_reset_steps")
    legacy.pop("optimizer_guard_rolling_interval_length")
    legacy["metrics"] = {
        "loss": "train/CE loss",
        "grad_norm": "optim/total grad norm",
        "optimizer_guard_skip": "optim/step skipped",
    }
    previous_sha = "0" * 64
    for event in legacy["events"]:
        event.pop("optimizer_guard_active")
        event.pop("optimizer_guard_loss_within")
        event.pop("optimizer_guard_gradient_within")
        event["previous_event_sha256"] = previous_sha
        event["event_sha256"] = perception.canonical_sha256(
            {name: value for name, value in event.items() if name != "event_sha256"}
        )
        previous_sha = event["event_sha256"]
    legacy["event_chain_sha256"] = previous_sha
    legacy["content_sha256"] = perception.canonical_sha256(
        {name: value for name, value in legacy.items() if name != "content_sha256"}
    )
    return legacy


def test_v2_manifest_template_locks_prospective_guard_policy(tmp_path: Path) -> None:
    template = (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_alignment/eval/ssmax_perception_pair_manifest_v2.json.template"
    )
    spec = json.loads(template.read_text())
    spec["model_variant"] = "ssmax_head_qknorm"
    spec["pair_id"] = "ssmax-head-qknorm-perception-v2"
    for arm in perception.ARMS:
        identity = perception.V2_ARM_IDENTITIES[spec["model_variant"]][arm]
        spec["arms"][arm] = {
            "run_name": identity["run_name"],
            "checkpoint_root": str(tmp_path / identity["run_name"]),
            "training_profile": str(tmp_path / identity["training_profile_name"]),
        }
    perception._validate_spec_common(spec)

    drifted = copy.deepcopy(spec)
    drifted["policy"]["maximum_optimizer_guard_skips"] = 9
    drifted_path = tmp_path / "drifted-v2-spec.json"
    drifted_path.write_text(json.dumps(drifted))
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="differs from the locked policy",
    ):
        perception.load_manifest_spec(drifted_path)

    for field, alias, message in (
        ("maximum_optimizer_guard_skips", 8.0, "must be an integer"),
        ("require_finite_gradient_only_skips", 1, "must be boolean"),
        ("require_uninterrupted_optimizer_guard_history", 1, "must be boolean"),
    ):
        drifted = copy.deepcopy(spec)
        drifted["policy"][field] = alias
        with pytest.raises(perception.SSMaxPerceptionEvidenceError, match=message):
            perception._validate_spec_common(drifted)

    drifted = copy.deepcopy(spec)
    drifted["policy"]["optimizer_guard"]["max_grad_norm"] = 1
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="must be a JSON float"):
        perception._validate_spec_common(drifted)

    drifted = copy.deepcopy(spec)
    drifted["arms"][perception.TREATMENT_ARM][
        "run_name"
    ] = "vision-ssmax-head-qknorm-1p4b-cx8-perception-treatment-v1"
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="fresh rerun"):
        perception._validate_spec_common(drifted)


def test_v2_optimizer_guard_boundaries_are_fail_closed() -> None:
    policy = perception._locked_promotion_policy(perception.PERCEPTION_V2_SCHEMA_VERSION)
    eight_steps = tuple(64 + 129 * index for index in range(8))
    eight = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({step: (2.0, 2.0) for step in eight_steps}),
        policy=policy,
        step=4000,
    )
    assert eight["passed"] is True
    assert eight["count"] == 8
    assert eight["minimum_step_distance"] == 129
    assert eight["required_minimum_step_distance"] == 129
    assert all(event["finite_gradient_only"] for event in eight["skip_events"])

    nine_steps = tuple(64 + 129 * index for index in range(9))
    nine = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({step: (2.0, 2.0) for step in nine_steps}),
        policy=policy,
        step=4000,
    )
    assert nine["maximum_passed"] is False
    assert nine["passed"] is False

    exact_spacing = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({64: (2.0, 2.0), 193: (2.0, 2.0)}),
        policy=policy,
        step=4000,
    )
    assert exact_spacing["spacing_passed"] is True
    too_close = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({64: (2.0, 2.0), 192: (2.0, 100.0)}),
        policy=policy,
        step=4000,
    )
    assert too_close["finite_gradient_only_passed"] is True
    assert too_close["minimum_step_distance"] == 128
    assert too_close["spacing_passed"] is False

    exact_final = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({3872: (2.0, 2.0)}), policy=policy, step=4000
    )
    assert exact_final["clean_final_steps"] == 128
    assert exact_final["final_window_passed"] is True
    too_late = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({3873: (2.0, 2.0)}), policy=policy, step=4000
    )
    assert too_late["clean_final_steps"] == 127
    assert too_late["final_window_passed"] is False

    loss_triggered = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({217: (3.0, 2.0)}), policy=policy, step=4000
    )
    assert loss_triggered["skip_events"][0]["loss_within_guard"] is False
    assert loss_triggered["skip_events"][0]["gradient_exceeds_guard"] is True
    assert loss_triggered["finite_gradient_only_passed"] is False


def test_v2_optimizer_guard_resume_is_truthfully_classified_but_ineligible() -> None:
    policy = perception._locked_promotion_policy(perception.PERCEPTION_V2_SCHEMA_VERSION)
    first_active = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger({564: (2.0, 2.0)}, resume_at=500),
        policy=policy,
        step=4000,
    )
    active_skip = first_active["skip_events"][0]
    assert active_skip["history_reset_step"] == 500
    assert active_skip["guard_active"] is True
    assert active_skip["loss_within_guard"] is True
    assert active_skip["gradient_exceeds_guard"] is True
    assert active_skip["finite_gradient_only"] is True
    assert first_active["optimizer_guard_history_reset_steps"] == [0, 500]
    assert first_active["resume_free_passed"] is False
    assert first_active["passed"] is False


def test_v2_optimizer_guard_uses_live_device_reason_not_cpu_replay() -> None:
    policy = perception._locked_promotion_policy(perception.PERCEPTION_V2_SCHEMA_VERSION)
    loss_only = perception.summarize_optimizer_guard_trajectory(
        _guard_ledger(
            {217: (2.0, 2.0)},
            live_decisions={217: (True, False, True)},
        ),
        policy=policy,
        step=4000,
    )
    event = loss_only["skip_events"][0]
    assert event["cpu_reconstruction"]["loss_within_guard"] is True
    assert event["cpu_reconstruction"]["gradient_exceeds_guard"] is True
    assert event["live_decision_matches_cpu_reconstruction"] is False
    assert event["loss_within_guard"] is False
    assert event["gradient_exceeds_guard"] is False
    assert event["finite_gradient_only"] is False
    assert loss_only["passed"] is False


def test_cross_model_comparison_separates_absolute_and_step0_adaptation(monkeypatch) -> None:
    shared_protocol = {
        "git": {"ref": "a" * 40},
        "producers": {"evaluation": "shared", "health": "shared"},
        "recipe": {"path": "/recipe", "sha256": "a" * 64},
        "perception_provenance": {
            "path": "/provenance",
            "sha256": "b" * 64,
            "content_sha256": "c" * 64,
        },
        "source_audit": {"path": "/audit", "sha256": "d" * 64},
        "source_audit_fingerprint": "e" * 64,
        "single_response_projection": {"contract": "shared"},
        "attention_probe": {"path": "/probe", "sha256": "f" * 64},
        "text_sentinel": {"path": "/sentinel", "sha256": "1" * 64},
        "pairings": {source: source for source in perception.SOURCES},
        "evaluation": {"bootstrap_samples": 32, "bootstrap_seed": 17},
        "topology": {"world_size": 16},
        "policy": {"descriptive": True},
        "loss_mass_targets": {source: 1 / len(perception.SOURCES) for source in perception.SOURCES},
    }

    def manifest(variant: str) -> dict[str, Any]:
        return {"model_variant": variant, "pair_id": f"pair-{variant}", **shared_protocol}

    left_manifest = manifest("ssmax_head_qknorm")
    right_manifest = manifest("ssmax_no_qknorm")

    def report(label: str) -> dict[str, Any]:
        return {
            "content_sha256": ("2" if label == "left" else "3") * 64,
            "summary": {
                "attention_trajectory": {
                    arm: {
                        str(step): {
                            "comparison_from_step0": (
                                None if step == 0 else {"label": label, "step": step}
                            )
                        }
                        for step in perception.REQUIRED_STEPS
                    }
                    for arm in perception.ARMS
                }
            },
        }

    left_rebuilt = {"manifest": left_manifest, "report": report("left")}
    right_rebuilt = {"manifest": right_manifest, "report": report("right")}

    def rows(side: str) -> dict[str, dict[int, dict[str, list[dict[str, Any]]]]]:
        gaps = {
            "left": {
                perception.CONTROL_ARM: {0: 0.2, 3000: 0.3, 4000: 0.4},
                perception.TREATMENT_ARM: {0: 0.2, 3000: 0.8, 4000: 1.2},
            },
            "right": {
                perception.CONTROL_ARM: {0: 0.5, 3000: 0.6, 4000: 0.7},
                perception.TREATMENT_ARM: {0: 0.5, 3000: 0.7, 4000: 0.8},
            },
        }
        return {
            arm: {
                step: {
                    source: _rows(correct=2.0, gap=gaps[side][arm][step])
                    for source in perception.SOURCES
                }
                for step in perception.REQUIRED_STEPS
            }
            for arm in perception.ARMS
        }

    def payloads(side: str) -> dict[str, dict[int, dict[str, Any]]]:
        return {
            arm: {
                step: {
                    "attention_diagnostics": {
                        "report_sha256": ("4" if side == "left" else "5") * 64,
                        "identity": f"{side}-{arm}-{step}",
                    }
                }
                for step in perception.REQUIRED_STEPS
            }
            for arm in perception.ARMS
        }

    monkeypatch.setattr(
        perception,
        "_rebuilt_promotion_for_model_comparison",
        lambda reference, **_kwargs: (
            left_rebuilt if reference["path"] == "/left" else right_rebuilt
        ),
    )
    monkeypatch.setattr(
        perception,
        "_model_comparison_evaluations",
        lambda rebuilt: (
            (left_manifest, payloads("left"), rows("left"))
            if rebuilt is left_rebuilt
            else (right_manifest, payloads("right"), rows("right"))
        ),
    )
    monkeypatch.setattr(
        perception,
        "compare_ssmax_attention_reports",
        lambda left, right: {"left": left["identity"], "right": right["identity"]},
    )

    result = perception.build_model_variant_comparison(
        left_promotion_report={"path": "/left", "sha256": "6" * 64},
        right_promotion_report={"path": "/right", "sha256": "7" * 64},
        created_at=_REPORT_TIME,
        verify_live_checkpoint=False,
    )
    assert result["winner"] is None
    adaptation = result["absolute_and_adaptation_trajectories"][perception.TREATMENT_ARM]["4000"][
        "sources"
    ][perception.SOURCES[0]]["all"]["step0_normalized_adaptation_left_minus_right"]["gap"]
    assert adaptation["mean"] == pytest.approx(0.7)
    assert adaptation["direction"] == "left_gained_more_visual_gap"
    causal = result["causal_adaptation_contrast"]["4000"]["sources"][perception.SOURCES[0]]["all"][
        "gap"
    ]
    assert causal["mean"] == pytest.approx(0.7)
    assert causal["direction"] == "left_larger_treatment_over_control_adaptation"
    assert result["attention_comparisons"][perception.CONTROL_ARM]["0"]["absolute_left_vs_right"][
        "left"
    ].startswith("left-")
    perception.validate_model_variant_comparison(result)

    drifted = copy.deepcopy(result)
    drifted["winner"] = "left"
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="identity differs"):
        perception.validate_model_variant_comparison(drifted)

    drifted = copy.deepcopy(result)
    drifted["causal_adaptation_contrast"]["4000"]["sources"][perception.SOURCES[0]]["all"]["gap"][
        "mean"
    ] = 999.0
    drifted["content_sha256"] = perception.canonical_sha256(
        {key: item for key, item in drifted.items() if key != "content_sha256"}
    )
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="rebuilt raw evidence"):
        perception.validate_model_variant_comparison(drifted, verify_live_checkpoint=False)


def _report_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: int = perception.SCHEMA_VERSION,
    treatment_skip_steps: tuple[int, ...] = (),
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_path, manifest, evaluator = _manifest_fixture(tmp_path, version=version)
    evidence_file = tmp_path / "health-producer.py"
    evidence_file.write_text("# health producer\n")
    evidence = perception.artifact_reference(evidence_file)
    evaluation_refs = {arm: {} for arm in perception.ARMS}
    health_refs = {arm: {} for arm in perception.ARMS}
    for arm in perception.ARMS:
        for step in perception.REQUIRED_STEPS:
            evaluation_refs[arm][step] = _write_json(
                tmp_path / "receipts" / f"{arm}-{step}-evaluation.json",
                _evaluation_receipt(manifest_path, manifest, evaluator, arm=arm, step=step),
            )
            health_refs[arm][step] = _write_json(
                tmp_path / "receipts" / f"{arm}-{step}-health.json",
                _health_receipt(
                    manifest_path,
                    manifest,
                    evidence,
                    arm=arm,
                    step=step,
                    skip_steps=(treatment_skip_steps if arm == perception.TREATMENT_ARM else ()),
                ),
            )
    monkeypatch.setattr(perception, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        perception, "validate_ssmax_attention_report", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        perception,
        "compare_ssmax_attention_reports",
        lambda *_args, **_kwargs: {"format": "test-attention-comparison", "flags": []},
    )
    report = perception.build_promotion_report(
        manifest_path=manifest_path,
        evaluation_receipts=evaluation_refs,
        health_receipts=health_refs,
        created_at=_REPORT_TIME,
        verify_live_manifest=False,
    )
    report_ref = _write_json(tmp_path / "promotion-report.json", report)
    return manifest_path, manifest, {"report": report, "reference": report_ref}


def test_promotion_rebuilds_all_bound_receipts_and_rejects_report_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, fixture = _report_fixture(tmp_path, monkeypatch)
    report = fixture["report"]
    assert report["status"] == "passed"
    assert report["deviations"] == []
    treatment = manifest["arms"][perception.TREATMENT_ARM]
    candidate = treatment["checkpoints"]["4000"]
    summary = perception.validate_promotion_report_reference(
        fixture["reference"],
        expected_checkpoint=Path(candidate["path"]),
        expected_checkpoint_config_sha256=candidate["config_sha256"],
        expected_model_variant=manifest["model_variant"],
        expected_data_contract_sha256=treatment["data_contract_sha256"],
        expected_trainable_contract_sha256=treatment["trainable_contract_sha256"],
        verify_live_checkpoint=False,
    )
    assert summary["candidate"]["identity_sha256"] == candidate["identity_sha256"]

    changed = copy.deepcopy(report)
    changed["summary"]["windows"]["all"]["macro_step4000_treatment_gap"] = 999.0
    changed["content_sha256"] = perception.canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    changed_ref = _write_json(tmp_path / "changed-report.json", changed)
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="rebuilt from raw"):
        perception.validate_promotion_report_reference(
            changed_ref,
            expected_checkpoint=Path(candidate["path"]),
            expected_checkpoint_config_sha256=candidate["config_sha256"],
            expected_model_variant=manifest["model_variant"],
            expected_data_contract_sha256=treatment["data_contract_sha256"],
            expected_trainable_contract_sha256=treatment["trainable_contract_sha256"],
            verify_live_checkpoint=False,
        )


def test_v2_promotion_accepts_bound_finite_gradient_only_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, fixture = _report_fixture(
        tmp_path,
        monkeypatch,
        version=perception.PERCEPTION_V2_SCHEMA_VERSION,
        treatment_skip_steps=(217,),
    )
    report = fixture["report"]
    assert report["version"] == perception.PERCEPTION_V2_SCHEMA_VERSION
    assert report["status"] == "passed"
    assert report["deviations"] == []
    trajectories = report["summary"]["optimizer_guard_trajectories"]
    assert trajectories[perception.CONTROL_ARM]["observed_steps"] == []
    treatment = trajectories[perception.TREATMENT_ARM]
    assert treatment["observed_steps"] == [217]
    assert treatment["finite_gradient_only_passed"] is True
    assert treatment["clean_final_steps"] == 3783

    candidate_arm = manifest["arms"][perception.TREATMENT_ARM]
    candidate = candidate_arm["checkpoints"]["4000"]
    perception.validate_promotion_report_reference(
        fixture["reference"],
        expected_checkpoint=Path(candidate["path"]),
        expected_checkpoint_config_sha256=candidate["config_sha256"],
        expected_model_variant=manifest["model_variant"],
        expected_data_contract_sha256=candidate_arm["data_contract_sha256"],
        expected_trainable_contract_sha256=candidate_arm["trainable_contract_sha256"],
        verify_live_checkpoint=False,
    )

    mixed = copy.deepcopy(report)
    mixed["version"] = perception.SCHEMA_VERSION
    mixed["content_sha256"] = perception.canonical_sha256(
        {key: item for key, item in mixed.items() if key != "content_sha256"}
    )
    mixed_ref = _write_json(tmp_path / "mixed-version-report.json", mixed)
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="incompatible manifest",
    ):
        perception.validate_promotion_report_reference(
            mixed_ref,
            expected_checkpoint=Path(candidate["path"]),
            expected_checkpoint_config_sha256=candidate["config_sha256"],
            expected_model_variant=manifest["model_variant"],
            expected_data_contract_sha256=candidate_arm["data_contract_sha256"],
            expected_trainable_contract_sha256=candidate_arm["trainable_contract_sha256"],
            verify_live_checkpoint=False,
        )


def test_perception_health_receipt_binds_ledger_to_trainer_state_bytes(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, evidence = _manifest_fixture(tmp_path)
    receipt = _health_receipt(
        manifest_path,
        manifest,
        evidence,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )
    perception._validate_health_receipt(
        receipt,
        manifest=manifest,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )

    historical = copy.deepcopy(receipt)
    historical["rank_states"][0]["health_ledger"] = _legacy_v2_ledger(
        historical["rank_states"][0]["health_ledger"]
    )
    perception._validate_health_receipt(
        historical,
        manifest=manifest,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )

    v2_fixture_root = tmp_path / "v2"
    v2_fixture_root.mkdir()
    v2_path, v2_manifest, v2_evidence = _manifest_fixture(
        v2_fixture_root, version=perception.PERCEPTION_V2_SCHEMA_VERSION
    )
    v2_receipt = _health_receipt(
        v2_path,
        v2_manifest,
        v2_evidence,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )
    v2_receipt["rank_states"][0]["health_ledger"] = _legacy_v2_ledger(
        v2_receipt["rank_states"][0]["health_ledger"]
    )
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="resume-aware v3"):
        perception._validate_health_receipt(
            v2_receipt,
            manifest=v2_manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )

    changed = copy.deepcopy(receipt)
    changed["rank_states"][0]["trainer_state_sha256"] = "9" * 64
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="trainer-state bytes differ",
    ):
        perception._validate_health_receipt(
            changed,
            manifest=manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )


def test_v2_health_receipt_rejects_cross_rank_resume_boundary_divergence(tmp_path: Path) -> None:
    manifest_path, manifest, evidence = _manifest_fixture(
        tmp_path, version=perception.PERCEPTION_V2_SCHEMA_VERSION
    )
    receipt = _health_receipt(
        manifest_path,
        manifest,
        evidence,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )
    rank1 = copy.deepcopy(receipt["rank_states"][0])
    rank1["rank"] = 1
    rank1["trainer_state_sha256"] = "d" * 64
    rank1["health_ledger"]["optimizer_guard_history_reset_steps"] = [0, 3000]
    rank1["health_ledger"]["content_sha256"] = perception.canonical_sha256(
        {name: value for name, value in rank1["health_ledger"].items() if name != "content_sha256"}
    )
    receipt["rank_states"].append(rank1)
    manifest["topology"]["world_size"] = 2
    checkpoint = manifest["arms"][perception.TREATMENT_ARM]["checkpoints"]["3000"]
    checkpoint["trainer_state_count"] = 2
    checkpoint["trainer_state_inventory_sha256"] = perception.canonical_sha256(
        [
            {
                "rank": state["rank"],
                "path": f"train/rank{state['rank']}.pt",
                "size": state["trainer_state_size_bytes"],
                "sha256": state["trainer_state_sha256"],
            }
            for state in receipt["rank_states"]
        ]
    )
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="reset steps differ"):
        perception._validate_health_receipt(
            receipt,
            manifest=manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )

    divergent_interval = copy.deepcopy(receipt)
    divergent_ledger = divergent_interval["rank_states"][1]["health_ledger"]
    divergent_ledger["optimizer_guard_history_reset_steps"] = [0]
    divergent_ledger["optimizer_guard_rolling_interval_length"] = 129
    divergent_ledger["content_sha256"] = perception.canonical_sha256(
        {name: value for name, value in divergent_ledger.items() if name != "content_sha256"}
    )
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="rolling intervals differ"):
        perception._validate_health_receipt(
            divergent_interval,
            manifest=manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )


def test_manifest_producer_source_is_git_blob_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    sources: dict[str, Path] = {}
    for producer, relative in perception.PRODUCER_RELATIVE_PATHS.items():
        source = repository_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# exact {producer} producer\n")
        sources[producer] = source
    git = {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "ref": "b" * 40,
    }
    manifest = {
        "git": git,
        "producers": {
            producer: {
                "repo_relative_path": relative,
                "sha256": perception.sha256_file(sources[producer]),
                "git_ref": git["ref"],
            }
            for producer, relative in perception.PRODUCER_RELATIVE_PATHS.items()
        },
    }
    monkeypatch.setattr(
        perception.bridge,
        "_validate_repository_checkout",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        perception.bridge,
        "_git_blob_bytes",
        lambda *_args, repo_relative_path, **_kwargs: (
            repository_root / repo_relative_path
        ).read_bytes(),
    )
    assert (
        perception.validate_manifest_producer_source(
            manifest,
            producer=perception.EVALUATION_PRODUCER,
            source_path=sources[perception.EVALUATION_PRODUCER],
        )
        == manifest["producers"][perception.EVALUATION_PRODUCER]
    )

    sources[perception.EVALUATION_PRODUCER].write_text("# locally changed\n")
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="source identity",
    ):
        perception.validate_manifest_producer_source(
            manifest,
            producer=perception.EVALUATION_PRODUCER,
            source_path=sources[perception.EVALUATION_PRODUCER],
        )


def test_receipts_reject_rehashed_fabricated_producer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest, evaluator = _manifest_fixture(tmp_path)
    monkeypatch.setattr(
        perception, "validate_ssmax_attention_report", lambda *_args, **_kwargs: None
    )
    evaluation = _evaluation_receipt(
        manifest_path,
        manifest,
        evaluator,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )
    evaluation["evaluator"]["sha256"] = "0" * 64
    evaluation["content_sha256"] = perception.canonical_sha256(
        {key: item for key, item in evaluation.items() if key != "content_sha256"}
    )
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="evaluator source identity differs",
    ):
        perception._validate_evaluation_receipt(
            evaluation,
            manifest=manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )

    health = _health_receipt(
        manifest_path,
        manifest,
        evaluator,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )
    health["evidence"]["producer"]["repo_relative_path"] = "src/scripts/eval/fabricated.py"
    health["content_sha256"] = perception.canonical_sha256(
        {key: item for key, item in health.items() if key != "content_sha256"}
    )
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="producer source identity differs",
    ):
        perception._validate_health_receipt(
            health,
            manifest=manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )

    health = _health_receipt(
        manifest_path,
        manifest,
        evaluator,
        arm=perception.TREATMENT_ARM,
        step=3000,
    )
    health["evidence"]["recipe"]["sha256"] = "0" * 64
    health["content_sha256"] = perception.canonical_sha256(
        {key: item for key, item in health.items() if key != "content_sha256"}
    )
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="recipe identity differs",
    ):
        perception._validate_health_receipt(
            health,
            manifest=manifest,
            arm=perception.TREATMENT_ARM,
            step=3000,
        )


@pytest.mark.parametrize(
    ("gate_version", "report_version"),
    ((5, perception.SCHEMA_VERSION), (6, perception.PERCEPTION_V2_SCHEMA_VERSION)),
)
def test_versioned_gate_binds_matching_treatment_candidate_and_forbids_waivers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_version: int,
    report_version: int,
) -> None:
    candidate = tmp_path / "treatment" / "step4000"
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n")
    summary = {
        "candidate": {"identity_sha256": "1" * 64},
        "report": {
            "version": report_version,
            "content_sha256": "2" * 64,
            "created_at": _REPORT_TIME,
        },
        "manifest_reference": {
            "path": str(manifest_path),
            "sha256": perception.sha256_file(manifest_path),
            "content_sha256": "3" * 64,
        },
    }
    monkeypatch.setattr(
        perception,
        "validate_promotion_report_reference",
        lambda *_args, **_kwargs: summary,
    )
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": gate_version,
        "status": "approved",
        "recipe_version": 3,
        "formatter_version": "v1",
        "phase": "perception",
        "model_variant": "ssmax_head_qknorm",
        "arm": perception.TREATMENT_ARM,
        "checkpoint": str(candidate),
        "checkpoint_config_sha256": "4" * 64,
        "checkpoint_identity_sha256": "1" * 64,
        "data_contract_sha256": "5" * 64,
        "trainable_contract_sha256": "6" * 64,
        "global_step": 4000,
        "metrics_artifact_sha256": "7" * 64,
        "promotion_report_path": str(report_path),
        "promotion_report_sha256": "7" * 64,
        "promotion_report_content_sha256": "2" * 64,
        "manifest_path": str(manifest_path),
        "manifest_sha256": summary["manifest_reference"]["sha256"],
        "manifest_content_sha256": "3" * 64,
        "approved_by": "reviewer@example.org",
        "approved_at": "2026-08-20T02:00:00+00:00",
        "waivers": [],
    }
    perception.validate_ssmax_perception_parent_gate(
        gate,
        expected_checkpoint=candidate,
        expected_checkpoint_config_sha256="4" * 64,
        expected_model_variant="ssmax_head_qknorm",
        expected_data_contract_sha256="5" * 64,
        expected_trainable_contract_sha256="6" * 64,
        verify_live_checkpoint=False,
    )
    changed = copy.deepcopy(gate)
    changed["waivers"] = [{"id": "legacy-s002"}]
    with pytest.raises(perception.SSMaxPerceptionEvidenceError, match="does not permit waivers"):
        perception.validate_ssmax_perception_parent_gate(
            changed,
            expected_checkpoint=candidate,
            expected_checkpoint_config_sha256="4" * 64,
            expected_model_variant="ssmax_head_qknorm",
            expected_data_contract_sha256="5" * 64,
            expected_trainable_contract_sha256="6" * 64,
            verify_live_checkpoint=False,
        )

    changed = copy.deepcopy(gate)
    changed["version"] = 6 if gate_version == 5 else 5
    with pytest.raises(
        perception.SSMaxPerceptionEvidenceError,
        match="does not match promotion protocol",
    ):
        perception.validate_ssmax_perception_parent_gate(
            changed,
            expected_checkpoint=candidate,
            expected_checkpoint_config_sha256="4" * 64,
            expected_model_variant="ssmax_head_qknorm",
            expected_data_contract_sha256="5" * 64,
            expected_trainable_contract_sha256="6" * 64,
            verify_live_checkpoint=False,
        )


@pytest.mark.parametrize(
    ("report_version", "expected_gate_version"),
    ((perception.SCHEMA_VERSION, 5), (perception.PERCEPTION_V2_SCHEMA_VERSION, 6)),
)
def test_parent_gate_builder_selects_version_from_rebuilt_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_version: int,
    expected_gate_version: int,
) -> None:
    _, manifest, fixture = _report_fixture(tmp_path, monkeypatch, version=report_version)
    candidate = Path(manifest["arms"][perception.TREATMENT_ARM]["checkpoints"]["4000"]["path"])
    candidate.mkdir(parents=True)
    (candidate / "config.json").write_text(
        json.dumps(
            {
                "vision_alignment": {
                    "recipe_version": 3,
                    "formatter_version": "v1",
                }
            }
        )
        + "\n"
    )
    report_reference = fixture["reference"]

    gate = perception.build_parent_gate(
        promotion_report_path=Path(report_reference["path"]),
        expected_promotion_report_sha256=report_reference["sha256"],
        approved_by="reviewer@example.org",
        approved_at="2026-08-20T02:00:00+00:00",
        verify_live_checkpoint=False,
    )

    assert gate["version"] == expected_gate_version
