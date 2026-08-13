"""Focused tests for the perception promotion evidence boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from olmo_core.eval import vision_alignment_perception_promotion as promotion


def _write_checkpoint(tmp_path: Path, arm: str) -> tuple[Path, dict]:
    checkpoint = tmp_path / arm / "step4000"
    state_dir = checkpoint / "model_and_optim"
    state_dir.mkdir(parents=True)
    freeze_params = (
        ["vision.*", "lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"]
        if arm == promotion.CONTROL_ARM
        else ["lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*"]
    )
    contract = promotion.EXPECTED_PROFILE_CONTRACTS[arm]
    config = {
        "phase": "perception",
        "perception_trainability_arm": arm,
        "required_run_name": contract["name"],
        "reviewed_profile_path": contract["repository_path"],
        "reviewed_profile_sha256": contract["sha256"],
        "expected_launch_command": [
            "src/scripts/train/Vision-Alignment.py",
            "train",
            contract["name"],
            f"--profile={contract['repository_path']}",
        ],
        "vision_alignment": {
            "phase": "perception",
            "lineage_id": contract["name"],
            "data_contract_sha256": "a" * 64,
            "trainable_contract_sha256": contract["trainable_contract_sha256"],
        },
        "model": {"lm": {"vocab_size": 100_352}},
        "train_module": {
            "freeze_params": freeze_params,
            "train_embedding_rows": list(promotion.IMAGE_TOKEN_ROWS),
            "optim": {
                "group_overrides": [
                    {
                        "params": ["*vision.*"],
                        "opts": {"lr": contract["vision_lr"]},
                    }
                ]
            },
        },
        "trainer": {
            "save_folder": str(checkpoint.parent),
            "callbacks": {"wandb": {"name": contract["name"]}},
        },
        "launch": {"git": {"ref": promotion.EXPECTED_GIT_REF}},
        "initialization": {
            "checkpoint": promotion.EXPECTED_PARENT_CHECKPOINT,
            "expected_parent_phase": "bridge",
            "parent_config_sha256": promotion.EXPECTED_PARENT_CONFIG_SHA256,
            "parent_gate_path": promotion.EXPECTED_PARENT_GATE_PATH,
            "parent_gate_sha256": promotion.EXPECTED_PARENT_GATE_SHA256,
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    (checkpoint / ".metadata.json").write_text('{"ephemeral": false}\n')
    (state_dir / ".metadata").write_bytes(b"dcp metadata")
    (state_dir / "__0_0.distcp").write_bytes(b"state")
    state_files = sorted(path for path in state_dir.iterdir() if path.is_file())
    inventory = [
        {
            "path": path.relative_to(checkpoint).as_posix(),
            "size": path.stat().st_size,
            "sha256": promotion.sha256_file(path),
        }
        for path in state_files
    ]
    identity = {
        "root": str(checkpoint),
        "state_dir": str(state_dir),
        "config_sha256": promotion.sha256_file(checkpoint / "config.json"),
        "checkpoint_marker_sha256": promotion.sha256_file(checkpoint / ".metadata.json"),
        "dcp_metadata_sha256": promotion.sha256_file(state_dir / ".metadata"),
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": promotion.canonical_sha256(inventory),
        "state_file_inventory": inventory,
        "identity_sha256": "",
    }
    identity["identity_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    return checkpoint, identity


def _outcome_receipt(control_identity: dict, treatment_identity: dict) -> dict:
    return {
        "format": promotion.OUTCOME_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "checkpoints": {
            promotion.CONTROL_ARM: {
                "step3000": control_identity,
                "step4000": control_identity,
            },
            promotion.TREATMENT_ARM: {
                "step3000": treatment_identity,
                "step4000": treatment_identity,
            },
        },
    }


def _policy_metrics() -> dict:
    return {
        "macro": {
            "did_ci_low": 0.01,
            "treatment_gap_ci_low": 0.02,
            "control_correct_ce": 2.0,
            "treatment_correct_ce": 2.04,
            "treatment_gap": 0.8,
            "step3000_treatment_gap": 1.0,
        },
        "sources": {
            source: {"control_correct_ce": 2.0, "treatment_correct_ce": 2.04}
            for source in promotion.SOURCES
        },
    }


def _validated_outcome_receipt() -> dict:
    return {
        "format": promotion.OUTCOME_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "protocol": {
            "examples_per_source": 256,
            "primary_step": 4000,
            "durability_step": 3000,
            "pairing_seed": promotion.EXPECTED_EVALUATION_SEED,
            "sources": list(promotion.SOURCES),
        },
    }


def _guard(steps: list[int]) -> dict:
    spacing = min((right - left for left, right in pairwise(steps)), default=4000)
    return {
        "rolling_interval_length": 128,
        "sigma_factor": 12,
        "observed_steps": steps,
        "count": len(steps),
        "rate": len(steps) / 4000,
        "minimum_spacing": spacing,
        "clean_final_steps": 4000 - steps[-1] if steps else 4000,
        "every_next_step_finite": True,
    }


def _loss_sources(shares: dict[str, float]) -> tuple[dict, dict]:
    sources = {
        source: {
            "examples": 1,
            "tokens": 1,
            "positive_tokens": 1,
            "loss_weight": share * 1000,
            "active_loss_weight": share * 900,
            "target_loss_mass": promotion.LOSS_MASS_TARGETS[source],
            "loss_mass_share": share,
            "active_loss_mass_share": share,
            "absolute_error": abs(share - promotion.LOSS_MASS_TARGETS[source]),
            "active_absolute_error": abs(share - promotion.LOSS_MASS_TARGETS[source]),
        }
        for source, share in shares.items()
    }
    return sources, {
        "total_loss_weight": 1000.0,
        "total_active_loss_weight": 900.0,
        "share_sum": 1.0,
        "active_share_sum": 1.0,
        "within_tolerance": True,
        "arm_final_cursor_equal": True,
    }


def test_candidate_from_outcome_rehashes_full_step4000_identity(
    monkeypatch, tmp_path: Path
) -> None:
    control_path, control_identity = _write_checkpoint(tmp_path, promotion.CONTROL_ARM)
    treatment_path, treatment_identity = _write_checkpoint(tmp_path, promotion.TREATMENT_ARM)
    receipt = _outcome_receipt(control_identity, treatment_identity)
    monkeypatch.setitem(
        promotion.EXPECTED_PROFILE_CONTRACTS[promotion.CONTROL_ARM],
        "save_folder",
        str(control_path.parent),
    )
    monkeypatch.setitem(
        promotion.EXPECTED_PROFILE_CONTRACTS[promotion.TREATMENT_ARM],
        "save_folder",
        str(treatment_path.parent),
    )

    treatment = promotion.candidate_from_outcome_receipt(treatment_path, receipt)
    control = promotion.candidate_from_outcome_receipt(control_path, receipt, role="control")

    assert treatment["phase"] == "perception"
    assert treatment["checkpoint_identity_sha256"] == treatment_identity["identity_sha256"]
    assert (
        control["trainable_contract_sha256"]
        == promotion.EXPECTED_PROFILE_CONTRACTS[promotion.CONTROL_ARM]["trainable_contract_sha256"]
    )
    (treatment_path / "model_and_optim" / "__0_0.distcp").write_bytes(b"changed")
    with pytest.raises(promotion.PromotionValidationError, match="inventory differs"):
        promotion.candidate_from_outcome_receipt(treatment_path, receipt)


def test_stable_checkpoint_rejects_extra_entries_and_symlinked_support_file(
    tmp_path: Path,
) -> None:
    checkpoint, identity = _write_checkpoint(tmp_path, promotion.TREATMENT_ARM)
    promotion._validate_live_checkpoint_identity_stable(identity, name="test checkpoint")

    extra = checkpoint / "model_and_optim" / "extra.distcp"
    extra.write_bytes(b"unattested")
    with pytest.raises(promotion.PromotionValidationError, match="entries differ"):
        promotion._validate_live_checkpoint_identity_stable(identity, name="test checkpoint")
    extra.unlink()

    config = checkpoint / "config.json"
    target = checkpoint / "config-copy.json"
    target.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(target.name)
    with pytest.raises(promotion.PromotionValidationError, match="symlink/non-file"):
        promotion._validate_live_checkpoint_identity_stable(identity, name="test checkpoint")


def test_receipt_versions_reject_json_booleans() -> None:
    receipt = {
        "format": "test-format",
        "version": True,
        "status": "passed",
        "created_at": "2026-08-13T00:00:00+00:00",
        "content_sha256": "0" * 64,
    }
    with pytest.raises(promotion.PromotionValidationError, match="identity or status"):
        promotion._validate_receipt_header(
            receipt,
            expected_format="test-format",
            expected_fields=frozenset(receipt),
            name="test receipt",
        )


def test_outcome_policy_uses_locked_predeclared_boundaries(monkeypatch) -> None:
    metrics = _policy_metrics()
    monkeypatch.setattr(
        promotion,
        "_load_outcome_module",
        lambda: SimpleNamespace(
            validate_outcome_receipt=lambda _receipt, verify_live_inputs=True: {
                "policy_metrics": metrics
            }
        ),
    )
    receipt = _validated_outcome_receipt()
    assert promotion.validate_counterfactual_outcome_receipt(receipt)["policy_metrics"] == metrics

    metrics["macro"]["did_ci_low"] = 0.0
    with pytest.raises(promotion.PromotionValidationError, match="DID lower"):
        promotion.validate_counterfactual_outcome_receipt(receipt)
    metrics["macro"]["did_ci_low"] = 0.01
    metrics["sources"][promotion.SOURCES[-1]]["treatment_correct_ce"] = 2.040_000_1
    with pytest.raises(promotion.PromotionValidationError, match="scalar_count correct CE"):
        promotion.validate_counterfactual_outcome_receipt(receipt)

    receipt["protocol"]["examples_per_source"] = 255
    with pytest.raises(promotion.PromotionValidationError, match=">= 256"):
        promotion.validate_counterfactual_outcome_receipt(receipt)


def test_optimizer_guard_accepts_only_the_exact_observed_seven_skip_deviation() -> None:
    steps = list(promotion.EXPECTED_TREATMENT_SKIP_STEPS)
    summary = promotion._validate_optimizer_guard(
        _guard(steps), expected_arm=promotion.TREATMENT_ARM
    )
    assert summary["steps"] == tuple(steps)

    unknown = steps + [3800]
    with pytest.raises(promotion.PromotionValidationError, match="exact evidence"):
        promotion._validate_optimizer_guard(_guard(unknown), expected_arm=promotion.TREATMENT_ARM)
    nonfinite_recovery = _guard(steps)
    nonfinite_recovery["every_next_step_finite"] = False
    with pytest.raises(promotion.PromotionValidationError, match="exact evidence"):
        promotion._validate_optimizer_guard(
            nonfinite_recovery, expected_arm=promotion.TREATMENT_ARM
        )


def test_pair_contract_reference_is_locked_to_the_published_raw_sha(tmp_path: Path) -> None:
    pair = tmp_path / "pair.json"
    pair.write_text("{}\n")
    with pytest.raises(promotion.PromotionValidationError, match="published v2"):
        promotion._load_published_pair_contract(
            {"path": str(pair), "sha256": promotion.sha256_file(pair)}
        )


def test_published_bridge_adapter_never_invokes_unsafe_legacy_pickle_validation(
    monkeypatch, tmp_path: Path
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n")
    reference = {
        "path": str(receipt_path),
        "sha256": promotion.sha256_file(receipt_path),
    }
    deviations = []
    waivers = []
    for index, deviation_id in enumerate(sorted(promotion.bridge.REQUIRED_WAIVER_IDS)):
        deviation = {
            "id": deviation_id,
            "waiver_required": True,
            "criterion": f"criterion-{index}",
        }
        deviation["sha256"] = promotion.canonical_sha256(deviation)
        deviations.append(deviation)
        waivers.append(
            {
                "id": deviation_id,
                "decision": "approved",
                "deviation_sha256": deviation["sha256"],
            }
        )
    parent_identity = "1" * 64
    gate = {
        "checkpoint_identity_sha256": parent_identity,
        "data_contract_sha256": "2" * 64,
        "trainable_contract_sha256": "3" * 64,
        "waivers": waivers,
    }
    bundle = {
        "format": promotion.bridge.PROMOTION_BUNDLE_FORMAT,
        "version": promotion.bridge.PROMOTION_BUNDLE_VERSION,
        "status": "ready_for_human_approval",
        "created_at": "2026-08-13T00:00:00+00:00",
        "policy": promotion.bridge._promotion_policy(),
        "candidate": {
            "checkpoint": promotion.EXPECTED_PARENT_CHECKPOINT,
            "global_step": 500,
            "phase": "bridge",
            "lineage_id": "vision-alignment-bridge-real-v1",
            "checkpoint_config_sha256": promotion.EXPECTED_PARENT_CONFIG_SHA256,
            "checkpoint_identity_sha256": parent_identity,
            "checkpoint_marker_sha256": "4" * 64,
            "dcp_metadata_sha256": "5" * 64,
            "state_file_inventory_sha256": "6" * 64,
            "data_contract_sha256": gate["data_contract_sha256"],
            "trainable_contract_sha256": gate["trainable_contract_sha256"],
            "vocab_size": 100_352,
            "image_embedding_rows": list(promotion.IMAGE_TOKEN_ROWS),
        },
        "receipts": {
            "frozen_state": reference,
            "text_retention": reference,
            "cumulative_loss_mass": reference,
            "optimizer_guard": reference,
            "matched_wrong": {
                role: reference
                for role in (
                    "canary_step250",
                    "bridge_step250",
                    "bridge_step500",
                    "independent_step0",
                    "independent_step500",
                )
            },
        },
        "deviations": deviations,
        "content_sha256": "",
    }
    bundle["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in bundle.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(
        promotion.bridge,
        "validate_promotion_bundle",
        lambda *args, **kwargs: pytest.fail("unsafe historical validator was invoked"),
    )
    assert promotion._validate_published_bridge_bundle(bundle, gate=gate) == parent_identity


def test_initialization_parity_requires_the_full_818_parameter_inventory(
    monkeypatch, tmp_path: Path
) -> None:
    producer = Path(promotion.__file__).resolve().parents[2] / (
        "scripts/eval/vision_alignment_perception_state_text.py"
    )
    parent = tmp_path / "bridge" / "step500"
    step0 = tmp_path / "arm" / "step0"
    parent.mkdir(parents=True)
    step0.mkdir(parents=True)
    parent_identity = {"root": str(parent), "config_sha256": "1" * 64}
    step0_identity = {"root": str(step0), "config_sha256": "2" * 64}
    comparison = {
        "name": "only.weight",
        "kind": "parameter",
        "dtype": "float32",
        "shape": [1],
        "numel": 1,
        "reference_sha256": "3" * 64,
        "step0_sha256": "3" * 64,
    }
    receipt = {
        "format": promotion.INITIALIZATION_PARITY_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": "2026-08-13T00:00:00+00:00",
        "producer": {
            "path": str(tmp_path / producer.name),
            "sha256": promotion.sha256_file(producer),
        },
        "native_helper": {
            "path": str(tmp_path / "vision_alignment_matched_wrong.py"),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_matched_wrong.py"
            ),
        },
        "snapshot_helper": {
            "path": str(tmp_path / "vision_alignment_perception_matched_wrong.py"),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_perception_matched_wrong.py"
            ),
        },
        "arm": promotion.TREATMENT_ARM,
        "reference_checkpoint": parent_identity,
        "perception_step0": step0_identity,
        "protocol": {
            "name": "logical-all-model-tensor-sha256-v1",
            "hash_algorithm": "sha256",
            "tensor_encoding": "dtype-shape-contiguous-little-endian-v1",
        },
        "comparisons": [comparison],
        "summary": {
            "complete": True,
            "expected_tensor_count": 1,
            "compared_tensor_count": 1,
            "mismatch_count": 0,
            "comparison_inventory_sha256": promotion.canonical_sha256([comparison]),
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(
        promotion.bridge,
        "_validate_checkpoint_identity",
        lambda value, name: value,
    )
    monkeypatch.setattr(
        promotion,
        "_validate_live_checkpoint_identity_stable",
        lambda value, name: None,
    )
    candidate = {
        "checkpoint": str(step0.parent / "step4000"),
        "checkpoint_config_sha256": "2" * 64,
    }
    with pytest.raises(promotion.PromotionValidationError, match="incomplete"):
        promotion.validate_initialization_parity_receipt(
            receipt, candidate=candidate, expected_arm=promotion.TREATMENT_ARM
        )


def test_text_retention_requires_the_published_sentinel(monkeypatch, tmp_path: Path) -> None:
    producer = Path(promotion.__file__).resolve().parents[2] / (
        "scripts/eval/vision_alignment_perception_state_text.py"
    )
    candidate = {
        "checkpoint": str(tmp_path / "arm" / "step4000"),
        "global_step": 4000,
        "checkpoint_config_sha256": "2" * 64,
        "checkpoint_identity_sha256": "3" * 64,
    }
    receipt = {
        "format": promotion.PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "evaluator": {
            "path": str(tmp_path / producer.name),
            "sha256": promotion.sha256_file(producer),
        },
        "native_helper": {
            "path": str(tmp_path / "vision_alignment_matched_wrong.py"),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_matched_wrong.py"
            ),
        },
        "snapshot_helper": {
            "path": str(tmp_path / "vision_alignment_perception_matched_wrong.py"),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_perception_matched_wrong.py"
            ),
        },
        "candidate": candidate,
        "reference_checkpoint": {
            "checkpoint": str(tmp_path / "arm" / "step0"),
            "global_step": 0,
            "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
            "checkpoint_identity_sha256": "4" * 64,
        },
        "dataset": {
            "path": str(tmp_path / "sentinel.json"),
            "sha256": "0" * 64,
            "fingerprint": "1" * 64,
            "examples": 128,
            "supervised_tokens": 32_768,
            "input_ids_sha256": "5" * 64,
            "labels_sha256": "6" * 64,
            "image_token_count": 0,
            "image_tensor_count": 0,
        },
    }
    monkeypatch.setattr(
        promotion.bridge,
        "validate_text_retention_receipt",
        lambda value, candidate: {"status": "passed"},
    )
    with pytest.raises(promotion.PromotionValidationError, match="locked sentinel"):
        promotion.validate_perception_text_retention_receipt(receipt, candidate=candidate)


def test_run_health_locks_the_single_control_prestart_failure(monkeypatch, tmp_path: Path) -> None:
    producer = Path(promotion.__file__).resolve().parents[2] / (
        "scripts/eval/vision_alignment_perception_run_health.py"
    )
    candidate = {
        "checkpoint": str(tmp_path / "control" / "step4000"),
        "global_step": 4000,
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_identity_sha256": "b" * 64,
    }
    beaker = tmp_path / "beaker.json"
    beaker.write_text("{}\n")
    prestart_log = tmp_path / "prestart.log"
    prestart_log.write_text("healthcheck failed before training\n")
    wandb_files = tmp_path / "control" / "wandb" / "wandb" / "run-test-control-run" / "files"
    wandb_files.mkdir(parents=True)
    output_log = wandb_files / "output.log"
    output_log.write_text("synthetic complete log\n")
    summary_file = wandb_files / "wandb-summary.json"
    summary_file.write_text('{"_step":4000}\n')

    def reference(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": promotion.sha256_file(path)}

    success_jobs = {0: "success-0", 1: "success-1"}
    locked_failure = {
        "job_id": "failed-replica-1",
        "replica_rank": 1,
        "canceled_code": 10,
        "reason": "healthcheck failure before user code",
    }
    receipt = {
        "format": promotion.RUN_HEALTH_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": "2026-08-13T00:00:00+00:00",
        "producer": {
            "path": str(tmp_path / "gantry-runtime" / producer.name),
            "sha256": promotion.sha256_file(producer),
        },
        "checkpoint_identity_helper": {
            "path": str(tmp_path / "vision_alignment_perception_matched_wrong.py"),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_perception_matched_wrong.py"
            ),
        },
        "arm": promotion.CONTROL_ARM,
        "candidate": candidate,
        "launch": {
            "workspace": "ai2/molmofication",
            "experiment_id": promotion.EXPECTED_EXPERIMENT_IDS[promotion.CONTROL_ARM],
            "successful_jobs": [
                {
                    "job_id": f"success-{rank}",
                    "replica_rank": rank,
                    "exit_code": 0,
                    "started_training": True,
                    "completed_training": True,
                    "log": reference(output_log),
                }
                for rank in range(2)
            ],
            "prestart_failures": [
                {
                    "job_id": "failed-replica-1",
                    "replica_rank": 1,
                    "exit_code": 10,
                    "started_training": False,
                    "reason": "healthcheck failure before user code",
                    "evidence": reference(prestart_log),
                }
            ],
        },
        "run": {
            "run_id": "control-run",
            "global_steps": 4000,
            "exit_code": 0,
            "rank_state_count": 16,
            "permanent_checkpoint_steps": [0, 1000, 2000, 3000, 4000],
            "metric_step_count": 4000,
            "numeric_metric_count": 1,
            "nonfinite_metric_count": 0,
            "unexpected_anomaly_count": 0,
            "total_data_errors": 0,
            "successful_terminal_marker": True,
        },
        "rank_state_inventory": [],
        "permanent_checkpoints": [],
        "optimizer_guard": _guard([]),
        "evidence": {
            "beaker_experiment": reference(beaker),
            "wandb_output": reference(output_log),
            "wandb_summary": reference(summary_file),
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    fake_module = SimpleNamespace(
        EXPECTED_SUCCESSFUL_JOBS={promotion.CONTROL_ARM: success_jobs},
        CONTROL_PRESTART_FAILURE=locked_failure,
        _audit_job_log=lambda path: (True, True),
        _audit_job_log_text=lambda text, name: (True, True),
        _verify_locked_experiment_snapshot=lambda value, arm: None,
        _audit_log=lambda path: {
            "metric_step_count": 4000,
            "numeric_metric_count": 1,
            "nonfinite_metric_count": 0,
            "unexpected_anomaly_count": 0,
            "guarded_skip_steps": [],
            "successful_terminal_marker": True,
            "every_next_step_finite": True,
        },
        _audit_log_text=lambda text: {
            "metric_step_count": 4000,
            "numeric_metric_count": 1,
            "nonfinite_metric_count": 0,
            "unexpected_anomaly_count": 0,
            "guarded_skip_steps": [],
            "successful_terminal_marker": True,
            "every_next_step_finite": True,
        },
        _audit_summary=lambda path, expected_run_id: None,
        _audit_summary_value=lambda value, expected_run_id: None,
    )
    monkeypatch.setattr(promotion, "_load_run_health_module", lambda: fake_module)
    monkeypatch.setattr(promotion, "_validate_rank_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(promotion, "_validate_permanent_checkpoints", lambda value, candidate: None)
    assert (
        promotion.validate_perception_run_health_receipt(
            receipt, candidate=candidate, expected_arm=promotion.CONTROL_ARM
        )["guarded_skip_steps"]
        == []
    )

    missing = deepcopy(receipt)
    missing["launch"]["prestart_failures"] = []
    missing["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in missing.items() if key != "content_sha256"}
    )
    with pytest.raises(promotion.PromotionValidationError, match="pre-start failure inventory"):
        promotion.validate_perception_run_health_receipt(
            missing, candidate=candidate, expected_arm=promotion.CONTROL_ARM
        )

    wrong_experiment = deepcopy(receipt)
    wrong_experiment["launch"]["experiment_id"] = "01WRONGEXPERIMENT0000000000"
    wrong_experiment["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in wrong_experiment.items() if key != "content_sha256"}
    )
    with pytest.raises(promotion.PromotionValidationError, match="locked run"):
        promotion.validate_perception_run_health_receipt(
            wrong_experiment, candidate=candidate, expected_arm=promotion.CONTROL_ARM
        )


def test_loss_mass_requires_each_predeclared_source_within_two_points() -> None:
    shares = dict(promotion.LOSS_MASS_TARGETS)
    sources, summary = _loss_sources(shares)
    result = promotion._validate_loss_sources(sources, summary)
    assert result["loss_mass_share"] == shares

    changed = dict(shares)
    changed["audited_alignment"] += 0.021
    changed["pixmo_caption"] -= 0.021
    sources, summary = _loss_sources(changed)
    with pytest.raises(promotion.PromotionValidationError, match="exceeds the 2% tolerance"):
        promotion._validate_loss_sources(sources, summary)


def test_loss_evidence_rederives_saved_dataset_fingerprint_identity(tmp_path: Path) -> None:
    loss_module = promotion._load_loss_mass_module()
    fingerprints = [{"source": "pinned", "sha256": "1" * 64}]
    inventories = {}
    arms = {}
    candidates = {}
    cursor_inventories = {}
    for arm in promotion.ARMS:
        checkpoint = tmp_path / arm / "step4000"
        train = checkpoint / "train"
        train.mkdir(parents=True)
        inventory = []
        loaded = []
        for rank in range(16):
            state = {
                "global_step": 4000,
                "world_size": 16,
                "data_loader": {
                    "batches_processed": 4000,
                    "total_data_errors": 0,
                    "packing_state": {
                        "version": 5,
                        "dp_world_size": 16,
                        "dp_rank": rank,
                        "packs_emitted": 32_000,
                        "dataset_fingerprints": fingerprints,
                    },
                },
            }
            path = train / f"rank{rank}.pt"
            torch.save(state, path)
            inventory.append(
                {"rank": rank, "path": str(path), "sha256": promotion.sha256_file(path)}
            )
            loaded.append(state)
        cursor = loss_module._loader_state_inventory(loaded)
        cursor_inventories[arm] = cursor
        cursor_sha = promotion.canonical_sha256(cursor)
        inventories[arm] = inventory
        arms[arm] = {
            "rank_state_inventory_sha256": promotion.canonical_sha256(inventory),
            "checkpoint_final_state_sha256": cursor_sha,
        }
        candidates[arm] = {"checkpoint": str(checkpoint)}
    assert cursor_inventories[promotion.CONTROL_ARM] == cursor_inventories[promotion.TREATMENT_ARM]
    final_cursor_sha = promotion.canonical_sha256(cursor_inventories[promotion.CONTROL_ARM])
    evidence = {
        "recipe": {
            "path": str(promotion._PERCEPTION_RECIPE_PATH),
            "sha256": promotion.sha256_file(promotion._PERCEPTION_RECIPE_PATH),
        },
        "producer": {
            "path": str(promotion._LOSS_MASS_PRODUCER_PATH),
            "sha256": promotion.sha256_file(promotion._LOSS_MASS_PRODUCER_PATH),
        },
        "rank_state_inventory": inventories,
    }
    promotion._validate_loss_evidence(
        evidence,
        arms=arms,
        candidates=candidates,
        expected_final_cursor_sha256=final_cursor_sha,
        expected_dataset_fingerprints_sha256=promotion.canonical_sha256(fingerprints),
    )
    with pytest.raises(promotion.PromotionValidationError, match="dataset fingerprint"):
        promotion._validate_loss_evidence(
            evidence,
            arms=arms,
            candidates=candidates,
            expected_final_cursor_sha256=final_cursor_sha,
            expected_dataset_fingerprints_sha256="f" * 64,
        )


def test_run_health_milestones_bind_the_outcome_checkpoint_identities() -> None:
    identities = {
        arm: {
            f"step{step}": {
                "root": f"/{arm}/step{step}",
                "config_sha256": "1" * 64,
                "checkpoint_marker_sha256": "2" * 64,
                "dcp_metadata_sha256": "3" * 64,
            }
            for step in (3000, 4000)
        }
        for arm in promotion.ARMS
    }
    run_health = {
        arm: {"permanent_checkpoints": deepcopy(identities[arm])} for arm in promotion.ARMS
    }
    promotion._bind_run_health_to_outcome(run_health, {"checkpoints": identities})
    run_health[promotion.TREATMENT_ARM]["permanent_checkpoints"]["step3000"][
        "dcp_metadata_sha256"
    ] = ("4" * 64)
    with pytest.raises(promotion.PromotionValidationError, match="run-health identity"):
        promotion._bind_run_health_to_outcome(run_health, {"checkpoints": identities})


def test_run_health_step0_binds_initialization_and_loss_rank_states() -> None:
    step0 = {
        arm: {
            "root": f"/{arm}/step0",
            "identity_sha256": ("1" if arm == promotion.CONTROL_ARM else "2") * 64,
        }
        for arm in promotion.ARMS
    }
    inventories = {
        arm: [
            {"rank": rank, "path": f"/{arm}/rank{rank}.pt", "sha256": f"{rank:x}" * 64}
            for rank in range(16)
        ]
        for arm in promotion.ARMS
    }
    run_health = {
        arm: {
            "permanent_checkpoints": {"step0": deepcopy(step0[arm])},
            "rank_state_inventory": deepcopy(inventories[arm]),
        }
        for arm in promotion.ARMS
    }
    initialization = {arm: {"step0_checkpoint": deepcopy(step0[arm])} for arm in promotion.ARMS}
    loss_mass = {"rank_state_inventory": deepcopy(inventories)}
    promotion._bind_run_health_to_initialization(run_health, initialization)
    promotion._bind_run_health_to_loss(run_health, loss_mass)

    initialization[promotion.TREATMENT_ARM]["step0_checkpoint"]["identity_sha256"] = "3" * 64
    with pytest.raises(promotion.PromotionValidationError, match="initialization evidence"):
        promotion._bind_run_health_to_initialization(run_health, initialization)
    loss_mass["rank_state_inventory"][promotion.TREATMENT_ARM][7]["sha256"] = "f" * 64
    with pytest.raises(promotion.PromotionValidationError, match="trainer states"):
        promotion._bind_run_health_to_loss(run_health, loss_mass)


def test_perception_state_receipt_uses_new_evaluator_without_weakening_bridge(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = Path(promotion.__file__).resolve().parents[2] / (
        "scripts/eval/vision_alignment_perception_state_text.py"
    )
    receipt = {
        "format": promotion.PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "evaluator": {
            "path": str(tmp_path / "gantry-runtime" / evaluator.name),
            "sha256": promotion.sha256_file(evaluator),
        },
        "native_helper": {
            "path": str(tmp_path / "gantry-runtime" / "vision_alignment_matched_wrong.py"),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_matched_wrong.py"
            ),
        },
        "snapshot_helper": {
            "path": str(
                tmp_path / "gantry-runtime" / "vision_alignment_perception_matched_wrong.py"
            ),
            "sha256": promotion.sha256_file(
                Path(promotion.__file__).resolve().parents[2]
                / "scripts/eval/vision_alignment_perception_matched_wrong.py"
            ),
        },
        "candidate": {"global_step": 4000},
        "reference_checkpoint": {"global_step": 0},
        "summary": {"mismatch_count": 0},
    }
    observed = {}

    def validate(adapted, *, candidate, expected_frozen_tensor_count):
        observed.update(adapted)
        assert expected_frozen_tensor_count == 806
        return {"frozen_tensor_count": 806}

    monkeypatch.setattr(promotion.bridge, "validate_frozen_state_receipt", validate)
    result = promotion.validate_perception_frozen_state_receipt(
        receipt, candidate={}, expected_frozen_tensor_count=806
    )
    assert result["frozen_tensor_count"] == 806
    assert observed["format"] == promotion.bridge.FROZEN_STATE_RECEIPT_FORMAT
    assert receipt["format"] == promotion.PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT


def test_bundle_requires_one_exact_human_waiver_and_is_content_hashed(monkeypatch) -> None:
    candidate = {"checkpoint": "/tmp/treatment/step4000"}
    comparator = {"checkpoint": "/tmp/control/step4000"}
    references = {
        "pair_contract": {"path": "/tmp/pair.json", "sha256": "1" * 64},
        "initialization_parity": {},
        "counterfactual_outcome": {"path": "/tmp/outcome.json", "sha256": "2" * 64},
        "frozen_state": {},
        "text_retention": {},
        "run_health": {
            promotion.CONTROL_ARM: {"path": "/tmp/control.json", "sha256": "3" * 64},
            promotion.TREATMENT_ARM: {"path": "/tmp/treatment.json", "sha256": "4" * 64},
        },
        "loss_mass_pair": {"path": "/tmp/loss.json", "sha256": "5" * 64},
    }
    component = {
        "run_health": {
            promotion.CONTROL_ARM: {"run_id": "control"},
            promotion.TREATMENT_ARM: {"run_id": "treatment"},
        }
    }
    deviation = promotion._guard_deviation(
        component["run_health"][promotion.TREATMENT_ARM],
        references["run_health"][promotion.TREATMENT_ARM],
    )
    bundle = {
        "format": promotion.PERCEPTION_PROMOTION_BUNDLE_FORMAT,
        "version": promotion.PERCEPTION_PROMOTION_BUNDLE_VERSION,
        "status": "ready_for_human_approval",
        "created_at": "2026-08-13T00:00:00+00:00",
        "policy": promotion._promotion_policy(),
        "candidate": candidate,
        "comparator": comparator,
        "receipts": references,
        "deviations": [deviation],
        "content_sha256": "",
    }
    bundle["content_sha256"] = promotion.canonical_sha256(
        {key: value for key, value in bundle.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(promotion, "_validate_bundle_candidate", lambda value, name: value)
    monkeypatch.setattr(
        promotion,
        "_validate_component_references",
        lambda refs, candidate, comparator: component,
    )
    validated = promotion.validate_perception_promotion_bundle(bundle)
    assert set(validated["deviation_sha256"]) == promotion.REQUIRED_WAIVER_IDS

    tampered = deepcopy(bundle)
    tampered["deviations"] = []
    with pytest.raises(promotion.PromotionValidationError, match="content SHA-256 differs"):
        promotion.validate_perception_promotion_bundle(tampered)
