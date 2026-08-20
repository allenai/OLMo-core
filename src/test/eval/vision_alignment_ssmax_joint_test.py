from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest

from olmo_core.eval import vision_alignment_ssmax_joint as joint

ZERO = "0" * 64


def _spec(variant: str = "ssmax_head_qknorm") -> dict[str, Any]:
    return {
        "format": joint.MANIFEST_SPEC_FORMAT,
        "version": joint.SCHEMA_VERSION,
        "run_id": f"{variant}-joint-v1",
        "model_variant": variant,
        "run_name": f"{variant}-joint",
        "checkpoint_root": "/checkpoints/joint",
        "training_profile": "/profiles/joint.yaml",
        "recipe": "/repo/Vision-Alignment.py",
        "perception_parent_gate": "/evidence/perception-v5.json",
        "joint_visual_projection": "/artifacts/projection.json",
        "source_audit": "/artifacts/audit.json",
        "attention_probe": "/artifacts/joint-attention.json",
        "pairing_paths": {source: f"/pairings/{source}.json" for source in joint.VISUAL_SOURCES},
        "evaluation": {
            "visual_sources": list(joint.VISUAL_SOURCES),
            "steps": list(joint.REQUIRED_STEPS),
            "windows": list(joint.WINDOWS),
            "examples_per_source": 496,
            "eligible_rows_per_source": dict(joint.ELIGIBLE_VISUAL_ROWS),
            "native_holdout_examples": 992,
            "pairing_seed": 6198,
            "single_response_projection_seed": 95818,
            "rank_batch_instances": 1,
        },
        "topology": {
            "world_size": 16,
            "num_nodes": 2,
            "gpus_per_node": 8,
            "data_parallel": "hsdp",
        },
        "policy": {
            "decision_scope": "descriptive_non_promotion",
            "maximum_data_errors": 0,
            "maximum_optimizer_guard_skips": 0,
            "maximum_nonfinite_losses": 0,
            "maximum_nonfinite_gradients": 0,
            "native_text_ce_max_relative_increase": 0.02,
            "native_text_bootstrap_samples": 10000,
            "native_text_bootstrap_seed": 6198,
            "require_exact_frozen_surfaces": True,
        },
        "companion_protocols": {"downstream_fast_pair": "/protocols/downstream.yaml"},
    }


@pytest.mark.parametrize(
    "filename",
    [
        "ssmax_head_qknorm_joint_manifest_v1.json.template",
        "ssmax_no_qknorm_joint_manifest_v1.json.template",
    ],
)
def test_checked_in_manifest_templates_lock_joint_trajectory(filename: str) -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_alignment/eval/joint"
        / filename
    )
    spec = joint._validate_spec(joint.load_json(path))

    assert spec["evaluation"]["steps"] == [0, 4000, 8000, 12000, 16000]
    assert spec["evaluation"]["examples_per_source"] == 496
    assert spec["evaluation"]["eligible_rows_per_source"] == joint.ELIGIBLE_VISUAL_ROWS
    assert spec["evaluation"]["native_holdout_examples"] == 992
    assert spec["topology"] == {
        "world_size": 16,
        "num_nodes": 2,
        "gpus_per_node": 8,
        "data_parallel": "hsdp",
    }
    assert spec["policy"]["decision_scope"] == "descriptive_non_promotion"


def test_spec_rejects_native_population_not_divisible_by_world() -> None:
    spec = _spec()
    spec["evaluation"]["native_holdout_examples"] = 1000

    with pytest.raises(joint.SSMaxJointEvidenceError, match="native holdout examples"):
        joint._validate_spec(spec)


def test_spec_rejects_padded_visual_population_or_eligibility_drift() -> None:
    spec = _spec()
    spec["evaluation"]["examples_per_source"] = 512
    with pytest.raises(joint.SSMaxJointEvidenceError, match="largest common"):
        joint._validate_spec(spec)


@pytest.mark.parametrize(
    ("pairing_seed", "projection_seed"),
    [(95818, 6198), (6198, 6198), (95818, 95818)],
)
def test_spec_rejects_swapped_or_conflated_independent_seeds(
    pairing_seed: int, projection_seed: int
) -> None:
    spec = _spec()
    spec["evaluation"]["pairing_seed"] = pairing_seed
    spec["evaluation"]["single_response_projection_seed"] = projection_seed

    with pytest.raises(joint.SSMaxJointEvidenceError, match="independent fixed contracts"):
        joint._validate_spec(spec)

    spec = _spec()
    spec["evaluation"]["eligible_rows_per_source"]["pixmo_caption"] = 512
    with pytest.raises(joint.SSMaxJointEvidenceError, match="live eligibility"):
        joint._validate_spec(spec)


def test_native_ce_ppl_receipt_is_recomputed() -> None:
    manifest = {"evaluation": {"native_holdout_examples": 992}}
    result = {
        "examples": 992,
        "tokens": 990,
        "loss_weight": 50.0,
        "summed_ce": 100.0,
        "ce": 2.0,
        "ppl": math.exp(2.0),
        "filtered_examples": 2,
        "dataset_order_sha256": ZERO,
        "row_provenance_sha256": ZERO,
        "native_identity_sha256": ZERO,
        "per_example": [
            {
                "position": position,
                "tokens": 0 if position < 2 else 1,
                "mask_weight": 0.0 if position < 2 else 1.0,
                "loss_weight": 0.0 if position < 2 else 50.0 / 990,
                "summed_ce": 0.0 if position < 2 else 100.0 / 990,
                "filtered": position < 2,
            }
            for position in range(992)
        ],
    }

    assert joint._validate_native_result(result, manifest=manifest) == result
    result["ppl"] += 0.1
    with pytest.raises(joint.SSMaxJointEvidenceError, match="PPL"):
        joint._validate_native_result(result, manifest=manifest)


def test_visual_rows_recompute_each_window_gap() -> None:
    manifest = {"evaluation": {"examples_per_source": 8}}
    rows = []
    for position in range(8):
        correct = {window: 1.0 + position / 100 for window in joint.WINDOWS}
        wrong = {window: value + 0.5 for window, value in correct.items()}
        rows.append(
            {
                "pairing_position": position,
                "recipient_index": position,
                "donor_index": (position + 1) % 8,
                "response_tokens": 40,
                "correct_ce": correct,
                "wrong_ce": wrong,
                "ce_gap_wrong_minus_correct": {window: 0.5 for window in joint.WINDOWS},
            }
        )

    assert (
        len(joint._validate_rows(rows, source="pixmo_caption", manifest=manifest, pairing=None))
        == 8
    )
    rows[0]["ce_gap_wrong_minus_correct"]["first_8"] = 0.4
    with pytest.raises(joint.SSMaxJointEvidenceError, match="inconsistent"):
        joint._validate_rows(rows, source="pixmo_caption", manifest=manifest, pairing=None)


def test_joint_health_reopens_and_validates_all_checkpoint_rank_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = 4000
    world = 16
    loader_state = {"batches_processed": step}
    rank_states = [
        {
            "rank": rank,
            "global_step": step,
            "batches_processed": step,
            "data_loader_state_sha256": joint.canonical_sha256(loader_state),
            "trainer_state_sha256": ZERO,
            "health_ledger": {"rank": rank},
        }
        for rank in range(world)
    ]
    targets = {source: 1 / len(joint.TRAIN_SOURCES) for source in joint.TRAIN_SOURCES}
    counters = {
        "data_errors": 0,
        "optimizer_guard_skips": 2,
        "nonfinite_losses": 0,
        "nonfinite_gradients": 0,
    }
    receipt = {
        "checkpoint": {"path": "/checkpoint"},
        "rank_states": rank_states,
        "sources": {
            source: {
                "examples": 1,
                "tokens": 2,
                "positive_tokens": 1,
                "loss_weight": 1.0,
                "active_loss_weight": 1.0,
                "target_loss_mass": target,
            }
            for source, target in targets.items()
        },
        "run_counters": counters,
        "evidence": {
            "recipe": {"path": "/recipe", "sha256": ZERO},
            "producer": {"path": "/producer", "sha256": ZERO},
        },
    }
    manifest = {
        "model_variant": "ssmax_head_qknorm",
        "run_name": "joint-run",
        "topology": {"world_size": world},
        "loss_mass_targets": targets,
    }
    monkeypatch.setattr(
        Path,
        "glob",
        lambda self, pattern: [self / f"rank{rank}.pt" for rank in range(world)],
    )
    monkeypatch.setattr(joint, "sha256_file", lambda path: ZERO)
    monkeypatch.setattr(
        joint.torch,
        "load",
        lambda path, **kwargs: {"data_loader": loader_state},
    )
    seen = {}

    def extract(states: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert len(states) == world
        return {
            "rank_ledgers": [{"rank": rank} for rank in range(world)],
            "counters": counters,
        }

    monkeypatch.setattr(joint, "extract_ssmax_health_ledgers", extract)
    monkeypatch.setattr(joint, "validate_artifact_reference", lambda *args, **kwargs: Path("/x"))

    validated = joint._validate_health_receipt(receipt, manifest=manifest, step=step)

    assert len(validated["rank_states"]) == 16
    assert seen["expected_world_size"] == 16
    assert seen["expected_phase"] == "joint"


def _surface(mismatches: int = 0) -> dict[str, Any]:
    return {
        "protocol": "logical-tensor-comparison-sha256-v1",
        "tensor_count": 1,
        "reference_inventory_sha256": ZERO,
        "candidate_inventory_sha256": ZERO if mismatches == 0 else "1" * 64,
        "mismatch_count": mismatches,
    }


def _evaluation(step: int, *, frozen_mismatches: int = 0) -> dict[str, Any]:
    rows = []
    for position in range(8):
        correct = {window: 1.0 + step / 100_000 for window in joint.WINDOWS}
        wrong = {window: value + 0.5 + step / 100_000 for window, value in correct.items()}
        rows.append(
            {
                "pairing_position": position,
                "recipient_index": position,
                "donor_index": (position + 1) % 8,
                "response_tokens": 8,
                "correct_ce": correct,
                "wrong_ce": wrong,
                "ce_gap_wrong_minus_correct": {
                    window: wrong[window] - correct[window] for window in joint.WINDOWS
                },
            }
        )
    native_ce = 2.0 + step / 1_000_000
    native_rows = [
        {
            "position": position,
            "tokens": 1,
            "mask_weight": 1.0,
            "loss_weight": 1.0,
            "summed_ce": native_ce,
            "filtered": False,
        }
        for position in range(8)
    ]
    return {
        "state": {
            "frozen_lexical_input_rows": _surface(frozen_mismatches),
            "frozen_output_projection": _surface(),
        },
        "native": {
            "ce": native_ce,
            "ppl": math.exp(native_ce),
            "native_identity_sha256": ZERO,
            "dataset_order_sha256": ZERO,
            "per_example": native_rows,
        },
        "rows": {source: copy.deepcopy(rows) for source in joint.VISUAL_SOURCES},
        "attention": {"step": step},
    }


def _health(step: int, *, nonfinite: int = 0, optimizer_guard_skips: int = 0) -> dict[str, Any]:
    return {
        "rank_states": [{"rank": 0}],
        "sources": {
            source: {"active_loss_weight": 0.0 if step == 0 else 100.0 / len(joint.TRAIN_SOURCES)}
            for source in joint.TRAIN_SOURCES
        },
        "run_counters": {
            "data_errors": 0,
            "optimizer_guard_skips": optimizer_guard_skips,
            "nonfinite_losses": nonfinite,
            "nonfinite_gradients": 0,
        },
    }


def _report_manifest() -> dict[str, Any]:
    return {
        "run_id": "qk-joint",
        "model_variant": "ssmax_head_qknorm",
        "topology": {"world_size": 1},
        "policy": {
            "maximum_data_errors": 0,
            "maximum_optimizer_guard_skips": 0,
            "maximum_nonfinite_losses": 0,
            "maximum_nonfinite_gradients": 0,
            "native_text_ce_max_relative_increase": 0.02,
            "native_text_bootstrap_samples": 10000,
            "native_text_bootstrap_seed": 6198,
        },
        "loss_mass_targets": {
            source: 1 / len(joint.TRAIN_SOURCES) for source in joint.TRAIN_SOURCES
        },
        "companion_protocols": {"downstream_fast_pair": {"path": "/x", "sha256": ZERO}},
        "content_sha256": ZERO,
    }


def test_trajectory_is_descriptive_and_only_hard_invariants_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _report_manifest()
    evaluations = {step: _evaluation(step) for step in joint.REQUIRED_STEPS}
    health = {step: _health(step) for step in joint.REQUIRED_STEPS}
    monkeypatch.setattr(joint, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        joint,
        "manifest_reference",
        lambda *args, **kwargs: {"path": "/manifest", "sha256": ZERO, "content_sha256": ZERO},
    )
    monkeypatch.setattr(
        joint,
        "_load_receipt",
        lambda *args, step, expected_format, **kwargs: {
            "status": "passed",
            "step": step,
            "format": expected_format,
        },
    )
    monkeypatch.setattr(
        joint,
        "_validate_evaluation_receipt",
        lambda receipt, *, manifest, step: evaluations[step],
    )
    monkeypatch.setattr(
        joint,
        "_validate_health_receipt",
        lambda receipt, *, manifest, step: health[step],
    )
    monkeypatch.setattr(
        joint,
        "compare_ssmax_attention_reports",
        lambda baseline, candidate: {"baseline": baseline["step"], "candidate": candidate["step"]},
    )
    references = {step: {"path": f"/{step}.json", "sha256": ZERO} for step in joint.REQUIRED_STEPS}

    report = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )

    assert report["status"] == "passed_hard_invariants"
    assert report["decision_scope"] == "descriptive_non_promotion"
    assert (
        report["trajectory"]["16000"]["visual"]["pixmo_caption"]["first_8"]["retention_vs_step0"]
        > 1
    )
    assert report["trajectory"]["16000"]["native_text"]["ce_change_vs_step0"] > 0
    assert report["attention_trajectory"]["16000"] == {
        "baseline": 0,
        "candidate": 16000,
    }

    health[8000] = _health(8000, nonfinite=1)
    failed = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )
    assert failed["status"] == "failed_hard_invariants"

    health[8000] = _health(8000, optimizer_guard_skips=1)
    skipped = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )
    assert skipped["status"] == "failed_hard_invariants"
    assert (
        skipped["hard_invariants"]["by_step"]["8000"]["optimizer_guard_skips_within_limit"] is False
    )

    health[8000] = _health(8000)
    regressed_native = evaluations[8000]["native"]
    for row in regressed_native["per_example"]:
        row["summed_ce"] = 2.4
    regressed_native["ce"] = 2.4
    regressed_native["ppl"] = math.exp(2.4)
    native_regression = joint.build_trajectory_report(
        manifest_path=tmp_path / "manifest.json",
        evaluation_receipts=references,
        health_receipts=references,
        created_at="2026-08-20T00:00:00+00:00",
    )
    assert native_regression["status"] == "failed_hard_invariants"
    assert (
        native_regression["hard_invariants"]["by_step"]["8000"]["native_text_ce_noninferior"]
        is False
    )


def test_pair_comparison_directly_compares_attention_at_every_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = {field: None for field in joint._REPORT_FIELDS}
    left.update(
        {
            "decision_scope": "descriptive_non_promotion",
            "run_id": "left",
            "model_variant": "ssmax_head_qknorm",
            "content_sha256": ZERO,
            "hard_invariants": {"passed": True},
            "trajectory": {},
            "paired_visual_rows": {},
            "attention_reports": {},
            "attention_trajectory": {},
        }
    )
    for step in joint.REQUIRED_STEPS:
        evaluation = _evaluation(step)
        left["trajectory"][str(step)] = {
            "visual": {
                source: {
                    window: {
                        "gap_wrong_minus_correct": 0.5,
                        "correct_ce": 1.0,
                        "retention_vs_step0": 1.0,
                    }
                    for window in joint.WINDOWS
                }
                for source in joint.VISUAL_SOURCES
            },
            "native_text": {"ce": 2.0, "ppl": math.exp(2.0)},
        }
        left["paired_visual_rows"][str(step)] = evaluation["rows"]
        left["attention_reports"][str(step)] = {
            "arm": "left",
            "step": step,
        }
        if step:
            left["attention_trajectory"][str(step)] = {"collapse_flags": []}
    right = copy.deepcopy(left)
    right["run_id"] = "right"
    right["model_variant"] = "ssmax_no_qknorm"
    for step in joint.REQUIRED_STEPS:
        right["attention_reports"][str(step)]["arm"] = "right"
        for source in joint.VISUAL_SOURCES:
            for row in right["paired_visual_rows"][str(step)][source]:
                for window in joint.WINDOWS:
                    row["correct_ce"][window] = 1.0
                    row["wrong_ce"][window] = 1.5
                    row["ce_gap_wrong_minus_correct"][window] = 0.5
    calls = []

    def compare_attention(left_report: Any, right_report: Any) -> dict[str, Any]:
        calls.append((left_report["step"], right_report["step"]))
        return {
            "left_arm": left_report["arm"],
            "right_arm": right_report["arm"],
            "step": left_report["step"],
        }

    monkeypatch.setattr(joint, "compare_ssmax_attention_reports", compare_attention)

    comparison = joint.compare_trajectory_reports(left, right)

    assert comparison["winner"] is None
    assert comparison["decision_scope"] == "descriptive_non_promotion"
    assert comparison["trajectory_deltas"]["16000"]["native_text"]["ce_delta_left_minus_right"] == 0
    adaptation = comparison["trajectory_deltas"]["16000"]["visual"]["pixmo_caption"]["first_8"][
        "paired_intervals"
    ]
    assert adaptation["gap_same_step_left_minus_right"]["mean"] == pytest.approx(0.16)
    assert adaptation["gap_adaptation_did_left_minus_right"]["mean"] == pytest.approx(0.16)
    assert adaptation["gap_adaptation_did_left_minus_right"]["direction"] == (
        "positive_left_minus_right"
    )
    signal = comparison["adaptation_interval_rule"]["signals"][
        "pixmo_caption/first_8/gap_adaptation_did_left_minus_right"
    ]
    assert signal["consistent_direction"] == "positive_left_minus_right"
    assert comparison["trajectory_deltas"]["0"]["attention"] == {
        "baseline": "left",
        "candidate": "right",
        "comparison": {
            "left_arm": "left",
            "right_arm": "right",
            "step": 0,
        },
    }
    assert calls == [(step, step) for step in joint.REQUIRED_STEPS]
