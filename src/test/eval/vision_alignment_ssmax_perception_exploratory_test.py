"""Focused tests for exploratory-only SSMax perception admission."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from olmo_core.eval import vision_alignment_ssmax_perception_direct as strict
from olmo_core.eval import vision_alignment_ssmax_perception_exploratory as exploratory


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _allowed_deviations() -> list[dict[str, str]]:
    return [
        {
            "kind": "source_nonpositive_absolute_gap",
            "source": "cosyn_point",
            "window": "first_8",
        },
        {
            "kind": "source_nonpositive_gap_improvement",
            "source": "pixmo_transcript",
            "window": "first_1",
        },
        {
            "kind": "source_gap_retention",
            "source": "scalar_count",
            "window": "first_1",
        },
    ]


def test_checked_in_authorization_has_exact_raw_and_semantic_identity() -> None:
    reference, authorization = exploratory._authorization_reference()

    assert reference == {
        "repo_relative_path": exploratory.AUTHORIZATION_RELATIVE_PATH,
        "raw_sha256": exploratory.AUTHORIZATION_RAW_SHA256,
        "content_sha256": exploratory.AUTHORIZATION_CONTENT_SHA256,
    }
    assert authorization["approved_by"] == "rustins"
    assert authorization["scope"] == exploratory.GATE_SCOPE
    assert authorization["strict_v7_preserved"] is True
    assert authorization["promotion_decision"] is False
    assert authorization["winner_selection"] is False


def test_deviation_boundary_accepts_only_short_prefix_source_visual_misses() -> None:
    assert (
        exploratory._validate_exploratory_deviations(_allowed_deviations()) == _allowed_deviations()
    )

    forbidden = (
        {"kind": "macro_absolute_gap_lower_ci", "window": "first_1"},
        {
            "kind": "source_nonpositive_absolute_gap",
            "source": "cosyn_point",
            "window": "first_32",
        },
        {
            "kind": "source_correct_ce_regression",
            "source": "cosyn_point",
            "window": "first_1",
        },
        {"kind": "run_health_counter", "step": 4000},
    )
    for deviation in forbidden:
        with pytest.raises(exploratory.SSMaxPerceptionExploratoryEvidenceError):
            exploratory._validate_exploratory_deviations([deviation])

    with pytest.raises(
        exploratory.SSMaxPerceptionExploratoryEvidenceError,
        match="requires a rejected strict report",
    ):
        exploratory._validate_exploratory_deviations([])


def test_audit_exactly_rebuilds_a_rejected_strict_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("manifest\n")
    manifest_raw_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = {
        "content_sha256": "b" * 64,
        "run_id": "head-direct-v3",
        "model_variant": "ssmax_head_qknorm",
        "run": {
            "checkpoints": {"4000": {"path": str(tmp_path / "step4000")}},
        },
    }
    receipts = {
        str(step): {
            kind: {"path": str(tmp_path / f"{step}-{kind}.json"), "sha256": "c" * 64}
            for kind in ("evaluation", "health")
        }
        for step in strict.REQUIRED_STEPS
    }
    report = {
        "format": strict.PROMOTION_REPORT_FORMAT,
        "version": strict.SCHEMA_VERSION,
        "status": "rejected",
        "decision_scope": "within_lineage_noncausal_joint_admission",
        "created_at": "2026-08-28T03:00:00Z",
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_raw_sha,
            "content_sha256": "b" * 64,
        },
        "run_id": "head-direct-v3",
        "model_variant": "ssmax_head_qknorm",
        "receipts": receipts,
        "summary": {"windows": {}},
        "deviations": _allowed_deviations(),
    }
    report["content_sha256"] = strict.canonical_sha256(report)
    report_path = tmp_path / "report.json"
    _write_json(report_path, report)
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    candidate = {
        "path": str(tmp_path / "step4000"),
        "config_sha256": "d" * 64,
        "identity_sha256": "e" * 64,
    }

    monkeypatch.setattr(strict, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(strict, "_checkpoint_reference", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(strict, "build_promotion_report", lambda **kwargs: report)

    summary = exploratory.audit_strict_report_reference(
        {"path": str(report_path), "sha256": report_sha}
    )

    assert summary["strict_report"] == report
    assert summary["candidate"] == candidate
    assert summary["acknowledged_deviations"] == _allowed_deviations()


def _gate_summary(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "step4000"
    checkpoint.mkdir()
    _write_json(
        checkpoint / "config.json",
        {
            "vision_alignment": {
                "recipe_version": 1,
                "formatter_version": "vision-alignment-document-v1",
            }
        },
    )
    candidate = {
        "path": str(checkpoint),
        "config_sha256": "1" * 64,
        "identity_sha256": "2" * 64,
    }
    manifest = {
        "run_id": "head-direct-v3",
        "model_variant": "ssmax_head_qknorm",
        "run": {
            "data_contract_sha256": "3" * 64,
            "trainable_contract_sha256": "4" * 64,
        },
        "protocol_amendment": {
            "path": str(tmp_path / "amendment.json"),
            "sha256": "5" * 64,
            "content_sha256": "6" * 64,
        },
        "training_git": {"ref": "7" * 40},
        "evidence_git": {"ref": "8" * 40},
    }
    report = {
        "created_at": "2026-08-28T03:00:00Z",
        "content_sha256": "9" * 64,
        "receipts": {str(step): {} for step in strict.REQUIRED_STEPS},
    }
    return {
        "strict_report": report,
        "strict_report_reference": {
            "path": str(tmp_path / "report.json"),
            "sha256": "a" * 64,
        },
        "manifest": manifest,
        "manifest_reference": {
            "path": str(tmp_path / "manifest.json"),
            "sha256": "b" * 64,
            "content_sha256": "c" * 64,
        },
        "candidate": candidate,
        "acknowledged_deviations": _allowed_deviations(),
    }


def test_gate_binds_rejected_report_receipts_authorization_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _gate_summary(tmp_path)
    monkeypatch.setattr(
        exploratory,
        "audit_strict_report_reference",
        lambda *args, **kwargs: summary,
    )
    gate = exploratory.build_parent_gate(
        strict_report_path=tmp_path / "report.json",
        expected_strict_report_sha256="a" * 64,
        approved_by="rustins",
        approved_at="2026-08-28T04:00:00Z",
    )

    assert set(gate) == exploratory._GATE_FIELDS
    assert gate["version"] == 8
    assert gate["scope"] == "exploratory_joint_only"
    assert gate["strict_report_status"] == "rejected"
    assert gate["strict_receipts"] == summary["strict_report"]["receipts"]
    assert gate["acknowledged_deviations"] == _allowed_deviations()
    assert gate["authorization"]["raw_sha256"] == exploratory.AUTHORIZATION_RAW_SHA256

    tampered = copy.deepcopy(gate)
    tampered["acknowledged_deviations"] = tampered["acknowledged_deviations"][:-1]
    with pytest.raises(
        exploratory.SSMaxPerceptionExploratoryEvidenceError,
        match="binding differs",
    ):
        exploratory.validate_ssmax_perception_exploratory_parent_gate(
            tampered,
            expected_checkpoint=Path(summary["candidate"]["path"]),
            expected_checkpoint_config_sha256="1" * 64,
            expected_model_variant="ssmax_head_qknorm",
            expected_data_contract_sha256="3" * 64,
            expected_trainable_contract_sha256="4" * 64,
        )
