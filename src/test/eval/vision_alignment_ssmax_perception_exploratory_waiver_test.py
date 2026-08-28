"""Focused tests for the evaluation-complete SSMax exploratory health waiver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from olmo_core.eval import vision_alignment_ssmax_perception_direct as strict
from olmo_core.eval import (
    vision_alignment_ssmax_perception_exploratory_waiver as waiver,
)


def _write_json(path: Path, value: Any) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_checked_in_authorization_is_exact_and_pins_raw_receipts() -> None:
    reference, authorization = waiver._authorization_reference()

    assert reference == {
        "repo_relative_path": waiver.AUTHORIZATION_RELATIVE_PATH,
        "raw_sha256": waiver.AUTHORIZATION_RAW_SHA256,
        "content_sha256": waiver.AUTHORIZATION_CONTENT_SHA256,
    }
    assert authorization["scope"] == waiver.GATE_SCOPE
    assert authorization["strict_v7_preserved"] is True
    assert authorization["strict_v8_preserved"] is True
    assert authorization["promotion_decision"] is False
    assert authorization["winner_selection"] is False
    assert authorization["policy"]["required_health_steps"] == [0]
    assert authorization["policy"]["waived_health_steps"] == [3000, 4000]
    assert (
        "sixteen_rank_trainer_state_replay" in authorization["policy"]["unverified_health_claims"]
    )
    assert (
        "checkpoint_native_health_ledger_event_chain_trajectory"
        in authorization["policy"]["unverified_health_claims"]
    )
    assert (
        authorization["authorized_evidence"]["ssmax_no_qknorm"]["step0_health_receipt_sha256"]
        == "4a358f415e075ec8d004ffe291ad83e7c5d4377804df6f2839e86b5d15ab823f"
    )


def test_only_short_prefix_visual_deviations_can_be_acknowledged() -> None:
    allowed = [
        {
            "kind": "source_nonpositive_absolute_gap",
            "source": "cosyn_point",
            "window": "first_8",
        }
    ]
    assert waiver._authorized_deviations(allowed) == allowed
    assert waiver._authorized_deviations([]) == []

    for forbidden in (
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
    ):
        with pytest.raises(waiver.SSMaxPerceptionExploratoryWaiverEvidenceError):
            waiver._authorized_deviations([forbidden])


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return {
        "created_at": "2026-08-28T03:00:00Z",
        "content_sha256": "2" * 64,
        "run_id": "head-direct-v4",
        "model_variant": "ssmax_head_qknorm",
        "policy": {
            "maximum_data_errors": 0,
            "maximum_optimizer_guard_skips": 8,
            "maximum_nonfinite_losses": 0,
            "maximum_nonfinite_gradients": 0,
        },
        "run": {
            "data_contract_sha256": "3" * 64,
            "trainable_contract_sha256": "4" * 64,
        },
    }


def _authorization_for(
    manifest_raw: str, evaluation_refs: dict[int, dict[str, str]], health_raw: str
) -> dict[str, Any]:
    return {
        "approved_by": "rustins",
        "approved_at": "2026-08-28T04:00:00Z",
        "authorized_evidence": {
            "ssmax_head_qknorm": {
                "manifest_sha256": manifest_raw,
                "manifest_content_sha256": "2" * 64,
                "evaluation_receipt_sha256": {
                    str(step): evaluation_refs[step]["sha256"] for step in strict.REQUIRED_STEPS
                },
                "step0_health_receipt_sha256": health_raw,
            }
        },
    }


def test_report_binds_three_evaluations_but_only_step0_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("manifest\n")
    manifest_raw = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    evaluation_refs = {
        step: _write_json(tmp_path / f"evaluation-{step}.json", {"step": step})
        for step in strict.REQUIRED_STEPS
    }
    health_ref = _write_json(tmp_path / "health-0.json", {"step": 0})
    manifest = _manifest(tmp_path)
    authorization = _authorization_for(manifest_raw, evaluation_refs, health_ref["sha256"])
    candidate = {
        "path": str(tmp_path / "step4000"),
        "config_sha256": "5" * 64,
        "identity_sha256": "6" * 64,
    }
    evaluations = {
        step: {
            "created_at": "2026-08-28T05:00:00Z",
            "status": "passed",
            "state": {},
            "text_sentinel": {},
            "attention_diagnostics": {},
        }
        for step in strict.REQUIRED_STEPS
    }
    health = {"created_at": "2026-08-28T05:00:00Z", "status": "passed"}

    monkeypatch.setattr(waiver, "_validate_admission_or_joint_checkout", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(strict, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(
        waiver,
        "_validate_live_manifest",
        lambda *a, **k: {0: {}, 3000: {}, 4000: candidate},
    )
    monkeypatch.setattr(
        waiver,
        "_authorization_reference",
        lambda *a, **k: (
            {
                "repo_relative_path": waiver.AUTHORIZATION_RELATIVE_PATH,
                "raw_sha256": "7" * 64,
                "content_sha256": "8" * 64,
            },
            authorization,
        ),
    )

    def load_receipt(reference, *, step, expected_format, **kwargs):
        del reference, kwargs
        return tmp_path / "receipt.json", (
            evaluations[step] if expected_format == strict.EVALUATION_RECEIPT_FORMAT else health
        )

    monkeypatch.setattr(strict, "_load_receipt_reference", load_receipt)
    monkeypatch.setattr(strict, "_validate_direct_evaluation_receipt", lambda receipt, **kwargs: {})
    monkeypatch.setattr(
        strict,
        "_validate_direct_health_receipt",
        lambda receipt, **kwargs: {
            "run_counters": {
                "data_errors": 0,
                "optimizer_guard_skips": 0,
                "nonfinite_losses": 0,
                "nonfinite_gradients": 0,
            }
        },
    )
    monkeypatch.setattr(
        waiver,
        "_evaluation_summary",
        lambda **kwargs: (
            {"windows": {}, "attention_trajectory": {}},
            [
                {
                    "kind": "source_nonpositive_absolute_gap",
                    "source": "cosyn_point",
                    "window": "first_8",
                }
            ],
        ),
    )
    monkeypatch.setattr(
        strict,
        "manifest_reference",
        lambda path, value: {
            "path": str(path),
            "sha256": manifest_raw,
            "content_sha256": value["content_sha256"],
        },
    )

    report = waiver.build_evidence_report(
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_raw,
        evaluation_receipts=evaluation_refs,
        health_receipts={0: health_ref},
        created_at="2026-08-28T06:00:00Z",
        admission_git_ref="a" * 40,
    )

    assert set(report) == waiver._REPORT_FIELDS
    assert report["status"] == waiver.REPORT_STATUS
    assert report["promotion_decision"] is False
    assert report["winner_selection"] is False
    assert set(report["receipts"]) == {"0", "3000", "4000"}
    assert set(report["receipts"]["0"]) == {"evaluation", "health"}
    assert set(report["receipts"]["3000"]) == {"evaluation"}
    assert set(report["receipts"]["4000"]) == {"evaluation"}
    assert report["required_waiver"] == waiver._required_waiver()
    assert report["summary"]["step0_health"]["run_counters"]["data_errors"] == 0

    with pytest.raises(
        waiver.SSMaxPerceptionExploratoryWaiverEvidenceError,
        match="predates its manifest or authorization",
    ):
        waiver.build_evidence_report(
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_raw,
            evaluation_receipts=evaluation_refs,
            health_receipts={0: health_ref},
            created_at="2026-08-28T03:30:00Z",
            admission_git_ref="a" * 40,
        )

    with pytest.raises(
        waiver.SSMaxPerceptionExploratoryWaiverEvidenceError,
        match=r"exactly steps \[0\]",
    ):
        waiver.build_evidence_report(
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_raw,
            evaluation_receipts=evaluation_refs,
            health_receipts={0: health_ref, 3000: evaluation_refs[3000]},
            created_at="2026-08-28T06:00:00Z",
            admission_git_ref="a" * 40,
        )


def test_live_manifest_replays_every_strict_live_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance_path = tmp_path / "provenance.json"
    audit_path = tmp_path / "audit.json"
    sentinel_path = tmp_path / "sentinel.json"
    provenance_unsigned = {"format": "provenance"}
    provenance_content = strict.canonical_sha256(provenance_unsigned)
    provenance_payload = {**provenance_unsigned, "content_sha256": provenance_content}
    audit_unsigned = {"format": "audit"}
    audit_fingerprint = strict.canonical_sha256(audit_unsigned)
    audit_payload = {**audit_unsigned, "fingerprint": audit_fingerprint}
    projection = {"calibration": {"sha256": "1" * 64}, "contract": {"version": 1}}
    training_git = {"repo": "repo", "repo_url": "url", "ref": "1" * 40}
    evidence_git = {
        "repo": "repo",
        "repo_url": "url",
        "ref": waiver.ADMISSION_PARENT_GIT_REF,
    }
    training_recipe = {"repo_relative_path": "recipe.py"}
    training_profile = {"repo_relative_path": "profile.yaml"}
    producers = {
        name: {"repo_relative_path": relative}
        for name, relative in strict.PRODUCER_RELATIVE_PATHS.items()
    }
    targets = {source: 1 / len(strict.SOURCES) for source in strict.SOURCES}
    bridge_parent = {"gate": {"path": "/gate", "sha256": "2" * 64}}
    amendment = {"path": "/amendment", "sha256": "3" * 64}
    config = {
        "data": {
            "perception_provenance_path": str(provenance_path),
            "perception_provenance_sha256": "4" * 64,
            "source_audit_path": str(audit_path),
            "source_audit_fingerprint": audit_fingerprint,
        },
        "artifacts": {"tokenizer_id": "tokenizer", "tokenizer_revision": "revision"},
    }
    manifest = {
        "training_git": training_git,
        "evidence_git": evidence_git,
        "producers": producers,
        "training_recipe": training_recipe,
        "run": {
            "training_profile": training_profile,
            "checkpoints": {str(step): {"path": f"/step{step}"} for step in strict.REQUIRED_STEPS},
            "checkpoint_root": str(tmp_path),
            "run_name": "run",
            "data_contract_sha256": "5" * 64,
            "trainable_contract_sha256": "6" * 64,
        },
        "loss_mass_targets": targets,
        "single_response_projection": projection,
        "bridge_parent": bridge_parent,
        "protocol_amendment": amendment,
        "model_variant": "ssmax_head_qknorm",
        "topology": {},
        "policy": {},
        "perception_provenance": {
            "path": str(provenance_path),
            "sha256": "4" * 64,
            "content_sha256": provenance_content,
        },
        "source_audit": {"path": str(audit_path), "sha256": "7" * 64},
        "source_audit_fingerprint": audit_fingerprint,
        "text_sentinel": {"path": str(sentinel_path), "sha256": "8" * 64},
        "attention_probe": {"path": "/probe", "sha256": "9" * 64},
        "pairings": {
            source: {"path": f"/{source}", "sha256": "a" * 64} for source in strict.SOURCES
        },
        "evaluation": {},
    }
    calls: list[str] = []

    class Provenance:
        @staticmethod
        def selection(source: str, split: str):
            assert source in strict.SOURCES
            assert split == "validation"
            return type(
                "Selection",
                (),
                {"indices": [0], "row_image_content_sha256": ["content"]},
            )()

    monkeypatch.setattr(
        waiver,
        "_validate_admission_or_joint_checkout",
        lambda *a, **k: calls.append("checkout"),
    )
    monkeypatch.setattr(strict, "_git_identity", lambda value, **kwargs: value)
    monkeypatch.setattr(
        strict,
        "validate_evidence_git_compatibility",
        lambda **kwargs: calls.append("git_compatibility"),
    )
    monkeypatch.setattr(strict, "_producer_references", lambda *a, **k: producers)
    monkeypatch.setattr(strict, "_source_reference", lambda value, **kwargs: value)
    monkeypatch.setattr(strict, "_exact", lambda value, fields, **kwargs: value)
    monkeypatch.setattr(
        strict.paired, "_validate_single_response_binding", lambda *a, **k: projection
    )

    def amendment_reference(value, **kwargs):
        del kwargs
        calls.append("amendment")
        return value

    monkeypatch.setattr(strict, "_authorized_amendment_reference", amendment_reference)
    monkeypatch.setattr(
        strict,
        "_checkpoint_reference",
        lambda value, **kwargs: {**value, "config_sha256": "b" * 64},
    )

    def git_blob(*, git, repo_relative_path, **kwargs):
        del kwargs
        if git == training_git:
            return training_recipe if repo_relative_path == "recipe.py" else training_profile
        return {"repo_relative_path": repo_relative_path}

    monkeypatch.setattr(strict, "_git_blob_reference", git_blob)
    monkeypatch.setattr(strict, "_mapping", lambda value, **kwargs: value)

    def load_json(path: Path):
        if path.name == "config.json":
            return config
        if path == provenance_path:
            return provenance_payload
        if path == audit_path:
            return audit_payload
        raise AssertionError(path)

    monkeypatch.setattr(strict, "load_json", load_json)
    monkeypatch.setattr(
        strict,
        "validate_saved_config",
        lambda *a, **k: {
            "data_contract_sha256": "5" * 64,
            "trainable_contract_sha256": "6" * 64,
            "loss_mass_targets": targets,
        },
    )
    monkeypatch.setattr(
        strict.paired, "_single_response_binding_from_config", lambda value: projection
    )
    monkeypatch.setattr(
        strict.paired,
        "_validate_calibration_git_blobs",
        lambda *a, **k: calls.append("calibration"),
    )
    monkeypatch.setattr(
        strict.paired,
        "_validate_bridge_parent",
        lambda *a, **k: (calls.append("bridge_parent"), bridge_parent)[1],
    )

    def artifact_reference(value, *, name):
        if name == "direct perception provenance":
            return provenance_path
        if name == "direct source audit":
            return audit_path
        if name == "direct native text sentinel":
            return sentinel_path
        raise AssertionError(name)

    monkeypatch.setattr(strict.paired, "validate_artifact_reference", artifact_reference)
    monkeypatch.setattr(
        strict.paired,
        "load_perception_provenance_manifest",
        lambda *a, **k: (calls.append("provenance_schema"), Provenance())[1],
    )
    monkeypatch.setattr(
        strict.paired,
        "_validate_text_sentinel",
        lambda path: {"tokenizer": {"identifier": "tokenizer", "revision": "revision"}},
    )
    monkeypatch.setattr(
        strict.paired,
        "_validate_attention_probe_reference",
        lambda *a, **k: calls.append("attention_probe"),
    )
    monkeypatch.setattr(strict.paired, "content_ids_sha256", lambda value: "content-ids")
    monkeypatch.setattr(
        strict.paired,
        "_validate_pairing_reference",
        lambda *a, **k: calls.append("pairing"),
    )

    checkpoints = waiver._validate_live_manifest(
        manifest,
        admission_git_ref="c" * 40,
        repository_root=tmp_path,
    )

    assert set(checkpoints) == set(strict.REQUIRED_STEPS)
    assert calls.count("checkout") == 1
    assert calls.count("git_compatibility") == 1
    assert calls.count("calibration") == 1
    assert calls.count("bridge_parent") == 1
    assert calls.count("provenance_schema") == 1
    assert calls.count("attention_probe") == 1
    assert calls.count("pairing") == len(strict.SOURCES)
    assert calls.count("amendment") == 2


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
    required = waiver._required_waiver()
    report = {
        "created_at": "2026-08-28T09:00:00Z",
        "content_sha256": "3" * 64,
        "receipts": {
            "0": {"evaluation": {}, "health": {}},
            "3000": {"evaluation": {}},
            "4000": {"evaluation": {}},
        },
        "acknowledged_deviations": [],
        "required_waiver": required,
        "admission_git_ref": "4" * 40,
    }
    manifest = {
        "run_id": "head-direct-v4",
        "model_variant": "ssmax_head_qknorm",
        "run": {
            "data_contract_sha256": "5" * 64,
            "trainable_contract_sha256": "6" * 64,
        },
        "protocol_amendment": {
            "path": str(tmp_path / "amendment.json"),
            "sha256": "7" * 64,
            "content_sha256": "8" * 64,
        },
        "training_git": {"ref": "9" * 40},
        "evidence_git": {"ref": "a" * 40},
    }
    return {
        "report": report,
        "report_reference": {"path": str(tmp_path / "report.json"), "sha256": "b" * 64},
        "manifest": manifest,
        "manifest_reference": {
            "path": str(tmp_path / "manifest.json"),
            "sha256": "c" * 64,
            "content_sha256": "d" * 64,
        },
        "candidate": candidate,
        "authorization": {},
        "acknowledged_deviations": [],
    }


def test_gate_records_one_content_bound_waiver_and_no_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _gate_summary(tmp_path)
    authorization = {
        "approved_by": "rustins",
        "approved_at": "2026-08-28T08:52:37Z",
    }
    monkeypatch.setattr(waiver, "validate_evidence_report_reference", lambda *a, **k: summary)
    monkeypatch.setattr(
        waiver,
        "_authorization_reference",
        lambda *a, **k: (
            {
                "repo_relative_path": waiver.AUTHORIZATION_RELATIVE_PATH,
                "raw_sha256": "e" * 64,
                "content_sha256": "f" * 64,
            },
            authorization,
        ),
    )
    monkeypatch.setattr(
        waiver,
        "validate_ssmax_perception_exploratory_waiver_parent_gate",
        lambda *a, **k: summary,
    )

    gate = waiver.build_parent_gate(
        evidence_report_path=tmp_path / "report.json",
        expected_evidence_report_sha256="b" * 64,
        approved_by="rustins",
        approved_at="2026-08-28T10:00:00Z",
    )

    assert set(gate) == waiver._GATE_FIELDS
    assert gate["version"] == 9
    assert gate["scope"] == waiver.GATE_SCOPE
    assert gate["evidence_report_status"] == waiver.REPORT_STATUS
    assert gate["promotion_decision"] is False
    assert gate["winner_selection"] is False
    assert gate["waivers"] == waiver._approved_waivers(summary["report"]["required_waiver"])
    assert gate["waivers"][0]["deviation_sha256"] == strict.canonical_sha256(
        summary["report"]["required_waiver"]
    )


def test_report_and_gate_refuse_disabled_live_verification() -> None:
    with pytest.raises(
        waiver.SSMaxPerceptionExploratoryWaiverEvidenceError,
        match="requires live checkpoint",
    ):
        waiver.validate_evidence_report_reference({}, verify_live_checkpoint=False)
    with pytest.raises(
        waiver.SSMaxPerceptionExploratoryWaiverEvidenceError,
        match="requires live checkpoint",
    ):
        waiver.validate_ssmax_perception_exploratory_waiver_parent_gate(
            {},
            expected_checkpoint=Path("/tmp/step4000"),
            expected_checkpoint_config_sha256="0" * 64,
            expected_model_variant="ssmax_head_qknorm",
            expected_data_contract_sha256="1" * 64,
            expected_trainable_contract_sha256="2" * 64,
            verify_live_checkpoint=False,
        )


def test_cli_requires_exact_evaluation_and_health_step_sets() -> None:
    from scripts.eval import vision_alignment_ssmax_perception_exploratory_waiver as cli

    assert cli._values(
        ["0=a", "3000=b", "4000=c"],
        option="--evaluation",
        required_steps=waiver.REQUIRED_EVALUATION_STEPS,
    ) == {0: "a", 3000: "b", 4000: "c"}
    assert cli._values(["0=a"], option="--health", required_steps=waiver.REQUIRED_HEALTH_STEPS) == {
        0: "a"
    }
    with pytest.raises(ValueError, match="selector is invalid"):
        cli._values(
            ["0=a", "3000=b"],
            option="--health",
            required_steps=waiver.REQUIRED_HEALTH_STEPS,
        )
