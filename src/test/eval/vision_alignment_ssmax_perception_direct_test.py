from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from olmo_core.eval import vision_alignment_ssmax_perception_direct as direct

_HASH = "a" * 64
_MANIFEST_TIME = "2026-08-22T02:00:00+00:00"
_RECEIPT_TIME = "2026-08-22T02:15:00+00:00"
_REPORT_TIME = "2026-08-22T03:00:00+00:00"


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return direct.artifact_reference(path)


def _git(ref: str) -> dict[str, str]:
    return {
        "repo": "allenai/OLMo-core",
        "repo_url": "https://github.com/allenai/OLMo-core",
        "branch": direct.TRAINING_GIT_BRANCH,
        "ref": ref,
    }


def _spec() -> dict[str, Any]:
    identity = direct.DIRECT_RUN_IDENTITIES["ssmax_head_qknorm"]
    return {
        "format": direct.MANIFEST_SPEC_FORMAT,
        "version": direct.SCHEMA_VERSION,
        "run_id": "head-direct-perception-v1",
        "model_variant": "ssmax_head_qknorm",
        "run_name": identity["run_name"],
        "checkpoint_root": identity["checkpoint_root"],
        "training_profile": identity["profile"],
        "recipe": "src/scripts/train/Vision-Alignment.py",
        "training_git": _git(direct.TRAINING_GIT_REF),
        "evidence_git": _git("b" * 40),
        "protocol_amendment": {
            "path": direct.AMENDMENT_RELATIVE_PATH,
            "sha256": direct.AMENDMENT_SHA256,
        },
        "bridge_parent_gate": "/evidence/bridge-parent-gate.json",
        "perception_provenance": "/evidence/perception-provenance.json",
        "source_audit": "/evidence/source-audit.json",
        "attention_probe": "/evidence/attention-probe.json",
        "text_sentinel": "/evidence/text-sentinel.json",
        "pairing_paths": {source: f"/evidence/{source}-pairing.json" for source in direct.SOURCES},
        "evaluation": copy.deepcopy(direct.EVALUATION_CONTRACT),
        "topology": copy.deepcopy(direct.TOPOLOGY_CONTRACT),
        "policy": copy.deepcopy(direct.DIRECT_POLICY),
    }


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_final_amendment_is_exactly_pinned_and_authorizes_no_controls() -> None:
    repository_root = direct._repository_root()
    path = repository_root / direct.AMENDMENT_RELATIVE_PATH
    assert direct.sha256_file(path) == direct.AMENDMENT_SHA256
    amendment = direct._validate_amendment_payload(direct.load_json(path))
    assert amendment["approved_by"] == "rustins"
    assert amendment["control_runs_required"] is False
    assert amendment["excluded_control_runs"] == {
        "ssmax_head_qknorm": {
            "checkpoint_root_created": False,
            "experiment_id": "01M0KBV398GTYAK7VZ9FMMHS8X",
            "status": "canceled_while_starting",
        },
        "ssmax_no_qknorm": {
            "checkpoint_root_created": False,
            "experiment_id": None,
            "status": "not_launched",
        },
    }

    changed = copy.deepcopy(amendment)
    changed["excluded_control_runs"]["ssmax_no_qknorm"]["checkpoint_root_created"] = 0
    with pytest.raises(
        direct.SSMaxPerceptionDirectEvidenceError, match="checkpoint-root disposition"
    ):
        direct._validate_amendment_payload(changed)

    changed = copy.deepcopy(amendment)
    changed["evidence_policy"][
        "candidate_visual_gap_improvement_vs_step0_ci_low_must_be_positive"
    ] = 1
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="must be boolean"):
        direct._validate_amendment_payload(changed)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("version",), True, "incompatible"),
        (("evaluation", "examples_per_source"), 480.0, "evaluation examples_per_source"),
        (("topology", "world_size"), 16.0, "topology world_size"),
        (("policy", "require_finite_gradient_only_skips"), 1, "must be boolean"),
        (("policy", "optimizer_guard", "rolling_interval_length"), 128.0, "non-canonical"),
        (("policy", "optimizer_guard", "max_grad_norm"), 1, "non-canonical"),
        (("run_name",), "unauthorized-run", "unauthorized"),
    ],
)
def test_manifest_spec_is_exact_and_rejects_json_type_aliases(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    spec = _spec()
    direct.validate_manifest_spec(spec)
    _set_nested(spec, path, value)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match=match):
        direct.validate_manifest_spec(spec)


def test_manifest_spec_requires_one_repository_for_training_and_evidence() -> None:
    spec = _spec()
    spec["evidence_git"]["repo_url"] = "https://example.invalid/other.git"
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="different repositories"):
        direct.validate_manifest_spec(spec)


def test_evidence_git_diff_is_restricted_to_additive_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = _git(direct.TRAINING_GIT_REF)
    evidence = _git("b" * 40)
    protocol = "src/olmo_core/eval/vision_alignment_ssmax_perception_direct.py"

    diff_output = [f"A\t{protocol}\nA\t{direct.AMENDMENT_RELATIVE_PATH}\n"]

    def output(command: list[str], **kwargs: Any) -> str:
        del kwargs
        if "rev-list" in command:
            return f"{evidence['ref']} {training['ref']}\n"
        if "diff" in command:
            return diff_output[0]
        raise AssertionError(command)

    monkeypatch.setattr(direct.subprocess, "check_output", output)
    assert direct.validate_evidence_git_compatibility(
        training_git=training, evidence_git=evidence, repository_root=tmp_path
    ) == (("A", protocol), ("A", direct.AMENDMENT_RELATIVE_PATH))

    diff_output[0] = f"A\t{protocol}\nM\tsrc/olmo_core/nn/transformer/config.py\n"
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="unauthorized change"):
        direct.validate_evidence_git_compatibility(
            training_git=training, evidence_git=evidence, repository_root=tmp_path
        )

    diff_output[0] = f"A\t{protocol}\nD\t{direct.AMENDMENT_RELATIVE_PATH}\n"
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="unauthorized change"):
        direct.validate_evidence_git_compatibility(
            training_git=training, evidence_git=evidence, repository_root=tmp_path
        )

    def wrong_parent(command: list[str], **kwargs: Any) -> str:
        if "rev-list" in command:
            return f"{evidence['ref']} {'d' * 40}\n"
        return output(command, **kwargs)

    monkeypatch.setattr(direct.subprocess, "check_output", wrong_parent)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="directly after"):
        direct.validate_evidence_git_compatibility(
            training_git=training, evidence_git=evidence, repository_root=tmp_path
        )


def test_joint_consumer_checkout_allows_only_the_exact_profile_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _git("b" * 40)
    descendant = "c" * 40
    diff = "".join(f"{status}\t{path}\n" for status, path in direct.JOINT_CONSUMER_GIT_DIFF)

    monkeypatch.setattr(
        direct.subprocess,
        "check_output",
        lambda command, **kwargs: (
            evidence["ref"] + "\n" if command[-2:] == ["rev-parse", "HEAD"] else ""
        ),
    )
    assert (
        direct._validate_evidence_or_joint_consumer_checkout(
            evidence_git=evidence,
            repository_root=tmp_path,
        )
        == ()
    )

    monkeypatch.setattr(
        direct.subprocess,
        "check_output",
        lambda command, **kwargs: (
            evidence["ref"] + "\n" if command[-2:] == ["rev-parse", "HEAD"] else "M uncommitted\n"
        ),
    )
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="not clean"):
        direct._validate_evidence_or_joint_consumer_checkout(
            evidence_git=evidence,
            repository_root=tmp_path,
        )

    wrong_repository = dict(evidence)
    wrong_repository["repo"] = "other/repository"
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="different repository"):
        direct._validate_evidence_or_joint_consumer_checkout(
            evidence_git=wrong_repository,
            repository_root=tmp_path,
        )

    def output(command: list[str], **kwargs: Any) -> str:
        del kwargs
        if command[-2:] == ["rev-parse", "HEAD"]:
            return descendant + "\n"
        if command[-2:] == ["status", "--porcelain"]:
            return ""
        if "rev-list" in command:
            return f"{descendant} {evidence['ref']}\n"
        if "diff" in command:
            return diff
        raise AssertionError(command)

    monkeypatch.setattr(direct.subprocess, "check_output", output)
    monkeypatch.setattr(direct, "_validate_joint_consumer_profiles", lambda root: None)
    rows = direct._validate_evidence_or_joint_consumer_checkout(
        evidence_git=evidence,
        repository_root=tmp_path,
    )
    assert frozenset(rows) == direct.JOINT_CONSUMER_GIT_DIFF

    def output_with_extra(command: list[str], **kwargs: Any) -> str:
        value = output(command, **kwargs)
        return value + "M\tsrc/scripts/train/Vision-Alignment.py\n" if "diff" in command else value

    monkeypatch.setattr(direct.subprocess, "check_output", output_with_extra)
    with pytest.raises(
        direct.SSMaxPerceptionDirectEvidenceError, match="joint-consumer diff differs"
    ):
        direct._validate_evidence_or_joint_consumer_checkout(
            evidence_git=evidence,
            repository_root=tmp_path,
        )

    def intervening_parent(command: list[str], **kwargs: Any) -> str:
        if "rev-list" in command:
            return f"{descendant} {'d' * 40}\n"
        return output(command, **kwargs)

    monkeypatch.setattr(direct.subprocess, "check_output", intervening_parent)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="directly after"):
        direct._validate_evidence_or_joint_consumer_checkout(
            evidence_git=evidence,
            repository_root=tmp_path,
        )


def test_joint_consumer_profile_allowlist_preserves_legacy_and_hashes_new_profiles(
    tmp_path: Path,
) -> None:
    profile_hashes: dict[str, str] = {}
    for index, path in enumerate(sorted(direct.JOINT_DIRECT_PROFILE_PATHS)):
        profile_path = tmp_path / path
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(f"version: 1\nname: direct-{index}\n")
        profile_hashes[path] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    allowlist = {
        "format": "vision_alignment_joint_profile_allowlist",
        "profiles": {**direct.JOINT_PROFILE_BASELINE, **profile_hashes},
        "version": 1,
    }
    allowlist_path = tmp_path / direct.JOINT_PROFILE_ALLOWLIST_RELATIVE_PATH
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text(
        json.dumps(allowlist, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    direct._validate_joint_consumer_profiles(tmp_path)

    changed = copy.deepcopy(allowlist)
    changed["profiles"][next(iter(direct.JOINT_PROFILE_BASELINE))] = "f" * 64
    allowlist_path.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="legacy entries"):
        direct._validate_joint_consumer_profiles(tmp_path)

    allowlist_path.write_text(
        json.dumps(allowlist, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    (tmp_path / min(direct.JOINT_DIRECT_PROFILE_PATHS)).write_text("changed\n")
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="profile bytes differ"):
        direct._validate_joint_consumer_profiles(tmp_path)


def test_direct_json_helpers_keep_the_direct_error_boundary(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"duplicate": 1, "duplicate": 2}\n')
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError):
        direct.load_json(malformed)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError):
        direct.artifact_reference(tmp_path / "absent.json")


def test_receipt_checkpoint_and_health_rank_cursors_reject_numeric_aliases(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_fixture(tmp_path)
    checkpoint = copy.deepcopy(manifest["run"]["checkpoints"]["3000"])
    checkpoint["global_step"] = 3000.0
    receipt: dict[str, Any] = {
        "format": direct.EVALUATION_RECEIPT_FORMAT,
        "version": direct.SCHEMA_VERSION,
        "status": "passed",
        "created_at": _RECEIPT_TIME,
        "manifest": direct.manifest_reference(manifest_path, manifest),
        "run_id": manifest["run_id"],
        "model_variant": manifest["model_variant"],
        "step": 3000,
        "checkpoint": checkpoint,
        "strict_generic_dcp_load": {},
        "state": {},
        "text_sentinel": {},
        "attention_diagnostics": {},
        "pairings": {},
        "results": {},
        "evaluator": {},
    }
    receipt["content_sha256"] = direct.canonical_sha256(receipt)
    receipt_ref = _write_json(tmp_path / "aliased-receipt.json", receipt)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="global_step"):
        direct._load_receipt_reference(
            receipt_ref,
            manifest=manifest,
            manifest_path=manifest_path,
            step=3000,
            expected_format=direct.EVALUATION_RECEIPT_FORMAT,
        )

    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="must be an integer"):
        direct._validate_direct_health_receipt(
            {"rank_states": [{"rank": 0.0, "global_step": 3000, "batches_processed": 3000}]},
            manifest={"topology": {"world_size": 1}},
            step=3000,
        )


def _checkpoint(tmp_path: Path, step: int) -> dict[str, Any]:
    path = tmp_path / "run" / f"step{step}"
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "path": str(path.resolve()),
        "global_step": step,
        "config_sha256": "1" * 64,
        "marker_sha256": "2" * 64,
        "dcp_metadata_sha256": "3" * 64,
        "state_file_count": 2,
        "state_file_inventory_sha256": "4" * 64,
        "trainer_state_count": 16,
        "trainer_state_inventory_sha256": "5" * 64,
    }
    payload["identity_sha256"] = direct.canonical_sha256(payload)
    return payload


def _manifest_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n")
    amendment = direct._validate_amendment_payload(
        direct.load_json(direct._repository_root() / direct.AMENDMENT_RELATIVE_PATH)
    )
    manifest: dict[str, Any] = {
        "created_at": _MANIFEST_TIME,
        "run_id": "head-direct-perception-v1",
        "model_variant": "ssmax_head_qknorm",
        "content_sha256": "c" * 64,
        "training_git": _git(direct.TRAINING_GIT_REF),
        "evidence_git": _git("b" * 40),
        "protocol_amendment": {
            "path": direct.AMENDMENT_RELATIVE_PATH,
            "sha256": direct.AMENDMENT_SHA256,
            "content_sha256": direct.canonical_sha256(amendment),
        },
        "policy": copy.deepcopy(direct.DIRECT_POLICY),
        "evaluation": {
            **copy.deepcopy(direct.EVALUATION_CONTRACT),
            "examples_per_source": 2,
            "bootstrap_samples": 64,
        },
        "loss_mass_targets": {source: 1 / len(direct.SOURCES) for source in direct.SOURCES},
        "run": {
            "run_name": direct.DIRECT_RUN_IDENTITIES["ssmax_head_qknorm"]["run_name"],
            "data_contract_sha256": "d" * 64,
            "trainable_contract_sha256": "e" * 64,
            "checkpoints": {
                str(step): _checkpoint(tmp_path, step) for step in direct.REQUIRED_STEPS
            },
        },
    }
    return manifest_path, manifest


def _surface(sha: str) -> dict[str, Any]:
    return {
        "mismatch_count": 0,
        "reference_inventory_sha256": sha,
        "candidate_inventory_sha256": sha,
    }


def _text_output_descriptor(sha: str, *, output: str) -> dict[str, Any]:
    shape = [1, 256, 100352] if output == "logits" else [1, 256]
    return {
        "dtype": "torch.bfloat16" if output == "logits" else "torch.float32",
        "shape": shape,
        "numel": int(np.prod(shape)),
        "sha256": sha,
        "finite": True,
    }


def _text_sentinel_payload(step: int) -> dict[str, Any]:
    rows = []
    for rank in range(direct.TOPOLOGY_CONTRACT["world_size"]):
        logits_sha = f"{10_000 + step + rank:064x}"
        ce_sha = f"{20_000 + step + rank:064x}"
        reference = {
            "logits": _text_output_descriptor(logits_sha, output="logits"),
            "ce": _text_output_descriptor(ce_sha, output="ce"),
        }
        rows.append(
            {
                "rank": rank,
                "reference": reference,
                "candidate": copy.deepcopy(reference),
                "logits_exact": True,
                "ce_exact": True,
                "passed": True,
            }
        )
    payload: dict[str, Any] = {
        "protocol": direct.DIRECT_TEXT_SENTINEL_PROTOCOL,
        "version": 1,
        "artifact_sha256": "6" * 64,
        "reference_step": 0,
        "reference_checkpoint_identity_sha256": "7" * 64,
        "candidate_step": step,
        "candidate_checkpoint_identity_sha256": f"{30_000 + step:064x}",
        "topology": copy.deepcopy(direct.TOPOLOGY_CONTRACT),
        "world_size": direct.TOPOLOGY_CONTRACT["world_size"],
        "input": {
            "dtype": "torch.int64",
            "shape": [1, 256],
            "numel": 256,
            "sha256": "8" * 64,
        },
        "labels": {
            "dtype": "torch.int64",
            "shape": [1, 256],
            "numel": 256,
            "sha256": "9" * 64,
        },
        "token_count": 256,
        "rank_count": direct.TOPOLOGY_CONTRACT["world_size"],
        "rank_rows": rows,
        "mismatch_count": 0,
        "all_ranks_passed": True,
        "rank_inventory_sha256": direct.canonical_sha256(rows),
    }
    payload["content_sha256"] = direct.canonical_sha256(payload)
    return payload


def _rows(*, correct_ce: float, gap: float) -> list[dict[str, Any]]:
    rows = []
    for position in range(2):
        correct = {window: correct_ce + position * 0.001 for window in direct.WINDOWS}
        rows.append(
            {
                "pairing_position": position,
                "recipient_index": position,
                "donor_index": 1 - position,
                "response_tokens": 32,
                "correct_ce": correct,
                "wrong_ce": {window: correct[window] + gap for window in direct.WINDOWS},
                "ce_gap_wrong_minus_correct": {window: gap for window in direct.WINDOWS},
            }
        )
    return rows


def _evaluation_payload(step: int) -> dict[str, Any]:
    return {
        "status": "passed",
        "created_at": _RECEIPT_TIME,
        "state": {
            "frozen_lm": _surface("6" * 64),
            "non_image_embedding_rows": _surface("7" * 64),
        },
        "text_sentinel": _text_sentinel_payload(step),
        "attention_diagnostics": {"report_sha256": f"{step:064x}"},
    }


def _health_summary() -> dict[str, Any]:
    source_values = {
        source: {"loss_weight": 1.0, "active_loss_weight": 1.0} for source in direct.SOURCES
    }
    return {
        "run_counters": {
            "data_errors": 0,
            "optimizer_guard_skips": 0,
            "nonfinite_losses": 0,
            "nonfinite_gradients": 0,
        },
        "total_loss_weight": float(len(direct.SOURCES)),
        "total_active_loss_weight": float(len(direct.SOURCES)),
        "sources": source_values,
        "optimizer_guard": {
            "optimizer_guard_history_reset_steps": [0],
            "required_optimizer_guard_history_reset_steps": [0],
            "skip_events": [],
            "minimum_step_distance": None,
            "required_minimum_step_distance": 129,
            "clean_final_steps": 4000,
            "required_clean_final_steps": 128,
            "resume_free_passed": True,
            "spacing_passed": True,
            "final_window_passed": True,
        },
    }


def _build_report_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    step0_gap: float = 0.1,
    step3000_gap: float = 0.5,
    step4000_gap: float = 0.45,
    step4000_correct_ce: float = 1.01,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path, manifest = _manifest_fixture(tmp_path)
    evaluations = {
        0: {source: _rows(correct_ce=1.0, gap=step0_gap) for source in direct.SOURCES},
        3000: {source: _rows(correct_ce=1.0, gap=step3000_gap) for source in direct.SOURCES},
        4000: {
            source: _rows(correct_ce=step4000_correct_ce, gap=step4000_gap)
            for source in direct.SOURCES
        },
    }
    evaluation_payloads = {step: _evaluation_payload(step) for step in direct.REQUIRED_STEPS}
    health_payloads = {
        step: {"status": "passed", "created_at": _RECEIPT_TIME} for step in direct.REQUIRED_STEPS
    }
    monkeypatch.setattr(direct, "load_manifest", lambda *args, **kwargs: manifest)

    def load_receipt(
        reference: Any,
        *,
        manifest: Mapping[str, Any],
        manifest_path: Path,
        step: int,
        expected_format: str,
    ) -> tuple[Path, Mapping[str, Any]]:
        del reference, manifest, manifest_path
        payload = (
            evaluation_payloads[step]
            if expected_format == direct.EVALUATION_RECEIPT_FORMAT
            else health_payloads[step]
        )
        return tmp_path / f"step{step}.json", payload

    monkeypatch.setattr(direct, "_load_receipt_reference", load_receipt)
    monkeypatch.setattr(
        direct,
        "_validate_direct_evaluation_receipt",
        lambda receipt, *, manifest, step: evaluations[step],
    )
    monkeypatch.setattr(
        direct,
        "_validate_direct_health_receipt",
        lambda receipt, *, manifest, step: _health_summary(),
    )
    monkeypatch.setattr(
        direct,
        "compare_ssmax_attention_reports",
        lambda baseline, candidate: {"status": "comparable"},
    )
    references = {
        step: {"path": str(tmp_path / f"receipt-{step}.json"), "sha256": _HASH}
        for step in direct.REQUIRED_STEPS
    }
    report = direct.build_promotion_report(
        manifest_path=manifest_path,
        evaluation_receipts=references,
        health_receipts=references,
        created_at=_REPORT_TIME,
        verify_live_manifest=False,
    )
    return report, manifest_path, manifest


def test_direct_report_requires_six_receipts_and_passes_all_four_visual_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _, _ = _build_report_fixture(tmp_path, monkeypatch)
    assert report["status"] == "passed"
    assert report["decision_scope"] == "within_lineage_noncausal_joint_admission"
    assert report["deviations"] == []
    assert set(report["receipts"]) == {"0", "3000", "4000"}
    for window in direct.WINDOWS:
        summary = report["summary"]["windows"][window]
        assert summary["candidate_absolute_gap"]["ci"]["low"] > 0
        assert summary["candidate_gap_improvement_from_step0"]["ci"]["low"] > 0
        assert summary["macro_step4000_correct_ce"] <= 1.02 * summary["macro_step0_correct_ce"]
        assert summary["macro_step4000_gap"] >= 0.8 * summary["macro_step3000_gap"]
        assert summary["macro_step0_gap"] == pytest.approx(0.1)

    references = {
        step: {"path": str(tmp_path / f"receipt-{step}.json"), "sha256": _HASH}
        for step in (0, 3000)
    }
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="exactly steps"):
        direct.build_promotion_report(
            manifest_path=tmp_path / "manifest.json",
            evaluation_receipts=references,
            health_receipts=references,
            created_at=_REPORT_TIME,
            verify_live_manifest=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "deviation"),
    [
        (
            {"step0_gap": -0.2, "step3000_gap": -0.2, "step4000_gap": -0.1},
            "source_nonpositive_absolute_gap",
        ),
        ({"step4000_gap": 0.1}, "source_nonpositive_gap_improvement"),
        ({"step4000_correct_ce": 1.03}, "source_correct_ce_regression"),
        ({"step4000_gap": 0.39}, "source_gap_retention"),
    ],
)
def test_direct_report_rejects_visual_gate_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: Mapping[str, float],
    deviation: str,
) -> None:
    report, _, _ = _build_report_fixture(tmp_path, monkeypatch, **kwargs)
    assert report["status"] == "rejected"
    assert deviation in {item["kind"] for item in report["deviations"]}


def test_version7_gate_is_human_approved_waiver_free_and_transitively_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, manifest_path, manifest = _build_report_fixture(tmp_path, monkeypatch)
    report_path = tmp_path / "promotion-report.json"
    report_ref = _write_json(report_path, report)
    candidate = manifest["run"]["checkpoints"]["4000"]
    _write_json(
        Path(candidate["path"]) / "config.json",
        {
            "vision_alignment": {
                "recipe_version": 12,
                "formatter_version": "ssmax-v3",
            }
        },
    )
    summary = {
        "report": report,
        "report_reference": report_ref,
        "manifest": manifest,
        "manifest_reference": report["manifest"],
        "candidate": candidate,
    }
    monkeypatch.setattr(direct, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        direct, "validate_promotion_report_reference", lambda *args, **kwargs: summary
    )
    gate = direct.build_parent_gate(
        promotion_report_path=report_path,
        expected_promotion_report_sha256=report_ref["sha256"],
        approved_by="rustins",
        approved_at="2026-08-22T04:00:00+00:00",
        verify_live_checkpoint=True,
    )
    assert gate["version"] == 7
    assert gate["lineage_kind"] == direct.LINEAGE_KIND
    assert gate["waivers"] == []
    assert gate["protocol_amendment_sha256"] == direct.AMENDMENT_SHA256
    assert gate["training_git_ref"] == direct.TRAINING_GIT_REF
    assert gate["evidence_git_ref"] == "b" * 40
    assert gate["manifest_path"] == str(manifest_path.resolve())

    changed = copy.deepcopy(gate)
    changed["waivers"] = [{"reason": "manual"}]
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="waiver-free"):
        direct.validate_ssmax_perception_direct_parent_gate(
            changed,
            expected_checkpoint=Path(candidate["path"]),
            expected_checkpoint_config_sha256=candidate["config_sha256"],
            expected_model_variant=manifest["model_variant"],
            expected_data_contract_sha256=manifest["run"]["data_contract_sha256"],
            expected_trainable_contract_sha256=manifest["run"]["trainable_contract_sha256"],
            verify_live_checkpoint=True,
        )

    changed = copy.deepcopy(gate)
    changed["protocol_amendment_sha256"] = "f" * 64
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="binding differs"):
        direct.validate_ssmax_perception_direct_parent_gate(
            changed,
            expected_checkpoint=Path(candidate["path"]),
            expected_checkpoint_config_sha256=candidate["config_sha256"],
            expected_model_variant=manifest["model_variant"],
            expected_data_contract_sha256=manifest["run"]["data_contract_sha256"],
            expected_trainable_contract_sha256=manifest["run"]["trainable_contract_sha256"],
            verify_live_checkpoint=True,
        )

    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="predates"):
        direct.build_parent_gate(
            promotion_report_path=report_path,
            expected_promotion_report_sha256=report_ref["sha256"],
            approved_by="rustins",
            approved_at="2026-08-22T02:59:59+00:00",
            verify_live_checkpoint=True,
        )

    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="requires live"):
        direct.build_parent_gate(
            promotion_report_path=report_path,
            expected_promotion_report_sha256=report_ref["sha256"],
            approved_by="rustins",
            approved_at="2026-08-22T04:00:00+00:00",
            verify_live_checkpoint=False,
        )


def test_direct_visual_gate_exact_ce_and_retention_thresholds_are_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, _, _ = _build_report_fixture(
        tmp_path,
        monkeypatch,
        step3000_gap=0.5,
        step4000_gap=0.4,
        step4000_correct_ce=1.02,
    )
    assert report["status"] == "passed"


@pytest.mark.parametrize("producer", ["evaluation", "health"])
def test_direct_receipt_clis_have_no_arm_selector(producer: str) -> None:
    if producer == "evaluation":
        from scripts.eval import vision_alignment_ssmax_perception_direct as cli

        argv = [
            "--manifest",
            "manifest.json",
            "--expected-manifest-sha256",
            _HASH,
            "--step",
            "0",
            "--output",
            "receipt.json",
            "--work-dir",
            "work",
        ]
    else:
        from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

        argv = [
            "--manifest",
            "manifest.json",
            "--expected-manifest-sha256",
            _HASH,
            "--step",
            "0",
            "--output",
            "receipt.json",
            "--work-dir",
            "work",
        ]
    parsed = cli._parse_args(argv)
    assert not hasattr(parsed, "arm")
    with pytest.raises(SystemExit):
        cli._parse_args([*argv, "--arm", "treatment"])


def test_direct_evaluator_contract_is_exactly_fifteen_full_global_batches(
    tmp_path: Path,
) -> None:
    from olmo_core.data.multimodal import MultimodalCollator, MultimodalDataLoader

    examples = direct.EVALUATION_CONTRACT["examples_per_source"]
    rank_instances = direct.EVALUATION_CONTRACT["rank_batch_instances"]
    world_size = direct.TOPOLOGY_CONTRACT["world_size"]
    assert (examples, rank_instances, world_size) == (480, 2, 16)
    assert examples % (rank_instances * world_size) == 0

    sequence_length = 16
    loader = MultimodalDataLoader(
        [None] * examples,
        MultimodalCollator(
            pad_token_id=0,
            label_ignore_index=-100,
            pad_sequence_length=sequence_length,
        ),
        work_dir=tmp_path,
        global_batch_size=rank_instances * world_size * sequence_length,
        shuffle=False,
        dp_world_size=world_size,
        dp_rank=0,
    )
    assert loader.total_batches == 15
    assert loader.rank_batch_size // sequence_length == 2


def _rehash_text_result(payload: dict[str, Any]) -> None:
    payload["rank_inventory_sha256"] = direct.canonical_sha256(payload["rank_rows"])
    payload.pop("content_sha256", None)
    payload["content_sha256"] = direct.canonical_sha256(payload)


def test_direct_text_sentinel_validates_same_rank_not_cross_rank_outputs(
    tmp_path: Path,
) -> None:
    sentinel: dict[str, Any] = {
        "format": "vision_alignment_ssmax_native_text_sentinel",
        "version": 1,
        "tokenizer": {"identifier": "tokenizer", "revision": "revision"},
        "input_ids": [1] * 256,
        "labels": [2] * 256,
    }
    sentinel["content_sha256"] = direct.canonical_sha256(sentinel)
    sentinel_ref = _write_json(tmp_path / "sentinel.json", sentinel)
    manifest = {
        "text_sentinel": sentinel_ref,
        "topology": copy.deepcopy(direct.TOPOLOGY_CONTRACT),
        "run": {
            "checkpoints": {
                "0": {"identity_sha256": "1" * 64},
                "3000": {"identity_sha256": "2" * 64},
            }
        },
    }
    result = _text_sentinel_payload(3000)
    result.update(direct._expected_direct_text_invariants(manifest=manifest, step=3000))
    _rehash_text_result(result)

    validated = direct._validate_direct_text_result(result, manifest=manifest, step=3000)
    assert validated["all_ranks_passed"] is True
    assert len({row["reference"]["logits"]["sha256"] for row in result["rank_rows"]}) == 16

    changed = copy.deepcopy(result)
    changed["rank_rows"][3]["candidate"]["logits"]["sha256"] = "f" * 64
    _rehash_text_result(changed)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="exact claim differs"):
        direct._validate_direct_text_result(changed, manifest=manifest, step=3000)

    changed = copy.deepcopy(result)
    changed["rank_rows"] = changed["rank_rows"][:-1]
    changed["rank_count"] = 15
    _rehash_text_result(changed)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="rank_count differs"):
        direct._validate_direct_text_result(changed, manifest=manifest, step=3000)

    changed = copy.deepcopy(result)
    changed["input"]["sha256"] = "f" * 64
    _rehash_text_result(changed)
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="input differs"):
        direct._validate_direct_text_result(changed, manifest=manifest, step=3000)

    changed = copy.deepcopy(result)
    for snapshot in ("reference", "candidate"):
        changed["rank_rows"][0][snapshot]["ce"]["finite"] = False
    changed["rank_rows"][0]["passed"] = False
    changed["mismatch_count"] = 1
    changed["all_ranks_passed"] = False
    _rehash_text_result(changed)
    assert not direct._validate_direct_text_result(changed, manifest=manifest, step=3000)[
        "all_ranks_passed"
    ]


def test_direct_text_sentinel_runs_two_forwards_and_gathers_only_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from olmo_core.nn.lm_head import LMOutputWithLoss
    from scripts.eval import vision_alignment_ssmax_perception_direct as cli

    class FakeTrainModule:
        calls = 0

        def eval_batch(self, batch: Mapping[str, Any]) -> LMOutputWithLoss:
            self.calls += 1
            assert batch["input_ids"].shape == (1, 256)
            logits = torch.ones((1, 2, 3), dtype=torch.float32)
            ce = torch.ones((1, 2), dtype=torch.float32)
            return LMOutputWithLoss(logits=logits, loss=ce.mean(), ce_loss=ce, z_loss=None)

    train_module = FakeTrainModule()
    input_ids = torch.ones((1, 256), dtype=torch.long)
    labels = torch.ones((1, 256), dtype=torch.long)
    reference = cli._snapshot_text_sentinel(train_module, input_ids, labels)
    candidate = cli._snapshot_text_sentinel(train_module, input_ids, labels)
    assert train_module.calls == 2
    assert reference["logits"].data_ptr() != candidate["logits"].data_ptr()

    monkeypatch.setattr(cli.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(cli.dist, "get_rank", lambda: 0)

    def gather(output: list[Any], packet: Mapping[str, Any]) -> None:
        def contains_tensor(value: Any) -> bool:
            if isinstance(value, torch.Tensor):
                return True
            if isinstance(value, Mapping):
                return any(contains_tensor(item) for item in value.values())
            if isinstance(value, list):
                return any(contains_tensor(item) for item in value)
            return False

        assert not contains_tensor(packet)
        output[0] = copy.deepcopy(packet)
        output[1] = copy.deepcopy(packet)
        output[1]["row"]["rank"] = 1
        for snapshot in ("reference", "candidate"):
            output[1]["row"][snapshot]["logits"]["sha256"] = "f" * 64

    monkeypatch.setattr(cli.dist, "all_gather_object", gather)
    result = cli._text_sentinel_result(
        reference=reference,
        candidate=candidate,
        invariants={
            "artifact_sha256": "a" * 64,
            "input": cli._tensor_descriptor(input_ids, include_finite=False),
            "labels": cli._tensor_descriptor(labels, include_finite=False),
            "token_count": 256,
        },
        reference_checkpoint={"identity_sha256": "b" * 64},
        candidate_checkpoint={"identity_sha256": "c" * 64},
        candidate_step=0,
        topology={**direct.TOPOLOGY_CONTRACT, "world_size": 2},
    )
    assert result["rank_count"] == 2
    assert result["all_ranks_passed"] is True
    assert (
        result["rank_rows"][0]["reference"]["logits"]["sha256"]
        != result["rank_rows"][1]["reference"]["logits"]["sha256"]
    )


def test_direct_projection_pairing_replays_the_exact_model_input_surface(tmp_path: Path) -> None:
    from olmo_core.data.multimodal.ssmax_single_response import (
        SSMaxSingleResponseDataset,
    )
    from olmo_core.eval import (
        MultimodalFixedValidationDataset,
        MultimodalMatchedWrongImageDataset,
    )
    from olmo_core.eval.vision_alignment_ssmax_data import create_or_validate_pairing
    from scripts.eval import vision_alignment_ssmax_perception as paired_runner

    class BranchedDataset:
        content_fingerprint = "projection-pairing-test-v1"

        def __len__(self) -> int:
            return 4

        def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
            assert epoch == 0
            return {
                "input_ids": np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
                "labels": np.asarray([11, 12, 13, 14, 15], dtype=np.int64),
                "loss_masks": np.asarray([0, 1, 1, 1, 1], dtype=np.float32),
                "position_ids": np.arange(5, dtype=np.int64),
                "token_type_ids": np.zeros(5, dtype=np.int64),
                "images": np.full((1, 2, 2), index + 1, dtype=np.float32),
                "pooled_patches_idx": np.zeros((2, 1), dtype=np.int64),
                "subsegment_ids": np.asarray([10_000, 1, 1, 2, 2], dtype=np.int64),
                "metadata": {"index": index, "must_not_reach_pairing": True},
            }

    projected = SSMaxSingleResponseDataset(
        BranchedDataset(),
        source_name="cosyn_point",
        logical_split="validation",
        seed=direct.EVALUATION_CONTRACT["pairing_seed"],
        loss_token_weighting="none",
    )
    dataset = paired_runner._ModelInputDataset(projected)
    assert set(dataset[0]) == paired_runner._ModelInputDataset.required
    assert "metadata" not in dataset[0]
    assert "subsegment_ids" not in dataset[0]

    content_ids = tuple(hashlib.sha256(f"image-{index}".encode()).hexdigest() for index in range(4))
    path = tmp_path / "direct-v1" / "cosyn_point.json"
    reference = create_or_validate_pairing(
        dataset,
        path=path,
        examples=4,
        seed=direct.EVALUATION_CONTRACT["pairing_seed"],
        content_ids=content_ids,
    )
    pairing = direct.load_json(Path(reference["path"]))
    fixed = MultimodalFixedValidationDataset(
        dataset,
        pairing=pairing,
        pairing_sha256=reference["sha256"],
    )
    wrong = MultimodalMatchedWrongImageDataset(
        dataset,
        pairing=pairing,
        pairing_sha256=reference["sha256"],
    )
    for index in range(4):
        correct_row = fixed[index]
        wrong_row = wrong[index]
        assert set(correct_row) == paired_runner._ModelInputDataset.required
        assert set(wrong_row) == paired_runner._ModelInputDataset.required
        assert not np.array_equal(correct_row["images"], wrong_row["images"])
        for field in set(correct_row) - {"images"}:
            assert np.array_equal(correct_row[field], wrong_row[field])


def test_direct_health_materializes_the_exact_training_ref_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    raw = b'RECIPE_SOURCE = "training-ref"\n'
    expected = hashlib.sha256(raw).hexdigest()
    manifest = {
        "training_git": {"ref": "1" * 40},
        "training_recipe": {
            "repo_relative_path": "src/scripts/train/Vision-Alignment.py",
            "sha256": expected,
            "git_ref": "1" * 40,
        },
    }
    observed: dict[str, Any] = {}

    def git_show(command: list[str], **kwargs: Any) -> bytes:
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        return raw

    monkeypatch.setattr(cli.subprocess, "check_output", git_show)
    materialized = cli._materialize_training_recipe(manifest, work_dir=tmp_path)
    assert materialized.read_bytes() == raw
    assert materialized == tmp_path.resolve() / "training-recipe" / f"{expected}.py"
    assert observed["command"] == [
        "git",
        "show",
        f"{'1' * 40}:src/scripts/train/Vision-Alignment.py",
    ]

    class Recipe:
        pass

    recipe = Recipe()
    monkeypatch.setattr(
        cli.paired_runner,
        "_load_recipe",
        lambda path: observed.update({"loaded_path": path}) or recipe,
    )
    loaded = cli._load_training_recipe(manifest, work_dir=tmp_path)
    assert loaded is recipe
    assert observed["loaded_path"] == materialized
    assert Path(loaded.__file__) == (
        Path(cli.__file__).resolve().parents[3] / "src/scripts/train/Vision-Alignment.py"
    )

    show_calls: list[list[str]] = []
    fetch_calls: list[list[str]] = []

    def shallow_clone_show(command: list[str], **kwargs: Any) -> bytes:
        del kwargs
        show_calls.append(command)
        if len(show_calls) == 1:
            raise cli.subprocess.CalledProcessError(128, command)
        return raw

    def fetch_training_ref(command: list[str], **kwargs: Any) -> int:
        assert kwargs["cwd"] == Path(cli.__file__).resolve().parents[3]
        assert kwargs["stderr"] is cli.subprocess.PIPE
        fetch_calls.append(command)
        return 0

    monkeypatch.setattr(cli.subprocess, "check_output", shallow_clone_show)
    monkeypatch.setattr(cli.subprocess, "check_call", fetch_training_ref)
    fallback = cli._materialize_training_recipe(manifest, work_dir=tmp_path / "shallow")
    assert fallback.read_bytes() == raw
    assert show_calls == [observed["command"], observed["command"]]
    assert fetch_calls == [["git", "fetch", "--no-tags", "--depth", "1", "origin", "1" * 40]]

    materialized.write_bytes(b"drifted evidence bytes\n")
    with pytest.raises(
        direct.SSMaxPerceptionDirectEvidenceError, match="Materialized training recipe differs"
    ):
        cli._materialize_training_recipe(manifest, work_dir=tmp_path)

    monkeypatch.setattr(cli.subprocess, "check_output", lambda *args, **kwargs: b"wrong blob\n")
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="Git blob differs"):
        cli._materialize_training_recipe(manifest, work_dir=tmp_path / "second")


def test_direct_health_uses_the_exact_translated_v2_optimizer_guard_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from olmo_core.eval import vision_alignment_ssmax_perception as paired
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    ledger = {
        "version": paired.SSMAX_HEALTH_LEDGER_VERSION,
        "events": [],
        "optimizer_guard_history_reset_steps": [0],
        "optimizer_guard_rolling_interval_length": 128,
    }
    with pytest.raises(paired.SSMaxPerceptionEvidenceError, match="locked v2 promotion policy"):
        paired.summarize_optimizer_guard_trajectory(
            ledger,
            policy=direct.DIRECT_POLICY,
            step=0,
        )
    expected = paired._locked_promotion_policy(paired.PERCEPTION_V2_SCHEMA_VERSION)
    assert paired.summarize_optimizer_guard_trajectory(
        ledger,
        policy=expected,
        step=0,
    )["passed"]

    observed: dict[str, Any] = {}

    def summarize(
        value: Mapping[str, Any], *, policy: Mapping[str, Any], step: int
    ) -> dict[str, Any]:
        observed.update({"ledger": value, "policy": dict(policy), "step": step})
        return {"passed": True}

    monkeypatch.setattr(cli.paired, "summarize_optimizer_guard_trajectory", summarize)
    assert cli._summarize_optimizer_guard(ledger, step=0) == {"passed": True}
    assert observed == {"ledger": ledger, "policy": expected, "step": 0}


@pytest.mark.parametrize(
    ("step", "epoch"),
    [(0, None), (3000, 1), (3000, 2), (4000, 1), (4000, 2)],
)
def test_direct_health_cursor_accepts_only_step_aware_epoch_types(
    step: int, epoch: int | None
) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    state = {
        "global_step": step,
        "world_size": 16,
        "data_loader": {"batches_processed": step, "epoch": epoch},
    }

    saved, validated_epoch = cli._validate_trainer_cursor(state, step=step, world_size=16, rank=3)

    assert saved == state["data_loader"]
    assert validated_epoch is epoch


@pytest.mark.parametrize(
    ("step", "epoch"),
    [
        (0, 0),
        (0, False),
        (0, 0.0),
        (0, -1),
        (0, 1),
        (3000, None),
        (3000, 0),
        (3000, False),
        (3000, 1.0),
        (3000, -1),
        (4000, None),
    ],
)
def test_direct_health_cursor_rejects_noncanonical_epoch(step: int, epoch: Any) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    state = {
        "global_step": step,
        "world_size": 16,
        "data_loader": {"batches_processed": step, "epoch": epoch},
    }

    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="rank3 epoch is invalid"):
        cli._validate_trainer_cursor(state, step=step, world_size=16, rank=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_step", 3000.0),
        ("global_step", 4000),
        ("world_size", 16.0),
        ("world_size", 8),
        ("batches_processed", 3000.0),
        ("batches_processed", 4000),
    ],
)
def test_direct_health_cursor_rejects_aliases_and_step_mismatches(field: str, value: Any) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    state = {
        "global_step": 3000,
        "world_size": 16,
        "data_loader": {"batches_processed": 3000, "epoch": 1},
    }
    target = state["data_loader"] if field == "batches_processed" else state
    target[field] = value

    with pytest.raises(
        direct.SSMaxPerceptionDirectEvidenceError, match="rank3 cursor is incompatible"
    ):
        cli._validate_trainer_cursor(state, step=3000, world_size=16, rank=3)


def test_direct_health_step0_cursor_requires_an_explicit_null_epoch() -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    state = {
        "global_step": 0,
        "world_size": 16,
        "data_loader": {"batches_processed": 0},
    }

    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="rank3 epoch is invalid"):
        cli._validate_trainer_cursor(state, step=0, world_size=16, rank=3)


def test_direct_health_hydrates_only_the_exact_real_saved_checkpointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.nn.vision import Molmo2TokenIds
    from scripts.eval import vision_alignment_ssmax_perception_direct_health as cli

    repository_root = Path(cli.__file__).resolve().parents[3]
    identity = direct.DIRECT_RUN_IDENTITIES["ssmax_head_qknorm"]
    recipe_relative = "src/scripts/train/Vision-Alignment.py"
    recipe_raw = cli.subprocess.check_output(
        ["git", "show", f"{direct.TRAINING_GIT_REF}:{recipe_relative}"],
        cwd=repository_root,
    )
    manifest = {
        "model_variant": "ssmax_head_qknorm",
        "run": {"run_name": identity["run_name"]},
        "training_git": {"ref": direct.TRAINING_GIT_REF},
        "training_recipe": {
            "repo_relative_path": recipe_relative,
            "sha256": hashlib.sha256(recipe_raw).hexdigest(),
            "git_ref": direct.TRAINING_GIT_REF,
        },
    }
    recipe = cli._load_training_recipe(manifest, work_dir=tmp_path / "recipe")
    profile_path = repository_root / identity["profile"]
    profile_raw = profile_path.read_bytes()
    assert hashlib.sha256(profile_raw).hexdigest() == identity["profile_sha256"]
    profile = yaml.safe_load(profile_raw)
    monkeypatch.setattr(
        recipe,
        "_load_tokenizer",
        lambda artifacts: (SimpleNamespace(pad_token_id=100_277), Molmo2TokenIds()),
    )
    monkeypatch.setattr(recipe, "_validate_phase_contract", lambda *args, **kwargs: None)
    config = recipe.build_config(
        recipe_relative,
        identity["run_name"],
        [f"--phase={profile['phase']}", *profile["overrides"]],
        reviewed_profile_path=identity["profile"],
        reviewed_profile_sha256=identity["profile_sha256"],
    )
    # Match the representation read from an actual saved config.json. Config
    # serialization can retain str-backed enum instances until the JSON
    # round-trip turns them into their canonical string values.
    raw = json.loads(json.dumps(config.as_config_dict()))
    checkpointer = raw["trainer"]["callbacks"]["checkpointer"]
    assert checkpointer == cli._DIRECT_SERIALIZED_CHECKPOINTER
    assert "save_interval" not in checkpointer
    with pytest.raises(OLMoConfigurationError, match="ephemeral_save_interval"):
        recipe.ExperimentConfig.from_dict(raw)

    hydrated = cli._hydrate_direct_saved_config(raw, manifest=manifest)
    assert "save_interval" not in raw["trainer"]["callbacks"]["checkpointer"]
    assert hydrated["trainer"]["callbacks"]["checkpointer"]["save_interval"] is None
    decoded = recipe.ExperimentConfig.from_dict(hydrated)
    assert decoded.trainer.callbacks["checkpointer"].save_interval is None

    present = copy.deepcopy(raw)
    present["trainer"]["callbacks"]["checkpointer"]["save_interval"] = None
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="must omit runtime-null"):
        cli._hydrate_direct_saved_config(present, manifest=manifest)

    for field, expected in cli._DIRECT_SERIALIZED_CHECKPOINTER.items():
        changed = copy.deepcopy(raw)
        if isinstance(expected, bool):
            replacement: Any = int(expected)
        elif isinstance(expected, int):
            replacement = float(expected)
        elif isinstance(expected, str):
            replacement = f"{expected}-drift"
        else:
            replacement = list(reversed(expected))
        changed["trainer"]["callbacks"]["checkpointer"][field] = replacement
        with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="reviewed contract"):
            cli._hydrate_direct_saved_config(changed, manifest=manifest)

    extra = copy.deepcopy(raw)
    extra["trainer"]["callbacks"]["checkpointer"]["unexpected"] = False
    with pytest.raises(direct.SSMaxPerceptionDirectEvidenceError, match="reviewed contract"):
        cli._hydrate_direct_saved_config(extra, manifest=manifest)

    drifted_profile = copy.deepcopy(raw)
    drifted_profile["reviewed_profile_sha256"] = "f" * 64
    with pytest.raises(
        direct.SSMaxPerceptionDirectEvidenceError, match="reviewed profile identity"
    ):
        cli._hydrate_direct_saved_config(drifted_profile, manifest=manifest)


def test_direct_promotion_selectors_require_exactly_three_unqualified_steps() -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_promotion as cli

    values = cli._values(
        ["0=step0.json", "3000=step3000.json", "4000=step4000.json"],
        option="--evaluation",
    )
    assert values == {0: "step0.json", 3000: "step3000.json", 4000: "step4000.json"}
    with pytest.raises(ValueError, match="step is invalid"):
        cli._values(
            ["treatment:0=step0.json", "3000=step3000.json", "4000=step4000.json"],
            option="--evaluation",
        )
    with pytest.raises(ValueError, match="exactly steps"):
        cli._values(["0=step0.json", "4000=step4000.json"], option="--health")


def test_direct_approval_refuses_existing_output_before_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_promotion as cli

    output = tmp_path / "parent-gate.json"
    output.write_text("preserve me\n")

    def unexpected_build(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise AssertionError("approval builder should not run when output exists")

    monkeypatch.setattr(cli, "build_parent_gate", unexpected_build)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(
            [
                "approve",
                "--report",
                str(tmp_path / "report.json"),
                "--expected-report-sha256",
                _HASH,
                "--approved-by",
                "rustins",
                "--approved-at",
                "2026-08-22T04:00:00+00:00",
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "preserve me\n"


def _comparison_summary(variant: str, value: float) -> dict[str, Any]:
    windows = {
        window: {
            "macro_step0_correct_ce": value + 1.0,
            "macro_step4000_correct_ce": value + 0.9,
            "macro_step0_gap": value + 0.1,
            "macro_step3000_gap": value + 0.4,
            "macro_step4000_gap": value + 0.5,
        }
        for window in direct.WINDOWS
    }
    shared = {
        "training_git": _git(direct.TRAINING_GIT_REF),
        "evidence_git": _git("b" * 40),
        "producers": {"protocol": {"git_ref": "b" * 40}},
        "training_recipe": {"git_ref": direct.TRAINING_GIT_REF},
        "protocol_amendment": {"sha256": direct.AMENDMENT_SHA256},
        "perception_provenance": {"sha256": "1" * 64},
        "source_audit": {"sha256": "2" * 64},
        "source_audit_fingerprint": "3" * 64,
        "single_response_projection": {"content_sha256": "4" * 64},
        "attention_probe": {"sha256": "5" * 64},
        "text_sentinel": {"sha256": "6" * 64},
        "pairings": {
            source: {"path": f"/evidence/direct-v1/{source}.json", "sha256": "7" * 64}
            for source in direct.SOURCES
        },
        "evaluation": copy.deepcopy(direct.EVALUATION_CONTRACT),
        "topology": copy.deepcopy(direct.TOPOLOGY_CONTRACT),
        "policy": copy.deepcopy(direct.DIRECT_POLICY),
        "loss_mass_targets": {source: 1 / len(direct.SOURCES) for source in direct.SOURCES},
    }
    return {
        "report": {
            "created_at": _REPORT_TIME,
            "model_variant": variant,
            "run_id": f"{variant}-direct",
            "content_sha256": f"{int(value * 10) + 1:064x}",
            "summary": {
                "windows": windows,
                "attention_trajectory": {"0": {}, "3000": {}, "4000": {}},
                "optimizer_guard_trajectory": {"passed": True},
            },
        },
        "manifest": {
            "model_variant": variant,
            "run_id": f"{variant}-direct",
            **shared,
        },
        "candidate": {"identity_sha256": f"{int(value * 10) + 2:064x}"},
    }


def test_direct_cross_model_comparison_is_descriptive_and_has_no_arm_labels(
    tmp_path: Path,
) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_compare as cli

    result = cli.build_comparison(
        left_summary=_comparison_summary("ssmax_no_qknorm", 0.2),
        left_report_path=tmp_path / "no-qk-report.json",
        left_report_sha256="1" * 64,
        right_summary=_comparison_summary("ssmax_head_qknorm", 0.5),
        right_report_path=tmp_path / "head-report.json",
        right_report_sha256="2" * 64,
        created_at="2026-08-22T05:00:00+00:00",
    )
    assert result["decision_scope"] == "descriptive_non_promotion"
    assert result["winner"] is None
    assert set(result["model_variants"]) == set(direct.MODEL_VARIANTS)
    for window in direct.WINDOWS:
        window_result = result["descriptive_difference"]["windows"][window]
        assert window_result["same_step_metric_difference"]["visual_gap"]["4000"] == pytest.approx(
            0.3
        )
        assert window_result["step0_normalized_adaptation_difference"]["visual_gap"][
            "3000"
        ] == pytest.approx(0.0)
        assert window_result["step0_normalized_adaptation_difference"]["visual_gap"][
            "4000"
        ] == pytest.approx(0.0)
    assert result["shared_protocol_inputs_sha256"] == direct.canonical_sha256(
        result["shared_protocol_inputs"]
    )
    serialized = json.dumps(result, sort_keys=True).lower()
    assert '"arm"' not in serialized
    assert "treatment" not in serialized
    assert "control" not in serialized
    assert "causal_adaptation" not in serialized


@pytest.mark.parametrize(
    "field",
    [
        "training_git",
        "evidence_git",
        "producers",
        "training_recipe",
        "protocol_amendment",
        "perception_provenance",
        "source_audit",
        "source_audit_fingerprint",
        "single_response_projection",
        "attention_probe",
        "text_sentinel",
        "pairings",
        "evaluation",
        "topology",
        "policy",
        "loss_mass_targets",
    ],
)
def test_direct_cross_model_comparison_rejects_shared_protocol_drift(
    field: str, tmp_path: Path
) -> None:
    from scripts.eval import vision_alignment_ssmax_perception_direct_compare as cli

    left = _comparison_summary("ssmax_head_qknorm", 0.5)
    right = _comparison_summary("ssmax_no_qknorm", 0.2)
    right["manifest"][field] = {"drifted": field}
    with pytest.raises(
        direct.SSMaxPerceptionDirectEvidenceError, match="shared protocol inputs differ"
    ):
        cli.build_comparison(
            left_summary=left,
            left_report_path=tmp_path / "head.json",
            left_report_sha256="1" * 64,
            right_summary=right,
            right_report_path=tmp_path / "no-qk.json",
            right_report_sha256="2" * 64,
            created_at="2026-08-22T05:00:00+00:00",
        )
