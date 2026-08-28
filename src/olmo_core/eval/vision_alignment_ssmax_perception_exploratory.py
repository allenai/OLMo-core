"""Exploratory-only admission for rejected direct SSMax perception evidence.

This module is additive to the strict version-7 direct-perception promotion protocol. It never
changes, waives, or reinterprets that protocol. Instead it exactly rebuilds a strict report from
its six immutable receipts and permits a separately authorized exploratory joint run only when
every strict deviation is a source-level short-prefix visual diagnostic. Macro deviations,
``first_32``/``all`` source deviations, correct-image CE regressions, and technical deviations
remain inadmissible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from olmo_core.eval import vision_alignment_ssmax_perception_direct as strict

PARENT_GATE_VERSION = 8
GATE_SCOPE = "exploratory_joint_only"

AUTHORIZATION_RELATIVE_PATH = (
    "configs/vision_moe/vision_alignment/eval/"
    "ssmax_perception_exploratory_joint_authorization_v1.json"
)
AUTHORIZATION_FORMAT = "vision_alignment_ssmax_perception_exploratory_joint_authorization"
AUTHORIZATION_VERSION = 1
AUTHORIZATION_RAW_SHA256 = "baa0b9eda20c2aaccdfcf84ba5c6922943267c75a68f3786e0896d2cd0ac9f67"
AUTHORIZATION_CONTENT_SHA256 = "b8104b4b9ee8c9109ba64a1d338dca6ce1cf3d4283c5f1b07b121848c4be069c"
AUTHORIZATION_APPROVED_BY = "rustins"
AUTHORIZATION_RESEARCH_QUESTION = (
    "Whether per-head QK norm changes downstream Molmo adaptability, BLINK-Jigsaw and "
    "MathVista-Geometry performance, or attention-logit, entropy, effective-context, and "
    "routing trajectories during matched joint alignment."
)

ALLOWED_DEVIATION_KINDS = frozenset(
    {
        "source_gap_retention",
        "source_nonpositive_absolute_gap",
        "source_nonpositive_gap_improvement",
    }
)
ALLOWED_SOURCE_WINDOWS = frozenset({"first_1", "first_8"})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DURABLE_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}")
_ARTIFACT_REFERENCE_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_REFERENCE_FIELDS = frozenset({"path", "sha256", "content_sha256"})
_AUTHORIZATION_REFERENCE_FIELDS = frozenset({"repo_relative_path", "raw_sha256", "content_sha256"})
_AUTHORIZATION_FIELDS = frozenset(
    {
        "approved_at",
        "approved_by",
        "format",
        "policy",
        "promotion_decision",
        "research_question",
        "scope",
        "strict_v7_preserved",
        "version",
        "winner_selection",
    }
)
_AUTHORIZATION_POLICY_FIELDS = frozenset(
    {
        "allowed_deviation_kinds",
        "allowed_source_windows",
        "require_zero_correct_ce_deviations",
        "require_zero_first_32_or_all_source_deviations",
        "require_zero_macro_deviations",
        "require_zero_technical_deviations",
    }
)
_DEVIATION_FIELDS = frozenset({"kind", "source", "window"})
_GATE_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "scope",
        "recipe_version",
        "formatter_version",
        "phase",
        "model_variant",
        "lineage_kind",
        "run_id",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "strict_report_path",
        "strict_report_sha256",
        "strict_report_content_sha256",
        "strict_report_status",
        "strict_receipts",
        "acknowledged_deviations",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_path",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
        "authorization",
        "training_git_ref",
        "evidence_git_ref",
        "approved_by",
        "approved_at",
        "waivers",
    }
)


class SSMaxPerceptionExploratoryEvidenceError(ValueError):
    """Raised when exploratory SSMax perception admission is not exactly authorized."""


def _exact(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise SSMaxPerceptionExploratoryEvidenceError(
            f"{name} fields differ: missing={sorted(fields - observed)}, "
            f"extra={sorted(observed - fields)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SSMaxPerceptionExploratoryEvidenceError(f"{name} must be a SHA-256 digest")
    return value


def _timestamp(value: Any, *, name: str) -> datetime:
    try:
        return strict._timestamp(value, name=name)
    except (TypeError, ValueError, strict.SSMaxPerceptionDirectEvidenceError) as error:
        raise SSMaxPerceptionExploratoryEvidenceError(str(error)) from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory authorization is not finite canonical JSON"
        ) from error


def _load_json(path: Path, *, name: str) -> Any:
    try:
        return strict.load_json(path)
    except (OSError, strict.SSMaxPerceptionDirectEvidenceError) as error:
        raise SSMaxPerceptionExploratoryEvidenceError(f"Could not load {name}: {error}") from error


def _pinned_file(path: Path, expected_sha256: str, *, name: str) -> bytes:
    expected_sha256 = _sha256(expected_sha256, name=f"expected {name} SHA-256")
    path = path.expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SSMaxPerceptionExploratoryEvidenceError(f"Could not read {name} {path}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SSMaxPerceptionExploratoryEvidenceError(f"{name} differs from its explicit pin")
    return raw


def _artifact_reference(value: Any, *, name: str) -> Mapping[str, str]:
    reference = _exact(value, _ARTIFACT_REFERENCE_FIELDS, name=name)
    path = reference["path"]
    digest = _sha256(reference["sha256"], name=f"{name} SHA-256")
    if not isinstance(path, str) or not Path(path).expanduser().is_absolute():
        raise SSMaxPerceptionExploratoryEvidenceError(f"{name} path must be absolute")
    return {"path": path, "sha256": digest}


def _authorization_reference(
    repository_root: Path | None = None,
) -> tuple[dict[str, str], Mapping[str, Any]]:
    root = (
        repository_root.expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    path = (root / AUTHORIZATION_RELATIVE_PATH).resolve()
    if not path.is_relative_to(root):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory authorization escapes the repository"
        )
    raw = _pinned_file(path, AUTHORIZATION_RAW_SHA256, name="exploratory authorization")
    value = _exact(
        _load_json(path, name="exploratory authorization"),
        _AUTHORIZATION_FIELDS,
        name="exploratory authorization",
    )
    if raw != _canonical_bytes(value):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory authorization bytes are not canonical"
        )
    policy = _exact(
        value["policy"],
        _AUTHORIZATION_POLICY_FIELDS,
        name="exploratory authorization policy",
    )
    expected_policy = {
        "allowed_deviation_kinds": sorted(ALLOWED_DEVIATION_KINDS),
        "allowed_source_windows": sorted(ALLOWED_SOURCE_WINDOWS),
        "require_zero_correct_ce_deviations": True,
        "require_zero_first_32_or_all_source_deviations": True,
        "require_zero_macro_deviations": True,
        "require_zero_technical_deviations": True,
    }
    if dict(policy) != expected_policy:
        raise SSMaxPerceptionExploratoryEvidenceError("Exploratory authorization policy differs")
    expected_values = {
        "format": AUTHORIZATION_FORMAT,
        "version": AUTHORIZATION_VERSION,
        "approved_by": AUTHORIZATION_APPROVED_BY,
        "scope": GATE_SCOPE,
        "research_question": AUTHORIZATION_RESEARCH_QUESTION,
        "strict_v7_preserved": True,
        "promotion_decision": False,
        "winner_selection": False,
    }
    if any(
        type(value[key]) is not type(expected) or value[key] != expected
        for key, expected in expected_values.items()
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory authorization identity or decision differs"
        )
    approved_at = _timestamp(value["approved_at"], name="authorization approved_at")
    if approved_at.date().isoformat() != "2026-08-28":
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory authorization approval date differs"
        )
    content_sha256 = strict.canonical_sha256(value)
    if content_sha256 != AUTHORIZATION_CONTENT_SHA256:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory authorization semantic identity differs"
        )
    return (
        {
            "repo_relative_path": AUTHORIZATION_RELATIVE_PATH,
            "raw_sha256": AUTHORIZATION_RAW_SHA256,
            "content_sha256": AUTHORIZATION_CONTENT_SHA256,
        },
        value,
    )


def _validate_exploratory_deviations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory admission requires a rejected strict report with deviations"
        )
    output: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        deviation = _exact(
            raw,
            _DEVIATION_FIELDS,
            name=f"strict deviation {index}",
        )
        kind = deviation["kind"]
        source = deviation["source"]
        window = deviation["window"]
        if (
            kind not in ALLOWED_DEVIATION_KINDS
            or source not in strict.SOURCES
            or window not in ALLOWED_SOURCE_WINDOWS
        ):
            raise SSMaxPerceptionExploratoryEvidenceError(
                "Strict report contains a non-authorized exploratory deviation: "
                f"{dict(deviation)!r}"
            )
        output.append({"kind": str(kind), "source": str(source), "window": str(window)})
    return output


def audit_strict_report_reference(
    value: Any,
    *,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Reopen and exactly rebuild one rejected strict direct-perception report."""

    if not verify_live_checkpoint:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory admission requires live checkpoint and Git verification"
        )
    reference = _artifact_reference(value, name="strict direct report")
    report_path = Path(reference["path"]).expanduser().resolve()
    _pinned_file(report_path, reference["sha256"], name="strict direct report")
    report = _exact(
        _load_json(report_path, name="strict direct report"),
        strict._PROMOTION_REPORT_FIELDS,
        name="strict direct report",
    )
    if (
        report["format"] != strict.PROMOTION_REPORT_FORMAT
        or type(report["version"]) is not int
        or report["version"] != strict.SCHEMA_VERSION
        or report["status"] != "rejected"
        or report["decision_scope"] != "within_lineage_noncausal_joint_admission"
        or report["model_variant"] not in strict.MODEL_VARIANTS
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Strict report is not a rejected direct-perception decision"
        )
    _timestamp(report["created_at"], name="strict report created_at")
    try:
        strict._content_sha(report, name="strict direct report")
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryEvidenceError(str(error)) from error
    acknowledged = _validate_exploratory_deviations(report["deviations"])

    manifest_ref = _exact(
        report["manifest"],
        _MANIFEST_REFERENCE_FIELDS,
        name="strict report manifest",
    )
    _sha256(manifest_ref["sha256"], name="strict manifest raw SHA-256")
    _sha256(manifest_ref["content_sha256"], name="strict manifest content SHA-256")
    manifest_path = Path(str(manifest_ref["path"])).expanduser().resolve()
    _pinned_file(
        manifest_path,
        str(manifest_ref["sha256"]),
        name="strict direct manifest",
    )
    try:
        manifest = strict.load_manifest(manifest_path, verify_live=True)
    except (OSError, strict.SSMaxPerceptionDirectEvidenceError) as error:
        raise SSMaxPerceptionExploratoryEvidenceError(str(error)) from error
    if (
        manifest["content_sha256"] != manifest_ref["content_sha256"]
        or manifest["run_id"] != report["run_id"]
        or manifest["model_variant"] != report["model_variant"]
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Strict report names an incompatible direct manifest"
        )
    try:
        candidate = strict._checkpoint_reference(
            manifest["run"]["checkpoints"]["4000"],
            step=4000,
            verify_live=True,
            workers=8,
        )
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryEvidenceError(str(error)) from error

    receipt_refs = _exact(
        report["receipts"],
        frozenset(str(step) for step in strict.REQUIRED_STEPS),
        name="strict report receipts",
    )
    evaluation_refs: dict[int, Mapping[str, str]] = {}
    health_refs: dict[int, Mapping[str, str]] = {}
    for step in strict.REQUIRED_STEPS:
        refs = _exact(
            receipt_refs[str(step)],
            frozenset({"evaluation", "health"}),
            name=f"strict step{step} receipts",
        )
        evaluation_refs[step] = _artifact_reference(
            refs["evaluation"], name=f"strict step{step} evaluation receipt"
        )
        health_refs[step] = _artifact_reference(
            refs["health"], name=f"strict step{step} health receipt"
        )
    try:
        rebuilt = strict.build_promotion_report(
            manifest_path=manifest_path,
            evaluation_receipts=evaluation_refs,
            health_receipts=health_refs,
            created_at=str(report["created_at"]),
            verify_live_manifest=True,
        )
    except (OSError, strict.SSMaxPerceptionDirectEvidenceError) as error:
        raise SSMaxPerceptionExploratoryEvidenceError(str(error)) from error
    if rebuilt != dict(report):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Strict report differs from its six bound raw receipts"
        )
    return {
        "strict_report": report,
        "strict_report_reference": dict(reference),
        "manifest": manifest,
        "manifest_reference": dict(manifest_ref),
        "candidate": dict(candidate),
        "acknowledged_deviations": acknowledged,
    }


def _candidate_metadata(candidate: Mapping[str, Any]) -> tuple[int, str]:
    path = Path(str(candidate["path"])).expanduser().resolve() / "config.json"
    config = _load_json(path, name="exploratory candidate config")
    if not isinstance(config, Mapping) or not isinstance(config.get("vision_alignment"), Mapping):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory candidate lacks vision-alignment metadata"
        )
    metadata = config["vision_alignment"]
    recipe_version = metadata.get("recipe_version")
    formatter_version = metadata.get("formatter_version")
    if type(recipe_version) is not int or recipe_version < 1:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory candidate recipe version is malformed"
        )
    if not isinstance(formatter_version, str) or not formatter_version:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory candidate formatter version is malformed"
        )
    return recipe_version, formatter_version


def build_parent_gate(
    *,
    strict_report_path: Path,
    expected_strict_report_sha256: str,
    approved_by: str,
    approved_at: str,
) -> dict[str, Any]:
    """Build one human-approved, exploratory-only version-8 parent gate."""

    summary = audit_strict_report_reference(
        {
            "path": str(strict_report_path.expanduser().resolve()),
            "sha256": expected_strict_report_sha256,
        }
    )
    authorization_ref, authorization = _authorization_reference()
    if (
        not isinstance(approved_by, str)
        or _DURABLE_IDENTITY_RE.fullmatch(approved_by) is None
        or approved_by != authorization["approved_by"]
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "approved_by must match the durable exploratory authorization identity"
        )
    approval_time = _timestamp(approved_at, name="exploratory gate approved_at")
    report_time = _timestamp(
        summary["strict_report"]["created_at"], name="strict report created_at"
    )
    authorization_time = _timestamp(authorization["approved_at"], name="authorization approved_at")
    if approval_time < report_time or approval_time < authorization_time:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory gate approval predates its report or authorization"
        )
    candidate = summary["candidate"]
    manifest = summary["manifest"]
    manifest_ref = summary["manifest_reference"]
    amendment = manifest["protocol_amendment"]
    recipe_version, formatter_version = _candidate_metadata(candidate)
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": PARENT_GATE_VERSION,
        "status": "approved",
        "scope": GATE_SCOPE,
        "recipe_version": recipe_version,
        "formatter_version": formatter_version,
        "phase": "perception",
        "model_variant": manifest["model_variant"],
        "lineage_kind": strict.LINEAGE_KIND,
        "run_id": manifest["run_id"],
        "checkpoint": candidate["path"],
        "checkpoint_config_sha256": candidate["config_sha256"],
        "checkpoint_identity_sha256": candidate["identity_sha256"],
        "data_contract_sha256": manifest["run"]["data_contract_sha256"],
        "trainable_contract_sha256": manifest["run"]["trainable_contract_sha256"],
        "global_step": 4000,
        "metrics_artifact_sha256": expected_strict_report_sha256,
        "strict_report_path": str(strict_report_path.expanduser().resolve()),
        "strict_report_sha256": expected_strict_report_sha256,
        "strict_report_content_sha256": summary["strict_report"]["content_sha256"],
        "strict_report_status": "rejected",
        "strict_receipts": copy.deepcopy(summary["strict_report"]["receipts"]),
        "acknowledged_deviations": copy.deepcopy(summary["acknowledged_deviations"]),
        "manifest_path": manifest_ref["path"],
        "manifest_sha256": manifest_ref["sha256"],
        "manifest_content_sha256": manifest_ref["content_sha256"],
        "protocol_amendment_path": amendment["path"],
        "protocol_amendment_sha256": amendment["sha256"],
        "protocol_amendment_content_sha256": amendment["content_sha256"],
        "authorization": authorization_ref,
        "training_git_ref": manifest["training_git"]["ref"],
        "evidence_git_ref": manifest["evidence_git"]["ref"],
        "approved_by": approved_by,
        "approved_at": approved_at,
        "waivers": [],
    }
    validate_ssmax_perception_exploratory_parent_gate(
        gate,
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(manifest["run"]["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(manifest["run"]["trainable_contract_sha256"]),
    )
    return gate


def validate_ssmax_perception_exploratory_parent_gate(
    gate: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_data_contract_sha256: str,
    expected_trainable_contract_sha256: str,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Validate one exact exploratory-only version-8 perception parent gate."""

    if not verify_live_checkpoint:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory eligibility requires live checkpoint and Git verification"
        )
    value = _exact(gate, _GATE_FIELDS, name="SSMax exploratory parent gate")
    expected_pairs = {
        "format": "vision_alignment_parent_gate",
        "version": PARENT_GATE_VERSION,
        "status": "approved",
        "scope": GATE_SCOPE,
        "phase": "perception",
        "model_variant": expected_model_variant,
        "lineage_kind": strict.LINEAGE_KIND,
        "global_step": 4000,
        "checkpoint_config_sha256": expected_checkpoint_config_sha256,
        "data_contract_sha256": expected_data_contract_sha256,
        "trainable_contract_sha256": expected_trainable_contract_sha256,
        "strict_report_status": "rejected",
        "waivers": [],
    }
    for key, expected in expected_pairs.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise SSMaxPerceptionExploratoryEvidenceError(
                f"SSMax exploratory parent gate {key} differs"
            )
    if expected_model_variant not in strict.MODEL_VARIANTS:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "SSMax exploratory parent gate model variant is unsupported"
        )
    if (
        not isinstance(value["run_id"], str)
        or not value["run_id"]
        or Path(str(value["checkpoint"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or expected_checkpoint.name != "step4000"
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "SSMax exploratory parent gate must name one direct step4000 lineage"
        )
    for key in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "metrics_artifact_sha256",
        "strict_report_sha256",
        "strict_report_content_sha256",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
    ):
        _sha256(value[key], name=f"SSMax exploratory gate {key}")
    if value["metrics_artifact_sha256"] != value["strict_report_sha256"]:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory metrics artifact differs from the strict report"
        )

    authorization_ref = _exact(
        value["authorization"],
        _AUTHORIZATION_REFERENCE_FIELDS,
        name="exploratory gate authorization",
    )
    live_authorization_ref, authorization = _authorization_reference()
    if dict(authorization_ref) != live_authorization_ref:
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory gate authorization reference differs"
        )
    summary = audit_strict_report_reference(
        {
            "path": value["strict_report_path"],
            "sha256": value["strict_report_sha256"],
        }
    )
    report = summary["strict_report"]
    manifest = summary["manifest"]
    manifest_ref = summary["manifest_reference"]
    candidate = summary["candidate"]
    amendment = manifest["protocol_amendment"]
    if (
        report["content_sha256"] != value["strict_report_content_sha256"]
        or report["receipts"] != value["strict_receipts"]
        or summary["acknowledged_deviations"] != value["acknowledged_deviations"]
        or manifest["run_id"] != value["run_id"]
        or manifest["model_variant"] != value["model_variant"]
        or manifest_ref["path"] != value["manifest_path"]
        or manifest_ref["sha256"] != value["manifest_sha256"]
        or manifest_ref["content_sha256"] != value["manifest_content_sha256"]
        or candidate["path"] != value["checkpoint"]
        or candidate["config_sha256"] != value["checkpoint_config_sha256"]
        or candidate["identity_sha256"] != value["checkpoint_identity_sha256"]
        or manifest["run"]["data_contract_sha256"] != value["data_contract_sha256"]
        or manifest["run"]["trainable_contract_sha256"] != value["trainable_contract_sha256"]
        or amendment["path"] != value["protocol_amendment_path"]
        or amendment["sha256"] != value["protocol_amendment_sha256"]
        or amendment["content_sha256"] != value["protocol_amendment_content_sha256"]
        or manifest["training_git"]["ref"] != value["training_git_ref"]
        or manifest["evidence_git"]["ref"] != value["evidence_git_ref"]
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory gate report, lineage, receipts, Git, or amendment binding differs"
        )
    recipe_version, formatter_version = _candidate_metadata(candidate)
    if (
        type(value["recipe_version"]) is not int
        or value["recipe_version"] != recipe_version
        or not isinstance(value["formatter_version"], str)
        or value["formatter_version"] != formatter_version
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory gate recipe or formatter identity differs"
        )
    approved_by = value["approved_by"]
    if (
        not isinstance(approved_by, str)
        or _DURABLE_IDENTITY_RE.fullmatch(approved_by) is None
        or approved_by != authorization["approved_by"]
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory gate approved_by differs from its authorization"
        )
    approval_time = _timestamp(value["approved_at"], name="exploratory gate approved_at")
    if approval_time < _timestamp(report["created_at"], name="strict report created_at") or (
        approval_time < _timestamp(authorization["approved_at"], name="authorization approved_at")
    ):
        raise SSMaxPerceptionExploratoryEvidenceError(
            "Exploratory gate approval predates its report or authorization"
        )
    return {
        "gate": value,
        "strict_report": report,
        "strict_report_reference": summary["strict_report_reference"],
        "manifest": manifest,
        "manifest_reference": manifest_ref,
        "candidate": candidate,
        "authorization": authorization,
        "authorization_reference": live_authorization_ref,
        "acknowledged_deviations": summary["acknowledged_deviations"],
    }


__all__ = [
    "ALLOWED_DEVIATION_KINDS",
    "ALLOWED_SOURCE_WINDOWS",
    "AUTHORIZATION_CONTENT_SHA256",
    "AUTHORIZATION_RELATIVE_PATH",
    "AUTHORIZATION_RAW_SHA256",
    "GATE_SCOPE",
    "PARENT_GATE_VERSION",
    "SSMaxPerceptionExploratoryEvidenceError",
    "audit_strict_report_reference",
    "build_parent_gate",
    "validate_ssmax_perception_exploratory_parent_gate",
]
