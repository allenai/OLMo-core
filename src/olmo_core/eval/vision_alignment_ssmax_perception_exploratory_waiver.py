"""Research-only SSMax perception admission with an explicit health-evidence waiver.

This protocol is additive to the strict version-7 and exploratory version-8 protocols. It does
not change either protocol and never creates a promotion decision. It validates all direct
evaluation receipts at steps 0, 3000, and 4000, validates the direct health receipt at step 0,
and records that checkpoint-native health evidence at steps 3000 and 4000 is unavailable. The
only permitted continuation is a matched exploratory SSMax joint run.
"""

from __future__ import annotations

import copy
import hashlib
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.eval import vision_alignment_ssmax_perception_direct as strict
from olmo_core.eval import vision_alignment_ssmax_perception_exploratory as v8
from olmo_core.eval.ssmax_attention_diagnostics import compare_ssmax_attention_reports

REPORT_FORMAT = "vision_alignment_ssmax_perception_exploratory_waiver_report"
REPORT_VERSION = 1
REPORT_STATUS = "eligible_with_required_waiver"
PARENT_GATE_VERSION = 9
GATE_SCOPE = "exploratory_joint_only_evaluation_complete_step0_health"
DECISION_SCOPE = "research_only_non_promotion"

REQUIRED_EVALUATION_STEPS = strict.REQUIRED_STEPS
REQUIRED_HEALTH_STEPS = (0,)
WAIVED_HEALTH_STEPS = (3000, 4000)
WAIVER_ID = "missing_direct_health_receipts_steps_3000_4000"
UNVERIFIED_HEALTH_CLAIMS = (
    "data_error_trajectory",
    "checkpoint_native_health_ledger_event_chain_trajectory",
    "nonfinite_gradient_trajectory",
    "nonfinite_loss_trajectory",
    "optimizer_guard_trajectory",
    "per_source_loss_mass_trajectory",
    "sixteen_rank_trainer_state_replay",
)

AUTHORIZATION_RELATIVE_PATH = (
    "configs/vision_moe/vision_alignment/eval/"
    "ssmax_perception_exploratory_health_waiver_authorization_v1.json"
)
AUTHORIZATION_FORMAT = "vision_alignment_ssmax_perception_exploratory_health_waiver_authorization"
AUTHORIZATION_VERSION = 1
AUTHORIZATION_RAW_SHA256 = "20ff3d8dc22d9188f0f08e58292968a1d8abb8024593e69b733289a93690e902"
AUTHORIZATION_CONTENT_SHA256 = "29a9e2b8586aa21688cafe365ac85f084484457d405a74a6892724bd94c9da94"
AUTHORIZATION_APPROVED_BY = "rustins"

# The admission revision is one exact commit after the already-published v8 evidence revision.
# Its child may be only the predeclared two-profile joint-consumer revision.
ADMISSION_PARENT_GIT_REF = "b94ed3a3d8ec6278819b854aedc7f33b8ba2b3b7"
ADMISSION_GIT_DIFF = frozenset(
    {
        ("A", AUTHORIZATION_RELATIVE_PATH),
        ("A", "src/olmo_core/eval/vision_alignment_ssmax_perception_exploratory_waiver.py"),
        ("A", "src/scripts/eval/vision_alignment_ssmax_perception_exploratory_waiver.py"),
        ("A", "src/test/eval/vision_alignment_ssmax_perception_exploratory_waiver_test.py"),
        ("M", "src/olmo_core/eval/vision_alignment_ssmax_joint.py"),
        ("M", "src/scripts/train/Vision-Alignment.py"),
        ("M", "src/test/eval/vision_alignment_ssmax_joint_test.py"),
        ("M", "src/test/scripts/vision_alignment_test.py"),
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REF_RE = re.compile(r"[0-9a-f]{40}")
_DURABLE_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}")
_ARTIFACT_REFERENCE_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_REFERENCE_FIELDS = frozenset({"path", "sha256", "content_sha256"})
_AUTHORIZATION_REFERENCE_FIELDS = frozenset({"repo_relative_path", "raw_sha256", "content_sha256"})
_AUTHORIZATION_FIELDS = frozenset(
    {
        "approved_at",
        "approved_by",
        "authorized_evidence",
        "format",
        "policy",
        "promotion_decision",
        "research_question",
        "scope",
        "strict_v7_preserved",
        "strict_v8_preserved",
        "version",
        "winner_selection",
    }
)
_AUTHORIZATION_POLICY_FIELDS = frozenset(
    {
        "allowed_deviation_kinds",
        "allowed_source_windows",
        "required_evaluation_steps",
        "required_health_steps",
        "require_zero_correct_ce_deviations",
        "require_zero_evaluation_technical_deviations",
        "require_zero_first_32_or_all_source_deviations",
        "require_zero_macro_deviations",
        "unverified_health_claims",
        "waived_health_steps",
        "waiver_id",
    }
)
_AUTHORIZED_EVIDENCE_FIELDS = frozenset(
    {
        "evaluation_receipt_sha256",
        "manifest_content_sha256",
        "manifest_sha256",
        "step0_health_receipt_sha256",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "scope",
        "decision_scope",
        "created_at",
        "promotion_decision",
        "winner_selection",
        "manifest",
        "run_id",
        "model_variant",
        "candidate",
        "receipts",
        "summary",
        "acknowledged_deviations",
        "required_waiver",
        "authorization",
        "admission_git_ref",
        "content_sha256",
    }
)
_REQUIRED_WAIVER_FIELDS = frozenset({"id", "receipt_kind", "steps", "unverified_claims"})
_APPROVED_WAIVER_FIELDS = frozenset({"id", "decision", "deviation_sha256"})
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
        "evidence_report_path",
        "evidence_report_sha256",
        "evidence_report_content_sha256",
        "evidence_report_status",
        "evidence_receipts",
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
        "admission_git_ref",
        "approved_by",
        "approved_at",
        "waivers",
        "promotion_decision",
        "winner_selection",
    }
)


class SSMaxPerceptionExploratoryWaiverEvidenceError(ValueError):
    """Raised when a v9 exploratory waiver artifact violates its exact contract."""


def _exact(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            f"{name} fields differ: missing={sorted(fields - observed)}, "
            f"extra={sorted(observed - fields)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(f"{name} must be a SHA-256 digest")
    return value


def _timestamp(value: Any, *, name: str) -> datetime:
    try:
        return strict._timestamp(value, name=name)
    except (TypeError, ValueError, strict.SSMaxPerceptionDirectEvidenceError) as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error


def _load_json(path: Path, *, name: str) -> Any:
    try:
        return strict.load_json(path)
    except (OSError, strict.SSMaxPerceptionDirectEvidenceError) as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            f"Could not load {name}: {error}"
        ) from error


def _pinned_file(path: Path, expected_sha256: str, *, name: str) -> bytes:
    expected_sha256 = _sha256(expected_sha256, name=f"expected {name} SHA-256")
    path = path.expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            f"Could not read {name} {path}"
        ) from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(f"{name} differs from its explicit pin")
    return raw


def _artifact_reference(value: Any, *, name: str) -> dict[str, str]:
    reference = _exact(value, _ARTIFACT_REFERENCE_FIELDS, name=name)
    path = reference["path"]
    digest = _sha256(reference["sha256"], name=f"{name} SHA-256")
    if not isinstance(path, str) or not Path(path).expanduser().is_absolute():
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(f"{name} path must be absolute")
    _pinned_file(Path(path), digest, name=name)
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
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver authorization escapes the repository"
        )
    raw = _pinned_file(path, AUTHORIZATION_RAW_SHA256, name="exploratory waiver authorization")
    authorization = _exact(
        _load_json(path, name="exploratory waiver authorization"),
        _AUTHORIZATION_FIELDS,
        name="exploratory waiver authorization",
    )
    if raw != bridge.canonical_json_bytes(authorization, trailing_newline=True):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver authorization bytes are not canonical"
        )
    policy = _exact(
        authorization["policy"],
        _AUTHORIZATION_POLICY_FIELDS,
        name="exploratory waiver authorization policy",
    )
    expected_policy = {
        "allowed_deviation_kinds": sorted(v8.ALLOWED_DEVIATION_KINDS),
        "allowed_source_windows": sorted(v8.ALLOWED_SOURCE_WINDOWS),
        "required_evaluation_steps": list(REQUIRED_EVALUATION_STEPS),
        "required_health_steps": list(REQUIRED_HEALTH_STEPS),
        "require_zero_correct_ce_deviations": True,
        "require_zero_evaluation_technical_deviations": True,
        "require_zero_first_32_or_all_source_deviations": True,
        "require_zero_macro_deviations": True,
        "unverified_health_claims": list(UNVERIFIED_HEALTH_CLAIMS),
        "waived_health_steps": list(WAIVED_HEALTH_STEPS),
        "waiver_id": WAIVER_ID,
    }
    if dict(policy) != expected_policy:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver authorization policy differs"
        )
    expected_identity = {
        "format": AUTHORIZATION_FORMAT,
        "version": AUTHORIZATION_VERSION,
        "approved_by": AUTHORIZATION_APPROVED_BY,
        "scope": GATE_SCOPE,
        "strict_v7_preserved": True,
        "strict_v8_preserved": True,
        "promotion_decision": False,
        "winner_selection": False,
    }
    if any(
        type(authorization[key]) is not type(expected) or authorization[key] != expected
        for key, expected in expected_identity.items()
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver authorization identity or decision differs"
        )
    _timestamp(authorization["approved_at"], name="waiver authorization approved_at")
    evidence = _exact(
        authorization["authorized_evidence"],
        frozenset(strict.MODEL_VARIANTS),
        name="authorized exploratory evidence",
    )
    for model_variant in strict.MODEL_VARIANTS:
        lineage = _exact(
            evidence[model_variant],
            _AUTHORIZED_EVIDENCE_FIELDS,
            name=f"authorized {model_variant} evidence",
        )
        evaluations = _exact(
            lineage["evaluation_receipt_sha256"],
            frozenset(str(step) for step in REQUIRED_EVALUATION_STEPS),
            name=f"authorized {model_variant} evaluations",
        )
        for step in REQUIRED_EVALUATION_STEPS:
            _sha256(
                evaluations[str(step)],
                name=f"authorized {model_variant} step{step} evaluation SHA-256",
            )
        for field in (
            "manifest_content_sha256",
            "manifest_sha256",
            "step0_health_receipt_sha256",
        ):
            _sha256(lineage[field], name=f"authorized {model_variant} {field}")
    if strict.canonical_sha256(authorization) != AUTHORIZATION_CONTENT_SHA256:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver authorization semantic identity differs"
        )
    return (
        {
            "repo_relative_path": AUTHORIZATION_RELATIVE_PATH,
            "raw_sha256": AUTHORIZATION_RAW_SHA256,
            "content_sha256": AUTHORIZATION_CONTENT_SHA256,
        },
        authorization,
    )


def _git_diff_rows(repository_root: Path, parent: str, child: str) -> tuple[tuple[str, str], ...]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository_root),
                "diff",
                "--name-status",
                "--no-renames",
                f"{parent}..{child}",
                "--",
            ],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Could not inspect the exploratory waiver Git revision"
        ) from error
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Exploratory waiver Git diff contains a non-canonical change"
            )
        rows.append((parts[0], parts[1]))
    return tuple(rows)


def _validate_admission_revision(admission_git_ref: str, *, repository_root: Path) -> None:
    if not isinstance(admission_git_ref, str) or _GIT_REF_RE.fullmatch(admission_git_ref) is None:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory admission Git ref is malformed"
        )
    try:
        strict._require_exact_git_parent(
            repository_root=repository_root,
            child_ref=admission_git_ref,
            parent_ref=ADMISSION_PARENT_GIT_REF,
            name="exploratory health-waiver evidence revision",
        )
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    rows = _git_diff_rows(repository_root, ADMISSION_PARENT_GIT_REF, admission_git_ref)
    if frozenset(rows) != ADMISSION_GIT_DIFF or len(rows) != len(ADMISSION_GIT_DIFF):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory health-waiver revision diff differs from its exact protocol surface"
        )


def _validate_admission_or_joint_checkout(admission_git_ref: str, *, repository_root: Path) -> str:
    root = repository_root.expanduser().resolve()
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Could not verify the exploratory waiver checkout"
        ) from error
    if dirty:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver checkout is not clean"
        )
    _validate_admission_revision(admission_git_ref, repository_root=root)
    if head == admission_git_ref:
        return head
    try:
        strict._require_exact_git_parent(
            repository_root=root,
            child_ref=head,
            parent_ref=admission_git_ref,
            name="exploratory waiver joint-consumer revision",
        )
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    rows = _git_diff_rows(root, admission_git_ref, head)
    if frozenset(rows) != strict.JOINT_CONSUMER_GIT_DIFF or len(rows) != len(
        strict.JOINT_CONSUMER_GIT_DIFF
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver joint-consumer diff differs from the two profiles and allowlist"
        )
    try:
        strict._validate_joint_consumer_profiles(root)
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    return head


def _manifest_reference(value: Any, *, name: str) -> dict[str, str]:
    reference = _exact(value, _MANIFEST_REFERENCE_FIELDS, name=name)
    path = reference["path"]
    if not isinstance(path, str) or not Path(path).expanduser().is_absolute():
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(f"{name} path must be absolute")
    return {
        "path": path,
        "sha256": _sha256(reference["sha256"], name=f"{name} raw SHA-256"),
        "content_sha256": _sha256(reference["content_sha256"], name=f"{name} content SHA-256"),
    }


def _validate_live_manifest(
    manifest: Mapping[str, Any],
    *,
    admission_git_ref: str,
    repository_root: Path,
) -> dict[int, dict[str, Any]]:
    """Run every strict live-manifest check with only checkout-shape validation substituted."""

    root = repository_root.expanduser().resolve()
    _validate_admission_or_joint_checkout(admission_git_ref, repository_root=root)
    try:
        training_git = strict._git_identity(manifest["training_git"], name="manifest training Git")
        evidence_git = strict._git_identity(manifest["evidence_git"], name="manifest evidence Git")
        if evidence_git["ref"] != ADMISSION_PARENT_GIT_REF:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Exploratory waiver manifest is not bound to the approved v8 evidence revision"
            )
        strict.validate_evidence_git_compatibility(
            training_git=training_git,
            evidence_git=evidence_git,
            repository_root=root,
        )
        producers = strict._producer_references(
            manifest["producers"], evidence_git_ref=evidence_git["ref"]
        )
        training_recipe = strict._source_reference(
            manifest["training_recipe"],
            name="manifest training recipe",
            expected_git_ref=training_git["ref"],
        )
        run = strict._exact(manifest["run"], strict._RUN_FIELDS, name="direct run")
        profile = strict._source_reference(
            run["training_profile"],
            name="manifest training profile",
            expected_git_ref=training_git["ref"],
        )
        targets = strict._exact(
            manifest["loss_mass_targets"],
            frozenset(strict.SOURCES),
            name="direct loss-mass targets",
        )
        projection = strict.paired._validate_single_response_binding(
            manifest["single_response_projection"], verify_live=True
        )
        bridge_parent = strict._exact(
            manifest["bridge_parent"],
            strict.paired._BRIDGE_PARENT_FIELDS,
            name="direct bridge parent",
        )
        amendment_reference = strict._authorized_amendment_reference(
            manifest["protocol_amendment"],
            repository_root=root,
            require_content_sha=True,
        )

        checkpoints: dict[int, dict[str, Any]] = {}
        for step in REQUIRED_EVALUATION_STEPS:
            checkpoints[step] = dict(
                strict._checkpoint_reference(
                    run["checkpoints"][str(step)],
                    step=step,
                    verify_live=True,
                    workers=8,
                )
            )

        actual_recipe = strict._git_blob_reference(
            git=training_git,
            repository_root=root,
            repo_relative_path=training_recipe["repo_relative_path"],
            require_live_equal=False,
        )
        actual_profile = strict._git_blob_reference(
            git=training_git,
            repository_root=root,
            repo_relative_path=profile["repo_relative_path"],
            require_live_equal=False,
        )
        if actual_recipe != training_recipe or actual_profile != profile:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError("Direct training Git blobs differ")
        actual_producers = {
            name: strict._git_blob_reference(
                git=evidence_git,
                repository_root=root,
                repo_relative_path=relative,
                require_live_equal=True,
            )
            for name, relative in strict.PRODUCER_RELATIVE_PATHS.items()
        }
        if actual_producers != producers:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Direct E4 producer Git blobs differ"
            )

        config = strict._mapping(
            strict.load_json(Path(run["checkpoint_root"]) / "step0" / "config.json"),
            name="live direct step0 config",
        )
        pseudo_spec = {
            "model_variant": manifest["model_variant"],
            "run_name": run["run_name"],
            "checkpoint_root": run["checkpoint_root"],
            "training_git": training_git,
            "topology": manifest["topology"],
            "policy": manifest["policy"],
        }
        saved_summary = strict.validate_saved_config(
            config, spec=pseudo_spec, training_profile=profile
        )
        if (
            saved_summary["data_contract_sha256"] != run["data_contract_sha256"]
            or saved_summary["trainable_contract_sha256"] != run["trainable_contract_sha256"]
            or saved_summary["loss_mass_targets"] != dict(targets)
        ):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Live direct saved contracts differ"
            )
        if strict.paired._single_response_binding_from_config(config) != dict(projection):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Live direct projection binding differs"
            )
        saved_data = strict._mapping(config.get("data"), name="live direct perception data")
        if (
            Path(str(saved_data.get("perception_provenance_path"))).expanduser().resolve()
            != Path(str(manifest["perception_provenance"]["path"])).expanduser().resolve()
            or saved_data.get("perception_provenance_sha256")
            != manifest["perception_provenance"]["sha256"]
            or Path(str(saved_data.get("source_audit_path"))).expanduser().resolve()
            != Path(str(manifest["source_audit"]["path"])).expanduser().resolve()
            or saved_data.get("source_audit_fingerprint") != manifest["source_audit_fingerprint"]
        ):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Live direct saved data artifacts differ from the manifest"
            )

        strict.paired._validate_calibration_git_blobs(
            {
                "repo": training_git["repo"],
                "repo_url": training_git["repo_url"],
                "ref": training_git["ref"],
            },
            recipe_path=root / training_recipe["repo_relative_path"],
            calibration=strict._mapping(
                projection["calibration"], name="direct projection calibration reference"
            ),
        )
        actual_parent = strict.paired._validate_bridge_parent(
            {strict.paired.CONTROL_ARM: config, strict.paired.TREATMENT_ARM: config},
            model_variant=str(manifest["model_variant"]),
            gate_reference=bridge_parent["gate"],
            verify_live_checkpoint=True,
        )
        if actual_parent != dict(bridge_parent):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError("Live direct bridge parent differs")

        provenance_path = strict.paired.validate_artifact_reference(
            {
                "path": manifest["perception_provenance"]["path"],
                "sha256": manifest["perception_provenance"]["sha256"],
            },
            name="direct perception provenance",
        )
        provenance_payload = strict._mapping(
            strict.load_json(provenance_path), name="direct perception provenance"
        )
        provenance_content = strict._sha(
            provenance_payload.get("content_sha256"),
            name="direct perception provenance content SHA-256",
        )
        if (
            provenance_content != manifest["perception_provenance"]["content_sha256"]
            or strict.canonical_sha256(
                {
                    field: item
                    for field, item in provenance_payload.items()
                    if field != "content_sha256"
                }
            )
            != provenance_content
        ):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Live direct perception provenance semantic SHA-256 differs"
            )
        typed_provenance = strict.paired.load_perception_provenance_manifest(
            provenance_path,
            expected_sha256=manifest["perception_provenance"]["sha256"],
            verify_finevision_materialization=False,
            load_image_path_signatures=False,
        )
        audit_path = strict.paired.validate_artifact_reference(
            manifest["source_audit"], name="direct source audit"
        )
        audit = strict._mapping(strict.load_json(audit_path), name="direct source audit")
        unsigned_audit = dict(audit)
        recorded_fingerprint = unsigned_audit.pop("fingerprint", None)
        if (
            recorded_fingerprint != manifest["source_audit_fingerprint"]
            or strict.canonical_sha256(unsigned_audit) != recorded_fingerprint
        ):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Live direct source-audit fingerprint differs"
            )
        sentinel_path = strict.paired.validate_artifact_reference(
            manifest["text_sentinel"], name="direct native text sentinel"
        )
        sentinel = strict.paired._validate_text_sentinel(sentinel_path)
        saved_artifacts = strict._mapping(config.get("artifacts"), name="live direct artifacts")
        if sentinel["tokenizer"] != {
            "identifier": saved_artifacts.get("tokenizer_id"),
            "revision": saved_artifacts.get("tokenizer_revision"),
        }:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Live direct text sentinel tokenizer differs from the saved config"
            )
        strict.paired._validate_attention_probe_reference(
            manifest["attention_probe"],
            provenance=typed_provenance,
            projection_contract=strict._mapping(
                manifest["single_response_projection"],
                name="direct single-response projection",
            )["contract"],
            verify_live=True,
        )
        for source in strict.SOURCES:
            selection = typed_provenance.selection(source, "validation")
            strict.paired._validate_pairing_reference(
                manifest["pairings"][source],
                source=source,
                evaluation=manifest["evaluation"],
                verify_live=True,
                dataset_size=len(selection.indices),
                expected_content_ids_sha256=strict.paired.content_ids_sha256(
                    selection.row_image_content_sha256
                ),
            )
        strict._authorized_amendment_reference(
            amendment_reference,
            repository_root=root,
            require_content_sha=True,
        )
    except (
        strict.SSMaxPerceptionDirectEvidenceError,
        strict.paired.SSMaxPerceptionEvidenceError,
    ) as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    return checkpoints


def _receipt_map(
    value: Any, *, required_steps: tuple[int, ...], name: str
) -> dict[int, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(f"{name} must be an object")
    output: dict[int, dict[str, str]] = {}
    for raw_step, raw_reference in value.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                f"{name} step {raw_step!r} is invalid"
            ) from error
        if step in output:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(f"{name} repeats step{step}")
        output[step] = _artifact_reference(raw_reference, name=f"{name} step{step}")
    if set(output) != set(required_steps):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            f"{name} must contain exactly steps {list(required_steps)}"
        )
    return output


def _authorized_deviations(value: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not value:
        return []
    try:
        return v8._validate_exploratory_deviations(value)
    except v8.SSMaxPerceptionExploratoryEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error


def _evaluation_summary(
    *,
    manifest: Mapping[str, Any],
    evaluation_payloads: Mapping[int, Mapping[str, Any]],
    evaluations: Mapping[int, Mapping[str, list[Mapping[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    baseline_state = evaluation_payloads[0]["state"]
    baseline_text = strict._direct_text_comparison_invariants(
        evaluation_payloads[0]["text_sentinel"]
    )
    for step in REQUIRED_EVALUATION_STEPS:
        state = evaluation_payloads[step]["state"]
        for surface in ("frozen_lm", "non_image_embedding_rows"):
            if (
                state[surface]["mismatch_count"] != 0
                or state[surface]["reference_inventory_sha256"]
                != state[surface]["candidate_inventory_sha256"]
                or state[surface]["reference_inventory_sha256"]
                != baseline_state[surface]["reference_inventory_sha256"]
            ):
                raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                    f"Direct step{step} frozen state differs"
                )
        if (
            strict._direct_text_comparison_invariants(evaluation_payloads[step]["text_sentinel"])
            != baseline_text
        ):
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                f"Direct step{step} native text sentinel invariants differ"
            )

    baseline_attention = evaluation_payloads[0]["attention_diagnostics"]
    attention_trajectory: dict[str, Any] = {
        "0": {
            "report_sha256": baseline_attention["report_sha256"],
            "comparison_from_step0": None,
        }
    }
    for step in REQUIRED_EVALUATION_STEPS[1:]:
        candidate_attention = evaluation_payloads[step]["attention_diagnostics"]
        try:
            comparison = compare_ssmax_attention_reports(baseline_attention, candidate_attention)
        except ValueError as error:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                f"Could not compare direct step{step} attention diagnostics: {error}"
            ) from error
        attention_trajectory[str(step)] = {
            "report_sha256": candidate_attention["report_sha256"],
            "comparison_from_step0": comparison,
        }

    policy = manifest["policy"]
    samples = int(manifest["evaluation"]["bootstrap_samples"])
    base_seed = int(manifest["evaluation"]["bootstrap_seed"])
    deviations: list[dict[str, Any]] = []
    windows: dict[str, Any] = {}
    for window_index, window in enumerate(strict.WINDOWS):
        arrays = {
            source: strict._metric_arrays(evaluations, source=source, window=window)
            for source in strict.SOURCES
        }
        absolute_values = {source: values["step4000_gap"] for source, values in arrays.items()}
        improvement_values = {
            source: values["step4000_gap"] - values["step0_gap"]
            for source, values in arrays.items()
        }
        absolute = strict._source_balanced_interval(
            absolute_values,
            seed=base_seed + window_index * 10_000,
            samples=samples,
        )
        improvement = strict._source_balanced_interval(
            improvement_values,
            seed=base_seed + 100_000 + window_index * 10_000,
            samples=samples,
        )
        source_summary: dict[str, Any] = {}
        for source_index, (source, values) in enumerate(arrays.items()):
            baseline_ce = float(values["step0_correct_ce"].mean())
            candidate_ce = float(values["step4000_correct_ce"].mean())
            baseline_gap = float(values["step0_gap"].mean())
            durability_gap = float(values["step3000_gap"].mean())
            candidate_gap = float(values["step4000_gap"].mean())
            source_absolute = bridge.summarize_paired_values(
                absolute_values[source],
                seed=base_seed + 200_000 + window_index * 10_000 + source_index,
                samples=samples,
            )
            source_improvement = bridge.summarize_paired_values(
                improvement_values[source],
                seed=base_seed + 300_000 + window_index * 10_000 + source_index,
                samples=samples,
            )
            source_summary[source] = {
                "candidate_absolute_gap": source_absolute,
                "candidate_gap_improvement_from_step0": source_improvement,
                "step0_correct_ce": baseline_ce,
                "step4000_correct_ce": candidate_ce,
                "step0_gap": baseline_gap,
                "step3000_gap": durability_gap,
                "step4000_gap": candidate_gap,
            }
            if source_absolute["mean_bootstrap_ci"]["low"] <= float(
                policy["candidate_gap_lower_ci_minimum"]
            ):
                deviations.append(
                    {
                        "kind": "source_nonpositive_absolute_gap",
                        "source": source,
                        "window": window,
                    }
                )
            if source_improvement["mean_bootstrap_ci"]["low"] <= float(
                policy["candidate_gap_improvement_lower_ci_minimum"]
            ):
                deviations.append(
                    {
                        "kind": "source_nonpositive_gap_improvement",
                        "source": source,
                        "window": window,
                    }
                )
            if candidate_ce > float(policy["maximum_correct_ce_fraction_of_step0"]) * baseline_ce:
                deviations.append(
                    {"kind": "source_correct_ce_regression", "source": source, "window": window}
                )
            if candidate_gap < float(policy["minimum_gap_retention"]) * durability_gap:
                deviations.append(
                    {"kind": "source_gap_retention", "source": source, "window": window}
                )

        macro_baseline_ce = float(
            np.mean([values["step0_correct_ce"].mean() for values in arrays.values()])
        )
        macro_candidate_ce = float(
            np.mean([values["step4000_correct_ce"].mean() for values in arrays.values()])
        )
        macro_baseline_gap = float(
            np.mean([values["step0_gap"].mean() for values in arrays.values()])
        )
        macro_durability_gap = float(
            np.mean([values["step3000_gap"].mean() for values in arrays.values()])
        )
        macro_candidate_gap = float(
            np.mean([values["step4000_gap"].mean() for values in arrays.values()])
        )
        if absolute["ci"]["low"] <= float(policy["candidate_gap_lower_ci_minimum"]):
            deviations.append({"kind": "macro_absolute_gap_lower_ci", "window": window})
        if improvement["ci"]["low"] <= float(policy["candidate_gap_improvement_lower_ci_minimum"]):
            deviations.append({"kind": "macro_gap_improvement_lower_ci", "window": window})
        if (
            macro_candidate_ce
            > float(policy["maximum_correct_ce_fraction_of_step0"]) * macro_baseline_ce
        ):
            deviations.append({"kind": "macro_correct_ce_regression", "window": window})
        if macro_candidate_gap < float(policy["minimum_gap_retention"]) * macro_durability_gap:
            deviations.append({"kind": "macro_gap_retention", "window": window})
        windows[window] = {
            "candidate_absolute_gap": absolute,
            "candidate_gap_improvement_from_step0": improvement,
            "macro_step0_correct_ce": macro_baseline_ce,
            "macro_step4000_correct_ce": macro_candidate_ce,
            "macro_step0_gap": macro_baseline_gap,
            "macro_step3000_gap": macro_durability_gap,
            "macro_step4000_gap": macro_candidate_gap,
            "sources": source_summary,
        }
    return (
        {"windows": windows, "attention_trajectory": attention_trajectory},
        _authorized_deviations(deviations),
    )


def _required_waiver() -> dict[str, Any]:
    return {
        "id": WAIVER_ID,
        "receipt_kind": "health",
        "steps": list(WAIVED_HEALTH_STEPS),
        "unverified_claims": list(UNVERIFIED_HEALTH_CLAIMS),
    }


def _approved_waivers(required_waiver: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "id": WAIVER_ID,
            "decision": "approved_for_exploratory_joint_only",
            "deviation_sha256": strict.canonical_sha256(required_waiver),
        }
    ]


def build_evidence_report(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    evaluation_receipts: Mapping[int, Mapping[str, str]],
    health_receipts: Mapping[int, Mapping[str, str]],
    created_at: str,
    admission_git_ref: str | None = None,
) -> dict[str, Any]:
    """Build one immutable research report from three evaluations and step-0 health."""

    root = Path(__file__).resolve().parents[3]
    if admission_git_ref is None:
        try:
            admission_git_ref = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Could not resolve the exploratory admission Git ref"
            ) from error
    _validate_admission_or_joint_checkout(admission_git_ref, repository_root=root)

    manifest_path = manifest_path.expanduser().resolve()
    _pinned_file(manifest_path, expected_manifest_sha256, name="direct manifest")
    try:
        manifest = strict.load_manifest(manifest_path, verify_live=False)
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    checkpoints = _validate_live_manifest(
        manifest,
        admission_git_ref=admission_git_ref,
        repository_root=root,
    )
    authorization_ref, authorization = _authorization_reference(root)
    authorized = authorization["authorized_evidence"][manifest["model_variant"]]
    if (
        expected_manifest_sha256 != authorized["manifest_sha256"]
        or manifest["content_sha256"] != authorized["manifest_content_sha256"]
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Direct manifest is not the exact authorized exploratory lineage"
        )

    evaluation_refs = _receipt_map(
        evaluation_receipts,
        required_steps=REQUIRED_EVALUATION_STEPS,
        name="direct evaluation receipts",
    )
    health_refs = _receipt_map(
        health_receipts,
        required_steps=REQUIRED_HEALTH_STEPS,
        name="direct health receipts",
    )
    expected_evaluations = authorized["evaluation_receipt_sha256"]
    if (
        any(
            evaluation_refs[step]["sha256"] != expected_evaluations[str(step)]
            for step in REQUIRED_EVALUATION_STEPS
        )
        or health_refs[0]["sha256"] != authorized["step0_health_receipt_sha256"]
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Receipt set is not the exact checked-in exploratory authorization"
        )

    report_time = _timestamp(created_at, name="exploratory evidence report created_at")
    manifest_time = _timestamp(manifest["created_at"], name="direct manifest created_at")
    authorization_time = _timestamp(
        authorization["approved_at"], name="waiver authorization approved_at"
    )
    if report_time < manifest_time or report_time < authorization_time:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory evidence report predates its manifest or authorization"
        )
    evaluation_payloads: dict[int, Mapping[str, Any]] = {}
    evaluations: dict[int, Mapping[str, list[Mapping[str, Any]]]] = {}
    receipt_output: dict[str, Any] = {}
    try:
        for step in REQUIRED_EVALUATION_STEPS:
            _, evaluation = strict._load_receipt_reference(
                evaluation_refs[step],
                manifest=manifest,
                manifest_path=manifest_path,
                step=step,
                expected_format=strict.EVALUATION_RECEIPT_FORMAT,
            )
            evaluation_time = _timestamp(
                evaluation["created_at"], name=f"direct step{step} evaluation created_at"
            )
            if evaluation_time < manifest_time or evaluation_time > report_time:
                raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                    "Direct evaluation ordering differs from manifest <= receipt <= report"
                )
            evaluations[step] = strict._validate_direct_evaluation_receipt(
                evaluation, manifest=manifest, step=step
            )
            if evaluation["status"] != "passed":
                raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                    f"Direct step{step} evaluation did not pass"
                )
            evaluation_payloads[step] = evaluation
            receipt_output[str(step)] = {"evaluation": dict(evaluation_refs[step])}

        _, health = strict._load_receipt_reference(
            health_refs[0],
            manifest=manifest,
            manifest_path=manifest_path,
            step=0,
            expected_format=strict.HEALTH_RECEIPT_FORMAT,
        )
        health_time = _timestamp(health["created_at"], name="direct step0 health created_at")
        if health_time < manifest_time or health_time > report_time:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                "Direct step0 health ordering differs from manifest <= receipt <= report"
            )
        health_summary = strict._validate_direct_health_receipt(health, manifest=manifest, step=0)
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    if health["status"] != "passed":
        raise SSMaxPerceptionExploratoryWaiverEvidenceError("Direct step0 health did not pass")
    receipt_output["0"]["health"] = dict(health_refs[0])
    for counter, maximum in (
        ("data_errors", int(manifest["policy"]["maximum_data_errors"])),
        ("optimizer_guard_skips", int(manifest["policy"]["maximum_optimizer_guard_skips"])),
        ("nonfinite_losses", int(manifest["policy"]["maximum_nonfinite_losses"])),
        ("nonfinite_gradients", int(manifest["policy"]["maximum_nonfinite_gradients"])),
    ):
        if int(health_summary["run_counters"][counter]) > maximum:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                f"Direct step0 health counter {counter} exceeds its strict maximum"
            )

    evaluation_summary, deviations = _evaluation_summary(
        manifest=manifest,
        evaluation_payloads=evaluation_payloads,
        evaluations=evaluations,
    )
    evaluation_summary["step0_health"] = {
        "receipt_status": "passed",
        "run_counters": copy.deepcopy(health_summary["run_counters"]),
    }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": REPORT_STATUS,
        "scope": GATE_SCOPE,
        "decision_scope": DECISION_SCOPE,
        "created_at": created_at,
        "promotion_decision": False,
        "winner_selection": False,
        "manifest": strict.manifest_reference(manifest_path, manifest),
        "run_id": manifest["run_id"],
        "model_variant": manifest["model_variant"],
        "candidate": copy.deepcopy(checkpoints[4000]),
        "receipts": receipt_output,
        "summary": evaluation_summary,
        "acknowledged_deviations": deviations,
        "required_waiver": _required_waiver(),
        "authorization": authorization_ref,
        "admission_git_ref": admission_git_ref,
    }
    report["content_sha256"] = strict.canonical_sha256(report)
    return report


def validate_evidence_report_reference(
    value: Any, *, verify_live_checkpoint: bool = True
) -> Mapping[str, Any]:
    """Reopen and exactly rebuild one v9 research report from its four raw receipts."""

    if not verify_live_checkpoint:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver admission requires live checkpoint and Git verification"
        )
    reference = _artifact_reference(value, name="exploratory waiver evidence report")
    report_path = Path(reference["path"]).expanduser().resolve()
    report = _exact(
        _load_json(report_path, name="exploratory waiver evidence report"),
        _REPORT_FIELDS,
        name="exploratory waiver evidence report",
    )
    expected_values = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": REPORT_STATUS,
        "scope": GATE_SCOPE,
        "decision_scope": DECISION_SCOPE,
        "promotion_decision": False,
        "winner_selection": False,
    }
    if any(
        type(report[key]) is not type(expected) or report[key] != expected
        for key, expected in expected_values.items()
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver report identity or decision differs"
        )
    _timestamp(report["created_at"], name="exploratory evidence report created_at")
    try:
        strict._content_sha(report, name="exploratory waiver evidence report")
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    manifest_ref = _manifest_reference(report["manifest"], name="exploratory report manifest")
    receipts = _exact(
        report["receipts"],
        frozenset(str(step) for step in REQUIRED_EVALUATION_STEPS),
        name="exploratory report receipts",
    )
    evaluation_refs: dict[int, Mapping[str, str]] = {}
    for step in REQUIRED_EVALUATION_STEPS:
        expected_fields = (
            frozenset({"evaluation", "health"}) if step == 0 else frozenset({"evaluation"})
        )
        step_receipts = _exact(
            receipts[str(step)], expected_fields, name=f"exploratory report step{step} receipts"
        )
        evaluation_refs[step] = _exact(
            step_receipts["evaluation"],
            _ARTIFACT_REFERENCE_FIELDS,
            name=f"exploratory report step{step} evaluation",
        )
    health_refs = {
        0: _exact(
            receipts["0"]["health"],
            _ARTIFACT_REFERENCE_FIELDS,
            name="exploratory report step0 health",
        )
    }
    rebuilt = build_evidence_report(
        manifest_path=Path(manifest_ref["path"]),
        expected_manifest_sha256=manifest_ref["sha256"],
        evaluation_receipts=evaluation_refs,
        health_receipts=health_refs,
        created_at=str(report["created_at"]),
        admission_git_ref=str(report["admission_git_ref"]),
    )
    if rebuilt != dict(report):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver report differs from its four bound raw receipts"
        )
    return {
        "report": report,
        "report_reference": reference,
        "manifest": strict.load_manifest(Path(manifest_ref["path"]), verify_live=False),
        "manifest_reference": manifest_ref,
        "candidate": dict(report["candidate"]),
        "authorization": _authorization_reference()[1],
        "acknowledged_deviations": copy.deepcopy(report["acknowledged_deviations"]),
    }


def build_parent_gate(
    *,
    evidence_report_path: Path,
    expected_evidence_report_sha256: str,
    approved_by: str,
    approved_at: str,
) -> dict[str, Any]:
    """Build one explicitly approved, research-only version-9 parent gate."""

    report_path = evidence_report_path.expanduser().resolve()
    summary = validate_evidence_report_reference(
        {"path": str(report_path), "sha256": expected_evidence_report_sha256}
    )
    authorization_ref, authorization = _authorization_reference()
    if (
        not isinstance(approved_by, str)
        or _DURABLE_IDENTITY_RE.fullmatch(approved_by) is None
        or approved_by != authorization["approved_by"]
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "approved_by must match the durable waiver authorization identity"
        )
    approval_time = _timestamp(approved_at, name="exploratory waiver gate approved_at")
    report_time = _timestamp(summary["report"]["created_at"], name="evidence report created_at")
    authorization_time = _timestamp(
        authorization["approved_at"], name="waiver authorization approved_at"
    )
    if approval_time < report_time or approval_time < authorization_time:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate approval predates its report or authorization"
        )
    candidate = summary["candidate"]
    manifest = summary["manifest"]
    manifest_ref = summary["manifest_reference"]
    amendment = manifest["protocol_amendment"]
    try:
        recipe_version, formatter_version = strict._candidate_metadata(candidate)
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    required_waiver = _exact(
        summary["report"]["required_waiver"],
        _REQUIRED_WAIVER_FIELDS,
        name="required exploratory health waiver",
    )
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
        "metrics_artifact_sha256": expected_evidence_report_sha256,
        "evidence_report_path": str(report_path),
        "evidence_report_sha256": expected_evidence_report_sha256,
        "evidence_report_content_sha256": summary["report"]["content_sha256"],
        "evidence_report_status": REPORT_STATUS,
        "evidence_receipts": copy.deepcopy(summary["report"]["receipts"]),
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
        "admission_git_ref": summary["report"]["admission_git_ref"],
        "approved_by": approved_by,
        "approved_at": approved_at,
        "waivers": _approved_waivers(required_waiver),
        "promotion_decision": False,
        "winner_selection": False,
    }
    validate_ssmax_perception_exploratory_waiver_parent_gate(
        gate,
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(manifest["run"]["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(manifest["run"]["trainable_contract_sha256"]),
    )
    return gate


def validate_ssmax_perception_exploratory_waiver_parent_gate(
    gate: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_data_contract_sha256: str,
    expected_trainable_contract_sha256: str,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Validate one exact, research-only version-9 perception parent gate."""

    if not verify_live_checkpoint:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver eligibility requires live checkpoint and Git verification"
        )
    value = _exact(gate, _GATE_FIELDS, name="SSMax exploratory waiver parent gate")
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
        "evidence_report_status": REPORT_STATUS,
        "promotion_decision": False,
        "winner_selection": False,
    }
    for key, expected in expected_pairs.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise SSMaxPerceptionExploratoryWaiverEvidenceError(
                f"SSMax exploratory waiver parent gate {key} differs"
            )
    if (
        expected_model_variant not in strict.MODEL_VARIANTS
        or not isinstance(value["run_id"], str)
        or not value["run_id"]
        or Path(str(value["checkpoint"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or expected_checkpoint.name != "step4000"
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "SSMax exploratory waiver gate names an incompatible direct step4000 lineage"
        )
    for key in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "metrics_artifact_sha256",
        "evidence_report_sha256",
        "evidence_report_content_sha256",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
    ):
        _sha256(value[key], name=f"SSMax exploratory waiver gate {key}")
    if value["metrics_artifact_sha256"] != value["evidence_report_sha256"]:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver metrics artifact differs from its evidence report"
        )
    authorization_ref = _exact(
        value["authorization"],
        _AUTHORIZATION_REFERENCE_FIELDS,
        name="exploratory waiver gate authorization",
    )
    live_authorization_ref, authorization = _authorization_reference()
    if dict(authorization_ref) != live_authorization_ref:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate authorization reference differs"
        )
    summary = validate_evidence_report_reference(
        {
            "path": value["evidence_report_path"],
            "sha256": value["evidence_report_sha256"],
        }
    )
    report = summary["report"]
    manifest = summary["manifest"]
    manifest_ref = summary["manifest_reference"]
    candidate = summary["candidate"]
    amendment = manifest["protocol_amendment"]
    required_waiver = _exact(
        report["required_waiver"],
        _REQUIRED_WAIVER_FIELDS,
        name="required exploratory health waiver",
    )
    waivers = value["waivers"]
    if (
        not isinstance(waivers, list)
        or len(waivers) != 1
        or set(_exact(waivers[0], _APPROVED_WAIVER_FIELDS, name="approved health waiver"))
        != _APPROVED_WAIVER_FIELDS
        or waivers != _approved_waivers(required_waiver)
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate does not approve exactly its missing health evidence"
        )
    if (
        report["content_sha256"] != value["evidence_report_content_sha256"]
        or report["receipts"] != value["evidence_receipts"]
        or report["acknowledged_deviations"] != value["acknowledged_deviations"]
        or report["admission_git_ref"] != value["admission_git_ref"]
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
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate report, lineage, receipts, Git, or amendment binding differs"
        )
    try:
        recipe_version, formatter_version = strict._candidate_metadata(candidate)
    except strict.SSMaxPerceptionDirectEvidenceError as error:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(str(error)) from error
    if value["recipe_version"] != recipe_version or value["formatter_version"] != formatter_version:
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate recipe or formatter identity differs"
        )
    approved_by = value["approved_by"]
    if (
        not isinstance(approved_by, str)
        or _DURABLE_IDENTITY_RE.fullmatch(approved_by) is None
        or approved_by != authorization["approved_by"]
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate approved_by differs from its authorization"
        )
    approval_time = _timestamp(value["approved_at"], name="exploratory waiver gate approved_at")
    if approval_time < _timestamp(report["created_at"], name="evidence report created_at") or (
        approval_time
        < _timestamp(authorization["approved_at"], name="waiver authorization approved_at")
    ):
        raise SSMaxPerceptionExploratoryWaiverEvidenceError(
            "Exploratory waiver gate approval predates its report or authorization"
        )
    return summary


__all__ = [
    "ADMISSION_GIT_DIFF",
    "ADMISSION_PARENT_GIT_REF",
    "AUTHORIZATION_CONTENT_SHA256",
    "AUTHORIZATION_RAW_SHA256",
    "GATE_SCOPE",
    "PARENT_GATE_VERSION",
    "REPORT_FORMAT",
    "REPORT_STATUS",
    "WAIVED_HEALTH_STEPS",
    "WAIVER_ID",
    "SSMaxPerceptionExploratoryWaiverEvidenceError",
    "build_evidence_report",
    "build_parent_gate",
    "validate_evidence_report_reference",
    "validate_ssmax_perception_exploratory_waiver_parent_gate",
]
