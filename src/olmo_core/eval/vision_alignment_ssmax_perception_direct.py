"""Fail-closed evidence for one direct SSMax perception lineage.

This protocol is additive.  It does not reinterpret the historical paired perception v1/v2
formats and it does not use a frozen-vision run.  One prospectively authorized, vision-unfrozen
run is evaluated at fixed steps 0, 3000, and 4000.  The training checkout and the later evidence
checkout have separate immutable Git identities so an already-running checkpoint can be audited
by a protocol that was added after the training source was frozen.

The resulting report supports a human-approved, waiver-free version-7 parent gate.  Its visual
claims are deliberately within-lineage: positive absolute correct-vs-wrong-image separation,
positive improvement from the run's own step 0, correct-image CE non-regression relative to that
same step 0, and late-trajectory retention.  It makes no causal claim about unfreezing the vision
encoder and never selects a winner between model variants.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.eval import vision_alignment_ssmax_perception as paired
from olmo_core.eval.ssmax_attention_diagnostics import compare_ssmax_attention_reports

MANIFEST_SPEC_FORMAT = "vision_alignment_ssmax_perception_direct_manifest_spec"
MANIFEST_FORMAT = "vision_alignment_ssmax_perception_direct_manifest"
EVALUATION_RECEIPT_FORMAT = "vision_alignment_ssmax_perception_direct_evaluation_receipt"
HEALTH_RECEIPT_FORMAT = "vision_alignment_ssmax_perception_direct_health_receipt"
PROMOTION_REPORT_FORMAT = "vision_alignment_ssmax_perception_direct_promotion_report"
DIRECT_TEXT_SENTINEL_PROTOCOL = "same-rank-step0-candidate-native-text-exact-v1"

SCHEMA_VERSION = 1
PARENT_GATE_VERSION = 7
LINEAGE_KIND = "direct_vision_unfrozen"
REQUIRED_STEPS = (0, 3000, 4000)
SOURCES = paired.SOURCES
WINDOWS = paired.WINDOWS
MODEL_VARIANTS = paired.MODEL_VARIANTS
TRAINING_ROLE = paired.TREATMENT_ARM

AMENDMENT_FORMAT = "vision_alignment_ssmax_perception_single_lineage_amendment"
AMENDMENT_VERSION = 1
AMENDMENT_RELATIVE_PATH = (
    "configs/vision_moe/vision_alignment/eval/" "ssmax_perception_single_lineage_amendment_v1.json"
)
AMENDMENT_SHA256 = "a67735449d44ee52c62a3562cf0925b534f916433779203ceedaa177cd1f5364"
AMENDMENT_RECORDED_AT = "2026-08-22T01:53:30Z"
AMENDMENT_APPROVED_BY = "rustins"
TRAINING_GIT_BRANCH = "rustin/vision-ssmax-molmofication"
TRAINING_GIT_REF = "1826b78105858a2163cb7689b151fadefa538bbc"
BASE_EVIDENCE_GIT_REF = "8bf81eb368e1ad33a5570cba4f5f5fff236760c3"
LEGACY_EVIDENCE_GIT_REFS = frozenset(
    {
        "86f170e72d68ffb46ed5f04390dc901cd41d3097",
        "197407c9db0aa1eb1120ed416ed002b236f113da",
        BASE_EVIDENCE_GIT_REF,
    }
)

DIRECT_RUN_IDENTITIES: Mapping[str, Mapping[str, str]] = {
    "ssmax_head_qknorm": {
        "run_name": "vision-ssmax-head-qknorm-1p4b-cx8-perception-treatment-v2",
        "checkpoint_root": (
            "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
            "vision-alignment/checkpoints/"
            "vision-ssmax-head-qknorm-1p4b-cx8-perception-treatment-v2"
        ),
        "profile": (
            "configs/vision_moe/vision_alignment/perception/"
            "ssmax_head_qknorm_1p4b_cx8_treatment_v2.yaml"
        ),
        "profile_sha256": ("5baf7aa4da2cedea44d6caed0e88882474f9821071e33ae838fe188f7d7a8a51"),
    },
    "ssmax_no_qknorm": {
        "run_name": "vision-ssmax-no-qknorm-1p4b-cx8-perception-treatment-v2",
        "checkpoint_root": (
            "/weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/"
            "vision-alignment/checkpoints/"
            "vision-ssmax-no-qknorm-1p4b-cx8-perception-treatment-v2"
        ),
        "profile": (
            "configs/vision_moe/vision_alignment/perception/"
            "ssmax_no_qknorm_1p4b_cx8_treatment_v2.yaml"
        ),
        "profile_sha256": ("834545f9352a56d539f3d377f915f58c2aca78a89965f7913164461e9551b9f0"),
    },
}

AMENDMENT_EVIDENCE_POLICY: Mapping[str, Any] = {
    "candidate_absolute_visual_gap_ci_low_must_be_positive": True,
    "candidate_visual_gap_improvement_vs_step0_ci_low_must_be_positive": True,
    "maximum_correct_ce_fraction_of_step0": 1.02,
    "maximum_data_errors": 0,
    "maximum_nonfinite_gradients": 0,
    "maximum_nonfinite_losses": 0,
    "maximum_optimizer_guard_skips": 8,
    "minimum_clean_final_steps": 128,
    "minimum_clean_steps_between_skips": 128,
    "minimum_step4000_gap_retention_fraction_of_step3000": 0.8,
    "require_finite_gradient_only_skips": True,
    "require_uninterrupted_optimizer_guard_history": True,
    "required_steps": list(REQUIRED_STEPS),
}

V2_OPTIMIZER_GUARD_CONTRACT = dict(paired.V2_OPTIMIZER_GUARD_CONTRACT)
DIRECT_POLICY: Mapping[str, Any] = {
    "baseline_step": 0,
    "durability_step": 3000,
    "candidate_step": 4000,
    "candidate_gap_lower_ci_minimum": 0.0,
    "candidate_gap_improvement_lower_ci_minimum": 0.0,
    "maximum_correct_ce_fraction_of_step0": 1.02,
    "minimum_gap_retention": 0.8,
    "loss_mass_share_tolerance": paired.LOSS_MASS_SHARE_TOLERANCE,
    "maximum_data_errors": 0,
    "optimizer_guard": V2_OPTIMIZER_GUARD_CONTRACT,
    "maximum_optimizer_guard_skips": paired.V2_MAXIMUM_OPTIMIZER_GUARD_SKIPS,
    "minimum_clean_steps_between_skips": paired.V2_MINIMUM_CLEAN_STEPS_BETWEEN_SKIPS,
    "minimum_clean_final_steps": paired.V2_MINIMUM_CLEAN_FINAL_STEPS,
    "require_finite_gradient_only_skips": True,
    "require_uninterrupted_optimizer_guard_history": True,
    "maximum_nonfinite_losses": 0,
    "maximum_nonfinite_gradients": 0,
}

EVALUATION_CONTRACT: Mapping[str, Any] = {
    "sources": list(SOURCES),
    "steps": list(REQUIRED_STEPS),
    "windows": list(WINDOWS),
    "examples_per_source": 480,
    "pairing_seed": paired.PAIRING_SEED,
    "bootstrap_seed": 1_006_201,
    "bootstrap_samples": 10_000,
    "rank_batch_instances": 2,
}
TOPOLOGY_CONTRACT: Mapping[str, Any] = {
    "world_size": 16,
    "num_nodes": 2,
    "gpus_per_node": 8,
    "data_parallel": "hsdp",
}

PROTOCOL_PRODUCER = "protocol"
EVALUATION_PRODUCER = paired.EVALUATION_PRODUCER
HEALTH_PRODUCER = paired.HEALTH_PRODUCER
PRODUCER_RELATIVE_PATHS: Mapping[str, str] = {
    PROTOCOL_PRODUCER: "src/olmo_core/eval/vision_alignment_ssmax_perception_direct.py",
    EVALUATION_PRODUCER: "src/scripts/eval/vision_alignment_ssmax_perception_direct.py",
    HEALTH_PRODUCER: "src/scripts/eval/vision_alignment_ssmax_perception_direct_health.py",
}

# The evidence checkout may add only the protocol and its narrowly scoped consumers. In
# particular, none of the active perception run's model, data, optimizer, or reviewed-profile
# sources may change between the training and evidence revisions. The training recipe itself is
# separately pinned to the training revision; the evidence revision may change only its audited
# gate-consumer surface for the subsequent joint phase.
EVIDENCE_GIT_DIFF_ALLOWLIST = frozenset(
    {
        AMENDMENT_RELATIVE_PATH,
        (
            "configs/vision_moe/vision_alignment/eval/"
            "ssmax_perception_exploratory_joint_authorization_v1.json"
        ),
        ("configs/vision_moe/vision_alignment/eval/" "SSMAX_PERCEPTION_EXPLORATORY_JOINT.md"),
        "configs/vision_moe/vision_alignment/eval/SSMAX_JOINT_EVIDENCE.md",
        "configs/vision_moe/vision_alignment/eval/SSMAX_PERCEPTION_DIRECT_EVALUATION.md",
        (
            "configs/vision_moe/vision_alignment/eval/joint/"
            "ssmax_head_qknorm_joint_manifest_v1.json.template"
        ),
        (
            "configs/vision_moe/vision_alignment/eval/joint/"
            "ssmax_no_qknorm_joint_manifest_v1.json.template"
        ),
        "configs/vision_moe/vision_alignment/eval/ssmax_perception_direct_manifest_v1.json.template",
        "configs/vision_moe/vision_alignment/README.md",
        "configs/vision_moe/vision_alignment/joint/README.md",
        "src/olmo_core/eval/vision_alignment_ssmax_perception_direct.py",
        "src/olmo_core/eval/vision_alignment_ssmax_perception_exploratory.py",
        "src/olmo_core/eval/vision_alignment_ssmax_joint.py",
        "src/scripts/beaker_launch_vision_ssmax_evidence.py",
        "src/scripts/eval/vision_alignment_ssmax_perception_direct.py",
        "src/scripts/eval/vision_alignment_ssmax_perception_direct_compare.py",
        "src/scripts/eval/vision_alignment_ssmax_perception_direct_health.py",
        "src/scripts/eval/vision_alignment_ssmax_perception_direct_manifest.py",
        "src/scripts/eval/vision_alignment_ssmax_perception_direct_promotion.py",
        "src/scripts/eval/vision_alignment_ssmax_perception_exploratory.py",
        "src/scripts/train/Vision-Alignment.py",
        "src/test/eval/vision_alignment_ssmax_perception_direct_test.py",
        "src/test/eval/vision_alignment_ssmax_perception_exploratory_test.py",
        "src/test/eval/vision_alignment_ssmax_joint_test.py",
        "src/test/scripts/beaker_launch_vision_ssmax_evidence_test.py",
        ("src/test/scripts/" "vision_alignment_ssmax_perception_direct_health_optimized_test.py"),
        "src/test/scripts/vision_alignment_test.py",
    }
)
JOINT_CONSUMER_GIT_DIFF = frozenset(
    {
        ("M", "configs/vision_moe/vision_alignment/joint/approved_profiles.json"),
        (
            "A",
            (
                "configs/vision_moe/vision_alignment/joint/"
                "ssmax_head_qknorm_1p4b_cx8_direct_v1.yaml"
            ),
        ),
        (
            "A",
            (
                "configs/vision_moe/vision_alignment/joint/"
                "ssmax_no_qknorm_1p4b_cx8_direct_v1.yaml"
            ),
        ),
    }
)
JOINT_PROFILE_ALLOWLIST_RELATIVE_PATH = (
    "configs/vision_moe/vision_alignment/joint/approved_profiles.json"
)
JOINT_PROFILE_BASELINE = {
    "configs/vision_moe/vision_alignment/joint/joint_v1.yaml": (
        "294da420f4f911fc96aad2a9eff43c59dc0831276fad5d1c0fbec37c6f78c2f5"
    )
}
JOINT_DIRECT_PROFILE_PATHS = frozenset(
    path for status, path in JOINT_CONSUMER_GIT_DIFF if status == "A"
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REF_RE = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_REF_FIELDS = frozenset({"path", "sha256", "content_sha256"})
_SOURCE_REF_FIELDS = frozenset({"repo_relative_path", "sha256", "git_ref"})
_GIT_FIELDS = frozenset({"repo", "repo_url", "branch", "ref"})
_CHECKPOINT_FIELDS = paired._CHECKPOINT_FIELDS
_AMENDMENT_REF_FIELDS = frozenset({"path", "sha256", "content_sha256"})
_SPEC_FIELDS = frozenset(
    {
        "format",
        "version",
        "run_id",
        "model_variant",
        "run_name",
        "checkpoint_root",
        "training_profile",
        "recipe",
        "training_git",
        "evidence_git",
        "protocol_amendment",
        "bridge_parent_gate",
        "perception_provenance",
        "source_audit",
        "attention_probe",
        "text_sentinel",
        "pairing_paths",
        "evaluation",
        "topology",
        "policy",
    }
)
_RUN_FIELDS = frozenset(
    {
        "run_name",
        "checkpoint_root",
        "training_profile",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "checkpoints",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "version",
        "created_at",
        "run_id",
        "lineage_kind",
        "model_variant",
        "training_git",
        "evidence_git",
        "producers",
        "training_recipe",
        "protocol_amendment",
        "bridge_parent",
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
        "run",
        "lineage_contract_sha256",
        "content_sha256",
    }
)
_EVALUATION_RECEIPT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "manifest",
        "run_id",
        "model_variant",
        "step",
        "checkpoint",
        "strict_generic_dcp_load",
        "state",
        "text_sentinel",
        "attention_diagnostics",
        "pairings",
        "results",
        "evaluator",
        "content_sha256",
    }
)
_HEALTH_RECEIPT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "manifest",
        "run_id",
        "model_variant",
        "step",
        "checkpoint",
        "rank_states",
        "sources",
        "run_counters",
        "evidence",
        "content_sha256",
    }
)
_PROMOTION_REPORT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "decision_scope",
        "created_at",
        "manifest",
        "run_id",
        "model_variant",
        "receipts",
        "summary",
        "deviations",
        "content_sha256",
    }
)
_PARENT_GATE_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
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
        "promotion_report_path",
        "promotion_report_sha256",
        "promotion_report_content_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_path",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
        "training_git_ref",
        "evidence_git_ref",
        "approved_by",
        "approved_at",
        "waivers",
    }
)
_DIRECT_TEXT_SENTINEL_INVARIANT_FIELDS = frozenset(
    {
        "protocol",
        "version",
        "artifact_sha256",
        "reference_step",
        "reference_checkpoint_identity_sha256",
        "candidate_step",
        "candidate_checkpoint_identity_sha256",
        "topology",
        "world_size",
        "input",
        "labels",
        "token_count",
        "rank_count",
    }
)
_DIRECT_TEXT_SENTINEL_RESULT_FIELDS = frozenset(
    {
        *_DIRECT_TEXT_SENTINEL_INVARIANT_FIELDS,
        "rank_rows",
        "mismatch_count",
        "all_ranks_passed",
        "rank_inventory_sha256",
        "content_sha256",
    }
)
_DIRECT_TEXT_SENTINEL_RANK_FIELDS = frozenset(
    {"rank", "reference", "candidate", "logits_exact", "ce_exact", "passed"}
)
_DIRECT_TEXT_SENTINEL_OUTPUT_FIELDS = frozenset({"logits", "ce"})
_DIRECT_TEXT_SENTINEL_INPUT_TENSOR_FIELDS = frozenset({"dtype", "shape", "numel", "sha256"})
_DIRECT_TEXT_SENTINEL_OUTPUT_TENSOR_FIELDS = frozenset(
    {*_DIRECT_TEXT_SENTINEL_INPUT_TENSOR_FIELDS, "finite"}
)


class SSMaxPerceptionDirectEvidenceError(ValueError):
    """Raised when direct single-lineage perception evidence violates its contract."""


def canonical_sha256(value: Any) -> str:
    """Return the canonical semantic SHA-256 used by direct evidence artifacts."""

    try:
        return paired.canonical_sha256(value)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def sha256_file(path: Path) -> str:
    """Return the raw SHA-256 of one file."""

    try:
        return paired.sha256_file(path)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def load_json(path: Path) -> Any:
    """Load strict JSON, rejecting duplicate keys and non-finite constants."""

    try:
        return paired.load_json(path)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create one immutable JSON artifact without overwriting."""

    try:
        paired.write_json_once(path, value)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def artifact_reference(path: Path) -> dict[str, str]:
    """Return an absolute raw-SHA artifact reference."""

    try:
        return paired.artifact_reference(path)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be an object")
    return value


def _exact(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    mapping = _mapping(value, name=name)
    if set(mapping) != set(fields):
        raise SSMaxPerceptionDirectEvidenceError(
            f"{name} fields differ: missing={sorted(fields - set(mapping))}, "
            f"extra={sorted(set(mapping) - fields)}"
        )
    return mapping


def _sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be finite")
    return result


def _timestamp(value: Any, *, name: str) -> datetime:
    try:
        return paired._timestamp(value, name=name)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def _content_sha(value: Mapping[str, Any], *, name: str) -> str:
    expected = _sha(value.get("content_sha256"), name=f"{name} content SHA-256")
    actual = canonical_sha256(
        {field: item for field, item in value.items() if field != "content_sha256"}
    )
    if actual != expected:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} content SHA-256 differs")
    return expected


def _git_identity(value: Any, *, name: str) -> dict[str, str]:
    git = _exact(value, _GIT_FIELDS, name=name)
    for field in ("repo", "repo_url", "branch"):
        if not isinstance(git[field], str) or not git[field]:
            raise SSMaxPerceptionDirectEvidenceError(f"{name} {field} must be non-empty")
    if not isinstance(git["ref"], str) or _GIT_REF_RE.fullmatch(git["ref"]) is None:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} ref must be a commit SHA")
    return {field: str(git[field]) for field in _GIT_FIELDS}


def _source_reference(value: Any, *, name: str, expected_git_ref: str) -> dict[str, str]:
    reference = _exact(value, _SOURCE_REF_FIELDS, name=name)
    path = reference["repo_relative_path"]
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} repository path is invalid")
    _sha(reference["sha256"], name=f"{name} SHA-256")
    if reference["git_ref"] != expected_git_ref:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} Git ref differs")
    return {field: str(reference[field]) for field in _SOURCE_REF_FIELDS}


def _git_blob_reference(
    *,
    git: Mapping[str, str],
    repository_root: Path,
    repo_relative_path: str,
    require_live_equal: bool,
) -> dict[str, str]:
    try:
        blob = bridge._git_blob_bytes(
            git,
            repository_root=repository_root,
            repo_relative_path=repo_relative_path,
            name=repo_relative_path,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    digest = hashlib.sha256(blob).hexdigest()
    live = (repository_root / repo_relative_path).resolve()
    if require_live_equal and (not live.is_file() or sha256_file(live) != digest):
        raise SSMaxPerceptionDirectEvidenceError(
            f"Live {repo_relative_path} differs from evidence Git blob"
        )
    return {
        "repo_relative_path": repo_relative_path,
        "sha256": digest,
        "git_ref": str(git["ref"]),
    }


def validate_evidence_git_compatibility(
    *,
    training_git: Mapping[str, str],
    evidence_git: Mapping[str, str],
    repository_root: Path,
) -> tuple[tuple[str, str], ...]:
    """Reject evidence revisions that touch anything outside the explicit additive allowlist.

    The returned tuples are ``(status, repository-relative-path)`` rows from Git.  Deletions,
    copies, renames, type changes, merge-conflict records, and multi-path status rows are always
    rejected even when one of their paths happens to be allowlisted.
    """

    training = _git_identity(training_git, name="compatibility training Git")
    evidence = _git_identity(evidence_git, name="compatibility evidence Git")
    if training["repo"] != evidence["repo"] or training["repo_url"] != evidence["repo_url"]:
        raise SSMaxPerceptionDirectEvidenceError(
            "Training and evidence Git identities name different repositories"
        )
    root = repository_root.expanduser().resolve()
    if evidence["ref"] in LEGACY_EVIDENCE_GIT_REFS:
        _require_exact_git_parent(
            repository_root=root,
            child_ref=evidence["ref"],
            parent_ref=training["ref"],
            name="legacy evidence revision",
        )
    else:
        # The exploratory continuation is additive to the already published direct-v3 evidence
        # revision. Preserve that immutable revision, require one new non-merge evidence commit,
        # and still prove the original evidence commit is a direct child of the training ref.
        _require_exact_git_parent(
            repository_root=root,
            child_ref=evidence["ref"],
            parent_ref=BASE_EVIDENCE_GIT_REF,
            name="exploratory evidence revision",
        )
        _require_exact_git_parent(
            repository_root=root,
            child_ref=BASE_EVIDENCE_GIT_REF,
            parent_ref=training["ref"],
            name="base evidence revision",
        )
    try:
        output = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-status",
                "--no-renames",
                f"{training['ref']}..{evidence['ref']}",
                "--",
            ],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxPerceptionDirectEvidenceError(
            "Could not compare the training and evidence Git revisions"
        ) from error
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            raise SSMaxPerceptionDirectEvidenceError(
                "Evidence Git diff contains a non-canonical multi-path change"
            )
        status, path = parts
        if status not in {"A", "M"} or path not in EVIDENCE_GIT_DIFF_ALLOWLIST:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Evidence Git diff contains unauthorized change {status} {path}"
            )
        rows.append((status, path))
    protocol_path = "src/olmo_core/eval/vision_alignment_ssmax_perception_direct.py"
    if not any(path == protocol_path for _, path in rows):
        raise SSMaxPerceptionDirectEvidenceError(
            "Evidence Git revision does not add the direct perception protocol"
        )
    return tuple(rows)


def _require_exact_git_parent(
    *, repository_root: Path, child_ref: str, parent_ref: str, name: str
) -> None:
    """Require ``child_ref`` to be a non-merge commit whose sole parent is ``parent_ref``."""

    try:
        output = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-list", "--parents", "-n", "1", child_ref],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxPerceptionDirectEvidenceError(
            f"Could not verify the exact parent of the {name}"
        ) from error
    if output.split() != [child_ref, parent_ref]:
        raise SSMaxPerceptionDirectEvidenceError(
            f"The {name} must be one non-merge commit directly after its approved parent"
        )


def _validate_joint_consumer_profiles(repository_root: Path) -> None:
    """Bind the exact legacy allowlist plus the two content-addressed direct profiles."""

    allowlist_path = repository_root / JOINT_PROFILE_ALLOWLIST_RELATIVE_PATH
    try:
        raw = allowlist_path.read_bytes()
        value = load_json(allowlist_path)
    except OSError as error:
        raise SSMaxPerceptionDirectEvidenceError(
            "Could not read the direct joint profile allowlist"
        ) from error
    mapping = _exact(
        value,
        frozenset({"format", "version", "profiles"}),
        name="direct joint profile allowlist",
    )
    if (
        mapping["format"] != "vision_alignment_joint_profile_allowlist"
        or type(mapping["version"]) is not int
        or mapping["version"] != 1
        or not isinstance(mapping["profiles"], Mapping)
        or raw != bridge.canonical_json_bytes(mapping, trailing_newline=True)
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct joint profile allowlist identity or canonical bytes differ"
        )
    profiles = dict(mapping["profiles"])
    expected_paths = set(JOINT_PROFILE_BASELINE) | set(JOINT_DIRECT_PROFILE_PATHS)
    if set(profiles) != expected_paths or any(
        profiles.get(path) != digest for path, digest in JOINT_PROFILE_BASELINE.items()
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct joint profile allowlist does not preserve the exact legacy entries"
        )
    for path in sorted(JOINT_DIRECT_PROFILE_PATHS):
        digest = profiles[path]
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct joint profile allowlist digest is invalid for {path}"
            )
        profile_path = repository_root / path
        if not profile_path.is_file() or sha256_file(profile_path) != digest:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct joint profile bytes differ from their allowlist pin for {path}"
            )


def _validate_evidence_or_joint_consumer_checkout(
    *, evidence_git: Mapping[str, str], repository_root: Path
) -> tuple[tuple[str, str], ...]:
    """Require either the evidence ref or its one exact joint-profile descendant.

    The v7 gate cannot know the two final profile hashes until its reports have been produced and
    human-approved. Joint training therefore runs from a later clean commit. That descendant may
    add only the two predeclared lineage profiles and update their dedicated allowlist; every
    evidence producer remains byte-bound to the earlier evidence ref.
    """

    evidence = _git_identity(evidence_git, name="consumer evidence Git")
    if evidence["repo"] != "allenai/OLMo-core" or evidence["repo_url"] != (
        "https://github.com/allenai/OLMo-core"
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct evidence consumer names a different repository"
        )
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
        raise SSMaxPerceptionDirectEvidenceError(
            "Could not verify the direct evidence consumer checkout"
        ) from error
    if dirty:
        raise SSMaxPerceptionDirectEvidenceError("Direct evidence consumer checkout is not clean")
    if head == evidence["ref"]:
        return ()
    _require_exact_git_parent(
        repository_root=root,
        child_ref=head,
        parent_ref=evidence["ref"],
        name="direct joint-consumer revision",
    )
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--name-status",
                "--no-renames",
                f"{evidence['ref']}..{head}",
                "--",
            ],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct joint consumer is not a descendant of its evidence ref"
        ) from error
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            raise SSMaxPerceptionDirectEvidenceError(
                "Direct joint-consumer diff contains a non-canonical multi-path change"
            )
        rows.append((parts[0], parts[1]))
    if frozenset(rows) != JOINT_CONSUMER_GIT_DIFF or len(rows) != len(JOINT_CONSUMER_GIT_DIFF):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct joint-consumer diff differs from the two profiles and their allowlist"
        )
    _validate_joint_consumer_profiles(root)
    return tuple(rows)


def _validate_amendment_payload(value: Any) -> Mapping[str, Any]:
    fields = frozenset(
        {
            "format",
            "version",
            "recorded_at",
            "approved_by",
            "scope",
            "control_runs_required",
            "excluded_claims",
            "excluded_control_runs",
            "cross_model_comparison",
            "training_git",
            "model_runs",
            "evidence_policy",
        }
    )
    amendment = _exact(value, fields, name="single-lineage protocol amendment")
    if (
        amendment["format"] != AMENDMENT_FORMAT
        or type(amendment["version"]) is not int
        or amendment["version"] != AMENDMENT_VERSION
        or amendment["scope"] != "two_direct_bridge_perception_joint_lineages"
        or amendment["control_runs_required"] is not False
        or amendment["excluded_claims"] != ["causal_effect_of_unfreezing_the_vision_encoder"]
        or amendment["cross_model_comparison"]
        != {"mode": "descriptive_non_promotion", "winner": None}
    ):
        raise SSMaxPerceptionDirectEvidenceError("Protocol amendment identity or scope differs")
    excluded_controls = _exact(
        amendment["excluded_control_runs"],
        frozenset(MODEL_VARIANTS),
        name="amendment excluded control runs",
    )
    expected_controls = {
        "ssmax_head_qknorm": {
            "experiment_id": "01M0KBV398GTYAK7VZ9FMMHS8X",
            "status": "canceled_while_starting",
            "checkpoint_root_created": False,
        },
        "ssmax_no_qknorm": {
            "experiment_id": None,
            "status": "not_launched",
            "checkpoint_root_created": False,
        },
    }
    if dict(excluded_controls) != expected_controls:
        raise SSMaxPerceptionDirectEvidenceError(
            "Protocol amendment excluded-control disposition differs"
        )
    for model_variant in MODEL_VARIANTS:
        disposition = _exact(
            excluded_controls[model_variant],
            frozenset({"experiment_id", "status", "checkpoint_root_created"}),
            name=f"amendment {model_variant} excluded control",
        )
        if type(disposition["checkpoint_root_created"]) is not bool:
            raise SSMaxPerceptionDirectEvidenceError(
                "Protocol amendment checkpoint-root disposition must be boolean"
            )
    if amendment["recorded_at"] != AMENDMENT_RECORDED_AT:
        raise SSMaxPerceptionDirectEvidenceError("Protocol amendment recorded_at differs")
    _timestamp(amendment["recorded_at"], name="protocol amendment recorded_at")
    approved_by = amendment["approved_by"]
    if (
        approved_by != AMENDMENT_APPROVED_BY
        or not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Protocol amendment approved_by is not a durable identity"
        )
    training_git = _exact(
        amendment["training_git"], frozenset({"branch", "ref"}), name="amendment training Git"
    )
    if training_git != {"branch": TRAINING_GIT_BRANCH, "ref": TRAINING_GIT_REF}:
        raise SSMaxPerceptionDirectEvidenceError("Protocol amendment training Git differs")
    runs = _exact(amendment["model_runs"], frozenset(MODEL_VARIANTS), name="amendment model runs")
    for model_variant, expected in DIRECT_RUN_IDENTITIES.items():
        run = _exact(
            runs[model_variant],
            frozenset({"run_name", "checkpoint_root", "profile", "profile_sha256"}),
            name=f"amendment {model_variant} run",
        )
        if dict(run) != dict(expected):
            raise SSMaxPerceptionDirectEvidenceError(
                f"Protocol amendment {model_variant} run identity differs"
            )
    policy = _mapping(amendment["evidence_policy"], name="amendment evidence policy")
    if dict(policy) != dict(AMENDMENT_EVIDENCE_POLICY):
        raise SSMaxPerceptionDirectEvidenceError("Protocol amendment evidence policy differs")
    for field in (
        "candidate_absolute_visual_gap_ci_low_must_be_positive",
        "candidate_visual_gap_improvement_vs_step0_ci_low_must_be_positive",
        "require_finite_gradient_only_skips",
        "require_uninterrupted_optimizer_guard_history",
    ):
        if type(policy[field]) is not bool:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Protocol amendment policy {field} must be boolean"
            )
    for field in (
        "maximum_data_errors",
        "maximum_nonfinite_gradients",
        "maximum_nonfinite_losses",
        "maximum_optimizer_guard_skips",
        "minimum_clean_final_steps",
        "minimum_clean_steps_between_skips",
    ):
        _integer(policy[field], name=f"amendment policy {field}")
    for field in (
        "maximum_correct_ce_fraction_of_step0",
        "minimum_step4000_gap_retention_fraction_of_step3000",
    ):
        if type(policy[field]) is not float:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Protocol amendment policy {field} must be a JSON float"
            )
        _finite(policy[field], name=f"amendment policy {field}")
    if not isinstance(policy["required_steps"], list) or any(
        type(step) is not int for step in policy["required_steps"]
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Protocol amendment required steps must be JSON integers"
        )
    return amendment


def _authorized_amendment_reference(
    value: Any, *, repository_root: Path, require_content_sha: bool
) -> dict[str, str]:
    fields = _AMENDMENT_REF_FIELDS if require_content_sha else _ARTIFACT_REF_FIELDS
    reference = _exact(value, fields, name="protocol amendment reference")
    if reference["path"] != AMENDMENT_RELATIVE_PATH or reference["sha256"] != AMENDMENT_SHA256:
        raise SSMaxPerceptionDirectEvidenceError("Protocol amendment raw identity differs")
    path = (repository_root / AMENDMENT_RELATIVE_PATH).resolve()
    if path != repository_root.resolve() / AMENDMENT_RELATIVE_PATH or not path.is_file():
        raise SSMaxPerceptionDirectEvidenceError("Authorized protocol amendment is absent")
    if sha256_file(path) != AMENDMENT_SHA256:
        raise SSMaxPerceptionDirectEvidenceError("Authorized protocol amendment bytes differ")
    amendment = _validate_amendment_payload(load_json(path))
    semantic = canonical_sha256(amendment)
    if require_content_sha and reference["content_sha256"] != semantic:
        raise SSMaxPerceptionDirectEvidenceError("Protocol amendment semantic identity differs")
    result = {"path": AMENDMENT_RELATIVE_PATH, "sha256": AMENDMENT_SHA256}
    if require_content_sha:
        result["content_sha256"] = semantic
    return result


def _validate_policy(value: Any) -> Mapping[str, Any]:
    policy = _mapping(value, name="direct promotion policy")
    if dict(policy) != dict(DIRECT_POLICY):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion policy differs from locked policy"
        )
    for field in (
        "baseline_step",
        "durability_step",
        "candidate_step",
        "maximum_data_errors",
        "maximum_optimizer_guard_skips",
        "minimum_clean_steps_between_skips",
        "minimum_clean_final_steps",
        "maximum_nonfinite_losses",
        "maximum_nonfinite_gradients",
    ):
        _integer(policy[field], name=f"policy {field}")
    for field in (
        "candidate_gap_lower_ci_minimum",
        "candidate_gap_improvement_lower_ci_minimum",
        "maximum_correct_ce_fraction_of_step0",
        "minimum_gap_retention",
        "loss_mass_share_tolerance",
    ):
        if type(policy[field]) is not float:
            raise SSMaxPerceptionDirectEvidenceError(f"policy {field} must be a JSON float")
        _finite(policy[field], name=f"policy {field}")
    guard = _exact(
        policy["optimizer_guard"],
        frozenset(V2_OPTIMIZER_GUARD_CONTRACT),
        name="optimizer guard",
    )
    if dict(guard) != V2_OPTIMIZER_GUARD_CONTRACT:
        raise SSMaxPerceptionDirectEvidenceError("Optimizer guard differs from v2 contract")
    if (
        type(guard["type"]) is not str
        or type(guard["rolling_interval_length"]) is not int
        or type(guard["sigma_factor"]) is not int
        or type(guard["max_grad_norm"]) is not float
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Optimizer guard fields have non-canonical JSON types"
        )
    for field in (
        "require_finite_gradient_only_skips",
        "require_uninterrupted_optimizer_guard_history",
    ):
        if type(policy[field]) is not bool:
            raise SSMaxPerceptionDirectEvidenceError(f"policy {field} must be boolean")
    return policy


def _validate_evaluation(value: Any) -> Mapping[str, Any]:
    evaluation = _mapping(value, name="direct evaluation contract")
    if dict(evaluation) != dict(EVALUATION_CONTRACT):
        raise SSMaxPerceptionDirectEvidenceError("Direct evaluation contract differs")
    for field in (
        "examples_per_source",
        "pairing_seed",
        "bootstrap_seed",
        "bootstrap_samples",
        "rank_batch_instances",
    ):
        _integer(evaluation[field], name=f"evaluation {field}")
    if (
        not isinstance(evaluation["steps"], list)
        or any(type(step) is not int for step in evaluation["steps"])
        or not isinstance(evaluation["sources"], list)
        or any(type(source) is not str for source in evaluation["sources"])
        or not isinstance(evaluation["windows"], list)
        or any(type(window) is not str for window in evaluation["windows"])
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct evaluation lists have non-canonical JSON types"
        )
    return evaluation


def _validate_topology(value: Any) -> Mapping[str, Any]:
    topology = _mapping(value, name="direct topology contract")
    if dict(topology) != dict(TOPOLOGY_CONTRACT):
        raise SSMaxPerceptionDirectEvidenceError("Direct topology contract differs")
    for field in ("world_size", "num_nodes", "gpus_per_node"):
        _integer(topology[field], name=f"topology {field}", minimum=1)
    if type(topology["data_parallel"]) is not str:
        raise SSMaxPerceptionDirectEvidenceError("Direct topology data_parallel must be a string")
    return topology


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_manifest_spec(value: Any, *, repository_root: Path | None = None) -> Mapping[str, Any]:
    """Validate one concrete, non-runnable direct-lineage manifest specification."""

    spec = _exact(value, _SPEC_FIELDS, name="direct manifest spec")
    if (
        spec["format"] != MANIFEST_SPEC_FORMAT
        or type(spec["version"]) is not int
        or spec["version"] != SCHEMA_VERSION
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct manifest spec is incompatible")
    for field in ("run_id", "run_name", "checkpoint_root", "training_profile", "recipe"):
        if not isinstance(spec[field], str) or not spec[field]:
            raise SSMaxPerceptionDirectEvidenceError(f"Direct manifest spec {field} is invalid")
    model_variant = spec["model_variant"]
    if model_variant not in MODEL_VARIANTS:
        raise SSMaxPerceptionDirectEvidenceError("Direct manifest model variant is unsupported")
    identity = DIRECT_RUN_IDENTITIES[str(model_variant)]
    for field in ("run_name", "checkpoint_root"):
        if spec[field] != identity[field]:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct manifest spec {field} is unauthorized"
            )
    if spec["training_profile"] != identity["profile"]:
        raise SSMaxPerceptionDirectEvidenceError("Direct manifest spec profile is unauthorized")
    training_git = _git_identity(spec["training_git"], name="training Git")
    if training_git["branch"] != TRAINING_GIT_BRANCH or training_git["ref"] != TRAINING_GIT_REF:
        raise SSMaxPerceptionDirectEvidenceError("Direct training Git is unauthorized")
    evidence_git = _git_identity(spec["evidence_git"], name="evidence Git")
    if (
        training_git["repo"] != evidence_git["repo"]
        or training_git["repo_url"] != evidence_git["repo_url"]
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct training and evidence Git identities name different repositories"
        )
    repo = (repository_root or _repository_root()).resolve()
    _authorized_amendment_reference(
        spec["protocol_amendment"], repository_root=repo, require_content_sha=False
    )
    if Path(spec["recipe"]).as_posix() != "src/scripts/train/Vision-Alignment.py":
        raise SSMaxPerceptionDirectEvidenceError("Direct training recipe path is non-canonical")
    for field in (
        "bridge_parent_gate",
        "perception_provenance",
        "source_audit",
        "attention_probe",
        "text_sentinel",
    ):
        if not isinstance(spec[field], str) or not spec[field]:
            raise SSMaxPerceptionDirectEvidenceError(f"Direct manifest spec {field} is invalid")
    pairings = _exact(spec["pairing_paths"], frozenset(SOURCES), name="direct pairing paths")
    if any(not isinstance(pairings[source], str) or not pairings[source] for source in SOURCES):
        raise SSMaxPerceptionDirectEvidenceError("Every direct pairing path must be non-empty")
    _validate_evaluation(spec["evaluation"])
    _validate_topology(spec["topology"])
    _validate_policy(spec["policy"])
    return spec


def load_manifest_spec(path: Path) -> Mapping[str, Any]:
    """Load and validate a concrete direct-lineage manifest specification."""

    return validate_manifest_spec(load_json(path))


def validate_saved_config(
    config: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    training_profile: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the one authorized vision-unfrozen training configuration."""

    model_variant = str(spec["model_variant"])
    run_name = str(spec["run_name"])
    for field, expected in (
        ("model_variant", model_variant),
        ("phase", "perception"),
        ("perception_trainability_arm", TRAINING_ROLE),
        ("required_run_name", run_name),
    ):
        if config.get(field) != expected:
            raise SSMaxPerceptionDirectEvidenceError(f"Saved direct config {field} differs")
    identity = DIRECT_RUN_IDENTITIES[model_variant]
    if (
        config.get("reviewed_profile_path") != identity["profile"]
        or config.get("reviewed_profile_sha256") != identity["profile_sha256"]
        or training_profile["repo_relative_path"] != identity["profile"]
        or training_profile["sha256"] != identity["profile_sha256"]
    ):
        raise SSMaxPerceptionDirectEvidenceError("Saved direct profile identity differs")
    expected_command = [
        "src/scripts/train/Vision-Alignment.py",
        "train",
        run_name,
        f"--profile={identity['profile']}",
    ]
    if config.get("expected_launch_command") != expected_command:
        raise SSMaxPerceptionDirectEvidenceError("Saved direct launch command differs")
    data = _mapping(config.get("data"), name="saved direct data")
    if (
        data.get("pack_sequences") is not False
        or data.get("allow_unpinned_synthetic_smoke") is not False
    ):
        raise SSMaxPerceptionDirectEvidenceError("Saved direct data enables an ineligible mode")
    metadata = _mapping(config.get("vision_alignment"), name="saved direct metadata")
    if (
        metadata.get("model_variant") != model_variant
        or metadata.get("phase") != "perception"
        or metadata.get("lineage_id") != run_name
    ):
        raise SSMaxPerceptionDirectEvidenceError("Saved direct lineage metadata differs")
    data_contract = _sha(metadata.get("data_contract_sha256"), name="direct data contract")
    trainable_contract = _sha(
        metadata.get("trainable_contract_sha256"), name="direct trainable contract"
    )
    initialization = _mapping(config.get("initialization"), name="direct initialization")
    if (
        initialization.get("expected_parent_phase") != "bridge"
        or initialization.get("mode") != "checkpoint"
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct run does not require a bridge parent")
    trainer = _mapping(config.get("trainer"), name="direct trainer")
    if (
        trainer.get("save_folder") != spec["checkpoint_root"]
        or trainer.get("load_path") != initialization.get("checkpoint")
        or trainer.get("load_strategy") != "always"
        or trainer.get("load_optim_state") is not False
        or trainer.get("load_trainer_state") is not False
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct trainer parent/output contract differs")
    duration = _mapping(trainer.get("max_duration"), name="direct duration")
    if duration.get("unit") != "steps" or duration.get("value") != 4000:
        raise SSMaxPerceptionDirectEvidenceError("Direct duration must be exactly 4000 steps")
    callbacks = _mapping(trainer.get("callbacks"), name="direct callbacks")
    checkpointer = _mapping(callbacks.get("checkpointer"), name="direct checkpointer")
    if (
        checkpointer.get("pre_train_checkpoint") is not True
        or checkpointer.get("save_async") is not False
        or checkpointer.get("fixed_steps") != [500, 1000, 2000, 3000, 4000]
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct checkpoint cadence differs")
    health = _mapping(callbacks.get("ssmax_health_ledger"), name="direct health ledger")
    if any(
        health.get(field) != expected
        for field, expected in (
            ("enabled", True),
            ("model_variant", model_variant),
            ("phase", "perception"),
            ("run_name", run_name),
        )
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct health-ledger identity differs")
    launch = _mapping(config.get("launch"), name="direct launch")
    topology = _mapping(spec["topology"], name="direct topology")
    if (
        launch.get("num_nodes") != topology["num_nodes"]
        or launch.get("num_gpus") != topology["gpus_per_node"]
        or launch.get("workspace") != "ai2/scaling-ladders"
        or launch.get("clusters") != ["ai2/holmes"]
        or launch.get("budget") != "ai2/oe-other"
        or launch.get("priority") != "urgent"
        or launch.get("min_runtime") not in ("8h", 28_800, 28_800.0)
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct launch contract differs")
    raw_git = _mapping(launch.get("git"), name="saved direct Git")
    allowed_git_fields = set(_GIT_FIELDS) | {"_CLASS_"}
    if not set(raw_git) <= allowed_git_fields or not set(_GIT_FIELDS) <= set(raw_git):
        raise SSMaxPerceptionDirectEvidenceError("Saved direct Git fields differ")
    saved_git = {field: raw_git[field] for field in _GIT_FIELDS}
    if saved_git != dict(spec["training_git"]):
        raise SSMaxPerceptionDirectEvidenceError("Saved direct training Git differs")
    train_module = _mapping(config.get("train_module"), name="direct train module")
    freeze = train_module.get("freeze_params")
    if not isinstance(freeze, list) or "vision.*" in freeze:
        raise SSMaxPerceptionDirectEvidenceError("Direct run does not leave vision trainable")
    guard = _mapping(spec["policy"], name="direct policy")["optimizer_guard"]
    optim = _mapping(train_module.get("optim"), name="direct optimizer")
    if (
        optim.get("type") != guard["type"]
        or optim.get("rolling_interval_length") != guard["rolling_interval_length"]
        or optim.get("sigma_factor") != guard["sigma_factor"]
        or not math.isclose(
            _finite(train_module.get("max_grad_norm"), name="direct max grad norm"),
            float(guard["max_grad_norm"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct optimizer guard differs")
    try:
        _, vision_group = paired._vision_group(config, arm=TRAINING_ROLE)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    vision_lr = _finite(
        _mapping(vision_group.get("opts"), name="direct vision optimizer options").get("lr"),
        name="direct vision LR",
    )
    if vision_lr <= 0:
        raise SSMaxPerceptionDirectEvidenceError("Direct vision LR must be positive")
    targets = _mapping(
        train_module.get("source_loss_mass_targets"), name="direct source loss-mass targets"
    )
    if set(targets) != set(SOURCES):
        raise SSMaxPerceptionDirectEvidenceError("Direct loss-mass source set differs")
    loss_mass_targets = {
        source: _finite(targets[source], name=f"{source} loss-mass target") for source in SOURCES
    }
    if any(value <= 0 for value in loss_mass_targets.values()) or not math.isclose(
        sum(loss_mass_targets.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct loss-mass targets are invalid")
    return {
        "training_git": dict(saved_git),
        "initialization": dict(initialization),
        "data_contract_sha256": data_contract,
        "trainable_contract_sha256": trainable_contract,
        "loss_mass_targets": loss_mass_targets,
        "vision_lr": vision_lr,
    }


def _lineage_contract_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    run = _mapping(manifest["run"], name="direct run")
    return {
        "run_id": manifest["run_id"],
        "lineage_kind": manifest["lineage_kind"],
        "model_variant": manifest["model_variant"],
        "training_git": manifest["training_git"],
        "evidence_git": manifest["evidence_git"],
        "producers": manifest["producers"],
        "training_recipe": manifest["training_recipe"],
        "protocol_amendment": manifest["protocol_amendment"],
        "bridge_parent": manifest["bridge_parent"],
        "perception_provenance": manifest["perception_provenance"],
        "source_audit": manifest["source_audit"],
        "source_audit_fingerprint": manifest["source_audit_fingerprint"],
        "single_response_projection": manifest["single_response_projection"],
        "attention_probe": manifest["attention_probe"],
        "text_sentinel": manifest["text_sentinel"],
        "pairings": manifest["pairings"],
        "evaluation": manifest["evaluation"],
        "topology": manifest["topology"],
        "policy": manifest["policy"],
        "loss_mass_targets": manifest["loss_mass_targets"],
        "run": {
            "run_name": run["run_name"],
            "checkpoint_root": run["checkpoint_root"],
            "training_profile": run["training_profile"],
            "data_contract_sha256": run["data_contract_sha256"],
            "trainable_contract_sha256": run["trainable_contract_sha256"],
        },
    }


def _checkpoint_reference(
    value: Any, *, step: int, verify_live: bool, workers: int
) -> Mapping[str, Any]:
    reference = _exact(value, _CHECKPOINT_FIELDS, name=f"direct step{step} checkpoint")
    _integer(reference["global_step"], name=f"direct step{step} checkpoint global_step")
    try:
        return bridge.validate_checkpoint_reference(
            reference,
            expected_step=step,
            verify_live=verify_live,
            workers=workers,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def build_manifest(
    spec: Mapping[str, Any], *, created_at: str, hash_workers: int = 8
) -> dict[str, Any]:
    """Finalize one direct-lineage manifest after all required checkpoints exist."""

    repository_root = _repository_root()
    spec = validate_manifest_spec(spec, repository_root=repository_root)
    manifest_time = _timestamp(created_at, name="direct manifest created_at")
    if manifest_time < _timestamp(AMENDMENT_RECORDED_AT, name="protocol amendment recorded_at"):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct manifest predates its authorizing protocol amendment"
        )
    if hash_workers <= 0:
        raise ValueError("hash_workers must be positive")
    training_git = _git_identity(spec["training_git"], name="training Git")
    evidence_git = _git_identity(spec["evidence_git"], name="evidence Git")
    try:
        bridge._validate_repository_checkout(evidence_git, repository_root=repository_root)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    validate_evidence_git_compatibility(
        training_git=training_git,
        evidence_git=evidence_git,
        repository_root=repository_root,
    )
    root = Path(str(spec["checkpoint_root"])).expanduser().resolve()
    missing = [
        str(root / f"step{step}") for step in REQUIRED_STEPS if not (root / f"step{step}").is_dir()
    ]
    if missing:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct manifest requires every fixed checkpoint; missing " + ", ".join(missing)
        )
    checkpoints = {
        str(step): bridge.checkpoint_identity(root / f"step{step}", workers=hash_workers)
        for step in REQUIRED_STEPS
    }
    if len({item["config_sha256"] for item in checkpoints.values()}) != 1:
        raise SSMaxPerceptionDirectEvidenceError("Direct checkpoints do not share one config")
    if {item["trainer_state_count"] for item in checkpoints.values()} != {
        spec["topology"]["world_size"]
    }:
        raise SSMaxPerceptionDirectEvidenceError("Direct checkpoint world sizes differ")
    training_recipe = _git_blob_reference(
        git=training_git,
        repository_root=repository_root,
        repo_relative_path="src/scripts/train/Vision-Alignment.py",
        require_live_equal=False,
    )
    training_profile = _git_blob_reference(
        git=training_git,
        repository_root=repository_root,
        repo_relative_path=str(spec["training_profile"]),
        require_live_equal=False,
    )
    config = _mapping(load_json(root / "step0" / "config.json"), name="direct step0 config")
    summary = validate_saved_config(
        config,
        spec=spec,
        training_profile=training_profile,
    )
    producers = {
        name: _git_blob_reference(
            git=evidence_git,
            repository_root=repository_root,
            repo_relative_path=relative,
            require_live_equal=True,
        )
        for name, relative in PRODUCER_RELATIVE_PATHS.items()
    }
    amendment_base = _authorized_amendment_reference(
        spec["protocol_amendment"],
        repository_root=repository_root,
        require_content_sha=False,
    )
    amendment = _validate_amendment_payload(load_json(repository_root / AMENDMENT_RELATIVE_PATH))
    amendment_reference = {
        **amendment_base,
        "content_sha256": canonical_sha256(amendment),
    }
    gate_reference = artifact_reference(Path(str(spec["bridge_parent_gate"])))
    try:
        bridge_parent = paired._validate_bridge_parent(
            {paired.CONTROL_ARM: config, paired.TREATMENT_ARM: config},
            model_variant=str(spec["model_variant"]),
            gate_reference=gate_reference,
            verify_live_checkpoint=True,
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    data = _mapping(config.get("data"), name="direct perception data")
    provenance_reference = artifact_reference(Path(str(spec["perception_provenance"])))
    source_audit_reference = artifact_reference(Path(str(spec["source_audit"])))
    if (
        Path(str(data.get("perception_provenance_path"))).expanduser().resolve()
        != Path(provenance_reference["path"])
        or data.get("perception_provenance_sha256") != provenance_reference["sha256"]
        or Path(str(data.get("source_audit_path"))).expanduser().resolve()
        != Path(source_audit_reference["path"])
    ):
        raise SSMaxPerceptionDirectEvidenceError("Saved direct data artifacts differ from spec")
    provenance = _mapping(
        load_json(Path(provenance_reference["path"])), name="direct perception provenance"
    )
    provenance_content_sha = _sha(
        provenance.get("content_sha256"), name="direct provenance content SHA"
    )
    if (
        canonical_sha256(
            {field: item for field, item in provenance.items() if field != "content_sha256"}
        )
        != provenance_content_sha
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct provenance semantic SHA differs")
    typed_provenance = paired.load_perception_provenance_manifest(
        provenance_reference["path"],
        expected_sha256=provenance_reference["sha256"],
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    source_audit = _mapping(load_json(Path(source_audit_reference["path"])), name="source audit")
    unsigned_audit = dict(source_audit)
    fingerprint = _sha(unsigned_audit.pop("fingerprint", None), name="source audit fingerprint")
    if (
        canonical_sha256(unsigned_audit) != fingerprint
        or data.get("source_audit_fingerprint") != fingerprint
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct source-audit fingerprint differs")
    text_sentinel_reference = artifact_reference(Path(str(spec["text_sentinel"])))
    try:
        sentinel = paired._validate_text_sentinel(Path(text_sentinel_reference["path"]))
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    artifacts = _mapping(config.get("artifacts"), name="direct artifacts")
    if sentinel["tokenizer"] != {
        "identifier": artifacts.get("tokenizer_id"),
        "revision": artifacts.get("tokenizer_revision"),
    }:
        raise SSMaxPerceptionDirectEvidenceError("Direct text sentinel tokenizer differs")
    attention_probe_reference = artifact_reference(Path(str(spec["attention_probe"])))
    try:
        single_response_projection = paired._single_response_binding_from_config(config)
        paired._validate_attention_probe_reference(
            attention_probe_reference,
            provenance=typed_provenance,
            projection_contract=single_response_projection["contract"],
            verify_live=True,
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    pairings: dict[str, dict[str, str]] = {}
    for source in SOURCES:
        reference = artifact_reference(Path(str(spec["pairing_paths"][source])))
        try:
            paired._validate_pairing_reference(
                reference,
                source=source,
                evaluation=spec["evaluation"],
                verify_live=True,
                dataset_size=len(typed_provenance.selection(source, "validation").indices),
                expected_content_ids_sha256=paired.content_ids_sha256(
                    typed_provenance.selection(source, "validation").row_image_content_sha256
                ),
            )
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
        pairings[source] = reference
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": SCHEMA_VERSION,
        "created_at": created_at,
        "run_id": spec["run_id"],
        "lineage_kind": LINEAGE_KIND,
        "model_variant": spec["model_variant"],
        "training_git": dict(training_git),
        "evidence_git": dict(evidence_git),
        "producers": producers,
        "training_recipe": training_recipe,
        "protocol_amendment": amendment_reference,
        "bridge_parent": bridge_parent,
        "perception_provenance": {
            **provenance_reference,
            "content_sha256": provenance_content_sha,
        },
        "source_audit": source_audit_reference,
        "source_audit_fingerprint": fingerprint,
        "single_response_projection": single_response_projection,
        "attention_probe": attention_probe_reference,
        "text_sentinel": text_sentinel_reference,
        "pairings": pairings,
        "evaluation": dict(spec["evaluation"]),
        "topology": dict(spec["topology"]),
        "policy": dict(spec["policy"]),
        "loss_mass_targets": summary["loss_mass_targets"],
        "run": {
            "run_name": spec["run_name"],
            "checkpoint_root": str(root),
            "training_profile": training_profile,
            "data_contract_sha256": summary["data_contract_sha256"],
            "trainable_contract_sha256": summary["trainable_contract_sha256"],
            "checkpoints": checkpoints,
        },
    }
    manifest["lineage_contract_sha256"] = canonical_sha256(_lineage_contract_payload(manifest))
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest, verify_live=True, hash_workers=hash_workers)
    return manifest


def _artifact_shape(value: Any, *, name: str, semantic: bool = False) -> Mapping[str, Any]:
    fields = _MANIFEST_REF_FIELDS if semantic else _ARTIFACT_REF_FIELDS
    reference = _exact(value, fields, name=name)
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} path is invalid")
    _sha(reference["sha256"], name=f"{name} raw SHA-256")
    if semantic:
        _sha(reference["content_sha256"], name=f"{name} semantic SHA-256")
    return reference


def _producer_references(value: Any, *, evidence_git_ref: str) -> dict[str, dict[str, str]]:
    producers = _exact(value, frozenset(PRODUCER_RELATIVE_PATHS), name="direct producers")
    output: dict[str, dict[str, str]] = {}
    for name, expected_path in PRODUCER_RELATIVE_PATHS.items():
        reference = _source_reference(
            producers[name], name=f"direct {name} producer", expected_git_ref=evidence_git_ref
        )
        if reference["repo_relative_path"] != expected_path:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct {name} producer path is non-canonical"
            )
        output[name] = reference
    return output


def validate_manifest(
    value: Any, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Validate a finalized direct-lineage manifest and optionally every live byte pin."""

    manifest = _exact(value, _MANIFEST_FIELDS, name="direct perception manifest")
    if (
        manifest["format"] != MANIFEST_FORMAT
        or type(manifest["version"]) is not int
        or manifest["version"] != SCHEMA_VERSION
        or manifest["lineage_kind"] != LINEAGE_KIND
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct perception manifest is incompatible")
    if _timestamp(manifest["created_at"], name="direct manifest created_at") < _timestamp(
        AMENDMENT_RECORDED_AT, name="protocol amendment recorded_at"
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct manifest predates its authorizing protocol amendment"
        )
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise SSMaxPerceptionDirectEvidenceError("Direct manifest run_id is invalid")
    model_variant = manifest["model_variant"]
    if model_variant not in MODEL_VARIANTS:
        raise SSMaxPerceptionDirectEvidenceError("Direct manifest model variant is unsupported")
    training_git = _git_identity(manifest["training_git"], name="manifest training Git")
    evidence_git = _git_identity(manifest["evidence_git"], name="manifest evidence Git")
    if training_git["branch"] != TRAINING_GIT_BRANCH or training_git["ref"] != TRAINING_GIT_REF:
        raise SSMaxPerceptionDirectEvidenceError("Manifest training Git is unauthorized")
    if (
        training_git["repo"] != evidence_git["repo"]
        or training_git["repo_url"] != evidence_git["repo_url"]
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Manifest training and evidence Git identities name different repositories"
        )
    producers = _producer_references(manifest["producers"], evidence_git_ref=evidence_git["ref"])
    training_recipe = _source_reference(
        manifest["training_recipe"],
        name="manifest training recipe",
        expected_git_ref=training_git["ref"],
    )
    if training_recipe["repo_relative_path"] != "src/scripts/train/Vision-Alignment.py":
        raise SSMaxPerceptionDirectEvidenceError("Manifest training recipe path differs")
    repository_root = _repository_root()
    amendment_reference = _authorized_amendment_reference(
        manifest["protocol_amendment"],
        repository_root=repository_root,
        require_content_sha=True,
    )
    bridge_parent = _exact(
        manifest["bridge_parent"], paired._BRIDGE_PARENT_FIELDS, name="direct bridge parent"
    )
    if not isinstance(bridge_parent["checkpoint"], str) or not bridge_parent["checkpoint"]:
        raise SSMaxPerceptionDirectEvidenceError("Direct bridge parent checkpoint is invalid")
    for field in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "gate_semantic_sha256",
    ):
        _sha(bridge_parent[field], name=f"direct bridge parent {field}")
    _artifact_shape(bridge_parent["gate"], name="direct bridge parent gate")
    _artifact_shape(
        manifest["perception_provenance"], name="direct perception provenance", semantic=True
    )
    _artifact_shape(manifest["source_audit"], name="direct source audit")
    _sha(manifest["source_audit_fingerprint"], name="direct source-audit fingerprint")
    try:
        projection = paired._validate_single_response_binding(
            manifest["single_response_projection"], verify_live=verify_live
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    _artifact_shape(manifest["attention_probe"], name="direct attention probe")
    _artifact_shape(manifest["text_sentinel"], name="direct text sentinel")
    _validate_evaluation(manifest["evaluation"])
    _validate_topology(manifest["topology"])
    _validate_policy(manifest["policy"])
    pairings = _exact(manifest["pairings"], frozenset(SOURCES), name="direct pairings")
    for source in SOURCES:
        _artifact_shape(pairings[source], name=f"direct {source} pairing")
    targets = _exact(
        manifest["loss_mass_targets"], frozenset(SOURCES), name="direct loss-mass targets"
    )
    total = 0.0
    for source in SOURCES:
        target = _finite(targets[source], name=f"{source} loss-mass target")
        if target <= 0:
            raise SSMaxPerceptionDirectEvidenceError("Direct loss-mass targets must be positive")
        total += target
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SSMaxPerceptionDirectEvidenceError("Direct loss-mass targets must sum to one")
    run = _exact(manifest["run"], _RUN_FIELDS, name="direct run")
    identity = DIRECT_RUN_IDENTITIES[str(model_variant)]
    if (
        run["run_name"] != identity["run_name"]
        or run["checkpoint_root"] != identity["checkpoint_root"]
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct run identity differs from amendment")
    profile = _source_reference(
        run["training_profile"],
        name="direct training profile",
        expected_git_ref=training_git["ref"],
    )
    if (
        profile["repo_relative_path"] != identity["profile"]
        or profile["sha256"] != identity["profile_sha256"]
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct training profile is unauthorized")
    for field in ("data_contract_sha256", "trainable_contract_sha256"):
        _sha(run[field], name=f"direct run {field}")
    checkpoints = _exact(
        run["checkpoints"],
        frozenset(str(step) for step in REQUIRED_STEPS),
        name="direct checkpoints",
    )
    for step in REQUIRED_STEPS:
        checkpoint = _checkpoint_reference(
            checkpoints[str(step)],
            step=step,
            verify_live=verify_live,
            workers=hash_workers,
        )
        if Path(str(checkpoint["path"])).resolve().parent != Path(run["checkpoint_root"]).resolve():
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct step{step} checkpoint is outside its authorized root"
            )
    if len({item["config_sha256"] for item in checkpoints.values()}) != 1:
        raise SSMaxPerceptionDirectEvidenceError("Direct checkpoint config identities differ")
    if {item["trainer_state_count"] for item in checkpoints.values()} != {
        manifest["topology"]["world_size"]
    }:
        raise SSMaxPerceptionDirectEvidenceError("Direct checkpoint trainer world sizes differ")
    expected_lineage = _sha(
        manifest["lineage_contract_sha256"], name="direct lineage contract SHA-256"
    )
    if canonical_sha256(_lineage_contract_payload(manifest)) != expected_lineage:
        raise SSMaxPerceptionDirectEvidenceError("Direct lineage contract SHA-256 differs")
    _content_sha(manifest, name="direct manifest")

    if verify_live:
        _validate_evidence_or_joint_consumer_checkout(
            evidence_git=evidence_git,
            repository_root=repository_root,
        )
        validate_evidence_git_compatibility(
            training_git=training_git,
            evidence_git=evidence_git,
            repository_root=repository_root,
        )
        actual_recipe = _git_blob_reference(
            git=training_git,
            repository_root=repository_root,
            repo_relative_path=training_recipe["repo_relative_path"],
            require_live_equal=False,
        )
        actual_profile = _git_blob_reference(
            git=training_git,
            repository_root=repository_root,
            repo_relative_path=profile["repo_relative_path"],
            require_live_equal=False,
        )
        if actual_recipe != training_recipe or actual_profile != profile:
            raise SSMaxPerceptionDirectEvidenceError("Direct training Git blobs differ")
        actual_producers = {
            name: _git_blob_reference(
                git=evidence_git,
                repository_root=repository_root,
                repo_relative_path=relative,
                require_live_equal=True,
            )
            for name, relative in PRODUCER_RELATIVE_PATHS.items()
        }
        if actual_producers != producers:
            raise SSMaxPerceptionDirectEvidenceError("Direct producer Git blobs differ")
        config = _mapping(
            load_json(Path(run["checkpoint_root"]) / "step0" / "config.json"),
            name="live direct step0 config",
        )
        pseudo_spec = {
            "model_variant": model_variant,
            "run_name": run["run_name"],
            "checkpoint_root": run["checkpoint_root"],
            "training_git": training_git,
            "topology": manifest["topology"],
            "policy": manifest["policy"],
        }
        summary = validate_saved_config(config, spec=pseudo_spec, training_profile=profile)
        if (
            summary["data_contract_sha256"] != run["data_contract_sha256"]
            or summary["trainable_contract_sha256"] != run["trainable_contract_sha256"]
            or summary["loss_mass_targets"] != dict(targets)
        ):
            raise SSMaxPerceptionDirectEvidenceError("Live direct saved contracts differ")
        if paired._single_response_binding_from_config(config) != dict(projection):
            raise SSMaxPerceptionDirectEvidenceError("Live direct projection binding differs")
        saved_data = _mapping(config.get("data"), name="live direct perception data")
        if (
            Path(str(saved_data.get("perception_provenance_path"))).expanduser().resolve()
            != Path(str(manifest["perception_provenance"]["path"])).expanduser().resolve()
            or saved_data.get("perception_provenance_sha256")
            != manifest["perception_provenance"]["sha256"]
            or Path(str(saved_data.get("source_audit_path"))).expanduser().resolve()
            != Path(str(manifest["source_audit"]["path"])).expanduser().resolve()
            or saved_data.get("source_audit_fingerprint") != manifest["source_audit_fingerprint"]
        ):
            raise SSMaxPerceptionDirectEvidenceError(
                "Live direct saved data artifacts differ from the manifest"
            )
        try:
            paired._validate_calibration_git_blobs(
                {
                    "repo": training_git["repo"],
                    "repo_url": training_git["repo_url"],
                    "ref": training_git["ref"],
                },
                recipe_path=repository_root / training_recipe["repo_relative_path"],
                calibration=_mapping(
                    projection["calibration"], name="direct projection calibration reference"
                ),
            )
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
        try:
            actual_parent = paired._validate_bridge_parent(
                {paired.CONTROL_ARM: config, paired.TREATMENT_ARM: config},
                model_variant=str(model_variant),
                gate_reference=bridge_parent["gate"],
                verify_live_checkpoint=True,
            )
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
        if actual_parent != dict(bridge_parent):
            raise SSMaxPerceptionDirectEvidenceError("Live direct bridge parent differs")
        try:
            provenance_path = paired.validate_artifact_reference(
                {
                    "path": manifest["perception_provenance"]["path"],
                    "sha256": manifest["perception_provenance"]["sha256"],
                },
                name="direct perception provenance",
            )
            provenance_payload = _mapping(
                load_json(provenance_path), name="direct perception provenance"
            )
            provenance_content = _sha(
                provenance_payload.get("content_sha256"),
                name="direct perception provenance content SHA-256",
            )
            if (
                provenance_content != manifest["perception_provenance"]["content_sha256"]
                or canonical_sha256(
                    {
                        field: item
                        for field, item in provenance_payload.items()
                        if field != "content_sha256"
                    }
                )
                != provenance_content
            ):
                raise SSMaxPerceptionDirectEvidenceError(
                    "Live direct perception provenance semantic SHA-256 differs"
                )
            typed_provenance = paired.load_perception_provenance_manifest(
                provenance_path,
                expected_sha256=manifest["perception_provenance"]["sha256"],
                verify_finevision_materialization=False,
                load_image_path_signatures=False,
            )
            audit_path = paired.validate_artifact_reference(
                manifest["source_audit"], name="direct source audit"
            )
            audit = _mapping(load_json(audit_path), name="direct source audit")
            unsigned_audit = dict(audit)
            recorded_fingerprint = unsigned_audit.pop("fingerprint", None)
            if (
                recorded_fingerprint != manifest["source_audit_fingerprint"]
                or canonical_sha256(unsigned_audit) != recorded_fingerprint
            ):
                raise SSMaxPerceptionDirectEvidenceError(
                    "Live direct source-audit fingerprint differs"
                )
            sentinel_path = paired.validate_artifact_reference(
                manifest["text_sentinel"], name="direct native text sentinel"
            )
            sentinel = paired._validate_text_sentinel(sentinel_path)
            saved_artifacts = _mapping(config.get("artifacts"), name="live direct artifacts")
            if sentinel["tokenizer"] != {
                "identifier": saved_artifacts.get("tokenizer_id"),
                "revision": saved_artifacts.get("tokenizer_revision"),
            }:
                raise SSMaxPerceptionDirectEvidenceError(
                    "Live direct text sentinel tokenizer differs from the saved config"
                )
            paired._validate_attention_probe_reference(
                manifest["attention_probe"],
                provenance=typed_provenance,
                projection_contract=_mapping(
                    manifest["single_response_projection"],
                    name="direct single-response projection",
                )["contract"],
                verify_live=True,
            )
            for source in SOURCES:
                selection = typed_provenance.selection(source, "validation")
                paired._validate_pairing_reference(
                    manifest["pairings"][source],
                    source=source,
                    evaluation=manifest["evaluation"],
                    verify_live=True,
                    dataset_size=len(selection.indices),
                    expected_content_ids_sha256=paired.content_ids_sha256(
                        selection.row_image_content_sha256
                    ),
                )
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
        _authorized_amendment_reference(
            amendment_reference,
            repository_root=repository_root,
            require_content_sha=True,
        )
    return manifest


def load_manifest(
    path: Path, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Load and validate one finalized direct perception manifest."""

    return validate_manifest(load_json(path), verify_live=verify_live, hash_workers=hash_workers)


def manifest_reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the raw and semantic identity of one finalized direct manifest."""

    reference = artifact_reference(path)
    reference["content_sha256"] = _sha(
        manifest.get("content_sha256"), name="direct manifest semantic SHA-256"
    )
    return reference


def validate_manifest_producer_source(
    manifest: Mapping[str, Any], *, producer: str, source_path: Path
) -> dict[str, str]:
    """Prove a direct evidence producer is the manifest-bound evidence Git blob."""

    if producer not in PRODUCER_RELATIVE_PATHS:
        raise SSMaxPerceptionDirectEvidenceError(f"Unknown direct producer {producer!r}")
    evidence_git = _git_identity(manifest["evidence_git"], name="manifest evidence Git")
    references = _producer_references(manifest["producers"], evidence_git_ref=evidence_git["ref"])
    expected_relative = PRODUCER_RELATIVE_PATHS[producer]
    source = source_path.expanduser().resolve()
    repository_root = source
    for _ in Path(expected_relative).parts:
        repository_root = repository_root.parent
    if source != (repository_root / expected_relative).resolve():
        raise SSMaxPerceptionDirectEvidenceError("Direct producer path is non-canonical")
    try:
        bridge._validate_repository_checkout(evidence_git, repository_root=repository_root)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    actual = _git_blob_reference(
        git=evidence_git,
        repository_root=repository_root,
        repo_relative_path=expected_relative,
        require_live_equal=True,
    )
    if actual != references[producer]:
        raise SSMaxPerceptionDirectEvidenceError("Direct producer source identity differs")
    return actual


def _validate_manifest_reference(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    expected_path: Path,
    name: str,
) -> dict[str, str]:
    reference = _artifact_shape(value, name=f"{name} manifest reference", semantic=True)
    if reference["content_sha256"] != manifest["content_sha256"]:
        raise SSMaxPerceptionDirectEvidenceError(
            f"{name} names a different manifest semantic SHA-256"
        )
    try:
        path = paired.validate_artifact_reference(
            {"path": reference["path"], "sha256": reference["sha256"]},
            name=f"{name} manifest",
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    if path != expected_path.expanduser().resolve():
        raise SSMaxPerceptionDirectEvidenceError(f"{name} names a different manifest path")
    return {field: str(reference[field]) for field in _MANIFEST_REF_FIELDS}


def _load_receipt_reference(
    reference: Any,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    step: int,
    expected_format: str,
) -> tuple[Path, Mapping[str, Any]]:
    try:
        path = paired.validate_artifact_reference(
            reference, name=f"direct step{step} {expected_format}"
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    payload = _mapping(load_json(path), name=f"direct step{step} receipt")
    fields = (
        _EVALUATION_RECEIPT_FIELDS
        if expected_format == EVALUATION_RECEIPT_FORMAT
        else _HEALTH_RECEIPT_FIELDS
    )
    _exact(payload, fields, name=f"direct step{step} receipt")
    if (
        payload["format"] != expected_format
        or type(payload["version"]) is not int
        or payload["version"] != SCHEMA_VERSION
        or payload["run_id"] != manifest["run_id"]
        or payload["model_variant"] != manifest["model_variant"]
        or type(payload["step"]) is not int
        or payload["step"] != step
    ):
        raise SSMaxPerceptionDirectEvidenceError(f"Direct step{step} receipt identity differs")
    embedded_checkpoint = _checkpoint_reference(
        payload["checkpoint"], step=step, verify_live=False, workers=1
    )
    if dict(embedded_checkpoint) != dict(manifest["run"]["checkpoints"][str(step)]):
        raise SSMaxPerceptionDirectEvidenceError(
            f"Direct step{step} receipt checkpoint identity differs"
        )
    if payload["status"] not in ("passed", "failed"):
        raise SSMaxPerceptionDirectEvidenceError(f"Direct step{step} receipt status is invalid")
    _timestamp(payload["created_at"], name=f"direct step{step} receipt created_at")
    _validate_manifest_reference(
        payload["manifest"],
        manifest=manifest,
        expected_path=manifest_path,
        name=f"direct step{step}",
    )
    _content_sha(payload, name=f"direct step{step} receipt")
    return path, payload


def _sentinel_int64_descriptor(values: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(values, list) or any(type(item) is not int for item in values):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} must be an integer list")
    tensor = np.asarray([values], dtype=np.int64)
    return {
        "dtype": "torch.int64",
        "shape": list(tensor.shape),
        "numel": int(tensor.size),
        "sha256": hashlib.sha256(tensor.tobytes(order="C")).hexdigest(),
    }


def _validate_direct_text_tensor_descriptor(
    value: Any, *, name: str, require_finite: bool
) -> dict[str, Any]:
    fields = (
        _DIRECT_TEXT_SENTINEL_OUTPUT_TENSOR_FIELDS
        if require_finite
        else _DIRECT_TEXT_SENTINEL_INPUT_TENSOR_FIELDS
    )
    descriptor = _exact(value, fields, name=name)
    dtype = descriptor["dtype"]
    if not isinstance(dtype, str) or not dtype.startswith("torch."):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} dtype is invalid")
    shape = descriptor["shape"]
    if not isinstance(shape, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise SSMaxPerceptionDirectEvidenceError(f"{name} shape is invalid")
    numel = _integer(descriptor["numel"], name=f"{name} numel", minimum=1)
    if math.prod(shape) != numel:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} shape/numel differ")
    _sha(descriptor["sha256"], name=f"{name} SHA-256")
    if require_finite and type(descriptor["finite"]) is not bool:
        raise SSMaxPerceptionDirectEvidenceError(f"{name} finite flag must be boolean")
    return dict(descriptor)


def _expected_direct_text_invariants(*, manifest: Mapping[str, Any], step: int) -> dict[str, Any]:
    try:
        sentinel_path = paired.validate_artifact_reference(
            manifest["text_sentinel"], name="direct native text sentinel"
        )
        sentinel = paired._validate_text_sentinel(sentinel_path)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    input_descriptor = _sentinel_int64_descriptor(
        sentinel["input_ids"], name="native text input_ids"
    )
    labels_descriptor = _sentinel_int64_descriptor(sentinel["labels"], name="native text labels")
    token_count = sum(label != -100 for label in sentinel["labels"])
    if (
        input_descriptor["dtype"] != "torch.int64"
        or input_descriptor["shape"] != [1, 256]
        or input_descriptor["numel"] != 256
        or labels_descriptor["dtype"] != "torch.int64"
        or labels_descriptor["shape"] != [1, 256]
        or labels_descriptor["numel"] != 256
        or token_count != 256
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct native text sentinel tensor/token geometry differs"
        )
    world_size = _integer(manifest["topology"]["world_size"], name="direct world size", minimum=1)
    reference = manifest["run"]["checkpoints"]["0"]
    candidate = manifest["run"]["checkpoints"][str(step)]
    return {
        "protocol": DIRECT_TEXT_SENTINEL_PROTOCOL,
        "version": 1,
        "artifact_sha256": manifest["text_sentinel"]["sha256"],
        "reference_step": 0,
        "reference_checkpoint_identity_sha256": reference["identity_sha256"],
        "candidate_step": step,
        "candidate_checkpoint_identity_sha256": candidate["identity_sha256"],
        "topology": dict(manifest["topology"]),
        "world_size": world_size,
        "input": input_descriptor,
        "labels": labels_descriptor,
        "token_count": token_count,
        "rank_count": world_size,
    }


def _validate_direct_text_result(
    value: Any, *, manifest: Mapping[str, Any], step: int
) -> Mapping[str, Any]:
    result = _exact(
        value,
        _DIRECT_TEXT_SENTINEL_RESULT_FIELDS,
        name=f"direct step{step} native text sentinel",
    )
    expected = _expected_direct_text_invariants(manifest=manifest, step=step)
    _validate_topology(result["topology"])
    for field, expected_value in expected.items():
        if field in {
            "version",
            "reference_step",
            "candidate_step",
            "world_size",
            "token_count",
            "rank_count",
        }:
            _integer(result[field], name=f"direct native text {field}")
        elif field in {
            "artifact_sha256",
            "reference_checkpoint_identity_sha256",
            "candidate_checkpoint_identity_sha256",
        }:
            _sha(result[field], name=f"direct native text {field}")
        if result[field] != expected_value:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct step{step} native text {field} differs"
            )
    _validate_direct_text_tensor_descriptor(
        result["input"], name="direct native text input", require_finite=False
    )
    _validate_direct_text_tensor_descriptor(
        result["labels"], name="direct native text labels", require_finite=False
    )
    rank_rows = result["rank_rows"]
    world_size = int(expected["world_size"])
    if not isinstance(rank_rows, list) or len(rank_rows) != world_size:
        raise SSMaxPerceptionDirectEvidenceError(
            f"Direct step{step} native text must contain exactly {world_size} rank rows"
        )
    computed_mismatches = 0
    for expected_rank, raw_row in enumerate(rank_rows):
        row = _exact(
            raw_row,
            _DIRECT_TEXT_SENTINEL_RANK_FIELDS,
            name=f"direct step{step} native text rank{expected_rank}",
        )
        rank = _integer(row["rank"], name=f"direct native text rank{expected_rank} rank")
        if rank != expected_rank:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct step{step} native text rank rows are not in global-rank order"
            )
        descriptors: dict[str, dict[str, dict[str, Any]]] = {}
        for snapshot_name in ("reference", "candidate"):
            snapshot = _exact(
                row[snapshot_name],
                _DIRECT_TEXT_SENTINEL_OUTPUT_FIELDS,
                name=f"direct native text rank{expected_rank} {snapshot_name}",
            )
            descriptors[snapshot_name] = {
                output_name: _validate_direct_text_tensor_descriptor(
                    snapshot[output_name],
                    name=(
                        f"direct native text rank{expected_rank} " f"{snapshot_name} {output_name}"
                    ),
                    require_finite=True,
                )
                for output_name in ("logits", "ce")
            }
            for output_name, expected_shape in (
                ("logits", [1, 256, 100352]),
                ("ce", [1, 256]),
            ):
                descriptor = descriptors[snapshot_name][output_name]
                if descriptor["shape"] != expected_shape or descriptor["dtype"] not in {
                    "torch.bfloat16",
                    "torch.float16",
                    "torch.float32",
                    "torch.float64",
                }:
                    raise SSMaxPerceptionDirectEvidenceError(
                        f"Direct native text rank{expected_rank} {snapshot_name} "
                        f"{output_name} geometry/dtype differs"
                    )
        for field in ("logits_exact", "ce_exact", "passed"):
            if type(row[field]) is not bool:
                raise SSMaxPerceptionDirectEvidenceError(
                    f"Direct native text rank{expected_rank} {field} must be boolean"
                )
        finite = all(
            descriptors[snapshot][output]["finite"]
            for snapshot in ("reference", "candidate")
            for output in ("logits", "ce")
        )
        logits_descriptors_equal = (
            descriptors["reference"]["logits"] == descriptors["candidate"]["logits"]
        )
        ce_descriptors_equal = descriptors["reference"]["ce"] == descriptors["candidate"]["ce"]
        if row["logits_exact"] is not logits_descriptors_equal:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct native text rank{expected_rank} logits exact claim differs"
            )
        if row["ce_exact"] is not ce_descriptors_equal:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct native text rank{expected_rank} CE exact claim differs"
            )
        derived_pass = bool(row["logits_exact"] and row["ce_exact"] and finite)
        if row["passed"] is not derived_pass:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct native text rank{expected_rank} pass flag differs"
            )
        computed_mismatches += int(not derived_pass)
    mismatch_count = _integer(result["mismatch_count"], name="direct native text mismatch_count")
    if mismatch_count != computed_mismatches:
        raise SSMaxPerceptionDirectEvidenceError("Direct native text mismatch count differs")
    expected_all_ranks_passed = computed_mismatches == 0
    if (
        type(result["all_ranks_passed"]) is not bool
        or result["all_ranks_passed"] is not expected_all_ranks_passed
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct native text all-ranks status differs")
    expected_inventory = canonical_sha256(rank_rows)
    if (
        _sha(result["rank_inventory_sha256"], name="direct native text rank inventory SHA-256")
        != expected_inventory
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct native text rank inventory SHA-256 differs"
        )
    _content_sha(result, name=f"direct step{step} native text sentinel")
    return result


def _direct_text_comparison_invariants(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: value[field]
        for field in _DIRECT_TEXT_SENTINEL_INVARIANT_FIELDS
        if field not in {"candidate_step", "candidate_checkpoint_identity_sha256"}
    }


def _validate_direct_evaluation_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], step: int
) -> dict[str, list[Mapping[str, Any]]]:
    try:
        paired._validate_strict_load(
            receipt["strict_generic_dcp_load"], name="direct strict generic DCP load"
        )
        state = paired._validate_state(receipt["state"], arm=TRAINING_ROLE, step=step)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    text_result = _validate_direct_text_result(
        receipt["text_sentinel"], manifest=manifest, step=step
    )
    expected_pass = bool(
        state["frozen_lm"]["mismatch_count"] == 0
        and state["non_image_embedding_rows"]["mismatch_count"] == 0
        and text_result["all_ranks_passed"]
    )
    if (receipt["status"] == "passed") is not expected_pass:
        raise SSMaxPerceptionDirectEvidenceError(
            f"Direct step{step} evaluation status differs from state/text evidence"
        )
    attention = _mapping(
        receipt["attention_diagnostics"], name=f"direct step{step} attention diagnostics"
    )
    if (
        attention.get("checkpoint") != manifest["run"]["checkpoints"][str(step)]
        or not isinstance(attention.get("protocol"), Mapping)
        or attention["protocol"].get("manifest_sha256") != manifest["attention_probe"]["sha256"]
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            f"Direct step{step} attention diagnostics differ from the manifest"
        )
    try:
        paired.validate_ssmax_attention_report(
            attention, label=f"direct step{step} attention diagnostics"
        )
    except ValueError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    if receipt["pairings"] != manifest["pairings"]:
        raise SSMaxPerceptionDirectEvidenceError(
            f"Direct step{step} evaluation changes fixed pairings"
        )
    evidence_git = _git_identity(manifest["evidence_git"], name="manifest evidence Git")
    producers = _producer_references(manifest["producers"], evidence_git_ref=evidence_git["ref"])
    evaluator = _source_reference(
        receipt["evaluator"],
        name=f"direct step{step} evaluator",
        expected_git_ref=evidence_git["ref"],
    )
    if evaluator != producers[EVALUATION_PRODUCER]:
        raise SSMaxPerceptionDirectEvidenceError("Direct evaluator source identity differs")
    results = _exact(receipt["results"], frozenset(SOURCES), name="direct evaluation results")
    output: dict[str, list[Mapping[str, Any]]] = {}
    for source in SOURCES:
        result = _exact(
            results[source],
            frozenset({"pairing_sha256", "examples", "per_example"}),
            name=f"direct {source} result",
        )
        if (
            result["pairing_sha256"] != manifest["pairings"][source]["sha256"]
            or type(result["examples"]) is not int
            or result["examples"] != manifest["evaluation"]["examples_per_source"]
        ):
            raise SSMaxPerceptionDirectEvidenceError(
                f"Direct {source} result pairing/count differs"
            )
        try:
            pairing_path = paired.validate_artifact_reference(
                manifest["pairings"][source], name=f"direct {source} pairing"
            )
            pairing = _mapping(load_json(pairing_path), name=f"direct {source} pairing")
            output[source] = paired._validate_rows(
                result["per_example"], source=source, manifest=manifest, pairing=pairing
            )
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    return output


def _paired_health_manifest_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Translate only the fields consumed by the historical v2 health validator."""

    evidence_git = _git_identity(manifest["evidence_git"], name="manifest evidence Git")
    producers = _producer_references(manifest["producers"], evidence_git_ref=evidence_git["ref"])
    translated_producers = {
        name: {
            **reference,
            "repo_relative_path": paired.PRODUCER_RELATIVE_PATHS[name],
        }
        for name, reference in producers.items()
        if name in (EVALUATION_PRODUCER, HEALTH_PRODUCER)
    }
    recipe = {
        "path": manifest["training_recipe"]["repo_relative_path"],
        "sha256": manifest["training_recipe"]["sha256"],
    }
    return {
        "version": paired.PERCEPTION_V2_SCHEMA_VERSION,
        "pair_id": manifest["run_id"],
        "model_variant": manifest["model_variant"],
        "git": {
            "repo": evidence_git["repo"],
            "repo_url": evidence_git["repo_url"],
            "ref": evidence_git["ref"],
        },
        "producers": translated_producers,
        "recipe": recipe,
        "topology": manifest["topology"],
        "policy": paired._locked_promotion_policy(paired.PERCEPTION_V2_SCHEMA_VERSION),
        "loss_mass_targets": manifest["loss_mass_targets"],
        "arms": {TRAINING_ROLE: manifest["run"]},
    }


def _validate_direct_health_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], step: int
) -> dict[str, Any]:
    rank_states = receipt.get("rank_states")
    world_size = int(manifest["topology"]["world_size"])
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise SSMaxPerceptionDirectEvidenceError(f"Direct step{step} health omits trainer ranks")
    for expected_rank, raw_state in enumerate(rank_states):
        state = _mapping(raw_state, name=f"direct step{step} rank{expected_rank}")
        for field, expected in (
            ("rank", expected_rank),
            ("global_step", step),
            ("batches_processed", step),
        ):
            observed = _integer(
                state.get(field), name=f"direct step{step} rank{expected_rank} {field}"
            )
            if observed != expected:
                raise SSMaxPerceptionDirectEvidenceError(
                    f"Direct step{step} rank{expected_rank} {field} differs"
                )
    evidence = _exact(
        receipt["evidence"],
        frozenset({"training_recipe", "producer"}),
        name=f"direct step{step} health evidence",
    )
    training_git = _git_identity(manifest["training_git"], name="manifest training Git")
    recipe = _source_reference(
        evidence["training_recipe"],
        name=f"direct step{step} health training recipe",
        expected_git_ref=training_git["ref"],
    )
    if recipe != manifest["training_recipe"]:
        raise SSMaxPerceptionDirectEvidenceError("Direct health training recipe differs")
    evidence_git = _git_identity(manifest["evidence_git"], name="manifest evidence Git")
    producers = _producer_references(manifest["producers"], evidence_git_ref=evidence_git["ref"])
    producer = _source_reference(
        evidence["producer"],
        name=f"direct step{step} health producer",
        expected_git_ref=evidence_git["ref"],
    )
    if producer != producers[HEALTH_PRODUCER]:
        raise SSMaxPerceptionDirectEvidenceError("Direct health producer source differs")
    compatibility_manifest = _paired_health_manifest_view(manifest)
    translated = dict(receipt)
    translated["evidence"] = {
        "recipe": compatibility_manifest["recipe"],
        "producer": compatibility_manifest["producers"][HEALTH_PRODUCER],
    }
    try:
        return paired._validate_health_receipt(
            translated,
            manifest=compatibility_manifest,
            arm=TRAINING_ROLE,
            step=step,
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def _receipt_map(value: Any, *, name: str) -> dict[int, Mapping[str, str]]:
    mapping = _mapping(value, name=name)
    converted: dict[int, Mapping[str, str]] = {}
    for raw_step, reference in mapping.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise SSMaxPerceptionDirectEvidenceError(
                f"{name} step {raw_step!r} is invalid"
            ) from error
        if step in converted:
            raise SSMaxPerceptionDirectEvidenceError(f"{name} repeats step{step}")
        converted[step] = _exact(reference, _ARTIFACT_REF_FIELDS, name=f"{name} step{step}")
    if set(converted) != set(REQUIRED_STEPS):
        raise SSMaxPerceptionDirectEvidenceError(
            f"{name} must contain exactly steps {list(REQUIRED_STEPS)}"
        )
    return converted


def _row_identity(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["pairing_position"]),
        int(row["recipient_index"]),
        int(row["donor_index"]),
        int(row["response_tokens"]),
    )


def _metric_arrays(
    evaluations: Mapping[int, Mapping[str, list[Mapping[str, Any]]]],
    *,
    source: str,
    window: str,
) -> dict[str, np.ndarray]:
    rows = {step: evaluations[step][source] for step in REQUIRED_STEPS}
    identities = [_row_identity(row) for row in rows[0]]
    if any([_row_identity(row) for row in candidate] != identities for candidate in rows.values()):
        raise SSMaxPerceptionDirectEvidenceError(
            f"Direct {source} rows are not exactly paired across steps"
        )

    def values(step: int, field: str) -> np.ndarray:
        return np.asarray([float(row[field][window]) for row in rows[step]], dtype=np.float64)

    return {
        "step0_gap": values(0, "ce_gap_wrong_minus_correct"),
        "step3000_gap": values(3000, "ce_gap_wrong_minus_correct"),
        "step4000_gap": values(4000, "ce_gap_wrong_minus_correct"),
        "step0_correct_ce": values(0, "correct_ce"),
        "step4000_correct_ce": values(4000, "correct_ce"),
    }


def _source_balanced_interval(
    values: Mapping[str, np.ndarray], *, seed: int, samples: int
) -> dict[str, Any]:
    try:
        return paired._source_balanced_interval(values, seed=seed, samples=samples)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error


def build_promotion_report(
    *,
    manifest_path: Path,
    evaluation_receipts: Mapping[int, Mapping[str, str]],
    health_receipts: Mapping[int, Mapping[str, str]],
    created_at: str,
    verify_live_manifest: bool = True,
) -> dict[str, Any]:
    """Rebuild one non-causal, within-lineage admission decision from six receipts."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_manifest)
    report_time = _timestamp(created_at, name="direct promotion created_at")
    manifest_time = _timestamp(manifest["created_at"], name="direct manifest created_at")
    if report_time < manifest_time:
        raise SSMaxPerceptionDirectEvidenceError("Direct promotion report predates its manifest")
    evaluation_refs = _receipt_map(evaluation_receipts, name="direct evaluation receipts")
    health_refs = _receipt_map(health_receipts, name="direct health receipts")
    evaluations: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
    evaluation_payloads: dict[int, Mapping[str, Any]] = {}
    health_summaries: dict[int, dict[str, Any]] = {}
    deviations: list[dict[str, Any]] = []
    receipt_output: dict[str, Any] = {}
    for step in REQUIRED_STEPS:
        _, evaluation = _load_receipt_reference(
            evaluation_refs[step],
            manifest=manifest,
            manifest_path=manifest_path,
            step=step,
            expected_format=EVALUATION_RECEIPT_FORMAT,
        )
        _, health = _load_receipt_reference(
            health_refs[step],
            manifest=manifest,
            manifest_path=manifest_path,
            step=step,
            expected_format=HEALTH_RECEIPT_FORMAT,
        )
        evaluation_time = _timestamp(evaluation["created_at"], name="direct evaluation created_at")
        health_time = _timestamp(health["created_at"], name="direct health created_at")
        if (
            evaluation_time < manifest_time
            or health_time < manifest_time
            or evaluation_time > report_time
            or health_time > report_time
        ):
            raise SSMaxPerceptionDirectEvidenceError(
                "Direct receipt ordering differs from manifest <= receipt <= report"
            )
        evaluations[step] = _validate_direct_evaluation_receipt(
            evaluation, manifest=manifest, step=step
        )
        health_summaries[step] = _validate_direct_health_receipt(
            health, manifest=manifest, step=step
        )
        evaluation_payloads[step] = evaluation
        receipt_output[str(step)] = {
            "evaluation": dict(evaluation_refs[step]),
            "health": dict(health_refs[step]),
        }
        if evaluation["status"] != "passed":
            deviations.append({"kind": "evaluation_receipt_status", "step": step})
        if health["status"] != "passed":
            deviations.append({"kind": "health_receipt_status", "step": step})

    canonical_text = _direct_text_comparison_invariants(evaluation_payloads[0]["text_sentinel"])
    baseline_state = evaluation_payloads[0]["state"]
    for step in REQUIRED_STEPS:
        state = evaluation_payloads[step]["state"]
        for surface in ("frozen_lm", "non_image_embedding_rows"):
            if (
                state[surface]["mismatch_count"] != 0
                or state[surface]["reference_inventory_sha256"]
                != state[surface]["candidate_inventory_sha256"]
                or state[surface]["reference_inventory_sha256"]
                != baseline_state[surface]["reference_inventory_sha256"]
            ):
                deviations.append(
                    {"kind": "frozen_state_regression", "step": step, "surface": surface}
                )
        if (
            _direct_text_comparison_invariants(evaluation_payloads[step]["text_sentinel"])
            != canonical_text
        ):
            deviations.append({"kind": "native_text_sentinel_invariants_changed", "step": step})

    policy = manifest["policy"]
    tolerance = float(policy["loss_mass_share_tolerance"])
    for step in REQUIRED_STEPS:
        health = health_summaries[step]
        counters = health["run_counters"]
        for counter, maximum in (
            ("data_errors", int(policy["maximum_data_errors"])),
            ("optimizer_guard_skips", int(policy["maximum_optimizer_guard_skips"])),
            ("nonfinite_losses", int(policy["maximum_nonfinite_losses"])),
            ("nonfinite_gradients", int(policy["maximum_nonfinite_gradients"])),
        ):
            if counters[counter] > maximum:
                deviations.append(
                    {
                        "kind": "run_health_counter",
                        "step": step,
                        "counter": counter,
                        "observed": counters[counter],
                        "maximum": maximum,
                    }
                )
        if step == 0:
            continue
        for field, total_field, mass_kind in (
            ("loss_weight", "total_loss_weight", "raw"),
            ("active_loss_weight", "total_active_loss_weight", "active"),
        ):
            total = float(health[total_field])
            if total <= 0:
                deviations.append({"kind": "empty_loss_mass", "step": step, "mass_kind": mass_kind})
                continue
            for source in SOURCES:
                share = float(health["sources"][source][field]) / total
                target = float(manifest["loss_mass_targets"][source])
                if abs(share - target) > tolerance:
                    deviations.append(
                        {
                            "kind": "loss_mass",
                            "step": step,
                            "mass_kind": mass_kind,
                            "source": source,
                            "observed": share,
                            "target": target,
                        }
                    )

    guard_summary = health_summaries[int(policy["candidate_step"])]["optimizer_guard"]
    if not guard_summary["resume_free_passed"]:
        deviations.append(
            {
                "kind": "optimizer_guard_history_reset",
                "observed": guard_summary["optimizer_guard_history_reset_steps"],
                "required": guard_summary["required_optimizer_guard_history_reset_steps"],
            }
        )
    invalid_steps = [
        event["step"] for event in guard_summary["skip_events"] if not event["finite_gradient_only"]
    ]
    if invalid_steps:
        deviations.append(
            {"kind": "optimizer_guard_not_finite_gradient_only", "steps": invalid_steps}
        )
    if not guard_summary["spacing_passed"]:
        deviations.append(
            {
                "kind": "optimizer_guard_spacing",
                "observed": guard_summary["minimum_step_distance"],
                "minimum": guard_summary["required_minimum_step_distance"],
            }
        )
    if not guard_summary["final_window_passed"]:
        deviations.append(
            {
                "kind": "optimizer_guard_final_clean_window",
                "observed": guard_summary["clean_final_steps"],
                "minimum": guard_summary["required_clean_final_steps"],
            }
        )

    baseline_attention = evaluation_payloads[0]["attention_diagnostics"]
    attention_trajectory: dict[str, Any] = {
        "0": {
            "report_sha256": baseline_attention["report_sha256"],
            "comparison_from_step0": None,
        }
    }
    for step in REQUIRED_STEPS[1:]:
        candidate_attention = evaluation_payloads[step]["attention_diagnostics"]
        try:
            comparison = compare_ssmax_attention_reports(baseline_attention, candidate_attention)
        except ValueError as error:
            raise SSMaxPerceptionDirectEvidenceError(
                f"Could not compare direct step{step} attention diagnostics: {error}"
            ) from error
        attention_trajectory[str(step)] = {
            "report_sha256": candidate_attention["report_sha256"],
            "comparison_from_step0": comparison,
        }

    summary: dict[str, Any] = {
        "windows": {},
        "attention_trajectory": attention_trajectory,
        "optimizer_guard_trajectory": guard_summary,
    }
    samples = int(manifest["evaluation"]["bootstrap_samples"])
    base_seed = int(manifest["evaluation"]["bootstrap_seed"])
    for window_index, window in enumerate(WINDOWS):
        arrays = {
            source: _metric_arrays(evaluations, source=source, window=window) for source in SOURCES
        }
        absolute_values = {source: values["step4000_gap"] for source, values in arrays.items()}
        improvement_values = {
            source: values["step4000_gap"] - values["step0_gap"]
            for source, values in arrays.items()
        }
        absolute = _source_balanced_interval(
            absolute_values,
            seed=base_seed + window_index * 10_000,
            samples=samples,
        )
        improvement = _source_balanced_interval(
            improvement_values,
            seed=base_seed + 100_000 + window_index * 10_000,
            samples=samples,
        )
        window_sources: dict[str, Any] = {}
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
            window_sources[source] = {
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
                    {"kind": "source_nonpositive_absolute_gap", "source": source, "window": window}
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
        macro_durability_gap = float(
            np.mean([values["step3000_gap"].mean() for values in arrays.values()])
        )
        macro_baseline_gap = float(
            np.mean([values["step0_gap"].mean() for values in arrays.values()])
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
        summary["windows"][window] = {
            "candidate_absolute_gap": absolute,
            "candidate_gap_improvement_from_step0": improvement,
            "macro_step0_correct_ce": macro_baseline_ce,
            "macro_step4000_correct_ce": macro_candidate_ce,
            "macro_step0_gap": macro_baseline_gap,
            "macro_step3000_gap": macro_durability_gap,
            "macro_step4000_gap": macro_candidate_gap,
            "sources": window_sources,
        }

    report: dict[str, Any] = {
        "format": PROMOTION_REPORT_FORMAT,
        "version": SCHEMA_VERSION,
        "status": "passed" if not deviations else "rejected",
        "decision_scope": "within_lineage_noncausal_joint_admission",
        "created_at": created_at,
        "manifest": manifest_reference(manifest_path, manifest),
        "run_id": manifest["run_id"],
        "model_variant": manifest["model_variant"],
        "receipts": receipt_output,
        "summary": summary,
        "deviations": deviations,
    }
    report["content_sha256"] = canonical_sha256(report)
    return report


def validate_promotion_report_reference(
    value: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_data_contract_sha256: str,
    expected_trainable_contract_sha256: str,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Re-open and exactly reproduce a passed direct report from its six raw receipts."""

    if expected_model_variant not in MODEL_VARIANTS:
        raise SSMaxPerceptionDirectEvidenceError("Expected direct model variant is unsupported")
    try:
        report_path = paired.validate_artifact_reference(
            value, name="direct perception promotion report"
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    report = _exact(
        load_json(report_path), _PROMOTION_REPORT_FIELDS, name="direct perception promotion report"
    )
    if (
        report["format"] != PROMOTION_REPORT_FORMAT
        or type(report["version"]) is not int
        or report["version"] != SCHEMA_VERSION
        or report["status"] != "passed"
        or report["decision_scope"] != "within_lineage_noncausal_joint_admission"
        or report["model_variant"] != expected_model_variant
        or report["deviations"] != []
    ):
        raise SSMaxPerceptionDirectEvidenceError("Direct promotion report is not eligible")
    _timestamp(report["created_at"], name="direct promotion report created_at")
    _content_sha(report, name="direct promotion report")
    manifest_ref = _artifact_shape(
        report["manifest"], name="direct promotion report manifest", semantic=True
    )
    try:
        manifest_path = paired.validate_artifact_reference(
            {"path": manifest_ref["path"], "sha256": manifest_ref["sha256"]},
            name="direct promotion report manifest",
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    if (
        manifest["content_sha256"] != manifest_ref["content_sha256"]
        or manifest["run_id"] != report["run_id"]
        or manifest["model_variant"] != expected_model_variant
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion report names an incompatible manifest"
        )
    candidate = _checkpoint_reference(
        manifest["run"]["checkpoints"]["4000"],
        step=4000,
        verify_live=verify_live_checkpoint,
        workers=8,
    )
    if (
        Path(str(candidate["path"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or candidate["config_sha256"] != expected_checkpoint_config_sha256
        or manifest["run"]["data_contract_sha256"] != expected_data_contract_sha256
        or manifest["run"]["trainable_contract_sha256"] != expected_trainable_contract_sha256
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion report names a different candidate"
        )
    receipt_refs = _exact(
        report["receipts"],
        frozenset(str(step) for step in REQUIRED_STEPS),
        name="direct report receipts",
    )
    evaluation_refs: dict[int, Mapping[str, str]] = {}
    health_refs: dict[int, Mapping[str, str]] = {}
    for step in REQUIRED_STEPS:
        refs = _exact(
            receipt_refs[str(step)],
            frozenset({"evaluation", "health"}),
            name=f"direct report step{step} receipts",
        )
        evaluation_refs[step] = _exact(
            refs["evaluation"], _ARTIFACT_REF_FIELDS, name="direct evaluation receipt reference"
        )
        health_refs[step] = _exact(
            refs["health"], _ARTIFACT_REF_FIELDS, name="direct health receipt reference"
        )
    rebuilt = build_promotion_report(
        manifest_path=manifest_path,
        evaluation_receipts=evaluation_refs,
        health_receipts=health_refs,
        created_at=str(report["created_at"]),
        verify_live_manifest=verify_live_checkpoint,
    )
    if rebuilt != dict(report):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion report differs from its six bound raw receipts"
        )
    return {
        "report": report,
        "report_reference": dict(value),
        "manifest": manifest,
        "manifest_reference": dict(manifest_ref),
        "candidate": dict(candidate),
    }


def _candidate_metadata(candidate: Mapping[str, Any]) -> tuple[int, str]:
    config = _mapping(
        load_json(Path(str(candidate["path"])) / "config.json"), name="direct candidate config"
    )
    metadata = _mapping(
        config.get("vision_alignment"), name="direct candidate vision-alignment metadata"
    )
    recipe_version = _integer(
        metadata.get("recipe_version"), name="direct recipe version", minimum=1
    )
    formatter_version = metadata.get("formatter_version")
    if not isinstance(formatter_version, str) or not formatter_version:
        raise SSMaxPerceptionDirectEvidenceError("Direct candidate formatter version is malformed")
    return recipe_version, formatter_version


def build_parent_gate(
    *,
    promotion_report_path: Path,
    expected_promotion_report_sha256: str,
    approved_by: str,
    approved_at: str,
    verify_live_checkpoint: bool = True,
) -> dict[str, Any]:
    """Build one explicit, waiver-free human approval gate from a rebuilt direct report."""

    if not verify_live_checkpoint:
        raise SSMaxPerceptionDirectEvidenceError(
            "SSMax v7 approval requires live checkpoint and evidence checkout verification"
        )
    promotion_report_path = promotion_report_path.expanduser().resolve()
    if sha256_file(promotion_report_path) != expected_promotion_report_sha256:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct promotion report differs from its explicit approval pin"
        )
    raw_report = _exact(
        load_json(promotion_report_path), _PROMOTION_REPORT_FIELDS, name="direct promotion report"
    )
    manifest_ref = _artifact_shape(
        raw_report["manifest"], name="direct promotion report manifest", semantic=True
    )
    manifest_path = Path(str(manifest_ref["path"])).expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    candidate = manifest["run"]["checkpoints"]["4000"]
    summary = validate_promotion_report_reference(
        {"path": str(promotion_report_path), "sha256": expected_promotion_report_sha256},
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(manifest["run"]["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(manifest["run"]["trainable_contract_sha256"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if (
        not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxPerceptionDirectEvidenceError("approved_by is not a durable human identity")
    approval_time = _timestamp(approved_at, name="direct approval timestamp")
    report_time = _timestamp(
        summary["report"]["created_at"], name="direct promotion report timestamp"
    )
    if approval_time < report_time:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct human approval predates the promotion report"
        )
    recipe_version, formatter_version = _candidate_metadata(candidate)
    amendment = manifest["protocol_amendment"]
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": PARENT_GATE_VERSION,
        "status": "approved",
        "recipe_version": recipe_version,
        "formatter_version": formatter_version,
        "phase": "perception",
        "model_variant": manifest["model_variant"],
        "lineage_kind": LINEAGE_KIND,
        "run_id": manifest["run_id"],
        "checkpoint": candidate["path"],
        "checkpoint_config_sha256": candidate["config_sha256"],
        "checkpoint_identity_sha256": candidate["identity_sha256"],
        "data_contract_sha256": manifest["run"]["data_contract_sha256"],
        "trainable_contract_sha256": manifest["run"]["trainable_contract_sha256"],
        "global_step": 4000,
        "metrics_artifact_sha256": expected_promotion_report_sha256,
        "promotion_report_path": str(promotion_report_path),
        "promotion_report_sha256": expected_promotion_report_sha256,
        "promotion_report_content_sha256": summary["report"]["content_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_ref["sha256"],
        "manifest_content_sha256": manifest["content_sha256"],
        "protocol_amendment_path": amendment["path"],
        "protocol_amendment_sha256": amendment["sha256"],
        "protocol_amendment_content_sha256": amendment["content_sha256"],
        "training_git_ref": manifest["training_git"]["ref"],
        "evidence_git_ref": manifest["evidence_git"]["ref"],
        "approved_by": approved_by,
        "approved_at": approved_at,
        "waivers": [],
    }
    validate_ssmax_perception_direct_parent_gate(
        gate,
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(manifest["run"]["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(manifest["run"]["trainable_contract_sha256"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    return gate


def validate_ssmax_perception_direct_parent_gate(
    gate: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_data_contract_sha256: str,
    expected_trainable_contract_sha256: str,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Validate an exact version-7 direct-perception parent gate for a joint phase."""

    if not verify_live_checkpoint:
        raise SSMaxPerceptionDirectEvidenceError(
            "SSMax v7 eligibility requires live checkpoint and evidence checkout verification"
        )
    value = _exact(gate, _PARENT_GATE_FIELDS, name="SSMax direct perception parent gate")
    expected_pairs = (
        ("format", "vision_alignment_parent_gate"),
        ("version", PARENT_GATE_VERSION),
        ("status", "approved"),
        ("phase", "perception"),
        ("model_variant", expected_model_variant),
        ("lineage_kind", LINEAGE_KIND),
        ("global_step", 4000),
        ("checkpoint_config_sha256", expected_checkpoint_config_sha256),
        ("data_contract_sha256", expected_data_contract_sha256),
        ("trainable_contract_sha256", expected_trainable_contract_sha256),
        ("training_git_ref", TRAINING_GIT_REF),
    )
    for name, expected in expected_pairs:
        if type(value[name]) is not type(expected) or value[name] != expected:
            raise SSMaxPerceptionDirectEvidenceError(f"SSMax v7 parent gate {name} differs")
    if expected_model_variant not in MODEL_VARIANTS:
        raise SSMaxPerceptionDirectEvidenceError(
            "SSMax v7 parent gate model variant is unsupported"
        )
    if (
        not isinstance(value["run_id"], str)
        or not value["run_id"]
        or Path(str(value["checkpoint"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or expected_checkpoint.name != "step4000"
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "SSMax v7 parent gate must name one direct step4000 lineage"
        )
    _integer(value["recipe_version"], name="SSMax v7 recipe version", minimum=1)
    if not isinstance(value["formatter_version"], str) or not value["formatter_version"]:
        raise SSMaxPerceptionDirectEvidenceError("SSMax v7 formatter version is malformed")
    for name in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "metrics_artifact_sha256",
        "promotion_report_sha256",
        "promotion_report_content_sha256",
        "manifest_sha256",
        "manifest_content_sha256",
        "protocol_amendment_sha256",
        "protocol_amendment_content_sha256",
    ):
        _sha(value[name], name=f"SSMax v7 parent gate {name}")
    if (
        value["waivers"] != []
        or value["metrics_artifact_sha256"] != value["promotion_report_sha256"]
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "SSMax v7 parent gate is not waiver-free or changes its metrics artifact"
        )
    approved_by = value["approved_by"]
    if (
        not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxPerceptionDirectEvidenceError("SSMax v7 approved_by is not a durable identity")
    approval_time = _timestamp(value["approved_at"], name="SSMax v7 approved_at")
    if (
        not isinstance(value["evidence_git_ref"], str)
        or _GIT_REF_RE.fullmatch(value["evidence_git_ref"]) is None
    ):
        raise SSMaxPerceptionDirectEvidenceError("SSMax v7 evidence Git ref is malformed")
    summary = validate_promotion_report_reference(
        {
            "path": value["promotion_report_path"],
            "sha256": value["promotion_report_sha256"],
        },
        expected_checkpoint=expected_checkpoint,
        expected_checkpoint_config_sha256=expected_checkpoint_config_sha256,
        expected_model_variant=expected_model_variant,
        expected_data_contract_sha256=expected_data_contract_sha256,
        expected_trainable_contract_sha256=expected_trainable_contract_sha256,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if verify_live_checkpoint:
        recipe_version, formatter_version = _candidate_metadata(summary["candidate"])
        if (
            value["recipe_version"] != recipe_version
            or value["formatter_version"] != formatter_version
        ):
            raise SSMaxPerceptionDirectEvidenceError(
                "SSMax v7 recipe/formatter identity differs from the live candidate"
            )
    if summary["candidate"]["identity_sha256"] != value["checkpoint_identity_sha256"]:
        raise SSMaxPerceptionDirectEvidenceError("SSMax v7 checkpoint identity differs")
    if summary["report"]["content_sha256"] != value["promotion_report_content_sha256"]:
        raise SSMaxPerceptionDirectEvidenceError("SSMax v7 promotion semantic SHA-256 differs")
    manifest = summary["manifest"]
    manifest_ref = summary["manifest_reference"]
    amendment = manifest["protocol_amendment"]
    if (
        manifest["run_id"] != value["run_id"]
        or manifest["training_git"]["ref"] != value["training_git_ref"]
        or manifest["evidence_git"]["ref"] != value["evidence_git_ref"]
        or Path(str(manifest_ref["path"])).resolve()
        != Path(str(value["manifest_path"])).expanduser().resolve()
        or manifest_ref["sha256"] != value["manifest_sha256"]
        or manifest_ref["content_sha256"] != value["manifest_content_sha256"]
        or amendment["path"] != value["protocol_amendment_path"]
        or amendment["sha256"] != value["protocol_amendment_sha256"]
        or amendment["content_sha256"] != value["protocol_amendment_content_sha256"]
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "SSMax v7 lineage, Git, manifest, or amendment binding differs"
        )
    if approval_time < _timestamp(summary["report"]["created_at"], name="direct report created_at"):
        raise SSMaxPerceptionDirectEvidenceError("SSMax v7 approval predates its promotion report")
    return summary


__all__ = [
    "AMENDMENT_RELATIVE_PATH",
    "AMENDMENT_SHA256",
    "BASE_EVIDENCE_GIT_REF",
    "DIRECT_POLICY",
    "DIRECT_RUN_IDENTITIES",
    "DIRECT_TEXT_SENTINEL_PROTOCOL",
    "EVALUATION_CONTRACT",
    "EVALUATION_PRODUCER",
    "EVALUATION_RECEIPT_FORMAT",
    "EVIDENCE_GIT_DIFF_ALLOWLIST",
    "HEALTH_PRODUCER",
    "HEALTH_RECEIPT_FORMAT",
    "JOINT_CONSUMER_GIT_DIFF",
    "LEGACY_EVIDENCE_GIT_REFS",
    "LINEAGE_KIND",
    "MANIFEST_FORMAT",
    "MANIFEST_SPEC_FORMAT",
    "MODEL_VARIANTS",
    "PARENT_GATE_VERSION",
    "PRODUCER_RELATIVE_PATHS",
    "PROMOTION_REPORT_FORMAT",
    "REQUIRED_STEPS",
    "SCHEMA_VERSION",
    "SOURCES",
    "TOPOLOGY_CONTRACT",
    "TRAINING_GIT_BRANCH",
    "TRAINING_GIT_REF",
    "WINDOWS",
    "SSMaxPerceptionDirectEvidenceError",
    "artifact_reference",
    "build_manifest",
    "build_parent_gate",
    "build_promotion_report",
    "canonical_sha256",
    "load_manifest",
    "load_manifest_spec",
    "manifest_reference",
    "sha256_file",
    "validate_evidence_git_compatibility",
    "validate_manifest",
    "validate_manifest_producer_source",
    "validate_manifest_spec",
    "validate_promotion_report_reference",
    "validate_saved_config",
    "validate_ssmax_perception_direct_parent_gate",
    "write_json_once",
]
