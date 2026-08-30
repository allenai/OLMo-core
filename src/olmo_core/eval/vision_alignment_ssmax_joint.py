"""Immutable, descriptive post-hoc evidence for dense SSMax joint alignment.

This protocol is intentionally separate from the historical s002 joint evaluators.  A manifest
binds one model variant, its approved versioned perception parent, reviewed joint profile, permanent
steps, visual projection, exact matched/wrong pairings, native replay population, and full
checkpoint byte identities.  Reports are reconstructed from raw receipts and are never promotion
gates: only load, frozen-surface, data-cursor, and non-finite failures are hard collapse signals.
All task-quality and retention measurements remain descriptive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from olmo_core.data.multimodal.ssmax_single_response import (
    SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM,
    SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT,
    SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION,
    ssmax_single_response_projection_contract,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
)
from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.eval import vision_alignment_ssmax_perception as perception
from olmo_core.eval import vision_alignment_ssmax_perception_direct as direct_perception
from olmo_core.eval import (
    vision_alignment_ssmax_perception_exploratory as exploratory_perception,
)
from olmo_core.eval import (
    vision_alignment_ssmax_perception_exploratory_waiver as exploratory_waiver_perception,
)
from olmo_core.eval.matched_wrong_image import (
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)
from olmo_core.eval.ssmax_attention_diagnostics import (
    compare_ssmax_attention_reports,
    validate_ssmax_attention_report,
)
from olmo_core.train.callbacks import (
    SSMaxHealthLedgerError,
    extract_ssmax_health_ledgers,
)

MANIFEST_SPEC_FORMAT = "vision_alignment_ssmax_joint_manifest_spec"
MANIFEST_FORMAT = "vision_alignment_ssmax_joint_manifest"
EVALUATION_RECEIPT_FORMAT = "vision_alignment_ssmax_joint_evaluation_receipt"
HEALTH_RECEIPT_FORMAT = "vision_alignment_ssmax_joint_health_receipt"
TRAJECTORY_REPORT_FORMAT = "vision_alignment_ssmax_joint_trajectory_report"
PAIR_COMPARISON_FORMAT = "vision_alignment_ssmax_joint_pair_comparison"
MANIFEST_SPEC_VERSION = 2
SCHEMA_VERSION = 2
LEGACY_PARENT_GATE_VERSION = 5
PARENT_GATE_VERSION = 6
DIRECT_PARENT_GATE_VERSION = direct_perception.PARENT_GATE_VERSION
EXPLORATORY_PARENT_GATE_VERSION = exploratory_perception.PARENT_GATE_VERSION
EXPLORATORY_WAIVER_PARENT_GATE_VERSION = exploratory_waiver_perception.PARENT_GATE_VERSION
SUPPORTED_PARENT_GATE_VERSIONS = frozenset(
    {
        LEGACY_PARENT_GATE_VERSION,
        PARENT_GATE_VERSION,
        DIRECT_PARENT_GATE_VERSION,
        EXPLORATORY_PARENT_GATE_VERSION,
        EXPLORATORY_WAIVER_PARENT_GATE_VERSION,
    }
)

REQUIRED_STEPS = (0, 4000, 8000, 12000, 16000)
VISUAL_SOURCES = tuple(JOINT_VISUAL_SOURCE_NAMES)
NATIVE_SOURCE = "native_text_replay"
TRAIN_SOURCES = (*VISUAL_SOURCES, NATIVE_SOURCE)
WINDOWS = ("first_1", "first_8", "first_32", "all")
MODEL_VARIANTS = bridge.MODEL_VARIANTS
IMAGE_TOKEN_ROWS = bridge.IMAGE_TOKEN_ROWS
VISUAL_DATASET_EXAMPLES = 512
VISUAL_EXAMPLES_PER_SOURCE = 496
PAIRING_SEED = 6198
SINGLE_RESPONSE_PROJECTION_SEED = 95818
ELIGIBLE_VISUAL_ROWS = {
    "audited_alignment": 511,
    "cosyn_point": 510,
    "count_numeric": 512,
    "ocr_document": 512,
    "pixmo_caption": 508,
    "pixmo_points_basic": 510,
    "pixmo_points_high_frequency": 509,
    "pixmo_transcript": 508,
}
BUILDER_REPO_RELATIVE_PATH = "src/olmo_core/eval/vision_alignment_ssmax_joint.py"
EVALUATOR_REPO_RELATIVE_PATH = "src/scripts/eval/vision_alignment_ssmax_joint.py"
STRUCTURAL_CONFIG_PROTOCOL = "joint-resume-structural-config-canonical-json-v1"
STRUCTURAL_CONFIG_IGNORED_PATHS = ("launch.name", "launch.git.ref")
CROSS_ARM_SCHEDULE_CLASSIFICATION = {
    "schedule": "asymmetric_code_transition",
    "causal_interpretation": "confounded",
    "decision_scope": "descriptive_only",
}
TRAINING_RESUME_SCHEDULES = {
    "ssmax_head_qknorm": {
        "0": {
            "config_sha256": "18b0ce331150767c71ead409c116f75cb3d1fa4c163365b80b06a43b51eb8d4e",
            "launch_name": "vision-ssmax-head-qknorm-1p4b-cx8-joint-v1-cf5067c3",
            "git_ref": "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
        },
        "4000": {
            "config_sha256": "18b0ce331150767c71ead409c116f75cb3d1fa4c163365b80b06a43b51eb8d4e",
            "launch_name": "vision-ssmax-head-qknorm-1p4b-cx8-joint-v1-cf5067c3",
            "git_ref": "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
        },
        "8000": {
            "config_sha256": "3a7b54a6f6e313f9d9288a1e177c21dfca434bc42f60cc48d7e7a0c30031ac14",
            "launch_name": "vision-ssmax-head-qknorm-1p4b-cx8-joint-v1-2ccabac1",
            "git_ref": "e53e8ee6db022366790e5a4ef3a94c62ab50928f",
        },
        "12000": {
            "config_sha256": "3cdc518325fd9c02cddb52c4803253b7525dd12dab11ebd5ff151f98d9f8b4b3",
            "launch_name": "vision-ssmax-head-qknorm-1p4b-cx8-joint-v1-cadcf2ef",
            "git_ref": "26eebf08c91caf407bdae31fb989c02682946a3c",
        },
        "16000": {
            "config_sha256": "3cdc518325fd9c02cddb52c4803253b7525dd12dab11ebd5ff151f98d9f8b4b3",
            "launch_name": "vision-ssmax-head-qknorm-1p4b-cx8-joint-v1-cadcf2ef",
            "git_ref": "26eebf08c91caf407bdae31fb989c02682946a3c",
        },
    },
    "ssmax_no_qknorm": {
        "0": {
            "config_sha256": "991eb1f631c99d82b791b52f37ebb9d1fe31a35b342aece9b29eaa0edc0edbe2",
            "launch_name": "vision-ssmax-no-qknorm-1p4b-cx8-joint-v1-06daa088",
            "git_ref": "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
        },
        "4000": {
            "config_sha256": "a11bc8da1699c012972b00cb6943e71668ac82e4762090695531bd99dfe5eaf7",
            "launch_name": "vision-ssmax-no-qknorm-1p4b-cx8-joint-v1-89a4e0b1",
            "git_ref": "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
        },
        "8000": {
            "config_sha256": "a11bc8da1699c012972b00cb6943e71668ac82e4762090695531bd99dfe5eaf7",
            "launch_name": "vision-ssmax-no-qknorm-1p4b-cx8-joint-v1-89a4e0b1",
            "git_ref": "7cc97a77cbfe9a625531653ac6ec64382a56e56c",
        },
        "12000": {
            "config_sha256": "17493e32d50252fc314d6eb7c1cb14576bb45ab37b8280428eafe6af6cc0afba",
            "launch_name": "vision-ssmax-no-qknorm-1p4b-cx8-joint-v1-043301f2",
            "git_ref": "26eebf08c91caf407bdae31fb989c02682946a3c",
        },
        "16000": {
            "config_sha256": "17493e32d50252fc314d6eb7c1cb14576bb45ab37b8280428eafe6af6cc0afba",
            "launch_name": "vision-ssmax-no-qknorm-1p4b-cx8-joint-v1-043301f2",
            "git_ref": "26eebf08c91caf407bdae31fb989c02682946a3c",
        },
    },
}

MIN_EXAMPLES_PER_SOURCE = 8
MIN_NATIVE_HOLDOUT_EXAMPLES = 1

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
_REPOSITORY_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256", "repo_relative_path"})
_MANIFEST_REF_FIELDS = frozenset({"path", "sha256", "content_sha256"})
_CHECKPOINT_FIELDS = frozenset(
    {
        "path",
        "global_step",
        "config_sha256",
        "marker_sha256",
        "dcp_metadata_sha256",
        "state_file_count",
        "state_file_inventory_sha256",
        "trainer_state_count",
        "trainer_state_inventory_sha256",
        "identity_sha256",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "format",
        "version",
        "run_id",
        "model_variant",
        "run_name",
        "checkpoint_root",
        "checkpoint_config_sha256s",
        "evidence_git",
        "training_profile",
        "recipe",
        "perception_parent_gate",
        "joint_visual_projection",
        "source_audit",
        "attention_probe",
        "pairing_paths",
        "evaluation",
        "topology",
        "policy",
        "companion_protocols",
    }
)
_EVIDENCE_GIT_FIELDS = frozenset({"repo", "repo_url", "ref"})
_EVIDENCE_GIT_REF_PLACEHOLDER = "<FILL_WITH_CLEAN_EVIDENCE_COMMIT_SHA>"
_MANIFEST_SPEC_REFERENCE_FIELDS = frozenset({"path", "sha256", "semantic_sha256"})
_BUILDER_SOURCE_FIELDS = frozenset({"repo_relative_path", "sha256", "git_ref"})
_EVALUATOR_SOURCE_FIELDS = frozenset({"repo_relative_path", "sha256", "git_ref"})
_TRAINING_RESUME_LINEAGE_FIELDS = frozenset({"cross_arm_schedule", "structural_config", "steps"})
_CROSS_ARM_SCHEDULE_FIELDS = frozenset({"schedule", "causal_interpretation", "decision_scope"})
_STRUCTURAL_CONFIG_FIELDS = frozenset({"protocol", "ignored_paths", "sha256"})
_RESUME_STEP_FIELDS = frozenset({"config_sha256", "launch_name", "training_git"})
_TRAINING_GIT_FIELDS = frozenset({"repo", "repo_url", "branch", "ref"})
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "version",
        "created_at",
        "run_id",
        "model_variant",
        "run_name",
        "git",
        "manifest_spec",
        "manifest_builder",
        "training_resume_lineage",
        "recipe",
        "training_profile",
        "perception_parent",
        "joint_visual_projection",
        "source_audit",
        "source_audit_fingerprint",
        "single_response_projection_contract",
        "attention_probe",
        "pairings",
        "native_replay",
        "evaluation",
        "topology",
        "policy",
        "loss_mass_targets",
        "companion_protocols",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "checkpoints",
        "run_contract_sha256",
        "content_sha256",
    }
)
_PARENT_FIELDS = frozenset(
    {
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "gate",
        "gate_semantic_sha256",
    }
)
_NATIVE_FIELDS = frozenset(
    {
        "train_config_sha256",
        "holdout_config_sha256",
        "train_manifest",
        "holdout_manifest",
        "verification_receipt",
        "train_fingerprint",
        "holdout_fingerprint",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "visual_sources",
        "steps",
        "windows",
        "examples_per_source",
        "eligible_rows_per_source",
        "native_holdout_examples",
        "pairing_seed",
        "single_response_projection_seed",
        "rank_batch_instances",
    }
)
_TOPOLOGY_FIELDS = frozenset({"world_size", "num_nodes", "gpus_per_node", "data_parallel"})
_POLICY_FIELDS = frozenset(
    {
        "decision_scope",
        "maximum_data_errors",
        "maximum_optimizer_guard_skips",
        "maximum_nonfinite_losses",
        "maximum_nonfinite_gradients",
        "native_text_ce_max_relative_increase",
        "native_text_bootstrap_samples",
        "native_text_bootstrap_seed",
        "require_exact_frozen_surfaces",
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
        "native_holdout",
        "pairings",
        "results",
        "attention_diagnostics",
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
_ROW_FIELDS = frozenset(
    {
        "pairing_position",
        "recipient_index",
        "donor_index",
        "response_tokens",
        "correct_ce",
        "wrong_ce",
        "ce_gap_wrong_minus_correct",
    }
)
_SURFACE_FIELDS = frozenset(
    {
        "protocol",
        "tensor_count",
        "reference_inventory_sha256",
        "candidate_inventory_sha256",
        "mismatch_count",
    }
)
_FULL_MODEL_FIELDS = frozenset({"protocol", "tensor_count", "inventory_sha256"})
_STATE_FIELDS = frozenset({"full_model", "frozen_lexical_input_rows", "frozen_output_projection"})
_NATIVE_RESULT_FIELDS = frozenset(
    {
        "examples",
        "tokens",
        "loss_weight",
        "summed_ce",
        "ce",
        "ppl",
        "filtered_examples",
        "dataset_order_sha256",
        "row_provenance_sha256",
        "native_identity_sha256",
        "per_example",
    }
)
_NATIVE_ROW_FIELDS = frozenset(
    {"position", "tokens", "mask_weight", "loss_weight", "summed_ce", "filtered"}
)
_RANK_STATE_FIELDS = frozenset(
    {
        "rank",
        "global_step",
        "batches_processed",
        "data_loader_state_sha256",
        "trainer_state_sha256",
        "health_ledger",
    }
)
_SOURCE_HEALTH_FIELDS = frozenset(
    {
        "examples",
        "tokens",
        "positive_tokens",
        "loss_weight",
        "active_loss_weight",
        "target_loss_mass",
    }
)
_RUN_COUNTER_FIELDS = frozenset(
    {
        "data_errors",
        "optimizer_guard_skips",
        "nonfinite_losses",
        "nonfinite_gradients",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "decision_scope",
        "cross_arm_schedule",
        "created_at",
        "manifest",
        "run_id",
        "model_variant",
        "receipts",
        "hard_invariants",
        "trajectory",
        "paired_visual_rows",
        "attention_reports",
        "attention_trajectory",
        "companion_protocols",
        "content_sha256",
    }
)


class SSMaxJointEvidenceError(ValueError):
    """Raised when SSMax joint evidence violates its immutable contract."""


def canonical_sha256(value: Any) -> str:
    """Return a deterministic semantic SHA-256 for finite JSON-compatible data."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the raw-byte SHA-256 of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    """Load finite JSON while rejecting duplicate object keys."""

    def hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SSMaxJointEvidenceError(f"JSON object repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SSMaxJointEvidenceError(f"JSON contains non-finite value {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SSMaxJointEvidenceError(f"Could not load JSON {path}: {error}") from error


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an immutable JSON artifact without replacing an existing file."""

    try:
        bridge.write_json_once(path, payload)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error


def _exact(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxJointEvidenceError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise SSMaxJointEvidenceError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxJointEvidenceError(f"{name} must be an object")
    return value


def _sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise SSMaxJointEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SSMaxJointEvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SSMaxJointEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise SSMaxJointEvidenceError(f"{name} must be finite and >= {minimum}")
    return result


def _timestamp(value: Any, *, name: str) -> None:
    if not isinstance(value, str):
        raise SSMaxJointEvidenceError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SSMaxJointEvidenceError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SSMaxJointEvidenceError(f"{name} must include a timezone")


def artifact_reference(path: Path) -> dict[str, str]:
    """Return an absolute raw-byte reference to a regular file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SSMaxJointEvidenceError(f"Required artifact is absent: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _builder_source_path() -> Path:
    return Path(__file__).resolve()


def _repository_root() -> Path:
    source = _builder_source_path()
    root = source
    for _ in Path(BUILDER_REPO_RELATIVE_PATH).parts:
        root = root.parent
    if source != (root / BUILDER_REPO_RELATIVE_PATH).resolve():
        raise SSMaxJointEvidenceError("joint manifest builder is outside its canonical repository")
    return root


def evaluator_source_reference(source_path: Path, *, git_ref: str) -> dict[str, str]:
    """Create a portable receipt reference to the canonical joint evaluator source."""

    source = source_path.expanduser().resolve()
    expected = (_repository_root() / EVALUATOR_REPO_RELATIVE_PATH).resolve()
    if source != expected or not source.is_file():
        raise SSMaxJointEvidenceError("joint evaluator source path is non-canonical")
    if re.fullmatch(r"[0-9a-f]{40}", git_ref) is None:
        raise SSMaxJointEvidenceError("joint evaluator git ref is invalid")
    return {
        "repo_relative_path": EVALUATOR_REPO_RELATIVE_PATH,
        "sha256": sha256_file(source),
        "git_ref": git_ref,
    }


def _validate_evaluator_source_reference(
    value: Any, *, evidence_git: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Prove a portable evaluator ref against this clean evidence checkout and Git blob."""

    reference = _exact(value, _EVALUATOR_SOURCE_FIELDS, name="joint evaluator")
    evidence_identity = _git_identity(evidence_git)
    if (
        reference["repo_relative_path"] != EVALUATOR_REPO_RELATIVE_PATH
        or reference["git_ref"] != evidence_identity["ref"]
    ):
        raise SSMaxJointEvidenceError("joint evaluator path or git ref differs")
    expected_sha256 = _sha(reference["sha256"], name="joint evaluator SHA")
    repository_root = _repository_root()
    source = (repository_root / EVALUATOR_REPO_RELATIVE_PATH).resolve()
    if not source.is_file() or not source.is_relative_to(repository_root):
        raise SSMaxJointEvidenceError("joint evaluator is absent from the evidence checkout")
    try:
        bridge._validate_repository_checkout(
            evidence_identity,
            repository_root=repository_root,
        )
        blob = bridge._git_blob_bytes(
            evidence_identity,
            repository_root=repository_root,
            repo_relative_path=EVALUATOR_REPO_RELATIVE_PATH,
            name="joint evaluator",
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error
    if (
        sha256_file(source) != expected_sha256
        or hashlib.sha256(blob).hexdigest() != expected_sha256
    ):
        raise SSMaxJointEvidenceError(
            "joint evaluator differs from the executing checkout or evidence Git blob"
        )
    return reference


def repository_artifact_reference(path: Path) -> dict[str, str]:
    """Pin build-time bytes and their portable path inside the builder repository."""

    reference = artifact_reference(path)
    resolved = Path(reference["path"])
    root = _repository_root()
    if not resolved.is_relative_to(root):
        raise SSMaxJointEvidenceError("repository artifact is outside the builder repository")
    return {**reference, "repo_relative_path": resolved.relative_to(root).as_posix()}


def resolve_repository_artifact(value: Any, *, name: str) -> Path:
    """Resolve a raw-pinned repository artifact under this executing module's checkout."""

    reference = _exact(value, _REPOSITORY_ARTIFACT_REF_FIELDS, name=f"{name} reference")
    _sha(reference["sha256"], name=f"{name} SHA")
    relative = reference["repo_relative_path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise SSMaxJointEvidenceError(f"{name} repository-relative path is invalid")
    path = (_repository_root() / relative).resolve()
    if not path.is_relative_to(_repository_root()) or not path.is_file():
        raise SSMaxJointEvidenceError(f"{name} is absent from the executing repository")
    if sha256_file(path) != reference["sha256"]:
        raise SSMaxJointEvidenceError(f"{name} executing-repository bytes differ")
    return path


def _repository_artifact_shape(
    value: Any, *, name: str, verify_stored_bytes: bool
) -> Mapping[str, Any]:
    reference = _exact(value, _REPOSITORY_ARTIFACT_REF_FIELDS, name=f"{name} reference")
    if verify_stored_bytes:
        validate_artifact_reference(
            {"path": reference["path"], "sha256": reference["sha256"]}, name=name
        )
    else:
        _artifact_shape({"path": reference["path"], "sha256": reference["sha256"]}, name=name)
    relative = reference["repo_relative_path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise SSMaxJointEvidenceError(f"{name} repository-relative path is invalid")
    return reference


def _manifest_spec_reference(path: Path, spec: Mapping[str, Any]) -> dict[str, str]:
    reference = artifact_reference(path)
    live_spec = _validate_spec(load_json(Path(reference["path"])))
    expected_semantic_sha256 = canonical_sha256(spec)
    if canonical_sha256(live_spec) != expected_semantic_sha256:
        raise SSMaxJointEvidenceError("joint manifest spec source differs semantically")
    return {**reference, "semantic_sha256": expected_semantic_sha256}


def validate_artifact_reference(value: Any, *, name: str) -> Path:
    """Re-open an artifact and prove its raw-byte identity."""

    reference = _exact(value, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    path_value = reference["path"]
    if not isinstance(path_value, str) or not path_value:
        raise SSMaxJointEvidenceError(f"{name} path must be non-empty")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != _sha(reference["sha256"], name=f"{name} SHA"):
        raise SSMaxJointEvidenceError(f"{name} differs from its immutable reference")
    return path


def _artifact_shape(value: Any, *, name: str) -> Mapping[str, Any]:
    reference = _exact(value, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise SSMaxJointEvidenceError(f"{name} path must be non-empty")
    _sha(reference["sha256"], name=f"{name} SHA")
    return reference


def manifest_reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return raw and semantic identity for a finalized joint manifest."""

    result = artifact_reference(path)
    result["content_sha256"] = _sha(manifest.get("content_sha256"), name="manifest content SHA")
    return result


def _git_identity(value: Any) -> Mapping[str, str]:
    git = _exact(value, frozenset({"repo", "repo_url", "ref"}), name="git identity")
    if any(not isinstance(git[field], str) or not git[field] for field in ("repo", "repo_url")):
        raise SSMaxJointEvidenceError("git repo and repo_url must be non-empty")
    if not isinstance(git["ref"], str) or re.fullmatch(r"[0-9a-f]{40}", git["ref"]) is None:
        raise SSMaxJointEvidenceError("git ref must be a 40-character commit SHA")
    return {field: str(git[field]) for field in ("repo", "repo_url", "ref")}


def _validate_evaluation(value: Any) -> Mapping[str, Any]:
    evaluation = _exact(value, _EVALUATION_FIELDS, name="evaluation contract")
    if (
        evaluation["visual_sources"] != list(VISUAL_SOURCES)
        or evaluation["steps"] != list(REQUIRED_STEPS)
        or evaluation["windows"] != list(WINDOWS)
    ):
        raise SSMaxJointEvidenceError("joint source/step/window contract differs")
    examples = _integer(
        evaluation["examples_per_source"],
        name="examples per source",
        minimum=MIN_EXAMPLES_PER_SOURCE,
    )
    eligible = _exact(
        evaluation["eligible_rows_per_source"],
        frozenset(VISUAL_SOURCES),
        name="eligible visual rows",
    )
    if examples != VISUAL_EXAMPLES_PER_SOURCE or dict(eligible) != ELIGIBLE_VISUAL_ROWS:
        raise SSMaxJointEvidenceError(
            "joint evidence requires the fixed largest common 16-way-divisible visual "
            "population and its live eligibility counts"
        )
    _integer(
        evaluation["native_holdout_examples"],
        name="native holdout examples",
        minimum=MIN_NATIVE_HOLDOUT_EXAMPLES,
    )
    pairing_seed = _integer(evaluation["pairing_seed"], name="pairing seed")
    projection_seed = _integer(
        evaluation["single_response_projection_seed"],
        name="single-response projection seed",
    )
    if pairing_seed != PAIRING_SEED or projection_seed != SINGLE_RESPONSE_PROJECTION_SEED:
        raise SSMaxJointEvidenceError(
            "joint pairing and single-response projection seeds differ from their "
            "independent fixed contracts"
        )
    rank_batch = _integer(
        evaluation["rank_batch_instances"], name="rank batch instances", minimum=1
    )
    if rank_batch != 1:
        raise SSMaxJointEvidenceError("joint evidence requires one unpacked response row per rank")
    if examples % rank_batch:
        raise SSMaxJointEvidenceError("visual examples must divide rank batch instances")
    return evaluation


def _validate_topology(value: Any, evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    topology = _exact(value, _TOPOLOGY_FIELDS, name="topology")
    world = _integer(topology["world_size"], name="world size", minimum=1)
    nodes = _integer(topology["num_nodes"], name="node count", minimum=1)
    gpus = _integer(topology["gpus_per_node"], name="GPUs per node", minimum=1)
    if world != nodes * gpus or topology["data_parallel"] != "hsdp":
        raise SSMaxJointEvidenceError("joint SSMax evidence requires a complete HSDP world")
    global_instances = world * int(evaluation["rank_batch_instances"])
    if int(evaluation["examples_per_source"]) % global_instances:
        raise SSMaxJointEvidenceError(
            "visual examples must divide the global evaluation instance batch"
        )
    if int(evaluation["native_holdout_examples"]) % global_instances:
        raise SSMaxJointEvidenceError(
            "native holdout examples must divide the global evaluation instance batch"
        )
    return topology


def _validate_policy(value: Any) -> Mapping[str, Any]:
    policy = _exact(value, _POLICY_FIELDS, name="joint evidence policy")
    expected = {
        "decision_scope": "descriptive_non_promotion",
        "maximum_data_errors": 0,
        "maximum_optimizer_guard_skips": 0,
        "maximum_nonfinite_losses": 0,
        "maximum_nonfinite_gradients": 0,
        "native_text_ce_max_relative_increase": 0.02,
        "native_text_bootstrap_samples": 10_000,
        "native_text_bootstrap_seed": 6198,
        "require_exact_frozen_surfaces": True,
    }
    if dict(policy) != expected:
        raise SSMaxJointEvidenceError("joint hard-invariant policy differs")
    return policy


def _validate_spec(value: Any) -> Mapping[str, Any]:
    spec = _exact(value, _SPEC_FIELDS, name="joint manifest spec")
    if spec["format"] != MANIFEST_SPEC_FORMAT or spec["version"] != MANIFEST_SPEC_VERSION:
        raise SSMaxJointEvidenceError("joint manifest spec is incompatible")
    for field in ("run_id", "run_name", "checkpoint_root", "training_profile", "recipe"):
        if not isinstance(spec[field], str) or not spec[field]:
            raise SSMaxJointEvidenceError(f"spec {field} must be non-empty")
    if spec["model_variant"] not in MODEL_VARIANTS:
        raise SSMaxJointEvidenceError("unsupported SSMax model variant")
    config_sha256s = _exact(
        spec["checkpoint_config_sha256s"],
        frozenset(str(step) for step in REQUIRED_STEPS),
        name="checkpoint config SHA-256 pins",
    )
    for step in REQUIRED_STEPS:
        _sha(config_sha256s[str(step)], name=f"step{step} checkpoint config SHA")
    expected_config_sha256s = {
        step: row["config_sha256"]
        for step, row in TRAINING_RESUME_SCHEDULES[str(spec["model_variant"])].items()
    }
    if dict(config_sha256s) != expected_config_sha256s:
        raise SSMaxJointEvidenceError(
            "checkpoint config SHA-256 pins differ from the reviewed resume schedule"
        )
    evidence_git = _exact(spec["evidence_git"], _EVIDENCE_GIT_FIELDS, name="evidence git")
    if evidence_git["repo"] != "allenai/OLMo-core" or evidence_git["repo_url"] != (
        "https://github.com/allenai/OLMo-core"
    ):
        raise SSMaxJointEvidenceError("evidence git names a different repository")
    evidence_ref = evidence_git["ref"]
    if not isinstance(evidence_ref, str) or (
        evidence_ref != _EVIDENCE_GIT_REF_PLACEHOLDER
        and re.fullmatch(r"[0-9a-f]{40}", evidence_ref) is None
    ):
        raise SSMaxJointEvidenceError(
            "evidence git ref must be the template placeholder or a 40-character commit SHA"
        )
    for field in (
        "perception_parent_gate",
        "joint_visual_projection",
        "source_audit",
        "attention_probe",
    ):
        if not isinstance(spec[field], str) or not spec[field]:
            raise SSMaxJointEvidenceError(f"spec {field} must be a path")
    pairings = _exact(spec["pairing_paths"], frozenset(VISUAL_SOURCES), name="pairing paths")
    if any(
        not isinstance(pairings[source], str) or not pairings[source] for source in VISUAL_SOURCES
    ):
        raise SSMaxJointEvidenceError("every pairing path must be non-empty")
    evaluation = _validate_evaluation(spec["evaluation"])
    _validate_topology(spec["topology"], evaluation)
    _validate_policy(spec["policy"])
    companions = _mapping(spec["companion_protocols"], name="companion protocols")
    if set(companions) != {"downstream_fast_pair"} or any(
        not isinstance(companions[name], str) or not companions[name] for name in companions
    ):
        raise SSMaxJointEvidenceError("companion protocol paths differ")
    return spec


def load_manifest_spec(path: Path) -> Mapping[str, Any]:
    """Load a checked-in, non-runnable SSMax joint manifest specification."""

    return _validate_spec(load_json(path))


def _checkpoint_reference(
    value: Any, *, step: int, verify_live: bool, workers: int
) -> Mapping[str, Any]:
    reference = _exact(value, _CHECKPOINT_FIELDS, name=f"step{step} checkpoint")
    try:
        return bridge.validate_checkpoint_reference(
            reference,
            expected_step=step,
            verify_live=verify_live,
            workers=workers,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error


def _config_artifact_reference(raw: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    manifest_path = raw.get("manifest_path")
    fingerprint = raw.get("expected_fingerprint")
    receipt_path = raw.get("verification_receipt_path")
    receipt_sha = raw.get("expected_verification_receipt_sha256")
    if (
        not isinstance(manifest_path, str)
        or not isinstance(fingerprint, str)
        or not isinstance(receipt_path, str)
        or not isinstance(receipt_sha, str)
    ):
        raise SSMaxJointEvidenceError(f"{name} lacks its pinned v3 replay artifacts")
    manifest = artifact_reference(Path(manifest_path))
    receipt = artifact_reference(Path(receipt_path))
    if receipt["sha256"] != _sha(receipt_sha, name=f"{name} verification receipt SHA"):
        raise SSMaxJointEvidenceError(f"{name} verification receipt differs")
    return {
        "config_sha256": canonical_sha256(dict(raw)),
        "manifest": manifest,
        "fingerprint": _sha(fingerprint, name=f"{name} fingerprint"),
        "verification_receipt": receipt,
    }


def _validate_saved_config(
    config: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    profile: Mapping[str, str],
) -> dict[str, Any]:
    for field, expected in (
        ("model_variant", spec["model_variant"]),
        ("phase", "joint"),
        ("required_run_name", spec["run_name"]),
    ):
        if config.get(field) != expected:
            raise SSMaxJointEvidenceError(f"saved config {field} differs")
    reviewed = config.get("reviewed_profile_path")
    if not isinstance(reviewed, str) or Path(reviewed).expanduser().resolve() != Path(
        profile["path"]
    ):
        raise SSMaxJointEvidenceError("saved config names a different reviewed profile")
    if config.get("reviewed_profile_sha256") != profile["sha256"]:
        raise SSMaxJointEvidenceError("saved profile SHA-256 differs")
    data = _mapping(config.get("data"), name="joint data")
    if data.get("pack_sequences") is not False or data.get("sequence_length") != 8192:
        raise SSMaxJointEvidenceError("SSMax GDN joint data must be unpacked at length 8192")
    if data.get("allow_unpinned_synthetic_smoke") is not False:
        raise SSMaxJointEvidenceError("production joint config enables synthetic bypass")
    projection = _mapping(
        data.get("ssmax_single_response_projection"),
        name="joint single-response projection",
    )
    if projection.get("seed") != spec["evaluation"]["single_response_projection_seed"]:
        raise SSMaxJointEvidenceError("joint single-response projection seed differs")
    try:
        projection_contract = ssmax_single_response_projection_contract(
            seed=int(projection["seed"]),
            loss_token_weighting=str(data.get("loss_token_weighting")),
            format=str(projection.get("format")),
            version=_integer(
                projection.get("version"),
                name="joint single-response projection version",
                minimum=1,
            ),
            algorithm=str(projection.get("algorithm")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SSMaxJointEvidenceError(
            f"joint single-response projection contract differs: {error}"
        ) from error
    trainer = _mapping(config.get("trainer"), name="joint trainer")
    duration = _mapping(trainer.get("max_duration"), name="joint duration")
    if duration.get("value") != 16000 or duration.get("unit") != "steps":
        raise SSMaxJointEvidenceError("joint duration must be exactly 16000 steps")
    checkpointer = _mapping(
        _mapping(trainer.get("callbacks"), name="joint callbacks").get("checkpointer"),
        name="joint checkpointer",
    )
    fixed = set(checkpointer.get("fixed_steps") or ())
    interval = checkpointer.get("save_interval")
    if isinstance(interval, int) and interval > 0:
        fixed.update(range(interval, 16001, interval))
    if checkpointer.get("pre_train_checkpoint") is not True or not set(REQUIRED_STEPS[1:]) <= fixed:
        raise SSMaxJointEvidenceError("joint checkpointer omits a required permanent step")
    launch = _mapping(config.get("launch"), name="joint launch")
    topology = _mapping(spec["topology"], name="spec topology")
    if (
        launch.get("num_nodes") != topology["num_nodes"]
        or launch.get("num_gpus") != topology["gpus_per_node"]
        or launch.get("workspace") != "ai2/scaling-ladders"
        or launch.get("priority") != "urgent"
        or launch.get("min_runtime") not in ("8h", 28800, 28_800.0)
    ):
        raise SSMaxJointEvidenceError("joint launch topology/workspace/priority/runtime differs")
    initialization = _mapping(config.get("initialization"), name="joint initialization")
    if initialization.get("expected_parent_phase") != "perception":
        raise SSMaxJointEvidenceError("joint config does not require a perception parent")
    metadata = _mapping(config.get("vision_alignment"), name="joint metadata")
    if (
        metadata.get("model_variant") != spec["model_variant"]
        or metadata.get("phase") != "joint"
        or metadata.get("lineage_id") != spec["run_name"]
    ):
        raise SSMaxJointEvidenceError("joint saved lineage metadata differs")
    train_module = _mapping(config.get("train_module"), name="joint train module")
    freeze = train_module.get("freeze_params")
    if freeze != ["lm.lm_head.w_out.weight"]:
        raise SSMaxJointEvidenceError("joint output projection freeze contract differs")
    rows = train_module.get("train_embedding_rows")
    if sorted(rows or ()) != sorted(IMAGE_TOKEN_ROWS):
        raise SSMaxJointEvidenceError("joint input embedding row-mask contract differs")
    targets = _mapping(train_module.get("source_loss_mass_targets"), name="loss-mass targets")
    if set(targets) != set(TRAIN_SOURCES):
        raise SSMaxJointEvidenceError("joint loss-mass source set differs")
    normalized_targets = {
        source: _finite(targets[source], name=f"{source} loss-mass target", minimum=0.0)
        for source in TRAIN_SOURCES
    }
    if any(value <= 0 for value in normalized_targets.values()) or not math.isclose(
        sum(normalized_targets.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SSMaxJointEvidenceError("joint loss-mass targets must be positive and sum to one")
    git = _mapping(launch.get("git"), name="joint saved git")
    return {
        "initialization": dict(initialization),
        "metadata": dict(metadata),
        "data": dict(data),
        "evaluation": dict(_mapping(config.get("evaluation"), name="joint evaluation config")),
        "loss_mass_targets": normalized_targets,
        "single_response_projection_contract": projection_contract,
        "git": {field: git.get(field) for field in ("repo", "repo_url", "branch", "ref")},
    }


def _resume_structural_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only separately recorded Gantry identities for a structural comparison."""

    normalized = copy.deepcopy(dict(config))
    launch = dict(_mapping(normalized.get("launch"), name="joint resume launch"))
    if "name" not in launch:
        raise SSMaxJointEvidenceError("joint resume config omits launch.name")
    git = dict(_mapping(launch.get("git"), name="joint resume git"))
    if "ref" not in git:
        raise SSMaxJointEvidenceError("joint resume config omits launch.git.ref")
    del launch["name"]
    del git["ref"]
    launch["git"] = git
    normalized["launch"] = launch
    return normalized


def _validate_resume_config_set(
    configs: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    profile: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every retained step to exact source lineage and one structural config."""

    step_keys = frozenset(str(step) for step in REQUIRED_STEPS)
    configs = _exact(configs, step_keys, name="joint checkpoint configs")
    checkpoints = _exact(checkpoints, step_keys, name="joint checkpoints")
    expected_sha256s = _exact(
        spec["checkpoint_config_sha256s"],
        step_keys,
        name="checkpoint config SHA-256 pins",
    )
    summaries: dict[str, dict[str, Any]] = {}
    structural_sha256: str | None = None
    resume_steps = {}
    expected_schedule = TRAINING_RESUME_SCHEDULES[str(spec["model_variant"])]
    for step in REQUIRED_STEPS:
        key = str(step)
        expected_sha256 = _sha(expected_sha256s[key], name=f"step{step} checkpoint config SHA")
        if checkpoints[key].get("config_sha256") != expected_sha256:
            raise SSMaxJointEvidenceError(
                f"step{step} checkpoint config differs from its reviewed raw SHA-256 pin"
            )
        config = _mapping(configs[key], name=f"step{step} joint saved config")
        summaries[key] = _validate_saved_config(config, spec=spec, profile=profile)
        launch = _mapping(config.get("launch"), name=f"step{step} joint launch")
        launch_name = launch.get("name")
        raw_git = _mapping(launch.get("git"), name=f"step{step} joint training git")
        training_git = {field: raw_git.get(field) for field in _TRAINING_GIT_FIELDS}
        if any(
            not isinstance(training_git[field], str) or not training_git[field]
            for field in training_git
        ):
            raise SSMaxJointEvidenceError(f"step{step} joint training git identity is incomplete")
        if re.fullmatch(r"[0-9a-f]{40}", training_git["ref"]) is None:
            raise SSMaxJointEvidenceError(f"step{step} joint training git ref is invalid")
        expected = expected_schedule[key]
        if (
            launch_name != expected["launch_name"]
            or training_git["ref"] != expected["git_ref"]
            or expected_sha256 != expected["config_sha256"]
        ):
            raise SSMaxJointEvidenceError(
                f"step{step} launch identity differs from the reviewed resume schedule"
            )
        current_structural_sha256 = canonical_sha256(_resume_structural_config(config))
        if structural_sha256 is None:
            structural_sha256 = current_structural_sha256
        elif current_structural_sha256 != structural_sha256:
            raise SSMaxJointEvidenceError(
                "joint checkpoint configs differ structurally outside launch.name and "
                "launch.git.ref"
            )
        resume_steps[key] = {
            "config_sha256": expected_sha256,
            "launch_name": launch_name,
            "training_git": training_git,
        }
    assert structural_sha256 is not None
    lineage = {
        "cross_arm_schedule": dict(CROSS_ARM_SCHEDULE_CLASSIFICATION),
        "structural_config": {
            "protocol": STRUCTURAL_CONFIG_PROTOCOL,
            "ignored_paths": list(STRUCTURAL_CONFIG_IGNORED_PATHS),
            "sha256": structural_sha256,
        },
        "steps": resume_steps,
    }
    return summaries[str(REQUIRED_STEPS[-1])], lineage


def _validate_evidence_git_checkout(
    value: Any, *, recipe_path: Path, profile_path: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Bind evidence production to the separately pinned clean evidence checkout."""

    evidence_git = dict(_git_identity(value))
    repository_root = _repository_root()
    recipe = recipe_path.expanduser().resolve()
    profile = profile_path.expanduser().resolve()
    for path, name in ((recipe, "recipe"), (profile, "reviewed profile")):
        if not path.is_file() or not path.is_relative_to(repository_root):
            raise SSMaxJointEvidenceError(
                f"{name} must be a file inside the manifest builder repository"
            )
    recipe_relative = recipe.relative_to(repository_root).as_posix()
    profile_relative = profile.relative_to(repository_root).as_posix()
    if recipe_relative != "src/scripts/train/Vision-Alignment.py":
        raise SSMaxJointEvidenceError("manifest recipe is not the canonical training script")
    try:
        bridge._validate_repository_checkout(evidence_git, repository_root=repository_root)
        for path, relative, name in (
            (recipe, recipe_relative, "recipe"),
            (profile, profile_relative, "reviewed profile"),
        ):
            blob = bridge._git_blob_bytes(
                evidence_git,
                repository_root=repository_root,
                repo_relative_path=relative,
                name=name,
            )
            if hashlib.sha256(blob).hexdigest() != sha256_file(path):
                raise SSMaxJointEvidenceError(f"live {name} differs from the evidence Git blob")
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error
    builder_source = _builder_source_path()
    if builder_source != (repository_root / BUILDER_REPO_RELATIVE_PATH).resolve():
        raise SSMaxJointEvidenceError("running joint manifest builder source is non-canonical")
    try:
        builder_blob = bridge._git_blob_bytes(
            evidence_git,
            repository_root=repository_root,
            repo_relative_path=BUILDER_REPO_RELATIVE_PATH,
            name="joint manifest builder",
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error
    builder_sha256 = sha256_file(builder_source)
    if hashlib.sha256(builder_blob).hexdigest() != builder_sha256:
        raise SSMaxJointEvidenceError("joint manifest builder differs from the evidence Git blob")
    return evidence_git, {
        "repo_relative_path": BUILDER_REPO_RELATIVE_PATH,
        "sha256": builder_sha256,
        "git_ref": evidence_git["ref"],
    }


def _consume_issued_v9_perception_parent_gate(
    gate: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_parent_metadata: Mapping[str, Any],
    verify_live_checkpoint: bool,
    hash_workers: int,
) -> dict[str, Any]:
    """Consume one immutable, raw-SHA-pinned v9 authorization for joint evidence."""

    if not verify_live_checkpoint:
        raise SSMaxJointEvidenceError(
            "issued v9 perception authorization requires live checkpoint verification"
        )
    value = _exact(
        gate,
        exploratory_waiver_perception._GATE_FIELDS,
        name="issued v9 perception parent gate",
    )
    checkpoint = expected_checkpoint.expanduser().resolve()
    if checkpoint.name != "step4000":
        raise SSMaxJointEvidenceError("issued v9 perception parent must be step4000")
    recipe_version = expected_parent_metadata.get("recipe_version")
    formatter_version = expected_parent_metadata.get("formatter_version")
    if type(recipe_version) is not int or not isinstance(formatter_version, str):
        raise SSMaxJointEvidenceError("perception parent recipe identity is malformed")
    data_contract_sha256 = _sha(
        expected_parent_metadata.get("data_contract_sha256"),
        name="perception parent data contract",
    )
    trainable_contract_sha256 = _sha(
        expected_parent_metadata.get("trainable_contract_sha256"),
        name="perception parent trainable contract",
    )
    expected_values = {
        "format": "vision_alignment_parent_gate",
        "version": EXPLORATORY_WAIVER_PARENT_GATE_VERSION,
        "status": "approved",
        "scope": exploratory_waiver_perception.GATE_SCOPE,
        "recipe_version": recipe_version,
        "formatter_version": formatter_version,
        "phase": "perception",
        "model_variant": expected_model_variant,
        "lineage_kind": direct_perception.LINEAGE_KIND,
        "checkpoint_config_sha256": expected_checkpoint_config_sha256,
        "data_contract_sha256": data_contract_sha256,
        "trainable_contract_sha256": trainable_contract_sha256,
        "global_step": 4000,
        "evidence_report_status": exploratory_waiver_perception.REPORT_STATUS,
        "promotion_decision": False,
        "winner_selection": False,
    }
    for field, expected in expected_values.items():
        if type(value[field]) is not type(expected) or value[field] != expected:
            raise SSMaxJointEvidenceError(f"issued v9 perception parent gate {field} differs")
    if (
        not isinstance(value["run_id"], str)
        or not value["run_id"]
        or not isinstance(value["checkpoint"], str)
        or Path(value["checkpoint"]).expanduser().resolve() != checkpoint
    ):
        raise SSMaxJointEvidenceError("issued v9 perception parent identity differs")
    gate_identity_sha256 = _sha(
        value["checkpoint_identity_sha256"],
        name="issued v9 perception checkpoint identity",
    )
    try:
        candidate = _mapping(
            bridge.checkpoint_identity(checkpoint, workers=hash_workers),
            name="live perception checkpoint identity",
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(f"live perception checkpoint failed: {error}") from error
    live_identity_sha256 = _sha(
        candidate.get("identity_sha256"), name="live perception checkpoint identity"
    )
    if (
        candidate.get("path") != str(checkpoint)
        or type(candidate.get("global_step")) is not int
        or candidate.get("global_step") != 4000
        or candidate.get("config_sha256") != expected_checkpoint_config_sha256
        or live_identity_sha256 != gate_identity_sha256
    ):
        raise SSMaxJointEvidenceError(
            "live perception checkpoint identity differs from the issued v9 gate"
        )
    return {"candidate": dict(candidate)}


def _validate_perception_parent(
    config_summary: Mapping[str, Any],
    *,
    gate_reference: Mapping[str, str],
    model_variant: str,
    verify_live_checkpoint: bool,
    hash_workers: int = 8,
) -> dict[str, Any]:
    initialization = _mapping(config_summary["initialization"], name="joint initialization")
    parent = Path(str(initialization.get("checkpoint"))).expanduser().resolve()
    config_path = parent / "config.json"
    expected_config_sha = _sha(
        initialization.get("parent_config_sha256"), name="perception parent config SHA"
    )
    if not config_path.is_file() or sha256_file(config_path) != expected_config_sha:
        raise SSMaxJointEvidenceError("perception parent config differs from its saved pin")
    gate_path = validate_artifact_reference(gate_reference, name="perception parent gate")
    if gate_path != Path(
        str(initialization.get("parent_gate_path"))
    ).expanduser().resolve() or gate_reference["sha256"] != _sha(
        initialization.get("parent_gate_sha256"), name="saved parent gate SHA"
    ):
        raise SSMaxJointEvidenceError("joint initialization names a different perception gate")
    parent_config = _mapping(load_json(config_path), name="perception parent config")
    parent_metadata = _mapping(
        parent_config.get("vision_alignment"), name="perception parent metadata"
    )
    if (
        parent_config.get("model_variant") != model_variant
        or parent_config.get("phase") != "perception"
        or parent_config.get("perception_trainability_arm") != perception.TREATMENT_ARM
        or parent_metadata.get("model_variant") != model_variant
        or parent_metadata.get("phase") != "perception"
    ):
        raise SSMaxJointEvidenceError("joint parent is not the selected perception treatment")
    gate = _mapping(load_json(gate_path), name="perception parent gate")
    gate_version = gate.get("version")
    if type(gate_version) is not int or gate_version not in SUPPORTED_PARENT_GATE_VERSIONS:
        raise SSMaxJointEvidenceError(
            "perception parent gate version must be exactly integer 5, 6, 7, 8, or 9"
        )
    data_contract_sha256 = _sha(
        parent_metadata.get("data_contract_sha256"), name="perception data contract"
    )
    trainable_contract_sha256 = _sha(
        parent_metadata.get("trainable_contract_sha256"),
        name="perception trainable contract",
    )
    validator_kwargs: dict[str, Any] = {
        "expected_checkpoint": parent,
        "expected_checkpoint_config_sha256": expected_config_sha,
        "expected_model_variant": model_variant,
        "expected_data_contract_sha256": data_contract_sha256,
        "expected_trainable_contract_sha256": trainable_contract_sha256,
        "verify_live_checkpoint": verify_live_checkpoint,
    }
    if gate_version in (LEGACY_PARENT_GATE_VERSION, PARENT_GATE_VERSION):
        try:
            result = perception.validate_ssmax_perception_parent_gate(
                gate,
                **validator_kwargs,
            )
        except perception.SSMaxPerceptionEvidenceError as error:
            raise SSMaxJointEvidenceError(
                f"paired perception parent gate failed: {error}"
            ) from error
    elif gate_version == DIRECT_PARENT_GATE_VERSION:
        try:
            result = direct_perception.validate_ssmax_perception_direct_parent_gate(
                gate,
                **validator_kwargs,
            )
        except direct_perception.SSMaxPerceptionDirectEvidenceError as error:
            raise SSMaxJointEvidenceError(
                f"direct perception parent gate failed: {error}"
            ) from error
    elif gate_version == EXPLORATORY_PARENT_GATE_VERSION:
        try:
            result = exploratory_perception.validate_ssmax_perception_exploratory_parent_gate(
                gate,
                **validator_kwargs,
            )
        except exploratory_perception.SSMaxPerceptionExploratoryEvidenceError as error:
            raise SSMaxJointEvidenceError(
                f"exploratory perception parent gate failed: {error}"
            ) from error
    else:
        result = _consume_issued_v9_perception_parent_gate(
            gate,
            expected_checkpoint=parent,
            expected_checkpoint_config_sha256=expected_config_sha,
            expected_model_variant=model_variant,
            expected_parent_metadata=parent_metadata,
            verify_live_checkpoint=verify_live_checkpoint,
            hash_workers=hash_workers,
        )
    candidate = _mapping(result.get("candidate"), name="perception candidate")
    return {
        "checkpoint": str(parent),
        "checkpoint_config_sha256": expected_config_sha,
        "checkpoint_identity_sha256": _sha(
            candidate.get("identity_sha256"), name="perception checkpoint identity"
        ),
        "data_contract_sha256": data_contract_sha256,
        "trainable_contract_sha256": trainable_contract_sha256,
        "gate": dict(gate_reference),
        "gate_semantic_sha256": canonical_sha256(gate),
    }


def build_manifest(
    spec: Mapping[str, Any], *, spec_path: Path, created_at: str, hash_workers: int = 8
) -> dict[str, Any]:
    """Finalize one arm manifest after all five permanent joint checkpoints exist."""

    spec = _validate_spec(spec)
    spec_reference = _manifest_spec_reference(spec_path, spec)
    _timestamp(created_at, name="manifest created_at")
    if hash_workers <= 0:
        raise SSMaxJointEvidenceError("hash_workers must be positive")
    root = Path(str(spec["checkpoint_root"])).expanduser().resolve()
    missing = [
        str(root / f"step{step}") for step in REQUIRED_STEPS if not (root / f"step{step}").is_dir()
    ]
    if missing:
        raise SSMaxJointEvidenceError(
            "joint manifest requires all permanent checkpoints; missing " + ", ".join(missing)
        )
    checkpoints = {
        str(step): bridge.checkpoint_identity(root / f"step{step}", workers=hash_workers)
        for step in REQUIRED_STEPS
    }
    if {item["trainer_state_count"] for item in checkpoints.values()} != {
        spec["topology"]["world_size"]
    }:
        raise SSMaxJointEvidenceError("joint trainer-state topology differs")
    profile = repository_artifact_reference(Path(str(spec["training_profile"])))
    recipe = repository_artifact_reference(Path(str(spec["recipe"])))
    configs = {
        str(step): _mapping(
            load_json(root / f"step{step}" / "config.json"),
            name=f"step{step} joint saved config",
        )
        for step in REQUIRED_STEPS
    }
    summary, training_resume_lineage = _validate_resume_config_set(
        configs,
        checkpoints,
        spec=spec,
        profile=profile,
    )
    git, manifest_builder = _validate_evidence_git_checkout(
        spec["evidence_git"],
        recipe_path=Path(recipe["path"]),
        profile_path=Path(profile["path"]),
    )
    training_git = _mapping(summary["git"], name="final training git")
    if any(training_git.get(field) != git[field] for field in ("repo", "repo_url")):
        raise SSMaxJointEvidenceError("training and evidence git name different repositories")
    gate_ref = artifact_reference(Path(str(spec["perception_parent_gate"])))
    perception_parent = _validate_perception_parent(
        summary,
        gate_reference=gate_ref,
        model_variant=str(spec["model_variant"]),
        verify_live_checkpoint=True,
        hash_workers=hash_workers,
    )
    data = _mapping(summary["data"], name="joint data")
    projection = artifact_reference(Path(str(spec["joint_visual_projection"])))
    source_audit = artifact_reference(Path(str(spec["source_audit"])))
    if (
        Path(str(data.get("joint_visual_projection_path"))).expanduser().resolve()
        != Path(projection["path"])
        or data.get("joint_visual_projection_sha256") != projection["sha256"]
        or Path(str(data.get("source_audit_path"))).expanduser().resolve()
        != Path(source_audit["path"])
    ):
        raise SSMaxJointEvidenceError("saved joint data artifacts differ from the spec")
    audit = _mapping(load_json(Path(source_audit["path"])), name="joint source audit")
    unsigned_audit = dict(audit)
    fingerprint = _sha(unsigned_audit.pop("fingerprint", None), name="source audit fingerprint")
    if (
        canonical_sha256(unsigned_audit) != fingerprint
        or data.get("source_audit_fingerprint") != fingerprint
    ):
        raise SSMaxJointEvidenceError("joint source-audit fingerprint differs")
    train_raw = _mapping(data.get("native_text_replay"), name="native train replay")
    holdout_raw = _mapping(
        _mapping(summary["evaluation"], name="joint evaluation").get("native_text_holdout"),
        name="native holdout replay",
    )
    train_native = _config_artifact_reference(train_raw, name="native train replay")
    holdout_native = _config_artifact_reference(holdout_raw, name="native holdout replay")
    if train_native["verification_receipt"] != holdout_native["verification_receipt"]:
        raise SSMaxJointEvidenceError(
            "native train and holdout do not share one verification receipt"
        )
    native_replay = {
        "train_config_sha256": train_native["config_sha256"],
        "holdout_config_sha256": holdout_native["config_sha256"],
        "train_manifest": train_native["manifest"],
        "holdout_manifest": holdout_native["manifest"],
        "verification_receipt": train_native["verification_receipt"],
        "train_fingerprint": train_native["fingerprint"],
        "holdout_fingerprint": holdout_native["fingerprint"],
    }
    pairings = {}
    for source in VISUAL_SOURCES:
        reference = artifact_reference(Path(str(spec["pairing_paths"][source])))
        payload = _mapping(load_json(Path(reference["path"])), name=f"{source} pairing")
        try:
            validate_matched_wrong_image_pairing(
                payload,
                recipient_count=int(spec["evaluation"]["examples_per_source"]),
                seed=int(spec["evaluation"]["pairing_seed"]),
                epoch=0,
            )
        except ValueError as error:
            raise SSMaxJointEvidenceError(f"{source} pairing is invalid: {error}") from error
        coverage = _mapping(payload.get("coverage"), name=f"{source} pairing coverage")
        if (
            payload.get("dataset_size") != VISUAL_DATASET_EXAMPLES
            or coverage.get("dataset_count") != VISUAL_DATASET_EXAMPLES
            or coverage.get("eligible_count") != ELIGIBLE_VISUAL_ROWS[source]
        ):
            raise SSMaxJointEvidenceError(f"{source} pairing live eligibility differs")
        if matched_wrong_image_pairing_sha256(payload) != reference["sha256"]:
            raise SSMaxJointEvidenceError(f"{source} pairing canonical SHA differs")
        pairings[source] = reference
    attention_probe = artifact_reference(Path(str(spec["attention_probe"])))
    companions = {
        "downstream_fast_pair": artifact_reference(
            Path(str(spec["companion_protocols"]["downstream_fast_pair"]))
        )
    }
    metadata = _mapping(summary["metadata"], name="joint metadata")
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": SCHEMA_VERSION,
        "created_at": created_at,
        "run_id": spec["run_id"],
        "model_variant": spec["model_variant"],
        "run_name": spec["run_name"],
        "git": dict(git),
        "manifest_spec": spec_reference,
        "manifest_builder": manifest_builder,
        "training_resume_lineage": training_resume_lineage,
        "recipe": recipe,
        "training_profile": profile,
        "perception_parent": perception_parent,
        "joint_visual_projection": projection,
        "source_audit": source_audit,
        "source_audit_fingerprint": fingerprint,
        "single_response_projection_contract": summary["single_response_projection_contract"],
        "attention_probe": attention_probe,
        "pairings": pairings,
        "native_replay": native_replay,
        "evaluation": dict(spec["evaluation"]),
        "topology": dict(spec["topology"]),
        "policy": dict(spec["policy"]),
        "loss_mass_targets": summary["loss_mass_targets"],
        "companion_protocols": companions,
        "data_contract_sha256": _sha(
            metadata.get("data_contract_sha256"), name="joint data contract"
        ),
        "trainable_contract_sha256": _sha(
            metadata.get("trainable_contract_sha256"), name="joint trainable contract"
        ),
        "checkpoints": checkpoints,
    }
    manifest["run_contract_sha256"] = canonical_sha256(
        {key: item for key, item in manifest.items() if key not in {"created_at", "checkpoints"}}
    )
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest, verify_live=True, hash_workers=hash_workers)
    return manifest


def _native_shape(value: Any, *, verify_live: bool) -> Mapping[str, Any]:
    native = _exact(value, _NATIVE_FIELDS, name="native replay")
    for field in (
        "train_config_sha256",
        "holdout_config_sha256",
        "train_fingerprint",
        "holdout_fingerprint",
    ):
        _sha(native[field], name=f"native {field}")
    for field in ("train_manifest", "holdout_manifest", "verification_receipt"):
        if verify_live:
            validate_artifact_reference(native[field], name=f"native {field}")
        else:
            _artifact_shape(native[field], name=f"native {field}")
    return native


def _manifest_spec_reference_shape(
    value: Any, *, verify_live: bool
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    reference = _exact(value, _MANIFEST_SPEC_REFERENCE_FIELDS, name="manifest spec reference")
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise SSMaxJointEvidenceError("manifest spec path must be non-empty")
    _sha(reference["sha256"], name="manifest spec raw SHA")
    semantic_sha256 = _sha(reference["semantic_sha256"], name="manifest spec semantic SHA")
    if verify_live:
        path = validate_artifact_reference(
            {"path": reference["path"], "sha256": reference["sha256"]},
            name="manifest spec",
        )
        live_spec = _validate_spec(load_json(path))
        if canonical_sha256(live_spec) != semantic_sha256:
            raise SSMaxJointEvidenceError("live manifest spec semantic SHA differs")
        return reference, live_spec
    return reference, None


def _builder_source_shape(value: Any, *, evidence_git: Mapping[str, str]) -> Mapping[str, Any]:
    source = _exact(value, _BUILDER_SOURCE_FIELDS, name="manifest builder source")
    if source["repo_relative_path"] != BUILDER_REPO_RELATIVE_PATH:
        raise SSMaxJointEvidenceError("manifest builder source path is non-canonical")
    _sha(source["sha256"], name="manifest builder source SHA")
    if source["git_ref"] != evidence_git["ref"]:
        raise SSMaxJointEvidenceError("manifest builder source Git ref differs")
    return source


def _training_resume_lineage_shape(value: Any, *, model_variant: str) -> Mapping[str, Any]:
    lineage = _exact(value, _TRAINING_RESUME_LINEAGE_FIELDS, name="training resume lineage")
    classification = _exact(
        lineage["cross_arm_schedule"],
        _CROSS_ARM_SCHEDULE_FIELDS,
        name="cross-arm resume schedule",
    )
    if canonical_sha256(classification) != canonical_sha256(CROSS_ARM_SCHEDULE_CLASSIFICATION):
        raise SSMaxJointEvidenceError("cross-arm resume schedule classification differs")
    structural = _exact(
        lineage["structural_config"],
        _STRUCTURAL_CONFIG_FIELDS,
        name="resume structural config",
    )
    if structural["protocol"] != STRUCTURAL_CONFIG_PROTOCOL or structural["ignored_paths"] != list(
        STRUCTURAL_CONFIG_IGNORED_PATHS
    ):
        raise SSMaxJointEvidenceError("resume structural config contract differs")
    _sha(structural["sha256"], name="resume structural config SHA")
    steps = _exact(
        lineage["steps"],
        frozenset(str(step) for step in REQUIRED_STEPS),
        name="training resume steps",
    )
    expected_schedule = TRAINING_RESUME_SCHEDULES[model_variant]
    for step in REQUIRED_STEPS:
        key = str(step)
        row = _exact(steps[key], _RESUME_STEP_FIELDS, name=f"step{step} resume lineage")
        expected = expected_schedule[key]
        if (
            row["config_sha256"] != expected["config_sha256"]
            or row["launch_name"] != expected["launch_name"]
        ):
            raise SSMaxJointEvidenceError(f"step{step} resume lineage schedule differs")
        _sha(row["config_sha256"], name=f"step{step} resume config SHA")
        training_git = _exact(
            row["training_git"], _TRAINING_GIT_FIELDS, name=f"step{step} training git"
        )
        if (
            training_git["repo"] != "allenai/OLMo-core"
            or training_git["repo_url"] != "https://github.com/allenai/OLMo-core"
            or training_git["branch"] != "rustin/vision-ssmax-molmofication"
            or training_git["ref"] != expected["git_ref"]
        ):
            raise SSMaxJointEvidenceError(f"step{step} training Git lineage differs")
    return lineage


def _validate_finalized_manifest_against_spec(
    manifest: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    """Rebind every spec-derived finalized-manifest field to the pinned live spec."""

    for field in ("run_id", "model_variant", "run_name"):
        if manifest[field] != spec[field]:
            raise SSMaxJointEvidenceError(f"finalized manifest {field} differs from its spec")
    if canonical_sha256(manifest["git"]) != canonical_sha256(spec["evidence_git"]):
        raise SSMaxJointEvidenceError("finalized manifest evidence Git differs from its spec")
    repository_root = _repository_root()

    def spec_repository_path(value: Any) -> Path:
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else repository_root / path).resolve()

    for manifest_field, spec_field in (
        ("recipe", "recipe"),
        ("training_profile", "training_profile"),
    ):
        if Path(str(manifest[manifest_field]["path"])).resolve() != spec_repository_path(
            spec[spec_field]
        ):
            raise SSMaxJointEvidenceError(
                f"finalized manifest {manifest_field} path differs from its spec"
            )
    for manifest_field, spec_field in (
        ("joint_visual_projection", "joint_visual_projection"),
        ("source_audit", "source_audit"),
        ("attention_probe", "attention_probe"),
    ):
        if (
            Path(str(manifest[manifest_field]["path"])).resolve()
            != Path(str(spec[spec_field])).expanduser().resolve()
        ):
            raise SSMaxJointEvidenceError(
                f"finalized manifest {manifest_field} path differs from its spec"
            )
    if (
        Path(str(manifest["perception_parent"]["gate"]["path"])).resolve()
        != Path(str(spec["perception_parent_gate"])).expanduser().resolve()
    ):
        raise SSMaxJointEvidenceError("finalized perception gate path differs from its spec")
    for source in VISUAL_SOURCES:
        if (
            Path(str(manifest["pairings"][source]["path"])).resolve()
            != Path(str(spec["pairing_paths"][source])).expanduser().resolve()
        ):
            raise SSMaxJointEvidenceError(f"finalized {source} pairing path differs from its spec")
    companion = manifest["companion_protocols"]["downstream_fast_pair"]
    if Path(str(companion["path"])).resolve() != spec_repository_path(
        spec["companion_protocols"]["downstream_fast_pair"]
    ):
        raise SSMaxJointEvidenceError("finalized companion protocol path differs from its spec")
    for field in ("evaluation", "topology", "policy"):
        if canonical_sha256(manifest[field]) != canonical_sha256(spec[field]):
            raise SSMaxJointEvidenceError(f"finalized manifest {field} differs from its spec")
    checkpoint_root = Path(str(spec["checkpoint_root"])).expanduser().resolve()
    for step in REQUIRED_STEPS:
        key = str(step)
        if (
            Path(str(manifest["checkpoints"][key]["path"])).resolve()
            != checkpoint_root / f"step{step}"
            or manifest["checkpoints"][key]["config_sha256"]
            != spec["checkpoint_config_sha256s"][key]
        ):
            raise SSMaxJointEvidenceError(f"finalized step{step} checkpoint differs from its spec")


def validate_manifest(
    value: Any, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Validate a finalized SSMax joint manifest and optionally all live byte pins."""

    manifest = _exact(value, _MANIFEST_FIELDS, name="joint manifest")
    if manifest["format"] != MANIFEST_FORMAT or manifest["version"] != SCHEMA_VERSION:
        raise SSMaxJointEvidenceError("joint manifest is incompatible")
    _timestamp(manifest["created_at"], name="manifest created_at")
    if manifest["model_variant"] not in MODEL_VARIANTS:
        raise SSMaxJointEvidenceError("joint manifest model variant is unsupported")
    for field in ("run_id", "run_name"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise SSMaxJointEvidenceError(f"manifest {field} must be non-empty")
    evidence_git = dict(_git_identity(manifest["git"]))
    _, live_spec = _manifest_spec_reference_shape(
        manifest["manifest_spec"], verify_live=verify_live
    )
    builder_source = _builder_source_shape(manifest["manifest_builder"], evidence_git=evidence_git)
    resume_lineage = _training_resume_lineage_shape(
        manifest["training_resume_lineage"], model_variant=str(manifest["model_variant"])
    )
    for field in ("recipe", "training_profile"):
        _repository_artifact_shape(manifest[field], name=field, verify_stored_bytes=verify_live)
    runtime_recipe = resolve_repository_artifact(manifest["recipe"], name="recipe")
    runtime_profile = resolve_repository_artifact(
        manifest["training_profile"], name="training profile"
    )
    for field in (
        "joint_visual_projection",
        "source_audit",
        "attention_probe",
    ):
        if verify_live:
            validate_artifact_reference(manifest[field], name=field)
        else:
            _artifact_shape(manifest[field], name=field)
    live_git, live_builder_source = _validate_evidence_git_checkout(
        evidence_git,
        recipe_path=runtime_recipe,
        profile_path=runtime_profile,
    )
    if live_git != evidence_git or live_builder_source != builder_source:
        raise SSMaxJointEvidenceError("live manifest builder identity differs")
    _sha(manifest["source_audit_fingerprint"], name="source audit fingerprint")
    parent = _exact(manifest["perception_parent"], _PARENT_FIELDS, name="perception parent")
    if not isinstance(parent["checkpoint"], str) or not parent["checkpoint"]:
        raise SSMaxJointEvidenceError("perception parent checkpoint must be non-empty")
    for field in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "gate_semantic_sha256",
    ):
        _sha(parent[field], name=f"perception parent {field}")
    if verify_live:
        validate_artifact_reference(parent["gate"], name="perception parent gate")
    else:
        _artifact_shape(parent["gate"], name="perception parent gate")
    evaluation = _validate_evaluation(manifest["evaluation"])
    projection_contract = _mapping(
        manifest["single_response_projection_contract"],
        name="single-response projection contract",
    )
    expected_projection_contract = ssmax_single_response_projection_contract(
        seed=int(evaluation["single_response_projection_seed"]),
        loss_token_weighting="root_subsegments_root_tokens",
        format=SSMAX_SINGLE_RESPONSE_PROJECTION_FORMAT,
        version=SSMAX_SINGLE_RESPONSE_PROJECTION_VERSION,
        algorithm=SSMAX_SINGLE_RESPONSE_PROJECTION_ALGORITHM,
    )
    if dict(projection_contract) != expected_projection_contract:
        raise SSMaxJointEvidenceError("single-response projection contract differs")
    topology = _validate_topology(manifest["topology"], evaluation)
    _validate_policy(manifest["policy"])
    pairings = _exact(manifest["pairings"], frozenset(VISUAL_SOURCES), name="joint pairings")
    for source in VISUAL_SOURCES:
        if verify_live:
            validate_artifact_reference(pairings[source], name=f"{source} pairing")
        else:
            _artifact_shape(pairings[source], name=f"{source} pairing")
    _native_shape(manifest["native_replay"], verify_live=verify_live)
    targets = _exact(
        manifest["loss_mass_targets"], frozenset(TRAIN_SOURCES), name="loss-mass targets"
    )
    target_values = [
        _finite(targets[source], name=f"{source} target", minimum=0.0) for source in TRAIN_SOURCES
    ]
    if any(value <= 0 for value in target_values) or not math.isclose(
        sum(target_values), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SSMaxJointEvidenceError("loss-mass targets are invalid")
    companions = _exact(
        manifest["companion_protocols"],
        frozenset({"downstream_fast_pair"}),
        name="companion protocols",
    )
    for name in companions:
        if verify_live:
            validate_artifact_reference(companions[name], name=f"{name} companion")
        else:
            _artifact_shape(companions[name], name=f"{name} companion")
    for field in ("data_contract_sha256", "trainable_contract_sha256", "run_contract_sha256"):
        _sha(manifest[field], name=field)
    checkpoints = _exact(
        manifest["checkpoints"], frozenset(str(step) for step in REQUIRED_STEPS), name="checkpoints"
    )
    for step in REQUIRED_STEPS:
        reference = _checkpoint_reference(
            checkpoints[str(step)], step=step, verify_live=verify_live, workers=hash_workers
        )
        if reference["trainer_state_count"] != topology["world_size"]:
            raise SSMaxJointEvidenceError(f"step{step} trainer-state topology differs")
        if reference["config_sha256"] != resume_lineage["steps"][str(step)]["config_sha256"]:
            raise SSMaxJointEvidenceError(f"step{step} checkpoint and resume config pins differ")
    if live_spec is not None:
        _validate_finalized_manifest_against_spec(manifest, live_spec)
    expected_run_contract = canonical_sha256(
        {
            key: item
            for key, item in manifest.items()
            if key not in {"created_at", "checkpoints", "run_contract_sha256", "content_sha256"}
        }
    )
    if expected_run_contract != manifest["run_contract_sha256"]:
        raise SSMaxJointEvidenceError("run contract semantic SHA differs")
    expected_content = canonical_sha256(
        {key: item for key, item in manifest.items() if key != "content_sha256"}
    )
    if expected_content != _sha(manifest["content_sha256"], name="manifest content SHA"):
        raise SSMaxJointEvidenceError("manifest content SHA differs")
    return manifest


def load_manifest(
    path: Path, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Load and validate a finalized joint manifest."""

    return validate_manifest(load_json(path), verify_live=verify_live, hash_workers=hash_workers)


def _validate_manifest_reference(value: Any, manifest: Mapping[str, Any], *, path: Path) -> None:
    reference = _exact(value, _MANIFEST_REF_FIELDS, name="receipt manifest reference")
    if reference["content_sha256"] != manifest["content_sha256"]:
        raise SSMaxJointEvidenceError("receipt names a different manifest semantic identity")
    live = validate_artifact_reference(
        {"path": reference["path"], "sha256": reference["sha256"]}, name="receipt manifest"
    )
    if live != path.expanduser().resolve():
        raise SSMaxJointEvidenceError("receipt names a different manifest path")


def _validate_content(payload: Mapping[str, Any], *, name: str) -> None:
    expected = _sha(payload.get("content_sha256"), name=f"{name} content SHA")
    if (
        canonical_sha256({key: value for key, value in payload.items() if key != "content_sha256"})
        != expected
    ):
        raise SSMaxJointEvidenceError(f"{name} content SHA differs")


def _validate_strict_load(value: Any) -> None:
    try:
        perception._validate_strict_load(value, name="joint strict generic DCP load")
    except perception.SSMaxPerceptionEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error


def _validate_surface(value: Any, *, name: str, step: int) -> Mapping[str, Any]:
    surface = _exact(value, _SURFACE_FIELDS, name=name)
    if surface["protocol"] != "logical-tensor-comparison-sha256-v1":
        raise SSMaxJointEvidenceError(f"{name} protocol differs")
    count = _integer(surface["tensor_count"], name=f"{name} tensor count", minimum=1)
    mismatch = _integer(surface["mismatch_count"], name=f"{name} mismatches")
    if mismatch > count:
        raise SSMaxJointEvidenceError(f"{name} mismatch count exceeds tensor count")
    for field in ("reference_inventory_sha256", "candidate_inventory_sha256"):
        _sha(surface[field], name=f"{name} {field}")
    if step == 0 and (
        mismatch != 0
        or surface["reference_inventory_sha256"] != surface["candidate_inventory_sha256"]
    ):
        raise SSMaxJointEvidenceError(f"step0 {name} is not self-equal")
    return surface


def _validate_state(value: Any, *, step: int) -> Mapping[str, Any]:
    state = _exact(value, _STATE_FIELDS, name=f"step{step} state")
    full = _exact(state["full_model"], _FULL_MODEL_FIELDS, name="full model state")
    if full["protocol"] != "logical-model-tensor-inventory-sha256-v1":
        raise SSMaxJointEvidenceError("full model state protocol differs")
    _integer(full["tensor_count"], name="full model tensor count", minimum=1)
    _sha(full["inventory_sha256"], name="full model inventory SHA")
    _validate_surface(state["frozen_lexical_input_rows"], name="frozen lexical rows", step=step)
    _validate_surface(state["frozen_output_projection"], name="frozen output projection", step=step)
    return state


def _validate_rows(
    value: Any,
    *,
    source: str,
    manifest: Mapping[str, Any],
    pairing: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    count = int(manifest["evaluation"]["examples_per_source"])
    if not isinstance(value, list) or len(value) != count:
        raise SSMaxJointEvidenceError(f"{source} must contain exactly {count} rows")
    pairs = pairing.get("pairs") if pairing is not None else None
    output = []
    for position, raw in enumerate(value):
        row = _exact(raw, _ROW_FIELDS, name=f"{source} row{position}")
        if row["pairing_position"] != position:
            raise SSMaxJointEvidenceError(f"{source} pairing positions differ")
        for field in ("recipient_index", "donor_index"):
            _integer(row[field], name=f"{source} {field}")
        _integer(row["response_tokens"], name=f"{source} response tokens", minimum=1)
        if pairs is not None:
            pair = _mapping(pairs[position], name=f"{source} pair{position}")
            if row["recipient_index"] != pair.get("recipient") or row["donor_index"] != pair.get(
                "donor"
            ):
                raise SSMaxJointEvidenceError(f"{source} receipt differs from fixed pairing")
        groups = {
            field: _exact(row[field], frozenset(WINDOWS), name=f"{source} {field}")
            for field in ("correct_ce", "wrong_ce", "ce_gap_wrong_minus_correct")
        }
        for window in WINDOWS:
            correct = _finite(groups["correct_ce"][window], name="correct CE", minimum=0.0)
            wrong = _finite(groups["wrong_ce"][window], name="wrong CE", minimum=0.0)
            gap = _finite(groups["ce_gap_wrong_minus_correct"][window], name="CE gap")
            if not math.isclose(gap, wrong - correct, rel_tol=0.0, abs_tol=1e-12):
                raise SSMaxJointEvidenceError(f"{source} {window} CE gap is inconsistent")
        output.append(row)
    return output


def _validate_native_result(value: Any, *, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _exact(value, _NATIVE_RESULT_FIELDS, name="native holdout result")
    if result["examples"] != manifest["evaluation"]["native_holdout_examples"]:
        raise SSMaxJointEvidenceError("native holdout example count differs")
    rows = result["per_example"]
    if not isinstance(rows, list) or len(rows) != result["examples"]:
        raise SSMaxJointEvidenceError("native holdout per-example rows are incomplete")
    validated_rows = []
    for position, raw_row in enumerate(rows):
        row = _exact(raw_row, _NATIVE_ROW_FIELDS, name=f"native row{position}")
        if row["position"] != position:
            raise SSMaxJointEvidenceError("native holdout row positions are not contiguous")
        tokens = _integer(row["tokens"], name=f"native row{position} tokens")
        mask_weight = _finite(
            row["mask_weight"], name=f"native row{position} mask weight", minimum=0.0
        )
        row_loss_weight = _finite(
            row["loss_weight"], name=f"native row{position} loss weight", minimum=0.0
        )
        row_summed_ce = _finite(
            row["summed_ce"], name=f"native row{position} summed CE", minimum=0.0
        )
        if type(row["filtered"]) is not bool:
            raise SSMaxJointEvidenceError("native holdout filtered flag must be boolean")
        if row["filtered"] != (tokens == 0):
            raise SSMaxJointEvidenceError("native holdout filtered flag differs from tokens")
        if row["filtered"]:
            if row_loss_weight != 0 or row_summed_ce != 0:
                raise SSMaxJointEvidenceError("filtered native row has nonzero loss mass")
        elif tokens <= 0 or mask_weight <= 0 or row_loss_weight <= 0:
            raise SSMaxJointEvidenceError("active native row has invalid supervision")
        validated_rows.append(row)
    for field in ("tokens", "filtered_examples"):
        _integer(result[field], name=f"native {field}")
    loss_weight = _finite(result["loss_weight"], name="native loss weight", minimum=0.0)
    summed = _finite(result["summed_ce"], name="native summed CE", minimum=0.0)
    ce = _finite(result["ce"], name="native CE", minimum=0.0)
    ppl = _finite(result["ppl"], name="native PPL", minimum=1.0)
    if loss_weight <= 0 or not math.isclose(ce, summed / loss_weight, rel_tol=1e-12, abs_tol=1e-12):
        raise SSMaxJointEvidenceError("native CE aggregation differs")
    if (
        result["tokens"] != sum(int(row["tokens"]) for row in validated_rows)
        or result["filtered_examples"] != sum(bool(row["filtered"]) for row in validated_rows)
        or not math.isclose(
            loss_weight,
            sum(float(row["loss_weight"]) for row in validated_rows),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            summed,
            sum(float(row["summed_ce"]) for row in validated_rows),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise SSMaxJointEvidenceError("native per-example rows do not reconstruct aggregates")
    if not math.isclose(ppl, math.exp(ce), rel_tol=1e-12, abs_tol=1e-12):
        raise SSMaxJointEvidenceError("native PPL differs from CE")
    for field in ("dataset_order_sha256", "row_provenance_sha256", "native_identity_sha256"):
        _sha(result[field], name=f"native {field}")
    return result


def _load_receipt(
    reference: Any,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    step: int,
    expected_format: str,
) -> Mapping[str, Any]:
    path = validate_artifact_reference(reference, name=f"step{step} {expected_format}")
    payload = _mapping(load_json(path), name=f"step{step} receipt")
    fields = (
        _EVALUATION_RECEIPT_FIELDS
        if expected_format == EVALUATION_RECEIPT_FORMAT
        else _HEALTH_RECEIPT_FIELDS
    )
    _exact(payload, fields, name=f"step{step} receipt")
    if (
        payload["format"] != expected_format
        or payload["version"] != SCHEMA_VERSION
        or payload["run_id"] != manifest["run_id"]
        or payload["model_variant"] != manifest["model_variant"]
        or payload["step"] != step
        or payload["checkpoint"] != manifest["checkpoints"][str(step)]
        or payload["status"] not in ("passed", "failed")
    ):
        raise SSMaxJointEvidenceError(f"step{step} receipt identity differs")
    _timestamp(payload["created_at"], name=f"step{step} receipt created_at")
    _validate_manifest_reference(payload["manifest"], manifest, path=manifest_path)
    _validate_content(payload, name=f"step{step} receipt")
    return payload


def _validate_evaluation_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], step: int
) -> dict[str, Any]:
    _validate_strict_load(receipt["strict_generic_dcp_load"])
    state = _validate_state(receipt["state"], step=step)
    native = _validate_native_result(receipt["native_holdout"], manifest=manifest)
    attention = _mapping(receipt["attention_diagnostics"], name="attention diagnostics")
    if (
        attention.get("format") != "ssmax_attention_diagnostics"
        or attention.get("checkpoint") != manifest["checkpoints"][str(step)]
        or attention.get("protocol", {}).get("manifest_sha256")
        != manifest["attention_probe"]["sha256"]
    ):
        raise SSMaxJointEvidenceError("attention diagnostics are not manifest/checkpoint bound")
    try:
        validate_ssmax_attention_report(attention, label=f"joint step{step} attention")
    except ValueError as error:
        raise SSMaxJointEvidenceError(str(error)) from error
    if receipt["pairings"] != manifest["pairings"]:
        raise SSMaxJointEvidenceError("evaluation changes fixed pairings")
    _validate_evaluator_source_reference(
        receipt["evaluator"],
        evidence_git=_mapping(manifest["git"], name="manifest evidence git"),
    )
    results = _exact(receipt["results"], frozenset(VISUAL_SOURCES), name="visual results")
    rows = {}
    for source in VISUAL_SOURCES:
        result = _exact(
            results[source],
            frozenset({"pairing_sha256", "examples", "per_example"}),
            name=f"{source} result",
        )
        if (
            result["pairing_sha256"] != manifest["pairings"][source]["sha256"]
            or result["examples"] != manifest["evaluation"]["examples_per_source"]
        ):
            raise SSMaxJointEvidenceError(f"{source} result identity differs")
        pairing = _mapping(
            load_json(
                validate_artifact_reference(manifest["pairings"][source], name=f"{source} pairing")
            ),
            name=f"{source} pairing",
        )
        rows[source] = _validate_rows(
            result["per_example"], source=source, manifest=manifest, pairing=pairing
        )
    return {"state": state, "native": native, "rows": rows, "attention": attention}


def _jsonable_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SSMaxJointEvidenceError(f"trainer state contains unsupported value {value!r}")


def _validate_health_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], step: int
) -> dict[str, Any]:
    ranks = receipt["rank_states"]
    world = int(manifest["topology"]["world_size"])
    if not isinstance(ranks, list) or len(ranks) != world:
        raise SSMaxJointEvidenceError(f"step{step} health omits trainer ranks")
    checkpoint = Path(str(receipt["checkpoint"]["path"]))
    trainer_paths = sorted(
        checkpoint.joinpath("train").glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    if len(trainer_paths) != world:
        raise SSMaxJointEvidenceError(f"step{step} checkpoint trainer ranks differ")
    trainer_states = []
    for rank, value in enumerate(ranks):
        state = _exact(value, _RANK_STATE_FIELDS, name=f"step{step} rank{rank}")
        if (
            state["rank"] != rank
            or state["global_step"] != step
            or state["batches_processed"] != step
        ):
            raise SSMaxJointEvidenceError(f"step{step} rank{rank} cursor differs")
        for field in ("data_loader_state_sha256", "trainer_state_sha256"):
            _sha(state[field], name=f"rank{rank} {field}")
        trainer_path = trainer_paths[rank]
        if (
            trainer_path.name != f"rank{rank}.pt"
            or sha256_file(trainer_path) != state["trainer_state_sha256"]
        ):
            raise SSMaxJointEvidenceError(f"step{step} rank{rank} trainer bytes differ")
        trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=False)
        if not isinstance(trainer_state, Mapping):
            raise SSMaxJointEvidenceError(f"step{step} rank{rank} trainer state is invalid")
        if (
            canonical_sha256(_jsonable_state(trainer_state.get("data_loader")))
            != state["data_loader_state_sha256"]
        ):
            raise SSMaxJointEvidenceError(f"step{step} rank{rank} data-loader state differs")
        trainer_states.append(trainer_state)
    try:
        ledger_summary = extract_ssmax_health_ledgers(
            trainer_states,
            expected_model_variant=str(manifest["model_variant"]),
            expected_phase="joint",
            expected_run_name=str(manifest["run_name"]),
            expected_step=step,
            expected_world_size=world,
        )
    except SSMaxHealthLedgerError as error:
        raise SSMaxJointEvidenceError(
            f"step{step} checkpoint health ledgers are invalid: {error}"
        ) from error
    for rank, state in enumerate(ranks):
        if state["health_ledger"] != ledger_summary["rank_ledgers"][rank]:
            raise SSMaxJointEvidenceError(f"step{step} rank{rank} health ledger differs")
    sources = _exact(receipt["sources"], frozenset(TRAIN_SOURCES), name="health sources")
    for source in TRAIN_SOURCES:
        values = _exact(sources[source], _SOURCE_HEALTH_FIELDS, name=f"{source} health")
        for field in ("examples", "tokens", "positive_tokens"):
            _integer(values[field], name=f"{source} {field}")
        for field in ("loss_weight", "active_loss_weight"):
            _finite(values[field], name=f"{source} {field}", minimum=0.0)
        if values["target_loss_mass"] != manifest["loss_mass_targets"][source]:
            raise SSMaxJointEvidenceError(f"{source} target loss mass differs")
    counters = _exact(receipt["run_counters"], _RUN_COUNTER_FIELDS, name="run counters")
    for field in counters:
        _integer(counters[field], name=field)
    if dict(counters) != ledger_summary["counters"]:
        raise SSMaxJointEvidenceError(f"step{step} health counters differ from checkpoint ledger")
    evidence = _exact(
        receipt["evidence"], frozenset({"recipe", "producer"}), name="health evidence"
    )
    for name in evidence:
        validate_artifact_reference(evidence[name], name=f"health {name}")
    return {"rank_states": ranks, "sources": sources, "run_counters": counters}


def _receipt_steps(value: Any, *, name: str) -> dict[int, Mapping[str, Any]]:
    mapping = _mapping(value, name=name)
    output: dict[int, Mapping[str, Any]] = {}
    for raw_step, reference in mapping.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise SSMaxJointEvidenceError(f"{name} has invalid step {raw_step!r}") from error
        if step in output:
            raise SSMaxJointEvidenceError(f"{name} repeats step{step}")
        output[step] = _exact(reference, _ARTIFACT_REF_FIELDS, name=f"{name} step{step}")
    if set(output) != set(REQUIRED_STEPS):
        raise SSMaxJointEvidenceError(f"{name} must contain exactly {list(REQUIRED_STEPS)}")
    return output


def _mean(rows: Sequence[Mapping[str, Any]], field: str, window: str) -> float:
    return float(np.mean([float(row[field][window]) for row in rows], dtype=np.float64))


def _native_text_noninferiority(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    maximum_relative_increase: float,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Return a row-paired bootstrap interval for native-text CE relative change."""

    reference_rows = reference["per_example"]
    candidate_rows = candidate["per_example"]
    if not isinstance(reference_rows, list) or not isinstance(candidate_rows, list):
        raise SSMaxJointEvidenceError("native noninferiority requires per-example rows")
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise SSMaxJointEvidenceError("native noninferiority row counts differ")
    invariant_fields = ("position", "tokens", "mask_weight", "loss_weight", "filtered")
    if any(
        any(left[field] != right[field] for field in invariant_fields)
        for left, right in zip(reference_rows, candidate_rows, strict=True)
    ):
        raise SSMaxJointEvidenceError("native noninferiority rows are not exactly paired")
    active = [index for index, row in enumerate(reference_rows) if not row["filtered"]]
    if len(active) < 2:
        raise SSMaxJointEvidenceError("native noninferiority needs at least two active rows")
    weights = np.asarray(
        [float(reference_rows[index]["loss_weight"]) for index in active], dtype=np.float64
    )
    reference_summed = np.asarray(
        [float(reference_rows[index]["summed_ce"]) for index in active], dtype=np.float64
    )
    candidate_summed = np.asarray(
        [float(candidate_rows[index]["summed_ce"]) for index in active], dtype=np.float64
    )
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(reference_summed).all()
        or not np.isfinite(candidate_summed).all()
        or np.any(weights <= 0)
    ):
        raise SSMaxJointEvidenceError("native noninferiority rows contain invalid loss mass")
    reference_ce = float(reference_summed.sum() / weights.sum())
    candidate_ce = float(candidate_summed.sum() / weights.sum())
    if reference_ce <= 0:
        raise SSMaxJointEvidenceError("native noninferiority reference CE must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    relative_changes = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size = 256
    for start in range(0, bootstrap_samples, chunk_size):
        count = min(chunk_size, bootstrap_samples - start)
        indices = rng.integers(0, len(active), size=(count, len(active)))
        sampled_weights = weights[indices].sum(axis=1)
        sampled_reference = reference_summed[indices].sum(axis=1) / sampled_weights
        sampled_candidate = candidate_summed[indices].sum(axis=1) / sampled_weights
        if np.any(sampled_reference <= 0):
            raise SSMaxJointEvidenceError("native bootstrap produced zero reference CE")
        relative_changes[start : start + count] = sampled_candidate / sampled_reference - 1.0
    lower, upper = np.quantile(relative_changes, (0.025, 0.975), method="linear")
    observed = candidate_ce / reference_ce - 1.0
    return {
        "method": "row_paired_weighted_ce_percentile_bootstrap_v1",
        "confidence": 0.95,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "active_examples": len(active),
        "reference_ce": reference_ce,
        "candidate_ce": candidate_ce,
        "observed_relative_change": observed,
        "relative_change_ci": {"low": float(lower), "high": float(upper)},
        "maximum_relative_increase": maximum_relative_increase,
        "passed": float(upper) <= maximum_relative_increase,
    }


def build_trajectory_report(
    *,
    manifest_path: Path,
    evaluation_receipts: Mapping[int | str, Mapping[str, str]],
    health_receipts: Mapping[int | str, Mapping[str, str]],
    created_at: str,
    verify_live_manifest: bool = True,
) -> dict[str, Any]:
    """Rebuild a descriptive trajectory report from every raw receipt."""

    _timestamp(created_at, name="trajectory report created_at")
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_manifest)
    eval_refs = _receipt_steps(evaluation_receipts, name="evaluation receipts")
    health_refs = _receipt_steps(health_receipts, name="health receipts")
    evaluations = {}
    health = {}
    raw_status = {}
    for step in REQUIRED_STEPS:
        eval_receipt = _load_receipt(
            eval_refs[step],
            manifest=manifest,
            manifest_path=manifest_path,
            step=step,
            expected_format=EVALUATION_RECEIPT_FORMAT,
        )
        health_receipt = _load_receipt(
            health_refs[step],
            manifest=manifest,
            manifest_path=manifest_path,
            step=step,
            expected_format=HEALTH_RECEIPT_FORMAT,
        )
        evaluations[step] = _validate_evaluation_receipt(eval_receipt, manifest=manifest, step=step)
        health[step] = _validate_health_receipt(health_receipt, manifest=manifest, step=step)
        raw_status[step] = {
            "evaluation": eval_receipt["status"],
            "health": health_receipt["status"],
        }
    native_identity = evaluations[0]["native"]["native_identity_sha256"]
    native_order = evaluations[0]["native"]["dataset_order_sha256"]
    if any(
        evaluations[step]["native"]["native_identity_sha256"] != native_identity
        or evaluations[step]["native"]["dataset_order_sha256"] != native_order
        for step in REQUIRED_STEPS
    ):
        raise SSMaxJointEvidenceError("native holdout identity/order drifts across steps")
    for source in VISUAL_SOURCES:
        base_ids = [
            (
                row["pairing_position"],
                row["recipient_index"],
                row["donor_index"],
                row["response_tokens"],
            )
            for row in evaluations[0]["rows"][source]
        ]
        for step in REQUIRED_STEPS[1:]:
            candidate_ids = [
                (
                    row["pairing_position"],
                    row["recipient_index"],
                    row["donor_index"],
                    row["response_tokens"],
                )
                for row in evaluations[step]["rows"][source]
            ]
            if candidate_ids != base_ids:
                raise SSMaxJointEvidenceError(f"{source} matched rows drift across steps")
    hard_by_step = {}
    all_hard = True
    trajectory = {}
    for step in REQUIRED_STEPS:
        state = evaluations[step]["state"]
        counters = health[step]["run_counters"]
        native = evaluations[step]["native"]
        baseline_native = evaluations[0]["native"]
        native_noninferiority = _native_text_noninferiority(
            baseline_native,
            native,
            maximum_relative_increase=float(
                manifest["policy"]["native_text_ce_max_relative_increase"]
            ),
            bootstrap_seed=int(manifest["policy"]["native_text_bootstrap_seed"]) + step,
            bootstrap_samples=int(manifest["policy"]["native_text_bootstrap_samples"]),
        )
        hard = {
            "receipt_status": raw_status[step]["evaluation"] == "passed"
            and raw_status[step]["health"] == "passed",
            "frozen_lexical_input_rows_exact": state["frozen_lexical_input_rows"]["mismatch_count"]
            == 0,
            "frozen_output_projection_exact": state["frozen_output_projection"]["mismatch_count"]
            == 0,
            "data_errors_within_limit": counters["data_errors"]
            <= manifest["policy"]["maximum_data_errors"],
            "optimizer_guard_skips_within_limit": counters["optimizer_guard_skips"]
            <= manifest["policy"]["maximum_optimizer_guard_skips"],
            "nonfinite_losses_within_limit": counters["nonfinite_losses"]
            <= manifest["policy"]["maximum_nonfinite_losses"],
            "nonfinite_gradients_within_limit": counters["nonfinite_gradients"]
            <= manifest["policy"]["maximum_nonfinite_gradients"],
            "native_text_ce_noninferior": native_noninferiority["passed"],
            "all_rank_cursors_exact": len(health[step]["rank_states"])
            == manifest["topology"]["world_size"],
        }
        hard["passed"] = all(hard.values())
        all_hard = all_hard and bool(hard["passed"])
        hard_by_step[str(step)] = hard
        active_total = sum(
            float(health[step]["sources"][source]["active_loss_weight"]) for source in TRAIN_SOURCES
        )
        loss_mass = {}
        for source in TRAIN_SOURCES:
            active = float(health[step]["sources"][source]["active_loss_weight"])
            observed = active / active_total if active_total > 0 else None
            target = float(manifest["loss_mass_targets"][source])
            loss_mass[source] = {
                "target": target,
                "observed": observed,
                "absolute_deviation": abs(observed - target) if observed is not None else None,
            }
        visual: dict[str, dict[str, Any]] = {}
        for source in VISUAL_SOURCES:
            visual[source] = {}
            for window in WINDOWS:
                gap = _mean(evaluations[step]["rows"][source], "ce_gap_wrong_minus_correct", window)
                baseline_gap = _mean(
                    evaluations[0]["rows"][source], "ce_gap_wrong_minus_correct", window
                )
                visual[source][window] = {
                    "gap_wrong_minus_correct": gap,
                    "correct_ce": _mean(evaluations[step]["rows"][source], "correct_ce", window),
                    "retention_vs_step0": (gap / baseline_gap if baseline_gap != 0 else None),
                }
        trajectory[str(step)] = {
            "visual": visual,
            "native_text": {
                "ce": native["ce"],
                "ppl": native["ppl"],
                "ce_change_vs_step0": native["ce"] - baseline_native["ce"],
                "ce_relative_change_vs_step0": (
                    native["ce"] / baseline_native["ce"] - 1.0
                    if baseline_native["ce"] > 0
                    else None
                ),
                "paired_noninferiority": native_noninferiority,
            },
            "loss_mass": loss_mass,
            "run_counters": dict(counters),
        }
    attention_trajectory = {
        str(step): compare_ssmax_attention_reports(
            evaluations[0]["attention"], evaluations[step]["attention"]
        )
        for step in REQUIRED_STEPS[1:]
    }
    attention_reports = {str(step): dict(evaluations[step]["attention"]) for step in REQUIRED_STEPS}
    paired_visual_rows = {
        str(step): {
            source: [dict(row) for row in evaluations[step]["rows"][source]]
            for source in VISUAL_SOURCES
        }
        for step in REQUIRED_STEPS
    }
    report: dict[str, Any] = {
        "format": TRAJECTORY_REPORT_FORMAT,
        "version": SCHEMA_VERSION,
        "status": "passed_hard_invariants" if all_hard else "failed_hard_invariants",
        "decision_scope": "descriptive_non_promotion",
        "cross_arm_schedule": dict(manifest["training_resume_lineage"]["cross_arm_schedule"]),
        "created_at": created_at,
        "manifest": manifest_reference(manifest_path, manifest),
        "run_id": manifest["run_id"],
        "model_variant": manifest["model_variant"],
        "receipts": {
            "evaluation": {str(step): dict(eval_refs[step]) for step in REQUIRED_STEPS},
            "health": {str(step): dict(health_refs[step]) for step in REQUIRED_STEPS},
        },
        "hard_invariants": {"passed": all_hard, "by_step": hard_by_step},
        "trajectory": trajectory,
        "paired_visual_rows": paired_visual_rows,
        "attention_reports": attention_reports,
        "attention_trajectory": attention_trajectory,
        "companion_protocols": dict(manifest["companion_protocols"]),
    }
    report["content_sha256"] = canonical_sha256(report)
    return report


def validate_trajectory_report(
    path: Path, *, verify_live_manifest: bool = True
) -> Mapping[str, Any]:
    """Rebuild a stored trajectory report from its raw-pinned receipts and require equality."""

    report = _exact(load_json(path), _REPORT_FIELDS, name="joint trajectory report")
    if (
        report["format"] != TRAJECTORY_REPORT_FORMAT
        or report["version"] != SCHEMA_VERSION
        or report["decision_scope"] != "descriptive_non_promotion"
        or canonical_sha256(report["cross_arm_schedule"])
        != canonical_sha256(CROSS_ARM_SCHEDULE_CLASSIFICATION)
        or report["status"] not in ("passed_hard_invariants", "failed_hard_invariants")
    ):
        raise SSMaxJointEvidenceError("joint trajectory report identity differs")
    _validate_content(report, name="joint trajectory report")
    manifest_ref = _exact(report["manifest"], _MANIFEST_REF_FIELDS, name="report manifest")
    manifest_path = validate_artifact_reference(
        {"path": manifest_ref["path"], "sha256": manifest_ref["sha256"]}, name="report manifest"
    )
    rebuilt = build_trajectory_report(
        manifest_path=manifest_path,
        evaluation_receipts=report["receipts"]["evaluation"],
        health_receipts=report["receipts"]["health"],
        created_at=str(report["created_at"]),
        verify_live_manifest=verify_live_manifest,
    )
    if rebuilt != report:
        raise SSMaxJointEvidenceError("stored trajectory report differs from raw evidence rebuild")
    return report


def _paired_normal_interval(values: Sequence[float]) -> dict[str, Any]:
    """Return the predeclared descriptive paired-mean 95% normal interval."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise SSMaxJointEvidenceError("paired interval requires at least two finite observations")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    half_width = 1.959963984540054 * standard_error
    lower = mean - half_width
    upper = mean + half_width
    direction = (
        "positive_left_minus_right"
        if lower > 0
        else "negative_left_minus_right"
        if upper < 0
        else "inconclusive"
    )
    return {
        "method": "paired_mean_normal_95_v1",
        "observations": len(array),
        "mean": mean,
        "standard_error": standard_error,
        "lower": lower,
        "upper": upper,
        "direction": direction,
    }


def _row_identity(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["pairing_position"]),
        int(row["recipient_index"]),
        int(row["donor_index"]),
        int(row["response_tokens"]),
    )


def compare_trajectory_reports(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Build a descriptive two-variant comparison without declaring a winner or promotion."""

    for name, report in (("left", left), ("right", right)):
        _exact(report, _REPORT_FIELDS, name=f"{name} trajectory report")
        if report["decision_scope"] != "descriptive_non_promotion":
            raise SSMaxJointEvidenceError(f"{name} report is not descriptive-only")
        if canonical_sha256(report["cross_arm_schedule"]) != canonical_sha256(
            CROSS_ARM_SCHEDULE_CLASSIFICATION
        ):
            raise SSMaxJointEvidenceError(f"{name} report omits the confounded resume schedule")
        _exact(
            report["trajectory"],
            frozenset(str(step) for step in REQUIRED_STEPS),
            name=f"{name} trajectory steps",
        )
        paired = _exact(
            report["paired_visual_rows"],
            frozenset(str(step) for step in REQUIRED_STEPS),
            name=f"{name} paired visual steps",
        )
        for step in REQUIRED_STEPS:
            sources = _exact(
                paired[str(step)],
                frozenset(VISUAL_SOURCES),
                name=f"{name} step{step} paired visual sources",
            )
            if any(not isinstance(sources[source], list) for source in VISUAL_SOURCES):
                raise SSMaxJointEvidenceError(f"{name} step{step} paired visual rows must be lists")
        _exact(
            report["attention_reports"],
            frozenset(str(step) for step in REQUIRED_STEPS),
            name=f"{name} raw attention reports",
        )
        _exact(
            report["attention_trajectory"],
            frozenset(str(step) for step in REQUIRED_STEPS[1:]),
            name=f"{name} attention trajectory steps",
        )
    if left["model_variant"] == right["model_variant"]:
        raise SSMaxJointEvidenceError("pair comparison requires different model variants")
    for source in VISUAL_SOURCES:
        expected_identities = [
            _row_identity(row) for row in left["paired_visual_rows"]["0"][source]
        ]
        for name, report in (("left", left), ("right", right)):
            for step in REQUIRED_STEPS:
                identities = [
                    _row_identity(row) for row in report["paired_visual_rows"][str(step)][source]
                ]
                if identities != expected_identities:
                    raise SSMaxJointEvidenceError(
                        f"{name} step{step} {source} rows are not paired across arms/checkpoints"
                    )
    rows = {}
    for step in REQUIRED_STEPS:
        step_key = str(step)
        visual: dict[str, dict[str, Any]] = {}
        for source in VISUAL_SOURCES:
            visual[source] = {}
            left_rows = left["paired_visual_rows"][step_key][source]
            right_rows = right["paired_visual_rows"][step_key][source]
            left_baseline_rows = left["paired_visual_rows"]["0"][source]
            right_baseline_rows = right["paired_visual_rows"]["0"][source]
            for window in WINDOWS:
                left_metric = left["trajectory"][step_key]["visual"][source][window]
                right_metric = right["trajectory"][step_key]["visual"][source][window]
                gap_same_step = [
                    float(left_row["ce_gap_wrong_minus_correct"][window])
                    - float(right_row["ce_gap_wrong_minus_correct"][window])
                    for left_row, right_row in zip(left_rows, right_rows)
                ]
                correct_same_step = [
                    float(left_row["correct_ce"][window]) - float(right_row["correct_ce"][window])
                    for left_row, right_row in zip(left_rows, right_rows)
                ]
                gap_adaptation_did = [
                    (
                        float(left_row["ce_gap_wrong_minus_correct"][window])
                        - float(left_base["ce_gap_wrong_minus_correct"][window])
                    )
                    - (
                        float(right_row["ce_gap_wrong_minus_correct"][window])
                        - float(right_base["ce_gap_wrong_minus_correct"][window])
                    )
                    for left_row, right_row, left_base, right_base in zip(
                        left_rows, right_rows, left_baseline_rows, right_baseline_rows
                    )
                ]
                correct_adaptation_did = [
                    (float(left_row["correct_ce"][window]) - float(left_base["correct_ce"][window]))
                    - (
                        float(right_row["correct_ce"][window])
                        - float(right_base["correct_ce"][window])
                    )
                    for left_row, right_row, left_base, right_base in zip(
                        left_rows, right_rows, left_baseline_rows, right_baseline_rows
                    )
                ]
                visual[source][window] = {
                    "gap_delta_left_minus_right": left_metric["gap_wrong_minus_correct"]
                    - right_metric["gap_wrong_minus_correct"],
                    "correct_ce_delta_left_minus_right": left_metric["correct_ce"]
                    - right_metric["correct_ce"],
                    "retention_delta_left_minus_right": (
                        left_metric["retention_vs_step0"] - right_metric["retention_vs_step0"]
                        if left_metric["retention_vs_step0"] is not None
                        and right_metric["retention_vs_step0"] is not None
                        else None
                    ),
                    "paired_intervals": {
                        "gap_same_step_left_minus_right": _paired_normal_interval(gap_same_step),
                        "correct_ce_same_step_left_minus_right": _paired_normal_interval(
                            correct_same_step
                        ),
                        "gap_adaptation_did_left_minus_right": _paired_normal_interval(
                            gap_adaptation_did
                        ),
                        "correct_ce_adaptation_did_left_minus_right": (
                            _paired_normal_interval(correct_adaptation_did)
                        ),
                    },
                }
        left_native = left["trajectory"][step_key]["native_text"]
        right_native = right["trajectory"][step_key]["native_text"]
        left_native_zero = left["trajectory"]["0"]["native_text"]
        right_native_zero = right["trajectory"]["0"]["native_text"]
        rows[step_key] = {
            "visual": visual,
            "native_text": {
                "ce_delta_left_minus_right": left["trajectory"][step_key]["native_text"]["ce"]
                - right["trajectory"][step_key]["native_text"]["ce"],
                "ppl_delta_left_minus_right": left["trajectory"][step_key]["native_text"]["ppl"]
                - right["trajectory"][step_key]["native_text"]["ppl"],
                "ce_adaptation_did_left_minus_right": (left_native["ce"] - left_native_zero["ce"])
                - (right_native["ce"] - right_native_zero["ce"]),
                "ppl_adaptation_did_left_minus_right": (
                    left_native["ppl"] - left_native_zero["ppl"]
                )
                - (right_native["ppl"] - right_native_zero["ppl"]),
            },
            "attention": {
                "baseline": "left",
                "candidate": "right",
                "comparison": compare_ssmax_attention_reports(
                    _mapping(
                        left["attention_reports"][step_key],
                        name=f"left step{step} attention report",
                    ),
                    _mapping(
                        right["attention_reports"][step_key],
                        name=f"right step{step} attention report",
                    ),
                ),
            },
        }
    interval_signals = {}
    for source in VISUAL_SOURCES:
        for window in WINDOWS:
            for metric in (
                "gap_adaptation_did_left_minus_right",
                "correct_ce_adaptation_did_left_minus_right",
            ):
                directions = [
                    rows[str(step)]["visual"][source][window]["paired_intervals"][metric][
                        "direction"
                    ]
                    for step in REQUIRED_STEPS[1:]
                ]
                consistent = (
                    directions[0]
                    if directions[0] != "inconclusive"
                    and all(direction == directions[0] for direction in directions)
                    else "inconclusive"
                )
                interval_signals[f"{source}/{window}/{metric}"] = {
                    "post_baseline_directions": directions,
                    "consistent_direction": consistent,
                }
    result: dict[str, Any] = {
        "format": PAIR_COMPARISON_FORMAT,
        "version": SCHEMA_VERSION,
        "decision_scope": "descriptive_non_promotion",
        "cross_arm_schedule": dict(CROSS_ARM_SCHEDULE_CLASSIFICATION),
        "winner": None,
        "left": {
            "run_id": left["run_id"],
            "model_variant": left["model_variant"],
            "content_sha256": left["content_sha256"],
        },
        "right": {
            "run_id": right["run_id"],
            "model_variant": right["model_variant"],
            "content_sha256": right["content_sha256"],
        },
        "hard_invariants": {
            "left": left["hard_invariants"]["passed"],
            "right": right["hard_invariants"]["passed"],
        },
        "adaptation_interval_rule": {
            "method": "paired_mean_normal_95_v1",
            "required_post_baseline_steps": list(REQUIRED_STEPS[1:]),
            "criterion": (
                "one strict nonzero interval direction at every post-baseline retained step"
            ),
            "scope": "descriptive_molmofiability_signal_not_a_promotion_gate",
            "signals": interval_signals,
        },
        "trajectory_deltas": rows,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result
