"""Immutable evidence contracts for the paired SSMax Vision Alignment bridge.

The bridge experiment deliberately evaluates a trajectory rather than a single chosen
checkpoint.  This module owns the model-neutral part of that contract: immutable run manifests,
checkpoint-content identities, paired/bootstrap statistics, strict generic DCP inventory checks,
and cross-step promotion validation.  GPU model construction and dataset replay live in the
corresponding scripts under :mod:`src.scripts.eval`.

Nothing in this module accepts a checkpoint override.  A finalized manifest names and hashes all
seven checkpoints, the fixed matched-wrong pairings, validation data, recipe, and training profile.
Evaluators select only a step from that closed set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.eval.matched_wrong_image import (
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)
from olmo_core.eval.ssmax_attention_diagnostics import (
    compare_ssmax_attention_reports,
    validate_ssmax_attention_report,
)
from olmo_core.train.callbacks.ssmax_health_ledger import (
    SSMaxHealthLedgerError,
    validate_ssmax_health_ledger_state,
)

MANIFEST_SPEC_FORMAT = "vision_alignment_ssmax_bridge_manifest_spec"
MANIFEST_FORMAT = "vision_alignment_ssmax_bridge_manifest"
MATCHED_STATE_RECEIPT_FORMAT = "vision_alignment_ssmax_bridge_matched_state_receipt"
HEALTH_RECEIPT_FORMAT = "vision_alignment_ssmax_bridge_health_receipt"
PROMOTION_REPORT_FORMAT = "vision_alignment_ssmax_bridge_promotion_report"
PAIR_COMPARISON_FORMAT = "vision_alignment_ssmax_bridge_pair_comparison"
SCHEMA_VERSION = 1

REQUIRED_STEPS = (0, 100, 200, 250, 300, 400, 500)
SOURCES = ("pixmo_caption", "pixmo_transcript")
WINDOWS = ("first_8", "first_32", "all")
IMAGE_TOKEN_ROWS = (100278, 100279, 100280, 100281, 100282, 100283)
MODEL_VARIANTS = ("ssmax_head_qknorm", "ssmax_no_qknorm")
BRIDGE_GLOBAL_BATCH_INSTANCES = 128

# These are scientific protocol constants, not tunable evaluator defaults.  Finalized production
# evidence must come from exactly this preregistered design; structurally valid weaker manifests
# are deliberately rejected.
BRIDGE_EVALUATION_CONTRACT: Mapping[str, Any] = {
    "sources": list(SOURCES),
    "steps": list(REQUIRED_STEPS),
    "examples_per_source": 512,
    "pairing_seed": 6198,
    "bootstrap_seed": 1006201,
    "bootstrap_samples": 10_000,
    "rank_batch_instances": 4,
    "windows": list(WINDOWS),
}
BRIDGE_TOPOLOGY_CONTRACT: Mapping[str, Any] = {
    "world_size": 16,
    "num_nodes": 2,
    "gpus_per_node": 8,
    "data_parallel": "hsdp",
}
BRIDGE_POLICY_CONTRACT: Mapping[str, Any] = {
    "positive_gap_ci_steps": [250, 300, 400, 500],
    "step0_gap_role": "descriptive_baseline_only",
    "retention_reference_step": 250,
    "retention_candidate_step": 500,
    "retention_windows": ["first_8", "first_32"],
    "minimum_gap_retention": 0.8,
    "correct_ce_reference_step": 250,
    "correct_ce_candidate_step": 500,
    "correct_ce_max_relative_increase": 0.02,
    "require_step0_to_final_correct_ce_improvement": True,
    "loss_mass_share_tolerance": 0.02,
    "maximum_data_errors": 0,
}
MANIFEST_SPEC_RELATIVE_PATHS = {
    "ssmax-qknorm-1p4b-cx8-bridge-v1": {
        "ssmax_head_qknorm": (
            "configs/vision_moe/vision_alignment/eval/" "ssmax_head_qknorm_bridge_manifest_v1.json"
        ),
        "ssmax_no_qknorm": (
            "configs/vision_moe/vision_alignment/eval/" "ssmax_no_qknorm_bridge_manifest_v1.json"
        ),
    },
    "ssmax-qknorm-1p4b-cx8-bridge-v2": {
        "ssmax_head_qknorm": (
            "configs/vision_moe/vision_alignment/eval/" "ssmax_head_qknorm_bridge_manifest_v2.json"
        ),
        "ssmax_no_qknorm": (
            "configs/vision_moe/vision_alignment/eval/" "ssmax_no_qknorm_bridge_manifest_v2.json"
        ),
    },
}

MATCHED_STATE_PRODUCER = "matched_state"
HEALTH_PRODUCER = "health"
PRODUCER_RELATIVE_PATHS = {
    MATCHED_STATE_PRODUCER: "src/scripts/eval/vision_alignment_ssmax_bridge.py",
    HEALTH_PRODUCER: "src/scripts/eval/vision_alignment_ssmax_bridge_health.py",
}

_SHA256_CHARS = frozenset("0123456789abcdef")
_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
_GIT_FIELDS = frozenset({"repo", "repo_url", "ref"})
_PRODUCER_SOURCE_REF_FIELDS = frozenset({"repo_relative_path", "sha256", "git_ref"})
_PARENT_FIELDS = frozenset(
    {
        "checkpoint",
        "config_sha256",
        "data_paths_sha256",
        "marker_sha256",
        "dcp_metadata_sha256",
        "trainer_state_sha256",
        "model_keyset_sha256",
        "model_inventory_sha256",
        "checkpoint_identity_sha256",
        "state_file_count",
        "state_file_inventory_sha256",
        "trainer_state_count",
        "trainer_state_inventory_sha256",
        "source_commit",
        "olmo_core_commit",
        "parameter_count",
        "tensor_count",
    }
)
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
        "pair_id",
        "arm",
        "model_variant",
        "run_name",
        "checkpoint_root",
        "parent",
        "training_profile",
        "recipe",
        "validation",
        "attention_probe",
        "pairing_paths",
        "evaluation",
        "topology",
        "policy",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "version",
        "created_at",
        "pair_id",
        "arm",
        "model_variant",
        "run_name",
        "parent",
        "parent_load_receipt",
        "git",
        "manifest_spec",
        "producers",
        "training_profile",
        "recipe",
        "validation",
        "attention_probe",
        "pairings",
        "evaluation",
        "topology",
        "policy",
        "checkpoints",
        "content_sha256",
    }
)
_MATCHED_STATE_RECEIPT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "manifest",
        "pair_id",
        "arm",
        "model_variant",
        "step",
        "checkpoint",
        "step0_checkpoint",
        "strict_generic_dcp_load",
        "step0_strict_generic_dcp_load",
        "frozen_state",
        "component_state",
        "validation",
        "pairings",
        "protocol",
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
        "pair_id",
        "arm",
        "model_variant",
        "step",
        "checkpoint",
        "protocol",
        "loader",
        "sources",
        "health_ledger",
        "summary",
        "evidence",
        "content_sha256",
    }
)
_PROMOTION_REPORT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "manifest",
        "pair_id",
        "arm",
        "model_variant",
        "receipts",
        "trajectory",
        "attention_trajectory",
        "deviations",
        "content_sha256",
    }
)
_SSMAX_PARENT_GATE_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "recipe_version",
        "formatter_version",
        "phase",
        "model_variant",
        "arm",
        "checkpoint",
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "global_step",
        "metrics_artifact_sha256",
        "promotion_report_path",
        "promotion_report_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "approved_by",
        "approved_at",
        "waivers",
    }
)


class SSMaxBridgeEvidenceError(ValueError):
    """Raised when paired SSMax bridge evidence violates its immutable contract."""


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Serialize finite JSON in the canonical form used for semantic identities."""

    suffix = "\n" if trailing_newline else ""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + suffix
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 over :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash ``path`` without reading it all into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SSMaxBridgeEvidenceError(
            f"Could not hash required artifact {path}: {error}"
        ) from error
    return digest.hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SSMaxBridgeEvidenceError(f"JSON repeats key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load strict finite JSON from ``path``."""

    def reject_constant(value: str) -> Any:
        raise SSMaxBridgeEvidenceError(f"JSON contains non-finite constant {value}")

    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SSMaxBridgeEvidenceError(f"Could not read JSON artifact {path}: {error}") from error


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create an immutable pretty-printed JSON artifact."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable artifact {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _exact_fields(value: Any, expected: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxBridgeEvidenceError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise SSMaxBridgeEvidenceError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise SSMaxBridgeEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SSMaxBridgeEvidenceError(f"{name} must be a {qualifier} integer")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SSMaxBridgeEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SSMaxBridgeEvidenceError(f"{name} must be finite")
    return result


def _timestamp(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise SSMaxBridgeEvidenceError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SSMaxBridgeEvidenceError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SSMaxBridgeEvidenceError(f"{name} must include a timezone")
    return value


def artifact_reference(path: Path) -> dict[str, str]:
    """Return an absolute raw-byte SHA-pinned reference to an existing file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SSMaxBridgeEvidenceError(f"Required artifact is absent: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_artifact_reference(value: Any, *, name: str) -> Path:
    """Validate and return the exact live file named by an artifact reference."""

    reference = _exact_fields(value, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    path_value = reference["path"]
    if not isinstance(path_value, str) or not path_value:
        raise SSMaxBridgeEvidenceError(f"{name} path must be non-empty")
    path = Path(path_value).expanduser().resolve()
    expected = _sha256(reference["sha256"], name=f"{name} SHA-256")
    if not path.is_file() or sha256_file(path) != expected:
        raise SSMaxBridgeEvidenceError(f"{name} differs from its immutable reference")
    return path


def _validate_artifact_reference_shape(value: Any, *, name: str) -> Mapping[str, str]:
    reference = _exact_fields(value, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    path = reference["path"]
    if not isinstance(path, str) or not path:
        raise SSMaxBridgeEvidenceError(f"{name} path must be non-empty")
    _sha256(reference["sha256"], name=f"{name} SHA-256")
    return {"path": str(reference["path"]), "sha256": str(reference["sha256"])}


def _validate_parent_contract(value: Any) -> Mapping[str, Any]:
    parent = _exact_fields(value, _PARENT_FIELDS, name="manifest parent")
    checkpoint = parent["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint:
        raise SSMaxBridgeEvidenceError("Manifest parent checkpoint must be non-empty")
    for hash_name in (
        "config_sha256",
        "data_paths_sha256",
        "marker_sha256",
        "dcp_metadata_sha256",
        "trainer_state_sha256",
        "model_keyset_sha256",
        "model_inventory_sha256",
        "checkpoint_identity_sha256",
        "state_file_inventory_sha256",
        "trainer_state_inventory_sha256",
    ):
        _sha256(parent[hash_name], name=f"parent {hash_name}")
    for commit_name in ("source_commit", "olmo_core_commit"):
        commit = parent[commit_name]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in _SHA256_CHARS for character in commit)
        ):
            raise SSMaxBridgeEvidenceError(f"Parent {commit_name} must be a lowercase git SHA")
    _positive_int(parent["parameter_count"], name="parent parameter count")
    _positive_int(parent["tensor_count"], name="parent tensor count")
    _positive_int(parent["state_file_count"], name="parent state file count")
    _positive_int(parent["trainer_state_count"], name="parent trainer state count")
    return parent


def _validate_git_identity(value: Any) -> Mapping[str, str]:
    git = _exact_fields(value, _GIT_FIELDS, name="manifest git identity")
    for name in ("repo", "repo_url"):
        if not isinstance(git[name], str) or not git[name]:
            raise SSMaxBridgeEvidenceError(f"Manifest git {name} must be non-empty")
    ref = git["ref"]
    if (
        not isinstance(ref, str)
        or len(ref) != 40
        or any(character not in _SHA256_CHARS for character in ref)
    ):
        raise SSMaxBridgeEvidenceError("Manifest git ref must be a lowercase 40-character SHA")
    return {name: str(git[name]) for name in ("repo", "repo_url", "ref")}


def _validate_repository_checkout(git: Mapping[str, str], *, repository_root: Path) -> None:
    if git["repo"] != "allenai/OLMo-core" or git["repo_url"] != (
        "https://github.com/allenai/OLMo-core"
    ):
        raise SSMaxBridgeEvidenceError("Saved bridge git identity names a different repository")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxBridgeEvidenceError("Could not verify the saved bridge git checkout") from error
    if head != git["ref"]:
        raise SSMaxBridgeEvidenceError("Evidence checkout HEAD differs from the manifest git ref")
    if dirty:
        raise SSMaxBridgeEvidenceError("Evidence checkout is not clean")


def _git_blob_bytes(
    git: Mapping[str, str], *, repository_root: Path, repo_relative_path: str, name: str
) -> bytes:
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(repository_root),
                "show",
                f"{git['ref']}:{repo_relative_path}",
            ],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SSMaxBridgeEvidenceError(f"Could not read the saved git {name} blob") from error


def _validate_producer_source_references(
    value: Any, *, git: Mapping[str, str]
) -> Mapping[str, Mapping[str, str]]:
    producers = _exact_fields(
        value,
        frozenset(PRODUCER_RELATIVE_PATHS),
        name="manifest evidence producers",
    )
    validated: dict[str, Mapping[str, str]] = {}
    for producer, expected_path in PRODUCER_RELATIVE_PATHS.items():
        reference = _exact_fields(
            producers[producer],
            _PRODUCER_SOURCE_REF_FIELDS,
            name=f"manifest {producer} producer source",
        )
        if reference["repo_relative_path"] != expected_path:
            raise SSMaxBridgeEvidenceError(
                f"Manifest {producer} producer is not the canonical repository source"
            )
        _sha256(reference["sha256"], name=f"manifest {producer} producer SHA-256")
        if reference["git_ref"] != git["ref"]:
            raise SSMaxBridgeEvidenceError(
                f"Manifest {producer} producer git ref differs from the manifest"
            )
        validated[producer] = {
            name: str(reference[name]) for name in ("repo_relative_path", "sha256", "git_ref")
        }
    return validated


def _producer_source_references(
    git: Mapping[str, str], *, repository_root: Path
) -> dict[str, dict[str, str]]:
    """Build exact source references from clean blobs at the manifest Git revision."""

    _validate_repository_checkout(git, repository_root=repository_root)
    references: dict[str, dict[str, str]] = {}
    for producer, relative in PRODUCER_RELATIVE_PATHS.items():
        path = (repository_root / relative).resolve()
        if path != repository_root.resolve() / relative or not path.is_file():
            raise SSMaxBridgeEvidenceError(f"Canonical {producer} producer source is absent")
        live_sha256 = sha256_file(path)
        blob_sha256 = hashlib.sha256(
            _git_blob_bytes(
                git,
                repository_root=repository_root,
                repo_relative_path=relative,
                name=f"{producer} producer",
            )
        ).hexdigest()
        if live_sha256 != blob_sha256:
            raise SSMaxBridgeEvidenceError(
                f"Live {producer} producer bytes differ from the manifest git blob"
            )
        references[producer] = {
            "repo_relative_path": relative,
            "sha256": live_sha256,
            "git_ref": git["ref"],
        }
    return references


def _canonical_manifest_spec_relative_path(*, pair_id: Any, model_variant: Any) -> str:
    """Resolve the sole checked-in manifest specification for one protocol arm."""

    if not isinstance(pair_id, str) or pair_id not in MANIFEST_SPEC_RELATIVE_PATHS:
        raise SSMaxBridgeEvidenceError("Manifest pair ID is not a registered bridge protocol")
    by_arm = MANIFEST_SPEC_RELATIVE_PATHS[pair_id]
    if not isinstance(model_variant, str) or model_variant not in by_arm:
        raise SSMaxBridgeEvidenceError(
            "Manifest model variant is not registered for its bridge protocol"
        )
    return by_arm[model_variant]


def _validate_manifest_spec_source_reference(
    value: Any, *, pair_id: str, model_variant: str, git: Mapping[str, str]
) -> dict[str, str]:
    reference = _exact_fields(
        value,
        _PRODUCER_SOURCE_REF_FIELDS,
        name="manifest specification source",
    )
    expected_relative = _canonical_manifest_spec_relative_path(
        pair_id=pair_id,
        model_variant=model_variant,
    )
    if reference["repo_relative_path"] != expected_relative:
        raise SSMaxBridgeEvidenceError(
            "Manifest specification is not the canonical per-arm repository source"
        )
    _sha256(reference["sha256"], name="manifest specification SHA-256")
    if reference["git_ref"] != git["ref"]:
        raise SSMaxBridgeEvidenceError("Manifest specification git ref differs from the manifest")
    return {name: str(reference[name]) for name in ("repo_relative_path", "sha256", "git_ref")}


def _manifest_spec_source_reference(
    spec_path: Path,
    spec: Mapping[str, Any],
    *,
    git: Mapping[str, str],
    repository_root: Path,
) -> dict[str, str]:
    """Bind a manifest spec to its canonical live file and clean Git blob."""

    _validate_repository_checkout(git, repository_root=repository_root)
    relative = _canonical_manifest_spec_relative_path(
        pair_id=spec.get("pair_id"),
        model_variant=spec.get("model_variant"),
    )
    canonical_path = (repository_root / relative).resolve()
    if spec_path.expanduser().resolve() != canonical_path or not canonical_path.is_file():
        raise SSMaxBridgeEvidenceError(
            "Manifest specification must be the canonical checked-in per-arm file"
        )
    if load_json(canonical_path) != dict(spec):
        raise SSMaxBridgeEvidenceError(
            "Loaded manifest specification differs from its canonical source bytes"
        )
    live_sha256 = sha256_file(canonical_path)
    blob_sha256 = hashlib.sha256(
        _git_blob_bytes(
            git,
            repository_root=repository_root,
            repo_relative_path=relative,
            name="manifest specification",
        )
    ).hexdigest()
    if live_sha256 != blob_sha256:
        raise SSMaxBridgeEvidenceError(
            "Live manifest specification bytes differ from the manifest git blob"
        )
    return {
        "repo_relative_path": relative,
        "sha256": live_sha256,
        "git_ref": git["ref"],
    }


def validate_manifest_producer_source(
    manifest: Mapping[str, Any], *, producer: str, source_path: Path
) -> dict[str, str]:
    """Prove a canonical evidence producer is running from the manifest's clean checkout."""

    if producer not in PRODUCER_RELATIVE_PATHS:
        raise SSMaxBridgeEvidenceError(f"Unknown bridge evidence producer {producer!r}")
    git = _validate_git_identity(manifest.get("git"))
    references = _validate_producer_source_references(manifest.get("producers"), git=git)
    expected_relative = PRODUCER_RELATIVE_PATHS[producer]
    source = source_path.expanduser().resolve()
    repository_root = source
    for _ in Path(expected_relative).parts:
        repository_root = repository_root.parent
    if source != (repository_root / expected_relative).resolve():
        raise SSMaxBridgeEvidenceError(
            f"Running {producer} producer is not the canonical repository source"
        )
    _validate_repository_checkout(git, repository_root=repository_root)
    reference = references[producer]
    live_sha256 = sha256_file(source)
    blob_sha256 = hashlib.sha256(
        _git_blob_bytes(
            git,
            repository_root=repository_root,
            repo_relative_path=expected_relative,
            name=f"{producer} producer",
        )
    ).hexdigest()
    if live_sha256 != reference["sha256"] or blob_sha256 != reference["sha256"]:
        raise SSMaxBridgeEvidenceError(
            f"Running {producer} producer differs from its manifest source identity"
        )
    return dict(reference)


def _validate_saved_git_checkout(
    git: Mapping[str, str], *, recipe_path: Path, profile_path: Path
) -> None:
    """Prove post-hoc source artifacts are clean blobs from the run's saved git ref."""

    repository_root = recipe_path.resolve().parents[3]
    try:
        recipe_relative = recipe_path.resolve().relative_to(repository_root).as_posix()
        profile_relative = profile_path.resolve().relative_to(repository_root).as_posix()
    except ValueError as error:
        raise SSMaxBridgeEvidenceError(
            "Recipe and reviewed profile must live in the saved git repository"
        ) from error
    if recipe_relative != "src/scripts/train/Vision-Alignment.py":
        raise SSMaxBridgeEvidenceError("Manifest recipe is not the canonical training script")
    _validate_repository_checkout(git, repository_root=repository_root)
    for path, relative, name in (
        (recipe_path, recipe_relative, "recipe"),
        (profile_path, profile_relative, "reviewed profile"),
    ):
        blob = _git_blob_bytes(
            git,
            repository_root=repository_root,
            repo_relative_path=relative,
            name=name,
        )
        if hashlib.sha256(blob).hexdigest() != sha256_file(path):
            raise SSMaxBridgeEvidenceError(f"Live {name} bytes differ from the saved git blob")


def _checkpoint_state_dir(checkpoint: Path) -> Path:
    nested = checkpoint / "model_and_optim"
    return nested if nested.is_dir() else checkpoint


def checkpoint_identity(checkpoint: Path, *, workers: int = 8) -> dict[str, Any]:
    """Hash one native checkpoint, including every DCP shard and trainer rank state."""

    root = checkpoint.expanduser().resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not root.is_dir() or not root.name.startswith("step"):
        raise SSMaxBridgeEvidenceError(f"Checkpoint is not a step directory: {root}")
    try:
        step = int(root.name.removeprefix("step"))
    except ValueError as error:
        raise SSMaxBridgeEvidenceError(
            f"Checkpoint name does not encode a step: {root.name}"
        ) from error
    config_path = root / "config.json"
    marker_path = root / ".metadata.json"
    state_dir = _checkpoint_state_dir(root)
    metadata_path = state_dir / ".metadata"
    for path in (config_path, marker_path, metadata_path):
        if not path.is_file():
            raise SSMaxBridgeEvidenceError(f"Checkpoint artifact is absent: {path}")
    marker = load_json(marker_path)
    if not isinstance(marker, Mapping) or marker.get("ephemeral") is not False:
        raise SSMaxBridgeEvidenceError(f"Checkpoint is not a completed permanent save: {root}")
    for marker_step_name in ("global_step", "step"):
        if marker_step_name in marker and marker[marker_step_name] != step:
            raise SSMaxBridgeEvidenceError(
                f"Checkpoint marker {marker_step_name} differs from directory step {step}"
            )
    state_paths = sorted(path for path in state_dir.iterdir() if path.is_file())
    trainer_paths = sorted(
        root.joinpath("train").glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    if not state_paths or not trainer_paths:
        raise SSMaxBridgeEvidenceError(f"Checkpoint lacks DCP or trainer state files: {root}")
    expected_ranks = list(range(len(trainer_paths)))
    actual_ranks = [int(path.stem.removeprefix("rank")) for path in trainer_paths]
    if actual_ranks != expected_ranks:
        raise SSMaxBridgeEvidenceError(f"Trainer rank-state inventory is not contiguous: {root}")
    for rank, path in enumerate(trainer_paths):
        try:
            trainer_state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise SSMaxBridgeEvidenceError(
                f"Could not read trainer rank{rank} state from {root}"
            ) from error
        if not isinstance(trainer_state, Mapping) or trainer_state.get("global_step") != step:
            raise SSMaxBridgeEvidenceError(
                f"Trainer rank{rank} global step differs from directory step {step}"
            )
        if trainer_state.get("world_size") != len(trainer_paths):
            raise SSMaxBridgeEvidenceError(
                f"Trainer rank{rank} world size differs from its checkpoint inventory"
            )
    all_paths = [*state_paths, *trainer_paths]
    with ThreadPoolExecutor(max_workers=min(workers, len(all_paths))) as executor:
        digests = list(executor.map(sha256_file, all_paths))
    state_digests = digests[: len(state_paths)]
    trainer_digests = digests[len(state_paths) :]
    state_inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(state_paths, state_digests, strict=True)
    ]
    trainer_inventory = [
        {
            "rank": rank,
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for rank, (path, digest) in enumerate(zip(trainer_paths, trainer_digests, strict=True))
    ]
    identity: dict[str, Any] = {
        "path": str(root),
        "global_step": step,
        "config_sha256": sha256_file(config_path),
        "marker_sha256": sha256_file(marker_path),
        "dcp_metadata_sha256": sha256_file(metadata_path),
        "state_file_count": len(state_paths),
        "state_file_inventory_sha256": canonical_sha256(state_inventory),
        "trainer_state_count": len(trainer_paths),
        "trainer_state_inventory_sha256": canonical_sha256(trainer_inventory),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def validate_checkpoint_reference(
    value: Any, *, expected_step: int, workers: int = 8, verify_live: bool = True
) -> Mapping[str, Any]:
    """Validate one manifest checkpoint reference and optionally rehash its live contents."""

    reference = _exact_fields(value, _CHECKPOINT_FIELDS, name=f"step{expected_step} checkpoint")
    if reference["global_step"] != expected_step:
        raise SSMaxBridgeEvidenceError(f"Checkpoint reference does not name step{expected_step}")
    for field in (
        "config_sha256",
        "marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "trainer_state_inventory_sha256",
        "identity_sha256",
    ):
        _sha256(reference[field], name=f"step{expected_step} {field}")
    _positive_int(reference["state_file_count"], name="state file count")
    _positive_int(reference["trainer_state_count"], name="trainer state count")
    path_value = reference["path"]
    if not isinstance(path_value, str) or Path(path_value).resolve().name != f"step{expected_step}":
        raise SSMaxBridgeEvidenceError(f"Checkpoint path does not end in step{expected_step}")
    expected_identity = canonical_sha256(
        {key: item for key, item in reference.items() if key != "identity_sha256"}
    )
    if expected_identity != reference["identity_sha256"]:
        raise SSMaxBridgeEvidenceError(f"step{expected_step} checkpoint identity is malformed")
    if verify_live:
        actual = checkpoint_identity(Path(path_value), workers=workers)
        if actual != dict(reference):
            differing = sorted(key for key in reference if reference[key] != actual.get(key))
            raise SSMaxBridgeEvidenceError(
                f"Live step{expected_step} checkpoint differs in fields {differing}"
            )
    return reference


def load_manifest_spec(path: Path) -> Mapping[str, Any]:
    """Load and structurally validate a checked-in per-arm manifest specification."""

    spec = _exact_fields(load_json(path), _SPEC_FIELDS, name="SSMax manifest spec")
    if spec["format"] != MANIFEST_SPEC_FORMAT or spec["version"] != SCHEMA_VERSION:
        raise SSMaxBridgeEvidenceError("SSMax manifest spec identity is incompatible")
    _validate_common_manifest_fields(spec, finalized=False)
    return spec


def _validate_common_manifest_fields(value: Mapping[str, Any], *, finalized: bool) -> None:
    for name in ("pair_id", "arm", "run_name"):
        if not isinstance(value[name], str) or not value[name]:
            raise SSMaxBridgeEvidenceError(f"Manifest {name} must be a non-empty string")
    if value["model_variant"] not in MODEL_VARIANTS or value["arm"] != value["model_variant"]:
        raise SSMaxBridgeEvidenceError("Manifest arm/model_variant is not a supported paired arm")
    _canonical_manifest_spec_relative_path(
        pair_id=value["pair_id"],
        model_variant=value["model_variant"],
    )
    topology = _exact_fields(
        value["topology"],
        frozenset({"world_size", "num_nodes", "gpus_per_node", "data_parallel"}),
        name="manifest topology",
    )
    if dict(topology) != dict(BRIDGE_TOPOLOGY_CONTRACT):
        raise SSMaxBridgeEvidenceError("Manifest topology differs from the locked contract")
    world_size = _positive_int(topology["world_size"], name="manifest world size")
    nodes = _positive_int(topology["num_nodes"], name="manifest node count")
    gpus = _positive_int(topology["gpus_per_node"], name="manifest GPUs per node")
    if world_size != nodes * gpus or topology["data_parallel"] != "hsdp":
        raise SSMaxBridgeEvidenceError("Manifest topology must be a complete HSDP world")
    evaluation = _exact_fields(
        value["evaluation"],
        frozenset(
            {
                "sources",
                "steps",
                "examples_per_source",
                "pairing_seed",
                "bootstrap_seed",
                "bootstrap_samples",
                "rank_batch_instances",
                "windows",
            }
        ),
        name="manifest evaluation",
    )
    if dict(evaluation) != dict(BRIDGE_EVALUATION_CONTRACT):
        raise SSMaxBridgeEvidenceError("Manifest evaluation differs from the locked contract")
    if (
        evaluation["sources"] != list(SOURCES)
        or evaluation["steps"] != list(REQUIRED_STEPS)
        or evaluation["windows"] != list(WINDOWS)
    ):
        raise SSMaxBridgeEvidenceError("Manifest evaluation source/step/window set differs")
    examples = _positive_int(evaluation["examples_per_source"], name="evaluation examples")
    rank_batch = _positive_int(evaluation["rank_batch_instances"], name="rank batch instances")
    if examples % (world_size * rank_batch):
        raise SSMaxBridgeEvidenceError("Evaluation examples must divide the global instance batch")
    for name in ("pairing_seed", "bootstrap_seed"):
        _positive_int(evaluation[name], name=name, allow_zero=True)
    _positive_int(evaluation["bootstrap_samples"], name="bootstrap samples")
    policy = _exact_fields(
        value["policy"],
        frozenset(
            {
                "positive_gap_ci_steps",
                "step0_gap_role",
                "retention_reference_step",
                "retention_candidate_step",
                "retention_windows",
                "minimum_gap_retention",
                "correct_ce_reference_step",
                "correct_ce_candidate_step",
                "correct_ce_max_relative_increase",
                "require_step0_to_final_correct_ce_improvement",
                "loss_mass_share_tolerance",
                "maximum_data_errors",
            }
        ),
        name="manifest policy",
    )
    if dict(policy) != dict(BRIDGE_POLICY_CONTRACT):
        raise SSMaxBridgeEvidenceError("Manifest policy differs from the locked contract")
    if any(step not in REQUIRED_STEPS for step in policy["positive_gap_ci_steps"]):
        raise SSMaxBridgeEvidenceError("Policy names a positive-CI step outside the trajectory")
    if policy["step0_gap_role"] != "descriptive_baseline_only":
        raise SSMaxBridgeEvidenceError("Step-0 gap role must remain descriptive baseline only")
    if policy["retention_windows"] != ["first_8", "first_32"]:
        raise SSMaxBridgeEvidenceError("Policy retention windows differ from the locked contract")
    for name in (
        "retention_reference_step",
        "retention_candidate_step",
        "correct_ce_reference_step",
        "correct_ce_candidate_step",
    ):
        if policy[name] not in REQUIRED_STEPS:
            raise SSMaxBridgeEvidenceError(f"Policy {name} is outside the saved steps")
    for name in (
        "minimum_gap_retention",
        "correct_ce_max_relative_increase",
        "loss_mass_share_tolerance",
    ):
        number = _finite(policy[name], name=f"policy {name}")
        if number < 0:
            raise SSMaxBridgeEvidenceError(f"Policy {name} must be non-negative")
    _positive_int(policy["maximum_data_errors"], name="maximum data errors", allow_zero=True)
    if type(policy["require_step0_to_final_correct_ce_improvement"]) is not bool:
        raise SSMaxBridgeEvidenceError(
            "Policy require_step0_to_final_correct_ce_improvement must be boolean"
        )
    _validate_parent_contract(value["parent"])
    if finalized:
        for name in (
            "parent_load_receipt",
            "training_profile",
            "recipe",
            "validation",
            "attention_probe",
        ):
            validate_artifact_reference(value[name], name=name.replace("_", " "))


def _validate_finalized_manifest_against_spec(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    checkpoint_root: Path,
) -> None:
    """Prove every finalized field derived from the checked-in spec remains identical."""

    for name in (
        "pair_id",
        "arm",
        "model_variant",
        "run_name",
        "parent",
        "evaluation",
        "topology",
        "policy",
    ):
        if manifest[name] != spec[name]:
            raise SSMaxBridgeEvidenceError(
                f"Finalized manifest {name} differs from its checked-in specification"
            )
    if Path(str(spec["checkpoint_root"])).expanduser().resolve() != checkpoint_root.resolve():
        raise SSMaxBridgeEvidenceError(
            "Finalized manifest checkpoint root differs from its checked-in specification"
        )
    for name in ("training_profile", "recipe", "validation", "attention_probe"):
        spec_path = Path(str(spec[name])).expanduser().resolve()
        manifest_path = Path(str(manifest[name]["path"])).expanduser().resolve()
        if spec_path != manifest_path:
            raise SSMaxBridgeEvidenceError(
                f"Finalized manifest {name} differs from its checked-in specification"
            )
    pairing_paths = _exact_fields(
        spec["pairing_paths"], frozenset(SOURCES), name="manifest specification pairing paths"
    )
    manifest_pairings = _exact_fields(
        manifest["pairings"], frozenset(SOURCES), name="finalized manifest pairings"
    )
    for source in SOURCES:
        reference = _validate_artifact_reference_shape(
            manifest_pairings[source], name=f"{source} pairing"
        )
        if (
            Path(str(pairing_paths[source])).expanduser().resolve()
            != Path(str(reference["path"])).expanduser().resolve()
        ):
            raise SSMaxBridgeEvidenceError(
                f"Finalized manifest {source} pairing differs from its checked-in specification"
            )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxBridgeEvidenceError(f"Saved bridge config {name} must be an object")
    return value


def _expect_saved(actual: Any, expected: Any, *, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise SSMaxBridgeEvidenceError(
            f"Saved bridge config {name} differs: expected={expected!r}, actual={actual!r}"
        )


def _resolved_saved_path(value: Any, *, repository_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SSMaxBridgeEvidenceError(f"Saved bridge config {name} path must be non-empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _validate_saved_bridge_config(
    spec: Mapping[str, Any],
    raw_config: Any,
    *,
    checkpoint_root: Path,
    profile_path: Path,
    profile_sha256: str,
    recipe_path: Path,
    validation_path: Path,
    validation_sha256: str,
) -> dict[str, str]:
    """Bind a saved bridge config to its preregistered spec and reviewed launch mechanics."""

    config = _mapping(raw_config, name="root")
    _expect_saved(config.get("model_variant"), spec["model_variant"], name="model_variant")
    _expect_saved(config.get("phase"), "bridge", name="phase")
    _expect_saved(config.get("required_run_name"), spec["run_name"], name="required_run_name")

    repository_root = recipe_path.resolve().parents[3]
    reviewed_path = _resolved_saved_path(
        config.get("reviewed_profile_path"),
        repository_root=repository_root,
        name="reviewed_profile_path",
    )
    if reviewed_path != profile_path.resolve():
        raise SSMaxBridgeEvidenceError("Saved bridge config names a different reviewed profile")
    _expect_saved(
        config.get("reviewed_profile_sha256"),
        profile_sha256,
        name="reviewed_profile_sha256",
    )
    profile_relative = profile_path.resolve().relative_to(repository_root).as_posix()
    expected_command = [
        "src/scripts/train/Vision-Alignment.py",
        "train",
        str(spec["run_name"]),
        f"--profile={profile_relative}",
    ]
    _expect_saved(
        config.get("expected_launch_command"),
        expected_command,
        name="expected_launch_command",
    )

    parent = _validate_parent_contract(spec["parent"])
    artifacts = _mapping(config.get("artifacts"), name="artifacts")
    parent_fields = {
        "base_config_sha256": "config_sha256",
        "base_data_paths_sha256": "data_paths_sha256",
        "base_checkpoint_marker_sha256": "marker_sha256",
        "base_checkpoint_metadata_sha256": "dcp_metadata_sha256",
        "base_trainer_state_sha256": "trainer_state_sha256",
        "base_model_keyset_sha256": "model_keyset_sha256",
        "base_model_inventory_sha256": "model_inventory_sha256",
        "base_checkpoint_identity_sha256": "checkpoint_identity_sha256",
        "base_checkpoint_state_file_count": "state_file_count",
        "base_checkpoint_state_file_inventory_sha256": "state_file_inventory_sha256",
        "base_checkpoint_trainer_state_count": "trainer_state_count",
        "base_checkpoint_trainer_state_inventory_sha256": "trainer_state_inventory_sha256",
        "source_commit": "source_commit",
        "source_olmo_core_commit": "olmo_core_commit",
        "expected_lm_parameter_count": "parameter_count",
        "expected_lm_tensor_count": "tensor_count",
    }
    for config_name, parent_name in parent_fields.items():
        _expect_saved(
            artifacts.get(config_name),
            parent[parent_name],
            name=f"artifacts.{config_name}",
        )
    configured_parent = _resolved_saved_path(
        artifacts.get("base_checkpoint"),
        repository_root=repository_root,
        name="artifacts.base_checkpoint",
    )
    if configured_parent != Path(str(parent["checkpoint"])).expanduser().resolve():
        raise SSMaxBridgeEvidenceError("Saved bridge config names a different pretraining parent")

    metadata = _mapping(config.get("vision_alignment"), name="vision_alignment")
    for name, expected in (
        ("model_variant", spec["model_variant"]),
        ("phase", "bridge"),
        ("lineage_id", spec["run_name"]),
    ):
        _expect_saved(metadata.get(name), expected, name=f"vision_alignment.{name}")

    data = _mapping(config.get("data"), name="data")
    _expect_saved(data.get("pack_sequences"), False, name="data.pack_sequences")
    _expect_saved(
        data.get("allow_unpinned_synthetic_smoke"),
        False,
        name="data.allow_unpinned_synthetic_smoke",
    )
    mixture = _mapping(data.get("mixture"), name="data.mixture")
    _expect_saved(mixture.get("phase"), "bridge", name="data.mixture.phase")
    sequence_length = _positive_int(data.get("sequence_length"), name="saved sequence length")

    evaluation = _mapping(config.get("evaluation"), name="evaluation")
    for name, expected in (
        ("interval", 100),
        ("examples_per_source", spec["evaluation"]["examples_per_source"]),
        ("rank_batch_instances", spec["evaluation"]["rank_batch_instances"]),
        ("seed", spec["evaluation"]["pairing_seed"]),
        ("eval_on_startup", True),
        ("eval_on_finish", True),
    ):
        _expect_saved(evaluation.get(name), expected, name=f"evaluation.{name}")
    configured_validation = _resolved_saved_path(
        evaluation.get("validation_manifest_path"),
        repository_root=repository_root,
        name="evaluation.validation_manifest_path",
    )
    if configured_validation != validation_path.resolve():
        raise SSMaxBridgeEvidenceError("Saved bridge config names a different validation manifest")
    _expect_saved(
        evaluation.get("validation_manifest_sha256"),
        validation_sha256,
        name="evaluation.validation_manifest_sha256",
    )

    launch = _mapping(config.get("launch"), name="launch")
    _expect_saved(launch.get("cmd"), expected_command, name="launch.cmd")
    topology = spec["topology"]
    _expect_saved(launch.get("num_nodes"), topology["num_nodes"], name="launch.num_nodes")
    _expect_saved(launch.get("num_gpus"), topology["gpus_per_node"], name="launch.num_gpus")
    for name, expected in (
        ("workspace", "ai2/scaling-ladders"),
        ("budget", "ai2/oe-other"),
        ("clusters", ["ai2/holmes"]),
        ("priority", "urgent"),
        ("min_runtime", "8h"),
        ("shared_filesystem", True),
    ):
        _expect_saved(launch.get(name), expected, name=f"launch.{name}")
    git_value = _mapping(launch.get("git"), name="launch.git")
    _expect_saved(
        git_value.get("branch"),
        "rustin/vision-ssmax-molmofication",
        name="launch.git.branch",
    )
    git = _validate_git_identity(
        {name: git_value.get(name) for name in ("repo", "repo_url", "ref")}
    )

    train_module = _mapping(config.get("train_module"), name="train_module")
    dp_config = _mapping(train_module.get("dp_config"), name="train_module.dp_config")
    _expect_saved(
        dp_config.get("name"), topology["data_parallel"], name="train_module.dp_config.name"
    )
    for name, expected in (
        ("param_dtype", "bfloat16"),
        ("reduce_dtype", "float32"),
        ("wrapping_strategy", "blocks"),
        ("reduce_grads_in_fp32", True),
        ("accumulate_grads_in_fp32", True),
    ):
        _expect_saved(dp_config.get(name), expected, name=f"train_module.dp_config.{name}")
    _expect_saved(
        train_module.get("source_loss_mass_targets"),
        {"pixmo_caption": 0.7, "pixmo_transcript": 0.3},
        name="train_module.source_loss_mass_targets",
    )
    rank_instances = spec["evaluation"]["rank_batch_instances"]
    _expect_saved(
        train_module.get("rank_microbatch_size"),
        rank_instances * sequence_length,
        name="train_module.rank_microbatch_size",
    )
    _expect_saved(
        train_module.get("new_component_init_seed"),
        spec["evaluation"]["pairing_seed"],
        name="train_module.new_component_init_seed",
    )
    rank_global_instances = topology["world_size"] * rank_instances
    if BRIDGE_GLOBAL_BATCH_INSTANCES % rank_global_instances:
        raise SSMaxBridgeEvidenceError(
            "Saved bridge topology does not divide the fixed global instance batch"
        )
    _expect_saved(
        config.get("global_batch_size"),
        BRIDGE_GLOBAL_BATCH_INSTANCES * sequence_length,
        name="global_batch_size",
    )

    trainer = _mapping(config.get("trainer"), name="trainer")
    configured_output = _resolved_saved_path(
        trainer.get("save_folder"),
        repository_root=repository_root,
        name="trainer.save_folder",
    )
    if configured_output != checkpoint_root.resolve():
        raise SSMaxBridgeEvidenceError("Saved bridge config names a different checkpoint root")
    duration = _mapping(trainer.get("max_duration"), name="trainer.max_duration")
    _expect_saved(duration.get("value"), REQUIRED_STEPS[-1], name="trainer.max_duration.value")
    _expect_saved(duration.get("unit"), "steps", name="trainer.max_duration.unit")
    _expect_saved(trainer.get("no_checkpoints"), False, name="trainer.no_checkpoints")
    callbacks = _mapping(trainer.get("callbacks"), name="trainer.callbacks")
    checkpointer = _mapping(callbacks.get("checkpointer"), name="trainer.callbacks.checkpointer")
    for name, expected in (
        ("save_interval", None),
        ("ephemeral_save_interval", 50),
        ("pre_train_checkpoint", True),
        ("fixed_steps", list(REQUIRED_STEPS[1:])),
        ("max_checkpoints", len(REQUIRED_STEPS)),
        ("enabled", True),
    ):
        _expect_saved(
            checkpointer.get(name), expected, name=f"trainer.callbacks.checkpointer.{name}"
        )
    ledger = _mapping(callbacks.get("ssmax_health_ledger"), name="SSMax health ledger callback")
    for name, expected in (
        ("model_variant", spec["model_variant"]),
        ("phase", "bridge"),
        ("run_name", spec["run_name"]),
        ("enabled", True),
    ):
        _expect_saved(
            ledger.get(name), expected, name=f"trainer.callbacks.ssmax_health_ledger.{name}"
        )
    return dict(git)


def _validate_parent_load_receipt_payload(
    payload: Any, *, parent: Mapping[str, Any], model_variant: str
) -> Mapping[str, Any]:
    receipt = _mapping(payload, name="parent-load receipt")
    expected_fields = {
        "format": "vision_alignment_ssmax_parent_load_receipt",
        "version": 1,
        "model_variant": model_variant,
        "parent_config_sha256": parent["config_sha256"],
        "parent_data_paths_sha256": parent["data_paths_sha256"],
        "parent_checkpoint_marker_sha256": parent["marker_sha256"],
        "parent_dcp_metadata_sha256": parent["dcp_metadata_sha256"],
        "parent_trainer_state_sha256": parent["trainer_state_sha256"],
        "parent_source_commit": parent["source_commit"],
        "parent_olmo_core_commit": parent["olmo_core_commit"],
        "parent_model_keyset_sha256": parent["model_keyset_sha256"],
        "parent_model_inventory_sha256": parent["model_inventory_sha256"],
        "parent_checkpoint_identity_sha256": parent["checkpoint_identity_sha256"],
        "parent_state_file_count": parent["state_file_count"],
        "parent_state_file_inventory_sha256": parent["state_file_inventory_sha256"],
        "parent_trainer_state_count": parent["trainer_state_count"],
        "parent_trainer_state_inventory_sha256": parent["trainer_state_inventory_sha256"],
        "loaded_parameter_numel": parent["parameter_count"],
        "loaded_model_tensor_count": parent["tensor_count"],
        "loaded_parameter_count": parent["tensor_count"],
    }
    for name, expected in expected_fields.items():
        if receipt.get(name) != expected:
            raise SSMaxBridgeEvidenceError(f"Parent-load receipt {name} differs")
    if (
        Path(str(receipt.get("parent_checkpoint"))).expanduser().resolve()
        != Path(str(parent["checkpoint"])).expanduser().resolve()
    ):
        raise SSMaxBridgeEvidenceError("Parent-load receipt names a different parent checkpoint")
    expected_state_dir = _checkpoint_state_dir(Path(str(parent["checkpoint"]))).resolve()
    if Path(str(receipt.get("checkpoint_dir"))).expanduser().resolve() != expected_state_dir:
        raise SSMaxBridgeEvidenceError("Parent-load receipt names a different DCP state directory")
    if receipt.get("loaded_tensor_dtype_counts") != {"torch.float32": parent["tensor_count"]}:
        raise SSMaxBridgeEvidenceError("Parent-load receipt dtype inventory differs")
    if receipt.get("loaded_tensor_layout_counts") != {"torch.strided": parent["tensor_count"]}:
        raise SSMaxBridgeEvidenceError("Parent-load receipt layout inventory differs")
    loaded = receipt.get("loaded_model_keys")
    parameters = receipt.get("loaded_parameter_keys")
    if (
        not isinstance(loaded, list)
        or not isinstance(parameters, list)
        or len(loaded) != parent["tensor_count"]
        or len(set(loaded)) != len(loaded)
        or loaded != sorted(loaded)
        or parameters != loaded
        or any(not isinstance(key, str) or not key.startswith("lm.") for key in loaded)
    ):
        raise SSMaxBridgeEvidenceError("Parent-load receipt loaded key inventory differs")
    source_keys = sorted(key.removeprefix("lm.") for key in loaded)
    if canonical_sha256(source_keys) != parent["model_keyset_sha256"]:
        raise SSMaxBridgeEvidenceError("Parent-load receipt keyset SHA-256 differs")
    fingerprint = _sha256(receipt.get("fingerprint"), name="parent-load receipt fingerprint")
    if (
        canonical_sha256({key: item for key, item in receipt.items() if key != "fingerprint"})
        != fingerprint
    ):
        raise SSMaxBridgeEvidenceError("Parent-load receipt fingerprint differs")
    return receipt


def build_manifest(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    pairing_references: Mapping[str, Mapping[str, str]],
    created_at: str,
    hash_workers: int = 8,
) -> dict[str, Any]:
    """Finalize ``spec`` by hashing all seven checkpoints and fixed pairing artifacts."""

    _exact_fields(spec, _SPEC_FIELDS, name="SSMax manifest spec")
    if spec.get("format") != MANIFEST_SPEC_FORMAT or spec.get("version") != SCHEMA_VERSION:
        raise SSMaxBridgeEvidenceError("SSMax manifest spec identity is incompatible")
    _validate_common_manifest_fields(spec, finalized=False)
    _timestamp(created_at, name="manifest created_at")
    checkpoint_root = Path(str(spec["checkpoint_root"])).expanduser().resolve()
    checkpoints = {
        str(step): checkpoint_identity(checkpoint_root / f"step{step}", workers=hash_workers)
        for step in REQUIRED_STEPS
    }
    config_hashes = {checkpoint["config_sha256"] for checkpoint in checkpoints.values()}
    trainer_counts = {checkpoint["trainer_state_count"] for checkpoint in checkpoints.values()}
    if len(config_hashes) != 1:
        raise SSMaxBridgeEvidenceError("Saved bridge checkpoints do not share one config")
    if trainer_counts != {spec["topology"]["world_size"]}:
        raise SSMaxBridgeEvidenceError("Trainer-state count differs from the manifest world")

    profile_path = Path(str(spec["training_profile"])).expanduser().resolve()
    recipe_path = Path(str(spec["recipe"])).expanduser().resolve()
    validation_path = Path(str(spec["validation"])).expanduser().resolve()
    attention_probe_path = Path(str(spec["attention_probe"])).expanduser().resolve()
    profile_reference = artifact_reference(profile_path)
    recipe_reference = artifact_reference(recipe_path)
    validation_reference = artifact_reference(validation_path)
    attention_probe_reference = artifact_reference(attention_probe_path)
    raw_config = load_json(checkpoint_root / "step0" / "config.json")
    git = _validate_saved_bridge_config(
        spec,
        raw_config,
        checkpoint_root=checkpoint_root,
        profile_path=profile_path,
        profile_sha256=profile_reference["sha256"],
        recipe_path=recipe_path,
        validation_path=validation_path,
        validation_sha256=validation_reference["sha256"],
    )
    _validate_saved_git_checkout(git, recipe_path=recipe_path, profile_path=profile_path)
    repository_root = recipe_path.resolve().parents[3]
    manifest_spec = _manifest_spec_source_reference(
        spec_path,
        spec,
        git=git,
        repository_root=repository_root,
    )
    producers = _producer_source_references(git, repository_root=repository_root)
    parent = _validate_parent_contract(spec["parent"])
    parent_load_path = checkpoint_root / "bridge-parent-load-receipt.json"
    parent_load_reference = artifact_reference(parent_load_path)
    _validate_parent_load_receipt_payload(
        load_json(parent_load_path),
        parent=parent,
        model_variant=str(spec["model_variant"]),
    )
    pairings: dict[str, dict[str, str]] = {}
    if set(pairing_references) != set(SOURCES):
        raise SSMaxBridgeEvidenceError("Finalization requires both fixed source pairings")
    pairing_paths = _exact_fields(
        spec["pairing_paths"], frozenset(SOURCES), name="manifest spec pairing paths"
    )
    for source in SOURCES:
        path = validate_artifact_reference(pairing_references[source], name=f"{source} pairing")
        if path != Path(str(pairing_paths[source])).expanduser().resolve():
            raise SSMaxBridgeEvidenceError(f"{source} pairing path differs from its manifest spec")
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise SSMaxBridgeEvidenceError(f"{source} pairing payload must be an object")
        validate_matched_wrong_image_pairing(
            payload,
            recipient_count=spec["evaluation"]["examples_per_source"],
            seed=spec["evaluation"]["pairing_seed"],
            epoch=0,
        )
        if matched_wrong_image_pairing_sha256(payload) != pairing_references[source]["sha256"]:
            raise SSMaxBridgeEvidenceError(f"{source} pairing canonical SHA-256 differs")
        pairings[source] = dict(pairing_references[source])

    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": SCHEMA_VERSION,
        "created_at": created_at,
        "pair_id": spec["pair_id"],
        "arm": spec["arm"],
        "model_variant": spec["model_variant"],
        "run_name": spec["run_name"],
        "parent": dict(spec["parent"]),
        "parent_load_receipt": parent_load_reference,
        "git": git,
        "manifest_spec": manifest_spec,
        "producers": producers,
        "training_profile": profile_reference,
        "recipe": recipe_reference,
        "validation": validation_reference,
        "attention_probe": attention_probe_reference,
        "pairings": pairings,
        "evaluation": dict(spec["evaluation"]),
        "topology": dict(spec["topology"]),
        "policy": dict(spec["policy"]),
        "checkpoints": checkpoints,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest, verify_live=True, hash_workers=hash_workers)
    return manifest


def validate_manifest(
    value: Any, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Validate a finalized per-arm manifest and every referenced live artifact."""

    manifest = _exact_fields(value, _MANIFEST_FIELDS, name="SSMax bridge manifest")
    if manifest["format"] != MANIFEST_FORMAT or manifest["version"] != SCHEMA_VERSION:
        raise SSMaxBridgeEvidenceError("SSMax bridge manifest identity is incompatible")
    _timestamp(manifest["created_at"], name="manifest created_at")
    _validate_common_manifest_fields(manifest, finalized=verify_live)
    git = _validate_git_identity(manifest["git"])
    manifest_spec = _validate_manifest_spec_source_reference(
        manifest["manifest_spec"],
        pair_id=str(manifest["pair_id"]),
        model_variant=str(manifest["model_variant"]),
        git=git,
    )
    producers = _validate_producer_source_references(manifest["producers"], git=git)
    for name in (
        "parent_load_receipt",
        "training_profile",
        "recipe",
        "validation",
        "attention_probe",
    ):
        _validate_artifact_reference_shape(manifest[name], name=name.replace("_", " "))
    content_hash = _sha256(manifest["content_sha256"], name="manifest content SHA-256")
    if (
        canonical_sha256({key: item for key, item in manifest.items() if key != "content_sha256"})
        != content_hash
    ):
        raise SSMaxBridgeEvidenceError("Manifest content SHA-256 differs")
    checkpoints = _exact_fields(
        manifest["checkpoints"],
        frozenset(str(step) for step in REQUIRED_STEPS),
        name="manifest checkpoints",
    )
    for step in REQUIRED_STEPS:
        validate_checkpoint_reference(
            checkpoints[str(step)],
            expected_step=step,
            workers=hash_workers,
            verify_live=verify_live,
        )
    config_hashes = {checkpoint["config_sha256"] for checkpoint in checkpoints.values()}
    if len(config_hashes) != 1:
        raise SSMaxBridgeEvidenceError("Manifest checkpoints do not share one saved config")
    if verify_live:
        checkpoint_root = Path(str(checkpoints["0"]["path"])).resolve().parent
        profile_path = validate_artifact_reference(
            manifest["training_profile"], name="training profile"
        )
        recipe_path = validate_artifact_reference(manifest["recipe"], name="recipe")
        validation_path = validate_artifact_reference(manifest["validation"], name="validation")
        git = _validate_saved_bridge_config(
            manifest,
            load_json(checkpoint_root / "step0" / "config.json"),
            checkpoint_root=checkpoint_root,
            profile_path=profile_path,
            profile_sha256=str(manifest["training_profile"]["sha256"]),
            recipe_path=recipe_path,
            validation_path=validation_path,
            validation_sha256=str(manifest["validation"]["sha256"]),
        )
        if git != manifest["git"]:
            raise SSMaxBridgeEvidenceError("Manifest git identity differs from the saved config")
        _validate_saved_git_checkout(
            git,
            recipe_path=recipe_path,
            profile_path=profile_path,
        )
        repository_root = recipe_path.resolve().parents[3]
        spec_path = repository_root / manifest_spec["repo_relative_path"]
        spec = load_manifest_spec(spec_path)
        actual_manifest_spec = _manifest_spec_source_reference(
            spec_path,
            spec,
            git=git,
            repository_root=repository_root,
        )
        if actual_manifest_spec != manifest_spec:
            raise SSMaxBridgeEvidenceError(
                "Manifest specification source differs from the saved git checkout"
            )
        _validate_finalized_manifest_against_spec(
            manifest,
            spec,
            checkpoint_root=checkpoint_root,
        )
        if _producer_source_references(git, repository_root=repository_root) != producers:
            raise SSMaxBridgeEvidenceError(
                "Manifest evidence producer sources differ from the saved git checkout"
            )
        parent_load_path = validate_artifact_reference(
            manifest["parent_load_receipt"], name="parent load receipt"
        )
        if parent_load_path != checkpoint_root / "bridge-parent-load-receipt.json":
            raise SSMaxBridgeEvidenceError("Manifest parent-load receipt is outside the run root")
        _validate_parent_load_receipt_payload(
            load_json(parent_load_path),
            parent=manifest["parent"],
            model_variant=str(manifest["model_variant"]),
        )
    pairings = _exact_fields(manifest["pairings"], frozenset(SOURCES), name="manifest pairings")
    for source in SOURCES:
        if verify_live:
            path = validate_artifact_reference(pairings[source], name=f"{source} pairing")
            payload = load_json(path)
            if not isinstance(payload, Mapping):
                raise SSMaxBridgeEvidenceError(f"{source} pairing must contain an object")
            validate_matched_wrong_image_pairing(
                payload,
                recipient_count=manifest["evaluation"]["examples_per_source"],
                seed=manifest["evaluation"]["pairing_seed"],
                epoch=0,
            )
            if matched_wrong_image_pairing_sha256(payload) != pairings[source]["sha256"]:
                raise SSMaxBridgeEvidenceError(f"{source} pairing canonical SHA-256 differs")
        else:
            _exact_fields(pairings[source], _ARTIFACT_REF_FIELDS, name=f"{source} pairing")
            _sha256(pairings[source]["sha256"], name=f"{source} pairing SHA-256")
    return manifest


def load_manifest(
    path: Path, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Load a finalized manifest and verify that its own raw-byte reference can be pinned."""

    return validate_manifest(load_json(path), verify_live=verify_live, hash_workers=hash_workers)


@dataclass(frozen=True)
class GenericDCPLoadInventory:
    """Exact correspondence between generic model state and native DCP model tensors."""

    checkpoint_key_count: int
    model_tensor_count: int
    model_parameter_tensor_count: int
    model_buffer_tensor_count: int
    model_keyset_sha256: str
    model_inventory_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible completed strict-load inventory."""

        payload = {
            "complete": True,
            "strict": True,
            "checkpoint_key_count": self.checkpoint_key_count,
            "model_tensor_count": self.model_tensor_count,
            "model_parameter_tensor_count": self.model_parameter_tensor_count,
            "model_buffer_tensor_count": self.model_buffer_tensor_count,
            "model_keyset_sha256": self.model_keyset_sha256,
            "model_inventory_sha256": self.model_inventory_sha256,
        }
        payload["sha256"] = canonical_sha256(payload)
        return payload


def verify_generic_dcp_load_inventory(
    *,
    metadata: Metadata,
    state_dict_to_load: Mapping[str, Any],
    parameter_names: Sequence[str],
    buffer_names: Sequence[str],
) -> GenericDCPLoadInventory:
    """Prove exact key/shape/dtype coverage before a generic multimodal DCP load.

    ``state_dict_to_load`` is the value returned by the generic train module with
    ``optim=False``.  Every checkpoint ``model.*`` tensor must map exactly once and optimizer or
    trainer entries are deliberately ignored.
    """

    if set(state_dict_to_load) != {"model"} or not isinstance(state_dict_to_load["model"], Mapping):
        raise SSMaxBridgeEvidenceError("Generic eval load must expose exactly one model mapping")
    current = state_dict_to_load["model"]
    checkpoint = {
        key.removeprefix("model."): value
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("model.")
    }
    if set(current) != set(checkpoint):
        raise SSMaxBridgeEvidenceError(
            "Generic DCP model key inventory differs; "
            f"missing={sorted(set(current) - set(checkpoint))[:10]}, "
            f"unexpected={sorted(set(checkpoint) - set(current))[:10]}"
        )
    parameters = set(parameter_names)
    buffers = set(buffer_names)
    if parameters & buffers or parameters | buffers != set(current):
        raise SSMaxBridgeEvidenceError(
            "Runtime parameter/buffer inventory does not exactly partition model state"
        )
    records: list[dict[str, Any]] = []
    for key in sorted(current):
        target = current[key]
        source = checkpoint[key]
        if not isinstance(target, torch.Tensor) or not isinstance(source, TensorStorageMetadata):
            raise SSMaxBridgeEvidenceError(f"Generic model entry {key!r} is not tensor metadata")
        if (
            tuple(target.shape) != tuple(source.size)
            or target.dtype != source.properties.dtype
            or target.layout != source.properties.layout
        ):
            raise SSMaxBridgeEvidenceError(
                f"Generic DCP tensor metadata differs for model.{key}: "
                f"runtime={tuple(target.shape)}/{target.dtype}/{target.layout}, "
                f"checkpoint={tuple(source.size)}/{source.properties.dtype}/"
                f"{source.properties.layout}"
            )
        records.append(
            {
                "key": key,
                "kind": "parameter" if key in parameters else "buffer",
                "shape": list(target.shape),
                "dtype": str(target.dtype),
                "layout": str(target.layout),
            }
        )
    return GenericDCPLoadInventory(
        checkpoint_key_count=len(metadata.state_dict_metadata),
        model_tensor_count=len(records),
        model_parameter_tensor_count=len(parameters),
        model_buffer_tensor_count=len(buffers),
        model_keyset_sha256=canonical_sha256(sorted(current)),
        model_inventory_sha256=canonical_sha256(records),
    )


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, samples: int
) -> dict[str, float]:
    """Return a deterministic percentile bootstrap interval for one paired statistic."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise SSMaxBridgeEvidenceError("Bootstrap input must be a non-empty finite vector")
    if seed < 0 or samples <= 0:
        raise ValueError("Bootstrap seed must be non-negative and samples must be positive")
    rng = np.random.RandomState(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = min(samples, 2048)
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        indices = rng.randint(0, len(array), size=(end - start, len(array)))
        means[start:end] = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"confidence": 0.95, "low": float(low), "high": float(high)}


def summarize_paired_values(values: Sequence[float], *, seed: int, samples: int) -> dict[str, Any]:
    """Summarize paired signed values with bootstrap CI, wins, losses, and ties."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise SSMaxBridgeEvidenceError("Paired values must be a non-empty finite vector")
    return {
        "examples": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "win_rate": float((array > 0).mean()),
        "loss_rate": float((array < 0).mean()),
        "tie_rate": float((array == 0).mean()),
        "mean_bootstrap_ci": bootstrap_mean_interval(array, seed=seed, samples=samples),
    }


def aggregate_matched_records(
    records: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    """Aggregate fixed correct/wrong-image rows over the locked response windows."""

    if not records:
        raise SSMaxBridgeEvidenceError("Cannot aggregate an empty matched-wrong record set")
    metrics: dict[str, Any] = {}
    for index, window in enumerate(WINDOWS):
        correct = np.asarray([row["correct_ce"][window] for row in records], dtype=np.float64)
        wrong = np.asarray([row["wrong_ce"][window] for row in records], dtype=np.float64)
        gaps = wrong - correct
        summary = summarize_paired_values(
            gaps,
            seed=bootstrap_seed + 10_000 * index,
            samples=bootstrap_samples,
        )
        metrics[window] = {
            "examples": len(records),
            "correct_ce_mean": float(correct.mean()),
            "wrong_ce_mean": float(wrong.mean()),
            "gap_wrong_minus_correct_mean": summary["mean"],
            "gap_median": summary["median"],
            "win_rate": summary["win_rate"],
            "tie_rate": summary["tie_rate"],
            "mean_gap_bootstrap_ci": summary["mean_bootstrap_ci"],
            "win_rate_bootstrap_ci": bootstrap_mean_interval(
                (gaps > 0).astype(np.float64),
                seed=bootstrap_seed + 10_000 * index + 1,
                samples=bootstrap_samples,
            ),
        }
    return metrics


def manifest_reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return a dual raw-byte and semantic identity for a finalized manifest."""

    reference = artifact_reference(path)
    reference["content_sha256"] = str(manifest["content_sha256"])
    return reference


def _validate_receipt_reference(
    value: Any,
    *,
    expected_format: str,
    manifest: Mapping[str, Any],
    step: int,
    name: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = validate_artifact_reference(value, name=name)
    fields = {
        MATCHED_STATE_RECEIPT_FORMAT: _MATCHED_STATE_RECEIPT_FIELDS,
        HEALTH_RECEIPT_FORMAT: _HEALTH_RECEIPT_FIELDS,
    }.get(expected_format)
    if fields is None:
        raise ValueError(f"Unsupported bridge receipt format {expected_format!r}")
    payload = _exact_fields(load_json(path), fields, name=f"{name} receipt")
    if (
        payload.get("format") != expected_format
        or payload.get("version") != SCHEMA_VERSION
        or payload.get("status") not in {"passed", "failed"}
        or payload.get("step") != step
        or payload.get("pair_id") != manifest["pair_id"]
        or payload.get("arm") != manifest["arm"]
        or payload.get("model_variant") != manifest["model_variant"]
    ):
        raise SSMaxBridgeEvidenceError(f"{name} receipt identity is incompatible")
    _timestamp(payload.get("created_at"), name=f"{name} created_at")
    manifest_identity = _exact_fields(
        payload.get("manifest"),
        frozenset({"path", "sha256", "content_sha256"}),
        name=f"{name} manifest reference",
    )
    if manifest_identity.get("content_sha256") != manifest["content_sha256"]:
        raise SSMaxBridgeEvidenceError(f"{name} names a different manifest")
    if not isinstance(manifest_identity["path"], str) or not manifest_identity["path"]:
        raise SSMaxBridgeEvidenceError(f"{name} manifest path is malformed")
    _sha256(manifest_identity["sha256"], name=f"{name} manifest SHA-256")
    candidate = payload.get("checkpoint")
    if candidate != manifest["checkpoints"][str(step)]:
        raise SSMaxBridgeEvidenceError(f"{name} names a different step{step} checkpoint")
    expected_producers = _validate_producer_source_references(
        manifest["producers"], git=_validate_git_identity(manifest["git"])
    )
    if expected_format == MATCHED_STATE_RECEIPT_FORMAT:
        evaluator = _exact_fields(
            payload["evaluator"],
            _PRODUCER_SOURCE_REF_FIELDS,
            name=f"{name} evaluator source",
        )
        if dict(evaluator) != dict(expected_producers[MATCHED_STATE_PRODUCER]):
            raise SSMaxBridgeEvidenceError(f"{name} evaluator source identity differs")
    else:
        evidence = _exact_fields(
            payload["evidence"],
            frozenset({"recipe", "producer", "rank_state_inventory"}),
            name=f"{name} evidence",
        )
        recipe = _exact_fields(
            evidence["recipe"],
            _ARTIFACT_REF_FIELDS,
            name=f"{name} recipe reference",
        )
        producer = _exact_fields(
            evidence["producer"],
            _PRODUCER_SOURCE_REF_FIELDS,
            name=f"{name} producer source",
        )
        if dict(recipe) != dict(manifest["recipe"]):
            raise SSMaxBridgeEvidenceError(f"{name} recipe identity differs")
        if dict(producer) != dict(expected_producers[HEALTH_PRODUCER]):
            raise SSMaxBridgeEvidenceError(f"{name} producer source identity differs")
    content_sha = _sha256(payload.get("content_sha256"), name=f"{name} content SHA-256")
    if (
        canonical_sha256({key: item for key, item in payload.items() if key != "content_sha256"})
        != content_sha
    ):
        raise SSMaxBridgeEvidenceError(f"{name} content SHA-256 differs")
    return path, payload


def _receipt_records(receipt: Mapping[str, Any], source: str) -> list[Mapping[str, Any]]:
    result = receipt.get("results", {}).get(source)
    if not isinstance(result, Mapping) or not isinstance(result.get("per_example"), list):
        raise SSMaxBridgeEvidenceError(f"Matched receipt omits {source} per-example rows")
    records = result["per_example"]
    expected_pairing = receipt["pairings"][source]["sha256"]
    if result.get("pairing_sha256") != expected_pairing:
        raise SSMaxBridgeEvidenceError(f"Matched receipt {source} result uses another pairing")
    positions = [row.get("pairing_position") for row in records]
    if positions != list(range(len(records))):
        raise SSMaxBridgeEvidenceError(
            f"Matched receipt {source} rows are not complete and ordered"
        )
    return records


def _paired_change(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    source: str,
    window: str,
    field: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    left = _receipt_records(reference, source)
    right = _receipt_records(candidate, source)
    identities = ("pairing_position", "recipient_index", "donor_index", "response_tokens")
    if len(left) != len(right) or any(
        any(lrow.get(name) != rrow.get(name) for name in identities)
        for lrow, rrow in zip(left, right, strict=True)
    ):
        raise SSMaxBridgeEvidenceError("Cross-step matched rows are not exactly paired")
    if field == "gap":
        values = [
            float(rrow["ce_gap_wrong_minus_correct"][window])
            - float(lrow["ce_gap_wrong_minus_correct"][window])
            for lrow, rrow in zip(left, right, strict=True)
        ]
    elif field == "correct_ce":
        # Positive means the candidate has lower (better) correct-image CE.
        values = [
            float(lrow["correct_ce"][window]) - float(rrow["correct_ce"][window])
            for lrow, rrow in zip(left, right, strict=True)
        ]
    else:
        raise ValueError(f"Unknown paired-change field {field!r}")
    return summarize_paired_values(values, seed=seed, samples=samples)


def _paired_policy_margin(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    source: str,
    window: str,
    policy: str,
    threshold: float,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Bootstrap a paired per-row policy margin; positive values satisfy the policy."""

    left = _receipt_records(reference, source)
    right = _receipt_records(candidate, source)
    identities = ("pairing_position", "recipient_index", "donor_index", "response_tokens")
    if len(left) != len(right) or any(
        any(lrow.get(name) != rrow.get(name) for name in identities)
        for lrow, rrow in zip(left, right, strict=True)
    ):
        raise SSMaxBridgeEvidenceError("Cross-step policy rows are not exactly paired")
    if policy == "gap_retention":
        values = [
            float(rrow["ce_gap_wrong_minus_correct"][window])
            - threshold * float(lrow["ce_gap_wrong_minus_correct"][window])
            for lrow, rrow in zip(left, right, strict=True)
        ]
    elif policy == "correct_ce_noninferiority":
        values = [
            (1.0 + threshold) * float(lrow["correct_ce"][window])
            - float(rrow["correct_ce"][window])
            for lrow, rrow in zip(left, right, strict=True)
        ]
    else:
        raise ValueError(f"Unknown paired policy margin {policy!r}")
    return summarize_paired_values(values, seed=seed, samples=samples)


def build_promotion_report(
    *,
    manifest_path: Path,
    matched_receipts: Mapping[int, Mapping[str, str]],
    health_receipts: Mapping[int, Mapping[str, str]],
    created_at: str,
    verify_live_manifest: bool = True,
) -> dict[str, Any]:
    """Validate all trajectory evidence and build a pass/reject bridge promotion report."""

    manifest = load_manifest(manifest_path, verify_live=verify_live_manifest)
    _timestamp(created_at, name="promotion created_at")
    if set(matched_receipts) != set(REQUIRED_STEPS) or set(health_receipts) != set(REQUIRED_STEPS):
        raise SSMaxBridgeEvidenceError(
            "Promotion requires matched/state and health receipts at all steps"
        )
    matched: dict[int, Mapping[str, Any]] = {}
    health: dict[int, Mapping[str, Any]] = {}
    refs: dict[str, Any] = {}
    for step in REQUIRED_STEPS:
        _, matched_payload = _validate_receipt_reference(
            matched_receipts[step],
            expected_format=MATCHED_STATE_RECEIPT_FORMAT,
            manifest=manifest,
            step=step,
            name=f"step{step} matched/state",
        )
        _, health_payload = _validate_receipt_reference(
            health_receipts[step],
            expected_format=HEALTH_RECEIPT_FORMAT,
            manifest=manifest,
            step=step,
            name=f"step{step} health",
        )
        matched[step] = matched_payload
        health[step] = health_payload
        refs[str(step)] = {
            "matched_state": dict(matched_receipts[step]),
            "health": dict(health_receipts[step]),
        }

    deviations: list[dict[str, Any]] = []
    policy = manifest["policy"]
    evaluation = manifest["evaluation"]
    samples = int(evaluation["bootstrap_samples"])
    seed = int(evaluation["bootstrap_seed"])
    pairing_pins = {source: manifest["pairings"][source]["sha256"] for source in SOURCES}
    for step, receipt in matched.items():
        if receipt.get("status") != "passed":
            deviations.append({"kind": "matched_state_receipt_status", "step": step})
        if receipt.get("pairings") != manifest["pairings"]:
            raise SSMaxBridgeEvidenceError(f"step{step} matched receipt changes fixed pairings")
        strict_load = receipt.get("strict_generic_dcp_load")
        frozen = receipt.get("frozen_state")
        if not isinstance(strict_load, Mapping) or (
            strict_load.get("complete") is not True
            or strict_load.get("strict") is not True
            or strict_load.get("load_completed") is not True
        ):
            raise SSMaxBridgeEvidenceError(f"step{step} lacks a completed strict generic DCP load")
        if not isinstance(frozen, Mapping) or (
            frozen.get("complete") is not True or frozen.get("mismatch_count") != 0
        ):
            deviations.append({"kind": "frozen_state", "step": step})
        attention = receipt.get("attention_diagnostics")
        if (
            not isinstance(attention, Mapping)
            or attention.get("format") != "ssmax_attention_diagnostics"
            or attention.get("checkpoint") != manifest["checkpoints"][str(step)]
            or attention.get("protocol", {}).get("manifest_sha256")
            != manifest["attention_probe"]["sha256"]
        ):
            raise SSMaxBridgeEvidenceError(
                f"step{step} lacks manifest-bound SSMax attention diagnostics"
            )
        try:
            validate_ssmax_attention_report(attention, label=f"step{step} attention diagnostics")
        except ValueError as error:
            raise SSMaxBridgeEvidenceError(str(error)) from error
        for source in SOURCES:
            result = receipt.get("results", {}).get(source)
            if (
                not isinstance(result, Mapping)
                or result.get("pairing_sha256") != pairing_pins[source]
            ):
                raise SSMaxBridgeEvidenceError(f"step{step} {source} result differs from pairing")
            records = _receipt_records(receipt, source)
            if len(records) != evaluation["examples_per_source"]:
                raise SSMaxBridgeEvidenceError(f"step{step} {source} example count differs")
            metrics = result.get("metrics")
            if not isinstance(metrics, Mapping) or set(metrics) != set(WINDOWS):
                raise SSMaxBridgeEvidenceError(f"step{step} {source} windows differ")
            recomputed = aggregate_matched_records(
                records,
                bootstrap_seed=seed + SOURCES.index(source) * 1_000_000,
                bootstrap_samples=samples,
            )
            if recomputed != metrics:
                raise SSMaxBridgeEvidenceError(f"step{step} {source} aggregate differs from rows")

    for step, receipt in health.items():
        if receipt.get("status") != "passed":
            deviations.append({"kind": "health_receipt_status", "step": step})
        loader = receipt.get("loader")
        sources = receipt.get("sources")
        if not isinstance(loader, Mapping) or not isinstance(sources, Mapping):
            raise SSMaxBridgeEvidenceError(f"step{step} health receipt is incomplete")
        if (
            loader.get("batches_replayed") != step
            or loader.get("rank_states_global_step") != step
            or loader.get("rank_states_batches_processed") != step
            or loader.get("checkpoint_final_state_sha256")
            != loader.get("replayed_final_state_sha256")
            or loader.get("total_data_errors") > policy["maximum_data_errors"]
            or loader.get("rank_state_inventory_sha256")
            != receipt["checkpoint"]["trainer_state_inventory_sha256"]
            or loader.get("rank_state_count") != receipt["checkpoint"]["trainer_state_count"]
        ):
            deviations.append({"kind": "data_health", "step": step})
        evidence = receipt.get("evidence")
        if not isinstance(evidence, Mapping):
            raise SSMaxBridgeEvidenceError(f"step{step} health evidence is incomplete")
        rank_inventory = evidence.get("rank_state_inventory")
        if (
            not isinstance(rank_inventory, list)
            or len(rank_inventory) != receipt["checkpoint"]["trainer_state_count"]
            or canonical_sha256(rank_inventory)
            != receipt["checkpoint"]["trainer_state_inventory_sha256"]
            or loader.get("rank_state_inventory_sha256") != canonical_sha256(rank_inventory)
        ):
            raise SSMaxBridgeEvidenceError(
                f"step{step} trainer-state bytes differ from the checkpoint identity"
            )
        for rank, item in enumerate(rank_inventory):
            if (
                not isinstance(item, Mapping)
                or item.get("rank") != rank
                or item.get("path") != f"train/rank{rank}.pt"
                or type(item.get("size")) is not int
                or item["size"] <= 0
            ):
                raise SSMaxBridgeEvidenceError(
                    f"step{step} trainer rank{rank} inventory is malformed"
                )
            _sha256(item.get("sha256"), name=f"step{step} trainer rank{rank} SHA-256")
        ledger_summary = receipt.get("health_ledger")
        if not isinstance(ledger_summary, Mapping) or set(ledger_summary) != {
            "rank_ledgers",
            "event_chain_sha256",
            "counters",
        }:
            raise SSMaxBridgeEvidenceError(f"step{step} health ledger summary fields differ")
        rank_ledgers = ledger_summary["rank_ledgers"]
        if (
            not isinstance(rank_ledgers, list)
            or len(rank_ledgers) != manifest["topology"]["world_size"]
        ):
            raise SSMaxBridgeEvidenceError(f"step{step} health ledger omits trainer ranks")
        try:
            validated_ledgers = [
                validate_ssmax_health_ledger_state(
                    ledger,
                    expected_model_variant=str(manifest["model_variant"]),
                    expected_phase="bridge",
                    expected_run_name=str(manifest["run_name"]),
                    expected_step=step,
                )
                for ledger in rank_ledgers
            ]
        except SSMaxHealthLedgerError as error:
            raise SSMaxBridgeEvidenceError(
                f"step{step} checkpoint-native health ledger is invalid: {error}"
            ) from error
        event_chain = validated_ledgers[0]["event_chain_sha256"]
        expected_counters = {
            "data_errors": sum(int(ledger["data_errors"]) for ledger in validated_ledgers),
            "optimizer_guard_skips": int(validated_ledgers[0]["optimizer_guard_skips"]),
            "nonfinite_losses": int(validated_ledgers[0]["nonfinite_losses"]),
            "nonfinite_gradients": int(validated_ledgers[0]["nonfinite_gradients"]),
        }
        if (
            any(ledger["event_chain_sha256"] != event_chain for ledger in validated_ledgers)
            or ledger_summary["event_chain_sha256"] != event_chain
            or ledger_summary["counters"] != expected_counters
            or expected_counters["data_errors"] != loader.get("total_data_errors")
        ):
            raise SSMaxBridgeEvidenceError(f"step{step} health ledger aggregate differs")
        for counter in (
            "optimizer_guard_skips",
            "nonfinite_losses",
            "nonfinite_gradients",
        ):
            if expected_counters[counter] != 0:
                deviations.append(
                    {
                        "kind": "run_health_counter",
                        "step": step,
                        "counter": counter,
                        "observed": expected_counters[counter],
                    }
                )
        shares = []
        for source in SOURCES:
            source_metrics = sources.get(source)
            if not isinstance(source_metrics, Mapping):
                raise SSMaxBridgeEvidenceError(f"step{step} health omits {source}")
            share = _finite(source_metrics.get("active_loss_mass_share"), name="loss-mass share")
            target = _finite(source_metrics.get("target_loss_mass"), name="loss-mass target")
            shares.append(share)
            if abs(share - target) > policy["loss_mass_share_tolerance"]:
                deviations.append(
                    {
                        "kind": "loss_mass",
                        "step": step,
                        "source": source,
                        "observed": share,
                        "target": target,
                    }
                )
        if step > 0 and not math.isclose(sum(shares), 1.0, rel_tol=0, abs_tol=1e-12):
            raise SSMaxBridgeEvidenceError(f"step{step} loss-mass shares do not sum to one")

    for step in policy["positive_gap_ci_steps"]:
        for source in SOURCES:
            for window in WINDOWS:
                metric = matched[step]["results"][source]["metrics"][window]
                if _finite(metric["mean_gap_bootstrap_ci"]["low"], name="gap CI low") <= 0:
                    deviations.append(
                        {
                            "kind": "nonpositive_gap_ci",
                            "step": step,
                            "source": source,
                            "window": window,
                        }
                    )
    ref_step = int(policy["retention_reference_step"])
    final_step = int(policy["retention_candidate_step"])
    retention_evidence: dict[str, dict[str, Any]] = {source: {} for source in SOURCES}
    for source in SOURCES:
        source_index = SOURCES.index(source)
        for window_index, window in enumerate(policy["retention_windows"]):
            retention = _paired_policy_margin(
                matched[ref_step],
                matched[final_step],
                source=source,
                window=window,
                policy="gap_retention",
                threshold=float(policy["minimum_gap_retention"]),
                seed=seed + 5_000_000 + source_index * 100_000 + window_index,
                samples=samples,
            )
            retention_evidence[source][window] = retention
            if retention["mean_bootstrap_ci"]["low"] <= 0:
                deviations.append(
                    {
                        "kind": "gap_retention",
                        "source": source,
                        "window": window,
                        "paired_margin": retention,
                    }
                )

    ce_ref_step = int(policy["correct_ce_reference_step"])
    ce_final_step = int(policy["correct_ce_candidate_step"])
    trajectory: dict[str, Any] = {}
    for source_index, source in enumerate(SOURCES):
        source_trajectory: dict[str, Any] = {}
        for window_index, window in enumerate(WINDOWS):
            correct_change = _paired_change(
                matched[0],
                matched[ce_final_step],
                source=source,
                window=window,
                field="correct_ce",
                seed=seed + 3_000_000 + source_index * 100_000 + window_index,
                samples=samples,
            )
            gap_change = _paired_change(
                matched[0],
                matched[ce_final_step],
                source=source,
                window=window,
                field="gap",
                seed=seed + 4_000_000 + source_index * 100_000 + window_index,
                samples=samples,
            )
            source_trajectory[window] = {
                "step0_to_final_correct_ce_improvement": correct_change,
                "step0_to_final_gap_change": gap_change,
            }
            if window in retention_evidence[source]:
                source_trajectory[window]["late_gap_retention_margin"] = retention_evidence[source][
                    window
                ]
            if (
                policy["require_step0_to_final_correct_ce_improvement"]
                and correct_change["mean_bootstrap_ci"]["low"] <= 0
            ):
                deviations.append(
                    {"kind": "correct_ce_no_improvement", "source": source, "window": window}
                )
        noninferiority = _paired_policy_margin(
            matched[ce_ref_step],
            matched[ce_final_step],
            source=source,
            window="all",
            policy="correct_ce_noninferiority",
            threshold=float(policy["correct_ce_max_relative_increase"]),
            seed=seed + 6_000_000 + source_index * 100_000,
            samples=samples,
        )
        source_trajectory["all"]["late_correct_ce_noninferiority_margin"] = noninferiority
        if noninferiority["mean_bootstrap_ci"]["low"] <= 0:
            deviations.append(
                {
                    "kind": "correct_ce_regression",
                    "source": source,
                    "reference_step": ce_ref_step,
                    "candidate_step": ce_final_step,
                    "paired_margin": noninferiority,
                }
            )
        trajectory[source] = source_trajectory

    attention_trajectory = {
        str(step): compare_ssmax_attention_reports(
            matched[0]["attention_diagnostics"],
            matched[step]["attention_diagnostics"],
        )
        for step in REQUIRED_STEPS
        if step != 0
    }

    report: dict[str, Any] = {
        "format": PROMOTION_REPORT_FORMAT,
        "version": SCHEMA_VERSION,
        "status": "passed" if not deviations else "rejected",
        "created_at": created_at,
        "manifest": manifest_reference(manifest_path, manifest),
        "pair_id": manifest["pair_id"],
        "arm": manifest["arm"],
        "model_variant": manifest["model_variant"],
        "receipts": refs,
        "trajectory": trajectory,
        "attention_trajectory": attention_trajectory,
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
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Validate a raw-pinned, deviation-free SSMax bridge promotion report.

    The returned summary is suitable for constructing or validating a v4 parent-quality gate.
    It binds the report to its finalized manifest and exact permanent step-500 candidate.
    """

    if expected_model_variant not in MODEL_VARIANTS:
        raise SSMaxBridgeEvidenceError("Expected promotion model variant is unsupported")
    report_path = validate_artifact_reference(value, name="SSMax bridge promotion report")
    report = _exact_fields(
        load_json(report_path), _PROMOTION_REPORT_FIELDS, name="SSMax bridge promotion report"
    )
    if (
        report["format"] != PROMOTION_REPORT_FORMAT
        or report["version"] != SCHEMA_VERSION
        or report["status"] != "passed"
        or report["arm"] != expected_model_variant
        or report["model_variant"] != expected_model_variant
        or report["deviations"] != []
    ):
        raise SSMaxBridgeEvidenceError("SSMax bridge promotion report is not eligible")
    _timestamp(report["created_at"], name="promotion report created_at")
    content_sha = _sha256(report["content_sha256"], name="promotion report content SHA-256")
    if (
        canonical_sha256({key: item for key, item in report.items() if key != "content_sha256"})
        != content_sha
    ):
        raise SSMaxBridgeEvidenceError("Promotion report content SHA-256 differs")

    manifest_reference_value = _exact_fields(
        report["manifest"],
        frozenset({"path", "sha256", "content_sha256"}),
        name="promotion report manifest",
    )
    manifest_path = validate_artifact_reference(
        {
            "path": manifest_reference_value["path"],
            "sha256": manifest_reference_value["sha256"],
        },
        name="promotion report manifest",
    )
    manifest = load_manifest(manifest_path, verify_live=False)
    if (
        manifest["content_sha256"] != manifest_reference_value["content_sha256"]
        or manifest["pair_id"] != report["pair_id"]
        or manifest["arm"] != expected_model_variant
        or manifest["model_variant"] != expected_model_variant
    ):
        raise SSMaxBridgeEvidenceError("Promotion report names an incompatible manifest")
    candidate = validate_checkpoint_reference(
        manifest["checkpoints"][str(REQUIRED_STEPS[-1])],
        expected_step=REQUIRED_STEPS[-1],
        verify_live=verify_live_checkpoint,
    )
    if (
        Path(str(candidate["path"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or candidate["config_sha256"] != expected_checkpoint_config_sha256
    ):
        raise SSMaxBridgeEvidenceError("Promotion report names a different step500 candidate")

    receipts = _exact_fields(
        report["receipts"],
        frozenset(str(step) for step in REQUIRED_STEPS),
        name="promotion report receipts",
    )
    for step in REQUIRED_STEPS:
        step_receipts = _exact_fields(
            receipts[str(step)],
            frozenset({"matched_state", "health"}),
            name=f"promotion report step{step} receipts",
        )
        for receipt_name in ("matched_state", "health"):
            validate_artifact_reference(
                step_receipts[receipt_name],
                name=f"promotion report step{step} {receipt_name}",
            )
    if not isinstance(report["trajectory"], Mapping) or set(report["trajectory"]) != set(SOURCES):
        raise SSMaxBridgeEvidenceError("Promotion report trajectory source set differs")
    if not isinstance(report["attention_trajectory"], Mapping) or set(
        report["attention_trajectory"]
    ) != {str(step) for step in REQUIRED_STEPS[1:]}:
        raise SSMaxBridgeEvidenceError("Promotion report attention trajectory differs")
    recomputed = build_promotion_report(
        manifest_path=manifest_path,
        matched_receipts={step: receipts[str(step)]["matched_state"] for step in REQUIRED_STEPS},
        health_receipts={step: receipts[str(step)]["health"] for step in REQUIRED_STEPS},
        created_at=str(report["created_at"]),
        # The candidate itself was optionally re-hashed above. Rebuilding a historical report
        # must not require the later perception checkout to have the bridge run's exact HEAD or
        # mutable source-artifact paths; their raw identities are already sealed by the manifest.
        verify_live_manifest=False,
    )
    if recomputed != dict(report):
        raise SSMaxBridgeEvidenceError(
            "Promotion report does not equal the report rebuilt from its pinned receipts"
        )
    return {
        "report": report,
        "report_reference": dict(value),
        "manifest": manifest,
        "manifest_reference": dict(manifest_reference_value),
        "candidate": dict(candidate),
    }


def validate_ssmax_bridge_parent_gate(
    gate: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_data_contract_sha256: str,
    expected_trainable_contract_sha256: str,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Validate the dedicated deviation-free v4 SSMax bridge parent gate."""

    value = _exact_fields(gate, _SSMAX_PARENT_GATE_FIELDS, name="SSMax v4 parent gate")
    for name, expected in (
        ("format", "vision_alignment_parent_gate"),
        ("version", 4),
        ("status", "approved"),
        ("recipe_version", 1),
        ("formatter_version", "vision-alignment-document-v1"),
        ("phase", "bridge"),
        ("model_variant", expected_model_variant),
        ("arm", expected_model_variant),
        ("global_step", REQUIRED_STEPS[-1]),
        ("checkpoint_config_sha256", expected_checkpoint_config_sha256),
        ("data_contract_sha256", expected_data_contract_sha256),
        ("trainable_contract_sha256", expected_trainable_contract_sha256),
    ):
        if value[name] != expected:
            raise SSMaxBridgeEvidenceError(f"SSMax v4 parent gate {name} differs")
    if Path(str(value["checkpoint"])).expanduser().resolve() != expected_checkpoint.resolve():
        raise SSMaxBridgeEvidenceError("SSMax v4 parent gate names a different checkpoint")
    if expected_checkpoint.name != f"step{REQUIRED_STEPS[-1]}":
        raise SSMaxBridgeEvidenceError("SSMax v4 parent gate candidate must be step500")
    for name in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "metrics_artifact_sha256",
        "promotion_report_sha256",
        "manifest_sha256",
        "manifest_content_sha256",
    ):
        _sha256(value[name], name=f"SSMax v4 parent gate {name}")
    if value["waivers"] != []:
        raise SSMaxBridgeEvidenceError("SSMax v4 parent gate does not permit waivers")
    approved_by = value["approved_by"]
    if (
        not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxBridgeEvidenceError("SSMax v4 parent gate approved_by is invalid")
    approved_at = _timestamp(value["approved_at"], name="SSMax v4 parent gate approved_at")

    report_reference = {
        "path": value["promotion_report_path"],
        "sha256": value["promotion_report_sha256"],
    }
    if value["metrics_artifact_sha256"] != value["promotion_report_sha256"]:
        raise SSMaxBridgeEvidenceError("SSMax v4 gate metrics artifact is not its promotion report")
    summary = validate_promotion_report_reference(
        report_reference,
        expected_checkpoint=expected_checkpoint,
        expected_checkpoint_config_sha256=expected_checkpoint_config_sha256,
        expected_model_variant=expected_model_variant,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    candidate = summary["candidate"]
    manifest_reference_value = summary["manifest_reference"]
    if candidate["identity_sha256"] != value["checkpoint_identity_sha256"]:
        raise SSMaxBridgeEvidenceError("SSMax v4 gate checkpoint identity differs")
    if (
        Path(str(manifest_reference_value["path"])).resolve()
        != Path(str(value["manifest_path"])).expanduser().resolve()
        or manifest_reference_value["sha256"] != value["manifest_sha256"]
        or manifest_reference_value["content_sha256"] != value["manifest_content_sha256"]
    ):
        raise SSMaxBridgeEvidenceError("SSMax v4 gate manifest reference differs")
    report_created = datetime.fromisoformat(
        str(summary["report"]["created_at"]).replace("Z", "+00:00")
    )
    approval_time = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    if approval_time < report_created:
        raise SSMaxBridgeEvidenceError("SSMax v4 gate approval predates its promotion report")
    return summary


def build_parent_gate(
    *,
    promotion_report_path: Path,
    expected_promotion_report_sha256: str,
    approved_by: str,
    approved_at: str,
    verify_live_checkpoint: bool = True,
) -> dict[str, Any]:
    """Build an explicit, waiver-free human approval for a passed bridge report.

    This is an approval action, so the human identity and timestamp are mandatory. The report,
    manifest, candidate checkpoint, and candidate training contracts are all reopened before the
    v4 gate is emitted.
    """

    report_path = promotion_report_path.expanduser().resolve()
    if sha256_file(report_path) != _sha256(
        expected_promotion_report_sha256,
        name="expected promotion report SHA-256",
    ):
        raise SSMaxBridgeEvidenceError("Promotion report differs from its explicit approval pin")
    raw_report = _exact_fields(
        load_json(report_path), _PROMOTION_REPORT_FIELDS, name="promotion report"
    )
    manifest_ref = _exact_fields(
        raw_report["manifest"],
        frozenset({"path", "sha256", "content_sha256"}),
        name="promotion report manifest",
    )
    manifest_path = Path(str(manifest_ref["path"])).expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    candidate = manifest["checkpoints"][str(REQUIRED_STEPS[-1])]
    candidate_path = Path(str(candidate["path"])).expanduser().resolve()
    config_path = candidate_path / "config.json"
    if sha256_file(config_path) != candidate["config_sha256"]:
        raise SSMaxBridgeEvidenceError("Candidate config differs from its manifest identity")
    candidate_config = _mapping(load_json(config_path), name="candidate config")
    metadata = _mapping(
        candidate_config.get("vision_alignment"), name="candidate vision-alignment metadata"
    )
    if (
        metadata.get("phase") != "bridge"
        or metadata.get("model_variant") != manifest["model_variant"]
    ):
        raise SSMaxBridgeEvidenceError("Candidate metadata differs from the bridge lineage")
    recipe_version = _positive_int(metadata.get("recipe_version"), name="recipe version")
    formatter_version = metadata.get("formatter_version")
    if not isinstance(formatter_version, str) or not formatter_version:
        raise SSMaxBridgeEvidenceError("Candidate formatter version is malformed")
    data_contract_sha256 = _sha256(
        metadata.get("data_contract_sha256"), name="candidate data contract SHA-256"
    )
    trainable_contract_sha256 = _sha256(
        metadata.get("trainable_contract_sha256"),
        name="candidate trainable contract SHA-256",
    )
    summary = validate_promotion_report_reference(
        {"path": str(report_path), "sha256": expected_promotion_report_sha256},
        expected_checkpoint=candidate_path,
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if (
        not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxBridgeEvidenceError("approved_by is not a durable human identity")
    approval_timestamp = _timestamp(approved_at, name="approval timestamp")
    report_timestamp = _timestamp(
        summary["report"]["created_at"], name="promotion report timestamp"
    )
    if datetime.fromisoformat(approval_timestamp.replace("Z", "+00:00")) < datetime.fromisoformat(
        report_timestamp.replace("Z", "+00:00")
    ):
        raise SSMaxBridgeEvidenceError("Human approval predates the promotion report")
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": 4,
        "status": "approved",
        "recipe_version": recipe_version,
        "formatter_version": formatter_version,
        "phase": "bridge",
        "model_variant": manifest["model_variant"],
        "arm": manifest["arm"],
        "checkpoint": candidate["path"],
        "checkpoint_config_sha256": candidate["config_sha256"],
        "checkpoint_identity_sha256": candidate["identity_sha256"],
        "data_contract_sha256": data_contract_sha256,
        "trainable_contract_sha256": trainable_contract_sha256,
        "global_step": REQUIRED_STEPS[-1],
        "metrics_artifact_sha256": expected_promotion_report_sha256,
        "promotion_report_path": str(report_path),
        "promotion_report_sha256": expected_promotion_report_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_ref["sha256"],
        "manifest_content_sha256": manifest["content_sha256"],
        "approved_by": approved_by,
        "approved_at": approved_at,
        "waivers": [],
    }
    validate_ssmax_bridge_parent_gate(
        gate,
        expected_checkpoint=candidate_path,
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=data_contract_sha256,
        expected_trainable_contract_sha256=trainable_contract_sha256,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    return gate


def _validate_promotion_report_for_comparison(
    value: Mapping[str, str], *, verify_live_checkpoint: bool
) -> Mapping[str, Any]:
    """Validate and rebuild a passed report while deriving its candidate from its manifest."""

    report_path = validate_artifact_reference(value, name="controlled-pair promotion report")
    report = _exact_fields(
        load_json(report_path),
        _PROMOTION_REPORT_FIELDS,
        name="controlled-pair promotion report",
    )
    manifest_reference_value = _exact_fields(
        report["manifest"],
        frozenset({"path", "sha256", "content_sha256"}),
        name="controlled-pair promotion manifest",
    )
    manifest_path = validate_artifact_reference(
        {
            "path": manifest_reference_value["path"],
            "sha256": manifest_reference_value["sha256"],
        },
        name="controlled-pair promotion manifest",
    )
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    candidate = manifest["checkpoints"][str(REQUIRED_STEPS[-1])]
    summary = validate_promotion_report_reference(
        value,
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if summary["manifest"] != manifest:
        raise SSMaxBridgeEvidenceError(
            "Controlled-pair promotion manifest changed during validation"
        )
    return summary


def build_pair_comparison(
    *,
    left_promotion_report: Mapping[str, str],
    right_promotion_report: Mapping[str, str],
    created_at: str,
    verify_live_checkpoint: bool = True,
) -> dict[str, Any]:
    """Compare two passed, fully rebuilt promotion trajectories row-for-row."""

    left_summary = _validate_promotion_report_for_comparison(
        left_promotion_report,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    right_summary = _validate_promotion_report_for_comparison(
        right_promotion_report,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    left_manifest = left_summary["manifest"]
    right_manifest = right_summary["manifest"]
    left_report = left_summary["report"]
    right_report = right_summary["report"]
    left_manifest_path = Path(str(left_summary["manifest_reference"]["path"]))
    right_manifest_path = Path(str(right_summary["manifest_reference"]["path"]))
    if (
        left_manifest["pair_id"] != right_manifest["pair_id"]
        or {left_manifest["model_variant"], right_manifest["model_variant"]} != set(MODEL_VARIANTS)
        or left_manifest["evaluation"] != right_manifest["evaluation"]
        or left_manifest["topology"] != right_manifest["topology"]
        or left_manifest["policy"] != right_manifest["policy"]
        or left_manifest["git"] != right_manifest["git"]
        or left_manifest["recipe"]["sha256"] != right_manifest["recipe"]["sha256"]
        or {source: left_manifest["pairings"][source]["sha256"] for source in SOURCES}
        != {source: right_manifest["pairings"][source]["sha256"] for source in SOURCES}
        or left_manifest["validation"]["sha256"] != right_manifest["validation"]["sha256"]
        or left_manifest["attention_probe"]["sha256"] != right_manifest["attention_probe"]["sha256"]
    ):
        raise SSMaxBridgeEvidenceError("Arm manifests are not one controlled paired experiment")
    left_receipts = left_report["receipts"]
    right_receipts = right_report["receipts"]
    left: dict[int, Mapping[str, Any]] = {}
    right: dict[int, Mapping[str, Any]] = {}
    left_health: dict[int, Mapping[str, Any]] = {}
    right_health: dict[int, Mapping[str, Any]] = {}
    refs: dict[str, Any] = {}
    health_compatibility: dict[str, Any] = {}
    compatible_loader_fields = (
        "data_contract_sha256",
        "dataset_fingerprints_sha256",
        "initial_state_sha256",
        "checkpoint_final_state_sha256",
        "replayed_final_state_sha256",
        "rank_state_count",
        "rank_states_global_step",
        "rank_states_batches_processed",
        "dp_world_size",
        "batches_replayed",
        "total_data_errors",
    )
    for step in REQUIRED_STEPS:
        left_step_refs = left_receipts[str(step)]
        right_step_refs = right_receipts[str(step)]
        _, left[step] = _validate_receipt_reference(
            left_step_refs["matched_state"],
            expected_format=MATCHED_STATE_RECEIPT_FORMAT,
            manifest=left_manifest,
            step=step,
            name=f"left step{step}",
        )
        _, right[step] = _validate_receipt_reference(
            right_step_refs["matched_state"],
            expected_format=MATCHED_STATE_RECEIPT_FORMAT,
            manifest=right_manifest,
            step=step,
            name=f"right step{step}",
        )
        _, left_health[step] = _validate_receipt_reference(
            left_step_refs["health"],
            expected_format=HEALTH_RECEIPT_FORMAT,
            manifest=left_manifest,
            step=step,
            name=f"left step{step} health",
        )
        _, right_health[step] = _validate_receipt_reference(
            right_step_refs["health"],
            expected_format=HEALTH_RECEIPT_FORMAT,
            manifest=right_manifest,
            step=step,
            name=f"right step{step} health",
        )
        left_loader = _mapping(left_health[step].get("loader"), name=f"left step{step} loader")
        right_loader = _mapping(right_health[step].get("loader"), name=f"right step{step} loader")
        loader_equal = {
            name: left_loader.get(name) == right_loader.get(name)
            and left_loader.get(name) is not None
            for name in compatible_loader_fields
        }
        same_sources = left_health[step].get("sources") == right_health[step].get("sources")
        same_protocol = left_health[step].get("protocol") == right_health[step].get("protocol")
        left_counters = left_health[step].get("health_ledger", {}).get("counters")
        right_counters = right_health[step].get("health_ledger", {}).get("counters")
        same_zero_counters = (
            left_counters == right_counters
            and isinstance(left_counters, Mapping)
            and all(left_counters.get(name) == 0 for name in left_counters)
        )
        if not all(loader_equal.values()) or not (
            same_sources and same_protocol and same_zero_counters
        ):
            raise SSMaxBridgeEvidenceError(
                f"Controlled-pair step{step} health/data trajectories differ"
            )
        health_compatibility[str(step)] = {
            "loader_fields_equal": loader_equal,
            "source_delivery_equal": same_sources,
            "protocol_equal": same_protocol,
            "zero_health_counters_equal": same_zero_counters,
            "left_health_sha256": left_step_refs["health"]["sha256"],
            "right_health_sha256": right_step_refs["health"]["sha256"],
        }
        refs[str(step)] = {
            "left": {
                "matched_state": dict(left_step_refs["matched_state"]),
                "health": dict(left_step_refs["health"]),
            },
            "right": {
                "matched_state": dict(right_step_refs["matched_state"]),
                "health": dict(right_step_refs["health"]),
            },
        }
    left_components = left[0].get("component_state")
    right_components = right[0].get("component_state")
    if not isinstance(left_components, Mapping) or not isinstance(right_components, Mapping):
        raise SSMaxBridgeEvidenceError("Step0 receipts omit new-component state descriptors")
    for label, components in (("left", left_components), ("right", right_components)):
        expected = _sha256(components.get("sha256"), name=f"{label} component-state SHA-256")
        if (
            canonical_sha256({key: item for key, item in components.items() if key != "sha256"})
            != expected
        ):
            raise SSMaxBridgeEvidenceError(f"{label} component-state descriptor is malformed")
    component_fields = {
        "vision": left_components.get("vision") == right_components.get("vision"),
        "connector": left_components.get("connector") == right_components.get("connector"),
        "image_embedding_rows": left_components.get("image_embedding_rows")
        == right_components.get("image_embedding_rows"),
    }
    if not all(component_fields.values()):
        differing = sorted(name for name, equal in component_fields.items() if not equal)
        raise SSMaxBridgeEvidenceError(
            f"Controlled-pair step0 new components are not bit-identical: {differing}"
        )
    component_attestation = {
        "same_topology": left_manifest["topology"] == right_manifest["topology"],
        "vision_bit_identical": True,
        "connector_bit_identical": True,
        "image_embedding_rows_bit_identical": True,
        "left_component_state_sha256": left_components["sha256"],
        "right_component_state_sha256": right_components["sha256"],
    }
    component_attestation["sha256"] = canonical_sha256(component_attestation)
    samples = int(left_manifest["evaluation"]["bootstrap_samples"])
    seed = int(left_manifest["evaluation"]["bootstrap_seed"])
    comparison: dict[str, Any] = {}
    attention_comparison: dict[str, Any] = {}
    absolute_directions: list[int] = []
    adaptation_directions: list[int] = []
    for step in REQUIRED_STEPS:
        left_attention = left[step].get("attention_diagnostics")
        right_attention = right[step].get("attention_diagnostics")
        if not isinstance(left_attention, Mapping) or not isinstance(right_attention, Mapping):
            raise SSMaxBridgeEvidenceError(
                f"Controlled-pair step{step} attention diagnostics are absent"
            )
        try:
            validate_ssmax_attention_report(
                left_attention, label=f"left step{step} attention diagnostics"
            )
            validate_ssmax_attention_report(
                right_attention, label=f"right step{step} attention diagnostics"
            )
            mechanism_delta = compare_ssmax_attention_reports(
                left_attention,
                right_attention,
            )
        except ValueError as error:
            raise SSMaxBridgeEvidenceError(
                f"Controlled-pair step{step} attention comparison failed: {error}"
            ) from error
        attention_comparison[str(step)] = {
            "left_report_sha256": left_attention["report_sha256"],
            "right_report_sha256": right_attention["report_sha256"],
            "left_minus_right": mechanism_delta,
        }
        step_result: dict[str, Any] = {}
        for source_index, source in enumerate(SOURCES):
            source_result: dict[str, Any] = {}
            left_baseline_rows = _receipt_records(left[0], source)
            right_baseline_rows = _receipt_records(right[0], source)
            for window_index, window in enumerate(WINDOWS):
                left_rows = _receipt_records(left[step], source)
                right_rows = _receipt_records(right[step], source)
                identities = (
                    "pairing_position",
                    "recipient_index",
                    "donor_index",
                    "response_tokens",
                )
                row_sets = (left_rows, right_rows, left_baseline_rows, right_baseline_rows)
                if any(len(rows) != len(left_rows) for rows in row_sets[1:]) or any(
                    any(rows[0].get(field) != row.get(field) for field in identities)
                    for rows in zip(*row_sets, strict=True)
                    for row in rows[1:]
                ):
                    raise SSMaxBridgeEvidenceError("Arm receipts are not paired row-for-row")
                gap_delta = [
                    float(a["ce_gap_wrong_minus_correct"][window])
                    - float(b["ce_gap_wrong_minus_correct"][window])
                    for a, b in zip(left_rows, right_rows, strict=True)
                ]
                # Positive means lower correct-image CE in the left arm.
                ce_advantage = [
                    float(b["correct_ce"][window]) - float(a["correct_ce"][window])
                    for a, b in zip(left_rows, right_rows, strict=True)
                ]
                left_gap_change = [
                    float(candidate["ce_gap_wrong_minus_correct"][window])
                    - float(baseline["ce_gap_wrong_minus_correct"][window])
                    for baseline, candidate in zip(left_baseline_rows, left_rows, strict=True)
                ]
                right_gap_change = [
                    float(candidate["ce_gap_wrong_minus_correct"][window])
                    - float(baseline["ce_gap_wrong_minus_correct"][window])
                    for baseline, candidate in zip(right_baseline_rows, right_rows, strict=True)
                ]
                gap_change_did = [
                    left_change - right_change
                    for left_change, right_change in zip(
                        left_gap_change, right_gap_change, strict=True
                    )
                ]
                # Positive improvement means that correct-image CE fell from step 0.
                left_correct_ce_improvement = [
                    float(baseline["correct_ce"][window]) - float(candidate["correct_ce"][window])
                    for baseline, candidate in zip(left_baseline_rows, left_rows, strict=True)
                ]
                right_correct_ce_improvement = [
                    float(baseline["correct_ce"][window]) - float(candidate["correct_ce"][window])
                    for baseline, candidate in zip(right_baseline_rows, right_rows, strict=True)
                ]
                correct_ce_improvement_did = [
                    left_change - right_change
                    for left_change, right_change in zip(
                        left_correct_ce_improvement,
                        right_correct_ce_improvement,
                        strict=True,
                    )
                ]
                base_seed = seed + step * 10_000 + source_index * 1_000 + window_index * 10
                gap_summary = summarize_paired_values(gap_delta, seed=base_seed, samples=samples)
                ce_summary = summarize_paired_values(
                    ce_advantage, seed=base_seed + 1, samples=samples
                )
                source_result[window] = {
                    "left_minus_right_gap": gap_summary,
                    "left_correct_ce_advantage": ce_summary,
                    "adaptation_from_step0": {
                        "left_gap_change": summarize_paired_values(
                            left_gap_change, seed=base_seed + 2, samples=samples
                        ),
                        "right_gap_change": summarize_paired_values(
                            right_gap_change, seed=base_seed + 3, samples=samples
                        ),
                        "gap_change_did_left_minus_right": summarize_paired_values(
                            gap_change_did, seed=base_seed + 4, samples=samples
                        ),
                        "left_correct_ce_improvement": summarize_paired_values(
                            left_correct_ce_improvement,
                            seed=base_seed + 5,
                            samples=samples,
                        ),
                        "right_correct_ce_improvement": summarize_paired_values(
                            right_correct_ce_improvement,
                            seed=base_seed + 6,
                            samples=samples,
                        ),
                        "correct_ce_improvement_did_left_minus_right": (
                            summarize_paired_values(
                                correct_ce_improvement_did,
                                seed=base_seed + 7,
                                samples=samples,
                            )
                        ),
                    },
                }
                if step == 500:
                    ci = gap_summary["mean_bootstrap_ci"]
                    absolute_directions.append(1 if ci["low"] > 0 else -1 if ci["high"] < 0 else 0)
                    adaptation_ci = source_result[window]["adaptation_from_step0"][
                        "gap_change_did_left_minus_right"
                    ]["mean_bootstrap_ci"]
                    adaptation_directions.append(
                        1 if adaptation_ci["low"] > 0 else -1 if adaptation_ci["high"] < 0 else 0
                    )
            step_result[source] = source_result
        comparison[str(step)] = step_result

    def _dominant_arm(directions: Sequence[int]) -> str | None:
        if directions and all(direction == 1 for direction in directions):
            return str(left_manifest["arm"])
        if directions and all(direction == -1 for direction in directions):
            return str(right_manifest["arm"])
        return None

    absolute_dominant = _dominant_arm(absolute_directions)
    adaptation_dominant = _dominant_arm(adaptation_directions)
    result: dict[str, Any] = {
        "format": PAIR_COMPARISON_FORMAT,
        "version": SCHEMA_VERSION,
        "created_at": created_at,
        "pair_id": left_manifest["pair_id"],
        "left_promotion_report": dict(left_promotion_report),
        "right_promotion_report": dict(right_promotion_report),
        "left_manifest": manifest_reference(left_manifest_path, left_manifest),
        "right_manifest": manifest_reference(right_manifest_path, right_manifest),
        "left_arm": left_manifest["arm"],
        "right_arm": right_manifest["arm"],
        "receipt_references": refs,
        "health_trajectory_attestation": health_compatibility,
        "step0_new_component_attestation": component_attestation,
        "comparison": comparison,
        "attention_comparison": attention_comparison,
        "step500_gap_dominant_arm": absolute_dominant,
        "step500_absolute_gap_dominant_arm": absolute_dominant,
        "step500_adaptation_gap_dominant_arm": adaptation_dominant,
        "interpretation": (
            "Absolute capability and adaptation are separate. An absolute dominant arm is "
            "reported only when all six step500 same-checkpoint source/window gap-delta "
            "bootstrap intervals exclude zero in one direction. An adaptation dominant arm is "
            "reported only when all six step0-normalized gap-change difference-in-differences "
            "intervals exclude zero in one direction; this is the bridge molmofiability signal. "
            "Otherwise that ranking is inconclusive. Correct-image CE is a separate paired "
            "axis. Attention comparisons are descriptive mechanism trajectories and never "
            "contribute to either ranking."
        ),
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "BRIDGE_EVALUATION_CONTRACT",
    "BRIDGE_GLOBAL_BATCH_INSTANCES",
    "BRIDGE_POLICY_CONTRACT",
    "BRIDGE_TOPOLOGY_CONTRACT",
    "HEALTH_RECEIPT_FORMAT",
    "IMAGE_TOKEN_ROWS",
    "MANIFEST_FORMAT",
    "MANIFEST_SPEC_FORMAT",
    "MANIFEST_SPEC_RELATIVE_PATHS",
    "MATCHED_STATE_RECEIPT_FORMAT",
    "MODEL_VARIANTS",
    "PAIR_COMPARISON_FORMAT",
    "PROMOTION_REPORT_FORMAT",
    "REQUIRED_STEPS",
    "SCHEMA_VERSION",
    "SOURCES",
    "WINDOWS",
    "GenericDCPLoadInventory",
    "SSMaxBridgeEvidenceError",
    "aggregate_matched_records",
    "artifact_reference",
    "bootstrap_mean_interval",
    "build_manifest",
    "build_parent_gate",
    "build_pair_comparison",
    "build_promotion_report",
    "canonical_json_bytes",
    "canonical_sha256",
    "checkpoint_identity",
    "load_json",
    "load_manifest",
    "load_manifest_spec",
    "manifest_reference",
    "sha256_file",
    "summarize_paired_values",
    "validate_artifact_reference",
    "validate_checkpoint_reference",
    "validate_manifest",
    "validate_promotion_report_reference",
    "validate_ssmax_bridge_parent_gate",
    "verify_generic_dcp_load_inventory",
    "write_json_once",
]
