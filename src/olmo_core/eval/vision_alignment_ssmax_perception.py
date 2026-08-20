"""Fail-closed paired perception evidence for dense SSMax vision alignment.

The historical perception promotion code is an audit of one completed s002 experiment.  Its
constants, tensor counts, optimizer skips, and waivers are evidence, not a reusable protocol.
This module therefore defines a separate model-variant-aware protocol for the two dense SSMax
lineages.  A finalized manifest is created only after both the vision-unfrozen treatment and its
frozen-vision control have permanent step-0, step-3000, and step-4000 checkpoints.

The manifest binds the reviewed profiles, saved git blobs, common approved bridge parent, data
provenance, pairings, topology, seeds, and every checkpoint byte identity.  Promotion is derived
again from raw per-example and run-health receipts whenever a report is audited.  Nothing in this
module trains a model, manufactures an approval, accepts a waiver, or falls back to the legacy
s002 allowlist.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.data.multimodal.ssmax_single_response import (
    ssmax_single_response_projection_contract,
    validate_ssmax_single_response_calibration,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    load_perception_provenance_manifest,
)
from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.eval.matched_wrong_image import (
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)
from olmo_core.eval.ssmax_attention_diagnostics import (
    SSMaxProbeManifest,
    compare_ssmax_attention_reports,
    validate_ssmax_attention_report,
)
from olmo_core.eval.vision_alignment_ssmax_data import content_ids_sha256
from olmo_core.train.callbacks.ssmax_health_ledger import (
    SSMaxHealthLedgerError,
    validate_ssmax_health_ledger_state,
)

MANIFEST_SPEC_FORMAT = "vision_alignment_ssmax_perception_pair_manifest_spec"
MANIFEST_FORMAT = "vision_alignment_ssmax_perception_pair_manifest"
EVALUATION_RECEIPT_FORMAT = "vision_alignment_ssmax_perception_evaluation_receipt"
HEALTH_RECEIPT_FORMAT = "vision_alignment_ssmax_perception_health_receipt"
PROMOTION_REPORT_FORMAT = "vision_alignment_ssmax_perception_promotion_report"
MODEL_COMPARISON_FORMAT = "vision_alignment_ssmax_perception_model_comparison"
SCHEMA_VERSION = 1
PARENT_GATE_VERSION = 5

CONTROL_ARM = "frozen_vision_control"
TREATMENT_ARM = "treatment"
ARMS = (CONTROL_ARM, TREATMENT_ARM)
REQUIRED_STEPS = (0, 3000, 4000)
SOURCES = tuple(PERCEPTION_SOURCE_NAMES)
WINDOWS = ("first_1", "first_8", "first_32", "all")
MODEL_VARIANTS = bridge.MODEL_VARIANTS
IMAGE_TOKEN_ROWS = bridge.IMAGE_TOKEN_ROWS

MIN_EXAMPLES_PER_SOURCE = 512
MIN_BOOTSTRAP_SAMPLES = 10_000
CORRECT_CE_MAX_RELATIVE_INCREASE = 0.02
MINIMUM_GAP_RETENTION = 0.8
LOSS_MASS_SHARE_TOLERANCE = 0.02
ATTENTION_PROBE_ROWS = 32
PROJECTION_SEED = 95818
PAIRING_SEED = 6198

EVALUATION_PRODUCER = "evaluation"
HEALTH_PRODUCER = "health"
PRODUCER_RELATIVE_PATHS = {
    EVALUATION_PRODUCER: "src/scripts/eval/vision_alignment_ssmax_perception.py",
    HEALTH_PRODUCER: "src/scripts/eval/vision_alignment_ssmax_perception_health.py",
}

_ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_REF_FIELDS = frozenset({"path", "sha256", "content_sha256"})
_PRODUCER_SOURCE_REF_FIELDS = frozenset({"repo_relative_path", "sha256", "git_ref"})
_ARM_SPEC_FIELDS = frozenset({"run_name", "checkpoint_root", "training_profile"})
_SPEC_FIELDS = frozenset(
    {
        "format",
        "version",
        "pair_id",
        "model_variant",
        "arms",
        "recipe",
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
_ARM_MANIFEST_FIELDS = frozenset(
    {
        "run_name",
        "checkpoint_root",
        "training_profile",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "checkpoints",
    }
)
_BRIDGE_PARENT_FIELDS = frozenset(
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
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "version",
        "created_at",
        "pair_id",
        "model_variant",
        "git",
        "producers",
        "recipe",
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
        "arms",
        "pair_contract_sha256",
        "content_sha256",
    }
)
_SINGLE_RESPONSE_FIELDS = frozenset({"contract", "calibration", "projected_mean_loss_weight"})
_PROJECTION_CONTRACT_FIELDS = frozenset(
    {
        "format",
        "version",
        "algorithm",
        "seed",
        "training_epoch_policy",
        "evaluation_epoch_policy",
        "shared_subsegment_id",
        "loss_token_weighting",
        "positive_weight_policy",
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
        "model_variant",
        "receipts",
        "summary",
        "deviations",
        "content_sha256",
    }
)
_MODEL_COMPARISON_FIELDS = frozenset(
    {
        "format",
        "version",
        "created_at",
        "decision_scope",
        "winner",
        "left",
        "right",
        "protocol",
        "absolute_and_adaptation_trajectories",
        "causal_adaptation_contrast",
        "attention_comparisons",
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
        "promotion_report_content_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_content_sha256",
        "approved_by",
        "approved_at",
        "waivers",
    }
)


class SSMaxPerceptionEvidenceError(ValueError):
    """Raised when SSMax perception evidence violates the locked causal contract."""


def canonical_sha256(value: Any) -> str:
    """Return the semantic SHA-256 used by SSMax perception artifacts."""

    return bridge.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    """Hash a file while translating bridge errors into this evidence boundary."""

    try:
        return bridge.sha256_file(path)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error


def load_json(path: Path) -> Any:
    """Strictly load finite JSON while rejecting duplicate keys."""

    try:
        return bridge.load_json(path)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create an immutable JSON artifact."""

    bridge.write_json_once(path, payload)


def _exact_fields(value: Any, expected: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxPerceptionEvidenceError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise SSMaxPerceptionEvidenceError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SSMaxPerceptionEvidenceError(f"{name} must be an object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SSMaxPerceptionEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SSMaxPerceptionEvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SSMaxPerceptionEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SSMaxPerceptionEvidenceError(f"{name} must be finite")
    return result


def _timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise SSMaxPerceptionEvidenceError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SSMaxPerceptionEvidenceError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SSMaxPerceptionEvidenceError(f"{name} must include a timezone")
    return parsed


def artifact_reference(path: Path) -> dict[str, str]:
    """Build an absolute raw-byte reference to an existing artifact."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SSMaxPerceptionEvidenceError(f"Required artifact is absent: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_artifact_reference(value: Any, *, name: str) -> Path:
    """Re-open and hash a raw-byte artifact reference."""

    reference = _exact_fields(value, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    path_value = reference["path"]
    if not isinstance(path_value, str) or not path_value:
        raise SSMaxPerceptionEvidenceError(f"{name} path must be non-empty")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != _sha256(
        reference["sha256"], name=f"{name} SHA-256"
    ):
        raise SSMaxPerceptionEvidenceError(f"{name} differs from its immutable reference")
    return path


def _artifact_reference_shape(value: Any, *, name: str) -> Mapping[str, Any]:
    reference = _exact_fields(value, _ARTIFACT_REF_FIELDS, name=f"{name} reference")
    if not isinstance(reference["path"], str) or not reference["path"]:
        raise SSMaxPerceptionEvidenceError(f"{name} path must be non-empty")
    _sha256(reference["sha256"], name=f"{name} SHA-256")
    return reference


def _single_response_binding_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(config.get("data"), name="checkpoint data")
    raw = _mapping(
        data.get("ssmax_single_response_projection"),
        name="SSMax single-response projection",
    )
    seed = raw.get("seed")
    if seed != PROJECTION_SEED:
        raise SSMaxPerceptionEvidenceError(
            f"SSMax single-response seed must be the data seed {PROJECTION_SEED}"
        )
    try:
        contract = ssmax_single_response_projection_contract(
            seed=seed,
            loss_token_weighting=str(data.get("loss_token_weighting")),
            format=str(raw.get("format")),
            version=int(raw.get("version")),
            algorithm=str(raw.get("algorithm")),
        )
    except (TypeError, ValueError) as error:
        raise SSMaxPerceptionEvidenceError(
            f"SSMax single-response projection contract is invalid: {error}"
        ) from error
    calibration_path = raw.get("calibration_path")
    calibration_sha = raw.get("calibration_sha256")
    if not isinstance(calibration_path, str) or not calibration_path:
        raise SSMaxPerceptionEvidenceError("SSMax projection calibration path is absent")
    calibration = artifact_reference(Path(calibration_path))
    if calibration["sha256"] != _sha256(
        calibration_sha, name="SSMax projection calibration config SHA-256"
    ):
        raise SSMaxPerceptionEvidenceError(
            "SSMax projection calibration differs from the saved config"
        )
    means = _mapping(raw.get("projected_mean_loss_weight"), name="projected mean loss weights")
    if set(means) != set(SOURCES) or any(
        _finite(value, name=f"projected mean {source}") <= 0 for source, value in means.items()
    ):
        raise SSMaxPerceptionEvidenceError(
            "Projected loss-mass calibration must contain every perception source"
        )
    calibration_payload = _mapping(
        load_json(Path(calibration["path"])), name="SSMax projection calibration"
    )
    calibration_content_sha = _sha256(
        calibration_payload.get("content_sha256"),
        name="SSMax projection calibration content SHA-256",
    )
    source_audit_path = Path(str(data.get("source_audit_path"))).expanduser().resolve()
    source_audit_raw = artifact_reference(source_audit_path)
    source_audit = {
        "path": str(source_audit_path),
        "raw_sha256": source_audit_raw["sha256"],
        "content_sha256": _sha256(
            data.get("source_audit_fingerprint"), name="source audit fingerprint"
        ),
    }
    provenance_path = Path(str(data.get("perception_provenance_path"))).expanduser().resolve()
    provenance_raw = artifact_reference(provenance_path)
    if provenance_raw["sha256"] != _sha256(
        data.get("perception_provenance_sha256"), name="perception provenance config SHA-256"
    ):
        raise SSMaxPerceptionEvidenceError("Perception provenance differs from the saved config")
    provenance_payload = _mapping(load_json(provenance_path), name="perception provenance")
    provenance = {
        "path": str(provenance_path),
        "raw_sha256": provenance_raw["sha256"],
        "content_sha256": _sha256(
            provenance_payload.get("content_sha256"),
            name="perception provenance content SHA-256",
        ),
    }
    evaluation = _mapping(config.get("evaluation"), name="checkpoint evaluation config")
    validation_rows = _positive_int(
        evaluation.get("examples_per_source"), name="validation examples per source"
    )
    try:
        validate_ssmax_single_response_calibration(
            calibration_payload,
            expected_phase="perception",
            expected_contract=contract,
            expected_source_audit=source_audit,
            expected_selection_manifest=provenance,
            expected_visual_sources=SOURCES,
            expected_unprojected_sources=(),
            expected_mean_loss_weight=means,
            expected_validation_rows_per_source={source: validation_rows for source in SOURCES},
        )
    except ValueError as error:
        raise SSMaxPerceptionEvidenceError(
            f"SSMax projection calibration failed semantic validation: {error}"
        ) from error
    return {
        "contract": contract,
        "calibration": {**calibration, "content_sha256": calibration_content_sha},
        "projected_mean_loss_weight": dict(means),
    }


def _validate_single_response_binding(value: Any, *, verify_live: bool) -> Mapping[str, Any]:
    binding = _exact_fields(value, _SINGLE_RESPONSE_FIELDS, name="SSMax single-response binding")
    contract = _exact_fields(
        binding["contract"],
        _PROJECTION_CONTRACT_FIELDS,
        name="SSMax single-response contract",
    )
    expected = ssmax_single_response_projection_contract(
        seed=_positive_int(contract["seed"], name="projection seed", minimum=0),
        loss_token_weighting=str(contract["loss_token_weighting"]),
        format=str(contract["format"]),
        version=_positive_int(contract["version"], name="projection version"),
        algorithm=str(contract["algorithm"]),
    )
    if dict(contract) != expected:
        raise SSMaxPerceptionEvidenceError("SSMax single-response contract differs")
    means = _mapping(binding["projected_mean_loss_weight"], name="projected mean loss weights")
    if set(means) != set(SOURCES) or any(
        _finite(value, name=f"projected mean {source}") <= 0 for source, value in means.items()
    ):
        raise SSMaxPerceptionEvidenceError(
            "Projected mean loss weights differ from the perception sources"
        )
    calibration = _exact_fields(
        binding["calibration"],
        _MANIFEST_REF_FIELDS,
        name="SSMax projection calibration reference",
    )
    if not isinstance(calibration["path"], str) or not calibration["path"]:
        raise SSMaxPerceptionEvidenceError("SSMax projection calibration path must be non-empty")
    _sha256(calibration["sha256"], name="SSMax projection calibration raw SHA-256")
    calibration_content_sha = _sha256(
        calibration["content_sha256"],
        name="SSMax projection calibration content SHA-256",
    )
    if verify_live:
        calibration_path = validate_artifact_reference(
            {"path": calibration["path"], "sha256": calibration["sha256"]},
            name="SSMax projection calibration",
        )
        payload = _mapping(load_json(calibration_path), name="SSMax projection calibration")
        if payload.get("content_sha256") != calibration_content_sha:
            raise SSMaxPerceptionEvidenceError(
                "SSMax projection calibration semantic SHA-256 differs"
            )
        try:
            validation_summaries = _mapping(
                payload.get("validation_preflight"),
                name="SSMax projection validation preflight",
            )
            validate_ssmax_single_response_calibration(
                payload,
                expected_phase="perception",
                expected_contract=contract,
                expected_source_audit=_mapping(
                    payload.get("source_audit"), name="calibration source audit"
                ),
                expected_selection_manifest=_mapping(
                    payload.get("selection_manifest"), name="calibration provenance"
                ),
                expected_visual_sources=SOURCES,
                expected_unprojected_sources=(),
                expected_mean_loss_weight=means,
                expected_validation_rows_per_source={
                    source: _positive_int(
                        _mapping(validation_summaries.get(source), name=source).get("rows"),
                        name=f"{source} validation rows",
                    )
                    for source in SOURCES
                },
            )
        except ValueError as error:
            raise SSMaxPerceptionEvidenceError(
                f"SSMax projection calibration failed semantic validation: {error}"
            ) from error
    return binding


def manifest_reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return raw and semantic pins for a finalized pair manifest."""

    reference = artifact_reference(path)
    reference["content_sha256"] = _sha256(
        manifest.get("content_sha256"), name="manifest content SHA-256"
    )
    return reference


def _expect_equal(actual: Any, expected: Any, *, name: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise SSMaxPerceptionEvidenceError(
            f"{name} differs: expected={expected!r}, actual={actual!r}"
        )


def _resolved_path(value: Any, *, repository_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SSMaxPerceptionEvidenceError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _vision_group(config: Mapping[str, Any], *, arm: str) -> tuple[int, Mapping[str, Any]]:
    train_module = _mapping(config.get("train_module"), name=f"{arm} train_module")
    optim = _mapping(train_module.get("optim"), name=f"{arm} optimizer")
    groups = optim.get("group_overrides")
    if not isinstance(groups, list):
        raise SSMaxPerceptionEvidenceError(f"{arm} optimizer groups must be a list")
    matches = [
        (index, group)
        for index, group in enumerate(groups)
        if isinstance(group, Mapping) and group.get("params") == ["*vision.*"]
    ]
    if len(matches) != 1:
        raise SSMaxPerceptionEvidenceError(
            f"{arm} must contain exactly one *vision.* optimizer group"
        )
    return matches[0]


def _normalize_pair_config(config: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    """Erase only the declared causal-arm and run-output identities from a saved config."""

    normalized = copy.deepcopy(dict(config))
    for field in (
        "required_run_name",
        "reviewed_profile_path",
        "reviewed_profile_sha256",
        "perception_trainability_arm",
        "expected_launch_command",
    ):
        normalized.pop(field, None)
    metadata = _mapping(normalized.get("vision_alignment"), name=f"{arm} vision_alignment")
    metadata = dict(metadata)
    metadata.pop("lineage_id", None)
    metadata.pop("trainable_contract_sha256", None)
    normalized["vision_alignment"] = metadata
    trainer = dict(_mapping(normalized.get("trainer"), name=f"{arm} trainer"))
    trainer.pop("save_folder", None)
    callbacks = dict(_mapping(trainer.get("callbacks"), name=f"{arm} callbacks"))
    wandb = callbacks.get("wandb")
    if isinstance(wandb, Mapping):
        wandb = dict(wandb)
        wandb.pop("name", None)
        callbacks["wandb"] = wandb
    ledger = callbacks.get("ssmax_health_ledger")
    if isinstance(ledger, Mapping):
        ledger = dict(ledger)
        ledger["run_name"] = "<derived-run-name>"
        callbacks["ssmax_health_ledger"] = ledger
    trainer["callbacks"] = callbacks
    normalized["trainer"] = trainer

    train_module = dict(_mapping(normalized.get("train_module"), name=f"{arm} train_module"))
    freeze = train_module.get("freeze_params")
    if not isinstance(freeze, list) or any(not isinstance(item, str) for item in freeze):
        raise SSMaxPerceptionEvidenceError(f"{arm} freeze_params must be a string list")
    train_module["freeze_params"] = [item for item in freeze if item != "vision.*"]
    optim = dict(_mapping(train_module.get("optim"), name=f"{arm} optimizer"))
    groups = copy.deepcopy(optim.get("group_overrides"))
    if not isinstance(groups, list):
        raise SSMaxPerceptionEvidenceError(f"{arm} optimizer groups must be a list")
    vision_index, _ = _vision_group(config, arm=arm)
    options = dict(_mapping(groups[vision_index].get("opts"), name=f"{arm} vision options"))
    options["lr"] = "<derived-vision-lr>"
    groups[vision_index] = dict(groups[vision_index])
    groups[vision_index]["opts"] = options
    optim["group_overrides"] = groups
    train_module["optim"] = optim
    normalized["train_module"] = train_module
    return normalized


def validate_saved_config_pair(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    profile_references: Mapping[str, Mapping[str, str]],
    recipe_path: Path,
) -> dict[str, Any]:
    """Prove treatment/control saved configs differ only by the derived vision intervention."""

    if set(configs) != set(ARMS) or set(profile_references) != set(ARMS):
        raise SSMaxPerceptionEvidenceError("Saved config validation requires both causal arms")
    repository_root = recipe_path.resolve().parents[3]
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        config = _mapping(configs[arm], name=f"{arm} saved config")
        arm_spec = _exact_fields(
            _mapping(spec["arms"], name="spec arms")[arm],
            _ARM_SPEC_FIELDS,
            name=f"{arm} spec",
        )
        for field, expected in (
            ("model_variant", spec["model_variant"]),
            ("phase", "perception"),
            ("perception_trainability_arm", arm),
            ("required_run_name", arm_spec["run_name"]),
        ):
            _expect_equal(config.get(field), expected, name=f"{arm} {field}")
        reviewed = _resolved_path(
            config.get("reviewed_profile_path"),
            repository_root=repository_root,
            name=f"{arm} reviewed profile",
        )
        profile_path = Path(str(profile_references[arm]["path"])).resolve()
        if reviewed != profile_path:
            raise SSMaxPerceptionEvidenceError(f"{arm} names a different reviewed profile")
        _expect_equal(
            config.get("reviewed_profile_sha256"),
            profile_references[arm]["sha256"],
            name=f"{arm} reviewed profile SHA-256",
        )
        data = _mapping(config.get("data"), name=f"{arm} data")
        _expect_equal(data.get("pack_sequences"), False, name=f"{arm} pack_sequences")
        _expect_equal(
            data.get("allow_unpinned_synthetic_smoke"),
            False,
            name=f"{arm} synthetic bypass",
        )
        metadata = _mapping(config.get("vision_alignment"), name=f"{arm} metadata")
        for field, expected in (
            ("model_variant", spec["model_variant"]),
            ("phase", "perception"),
            ("lineage_id", arm_spec["run_name"]),
        ):
            _expect_equal(metadata.get(field), expected, name=f"{arm} metadata {field}")
        data_sha = _sha256(metadata.get("data_contract_sha256"), name=f"{arm} data contract")
        trainable_sha = _sha256(
            metadata.get("trainable_contract_sha256"), name=f"{arm} trainable contract"
        )
        initialization = _mapping(config.get("initialization"), name=f"{arm} initialization")
        if initialization.get("expected_parent_phase") != "bridge":
            raise SSMaxPerceptionEvidenceError(f"{arm} does not require a bridge parent")
        launch = _mapping(config.get("launch"), name=f"{arm} launch")
        topology = _mapping(spec["topology"], name="spec topology")
        for field, expected in (
            ("num_nodes", topology["num_nodes"]),
            ("num_gpus", topology["gpus_per_node"]),
            ("workspace", "ai2/scaling-ladders"),
            ("priority", "urgent"),
        ):
            _expect_equal(launch.get(field), expected, name=f"{arm} launch {field}")
        min_runtime = launch.get("min_runtime")
        if min_runtime not in ("8h", 28800, 28_800.0):
            raise SSMaxPerceptionEvidenceError(f"{arm} launch min_runtime must be exactly 8h")
        trainer = _mapping(config.get("trainer"), name=f"{arm} trainer")
        duration = _mapping(trainer.get("max_duration"), name=f"{arm} max duration")
        if duration.get("value") != 4000 or duration.get("unit") != "steps":
            raise SSMaxPerceptionEvidenceError(f"{arm} duration must be exactly 4000 steps")
        callbacks = _mapping(trainer.get("callbacks"), name=f"{arm} callbacks")
        checkpointer = _mapping(
            callbacks.get("checkpointer"),
            name=f"{arm} checkpointer",
        )
        fixed = checkpointer.get("fixed_steps")
        save_interval = checkpointer.get("save_interval")
        saved_steps = set(fixed or ())
        if isinstance(save_interval, int) and save_interval > 0:
            saved_steps.update(range(save_interval, 4001, save_interval))
        required_saved_steps = {500, 1000, 2000, 3000, 4000}
        if (
            not required_saved_steps <= saved_steps
            or checkpointer.get("pre_train_checkpoint") is not True
        ):
            raise SSMaxPerceptionEvidenceError(
                f"{arm} checkpointer must permanently save step0 and steps "
                "500/1000/2000/3000/4000"
            )
        ledger = _mapping(callbacks.get("ssmax_health_ledger"), name=f"{arm} health ledger")
        for field, expected in (
            ("model_variant", spec["model_variant"]),
            ("phase", "perception"),
            ("run_name", arm_spec["run_name"]),
            ("enabled", True),
        ):
            _expect_equal(ledger.get(field), expected, name=f"{arm} health ledger {field}")
        freeze = _mapping(config.get("train_module"), name=f"{arm} train module").get(
            "freeze_params"
        )
        if not isinstance(freeze, list):
            raise SSMaxPerceptionEvidenceError(f"{arm} freeze_params must be a list")
        _, vision_group = _vision_group(config, arm=arm)
        vision_lr = _finite(
            _mapping(vision_group.get("opts"), name=f"{arm} vision options").get("lr"),
            name=f"{arm} vision LR",
        )
        if arm == CONTROL_ARM:
            if freeze.count("vision.*") != 1 or vision_lr != 0:
                raise SSMaxPerceptionEvidenceError(
                    "Frozen-vision control must add vision.* and set its LR to zero"
                )
        elif "vision.*" in freeze or vision_lr <= 0:
            raise SSMaxPerceptionEvidenceError(
                "Treatment must leave vision trainable with a positive vision LR"
            )
        summaries[arm] = {
            "data_contract_sha256": data_sha,
            "trainable_contract_sha256": trainable_sha,
            "initialization": dict(initialization),
            "git": {
                field: _mapping(launch.get("git"), name=f"{arm} git").get(field)
                for field in ("repo", "repo_url", "ref")
            },
            "vision_lr": vision_lr,
            "freeze_params": list(freeze),
        }

    if _normalize_pair_config(configs[CONTROL_ARM], arm=CONTROL_ARM) != _normalize_pair_config(
        configs[TREATMENT_ARM], arm=TREATMENT_ARM
    ):
        raise SSMaxPerceptionEvidenceError(
            "Treatment/control saved configs differ outside the derived vision freeze/LR and "
            "run/profile/output identities"
        )
    if (
        summaries[CONTROL_ARM]["data_contract_sha256"]
        != summaries[TREATMENT_ARM]["data_contract_sha256"]
    ):
        raise SSMaxPerceptionEvidenceError("Treatment/control data contracts differ")
    if summaries[CONTROL_ARM]["initialization"] != summaries[TREATMENT_ARM]["initialization"]:
        raise SSMaxPerceptionEvidenceError("Treatment/control bridge initialization differs")
    if summaries[CONTROL_ARM]["git"] != summaries[TREATMENT_ARM]["git"]:
        raise SSMaxPerceptionEvidenceError("Treatment/control saved git identities differ")
    return summaries


def _validate_spec_common(spec: Mapping[str, Any]) -> None:
    if spec.get("format") != MANIFEST_SPEC_FORMAT or spec.get("version") != SCHEMA_VERSION:
        raise SSMaxPerceptionEvidenceError("SSMax perception manifest spec is incompatible")
    for field in ("pair_id", "model_variant"):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise SSMaxPerceptionEvidenceError(f"Manifest spec {field} must be non-empty")
    if spec["model_variant"] not in MODEL_VARIANTS:
        raise SSMaxPerceptionEvidenceError("Manifest spec model variant is unsupported")
    arms = _exact_fields(spec["arms"], frozenset(ARMS), name="manifest spec arms")
    for arm in ARMS:
        arm_spec = _exact_fields(arms[arm], _ARM_SPEC_FIELDS, name=f"{arm} manifest spec")
        for field in _ARM_SPEC_FIELDS:
            if not isinstance(arm_spec[field], str) or not arm_spec[field]:
                raise SSMaxPerceptionEvidenceError(f"{arm} spec {field} must be non-empty")
    for field in (
        "recipe",
        "bridge_parent_gate",
        "perception_provenance",
        "source_audit",
        "attention_probe",
        "text_sentinel",
    ):
        if not isinstance(spec[field], str) or not spec[field]:
            raise SSMaxPerceptionEvidenceError(f"Manifest spec {field} must be a path")
    pairing_paths = _exact_fields(
        spec["pairing_paths"], frozenset(SOURCES), name="manifest spec pairing paths"
    )
    if any(
        not isinstance(pairing_paths[source], str) or not pairing_paths[source]
        for source in SOURCES
    ):
        raise SSMaxPerceptionEvidenceError("Every manifest pairing path must be non-empty")

    evaluation = _exact_fields(
        spec["evaluation"],
        frozenset(
            {
                "sources",
                "steps",
                "windows",
                "examples_per_source",
                "pairing_seed",
                "bootstrap_seed",
                "bootstrap_samples",
                "rank_batch_instances",
            }
        ),
        name="manifest evaluation",
    )
    if (
        evaluation["sources"] != list(SOURCES)
        or evaluation["steps"] != list(REQUIRED_STEPS)
        or evaluation["windows"] != list(WINDOWS)
    ):
        raise SSMaxPerceptionEvidenceError("Manifest source/step/window contract differs")
    examples = _positive_int(
        evaluation["examples_per_source"],
        name="examples per source",
        minimum=MIN_EXAMPLES_PER_SOURCE,
    )
    samples = _positive_int(evaluation["bootstrap_samples"], name="bootstrap samples")
    if samples < MIN_BOOTSTRAP_SAMPLES:
        raise SSMaxPerceptionEvidenceError(
            f"Promotion requires at least {MIN_BOOTSTRAP_SAMPLES} bootstrap samples"
        )
    for name in ("pairing_seed", "bootstrap_seed"):
        _positive_int(evaluation[name], name=name, minimum=0)
    if evaluation["pairing_seed"] != PAIRING_SEED:
        raise SSMaxPerceptionEvidenceError(
            f"Perception donor pairing seed must be {PAIRING_SEED}, independently of the "
            f"projection seed {PROJECTION_SEED}"
        )
    rank_batch = _positive_int(evaluation["rank_batch_instances"], name="rank batch instances")

    topology = _exact_fields(
        spec["topology"],
        frozenset({"world_size", "num_nodes", "gpus_per_node", "data_parallel"}),
        name="manifest topology",
    )
    world_size = _positive_int(topology["world_size"], name="world size")
    nodes = _positive_int(topology["num_nodes"], name="node count")
    gpus = _positive_int(topology["gpus_per_node"], name="GPUs per node")
    if world_size != nodes * gpus or topology["data_parallel"] != "hsdp":
        raise SSMaxPerceptionEvidenceError("SSMax perception requires a complete HSDP world")
    if examples % (world_size * rank_batch):
        raise SSMaxPerceptionEvidenceError(
            "Examples per source must divide the global evaluation instance batch"
        )

    policy = _exact_fields(
        spec["policy"],
        frozenset(
            {
                "baseline_step",
                "durability_step",
                "candidate_step",
                "did_lower_ci_minimum",
                "treatment_gap_lower_ci_minimum",
                "correct_ce_max_relative_increase",
                "minimum_gap_retention",
                "loss_mass_share_tolerance",
                "maximum_data_errors",
                "maximum_optimizer_guard_skips",
            }
        ),
        name="manifest policy",
    )
    locked = {
        "baseline_step": 0,
        "durability_step": 3000,
        "candidate_step": 4000,
        "did_lower_ci_minimum": 0.0,
        "treatment_gap_lower_ci_minimum": 0.0,
        "correct_ce_max_relative_increase": CORRECT_CE_MAX_RELATIVE_INCREASE,
        "minimum_gap_retention": MINIMUM_GAP_RETENTION,
        "loss_mass_share_tolerance": LOSS_MASS_SHARE_TOLERANCE,
        "maximum_data_errors": 0,
        "maximum_optimizer_guard_skips": 0,
    }
    if dict(policy) != locked:
        raise SSMaxPerceptionEvidenceError(
            "Manifest promotion policy differs from the locked policy"
        )


def load_manifest_spec(path: Path) -> Mapping[str, Any]:
    """Load and validate one checked-in, non-runnable pair manifest specification."""

    spec = _exact_fields(load_json(path), _SPEC_FIELDS, name="SSMax perception manifest spec")
    _validate_spec_common(spec)
    return spec


def _validate_text_sentinel(path: Path) -> Mapping[str, Any]:
    sentinel = _exact_fields(
        load_json(path),
        frozenset({"format", "version", "tokenizer", "input_ids", "labels", "content_sha256"}),
        name="native text sentinel",
    )
    if (
        sentinel["format"] != "vision_alignment_ssmax_native_text_sentinel"
        or sentinel["version"] != 1
    ):
        raise SSMaxPerceptionEvidenceError("Native text sentinel identity is incompatible")
    input_ids = sentinel["input_ids"]
    labels = sentinel["labels"]
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or len(input_ids) != len(labels)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in input_ids
        )
        or any(
            isinstance(item, bool) or not isinstance(item, int) or (item < 0 and item != -100)
            for item in labels
        )
        or any(item in IMAGE_TOKEN_ROWS for item in input_ids)
    ):
        raise SSMaxPerceptionEvidenceError("Native text sentinel token rows are malformed")
    tokenizer = _mapping(sentinel["tokenizer"], name="native text sentinel tokenizer")
    if set(tokenizer) != {"identifier", "revision"} or any(
        not isinstance(tokenizer[name], str) or not tokenizer[name] for name in tokenizer
    ):
        raise SSMaxPerceptionEvidenceError("Native text sentinel tokenizer pin is malformed")
    content_sha = _sha256(sentinel["content_sha256"], name="native text sentinel content SHA")
    if (
        canonical_sha256({key: item for key, item in sentinel.items() if key != "content_sha256"})
        != content_sha
    ):
        raise SSMaxPerceptionEvidenceError("Native text sentinel content SHA-256 differs")
    return sentinel


def _validate_pairing_reference(
    value: Mapping[str, str],
    *,
    source: str,
    evaluation: Mapping[str, Any],
    verify_live: bool,
    dataset_size: int | None = None,
    expected_content_ids_sha256: str | None = None,
) -> None:
    _artifact_reference_shape(value, name=f"{source} pairing")
    if not verify_live:
        return
    path = validate_artifact_reference(value, name=f"{source} pairing")
    pairing = load_json(path)
    if not isinstance(pairing, Mapping):
        raise SSMaxPerceptionEvidenceError(f"{source} pairing must be an object")
    try:
        validate_matched_wrong_image_pairing(
            pairing,
            dataset_size=dataset_size,
            recipient_count=int(evaluation["examples_per_source"]),
            seed=int(evaluation["pairing_seed"]),
            epoch=0,
            content_ids_sha256=expected_content_ids_sha256,
        )
        pairing_sha = matched_wrong_image_pairing_sha256(pairing)
    except ValueError as error:
        raise SSMaxPerceptionEvidenceError(f"{source} pairing is invalid: {error}") from error
    if pairing_sha != value["sha256"]:
        raise SSMaxPerceptionEvidenceError(f"{source} canonical pairing SHA-256 differs")


def _validate_attention_probe_reference(
    value: Mapping[str, str],
    *,
    provenance: Any | None,
    projection_contract: Mapping[str, Any] | None,
    verify_live: bool,
) -> None:
    """Bind the fixed probe to exact logical rows of the perception provenance artifact."""

    _artifact_reference_shape(value, name="SSMax attention probe")
    if not verify_live:
        return
    if provenance is None:
        raise SSMaxPerceptionEvidenceError("Live attention validation requires provenance")
    path = validate_artifact_reference(value, name="SSMax attention probe")
    try:
        probe = SSMaxProbeManifest.load(
            path,
            expected_sha256=value["sha256"],
            verify_validation_manifest=True,
        )
    except ValueError as error:
        raise SSMaxPerceptionEvidenceError(f"SSMax attention probe is invalid: {error}") from error
    validation = _mapping(
        probe.payload.get("validation_manifest"), name="attention probe provenance"
    )
    if (
        Path(str(validation.get("path"))).expanduser().resolve() != provenance.path
        or validation.get("sha256") != provenance.raw_sha256
    ):
        raise SSMaxPerceptionEvidenceError(
            "SSMax perception attention probe names a different provenance artifact"
        )
    population = _mapping(probe.payload.get("population"), name="attention probe population")
    if (
        population.get("source") != "pixmo_caption"
        or population.get("split") != "validation"
        or population.get("epoch") != 0
        or population.get("row_selection_algorithm") != "sha256-priority-over-content-id-v1"
        or len(probe.rows_by_sample_id) != ATTENTION_PROBE_ROWS
    ):
        raise SSMaxPerceptionEvidenceError(
            "SSMax perception attention probe population contract differs"
        )
    indices = population.get("selected_dataset_indices")
    content_ids = population.get("selected_content_ids")
    selection = provenance.selection("pixmo_caption", "validation")
    if (
        not isinstance(indices, list)
        or not isinstance(content_ids, list)
        or len(indices) != ATTENTION_PROBE_ROWS
        or len(content_ids) != ATTENTION_PROBE_ROWS
        or any(
            type(index) is not int or not 0 <= index < len(selection.indices) for index in indices
        )
        or content_ids != [selection.row_image_content_sha256[int(index)] for index in indices]
    ):
        raise SSMaxPerceptionEvidenceError(
            "SSMax perception attention probe row/content identities differ from provenance"
        )
    live_dataset = _mapping(population.get("live_dataset"), name="attention probe live dataset")
    if projection_contract is None:
        raise SSMaxPerceptionEvidenceError(
            "Live attention validation requires the single-response contract"
        )
    base_fingerprint = selection.runtime_dataset_fingerprint
    projected_fingerprint = canonical_sha256(
        {
            "version": "ssmax-single-response-dataset-v1",
            "base_content_fingerprint": base_fingerprint,
            "source": "pixmo_caption",
            "logical_split": "validation",
            "projection_contract_sha256": projection_contract["content_sha256"],
        }
    )
    required_live = {
        "contract": "perception-provenance-selected-validation-v1",
        "dataset_fingerprint": projected_fingerprint,
        "base_dataset_fingerprint": base_fingerprint,
        "examples": len(selection.indices),
        "logical_split": "validation",
        "physical_split": selection.physical_split,
        "selection_indices_sha256": selection.selection_indices_sha256,
        "row_image_content_sha256": content_ids_sha256(selection.row_image_content_sha256),
        "provenance_content_sha256": provenance.content_sha256,
        "source_spec_sha256": provenance.source_spec_sha256,
        "single_response_projection": dict(projection_contract),
    }
    if dict(live_dataset) != required_live:
        raise SSMaxPerceptionEvidenceError(
            "SSMax perception attention probe live-dataset identity differs"
        )


def _checkpoint_reference(
    value: Any, *, step: int, verify_live: bool, workers: int
) -> Mapping[str, Any]:
    reference = _exact_fields(value, _CHECKPOINT_FIELDS, name=f"step{step} checkpoint")
    try:
        return bridge.validate_checkpoint_reference(
            reference,
            expected_step=step,
            verify_live=verify_live,
            workers=workers,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error


def _git_identity(value: Any) -> Mapping[str, str]:
    git = _exact_fields(value, frozenset({"repo", "repo_url", "ref"}), name="saved git identity")
    for name in ("repo", "repo_url"):
        if not isinstance(git[name], str) or not git[name]:
            raise SSMaxPerceptionEvidenceError(f"Saved git {name} must be non-empty")
    ref = git["ref"]
    if not isinstance(ref, str) or re.fullmatch(r"[0-9a-f]{40}", ref) is None:
        raise SSMaxPerceptionEvidenceError("Saved git ref must be a 40-character commit SHA")
    return {name: str(git[name]) for name in ("repo", "repo_url", "ref")}


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
            raise SSMaxPerceptionEvidenceError(
                f"Manifest {producer} producer is not the canonical repository source"
            )
        _sha256(reference["sha256"], name=f"manifest {producer} producer SHA-256")
        if reference["git_ref"] != git["ref"]:
            raise SSMaxPerceptionEvidenceError(
                f"Manifest {producer} producer git ref differs from the manifest"
            )
        validated[producer] = {
            name: str(reference[name]) for name in ("repo_relative_path", "sha256", "git_ref")
        }
    return validated


def _producer_source_references(
    git: Mapping[str, str], *, repository_root: Path
) -> dict[str, dict[str, str]]:
    """Build exact evidence-producer references from a clean saved Git checkout."""

    try:
        bridge._validate_repository_checkout(git, repository_root=repository_root)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error
    references: dict[str, dict[str, str]] = {}
    for producer, relative in PRODUCER_RELATIVE_PATHS.items():
        path = (repository_root / relative).resolve()
        if path != repository_root.resolve() / relative or not path.is_file():
            raise SSMaxPerceptionEvidenceError(f"Canonical {producer} producer source is absent")
        try:
            blob = bridge._git_blob_bytes(
                git,
                repository_root=repository_root,
                repo_relative_path=relative,
                name=f"{producer} producer",
            )
        except bridge.SSMaxBridgeEvidenceError as error:
            raise SSMaxPerceptionEvidenceError(str(error)) from error
        live_sha = sha256_file(path)
        if hashlib.sha256(blob).hexdigest() != live_sha:
            raise SSMaxPerceptionEvidenceError(
                f"Live {producer} producer bytes differ from the manifest git blob"
            )
        references[producer] = {
            "repo_relative_path": relative,
            "sha256": live_sha,
            "git_ref": git["ref"],
        }
    return references


def validate_manifest_producer_source(
    manifest: Mapping[str, Any], *, producer: str, source_path: Path
) -> dict[str, str]:
    """Prove an evidence producer is the canonical blob in the manifest's clean checkout."""

    if producer not in PRODUCER_RELATIVE_PATHS:
        raise SSMaxPerceptionEvidenceError(f"Unknown perception evidence producer {producer!r}")
    git = _git_identity(manifest.get("git"))
    references = _validate_producer_source_references(manifest.get("producers"), git=git)
    expected_relative = PRODUCER_RELATIVE_PATHS[producer]
    source = source_path.expanduser().resolve()
    repository_root = source
    for _ in Path(expected_relative).parts:
        repository_root = repository_root.parent
    if source != (repository_root / expected_relative).resolve():
        raise SSMaxPerceptionEvidenceError(
            f"Running {producer} producer is not the canonical repository source"
        )
    try:
        bridge._validate_repository_checkout(git, repository_root=repository_root)
        blob = bridge._git_blob_bytes(
            git,
            repository_root=repository_root,
            repo_relative_path=expected_relative,
            name=f"{producer} producer",
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error
    reference = references[producer]
    if (
        sha256_file(source) != reference["sha256"]
        or hashlib.sha256(blob).hexdigest() != reference["sha256"]
    ):
        raise SSMaxPerceptionEvidenceError(
            f"Running {producer} producer differs from its manifest source identity"
        )
    return dict(reference)


def _validate_calibration_git_blobs(
    git: Mapping[str, str], *, recipe_path: Path, calibration: Mapping[str, Any]
) -> None:
    """Bind the calibration producer and projection implementation to the saved run git ref."""

    calibration_path = validate_artifact_reference(
        {"path": calibration["path"], "sha256": calibration["sha256"]},
        name="SSMax projection calibration",
    )
    payload = _mapping(load_json(calibration_path), name="SSMax projection calibration")
    repository_root = recipe_path.resolve().parents[3]
    expected_paths = {
        "producer": "src/scripts/data/build_ssmax_single_response_calibration.py",
        "projection_implementation": ("src/olmo_core/data/multimodal/ssmax_single_response.py"),
    }
    for name, expected_relative in expected_paths.items():
        reference = _exact_fields(
            payload.get(name), frozenset({"path", "sha256"}), name=f"calibration {name}"
        )
        if reference["path"] != expected_relative:
            raise SSMaxPerceptionEvidenceError(
                f"Calibration {name} is not the canonical checked-in source"
            )
        expected_sha = _sha256(reference["sha256"], name=f"calibration {name} SHA-256")
        live_path = (repository_root / expected_relative).resolve()
        if (
            not live_path.is_relative_to(repository_root)
            or not live_path.is_file()
            or sha256_file(live_path) != expected_sha
        ):
            raise SSMaxPerceptionEvidenceError(f"Live calibration {name} bytes differ")
        try:
            blob = subprocess.check_output(
                ["git", "-C", str(repository_root), "show", f"{git['ref']}:{expected_relative}"],
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise SSMaxPerceptionEvidenceError(
                f"Could not read calibration {name} from the saved git ref"
            ) from error
        if hashlib.sha256(blob).hexdigest() != expected_sha:
            raise SSMaxPerceptionEvidenceError(
                f"Calibration {name} differs from the saved git blob"
            )


def _validate_bridge_parent(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    model_variant: str,
    gate_reference: Mapping[str, str],
    verify_live_checkpoint: bool,
) -> dict[str, Any]:
    initializations = {
        arm: _mapping(configs[arm].get("initialization"), name=f"{arm} initialization")
        for arm in ARMS
    }
    if initializations[CONTROL_ARM] != initializations[TREATMENT_ARM]:
        raise SSMaxPerceptionEvidenceError("Perception arms do not share an exact bridge parent")
    initialization = initializations[TREATMENT_ARM]
    parent_path = Path(str(initialization.get("checkpoint"))).expanduser().resolve()
    config_path = parent_path / "config.json"
    if not config_path.is_file():
        raise SSMaxPerceptionEvidenceError(f"Bridge parent config is absent: {config_path}")
    expected_config_sha = _sha256(
        initialization.get("parent_config_sha256"), name="bridge parent config SHA-256"
    )
    if sha256_file(config_path) != expected_config_sha:
        raise SSMaxPerceptionEvidenceError("Bridge parent config differs from its saved pin")
    gate_path = validate_artifact_reference(gate_reference, name="bridge parent gate")
    if gate_path != Path(
        str(initialization.get("parent_gate_path"))
    ).expanduser().resolve() or gate_reference["sha256"] != _sha256(
        initialization.get("parent_gate_sha256"), name="bridge parent gate SHA-256"
    ):
        raise SSMaxPerceptionEvidenceError("Saved initialization names a different bridge gate")
    parent_config = _mapping(load_json(config_path), name="bridge parent config")
    parent_metadata = _mapping(parent_config.get("vision_alignment"), name="bridge parent metadata")
    if (
        parent_config.get("model_variant") != model_variant
        or parent_config.get("phase") != "bridge"
        or parent_metadata.get("model_variant") != model_variant
        or parent_metadata.get("phase") != "bridge"
    ):
        raise SSMaxPerceptionEvidenceError("Bridge parent model lineage is incompatible")
    gate = _mapping(load_json(gate_path), name="bridge parent gate")
    try:
        summary = bridge.validate_ssmax_bridge_parent_gate(
            gate,
            expected_checkpoint=parent_path,
            expected_checkpoint_config_sha256=expected_config_sha,
            expected_model_variant=model_variant,
            expected_data_contract_sha256=str(parent_metadata.get("data_contract_sha256")),
            expected_trainable_contract_sha256=str(
                parent_metadata.get("trainable_contract_sha256")
            ),
            verify_live_checkpoint=verify_live_checkpoint,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(f"Bridge v4 gate failed validation: {error}") from error
    candidate = _mapping(summary.get("candidate"), name="bridge promotion candidate")
    return {
        "checkpoint": str(parent_path),
        "checkpoint_config_sha256": expected_config_sha,
        "checkpoint_identity_sha256": _sha256(
            candidate.get("identity_sha256"), name="bridge checkpoint identity"
        ),
        "data_contract_sha256": _sha256(
            parent_metadata.get("data_contract_sha256"), name="bridge data contract"
        ),
        "trainable_contract_sha256": _sha256(
            parent_metadata.get("trainable_contract_sha256"), name="bridge trainable contract"
        ),
        "gate": dict(gate_reference),
        "gate_semantic_sha256": canonical_sha256(gate),
    }


def _pair_contract_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": manifest["pair_id"],
        "model_variant": manifest["model_variant"],
        "git": manifest["git"],
        "producers": manifest["producers"],
        "recipe": manifest["recipe"],
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
        "arms": {
            arm: {
                "run_name": manifest["arms"][arm]["run_name"],
                "training_profile": manifest["arms"][arm]["training_profile"],
                "data_contract_sha256": manifest["arms"][arm]["data_contract_sha256"],
                "trainable_contract_sha256": manifest["arms"][arm]["trainable_contract_sha256"],
            }
            for arm in ARMS
        },
    }


def build_manifest(
    spec: Mapping[str, Any], *, created_at: str, hash_workers: int = 8
) -> dict[str, Any]:
    """Finalize a causal-pair manifest after every required checkpoint exists."""

    _exact_fields(spec, _SPEC_FIELDS, name="SSMax perception manifest spec")
    _validate_spec_common(spec)
    _timestamp(created_at, name="manifest created_at")
    if hash_workers <= 0:
        raise ValueError("hash_workers must be positive")
    arm_specs = _mapping(spec["arms"], name="manifest spec arms")
    roots = {
        arm: Path(str(arm_specs[arm]["checkpoint_root"])).expanduser().resolve() for arm in ARMS
    }
    missing = [
        str(roots[arm] / f"step{step}")
        for arm in ARMS
        for step in REQUIRED_STEPS
        if not (roots[arm] / f"step{step}").is_dir()
    ]
    if missing:
        raise SSMaxPerceptionEvidenceError(
            "Pair manifest cannot be built until both completed runs exist; missing "
            + ", ".join(missing)
        )
    checkpoints = {
        arm: {
            str(step): bridge.checkpoint_identity(roots[arm] / f"step{step}", workers=hash_workers)
            for step in REQUIRED_STEPS
        }
        for arm in ARMS
    }
    for arm in ARMS:
        if len({item["config_sha256"] for item in checkpoints[arm].values()}) != 1:
            raise SSMaxPerceptionEvidenceError(f"{arm} checkpoints do not share one config")
        if {item["trainer_state_count"] for item in checkpoints[arm].values()} != {
            spec["topology"]["world_size"]
        }:
            raise SSMaxPerceptionEvidenceError(f"{arm} trainer-state world size differs")

    recipe_path = Path(str(spec["recipe"])).expanduser().resolve()
    recipe_reference = artifact_reference(recipe_path)
    profiles = {
        arm: artifact_reference(Path(str(arm_specs[arm]["training_profile"]))) for arm in ARMS
    }
    configs = {
        arm: _mapping(load_json(roots[arm] / "step0" / "config.json"), name=f"{arm} config")
        for arm in ARMS
    }
    summaries = validate_saved_config_pair(
        configs,
        spec=spec,
        profile_references=profiles,
        recipe_path=recipe_path,
    )
    git = _git_identity(summaries[TREATMENT_ARM]["git"])
    try:
        for arm in ARMS:
            bridge._validate_saved_git_checkout(
                git,
                recipe_path=recipe_path,
                profile_path=Path(profiles[arm]["path"]),
            )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error
    repository_root = recipe_path.resolve().parents[3]
    producers = _producer_source_references(git, repository_root=repository_root)

    gate_reference = artifact_reference(Path(str(spec["bridge_parent_gate"])))
    bridge_parent = _validate_bridge_parent(
        configs,
        model_variant=str(spec["model_variant"]),
        gate_reference=gate_reference,
        verify_live_checkpoint=True,
    )
    data = _mapping(configs[TREATMENT_ARM].get("data"), name="perception data")
    target_values = _mapping(
        _mapping(configs[TREATMENT_ARM].get("train_module"), name="treatment train module").get(
            "source_loss_mass_targets"
        ),
        name="source loss-mass targets",
    )
    control_targets = _mapping(
        _mapping(configs[CONTROL_ARM].get("train_module"), name="control train module").get(
            "source_loss_mass_targets"
        ),
        name="control source loss-mass targets",
    )
    if target_values != control_targets or set(target_values) != set(SOURCES):
        raise SSMaxPerceptionEvidenceError("Perception arm source loss-mass targets differ")
    loss_mass_targets = {
        source: _finite(target_values[source], name=f"{source} loss-mass target")
        for source in SOURCES
    }
    if any(value <= 0 for value in loss_mass_targets.values()) or not math.isclose(
        sum(loss_mass_targets.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SSMaxPerceptionEvidenceError("Perception source loss-mass targets are invalid")
    provenance_reference = artifact_reference(Path(str(spec["perception_provenance"])))
    source_audit_reference = artifact_reference(Path(str(spec["source_audit"])))
    if (
        Path(str(data.get("perception_provenance_path"))).expanduser().resolve()
        != Path(provenance_reference["path"])
        or data.get("perception_provenance_sha256") != provenance_reference["sha256"]
        or Path(str(data.get("source_audit_path"))).expanduser().resolve()
        != Path(source_audit_reference["path"])
    ):
        raise SSMaxPerceptionEvidenceError("Saved perception data artifacts differ from the spec")
    provenance = _mapping(
        load_json(Path(provenance_reference["path"])), name="perception provenance"
    )
    provenance_content_sha = _sha256(
        provenance.get("content_sha256"), name="perception provenance content SHA"
    )
    if (
        canonical_sha256({key: item for key, item in provenance.items() if key != "content_sha256"})
        != provenance_content_sha
    ):
        raise SSMaxPerceptionEvidenceError("Perception provenance content SHA-256 differs")
    typed_provenance = load_perception_provenance_manifest(
        provenance_reference["path"],
        expected_sha256=provenance_reference["sha256"],
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    source_audit = _mapping(load_json(Path(source_audit_reference["path"])), name="source audit")
    unsigned_audit = dict(source_audit)
    recorded_fingerprint = unsigned_audit.pop("fingerprint", None)
    fingerprint = _sha256(recorded_fingerprint, name="source audit fingerprint")
    if (
        canonical_sha256(unsigned_audit) != fingerprint
        or data.get("source_audit_fingerprint") != fingerprint
    ):
        raise SSMaxPerceptionEvidenceError("Perception source-audit fingerprint differs")

    text_sentinel_reference = artifact_reference(Path(str(spec["text_sentinel"])))
    sentinel = _validate_text_sentinel(Path(text_sentinel_reference["path"]))
    artifacts = _mapping(configs[TREATMENT_ARM].get("artifacts"), name="treatment artifacts")
    if sentinel["tokenizer"] != {
        "identifier": artifacts.get("tokenizer_id"),
        "revision": artifacts.get("tokenizer_revision"),
    }:
        raise SSMaxPerceptionEvidenceError(
            "Native text sentinel tokenizer differs from the paired training tokenizer"
        )
    attention_probe_reference = artifact_reference(Path(str(spec["attention_probe"])))
    single_response_projection = _single_response_binding_from_config(configs[TREATMENT_ARM])
    _validate_attention_probe_reference(
        attention_probe_reference,
        provenance=typed_provenance,
        projection_contract=single_response_projection["contract"],
        verify_live=True,
    )
    pairings: dict[str, dict[str, str]] = {}
    for source in SOURCES:
        reference = artifact_reference(Path(str(spec["pairing_paths"][source])))
        _validate_pairing_reference(
            reference,
            source=source,
            evaluation=spec["evaluation"],
            verify_live=True,
            dataset_size=len(typed_provenance.selection(source, "validation").indices),
            expected_content_ids_sha256=content_ids_sha256(
                typed_provenance.selection(source, "validation").row_image_content_sha256
            ),
        )
        pairings[source] = reference

    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": SCHEMA_VERSION,
        "created_at": created_at,
        "pair_id": spec["pair_id"],
        "model_variant": spec["model_variant"],
        "git": dict(git),
        "producers": producers,
        "recipe": recipe_reference,
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
        "loss_mass_targets": loss_mass_targets,
        "arms": {
            arm: {
                "run_name": arm_specs[arm]["run_name"],
                "checkpoint_root": str(roots[arm]),
                "training_profile": profiles[arm],
                "data_contract_sha256": summaries[arm]["data_contract_sha256"],
                "trainable_contract_sha256": summaries[arm]["trainable_contract_sha256"],
                "checkpoints": checkpoints[arm],
            }
            for arm in ARMS
        },
    }
    manifest["pair_contract_sha256"] = canonical_sha256(_pair_contract_payload(manifest))
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest, verify_live=True, hash_workers=hash_workers)
    return manifest


def validate_manifest(
    value: Any, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Validate a finalized causal-pair manifest and optionally every live byte pin."""

    manifest = _exact_fields(value, _MANIFEST_FIELDS, name="SSMax perception pair manifest")
    if manifest["format"] != MANIFEST_FORMAT or manifest["version"] != SCHEMA_VERSION:
        raise SSMaxPerceptionEvidenceError("SSMax perception pair manifest is incompatible")
    _timestamp(manifest["created_at"], name="manifest created_at")
    if manifest["model_variant"] not in MODEL_VARIANTS:
        raise SSMaxPerceptionEvidenceError("Manifest model variant is unsupported")
    if not isinstance(manifest["pair_id"], str) or not manifest["pair_id"]:
        raise SSMaxPerceptionEvidenceError("Manifest pair_id must be non-empty")
    git = _git_identity(manifest["git"])
    producers = _validate_producer_source_references(manifest["producers"], git=git)
    _artifact_reference_shape(manifest["recipe"], name="training recipe")
    parent = _exact_fields(manifest["bridge_parent"], _BRIDGE_PARENT_FIELDS, name="bridge parent")
    if not isinstance(parent["checkpoint"], str) or not parent["checkpoint"]:
        raise SSMaxPerceptionEvidenceError("Bridge parent checkpoint must be non-empty")
    for field in (
        "checkpoint_config_sha256",
        "checkpoint_identity_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "gate_semantic_sha256",
    ):
        _sha256(parent[field], name=f"bridge parent {field}")
    _artifact_reference_shape(parent["gate"], name="bridge parent gate")
    provenance = _exact_fields(
        manifest["perception_provenance"],
        frozenset({"path", "sha256", "content_sha256"}),
        name="perception provenance reference",
    )
    if not isinstance(provenance["path"], str) or not provenance["path"]:
        raise SSMaxPerceptionEvidenceError("Perception provenance path must be non-empty")
    _sha256(provenance["sha256"], name="perception provenance raw SHA-256")
    _sha256(provenance["content_sha256"], name="perception provenance content SHA-256")
    _artifact_reference_shape(manifest["source_audit"], name="source audit")
    _sha256(manifest["source_audit_fingerprint"], name="source audit fingerprint")
    single_response_projection = _validate_single_response_binding(
        manifest["single_response_projection"], verify_live=verify_live
    )
    _artifact_reference_shape(manifest["attention_probe"], name="SSMax attention probe")
    _artifact_reference_shape(manifest["text_sentinel"], name="native text sentinel")

    pseudo_spec: dict[str, Any] = {
        "format": MANIFEST_SPEC_FORMAT,
        "version": SCHEMA_VERSION,
        "pair_id": manifest["pair_id"],
        "model_variant": manifest["model_variant"],
        "arms": {},
        "recipe": manifest["recipe"]["path"],
        "bridge_parent_gate": parent["gate"]["path"],
        "perception_provenance": provenance["path"],
        "source_audit": manifest["source_audit"]["path"],
        "attention_probe": manifest["attention_probe"]["path"],
        "text_sentinel": manifest["text_sentinel"]["path"],
        "pairing_paths": {},
        "evaluation": manifest["evaluation"],
        "topology": manifest["topology"],
        "policy": manifest["policy"],
    }
    arms = _exact_fields(manifest["arms"], frozenset(ARMS), name="manifest arms")
    for arm in ARMS:
        arm_value = _exact_fields(arms[arm], _ARM_MANIFEST_FIELDS, name=f"{arm} manifest arm")
        for field in ("run_name", "checkpoint_root"):
            if not isinstance(arm_value[field], str) or not arm_value[field]:
                raise SSMaxPerceptionEvidenceError(f"{arm} {field} must be non-empty")
        _artifact_reference_shape(arm_value["training_profile"], name=f"{arm} profile")
        for field in ("data_contract_sha256", "trainable_contract_sha256"):
            _sha256(arm_value[field], name=f"{arm} {field}")
        checkpoints = _exact_fields(
            arm_value["checkpoints"],
            frozenset(str(step) for step in REQUIRED_STEPS),
            name=f"{arm} checkpoints",
        )
        for step in REQUIRED_STEPS:
            reference = _checkpoint_reference(
                checkpoints[str(step)],
                step=step,
                verify_live=verify_live,
                workers=hash_workers,
            )
            if (
                Path(str(reference["path"])).resolve().parent
                != Path(str(arm_value["checkpoint_root"])).expanduser().resolve()
            ):
                raise SSMaxPerceptionEvidenceError(f"{arm} step{step} is outside its run root")
        if len({item["config_sha256"] for item in checkpoints.values()}) != 1:
            raise SSMaxPerceptionEvidenceError(f"{arm} checkpoints do not share one config")
        if {item["trainer_state_count"] for item in checkpoints.values()} != {
            manifest["topology"]["world_size"]
        }:
            raise SSMaxPerceptionEvidenceError(f"{arm} trainer rank-state counts differ")
        pseudo_spec["arms"][arm] = {
            "run_name": arm_value["run_name"],
            "checkpoint_root": arm_value["checkpoint_root"],
            "training_profile": arm_value["training_profile"]["path"],
        }

    typed_provenance = None
    if verify_live:
        typed_provenance = load_perception_provenance_manifest(
            provenance["path"],
            expected_sha256=provenance["sha256"],
            verify_finevision_materialization=False,
            load_image_path_signatures=False,
        )
    pairings = _exact_fields(manifest["pairings"], frozenset(SOURCES), name="manifest pairings")
    for source in SOURCES:
        selection = (
            typed_provenance.selection(source, "validation")
            if typed_provenance is not None
            else None
        )
        _validate_pairing_reference(
            pairings[source],
            source=source,
            evaluation=manifest["evaluation"],
            verify_live=verify_live,
            dataset_size=len(selection.indices) if selection is not None else None,
            expected_content_ids_sha256=(
                content_ids_sha256(selection.row_image_content_sha256)
                if selection is not None
                else None
            ),
        )
        pseudo_spec["pairing_paths"][source] = pairings[source]["path"]
    _validate_spec_common(pseudo_spec)
    targets = _exact_fields(
        manifest["loss_mass_targets"], frozenset(SOURCES), name="manifest loss-mass targets"
    )
    target_sum = 0.0
    for source in SOURCES:
        target = _finite(targets[source], name=f"{source} loss-mass target")
        if target <= 0:
            raise SSMaxPerceptionEvidenceError("Loss-mass targets must be positive")
        target_sum += target
    if not math.isclose(target_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SSMaxPerceptionEvidenceError("Loss-mass targets must sum to one")

    expected_pair_contract = _sha256(manifest["pair_contract_sha256"], name="pair contract SHA-256")
    if canonical_sha256(_pair_contract_payload(manifest)) != expected_pair_contract:
        raise SSMaxPerceptionEvidenceError("Manifest pair contract SHA-256 differs")
    content_sha = _sha256(manifest["content_sha256"], name="manifest content SHA-256")
    if (
        canonical_sha256({key: item for key, item in manifest.items() if key != "content_sha256"})
        != content_sha
    ):
        raise SSMaxPerceptionEvidenceError("Manifest content SHA-256 differs")

    if verify_live:
        recipe_path = validate_artifact_reference(manifest["recipe"], name="training recipe")
        profile_paths = {
            arm: validate_artifact_reference(
                arms[arm]["training_profile"], name=f"{arm} training profile"
            )
            for arm in ARMS
        }
        configs = {
            arm: _mapping(
                load_json(Path(str(arms[arm]["checkpoint_root"])) / "step0" / "config.json"),
                name=f"{arm} saved config",
            )
            for arm in ARMS
        }
        summaries = validate_saved_config_pair(
            configs,
            spec=pseudo_spec,
            profile_references={arm: arms[arm]["training_profile"] for arm in ARMS},
            recipe_path=recipe_path,
        )
        live_projection = _single_response_binding_from_config(configs[TREATMENT_ARM])
        if live_projection != dict(single_response_projection):
            raise SSMaxPerceptionEvidenceError(
                "Live checkpoint single-response projection differs from manifest"
            )
        for arm in ARMS:
            if (
                summaries[arm]["data_contract_sha256"] != arms[arm]["data_contract_sha256"]
                or summaries[arm]["trainable_contract_sha256"]
                != arms[arm]["trainable_contract_sha256"]
            ):
                raise SSMaxPerceptionEvidenceError(f"{arm} saved contracts differ from manifest")
            try:
                bridge._validate_saved_git_checkout(
                    manifest["git"], recipe_path=recipe_path, profile_path=profile_paths[arm]
                )
            except bridge.SSMaxBridgeEvidenceError as error:
                raise SSMaxPerceptionEvidenceError(str(error)) from error
        repository_root = recipe_path.resolve().parents[3]
        if _producer_source_references(git, repository_root=repository_root) != producers:
            raise SSMaxPerceptionEvidenceError(
                "Manifest evidence producer sources differ from the saved git checkout"
            )
        _validate_calibration_git_blobs(
            git,
            recipe_path=recipe_path,
            calibration=_mapping(
                single_response_projection["calibration"],
                name="SSMax projection calibration reference",
            ),
        )
        actual_parent = _validate_bridge_parent(
            configs,
            model_variant=str(manifest["model_variant"]),
            gate_reference=parent["gate"],
            verify_live_checkpoint=True,
        )
        if actual_parent != dict(parent):
            raise SSMaxPerceptionEvidenceError("Live bridge parent differs from the manifest")
        provenance_path = validate_artifact_reference(
            {"path": provenance["path"], "sha256": provenance["sha256"]},
            name="perception provenance",
        )
        provenance_payload = _mapping(load_json(provenance_path), name="perception provenance")
        if provenance_payload.get("content_sha256") != provenance["content_sha256"]:
            raise SSMaxPerceptionEvidenceError("Live perception provenance semantic SHA differs")
        audit_path = validate_artifact_reference(manifest["source_audit"], name="source audit")
        audit = _mapping(load_json(audit_path), name="source audit")
        unsigned_audit = dict(audit)
        recorded = unsigned_audit.pop("fingerprint", None)
        if (
            recorded != manifest["source_audit_fingerprint"]
            or canonical_sha256(unsigned_audit) != recorded
        ):
            raise SSMaxPerceptionEvidenceError("Live source-audit fingerprint differs")
        sentinel_path = validate_artifact_reference(
            manifest["text_sentinel"], name="native text sentinel"
        )
        _validate_text_sentinel(sentinel_path)
        _validate_attention_probe_reference(
            manifest["attention_probe"],
            provenance=typed_provenance,
            projection_contract=_mapping(
                manifest["single_response_projection"],
                name="single-response projection",
            )["contract"],
            verify_live=True,
        )
    return manifest


def load_manifest(
    path: Path, *, verify_live: bool = True, hash_workers: int = 8
) -> Mapping[str, Any]:
    """Load and validate a finalized SSMax perception pair manifest."""

    return validate_manifest(load_json(path), verify_live=verify_live, hash_workers=hash_workers)


_EVALUATION_RECEIPT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "manifest",
        "pair_id",
        "model_variant",
        "arm",
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
        "pair_id",
        "model_variant",
        "arm",
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
_STATE_FIELDS = frozenset({"full_model", "frozen_lm", "non_image_embedding_rows", "vision"})
_FULL_MODEL_FIELDS = frozenset({"protocol", "tensor_count", "inventory_sha256"})
_SURFACE_FIELDS = frozenset(
    {
        "protocol",
        "tensor_count",
        "reference_inventory_sha256",
        "candidate_inventory_sha256",
        "mismatch_count",
    }
)
_TEXT_SENTINEL_RESULT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "input_sha256",
        "labels_sha256",
        "token_count",
        "logits_sha256",
        "ce_sha256",
    }
)
_RANK_STATE_FIELDS = frozenset(
    {
        "rank",
        "global_step",
        "batches_processed",
        "data_loader_state_sha256",
        "trainer_state_sha256",
        "trainer_state_size_bytes",
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


def _validate_manifest_reference(
    value: Any, manifest: Mapping[str, Any], *, expected_path: Path, name: str
) -> None:
    reference = _exact_fields(value, _MANIFEST_REF_FIELDS, name=f"{name} manifest reference")
    for field in ("sha256", "content_sha256"):
        _sha256(reference[field], name=f"{name} manifest {field}")
    if reference["content_sha256"] != manifest["content_sha256"]:
        raise SSMaxPerceptionEvidenceError(f"{name} names a different manifest semantic SHA")
    path = validate_artifact_reference(
        {"path": reference["path"], "sha256": reference["sha256"]},
        name=f"{name} manifest",
    )
    if path != expected_path.expanduser().resolve():
        raise SSMaxPerceptionEvidenceError(f"{name} names a different manifest path")


def _validate_content_sha(payload: Mapping[str, Any], *, name: str) -> None:
    expected = _sha256(payload.get("content_sha256"), name=f"{name} content SHA-256")
    actual = canonical_sha256(
        {key: item for key, item in payload.items() if key != "content_sha256"}
    )
    if actual != expected:
        raise SSMaxPerceptionEvidenceError(f"{name} content SHA-256 differs")


def _validate_strict_load(value: Any, *, name: str) -> None:
    load = _mapping(value, name=name)
    required = {
        "complete",
        "strict",
        "load_completed",
        "checkpoint_key_count",
        "model_tensor_count",
        "model_parameter_tensor_count",
        "model_buffer_tensor_count",
        "model_keyset_sha256",
        "model_inventory_sha256",
        "sha256",
    }
    if set(load) != required:
        raise SSMaxPerceptionEvidenceError(f"{name} fields differ")
    if (
        load["complete"] is not True
        or load["strict"] is not True
        or load["load_completed"] is not True
    ):
        raise SSMaxPerceptionEvidenceError(f"{name} is not a completed strict generic DCP load")
    counts = [
        _positive_int(load[field], name=f"{name} {field}", minimum=0)
        for field in (
            "checkpoint_key_count",
            "model_tensor_count",
            "model_parameter_tensor_count",
            "model_buffer_tensor_count",
        )
    ]
    # DCP metadata includes optimizer and trainer entries in addition to model tensors.  A
    # successful strict model load therefore proves that the checkpoint inventory is a
    # superset of the requested model state, not that both inventories have equal cardinality.
    if counts[1] <= 0 or counts[0] < counts[1] or counts[1] != counts[2] + counts[3]:
        raise SSMaxPerceptionEvidenceError(f"{name} model inventory counts differ")
    for field in ("model_keyset_sha256", "model_inventory_sha256", "sha256"):
        _sha256(load[field], name=f"{name} {field}")
    if (
        canonical_sha256({key: item for key, item in load.items() if key != "sha256"})
        != load["sha256"]
    ):
        raise SSMaxPerceptionEvidenceError(f"{name} semantic SHA-256 differs")


def _validate_state(value: Any, *, arm: str, step: int) -> Mapping[str, Any]:
    state = _exact_fields(value, _STATE_FIELDS, name=f"{arm} step{step} state")
    full = _exact_fields(
        state["full_model"], _FULL_MODEL_FIELDS, name=f"{arm} step{step} full model"
    )
    if full["protocol"] != "logical-model-tensor-inventory-sha256-v1":
        raise SSMaxPerceptionEvidenceError("Full-model state protocol differs")
    _positive_int(full["tensor_count"], name="full-model tensor count")
    _sha256(full["inventory_sha256"], name="full-model inventory SHA-256")
    for surface_name in ("frozen_lm", "non_image_embedding_rows", "vision"):
        surface = _exact_fields(
            state[surface_name],
            _SURFACE_FIELDS,
            name=f"{arm} step{step} {surface_name}",
        )
        if surface["protocol"] != "logical-tensor-comparison-sha256-v1":
            raise SSMaxPerceptionEvidenceError(f"{surface_name} state protocol differs")
        count = _positive_int(surface["tensor_count"], name=f"{surface_name} tensor count")
        mismatch = _positive_int(
            surface["mismatch_count"], name=f"{surface_name} mismatch count", minimum=0
        )
        if mismatch > count:
            raise SSMaxPerceptionEvidenceError(f"{surface_name} mismatch count exceeds inventory")
        for field in ("reference_inventory_sha256", "candidate_inventory_sha256"):
            _sha256(surface[field], name=f"{surface_name} {field}")
        if step == 0 and (
            mismatch != 0
            or surface["reference_inventory_sha256"] != surface["candidate_inventory_sha256"]
        ):
            raise SSMaxPerceptionEvidenceError(f"{arm} step0 {surface_name} is not self-equal")
    return state


def _validate_text_result(
    value: Any, *, manifest: Mapping[str, Any], arm: str, step: int
) -> Mapping[str, Any]:
    result = _exact_fields(
        value,
        _TEXT_SENTINEL_RESULT_FIELDS,
        name=f"{arm} step{step} native text sentinel",
    )
    if result["artifact_sha256"] != manifest["text_sentinel"]["sha256"]:
        raise SSMaxPerceptionEvidenceError("Native text result names a different sentinel")
    for field in ("artifact_sha256", "input_sha256", "labels_sha256", "logits_sha256", "ce_sha256"):
        _sha256(result[field], name=f"native text {field}")
    _positive_int(result["token_count"], name="native text token count")
    return result


def _validate_rows(
    rows: Any,
    *,
    source: str,
    manifest: Mapping[str, Any],
    pairing: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    examples = int(manifest["evaluation"]["examples_per_source"])
    if not isinstance(rows, list) or len(rows) != examples:
        raise SSMaxPerceptionEvidenceError(f"{source} must contain exactly {examples} rows")
    pairs = pairing.get("pairs") if pairing is not None else None
    if pairs is not None and (not isinstance(pairs, list) or len(pairs) != examples):
        raise SSMaxPerceptionEvidenceError(f"{source} live pairing row count differs")
    result: list[Mapping[str, Any]] = []
    for position, raw in enumerate(rows):
        row = _exact_fields(raw, _ROW_FIELDS, name=f"{source} row {position}")
        if row["pairing_position"] != position:
            raise SSMaxPerceptionEvidenceError(f"{source} pairing positions are not contiguous")
        for field in ("recipient_index", "donor_index"):
            _positive_int(row[field], name=f"{source} {field}", minimum=0)
        _positive_int(row["response_tokens"], name=f"{source} response tokens")
        if pairs is not None:
            pair = _mapping(pairs[position], name=f"{source} pairing row {position}")
            if row["recipient_index"] != pair.get("recipient") or row["donor_index"] != pair.get(
                "donor"
            ):
                raise SSMaxPerceptionEvidenceError(f"{source} row differs from its pairing")
        windows = {
            name: _exact_fields(
                row[name], frozenset(WINDOWS), name=f"{source} row {position} {name}"
            )
            for name in ("correct_ce", "wrong_ce", "ce_gap_wrong_minus_correct")
        }
        for window in WINDOWS:
            correct = _finite(windows["correct_ce"][window], name="correct CE")
            wrong = _finite(windows["wrong_ce"][window], name="wrong CE")
            gap = _finite(windows["ce_gap_wrong_minus_correct"][window], name="CE gap")
            if (
                correct < 0
                or wrong < 0
                or not math.isclose(gap, wrong - correct, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise SSMaxPerceptionEvidenceError(
                    f"{source} row {position} {window} CE fields are inconsistent"
                )
        result.append(row)
    return result


def _load_receipt_reference(
    reference: Any,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    arm: str,
    step: int,
    expected_format: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = validate_artifact_reference(reference, name=f"{arm} step{step} {expected_format}")
    payload = _mapping(load_json(path), name=f"{arm} step{step} receipt")
    fields = (
        _EVALUATION_RECEIPT_FIELDS
        if expected_format == EVALUATION_RECEIPT_FORMAT
        else _HEALTH_RECEIPT_FIELDS
    )
    _exact_fields(payload, fields, name=f"{arm} step{step} receipt")
    if (
        payload["format"] != expected_format
        or payload["version"] != SCHEMA_VERSION
        or payload["pair_id"] != manifest["pair_id"]
        or payload["model_variant"] != manifest["model_variant"]
        or payload["arm"] != arm
        or payload["step"] != step
        or payload["checkpoint"] != manifest["arms"][arm]["checkpoints"][str(step)]
    ):
        raise SSMaxPerceptionEvidenceError(f"{arm} step{step} receipt identity differs")
    if payload["status"] not in ("passed", "failed"):
        raise SSMaxPerceptionEvidenceError(f"{arm} step{step} receipt status is invalid")
    _timestamp(payload["created_at"], name=f"{arm} step{step} receipt created_at")
    _validate_manifest_reference(
        payload["manifest"],
        manifest,
        expected_path=manifest_path,
        name=f"{arm} step{step}",
    )
    _validate_content_sha(payload, name=f"{arm} step{step} receipt")
    return path, payload


def _validate_evaluation_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], arm: str, step: int
) -> dict[str, list[Mapping[str, Any]]]:
    _validate_strict_load(receipt["strict_generic_dcp_load"], name="strict generic DCP load")
    _validate_state(receipt["state"], arm=arm, step=step)
    _validate_text_result(receipt["text_sentinel"], manifest=manifest, arm=arm, step=step)
    attention = _mapping(
        receipt["attention_diagnostics"], name=f"{arm} step{step} attention diagnostics"
    )
    if (
        attention.get("checkpoint") != manifest["arms"][arm]["checkpoints"][str(step)]
        or not isinstance(attention.get("protocol"), Mapping)
        or attention["protocol"].get("manifest_sha256") != manifest["attention_probe"]["sha256"]
    ):
        raise SSMaxPerceptionEvidenceError(
            f"{arm} step{step} attention diagnostics differ from the manifest"
        )
    try:
        validate_ssmax_attention_report(attention, label=f"{arm} step{step} attention diagnostics")
    except ValueError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error
    if receipt["pairings"] != manifest["pairings"]:
        raise SSMaxPerceptionEvidenceError(f"{arm} step{step} changes fixed pairings")
    evaluator = _exact_fields(
        receipt["evaluator"],
        _PRODUCER_SOURCE_REF_FIELDS,
        name=f"{arm} step{step} evaluator",
    )
    producers = _validate_producer_source_references(
        manifest["producers"], git=_git_identity(manifest["git"])
    )
    if dict(evaluator) != dict(producers[EVALUATION_PRODUCER]):
        raise SSMaxPerceptionEvidenceError("Perception evaluator source identity differs")
    results = _exact_fields(receipt["results"], frozenset(SOURCES), name="evaluation results")
    output: dict[str, list[Mapping[str, Any]]] = {}
    for source in SOURCES:
        result = _exact_fields(
            results[source],
            frozenset({"pairing_sha256", "examples", "per_example"}),
            name=f"{source} result",
        )
        if (
            result["pairing_sha256"] != manifest["pairings"][source]["sha256"]
            or result["examples"] != manifest["evaluation"]["examples_per_source"]
        ):
            raise SSMaxPerceptionEvidenceError(f"{source} result pairing/count differs")
        pairing_path = validate_artifact_reference(
            manifest["pairings"][source], name=f"{source} pairing"
        )
        pairing = _mapping(load_json(pairing_path), name=f"{source} pairing")
        output[source] = _validate_rows(
            result["per_example"], source=source, manifest=manifest, pairing=pairing
        )
    return output


def _validate_health_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], arm: str, step: int
) -> dict[str, Any]:
    rank_states = receipt["rank_states"]
    world_size = int(manifest["topology"]["world_size"])
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise SSMaxPerceptionEvidenceError(f"{arm} step{step} health omits trainer ranks")
    ledgers: list[Mapping[str, Any]] = []
    for rank, raw in enumerate(rank_states):
        state = _exact_fields(raw, _RANK_STATE_FIELDS, name=f"{arm} step{step} rank{rank}")
        if (
            state["rank"] != rank
            or state["global_step"] != step
            or state["batches_processed"] != step
        ):
            raise SSMaxPerceptionEvidenceError(f"{arm} step{step} rank{rank} cursor differs")
        for field in ("data_loader_state_sha256", "trainer_state_sha256"):
            _sha256(state[field], name=f"rank{rank} {field}")
        _positive_int(
            state["trainer_state_size_bytes"],
            name=f"rank{rank} trainer-state byte size",
        )
        try:
            ledger = validate_ssmax_health_ledger_state(
                state["health_ledger"],
                expected_model_variant=str(manifest["model_variant"]),
                expected_phase="perception",
                expected_run_name=str(manifest["arms"][arm]["run_name"]),
                expected_step=step,
            )
        except SSMaxHealthLedgerError as error:
            raise SSMaxPerceptionEvidenceError(
                f"{arm} step{step} rank{rank} health ledger is invalid: {error}"
            ) from error
        ledgers.append(ledger)
    trainer_inventory = [
        {
            "rank": rank,
            "path": f"train/rank{rank}.pt",
            "size": state["trainer_state_size_bytes"],
            "sha256": state["trainer_state_sha256"],
        }
        for rank, state in enumerate(rank_states)
    ]
    if (
        canonical_sha256(trainer_inventory)
        != manifest["arms"][arm]["checkpoints"][str(step)]["trainer_state_inventory_sha256"]
    ):
        raise SSMaxPerceptionEvidenceError(
            f"{arm} step{step} trainer-state bytes differ from the checkpoint identity"
        )
    event_chain = ledgers[0]["event_chain_sha256"]
    if any(ledger["event_chain_sha256"] != event_chain for ledger in ledgers):
        raise SSMaxPerceptionEvidenceError(
            f"{arm} step{step} health event chains differ across ranks"
        )
    sources = _exact_fields(receipt["sources"], frozenset(SOURCES), name="health sources")
    total_loss = 0.0
    total_active = 0.0
    for source in SOURCES:
        values = _exact_fields(sources[source], _SOURCE_HEALTH_FIELDS, name=f"{source} health")
        for field in ("examples", "tokens", "positive_tokens"):
            _positive_int(values[field], name=f"{source} {field}", minimum=0)
        for field in ("loss_weight", "active_loss_weight"):
            number = _finite(values[field], name=f"{source} {field}")
            if number < 0:
                raise SSMaxPerceptionEvidenceError(f"{source} {field} must be non-negative")
        if values["target_loss_mass"] != manifest["loss_mass_targets"][source]:
            raise SSMaxPerceptionEvidenceError(f"{source} loss-mass target differs")
        total_loss += float(values["loss_weight"])
        total_active += float(values["active_loss_weight"])
    counters = _exact_fields(
        receipt["run_counters"], _RUN_COUNTER_FIELDS, name="run health counters"
    )
    for field in _RUN_COUNTER_FIELDS:
        _positive_int(counters[field], name=field, minimum=0)
    expected_counters = {
        "data_errors": sum(int(ledger["data_errors"]) for ledger in ledgers),
        "optimizer_guard_skips": int(ledgers[0]["optimizer_guard_skips"]),
        "nonfinite_losses": int(ledgers[0]["nonfinite_losses"]),
        "nonfinite_gradients": int(ledgers[0]["nonfinite_gradients"]),
    }
    if dict(counters) != expected_counters:
        raise SSMaxPerceptionEvidenceError(
            f"{arm} step{step} counters differ from checkpoint-native health ledgers"
        )
    evidence = _exact_fields(
        receipt["evidence"],
        frozenset({"recipe", "producer"}),
        name="health evidence",
    )
    recipe = _exact_fields(evidence["recipe"], _ARTIFACT_REF_FIELDS, name="health recipe reference")
    producer = _exact_fields(
        evidence["producer"],
        _PRODUCER_SOURCE_REF_FIELDS,
        name="health producer source",
    )
    producers = _validate_producer_source_references(
        manifest["producers"], git=_git_identity(manifest["git"])
    )
    if dict(recipe) != dict(manifest["recipe"]):
        raise SSMaxPerceptionEvidenceError("Health recipe identity differs from the manifest")
    if dict(producer) != dict(producers[HEALTH_PRODUCER]):
        raise SSMaxPerceptionEvidenceError("Health producer source identity differs")
    return {
        "rank_states": list(rank_states),
        "total_loss_weight": total_loss,
        "total_active_loss_weight": total_active,
        "sources": sources,
        "run_counters": counters,
    }


def _source_balanced_interval(
    values: Mapping[str, np.ndarray], *, seed: int, samples: int
) -> dict[str, Any]:
    """Compute a deterministic paired, source-balanced percentile bootstrap interval."""

    if set(values) != set(SOURCES):
        raise SSMaxPerceptionEvidenceError("Bootstrap source set differs")
    arrays = {source: np.asarray(values[source], dtype=np.float64) for source in SOURCES}
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise SSMaxPerceptionEvidenceError("Bootstrap arrays must be non-empty and equally sized")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise SSMaxPerceptionEvidenceError("Bootstrap arrays contain non-finite values")
    _positive_int(samples, name="bootstrap samples")
    population = next(iter(lengths))
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(samples, dtype=np.float64)
    chunk_size = min(samples, 256)
    for start in range(0, samples, chunk_size):
        width = min(chunk_size, samples - start)
        macro = np.zeros(width, dtype=np.float64)
        for source in SOURCES:
            indices = rng.integers(0, population, size=(width, population))
            macro += arrays[source][indices].mean(axis=1)
        bootstrapped[start : start + width] = macro / len(SOURCES)
    per_source = {source: float(arrays[source].mean()) for source in SOURCES}
    return {
        "mean": float(np.mean(list(per_source.values()))),
        "ci": {
            "low": float(np.quantile(bootstrapped, 0.025)),
            "high": float(np.quantile(bootstrapped, 0.975)),
            "confidence": 0.95,
            "method": "paired-source-balanced-percentile-bootstrap-v1",
            "seed": seed,
            "samples": samples,
        },
        "per_source_mean": per_source,
    }


def _row_identity(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["pairing_position"]),
        int(row["recipient_index"]),
        int(row["donor_index"]),
        int(row["response_tokens"]),
    )


def _metric_arrays(
    evaluations: Mapping[str, Mapping[int, Mapping[str, list[Mapping[str, Any]]]]],
    *,
    source: str,
    window: str,
) -> dict[str, np.ndarray]:
    rows = {
        f"{arm}:{step}": evaluations[arm][step][source] for arm in ARMS for step in REQUIRED_STEPS
    }
    canonical = rows[f"{CONTROL_ARM}:0"]
    identities = [_row_identity(row) for row in canonical]
    if any([_row_identity(row) for row in candidate] != identities for candidate in rows.values()):
        raise SSMaxPerceptionEvidenceError(
            f"{source} rows are not exactly paired across arms and steps"
        )

    def values(arm: str, step: int, field: str) -> np.ndarray:
        return np.asarray(
            [float(row[field][window]) for row in rows[f"{arm}:{step}"]], dtype=np.float64
        )

    return {
        "control_step0_gap": values(CONTROL_ARM, 0, "ce_gap_wrong_minus_correct"),
        "treatment_step0_gap": values(TREATMENT_ARM, 0, "ce_gap_wrong_minus_correct"),
        "control_step4000_gap": values(CONTROL_ARM, 4000, "ce_gap_wrong_minus_correct"),
        "treatment_step3000_gap": values(TREATMENT_ARM, 3000, "ce_gap_wrong_minus_correct"),
        "treatment_step4000_gap": values(TREATMENT_ARM, 4000, "ce_gap_wrong_minus_correct"),
        "control_step4000_correct_ce": values(CONTROL_ARM, 4000, "correct_ce"),
        "treatment_step4000_correct_ce": values(TREATMENT_ARM, 4000, "correct_ce"),
    }


def _receipt_map(value: Any, *, name: str) -> Mapping[str, Mapping[int, Mapping[str, str]]]:
    mapping = _exact_fields(value, frozenset(ARMS), name=name)
    normalized: dict[str, Mapping[int, Mapping[str, str]]] = {}
    for arm in ARMS:
        arm_values = _mapping(mapping[arm], name=f"{name} {arm}")
        converted: dict[int, Mapping[str, str]] = {}
        for raw_step, reference in arm_values.items():
            try:
                step = int(raw_step)
            except (TypeError, ValueError) as error:
                raise SSMaxPerceptionEvidenceError(
                    f"{name} step {raw_step!r} is invalid"
                ) from error
            if step in converted:
                raise SSMaxPerceptionEvidenceError(f"{name} repeats step{step}")
            converted[step] = _exact_fields(
                reference, _ARTIFACT_REF_FIELDS, name=f"{name} {arm} step{step}"
            )
        if set(converted) != set(REQUIRED_STEPS):
            raise SSMaxPerceptionEvidenceError(
                f"{name} {arm} must contain exactly steps {list(REQUIRED_STEPS)}"
            )
        normalized[arm] = converted
    return normalized


def build_promotion_report(
    *,
    manifest_path: Path,
    evaluation_receipts: Mapping[str, Mapping[int, Mapping[str, str]]],
    health_receipts: Mapping[str, Mapping[int, Mapping[str, str]]],
    created_at: str,
    verify_live_manifest: bool = True,
) -> dict[str, Any]:
    """Rebuild the SSMax perception promotion decision from immutable raw receipts."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_manifest)
    report_time = _timestamp(created_at, name="promotion created_at")
    evaluation_refs = _receipt_map(evaluation_receipts, name="evaluation receipts")
    health_refs = _receipt_map(health_receipts, name="health receipts")
    evaluations: dict[str, dict[int, dict[str, list[Mapping[str, Any]]]]] = {
        arm: {} for arm in ARMS
    }
    evaluation_payloads: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in ARMS}
    health_payloads: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in ARMS}
    health_summaries: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    deviations: list[dict[str, Any]] = []
    receipt_output: dict[str, Any] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for step in REQUIRED_STEPS:
            _, evaluation = _load_receipt_reference(
                evaluation_refs[arm][step],
                manifest=manifest,
                manifest_path=manifest_path,
                arm=arm,
                step=step,
                expected_format=EVALUATION_RECEIPT_FORMAT,
            )
            _, health = _load_receipt_reference(
                health_refs[arm][step],
                manifest=manifest,
                manifest_path=manifest_path,
                arm=arm,
                step=step,
                expected_format=HEALTH_RECEIPT_FORMAT,
            )
            if (
                _timestamp(evaluation["created_at"], name="evaluation created_at") > report_time
                or _timestamp(health["created_at"], name="health created_at") > report_time
            ):
                raise SSMaxPerceptionEvidenceError("Promotion report predates one of its receipts")
            evaluations[arm][step] = _validate_evaluation_receipt(
                evaluation, manifest=manifest, arm=arm, step=step
            )
            health_summaries[arm][step] = _validate_health_receipt(
                health, manifest=manifest, arm=arm, step=step
            )
            evaluation_payloads[arm][step] = evaluation
            health_payloads[arm][step] = health
            receipt_output[arm][str(step)] = {
                "evaluation": dict(evaluation_refs[arm][step]),
                "health": dict(health_refs[arm][step]),
            }
            if evaluation["status"] != "passed":
                deviations.append({"kind": "evaluation_receipt_status", "arm": arm, "step": step})
            if health["status"] != "passed":
                deviations.append({"kind": "health_receipt_status", "arm": arm, "step": step})

    control_initial = evaluation_payloads[CONTROL_ARM][0]["state"]["full_model"]
    treatment_initial = evaluation_payloads[TREATMENT_ARM][0]["state"]["full_model"]
    if control_initial != treatment_initial:
        deviations.append({"kind": "step0_model_state_inequality"})
    canonical_text = evaluation_payloads[CONTROL_ARM][0]["text_sentinel"]
    for arm in ARMS:
        for step in REQUIRED_STEPS:
            state = evaluation_payloads[arm][step]["state"]
            for surface in ("frozen_lm", "non_image_embedding_rows"):
                if (
                    state[surface]["mismatch_count"] != 0
                    or state[surface]["reference_inventory_sha256"]
                    != state[surface]["candidate_inventory_sha256"]
                ):
                    deviations.append(
                        {
                            "kind": "frozen_state_regression",
                            "arm": arm,
                            "step": step,
                            "surface": surface,
                        }
                    )
            if arm == CONTROL_ARM and (
                state["vision"]["mismatch_count"] != 0
                or state["vision"]["reference_inventory_sha256"]
                != state["vision"]["candidate_inventory_sha256"]
            ):
                deviations.append({"kind": "control_vision_changed", "arm": arm, "step": step})
            if evaluation_payloads[arm][step]["text_sentinel"] != canonical_text:
                deviations.append(
                    {"kind": "native_text_sentinel_changed", "arm": arm, "step": step}
                )

    policy = manifest["policy"]
    tolerance = float(policy["loss_mass_share_tolerance"])
    for step in REQUIRED_STEPS:
        control_ranks = health_summaries[CONTROL_ARM][step]["rank_states"]
        treatment_ranks = health_summaries[TREATMENT_ARM][step]["rank_states"]
        for left, right in zip(control_ranks, treatment_ranks, strict=True):
            if left["data_loader_state_sha256"] != right["data_loader_state_sha256"]:
                deviations.append(
                    {"kind": "arm_cursor_mismatch", "step": step, "rank": left["rank"]}
                )
        for arm in ARMS:
            health = health_summaries[arm][step]
            counters = health["run_counters"]
            for counter, maximum in (
                ("data_errors", int(policy["maximum_data_errors"])),
                ("optimizer_guard_skips", int(policy["maximum_optimizer_guard_skips"])),
                ("nonfinite_losses", 0),
                ("nonfinite_gradients", 0),
            ):
                if counters[counter] > maximum:
                    deviations.append(
                        {
                            "kind": "run_health_counter",
                            "arm": arm,
                            "step": step,
                            "counter": counter,
                            "observed": counters[counter],
                            "maximum": maximum,
                        }
                    )
            if step > 0:
                for field, total_field, mass_kind in (
                    ("loss_weight", "total_loss_weight", "raw"),
                    ("active_loss_weight", "total_active_loss_weight", "active"),
                ):
                    total = float(health[total_field])
                    if total <= 0:
                        deviations.append(
                            {
                                "kind": "empty_loss_mass",
                                "arm": arm,
                                "step": step,
                                "mass_kind": mass_kind,
                            }
                        )
                        continue
                    for source in SOURCES:
                        share = float(health["sources"][source][field]) / total
                        target = float(manifest["loss_mass_targets"][source])
                        if abs(share - target) > tolerance:
                            deviations.append(
                                {
                                    "kind": "loss_mass",
                                    "arm": arm,
                                    "step": step,
                                    "mass_kind": mass_kind,
                                    "source": source,
                                    "observed": share,
                                    "target": target,
                                }
                            )

    attention_trajectory: dict[str, Any] = {}
    for arm in ARMS:
        baseline_attention = evaluation_payloads[arm][0]["attention_diagnostics"]
        arm_trajectory: dict[str, Any] = {
            "0": {
                "report_sha256": baseline_attention["report_sha256"],
                "comparison_from_step0": None,
            }
        }
        for step in REQUIRED_STEPS[1:]:
            candidate_attention = evaluation_payloads[arm][step]["attention_diagnostics"]
            try:
                comparison = compare_ssmax_attention_reports(
                    baseline_attention, candidate_attention
                )
            except ValueError as error:
                raise SSMaxPerceptionEvidenceError(
                    f"Could not compare {arm} step{step} attention diagnostics: {error}"
                ) from error
            arm_trajectory[str(step)] = {
                "report_sha256": candidate_attention["report_sha256"],
                "comparison_from_step0": comparison,
            }
        attention_trajectory[arm] = arm_trajectory
    summary: dict[str, Any] = {
        "windows": {},
        "attention_trajectory": attention_trajectory,
    }
    samples = int(manifest["evaluation"]["bootstrap_samples"])
    base_seed = int(manifest["evaluation"]["bootstrap_seed"])
    for window_index, window in enumerate(WINDOWS):
        arrays = {
            source: _metric_arrays(evaluations, source=source, window=window) for source in SOURCES
        }
        did_values = {
            source: (
                values["treatment_step4000_gap"]
                - values["treatment_step0_gap"]
                - values["control_step4000_gap"]
                + values["control_step0_gap"]
            )
            for source, values in arrays.items()
        }
        treatment_values = {
            source: values["treatment_step4000_gap"] for source, values in arrays.items()
        }
        did = _source_balanced_interval(
            did_values, seed=base_seed + window_index * 10_000, samples=samples
        )
        treatment_gap = _source_balanced_interval(
            treatment_values,
            seed=base_seed + 100_000 + window_index * 10_000,
            samples=samples,
        )
        window_sources: dict[str, Any] = {}
        for source_index, (source, values) in enumerate(arrays.items()):
            control_ce = float(values["control_step4000_correct_ce"].mean())
            treatment_ce = float(values["treatment_step4000_correct_ce"].mean())
            durability_gap = float(values["treatment_step3000_gap"].mean())
            candidate_gap = float(values["treatment_step4000_gap"].mean())
            source_did = bridge.summarize_paired_values(
                did_values[source],
                seed=base_seed + 200_000 + window_index * 10_000 + source_index,
                samples=samples,
            )
            source_treatment_gap = bridge.summarize_paired_values(
                treatment_values[source],
                seed=base_seed + 300_000 + window_index * 10_000 + source_index,
                samples=samples,
            )
            window_sources[source] = {
                "did": source_did,
                "treatment_gap": source_treatment_gap,
                "control_correct_ce": control_ce,
                "treatment_correct_ce": treatment_ce,
                "step3000_treatment_gap": durability_gap,
                "step4000_treatment_gap": candidate_gap,
            }
            if source_did["mean_bootstrap_ci"]["low"] <= float(policy["did_lower_ci_minimum"]):
                deviations.append(
                    {"kind": "source_nonpositive_did", "source": source, "window": window}
                )
            if source_treatment_gap["mean_bootstrap_ci"]["low"] <= float(
                policy["treatment_gap_lower_ci_minimum"]
            ):
                deviations.append(
                    {
                        "kind": "source_nonpositive_treatment_gap",
                        "source": source,
                        "window": window,
                    }
                )
            if treatment_ce > (1 + float(policy["correct_ce_max_relative_increase"])) * control_ce:
                deviations.append(
                    {"kind": "source_correct_ce_regression", "source": source, "window": window}
                )
            if candidate_gap < float(policy["minimum_gap_retention"]) * durability_gap:
                deviations.append(
                    {"kind": "source_gap_retention", "source": source, "window": window}
                )
        macro_control_ce = float(
            np.mean([values["control_step4000_correct_ce"].mean() for values in arrays.values()])
        )
        macro_treatment_ce = float(
            np.mean([values["treatment_step4000_correct_ce"].mean() for values in arrays.values()])
        )
        macro_durability_gap = float(
            np.mean([values["treatment_step3000_gap"].mean() for values in arrays.values()])
        )
        macro_candidate_gap = float(
            np.mean([values["treatment_step4000_gap"].mean() for values in arrays.values()])
        )
        if did["ci"]["low"] <= float(policy["did_lower_ci_minimum"]):
            deviations.append({"kind": "macro_did_lower_ci", "window": window})
        if treatment_gap["ci"]["low"] <= float(policy["treatment_gap_lower_ci_minimum"]):
            deviations.append({"kind": "macro_treatment_gap_lower_ci", "window": window})
        if (
            macro_treatment_ce
            > (1 + float(policy["correct_ce_max_relative_increase"])) * macro_control_ce
        ):
            deviations.append({"kind": "macro_correct_ce_regression", "window": window})
        if macro_candidate_gap < float(policy["minimum_gap_retention"]) * macro_durability_gap:
            deviations.append({"kind": "macro_gap_retention", "window": window})
        summary["windows"][window] = {
            "did": did,
            "treatment_absolute_gap": treatment_gap,
            "macro_control_correct_ce": macro_control_ce,
            "macro_treatment_correct_ce": macro_treatment_ce,
            "macro_step3000_treatment_gap": macro_durability_gap,
            "macro_step4000_treatment_gap": macro_candidate_gap,
            "sources": window_sources,
        }

    report: dict[str, Any] = {
        "format": PROMOTION_REPORT_FORMAT,
        "version": SCHEMA_VERSION,
        "status": "passed" if not deviations else "rejected",
        "created_at": created_at,
        "manifest": manifest_reference(manifest_path, manifest),
        "pair_id": manifest["pair_id"],
        "model_variant": manifest["model_variant"],
        "receipts": receipt_output,
        "summary": summary,
        "deviations": deviations,
    }
    report["content_sha256"] = canonical_sha256(report)
    return report


def _rebuilt_promotion_for_model_comparison(
    reference: Mapping[str, Any], *, verify_live_checkpoint: bool
) -> Mapping[str, Any]:
    """Rebuild one passed v5 evidence report without accepting caller-supplied expectations."""

    report_path = validate_artifact_reference(reference, name="model-comparison promotion report")
    report = _exact_fields(
        load_json(report_path), _PROMOTION_REPORT_FIELDS, name="model-comparison promotion report"
    )
    manifest_ref = _exact_fields(
        report["manifest"], _MANIFEST_REF_FIELDS, name="model-comparison manifest reference"
    )
    manifest_path = Path(str(manifest_ref["path"])).expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    treatment = manifest["arms"][TREATMENT_ARM]
    candidate = treatment["checkpoints"]["4000"]
    return validate_promotion_report_reference(
        reference,
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(treatment["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(treatment["trainable_contract_sha256"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )


def _comparison_direction(summary: Mapping[str, Any], *, positive: str, negative: str) -> str:
    interval = _mapping(summary.get("mean_bootstrap_ci"), name="paired comparison interval")
    low = _finite(interval.get("low"), name="paired comparison lower bound")
    high = _finite(interval.get("high"), name="paired comparison upper bound")
    return positive if low > 0 else negative if high < 0 else "inconclusive"


def _summarize_model_difference(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
    positive: str,
    negative: str,
) -> dict[str, Any]:
    try:
        summary = bridge.summarize_paired_values(values, seed=seed, samples=samples)
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error
    summary["direction"] = _comparison_direction(summary, positive=positive, negative=negative)
    return summary


def _model_comparison_evaluations(
    rebuilt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, dict[int, Mapping[str, Any]]], dict[str, Any]]:
    report = _mapping(rebuilt["report"], name="rebuilt promotion report")
    manifest = _mapping(rebuilt["manifest"], name="rebuilt perception manifest")
    manifest_ref = _mapping(report["manifest"], name="promotion manifest reference")
    manifest_path = Path(str(manifest_ref["path"])).expanduser().resolve()
    receipt_refs = _exact_fields(report["receipts"], frozenset(ARMS), name="report receipts")
    payloads: dict[str, dict[int, Mapping[str, Any]]] = {arm: {} for arm in ARMS}
    rows: dict[str, dict[int, Mapping[str, list[Mapping[str, Any]]]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        arm_refs = _exact_fields(
            receipt_refs[arm],
            frozenset(str(step) for step in REQUIRED_STEPS),
            name=f"{arm} report receipts",
        )
        for step in REQUIRED_STEPS:
            step_refs = _exact_fields(
                arm_refs[str(step)],
                frozenset({"evaluation", "health"}),
                name=f"{arm} step{step} report receipts",
            )
            _, payload = _load_receipt_reference(
                step_refs["evaluation"],
                manifest=manifest,
                manifest_path=manifest_path,
                arm=arm,
                step=step,
                expected_format=EVALUATION_RECEIPT_FORMAT,
            )
            rows[arm][step] = _validate_evaluation_receipt(
                payload, manifest=manifest, arm=arm, step=step
            )
            payloads[arm][step] = payload
    return manifest, payloads, rows


def build_model_variant_comparison(
    *,
    left_promotion_report: Mapping[str, Any],
    right_promotion_report: Mapping[str, Any],
    created_at: str,
    verify_live_checkpoint: bool = True,
) -> dict[str, Any]:
    """Build a descriptive QK-vs-no-QK perception comparison from rebuilt v5 evidence.

    Absolute same-checkpoint differences and step-0-normalized adaptation differences are kept
    separate.  This function deliberately emits no winner and cannot authorize a checkpoint.
    """

    _timestamp(created_at, name="model comparison created_at")
    left_rebuilt = _rebuilt_promotion_for_model_comparison(
        left_promotion_report, verify_live_checkpoint=verify_live_checkpoint
    )
    right_rebuilt = _rebuilt_promotion_for_model_comparison(
        right_promotion_report, verify_live_checkpoint=verify_live_checkpoint
    )
    left_manifest, left_payloads, left_rows = _model_comparison_evaluations(left_rebuilt)
    right_manifest, right_payloads, right_rows = _model_comparison_evaluations(right_rebuilt)
    variants = {left_manifest["model_variant"], right_manifest["model_variant"]}
    if variants != set(MODEL_VARIANTS):
        raise SSMaxPerceptionEvidenceError(
            "Perception model comparison requires exactly QK-norm and no-QK-norm variants"
        )
    protocol_fields = (
        "git",
        "producers",
        "recipe",
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
    )
    differing_protocols = [
        field for field in protocol_fields if left_manifest[field] != right_manifest[field]
    ]
    if differing_protocols:
        raise SSMaxPerceptionEvidenceError(
            f"Perception model-comparison protocols differ in {differing_protocols}"
        )

    canonical_identities = {
        source: [_row_identity(row) for row in left_rows[CONTROL_ARM][0][source]]
        for source in SOURCES
    }
    for side_name, side_rows in (("left", left_rows), ("right", right_rows)):
        for arm in ARMS:
            for step in REQUIRED_STEPS:
                for source in SOURCES:
                    identities = [_row_identity(row) for row in side_rows[arm][step][source]]
                    if identities != canonical_identities[source]:
                        raise SSMaxPerceptionEvidenceError(
                            f"{side_name} {arm} step{step} {source} rows are not exactly paired"
                        )

    samples = int(left_manifest["evaluation"]["bootstrap_samples"])
    seed = int(left_manifest["evaluation"]["bootstrap_seed"])

    def metric(
        side_rows: Mapping[str, Mapping[int, Mapping[str, list[Mapping[str, Any]]]]],
        arm: str,
        step: int,
        source: str,
        window: str,
        field: str,
    ) -> np.ndarray:
        return np.asarray(
            [float(row[field][window]) for row in side_rows[arm][step][source]],
            dtype=np.float64,
        )

    trajectory: dict[str, Any] = {}
    causal: dict[str, Any] = {}
    attention: dict[str, Any] = {}
    stream = 0
    for arm in ARMS:
        arm_steps: dict[str, Any] = {}
        arm_attention: dict[str, Any] = {}
        for step in REQUIRED_STEPS:
            step_sources: dict[str, Any] = {}
            for source in SOURCES:
                source_windows: dict[str, Any] = {}
                for window in WINDOWS:
                    left_gap = metric(
                        left_rows, arm, step, source, window, "ce_gap_wrong_minus_correct"
                    )
                    right_gap = metric(
                        right_rows, arm, step, source, window, "ce_gap_wrong_minus_correct"
                    )
                    left_correct = metric(left_rows, arm, step, source, window, "correct_ce")
                    right_correct = metric(right_rows, arm, step, source, window, "correct_ce")
                    left_gap_zero = metric(
                        left_rows, arm, 0, source, window, "ce_gap_wrong_minus_correct"
                    )
                    right_gap_zero = metric(
                        right_rows, arm, 0, source, window, "ce_gap_wrong_minus_correct"
                    )
                    left_correct_zero = metric(left_rows, arm, 0, source, window, "correct_ce")
                    right_correct_zero = metric(right_rows, arm, 0, source, window, "correct_ce")
                    differences = {
                        "gap_absolute": left_gap - right_gap,
                        "correct_absolute": left_correct - right_correct,
                        "gap_adaptation": (left_gap - left_gap_zero) - (right_gap - right_gap_zero),
                        "correct_adaptation": (left_correct - left_correct_zero)
                        - (right_correct - right_correct_zero),
                    }
                    source_windows[window] = {
                        "absolute_checkpoint_advantage": {
                            "gap_left_minus_right": _summarize_model_difference(
                                differences["gap_absolute"],
                                seed=seed + stream,
                                samples=samples,
                                positive="left_larger_visual_gap",
                                negative="right_larger_visual_gap",
                            ),
                            "correct_ce_left_minus_right": _summarize_model_difference(
                                differences["correct_absolute"],
                                seed=seed + stream + 1,
                                samples=samples,
                                positive="left_higher_worse_correct_ce",
                                negative="right_higher_worse_correct_ce",
                            ),
                        },
                        "step0_normalized_adaptation_left_minus_right": {
                            "formula": "(left_step-left_step0)-(right_step-right_step0)",
                            "gap": _summarize_model_difference(
                                differences["gap_adaptation"],
                                seed=seed + stream + 2,
                                samples=samples,
                                positive="left_gained_more_visual_gap",
                                negative="right_gained_more_visual_gap",
                            ),
                            "correct_ce": _summarize_model_difference(
                                differences["correct_adaptation"],
                                seed=seed + stream + 3,
                                samples=samples,
                                positive="left_correct_ce_regressed_more",
                                negative="right_correct_ce_regressed_more",
                            ),
                        },
                    }
                    stream += 4
                step_sources[source] = source_windows
            arm_steps[str(step)] = {"sources": step_sources}
            left_attention = left_payloads[arm][step]["attention_diagnostics"]
            right_attention = right_payloads[arm][step]["attention_diagnostics"]
            try:
                attention_comparison = compare_ssmax_attention_reports(
                    left_attention, right_attention
                )
            except ValueError as error:
                raise SSMaxPerceptionEvidenceError(
                    f"Could not compare {arm} step{step} attention diagnostics: {error}"
                ) from error
            arm_attention[str(step)] = {
                "left_report_sha256": left_attention["report_sha256"],
                "right_report_sha256": right_attention["report_sha256"],
                "absolute_left_vs_right": attention_comparison,
                "left_comparison_from_step0": left_rebuilt["report"]["summary"][
                    "attention_trajectory"
                ][arm][str(step)]["comparison_from_step0"],
                "right_comparison_from_step0": right_rebuilt["report"]["summary"][
                    "attention_trajectory"
                ][arm][str(step)]["comparison_from_step0"],
            }
        trajectory[arm] = arm_steps
        attention[arm] = arm_attention

    for step in REQUIRED_STEPS[1:]:
        step_sources = {}
        for source in SOURCES:
            source_windows = {}
            for window in WINDOWS:
                model_values: dict[str, dict[str, np.ndarray]] = {}
                for side, side_rows in (("left", left_rows), ("right", right_rows)):
                    model_values[side] = {}
                    for field_name, field in (
                        ("gap", "ce_gap_wrong_minus_correct"),
                        ("correct_ce", "correct_ce"),
                    ):
                        treatment_change = metric(
                            side_rows, TREATMENT_ARM, step, source, window, field
                        ) - metric(side_rows, TREATMENT_ARM, 0, source, window, field)
                        control_change = metric(
                            side_rows, CONTROL_ARM, step, source, window, field
                        ) - metric(side_rows, CONTROL_ARM, 0, source, window, field)
                        model_values[side][field_name] = treatment_change - control_change
                gap = model_values["left"]["gap"] - model_values["right"]["gap"]
                correct = model_values["left"]["correct_ce"] - model_values["right"]["correct_ce"]
                source_windows[window] = {
                    "formula": (
                        "[(left_treatment_step-left_treatment_step0)-"
                        "(left_control_step-left_control_step0)]-"
                        "[(right_treatment_step-right_treatment_step0)-"
                        "(right_control_step-right_control_step0)]"
                    ),
                    "gap": _summarize_model_difference(
                        gap,
                        seed=seed + stream,
                        samples=samples,
                        positive="left_larger_treatment_over_control_adaptation",
                        negative="right_larger_treatment_over_control_adaptation",
                    ),
                    "correct_ce": _summarize_model_difference(
                        correct,
                        seed=seed + stream + 1,
                        samples=samples,
                        positive="left_larger_causal_correct_ce_regression",
                        negative="right_larger_causal_correct_ce_regression",
                    ),
                }
                stream += 2
            step_sources[source] = source_windows
        causal[str(step)] = {"sources": step_sources}

    result: dict[str, Any] = {
        "format": MODEL_COMPARISON_FORMAT,
        "version": SCHEMA_VERSION,
        "created_at": created_at,
        "decision_scope": "descriptive_non_promotion_molmofiability_signal",
        "winner": None,
        "left": {
            "model_variant": left_manifest["model_variant"],
            "pair_id": left_manifest["pair_id"],
            "promotion_report": {
                **dict(left_promotion_report),
                "content_sha256": left_rebuilt["report"]["content_sha256"],
            },
        },
        "right": {
            "model_variant": right_manifest["model_variant"],
            "pair_id": right_manifest["pair_id"],
            "promotion_report": {
                **dict(right_promotion_report),
                "content_sha256": right_rebuilt["report"]["content_sha256"],
            },
        },
        "protocol": {
            "required_steps": list(REQUIRED_STEPS),
            "arms": list(ARMS),
            "sources": list(SOURCES),
            "windows": list(WINDOWS),
            "paired_bootstrap_samples": samples,
            "absolute_and_adaptation_are_separate": True,
        },
        "absolute_and_adaptation_trajectories": trajectory,
        "causal_adaptation_contrast": causal,
        "attention_comparisons": attention,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def validate_model_variant_comparison(
    value: Any, *, verify_live_checkpoint: bool = True
) -> Mapping[str, Any]:
    """Rebuild the closed descriptive comparison from both pinned v5 reports."""

    comparison = _exact_fields(
        value, _MODEL_COMPARISON_FIELDS, name="SSMax perception model comparison"
    )
    if (
        comparison["format"] != MODEL_COMPARISON_FORMAT
        or comparison["version"] != SCHEMA_VERSION
        or comparison["decision_scope"] != "descriptive_non_promotion_molmofiability_signal"
        or comparison["winner"] is not None
    ):
        raise SSMaxPerceptionEvidenceError("SSMax perception model comparison identity differs")
    _timestamp(comparison["created_at"], name="model comparison created_at")
    _validate_content_sha(comparison, name="model comparison")
    references: dict[str, dict[str, str]] = {}
    for side in ("left", "right"):
        identity = _exact_fields(
            comparison[side],
            frozenset({"model_variant", "pair_id", "promotion_report"}),
            name=f"model comparison {side}",
        )
        if identity["model_variant"] not in MODEL_VARIANTS:
            raise SSMaxPerceptionEvidenceError(f"Model comparison {side} variant is unsupported")
        if not isinstance(identity["pair_id"], str) or not identity["pair_id"]:
            raise SSMaxPerceptionEvidenceError(f"Model comparison {side} pair_id is malformed")
        report = _exact_fields(
            identity["promotion_report"],
            _MANIFEST_REF_FIELDS,
            name=f"model comparison {side} promotion report",
        )
        if not isinstance(report["path"], str) or not report["path"]:
            raise SSMaxPerceptionEvidenceError(
                f"Model comparison {side} promotion report path is malformed"
            )
        references[side] = {
            "path": report["path"],
            "sha256": _sha256(
                report["sha256"], name=f"model comparison {side} promotion report raw SHA"
            ),
        }
        _sha256(
            report["content_sha256"],
            name=f"model comparison {side} promotion report semantic SHA",
        )
    rebuilt = build_model_variant_comparison(
        left_promotion_report=references["left"],
        right_promotion_report=references["right"],
        created_at=str(comparison["created_at"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if dict(comparison) != rebuilt:
        raise SSMaxPerceptionEvidenceError(
            "SSMax perception model comparison differs from its rebuilt raw evidence"
        )
    return comparison


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
    """Re-open a passed report and reproduce it exactly from every raw receipt.

    The returned candidate/manifest summary is the only input accepted by the v5 parent gate.
    """

    if expected_model_variant not in MODEL_VARIANTS:
        raise SSMaxPerceptionEvidenceError("Expected SSMax perception model variant is unsupported")
    report_path = validate_artifact_reference(value, name="SSMax perception promotion report")
    report = _exact_fields(
        load_json(report_path), _PROMOTION_REPORT_FIELDS, name="SSMax perception promotion report"
    )
    if (
        report["format"] != PROMOTION_REPORT_FORMAT
        or report["version"] != SCHEMA_VERSION
        or report["status"] != "passed"
        or report["model_variant"] != expected_model_variant
        or report["deviations"] != []
    ):
        raise SSMaxPerceptionEvidenceError("SSMax perception promotion report is not eligible")
    _timestamp(report["created_at"], name="promotion report created_at")
    _validate_content_sha(report, name="promotion report")
    manifest_ref = _exact_fields(
        report["manifest"], _MANIFEST_REF_FIELDS, name="promotion report manifest"
    )
    manifest_path = validate_artifact_reference(
        {"path": manifest_ref["path"], "sha256": manifest_ref["sha256"]},
        name="promotion report manifest",
    )
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    if (
        manifest["content_sha256"] != manifest_ref["content_sha256"]
        or manifest["pair_id"] != report["pair_id"]
        or manifest["model_variant"] != expected_model_variant
    ):
        raise SSMaxPerceptionEvidenceError("Promotion report names an incompatible manifest")
    candidate = _checkpoint_reference(
        manifest["arms"][TREATMENT_ARM]["checkpoints"]["4000"],
        step=4000,
        verify_live=verify_live_checkpoint,
        workers=8,
    )
    if (
        Path(str(candidate["path"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or candidate["config_sha256"] != expected_checkpoint_config_sha256
        or manifest["arms"][TREATMENT_ARM]["data_contract_sha256"] != expected_data_contract_sha256
        or manifest["arms"][TREATMENT_ARM]["trainable_contract_sha256"]
        != expected_trainable_contract_sha256
    ):
        raise SSMaxPerceptionEvidenceError("Promotion report names a different treatment candidate")
    receipt_refs = _exact_fields(report["receipts"], frozenset(ARMS), name="report receipts")
    evaluation_refs: dict[str, dict[int, Mapping[str, str]]] = {arm: {} for arm in ARMS}
    health_refs: dict[str, dict[int, Mapping[str, str]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        steps = _exact_fields(
            receipt_refs[arm],
            frozenset(str(step) for step in REQUIRED_STEPS),
            name=f"report {arm} receipts",
        )
        for step in REQUIRED_STEPS:
            refs = _exact_fields(
                steps[str(step)],
                frozenset({"evaluation", "health"}),
                name=f"report {arm} step{step} receipts",
            )
            evaluation_refs[arm][step] = _exact_fields(
                refs["evaluation"], _ARTIFACT_REF_FIELDS, name="evaluation receipt reference"
            )
            health_refs[arm][step] = _exact_fields(
                refs["health"], _ARTIFACT_REF_FIELDS, name="health receipt reference"
            )
    rebuilt = build_promotion_report(
        manifest_path=manifest_path,
        evaluation_receipts=evaluation_refs,
        health_receipts=health_refs,
        created_at=str(report["created_at"]),
        verify_live_manifest=verify_live_checkpoint,
    )
    if rebuilt != dict(report):
        raise SSMaxPerceptionEvidenceError(
            "Promotion report differs from the report rebuilt from raw bound receipts"
        )
    return {
        "report": report,
        "report_reference": dict(value),
        "manifest": manifest,
        "manifest_reference": dict(manifest_ref),
        "candidate": dict(candidate),
    }


def build_parent_gate(
    *,
    promotion_report_path: Path,
    expected_promotion_report_sha256: str,
    approved_by: str,
    approved_at: str,
    verify_live_checkpoint: bool = True,
) -> dict[str, Any]:
    """Build an explicit human approval gate from a fully rebuilt passed report.

    Calling this function is an approval action.  It requires the caller to supply a durable
    identity and timestamp; it never inserts defaults or accepts waiver records.
    """

    if (
        sha256_file(promotion_report_path.expanduser().resolve())
        != expected_promotion_report_sha256
    ):
        raise SSMaxPerceptionEvidenceError(
            "Promotion report differs from its explicit approval pin"
        )
    raw_report = _exact_fields(
        load_json(promotion_report_path),
        _PROMOTION_REPORT_FIELDS,
        name="promotion report",
    )
    manifest_ref = _exact_fields(
        raw_report["manifest"], _MANIFEST_REF_FIELDS, name="promotion report manifest"
    )
    manifest_path = Path(str(manifest_ref["path"])).expanduser().resolve()
    manifest = load_manifest(manifest_path, verify_live=verify_live_checkpoint)
    treatment = manifest["arms"][TREATMENT_ARM]
    candidate = treatment["checkpoints"]["4000"]
    summary = validate_promotion_report_reference(
        {"path": str(promotion_report_path.resolve()), "sha256": expected_promotion_report_sha256},
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(treatment["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(treatment["trainable_contract_sha256"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if (
        not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxPerceptionEvidenceError("approved_by is not a durable human identity")
    approval_time = _timestamp(approved_at, name="approval timestamp")
    report_time = _timestamp(summary["report"]["created_at"], name="promotion report timestamp")
    if approval_time < report_time:
        raise SSMaxPerceptionEvidenceError("Human approval predates the promotion report")
    checkpoint_config = _mapping(
        load_json(Path(str(candidate["path"])) / "config.json"), name="candidate config"
    )
    metadata = _mapping(
        checkpoint_config.get("vision_alignment"), name="candidate vision-alignment metadata"
    )
    recipe_version = _positive_int(metadata.get("recipe_version"), name="recipe version")
    formatter_version = metadata.get("formatter_version")
    if not isinstance(formatter_version, str) or not formatter_version:
        raise SSMaxPerceptionEvidenceError("Candidate formatter version is malformed")
    gate = {
        "format": "vision_alignment_parent_gate",
        "version": PARENT_GATE_VERSION,
        "status": "approved",
        "recipe_version": recipe_version,
        "formatter_version": formatter_version,
        "phase": "perception",
        "model_variant": manifest["model_variant"],
        "arm": TREATMENT_ARM,
        "checkpoint": candidate["path"],
        "checkpoint_config_sha256": candidate["config_sha256"],
        "checkpoint_identity_sha256": candidate["identity_sha256"],
        "data_contract_sha256": treatment["data_contract_sha256"],
        "trainable_contract_sha256": treatment["trainable_contract_sha256"],
        "global_step": 4000,
        "metrics_artifact_sha256": expected_promotion_report_sha256,
        "promotion_report_path": str(promotion_report_path.resolve()),
        "promotion_report_sha256": expected_promotion_report_sha256,
        "promotion_report_content_sha256": summary["report"]["content_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_ref["sha256"],
        "manifest_content_sha256": manifest["content_sha256"],
        "approved_by": approved_by,
        "approved_at": approved_at,
        "waivers": [],
    }
    validate_ssmax_perception_parent_gate(
        gate,
        expected_checkpoint=Path(str(candidate["path"])),
        expected_checkpoint_config_sha256=str(candidate["config_sha256"]),
        expected_model_variant=str(manifest["model_variant"]),
        expected_data_contract_sha256=str(treatment["data_contract_sha256"]),
        expected_trainable_contract_sha256=str(treatment["trainable_contract_sha256"]),
        verify_live_checkpoint=verify_live_checkpoint,
    )
    return gate


def validate_ssmax_perception_parent_gate(
    gate: Any,
    *,
    expected_checkpoint: Path,
    expected_checkpoint_config_sha256: str,
    expected_model_variant: str,
    expected_data_contract_sha256: str,
    expected_trainable_contract_sha256: str,
    verify_live_checkpoint: bool = True,
) -> Mapping[str, Any]:
    """Validate the deviation-free v5 gate used exclusively by an SSMax joint phase."""

    value = _exact_fields(gate, _PARENT_GATE_FIELDS, name="SSMax v5 perception parent gate")
    expected_pairs = (
        ("format", "vision_alignment_parent_gate"),
        ("version", PARENT_GATE_VERSION),
        ("status", "approved"),
        ("phase", "perception"),
        ("model_variant", expected_model_variant),
        ("arm", TREATMENT_ARM),
        ("global_step", 4000),
        ("checkpoint_config_sha256", expected_checkpoint_config_sha256),
        ("data_contract_sha256", expected_data_contract_sha256),
        ("trainable_contract_sha256", expected_trainable_contract_sha256),
    )
    for name, expected in expected_pairs:
        if value[name] != expected:
            raise SSMaxPerceptionEvidenceError(f"SSMax v5 parent gate {name} differs")
    if expected_model_variant not in MODEL_VARIANTS:
        raise SSMaxPerceptionEvidenceError("SSMax v5 parent gate model variant is unsupported")
    if (
        Path(str(value["checkpoint"])).expanduser().resolve()
        != expected_checkpoint.expanduser().resolve()
        or expected_checkpoint.name != "step4000"
    ):
        raise SSMaxPerceptionEvidenceError("SSMax v5 gate must name the treatment step4000")
    _positive_int(value["recipe_version"], name="SSMax v5 recipe version")
    if not isinstance(value["formatter_version"], str) or not value["formatter_version"]:
        raise SSMaxPerceptionEvidenceError("SSMax v5 formatter version is malformed")
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
    ):
        _sha256(value[name], name=f"SSMax v5 parent gate {name}")
    if value["waivers"] != []:
        raise SSMaxPerceptionEvidenceError("SSMax v5 parent gate does not permit waivers")
    if value["metrics_artifact_sha256"] != value["promotion_report_sha256"]:
        raise SSMaxPerceptionEvidenceError("SSMax v5 metrics artifact must be its promotion report")
    approved_by = value["approved_by"]
    if (
        not isinstance(approved_by, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:/+\-]{2,127}", approved_by) is None
    ):
        raise SSMaxPerceptionEvidenceError("SSMax v5 approved_by is not a durable identity")
    approval_time = _timestamp(value["approved_at"], name="SSMax v5 approved_at")
    report_reference = {
        "path": value["promotion_report_path"],
        "sha256": value["promotion_report_sha256"],
    }
    summary = validate_promotion_report_reference(
        report_reference,
        expected_checkpoint=expected_checkpoint,
        expected_checkpoint_config_sha256=expected_checkpoint_config_sha256,
        expected_model_variant=expected_model_variant,
        expected_data_contract_sha256=expected_data_contract_sha256,
        expected_trainable_contract_sha256=expected_trainable_contract_sha256,
        verify_live_checkpoint=verify_live_checkpoint,
    )
    if verify_live_checkpoint:
        checkpoint_config = _mapping(
            load_json(expected_checkpoint.expanduser().resolve() / "config.json"),
            name="SSMax v5 candidate config",
        )
        metadata = _mapping(
            checkpoint_config.get("vision_alignment"),
            name="SSMax v5 candidate vision-alignment metadata",
        )
        if value["recipe_version"] != metadata.get("recipe_version") or value[
            "formatter_version"
        ] != metadata.get("formatter_version"):
            raise SSMaxPerceptionEvidenceError(
                "SSMax v5 recipe/formatter identity differs from the live candidate"
            )
    if summary["candidate"]["identity_sha256"] != value["checkpoint_identity_sha256"]:
        raise SSMaxPerceptionEvidenceError("SSMax v5 checkpoint identity differs")
    if summary["report"]["content_sha256"] != value["promotion_report_content_sha256"]:
        raise SSMaxPerceptionEvidenceError("SSMax v5 promotion semantic SHA differs")
    manifest_ref = summary["manifest_reference"]
    if (
        Path(str(manifest_ref["path"])).resolve()
        != Path(str(value["manifest_path"])).expanduser().resolve()
        or manifest_ref["sha256"] != value["manifest_sha256"]
        or manifest_ref["content_sha256"] != value["manifest_content_sha256"]
    ):
        raise SSMaxPerceptionEvidenceError("SSMax v5 manifest reference differs")
    if approval_time < _timestamp(summary["report"]["created_at"], name="report created_at"):
        raise SSMaxPerceptionEvidenceError("SSMax v5 approval predates its promotion report")
    return summary


__all__ = [
    "ARMS",
    "CONTROL_ARM",
    "EVALUATION_PRODUCER",
    "EVALUATION_RECEIPT_FORMAT",
    "HEALTH_PRODUCER",
    "HEALTH_RECEIPT_FORMAT",
    "MANIFEST_FORMAT",
    "MANIFEST_SPEC_FORMAT",
    "MODEL_COMPARISON_FORMAT",
    "MODEL_VARIANTS",
    "PARENT_GATE_VERSION",
    "PRODUCER_RELATIVE_PATHS",
    "PROMOTION_REPORT_FORMAT",
    "REQUIRED_STEPS",
    "SCHEMA_VERSION",
    "SOURCES",
    "SSMaxPerceptionEvidenceError",
    "TREATMENT_ARM",
    "WINDOWS",
    "artifact_reference",
    "build_manifest",
    "build_model_variant_comparison",
    "build_parent_gate",
    "build_promotion_report",
    "canonical_sha256",
    "load_manifest",
    "load_manifest_spec",
    "validate_manifest",
    "validate_manifest_producer_source",
    "validate_model_variant_comparison",
    "validate_promotion_report_reference",
    "validate_saved_config_pair",
    "validate_ssmax_perception_parent_gate",
    "write_json_once",
]
