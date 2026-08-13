"""Build and validate the paired perception control-versus-treatment outcome receipt.

The comparator consumes four immutable schema-v4 evaluator receipts: both causal arms at matched
step3000 and step4000 endpoints. It joins exact recipient/donor rows, recomputes paired effects,
and emits deterministic source-balanced bootstrap evidence without loading model weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
)
from olmo_core.eval import (
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)
from olmo_core.eval import vision_alignment_promotion as promotion_bridge

FORMAT = "vision_alignment_perception_outcome_receipt"
VERSION = 1
PROTOCOL_NAME = "vision-alignment-perception-paired-counterfactual-outcome-v1"
ARMS = ("frozen_vision_control", "treatment")
STEPS = (3000, 4000)
STEP_KEYS = ("step3000", "step4000")
WINDOWS = ("all", "first_1", "first_8", "first_32")
DEFAULT_BOOTSTRAP_SEED = 7_208_2026
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_METHOD = "deterministic paired stratified example bootstrap percentile interval"
BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_PROFILE_PAIR_RECEIPT_SHA256 = (
    "5c7d9f3b2a882ed3147ca239eaaf00e9089d8e47c552a5cd19c351fdd806ea04"
)
EXPECTED_TRAINING_GIT_REF = "d8ec4f57cf026424ccd13f20452365b6b1df34e5"
EXPECTED_RUNTIME_CONFIG_SHA256 = {
    "frozen_vision_control": ("32d4baa68a9363b1ead672bbee737aa3f51fd712ec3ca28584cbde5416d2da5c"),
    "treatment": "6e6da90df7048d74fe611c45032b8c7b5c9846725a2029492b82353589ceca23",
}
EXPECTED_RUNTIME_ARM_CONFIG_SHA256 = {
    "frozen_vision_control": ("5f05b44dbd9e5a7f84ea6f3385be1f77ff24d775c70bbcab7d2a1cb32f1e0c79"),
    "treatment": "35d4245a62e0baa88dcb77bd0e6ffb079702315291723aba470a953223656a12",
}
EXPECTED_RUNTIME_SHARED_CONFIG_SHA256 = (
    "1cde5916a0c8f2aca32539e1aa4bb5c843c43f452384ee58d4a06d8122e6b7d4"
)
EXPECTED_IDENTITY_CONFIG_PATHS = (
    "/expected_launch_command",
    "/launch/cmd",
    "/launch/description",
    "/launch/name",
    "/required_run_name",
    "/reviewed_profile_path",
    "/reviewed_profile_sha256",
    "/trainer/callbacks/wandb/name",
    "/trainer/save_folder",
    "/vision_alignment/lineage_id",
)
EXPECTED_ARM_CONFIG_PATHS = (
    "/perception_trainability_arm",
    "/train_module/freeze_params",
    "/train_module/optim/group_overrides/<vision>/opts/lr",
    "/vision_alignment/trainable_contract_sha256",
)
TOP_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "inputs",
        "checkpoints",
        "protocol",
        "sources",
        "summary",
        "content_sha256",
    }
)


def _load_evaluator() -> ModuleType:
    path = Path(__file__).resolve().with_name("vision_alignment_perception_matched_wrong.py")
    spec = importlib.util.spec_from_file_location("_perception_matched_wrong_for_outcome", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load perception evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator()
bridge = evaluator.bridge


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        cli_arm = "control" if arm == "frozen_vision_control" else "treatment"
        for step in STEPS:
            parser.add_argument(f"--{cli_arm}-step{step}", required=True)
            parser.add_argument(f"--expected-{cli_arm}-step{step}-sha256", required=True)
    parser.add_argument("--profile-pair-receipt", required=True)
    parser.add_argument("--expected-profile-pair-receipt-sha256", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return bridge._sha256_file(path)


def _is_sha256(value: Any) -> bool:
    return evaluator._is_sha256(value)


def _load_json_bytes(
    path: Path, *, expected_sha256: str | None = None, name: str
) -> tuple[Any, str]:
    """Hash and strictly decode one immutable JSON byte snapshot."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"{name} SHA-256 differs: expected {expected_sha256}, got {digest}")
        payload = json.loads(
            raw,
            object_pairs_hook=promotion_bridge._strict_json_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {name} from {path}: {error}") from error
    return payload, digest


def _exact(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _artifact_ref(path_value: str | Path, expected_sha256: str, *, name: str) -> dict[str, str]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{name} expected identity must be a lowercase SHA-256")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} bytes differ from their exact SHA-256 pin")
    return {"path": str(path), "sha256": expected_sha256}


def _read_ref(value: Any, *, name: str) -> tuple[dict[str, str], Mapping[str, Any]]:
    reference = _exact(value, frozenset({"path", "sha256"}), name=f"{name} reference")
    expected_sha256 = reference["sha256"]
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{name} expected identity must be a lowercase SHA-256")
    path = Path(reference["path"]).expanduser().resolve()
    payload, _ = _load_json_bytes(path, expected_sha256=expected_sha256, name=name)
    normalized = {"path": str(path), "sha256": expected_sha256}
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return normalized, payload


def _implementation_reference(
    *,
    path: Any,
    sha256: Any,
    name: str,
    canonical_path: Path,
    verify_live: bool,
) -> dict[str, str]:
    """Validate a code pin while tolerating an expired Gantry checkout path."""
    if not isinstance(path, str) or not _is_sha256(sha256):
        raise ValueError(f"{name} implementation reference is malformed")
    try:
        promotion_bridge._validate_implementation_reference(
            {"path": path, "sha256": sha256},
            name=name,
            expected_basename=canonical_path.name,
            canonical_path=canonical_path,
            verify_live=verify_live,
        )
    except Exception as error:
        raise ValueError(str(error)) from error
    return {"basename": Path(path).name, "sha256": sha256}


def _checkpoint_identity(value: Any, *, arm: str, step: int) -> Mapping[str, Any]:
    identity = promotion_bridge._validate_checkpoint_identity(value, name=f"{arm} step{step}")
    root = Path(str(identity["root"])).expanduser().resolve()
    if root.name != f"step{step}":
        raise ValueError(f"{arm} receipt does not identify step{step}")
    return identity


def _validated_profile_pair_payload(
    payload: Mapping[str, Any], *, raw_sha256: str
) -> Mapping[str, Any]:
    if raw_sha256 != EXPECTED_PROFILE_PAIR_RECEIPT_SHA256:
        raise ValueError("Profile-pair receipt is not the exact reviewed causal-pair receipt")
    if (
        payload.get("format") != evaluator.PROFILE_PAIR_FORMAT
        or type(payload.get("version")) is not int
        or payload.get("version") != evaluator.PROFILE_PAIR_VERSION
        or payload.get("status") != "passed"
    ):
        raise ValueError("Profile-pair receipt identity or status is incompatible")
    comparison = payload.get("comparison")
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("allowed_identity_config_paths") != list(EXPECTED_IDENTITY_CONFIG_PATHS)
        or comparison.get("allowed_arm_config_paths") != list(EXPECTED_ARM_CONFIG_PATHS)
    ):
        raise ValueError("Profile-pair receipt causal-difference paths differ")
    git = payload.get("git")
    perception_contract = payload.get("perception_contract")
    evaluation = (
        perception_contract.get("evaluation") if isinstance(perception_contract, Mapping) else None
    )
    if (
        not isinstance(git, Mapping)
        or git.get("ref") != EXPECTED_TRAINING_GIT_REF
        or not isinstance(evaluation, Mapping)
        or _integer(evaluation.get("seed"), name="profile-pair evaluation seed") != 6198
    ):
        raise ValueError("Profile-pair training revision or evaluation seed differs")
    return payload


def _vision_group_index(config: Mapping[str, Any], *, arm: str) -> int:
    train_module = config.get("train_module")
    optim = train_module.get("optim") if isinstance(train_module, Mapping) else None
    groups = optim.get("group_overrides") if isinstance(optim, Mapping) else None
    if not isinstance(groups, list):
        raise TypeError(f"{arm} config optimizer groups are missing")
    matches = [
        index
        for index, group in enumerate(groups)
        if isinstance(group, Mapping) and group.get("params") == ["*vision.*"]
    ]
    if len(matches) != 1:
        raise ValueError(f"{arm} config must contain exactly one vision optimizer group")
    return matches[0]


def _normalize_config_identity(config: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    """Apply only the identity normalization attested by the pinned pair receipt."""
    normalized = copy.deepcopy(dict(config))
    try:
        normalized["expected_launch_command"] = [
            "<recipe>",
            "train",
            "<run>",
            "--profile=<profile>",
        ]
        normalized["required_run_name"] = "<run>"
        normalized["reviewed_profile_path"] = "<profile>"
        normalized["reviewed_profile_sha256"] = "<profile-sha256>"
        normalized["launch"]["cmd"] = [
            "<recipe>",
            "train",
            "<run>",
            "--profile=<profile>",
        ]
        normalized["launch"]["name"] = "<run>-<uuid>"
        normalized["launch"].pop("description", None)
        normalized["trainer"]["save_folder"] = "<save-folder>"
        normalized["trainer"]["callbacks"]["wandb"]["name"] = "<run>"
        normalized["vision_alignment"]["lineage_id"] = "<run>"
    except (KeyError, TypeError) as error:
        raise ValueError(f"{arm} config lacks a reviewed identity path") from error
    return normalized


def _normalize_config_arm(config: dict[str, Any], *, arm: str) -> None:
    """Apply only the causal-arm normalization attested by the pinned pair receipt."""
    vision_index = _vision_group_index(config, arm=arm)
    try:
        config["perception_trainability_arm"] = "<arm>"
        config["train_module"]["freeze_params"] = "<arm-freeze-params>"
        config["train_module"]["optim"]["group_overrides"][vision_index]["opts"][
            "lr"
        ] = "<arm-vision-lr>"
        config["vision_alignment"]["trainable_contract_sha256"] = "<arm-contract-sha256>"
    except (KeyError, TypeError) as error:
        raise ValueError(f"{arm} config lacks a reviewed causal-arm path") from error


def _causal_config_identity(
    config: Mapping[str, Any], *, arm: str, profile_pair: Mapping[str, Any]
) -> bytes:
    profiles = profile_pair.get("profiles")
    comparison = profile_pair.get("comparison")
    save_folders = profile_pair.get("save_folders")
    pair_data = profile_pair.get("data")
    pair_initialization = profile_pair.get("initialization")
    pair_launch = profile_pair.get("launch_contract")
    pair_contract = profile_pair.get("perception_contract")
    profile = profiles.get(arm) if isinstance(profiles, Mapping) else None
    trainable = (
        comparison.get("trainable_contract_sha256") if isinstance(comparison, Mapping) else None
    )
    if (
        not isinstance(profile, Mapping)
        or not isinstance(trainable, Mapping)
        or not isinstance(save_folders, Mapping)
        or not isinstance(pair_data, Mapping)
        or not isinstance(pair_initialization, Mapping)
        or not isinstance(pair_launch, Mapping)
        or not isinstance(pair_contract, Mapping)
    ):
        raise TypeError("Profile-pair arm identities are incomplete")
    expected_freeze = list(evaluator.EXPECTED_FREEZE)
    expected_lr = 3e-6
    if arm == "frozen_vision_control":
        expected_freeze = ["vision.*", *expected_freeze]
        expected_lr = 0.0
    vision_index = _vision_group_index(config, arm=arm)
    try:
        vision_group = config["train_module"]["optim"]["group_overrides"][vision_index]
        exact = (
            (config.get("phase"), "perception"),
            (config.get("perception_trainability_arm"), arm),
            (config.get("required_run_name"), profile.get("name")),
            (config.get("reviewed_profile_path"), profile.get("repository_path")),
            (config.get("reviewed_profile_sha256"), profile.get("sha256")),
            (config["trainer"]["save_folder"], save_folders.get(arm)),
            (config["train_module"]["freeze_params"], expected_freeze),
            (float(vision_group["opts"]["lr"]), expected_lr),
            (
                config["vision_alignment"]["trainable_contract_sha256"],
                trainable.get(arm),
            ),
            (
                config["vision_alignment"]["data_contract_sha256"],
                pair_data.get("data_contract_sha256"),
            ),
            (
                config["data"]["perception_provenance_sha256"],
                pair_data.get("perception_provenance_sha256"),
            ),
            (
                config["data"]["source_audit_fingerprint"],
                pair_data.get("source_audit_fingerprint"),
            ),
            (config["data"]["sequence_length"], pair_contract.get("data_sequence_length")),
            (
                config["initialization"]["checkpoint"],
                pair_initialization.get("checkpoint"),
            ),
            (
                config["initialization"]["parent_config_sha256"],
                pair_initialization.get("parent_config_sha256"),
            ),
            (
                config["initialization"]["parent_gate_sha256"],
                pair_initialization.get("parent_gate_sha256"),
            ),
            (config["launch"]["workspace"], pair_launch.get("workspace")),
            (config["launch"]["budget"], pair_launch.get("budget")),
            (config["launch"]["clusters"], [pair_launch.get("cluster")]),
            (config["launch"]["num_nodes"], pair_launch.get("num_nodes")),
            (config["launch"]["num_gpus"], pair_launch.get("num_gpus")),
            (config["launch"]["priority"], pair_launch.get("priority")),
            (config["launch"]["min_runtime"], pair_launch.get("min_runtime")),
            (
                config["evaluation"]["seed"],
                pair_contract.get("evaluation", {}).get("seed"),
            ),
            (
                config["evaluation"]["examples_per_source"],
                pair_contract.get("evaluation", {}).get("examples_per_source"),
            ),
            (
                config["evaluation"]["rank_batch_instances"],
                pair_contract.get("evaluation", {}).get("rank_batch_instances"),
            ),
            (config["launch"]["git"]["ref"], EXPECTED_TRAINING_GIT_REF),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{arm} config lacks an exact profile-pair field") from error
    if any(actual != expected for actual, expected in exact):
        raise ValueError(f"{arm} config differs from its exact profile-pair arm")
    normalized = _normalize_config_identity(config, arm=arm)
    if _canonical_sha256(normalized) != EXPECTED_RUNTIME_ARM_CONFIG_SHA256[arm]:
        raise ValueError(f"{arm} realized runtime config differs from its immutable checkpoint")
    _normalize_config_arm(normalized, arm=arm)
    normalized_bytes = _canonical_bytes(normalized)
    if hashlib.sha256(normalized_bytes).hexdigest() != EXPECTED_RUNTIME_SHARED_CONFIG_SHA256:
        raise ValueError("Realized shared runtime config differs from the immutable causal pair")
    return normalized_bytes


def _validate_per_example_rows(
    rows: Any, *, source: str, arm: str, step: int, examples: int
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != examples:
        raise ValueError(f"{arm} step{step} {source} has incomplete per-example rows")
    expected_positions = list(range(examples))
    positions: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "pairing_position",
            "recipient_index",
            "donor_index",
            "response_tokens",
            "correct_ce",
            "wrong_ce",
            "ce_gap_wrong_minus_correct",
        }:
            raise ValueError(f"{arm} step{step} {source} row {index} fields differ")
        position = row["pairing_position"]
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("Pairing position must be an integer")
        positions.append(position)
        for field in ("recipient_index", "donor_index", "response_tokens"):
            value = row[field]
            minimum = 1 if field == "response_tokens" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Per-example {field} is invalid")
        for field in ("correct_ce", "wrong_ce", "ce_gap_wrong_minus_correct"):
            values = row[field]
            if not isinstance(values, Mapping) or set(values) != set(WINDOWS):
                raise ValueError(f"Per-example {field} windows differ")
        for window in WINDOWS:
            correct = _finite(row["correct_ce"][window], name="correct CE")
            wrong = _finite(row["wrong_ce"][window], name="wrong CE")
            gap = _finite(row["ce_gap_wrong_minus_correct"][window], name="CE gap")
            if correct < 0 or wrong < 0:
                raise ValueError("Per-example cross entropy must be non-negative")
            if not math.isclose(gap, wrong - correct, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Per-example CE gap does not equal wrong minus correct")
    if positions != expected_positions:
        raise ValueError(f"{arm} step{step} {source} pairing positions are not complete")
    return rows


def _validate_rows_against_pairing(
    rows: Sequence[Mapping[str, Any]], pairing: Mapping[str, Any], *, source: str
) -> None:
    pairs = pairing.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(rows):
        raise ValueError(f"{source} pairing does not expose the exact recipient rows")
    for position, (row, pair) in enumerate(zip(rows, pairs, strict=True)):
        if (
            not isinstance(pair, Mapping)
            or set(pair) != {"recipient", "donor"}
            or row["pairing_position"] != position
            or row["recipient_index"] != pair["recipient"]
            or row["donor_index"] != pair["donor"]
        ):
            raise ValueError(f"{source} evaluator rows differ from the pinned pairing")


def _validate_evaluator_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    step: int,
    profile_pair_sha256: str,
    profile_pair_path: Path,
    profile_pair_payload: Mapping[str, Any],
    verify_live_inputs: bool,
) -> dict[str, Any]:
    _exact(
        receipt,
        frozenset(
            {
                "schema_version",
                "created_at",
                "checkpoint",
                "native_checkpoint_load",
                "config_path",
                "git",
                "evaluator",
                "profile_pair",
                "validation",
                "pairings",
                "artifact_policy",
                "protocol",
                "results",
                "config_and_protocol_sha256",
            }
        ),
        name=f"{arm} step{step} evaluator receipt",
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != evaluator.SCHEMA_VERSION
    ):
        raise ValueError(f"{arm} step{step} evaluator receipt is not schema v4")
    identity = _checkpoint_identity(receipt["checkpoint"], arm=arm, step=step)
    if identity["config_sha256"] != EXPECTED_RUNTIME_CONFIG_SHA256[arm]:
        raise ValueError(f"{arm} step{step} config is not the completed production config")
    checkpoint_root = Path(str(identity["root"])).expanduser().resolve()
    config_path = Path(str(receipt["config_path"])).expanduser().resolve()
    if config_path != checkpoint_root / "config.json":
        raise ValueError(f"{arm} step{step} config path is not rooted in its checkpoint")
    config, _ = _load_json_bytes(
        config_path,
        expected_sha256=identity["config_sha256"],
        name=f"{arm} step{step} checkpoint config",
    )
    if not isinstance(config, Mapping):
        raise TypeError(f"{arm} step{step} checkpoint config must be an object")
    normalized_config = _causal_config_identity(config, arm=arm, profile_pair=profile_pair_payload)
    if verify_live_inputs:
        try:
            evaluator._validate_live_checkpoint_identity_stable(identity, name=f"{arm} step{step}")
        except Exception as error:
            raise ValueError(str(error)) from error
    native = receipt["native_checkpoint_load"]
    native_fields = {
        "complete",
        "checkpoint_key_count",
        "model_parameter_count",
        "model_parameter_checkpoint_key_count",
        "model_parameter_checkpoint_keys_sha256",
        "model_parameter_assignments_sha256",
        "eval_state_key_count",
        "frozen_state_key_count",
        "persistent_buffer_count",
        "persistent_buffer_keys_sha256",
        "shadowed_frozen_key_count",
        "shadowed_frozen_keys_sha256",
        "unused_model_bearing_key_count",
        "prepared_load_key_count",
        "sha256",
        "load_completed",
    }
    model_parameter_count = (
        native.get("model_parameter_count") if isinstance(native, Mapping) else None
    )
    if (
        not isinstance(native, Mapping)
        or set(native) != native_fields
        or native.get("complete") is not True
        or native.get("load_completed") is not True
        or native.get("model_parameter_count") != native.get("model_parameter_checkpoint_key_count")
        or not isinstance(model_parameter_count, int)
        or isinstance(model_parameter_count, bool)
        or model_parameter_count <= 0
        or native.get("unused_model_bearing_key_count") != 0
    ):
        raise ValueError(f"{arm} step{step} native checkpoint load is incomplete")
    for field in native_fields - {
        "complete",
        "load_completed",
        "sha256",
        "model_parameter_checkpoint_keys_sha256",
        "model_parameter_assignments_sha256",
        "persistent_buffer_keys_sha256",
        "shadowed_frozen_keys_sha256",
    }:
        value = native[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{arm} step{step} native load {field} is invalid")
    for field in (
        "model_parameter_checkpoint_keys_sha256",
        "model_parameter_assignments_sha256",
        "persistent_buffer_keys_sha256",
        "shadowed_frozen_keys_sha256",
    ):
        if not _is_sha256(native[field]):
            raise ValueError(f"{arm} step{step} native load {field} is invalid")
    unsigned_native = {key: value for key, value in native.items() if key != "sha256"}
    if native["sha256"] != _canonical_sha256(unsigned_native):
        raise ValueError(f"{arm} step{step} native checkpoint-load SHA-256 differs")
    expected_frozen = 806 if arm == "frozen_vision_control" else 403
    if (
        native["model_parameter_count"] != 818
        or native["model_parameter_checkpoint_key_count"] != 818
        or native["eval_state_key_count"] != 818 - expected_frozen
        or native["frozen_state_key_count"] != expected_frozen
        or native["persistent_buffer_count"] != 0
        or native["prepared_load_key_count"] != 818
        or native["unused_model_bearing_key_count"] != 0
    ):
        raise ValueError(f"{arm} step{step} native checkpoint load surface differs")
    profile = receipt["profile_pair"]
    expected_profile_fields = {
        "path",
        "sha256",
        "format",
        "version",
        "arm",
        "step",
        "profile",
        "shared_config_sha256",
        "arm_config_sha256",
        "data_contract_sha256",
        "trainable_contract_sha256",
        "git_ref",
    }
    profiles = profile_pair_payload.get("profiles")
    comparison = profile_pair_payload.get("comparison")
    pair_data = profile_pair_payload.get("data")
    expected_profile = profiles.get(arm) if isinstance(profiles, Mapping) else None
    arm_configs = comparison.get("arm_config_sha256") if isinstance(comparison, Mapping) else None
    trainable = (
        comparison.get("trainable_contract_sha256") if isinstance(comparison, Mapping) else None
    )
    if (
        not isinstance(profiles, Mapping)
        or not isinstance(comparison, Mapping)
        or not isinstance(pair_data, Mapping)
        or not isinstance(expected_profile, Mapping)
        or not isinstance(arm_configs, Mapping)
        or not isinstance(trainable, Mapping)
    ):
        raise TypeError(f"{arm} step{step} profile-pair payload is incomplete")
    if (
        not isinstance(profile, Mapping)
        or set(profile) != expected_profile_fields
        or profile.get("sha256") != profile_pair_sha256
        or Path(str(profile.get("path"))).expanduser().resolve() != profile_pair_path
        or profile.get("format") != evaluator.PROFILE_PAIR_FORMAT
        or type(profile.get("version")) is not int
        or profile.get("version") != evaluator.PROFILE_PAIR_VERSION
        or profile.get("arm") != arm
        or _integer(profile.get("step"), name=f"{arm} profile step") != step
        or profile.get("profile")
        != {
            "name": expected_profile.get("name"),
            "repository_path": expected_profile.get("repository_path"),
            "sha256": expected_profile.get("sha256"),
        }
        or profile.get("shared_config_sha256") != comparison.get("shared_config_sha256")
        or profile.get("arm_config_sha256") != arm_configs.get(arm)
        or profile.get("data_contract_sha256") != pair_data.get("data_contract_sha256")
        or profile.get("trainable_contract_sha256") != trainable.get(arm)
        or profile.get("git_ref") != EXPECTED_TRAINING_GIT_REF
    ):
        raise ValueError(f"{arm} step{step} profile-pair binding differs")
    git = receipt["git"]
    if (
        not isinstance(git, Mapping)
        or set(git) != {"revision", "dirty", "status_sha256", "tracked_diff_sha256"}
        or not isinstance(git.get("revision"), str)
        or len(git["revision"]) != 40
        or any(character not in "0123456789abcdef" for character in git["revision"])
        or git.get("dirty") is not False
        or git.get("status_sha256") != hashlib.sha256(b"").hexdigest()
        or git.get("tracked_diff_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError(f"{arm} step{step} evaluator git state is not clean and pinned")
    if verify_live_inputs and git != evaluator.bridge._git_identity():
        raise ValueError(f"{arm} step{step} evaluator Git revision differs from this checkout")
    protocol = receipt["protocol"]
    protocol = _exact(
        protocol,
        frozenset(
            {
                "name",
                "sources",
                "dataset_split",
                "evaluation_population",
                "examples_per_source",
                "source_epoch",
                "pairing_seed",
                "pairing_sha256",
                "pairing_pin_policy",
                "pairing_rule",
                "recipient_replay",
                "response_logits",
                "per_example_ce",
                "gap_sign",
                "bootstrap",
                "windows",
                "message_format",
                "loss_token_weighting",
                "sequence_length",
                "rank_batch_instances",
                "global_batch_instances",
                "world_size",
                "ep_degree",
                "dp_process_group_size",
                "source_registry_sha256",
                "profile_pair_receipt_sha256",
                "perception_provenance_sha256",
                "source_audit_fingerprint",
                "tokenizer",
                "sha256",
            }
        ),
        name=f"{arm} step{step} evaluator protocol",
    )
    unsigned_protocol = dict(protocol)
    protocol_sha = unsigned_protocol.pop("sha256", None)
    expected_bootstrap = {
        "method": "deterministic iid example bootstrap percentile interval",
        "confidence": 0.95,
        "samples": 10_000,
        "seed": 1_006_201,
    }
    expected_token_ids = {
        "im_start_id": 100278,
        "im_end_id": 100279,
        "im_patch_id": 100280,
        "im_col_id": 100281,
        "low_res_im_start_id": 100282,
        "image_placeholder_id": 100283,
        "im_end_turn_id": 100265,
        "_CLASS_": "olmo_core.nn.vision.molmo2_tokens.Molmo2TokenIds",
    }
    expected_tokenizer = {
        "id": config["artifacts"]["tokenizer_id"],
        "revision": config["artifacts"]["tokenizer_revision"],
        "fingerprint": config["artifacts"]["tokenizer_fingerprint"],
        "token_ids": expected_token_ids,
    }
    bootstrap = _exact(
        protocol.get("bootstrap"),
        frozenset({"method", "confidence", "samples", "seed"}),
        name=f"{arm} step{step} bootstrap protocol",
    )
    if (
        bootstrap["method"] != expected_bootstrap["method"]
        or _finite(bootstrap["confidence"], name="bootstrap confidence") != 0.95
        or _integer(bootstrap["samples"], name="bootstrap samples", minimum=1) != 10_000
        or _integer(bootstrap["seed"], name="bootstrap seed") != 1_006_201
    ):
        raise ValueError(f"{arm} step{step} bootstrap protocol differs")
    for field, expected in (
        ("source_epoch", 0),
        ("pairing_seed", 6198),
        ("sequence_length", 2560),
        ("world_size", 8),
        ("ep_degree", 8),
        ("dp_process_group_size", 8),
        ("rank_batch_instances", 4),
        ("global_batch_instances", 32),
    ):
        if _integer(protocol.get(field), name=f"protocol {field}") != expected:
            raise ValueError(f"{arm} step{step} evaluator protocol {field} differs")
    if (
        protocol.get("name") != evaluator.PROTOCOL_NAME
        or protocol.get("sources") != list(PERCEPTION_SOURCE_NAMES)
        or protocol.get("dataset_split") != "validation"
        or protocol.get("evaluation_population") != "matched_eligible_validation_subset"
        or protocol.get("pairing_pin_policy")
        != "all eight existing pairing files require exact CLI pins"
        or protocol.get("pairing_rule")
        != (
            "distinct pinned content and materialized pixels; exact image tensor shape and "
            "byte-identical pooled_patches_idx; explicit unique donors"
        )
        or protocol.get("recipient_replay")
        != "correct and wrong forwards use exactly the same recipients"
        or protocol.get("response_logits") != "only positive-loss-mask positions are materialized"
        or protocol.get("per_example_ce")
        != "loss-mask-weighted mean over all or first min(K,N) supervised response tokens"
        or protocol.get("gap_sign") != "wrong_ce - correct_ce; positive is a correct-image win"
        or _canonical_bytes(protocol.get("windows"))
        != _canonical_bytes({name: limit for name, limit in evaluator.bridge.WINDOWS})
        or protocol.get("message_format") != config["data"]["message_format"]
        or protocol.get("loss_token_weighting") != config["data"]["loss_token_weighting"]
        or protocol.get("source_registry_sha256")
        != evaluator.vision_alignment_perception_source_registry_sha256()
        or protocol.get("profile_pair_receipt_sha256") != profile_pair_sha256
        or protocol.get("perception_provenance_sha256")
        != pair_data.get("perception_provenance_sha256")
        or protocol.get("source_audit_fingerprint") != pair_data.get("source_audit_fingerprint")
        or _canonical_bytes(protocol.get("tokenizer")) != _canonical_bytes(expected_tokenizer)
        or protocol_sha != _canonical_sha256(unsigned_protocol)
    ):
        raise ValueError(f"{arm} step{step} evaluator protocol differs")
    examples = protocol.get("examples_per_source")
    if (
        isinstance(examples, bool)
        or not isinstance(examples, int)
        or examples <= 0
        or examples % 32
    ):
        raise ValueError("Evaluator examples/source must be a positive multiple of 32")
    pairing_sha = protocol.get("pairing_sha256")
    policy = _exact(
        receipt["artifact_policy"],
        frozenset(
            {
                "all_pairings_require_sha256_pins",
                "expected_pairing_sha256",
                "output_overwrite_enabled",
            }
        ),
        name=f"{arm} step{step} evaluator artifact policy",
    )
    if (
        not isinstance(pairing_sha, Mapping)
        or set(pairing_sha) != set(PERCEPTION_SOURCE_NAMES)
        or not isinstance(policy, Mapping)
        or policy.get("all_pairings_require_sha256_pins") is not True
        or policy.get("expected_pairing_sha256") != pairing_sha
        or policy.get("output_overwrite_enabled") is not False
    ):
        raise ValueError("Evaluator receipt does not bind all eight immutable pairings")
    expected_config_and_protocol = _canonical_sha256(
        {
            "checkpoint_config_sha256": identity["config_sha256"],
            "protocol_sha256": protocol["sha256"],
            "pairing_sha256": pairing_sha,
        }
    )
    if receipt["config_and_protocol_sha256"] != expected_config_and_protocol:
        raise ValueError(f"{arm} step{step} config/protocol binding differs")
    validation = _exact(
        receipt["validation"],
        frozenset(
            {
                "mode",
                "manifest_path",
                "manifest_sha256",
                "content_sha256",
                "source_spec_sha256",
                "validation_union_disjoint_from_train",
                "validation_examples_per_source",
                "source_runtime_identities",
                "source_audit",
            }
        ),
        name=f"{arm} step{step} validation provenance",
    )
    source_audit = _exact(
        validation["source_audit"],
        frozenset({"path", "raw_sha256", "fingerprint", "source_registry_sha256"}),
        name=f"{arm} step{step} source-audit identity",
    )
    manifest_path = Path(str(validation["manifest_path"])).expanduser().resolve()
    source_audit_path = Path(str(source_audit["path"])).expanduser().resolve()
    if (
        validation.get("mode") != "perception_union_provenance_v2"
        or validation.get("validation_union_disjoint_from_train") is not True
        or _integer(
            validation.get("validation_examples_per_source"),
            name=f"{arm} validation examples/source",
        )
        != 512
        or validation.get("manifest_sha256") != pair_data.get("perception_provenance_sha256")
        or manifest_path != Path(str(config["data"]["perception_provenance_path"])).resolve()
        or source_audit.get("fingerprint") != pair_data.get("source_audit_fingerprint")
        or any(
            not _is_sha256(value)
            for value in (
                validation.get("content_sha256"),
                validation.get("source_spec_sha256"),
                source_audit.get("raw_sha256"),
            )
        )
        or source_audit.get("source_registry_sha256")
        != evaluator.vision_alignment_perception_source_registry_sha256()
        or source_audit_path != Path(str(config["data"]["source_audit_path"])).resolve()
    ):
        raise ValueError("Evaluator validation provenance differs")
    live_manifest = None
    if verify_live_inputs:
        try:
            live_manifest = evaluator.load_perception_provenance_manifest(
                manifest_path,
                expected_sha256=validation["manifest_sha256"],
                verify_finevision_materialization=False,
                load_image_path_signatures=False,
            )
            if (
                live_manifest.content_sha256 != validation["content_sha256"]
                or live_manifest.source_spec_sha256 != validation["source_spec_sha256"]
            ):
                raise ValueError("Live perception provenance semantic identity differs")
            if evaluator._source_audit_identity(config, live_manifest) != source_audit:
                raise ValueError("Live perception source-audit identity differs")
        except Exception as error:
            raise ValueError(
                f"Could not verify live perception validation evidence: {error}"
            ) from error
    evaluator_identity = receipt["evaluator"]
    expected_evaluator_fields = {
        "path",
        "sha256",
        "bridge_helper_path",
        "bridge_helper_sha256",
        "pairing_implementation_path",
        "pairing_implementation_sha256",
    }
    if (
        not isinstance(evaluator_identity, Mapping)
        or set(evaluator_identity) != expected_evaluator_fields
    ):
        raise ValueError("Evaluator implementation identity fields differ")
    pairing_source = inspect.getsourcefile(evaluator.build_matched_wrong_image_pairing)
    bridge_file = getattr(evaluator.bridge, "__file__", None)
    if pairing_source is None or not isinstance(bridge_file, str):
        raise RuntimeError("Canonical perception evaluator dependencies are unavailable")
    evaluator_file = getattr(evaluator, "__file__", None)
    if not isinstance(evaluator_file, str):
        raise TypeError("Canonical perception evaluator path is unavailable")
    evaluator_semantic = {
        "evaluator": _implementation_reference(
            path=evaluator_identity["path"],
            sha256=evaluator_identity["sha256"],
            name="perception evaluator",
            canonical_path=Path(evaluator_file).resolve(),
            verify_live=verify_live_inputs,
        ),
        "bridge_helper": _implementation_reference(
            path=evaluator_identity["bridge_helper_path"],
            sha256=evaluator_identity["bridge_helper_sha256"],
            name="bridge helper",
            canonical_path=Path(bridge_file).resolve(),
            verify_live=verify_live_inputs,
        ),
        "pairing_implementation": _implementation_reference(
            path=evaluator_identity["pairing_implementation_path"],
            sha256=evaluator_identity["pairing_implementation_sha256"],
            name="pairing implementation",
            canonical_path=Path(pairing_source).resolve(),
            verify_live=verify_live_inputs,
        ),
    }

    pairings = receipt["pairings"]
    results = receipt["results"]
    if (
        not isinstance(pairings, Mapping)
        or set(pairings) != set(PERCEPTION_SOURCE_NAMES)
        or not isinstance(results, Mapping)
        or set(results) != set(PERCEPTION_SOURCE_NAMES)
    ):
        raise ValueError("Evaluator pairings/results do not cover the exact eight sources")
    pairing_payloads: dict[str, Mapping[str, Any]] = {}
    rows: dict[str, list[Mapping[str, Any]]] = {}
    runtime_identities = validation.get("source_runtime_identities")
    if not isinstance(runtime_identities, Mapping) or set(runtime_identities) != set(
        PERCEPTION_SOURCE_NAMES
    ):
        raise ValueError("Validation runtime identities do not cover the exact eight sources")
    for source in PERCEPTION_SOURCE_NAMES:
        meta = _exact(
            pairings[source],
            frozenset(
                {
                    "path",
                    "sha256",
                    "expected_sha256",
                    "provenance",
                    "population",
                    "pairing_schema_version",
                    "coverage",
                    "recipient_indices_sha256",
                    "donor_indices_sha256",
                }
            ),
            name=f"{arm} step{step} {source} pairing metadata",
        )
        result = _exact(
            results[source],
            frozenset(
                {
                    "pairing_sha256",
                    "examples",
                    "elapsed_seconds",
                    "metrics",
                    "per_example",
                    "population",
                    "coverage",
                }
            ),
            name=f"{arm} step{step} {source} result",
        )
        runtime_identity = runtime_identities[source]
        if (
            not isinstance(runtime_identity, Mapping)
            or set(runtime_identity)
            != {
                "runtime_dataset_fingerprint",
                "selection_indices_sha256",
                "row_image_content_sha256",
                "live_image_validation_sha256",
            }
            or any(not _is_sha256(value) for value in runtime_identity.values())
        ):
            raise ValueError(f"{source} validation runtime identity is malformed")
        if live_manifest is not None:
            selection = live_manifest.selection(source, "validation")
            expected_runtime_identity = {
                "runtime_dataset_fingerprint": selection.runtime_dataset_fingerprint,
                "selection_indices_sha256": selection.selection_indices_sha256,
                "row_image_content_sha256": evaluator.bridge._content_ids_sha256(
                    selection.row_image_content_sha256
                ),
                "live_image_validation_sha256": _canonical_sha256(
                    [
                        {"index": index, "image_sha256": image_sha256}
                        for index, image_sha256 in enumerate(selection.row_image_content_sha256)
                    ]
                ),
            }
            if any(
                runtime_identity[field] != value
                for field, value in expected_runtime_identity.items()
            ):
                raise ValueError(f"{source} runtime identity differs from live provenance")
        if (
            not isinstance(meta, Mapping)
            or meta.get("sha256") != pairing_sha[source]
            or meta.get("expected_sha256") != pairing_sha[source]
            or meta.get("provenance") != "loaded"
            or meta.get("population") != "matched_eligible_validation_subset"
            or result.get("pairing_sha256") != pairing_sha[source]
            or _integer(result.get("examples"), name=f"{source} result examples") != examples
            or result.get("population") != "matched_eligible_validation_subset"
            or not isinstance(result.get("elapsed_seconds"), (int, float))
            or isinstance(result.get("elapsed_seconds"), bool)
            or not math.isfinite(float(result["elapsed_seconds"]))
            or float(result["elapsed_seconds"]) < 0
            or not isinstance(result.get("metrics"), Mapping)
        ):
            raise ValueError(f"{arm} step{step} {source} pairing/result binding differs")
        path = Path(str(meta.get("path"))).expanduser().resolve()
        pairing, _ = _load_json_bytes(
            path,
            expected_sha256=pairing_sha[source],
            name=f"live {source} pairing",
        )
        validate_matched_wrong_image_pairing(
            pairing,
            dataset_size=512,
            recipient_count=examples,
            seed=protocol.get("pairing_seed"),
            epoch=0,
            content_ids_sha256=runtime_identity["row_image_content_sha256"],
        )
        if matched_wrong_image_pairing_sha256(pairing) != pairing_sha[source]:
            raise ValueError(f"Canonical {source} pairing SHA-256 differs")
        if (
            _integer(meta.get("pairing_schema_version"), name=f"{source} pairing version")
            != _integer(pairing.get("version"), name=f"{source} pairing payload version")
            or _canonical_bytes(meta.get("coverage")) != _canonical_bytes(pairing.get("coverage"))
            or _canonical_bytes(result.get("coverage")) != _canonical_bytes(pairing.get("coverage"))
            or meta.get("recipient_indices_sha256")
            != _canonical_sha256([pair["recipient"] for pair in pairing["pairs"]])
            or meta.get("donor_indices_sha256")
            != _canonical_sha256([pair["donor"] for pair in pairing["pairs"]])
        ):
            raise ValueError(f"{arm} step{step} {source} pairing metadata differs from bytes")
        pairing_payloads[source] = pairing
        source_rows = _validate_per_example_rows(
            result.get("per_example"),
            source=source,
            arm=arm,
            step=step,
            examples=examples,
        )
        _validate_rows_against_pairing(source_rows, pairing, source=source)
        rows[source] = source_rows
    return {
        "checkpoint": identity,
        "native_checkpoint_load": dict(native),
        "config": config,
        "normalized_config": normalized_config,
        "profile_pair": profile,
        "protocol": protocol,
        "git": git,
        "validation": validation,
        "evaluator": evaluator_identity,
        "evaluator_semantic": evaluator_semantic,
        "pairings": pairings,
        "pairing_payloads": pairing_payloads,
        "rows": rows,
        "examples": examples,
    }


def _interval(values: np.ndarray, *, seed: int, samples: int) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a finite non-empty vector")
    rng = np.random.RandomState(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = min(samples, 2048)
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        indices = rng.randint(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"confidence": 0.95, "low": float(low), "high": float(high)}


def _stratified_interval(
    values: Mapping[str, np.ndarray], *, seed: int, samples: int
) -> dict[str, float]:
    """Bootstrap rows within each source, then average eight equally weighted source means."""
    if set(values) != set(PERCEPTION_SOURCE_NAMES):
        raise ValueError("Source-balanced bootstrap lacks the exact source set")
    rng = np.random.RandomState(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = min(samples, 512)
    ordered = [values[source] for source in PERCEPTION_SOURCE_NAMES]
    if any(array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all() for array in ordered):
        raise ValueError("Source-balanced bootstrap arrays are invalid")
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        combined = np.zeros(end - start, dtype=np.float64)
        for array in ordered:
            indices = rng.randint(0, len(array), size=(end - start, len(array)))
            combined += array[indices].mean(axis=1)
        means[start:end] = combined / len(ordered)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"confidence": 0.95, "low": float(low), "high": float(high)}


def _join_source_step(
    control_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    source_index: int,
    step_index: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    if len(control_rows) != len(treatment_rows):
        raise ValueError("Control and treatment row counts differ")
    joined: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, list[float]]] = {
        window: {
            "control_correct": [],
            "control_wrong": [],
            "control_gap": [],
            "treatment_correct": [],
            "treatment_wrong": [],
            "treatment_gap": [],
            "correct_reduction": [],
            "wrong_reduction": [],
            "did": [],
        }
        for window in WINDOWS
    }
    for position, (control, treatment) in enumerate(zip(control_rows, treatment_rows, strict=True)):
        identity_fields = (
            "pairing_position",
            "recipient_index",
            "donor_index",
            "response_tokens",
        )
        if any(control[field] != treatment[field] for field in identity_fields):
            raise ValueError("Control/treatment per-example pairing or response identity differs")
        if control["pairing_position"] != position:
            raise ValueError("Joined pairing positions are not canonical")
        control_block = {
            "correct_ce": dict(control["correct_ce"]),
            "wrong_ce": dict(control["wrong_ce"]),
            "gap_wrong_minus_correct": dict(control["ce_gap_wrong_minus_correct"]),
        }
        treatment_block = {
            "correct_ce": dict(treatment["correct_ce"]),
            "wrong_ce": dict(treatment["wrong_ce"]),
            "gap_wrong_minus_correct": dict(treatment["ce_gap_wrong_minus_correct"]),
        }
        effects: dict[str, dict[str, float]] = {
            "correct_ce_reduction": {},
            "wrong_ce_reduction": {},
            "gap_improvement_did": {},
        }
        for window in WINDOWS:
            cc = float(control["correct_ce"][window])
            cw = float(control["wrong_ce"][window])
            cg = float(control["ce_gap_wrong_minus_correct"][window])
            tc = float(treatment["correct_ce"][window])
            tw = float(treatment["wrong_ce"][window])
            tg = float(treatment["ce_gap_wrong_minus_correct"][window])
            correct_reduction = cc - tc
            wrong_reduction = cw - tw
            did = tg - cg
            effects["correct_ce_reduction"][window] = correct_reduction
            effects["wrong_ce_reduction"][window] = wrong_reduction
            effects["gap_improvement_did"][window] = did
            for name, value in (
                ("control_correct", cc),
                ("control_wrong", cw),
                ("control_gap", cg),
                ("treatment_correct", tc),
                ("treatment_wrong", tw),
                ("treatment_gap", tg),
                ("correct_reduction", correct_reduction),
                ("wrong_reduction", wrong_reduction),
                ("did", did),
            ):
                arrays[window][name].append(value)
        joined.append(
            {
                "pairing_position": position,
                "recipient_index": control["recipient_index"],
                "donor_index": control["donor_index"],
                "response_tokens": control["response_tokens"],
                "control": control_block,
                "treatment": treatment_block,
                "effects": effects,
            }
        )

    output_arrays = {
        window: {name: np.asarray(values, dtype=np.float64) for name, values in fields.items()}
        for window, fields in arrays.items()
    }
    window_metrics: dict[str, Any] = {}
    for window_index, window in enumerate(WINDOWS):
        values = output_arrays[window]
        seed = (
            bootstrap_seed + step_index * 10_000_000 + source_index * 100_000 + window_index * 100
        )
        window_metrics[window] = {
            "examples": len(joined),
            "control": {
                "correct_ce": float(values["control_correct"].mean()),
                "wrong_ce": float(values["control_wrong"].mean()),
                "gap": float(values["control_gap"].mean()),
            },
            "treatment": {
                "correct_ce": float(values["treatment_correct"].mean()),
                "wrong_ce": float(values["treatment_wrong"].mean()),
                "gap": {
                    "mean": float(values["treatment_gap"].mean()),
                    "ci": _interval(values["treatment_gap"], seed=seed, samples=bootstrap_samples),
                },
            },
            "effects": {
                "correct_ce_reduction": {
                    "mean": float(values["correct_reduction"].mean()),
                    "ci": _interval(
                        values["correct_reduction"], seed=seed + 1, samples=bootstrap_samples
                    ),
                },
                "wrong_ce_reduction": {
                    "mean": float(values["wrong_reduction"].mean()),
                    "ci": _interval(
                        values["wrong_reduction"], seed=seed + 2, samples=bootstrap_samples
                    ),
                },
                "gap_improvement_did": {
                    "mean": float(values["did"].mean()),
                    "ci": _interval(values["did"], seed=seed + 3, samples=bootstrap_samples),
                    "win_rate": float((values["did"] > 0).mean()),
                },
            },
        }
    return {"per_example": joined, "windows": window_metrics}, output_arrays


def _macro_step(
    arrays: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    step_index: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for window_index, window in enumerate(WINDOWS):
        seed = bootstrap_seed + 50_000_000 + step_index * 1_000_000 + window_index * 10_000

        def source_values(name: str, selected_window: str = window) -> dict[str, np.ndarray]:
            return {
                source: arrays[source][selected_window][name] for source in PERCEPTION_SOURCE_NAMES
            }

        def macro_mean(name: str, selected_window: str = window) -> float:
            return float(
                np.mean(
                    [
                        arrays[source][selected_window][name].mean()
                        for source in PERCEPTION_SOURCE_NAMES
                    ]
                )
            )

        did_wins = int(
            sum(arrays[source][window]["did"].mean() > 0 for source in PERCEPTION_SOURCE_NAMES)
        )
        correct_ce_wins = int(
            sum(
                arrays[source][window]["correct_reduction"].mean() > 0
                for source in PERCEPTION_SOURCE_NAMES
            )
        )
        if not (
            0 <= did_wins <= len(PERCEPTION_SOURCE_NAMES)
            and 0 <= correct_ce_wins <= len(PERCEPTION_SOURCE_NAMES)
        ):
            raise RuntimeError("Source-win counts exceed the exact eight-source population")
        output[window] = {
            "source_weighting": "equal_weight_per_source",
            "control": {
                "correct_ce": macro_mean("control_correct"),
                "wrong_ce": macro_mean("control_wrong"),
                "gap": macro_mean("control_gap"),
            },
            "treatment": {
                "correct_ce": macro_mean("treatment_correct"),
                "wrong_ce": macro_mean("treatment_wrong"),
                "gap": {
                    "mean": macro_mean("treatment_gap"),
                    "ci": _stratified_interval(
                        source_values("treatment_gap"), seed=seed, samples=bootstrap_samples
                    ),
                },
            },
            "did": {
                "mean": macro_mean("did"),
                "ci": _stratified_interval(
                    source_values("did"), seed=seed + 1, samples=bootstrap_samples
                ),
            },
            "correct_ce_reduction": {
                "mean": macro_mean("correct_reduction"),
                "ci": _stratified_interval(
                    source_values("correct_reduction"),
                    seed=seed + 2,
                    samples=bootstrap_samples,
                ),
            },
            "source_wins": {
                "did_positive": did_wins,
                "treatment_correct_ce_lower": correct_ce_wins,
                "source_count": len(PERCEPTION_SOURCE_NAMES),
            },
        }
    return output


def _durability(
    step_arrays: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    macro: dict[str, Any] = {}
    sources: dict[str, dict[str, Any]] = {source: {} for source in PERCEPTION_SOURCE_NAMES}
    for window_index, window in enumerate(WINDOWS):
        treatment_changes: dict[str, np.ndarray] = {}
        did_changes: dict[str, np.ndarray] = {}
        for source in PERCEPTION_SOURCE_NAMES:
            before = step_arrays[3000][source][window]
            after = step_arrays[4000][source][window]
            treatment_change = after["treatment_gap"] - before["treatment_gap"]
            did_change = after["did"] - before["did"]
            treatment_changes[source] = treatment_change
            did_changes[source] = did_change
            gap3000 = float(before["treatment_gap"].mean())
            gap4000 = float(after["treatment_gap"].mean())
            sources[source][window] = {
                "treatment_gap_step3000": gap3000,
                "treatment_gap_step4000": gap4000,
                "treatment_gap_retention_fraction": (gap4000 / gap3000 if gap3000 != 0 else None),
                "treatment_gap_change": float(treatment_change.mean()),
                "did_change": float(did_change.mean()),
            }
        gap3000 = float(
            np.mean(
                [
                    step_arrays[3000][source][window]["treatment_gap"].mean()
                    for source in PERCEPTION_SOURCE_NAMES
                ]
            )
        )
        gap4000 = float(
            np.mean(
                [
                    step_arrays[4000][source][window]["treatment_gap"].mean()
                    for source in PERCEPTION_SOURCE_NAMES
                ]
            )
        )
        seed = bootstrap_seed + 80_000_000 + window_index * 10_000
        macro[window] = {
            "treatment_gap_step3000": gap3000,
            "treatment_gap_step4000": gap4000,
            "treatment_gap_retention_fraction": gap4000 / gap3000 if gap3000 != 0 else None,
            "treatment_gap_change": {
                "mean": float(np.mean([value.mean() for value in treatment_changes.values()])),
                "ci": _stratified_interval(treatment_changes, seed=seed, samples=bootstrap_samples),
            },
            "did_change": {
                "mean": float(np.mean([value.mean() for value in did_changes.values()])),
                "ci": _stratified_interval(did_changes, seed=seed + 1, samples=bootstrap_samples),
            },
        }
    return macro, sources


def _build_components(
    evaluations: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    profile_pair_ref: Mapping[str, str],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    packets = [evaluations[arm][step] for arm in ARMS for step in STEPS]
    first = packets[0]
    examples = first["examples"]
    if first["protocol"].get("pairing_seed") != 6198:
        raise ValueError("Evaluator pairing seed differs from the reviewed profile pair")
    for packet in packets[1:]:
        if packet["examples"] != examples:
            raise ValueError("Four evaluator receipts use different examples/source")
        for field in ("protocol", "validation", "evaluator_semantic", "pairings", "git"):
            if _canonical_bytes(packet[field]) != _canonical_bytes(first[field]):
                raise ValueError(f"Four evaluator receipts use different {field} identities")
        for source in PERCEPTION_SOURCE_NAMES:
            if _canonical_bytes(packet["pairing_payloads"][source]) != _canonical_bytes(
                first["pairing_payloads"][source]
            ):
                raise ValueError("Four evaluator receipts do not replay identical pairings")
    for arm in ARMS:
        step3000_root = Path(str(evaluations[arm][3000]["checkpoint"]["root"])).resolve()
        step4000_root = Path(str(evaluations[arm][4000]["checkpoint"]["root"])).resolve()
        if step3000_root.parent != step4000_root.parent:
            raise ValueError(f"{arm} step3000/4000 checkpoints do not share one lineage root")
        if (
            evaluations[arm][3000]["checkpoint"]["config_sha256"]
            != evaluations[arm][4000]["checkpoint"]["config_sha256"]
        ):
            raise ValueError(f"{arm} step3000/4000 config identities differ")
        if evaluations[arm][3000]["config"] != evaluations[arm][4000]["config"]:
            raise ValueError(f"{arm} step3000/4000 saved configs differ")
        if (
            evaluations[arm][3000]["native_checkpoint_load"]
            != evaluations[arm][4000]["native_checkpoint_load"]
        ):
            raise ValueError(f"{arm} step3000/4000 native checkpoint load surfaces differ")
    if (
        evaluations["frozen_vision_control"][4000]["checkpoint"]["config_sha256"]
        == evaluations["treatment"][4000]["checkpoint"]["config_sha256"]
    ):
        raise ValueError("Control and treatment checkpoint configs must be distinct")
    if any(packet["normalized_config"] != first["normalized_config"] for packet in packets[1:]):
        raise ValueError(
            "Control/treatment configs differ outside reviewed identity and causal-arm fields"
        )

    identity_fields = (
        "pairing_position",
        "recipient_index",
        "donor_index",
        "response_tokens",
    )
    for source in PERCEPTION_SOURCE_NAMES:
        canonical_rows = first["rows"][source]
        for packet in packets[1:]:
            candidate_rows = packet["rows"][source]
            if len(candidate_rows) != len(canonical_rows):
                raise ValueError("Four evaluator receipts use different row populations")
            for canonical, candidate in zip(canonical_rows, candidate_rows, strict=True):
                if any(canonical[field] != candidate[field] for field in identity_fields):
                    raise ValueError("Step3000/step4000 pairing or response-token identity differs")

    checkpoints = {
        arm: {f"step{step}": dict(evaluations[arm][step]["checkpoint"]) for step in STEPS}
        for arm in ARMS
    }
    source_output: dict[str, Any] = {}
    step_arrays: dict[int, dict[str, Mapping[str, Mapping[str, np.ndarray]]]] = {
        step: {} for step in STEPS
    }
    for source_index, source in enumerate(PERCEPTION_SOURCE_NAMES):
        steps_output: dict[str, Any] = {}
        for step_index, step in enumerate(STEPS):
            joined, arrays = _join_source_step(
                evaluations["frozen_vision_control"][step]["rows"][source],
                evaluations["treatment"][step]["rows"][source],
                source_index=source_index,
                step_index=step_index,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            )
            steps_output[f"step{step}"] = joined
            step_arrays[step][source] = arrays
        source_output[source] = {
            "pairing": dict(first["pairings"][source]),
            "steps": steps_output,
        }
    macro_steps = {
        f"step{step}": {
            "windows": _macro_step(
                step_arrays[step],
                step_index=step_index,
                bootstrap_seed=bootstrap_seed,
                bootstrap_samples=bootstrap_samples,
            )
        }
        for step_index, step in enumerate(STEPS)
    }
    macro_durability, source_durability = _durability(
        step_arrays,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    for source in PERCEPTION_SOURCE_NAMES:
        source_output[source]["durability"] = source_durability[source]
    protocol = {
        "name": PROTOCOL_NAME,
        "arms": list(ARMS),
        "steps": list(STEPS),
        "primary_step": 4000,
        "durability_step": 3000,
        "sources": list(PERCEPTION_SOURCE_NAMES),
        "windows": list(WINDOWS),
        "examples_per_source": examples,
        "pairing_seed": first["protocol"]["pairing_seed"],
        "pairing_sha256": dict(first["protocol"]["pairing_sha256"]),
        "perception_provenance_sha256": first["protocol"]["perception_provenance_sha256"],
        "source_audit_fingerprint": first["protocol"]["source_audit_fingerprint"],
        "profile_pair_receipt_sha256": profile_pair_ref["sha256"],
        "evaluator_protocol_sha256": first["protocol"]["sha256"],
        "gap_sign": "wrong_ce - correct_ce; positive means correct-image reliance",
        "correct_ce_effect_sign": "control_correct_ce - treatment_correct_ce",
        "did_sign": "treatment_gap - control_gap; positive favors vision adaptation",
        "source_aggregation": "equal-weight mean over the exact eight sources",
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
    }
    protocol["sha256"] = _canonical_sha256(protocol)
    summary = {
        "steps": macro_steps,
        "durability": {"windows": macro_durability},
    }
    return {
        "checkpoints": checkpoints,
        "protocol": protocol,
        "sources": source_output,
        "summary": summary,
    }


def _policy_metrics(components: Mapping[str, Any]) -> dict[str, Any]:
    summary = components["summary"]
    endpoint = summary["steps"]["step4000"]["windows"]["all"]
    durability = summary["durability"]["windows"]["all"]
    sources = components["sources"]
    return {
        "macro": {
            "did_ci_low": endpoint["did"]["ci"]["low"],
            "treatment_gap_ci_low": endpoint["treatment"]["gap"]["ci"]["low"],
            "control_correct_ce": endpoint["control"]["correct_ce"],
            "treatment_correct_ce": endpoint["treatment"]["correct_ce"],
            "treatment_gap": endpoint["treatment"]["gap"]["mean"],
            "step3000_treatment_gap": durability["treatment_gap_step3000"],
        },
        "sources": {
            source: {
                "control_correct_ce": sources[source]["steps"]["step4000"]["windows"]["all"][
                    "control"
                ]["correct_ce"],
                "treatment_correct_ce": sources[source]["steps"]["step4000"]["windows"]["all"][
                    "treatment"
                ]["correct_ce"],
            }
            for source in PERCEPTION_SOURCE_NAMES
        },
    }


def _load_inputs(
    inputs: Any, *, verify_live_inputs: bool
) -> tuple[dict[str, Mapping[int, Mapping[str, Any]]], dict[str, str]]:
    inputs = _exact(
        inputs,
        frozenset({"profile_pair_receipt", "evaluations"}),
        name="outcome inputs",
    )
    profile_ref, profile_payload = _read_ref(
        inputs["profile_pair_receipt"], name="profile-pair receipt"
    )
    profile_payload = _validated_profile_pair_payload(
        profile_payload, raw_sha256=profile_ref["sha256"]
    )
    profile_path = Path(profile_ref["path"])
    evaluations_value = _exact(inputs["evaluations"], frozenset(ARMS), name="evaluation inputs")
    evaluations: dict[str, Mapping[int, Mapping[str, Any]]] = {}
    for arm in ARMS:
        arm_value = _exact(
            evaluations_value[arm], frozenset(STEP_KEYS), name=f"{arm} evaluation inputs"
        )
        packets: dict[int, Mapping[str, Any]] = {}
        for step in STEPS:
            _, receipt = _read_ref(arm_value[f"step{step}"], name=f"{arm} step{step}")
            packets[step] = _validate_evaluator_receipt(
                receipt,
                arm=arm,
                step=step,
                profile_pair_sha256=profile_ref["sha256"],
                profile_pair_path=profile_path,
                profile_pair_payload=profile_payload,
                verify_live_inputs=verify_live_inputs,
            )
        evaluations[arm] = packets
    return evaluations, profile_ref


def validate_outcome_receipt(
    receipt: Mapping[str, Any], *, verify_live_inputs: bool = True
) -> dict[str, Any]:
    """Re-open all four inputs and exactly rederive the paired outcome.

    :returns: Normalized checkpoint identities and policy metrics consumed by promotion.
    """
    _exact(receipt, TOP_FIELDS, name="perception outcome receipt")
    if (
        receipt["format"] != FORMAT
        or type(receipt["version"]) is not int
        or receipt["version"] != VERSION
        or receipt["status"] != "passed"
    ):
        raise ValueError("Perception outcome receipt identity or status is incompatible")
    created_at = receipt["created_at"]
    if not isinstance(created_at, str):
        raise TypeError("Outcome created_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Outcome created_at is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("Outcome created_at must contain a timezone")
    unsigned = dict(receipt)
    content_sha = unsigned.pop("content_sha256")
    if not _is_sha256(content_sha) or content_sha != _canonical_sha256(unsigned):
        raise ValueError("Outcome content SHA-256 differs")
    producer = _exact(receipt["producer"], frozenset({"path", "sha256"}), name="outcome producer")
    _implementation_reference(
        path=producer["path"],
        sha256=producer["sha256"],
        name="outcome producer",
        canonical_path=Path(__file__).resolve(),
        verify_live=verify_live_inputs,
    )
    evaluations, profile_ref = _load_inputs(
        receipt["inputs"], verify_live_inputs=verify_live_inputs
    )
    protocol = receipt["protocol"]
    if not isinstance(protocol, Mapping):
        raise TypeError("Outcome protocol must be an object")
    bootstrap = protocol.get("bootstrap")
    if (
        not isinstance(bootstrap, Mapping)
        or set(bootstrap) != {"method", "confidence", "samples", "seed"}
        or bootstrap.get("method") != BOOTSTRAP_METHOD
        or bootstrap.get("confidence") != BOOTSTRAP_CONFIDENCE
        or bootstrap.get("seed") != DEFAULT_BOOTSTRAP_SEED
        or bootstrap.get("samples") != DEFAULT_BOOTSTRAP_SAMPLES
    ):
        raise ValueError("Outcome bootstrap protocol is missing")
    seed = bootstrap.get("seed")
    samples = bootstrap.get("samples")
    if seed != DEFAULT_BOOTSTRAP_SEED or samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError("Outcome bootstrap seed/sample count differs from locked policy")
    components = _build_components(
        evaluations,
        profile_pair_ref=profile_ref,
        bootstrap_seed=seed,
        bootstrap_samples=samples,
    )
    for field in ("checkpoints", "protocol", "sources", "summary"):
        if _canonical_bytes(receipt[field]) != _canonical_bytes(components[field]):
            raise ValueError(f"Stored outcome {field} differs from exact rederivation")
    return {
        "checkpoints": components["checkpoints"],
        "policy_metrics": _policy_metrics(components),
    }


def _cli_input_refs(args: argparse.Namespace) -> dict[str, Any]:
    evaluations: dict[str, Any] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        cli_arm = "control" if arm == "frozen_vision_control" else "treatment"
        for step in STEPS:
            path = getattr(args, f"{cli_arm}_step{step}")
            digest = getattr(args, f"expected_{cli_arm}_step{step}_sha256")
            evaluations[arm][f"step{step}"] = _artifact_ref(path, digest, name=f"{arm} step{step}")
    return {
        "profile_pair_receipt": _artifact_ref(
            args.profile_pair_receipt,
            args.expected_profile_pair_receipt_sha256,
            name="profile-pair receipt",
        ),
        "evaluations": evaluations,
    }


def build_outcome_receipt(
    *,
    inputs: Mapping[str, Any],
    output: Path,
    bootstrap_seed: int,
    bootstrap_samples: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build, self-validate, and atomically publish one immutable outcome receipt."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable outcome receipt {output}")
    if bootstrap_seed != DEFAULT_BOOTSTRAP_SEED or bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError("Bootstrap seed/sample count differs from locked policy")
    evaluations, profile_ref = _load_inputs(inputs, verify_live_inputs=True)
    components = _build_components(
        evaluations,
        profile_pair_ref=profile_ref,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    producer = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "status": "passed",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "producer": {"path": str(producer), "sha256": _sha256_file(producer)},
        "inputs": dict(inputs),
        **components,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    # The initial load above already re-hashed every live checkpoint and implementation. Re-run
    # the full semantic derivation without duplicating the multi-hundred-GB checkpoint scan.
    validate_outcome_receipt(payload, verify_live_inputs=False)
    bridge._write_json_atomic(output, payload, overwrite=False)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
        or args.bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES
    ):
        raise ValueError("Bootstrap seed/sample count differs from locked policy")
    inputs = _cli_input_refs(args)
    build_outcome_receipt(
        inputs=inputs,
        output=Path(args.output).expanduser().resolve(),
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
