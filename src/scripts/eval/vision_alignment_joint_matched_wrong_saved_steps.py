"""Evaluate exact late saved endpoints of the frozen joint matched/wrong protocol.

This V2 wrapper leaves the step-4000/8000 evaluator byte-for-byte unchanged and reuses it only
as a frozen scoring engine. It admits exactly steps 12000, 14400, and 16000, preserves the
step-14400 ephemeral marker in the receipt, and reuses the exact V1 pairing manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def _load_scoring_engine() -> ModuleType:
    path = Path(__file__).resolve().with_name("vision_alignment_joint_matched_wrong.py")
    name = "_vision_alignment_joint_matched_wrong_v1_for_saved_steps"
    cached = sys.modules.get(name)
    if isinstance(cached, ModuleType):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen joint scoring engine {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scoring = _load_scoring_engine()

RECEIPT_FORMAT = "vision_alignment_joint_matched_wrong_receipt"
RECEIPT_VERSION = 2
PROTOCOL_NAME = "vision-alignment-joint-native-matched-wrong-saved-endpoints-v2"
ENDPOINT_CONTRACT = "vision-alignment-joint-saved-endpoints-v1"
ENDPOINT_SET_VERSION = "joint-saved-endpoints-v1"
ADMISSIBLE_STEPS = (12000, 14400, 16000)
EXPECTED_CONFIG_SHA256 = scoring.EXPECTED_CONFIG_SHA256
EXPECTED_SCORING_ENGINE_SHA256 = "3daf99a9996de0c3deaf62653dc43fa47eaca915785bda0b94cc1d897b25e058"
EXPECTED_V1_COMPARATOR_SHA256 = "56b3526293332b70adf73977d2edbf050a6d507b2ff7b9ef9a3ddc2bd75f53e1"
EXPECTED_PAIRING_MANIFEST_SHA256 = (
    "24a768b89ca0b73386362c3aa6db9afb27b36318e159766ac3a0cb62cf978739"
)
EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256 = (
    "f021873ba46192f62eab44b27d9f198db5b36081b548326b1547b25020b457de"
)
EXPECTED_CHECKPOINT_PARENT = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/"
    "vision-alignment-joint-v1"
)
EXPECTED_PAIRING_MANIFEST_PATH = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
    "joint-v1-matched-wrong-v1/pairing-manifest-12f5623a.json"
)
EXPECTED_PAIRING_DIR = Path(
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/"
    "joint-v1-matched-wrong-v1/pairings-12f5623a"
)
EXPECTED_MODEL_AND_OPTIM_FILES = 257
EXPECTED_MODEL_AND_OPTIM_BYTES = 384_970_228_158
EXPECTED_CHECKPOINT_REGULAR_FILES = 275
EXPECTED_WANDB_RUN_ID = "4gxnu6we"
PRIMARY_STATISTIC = "all-response wrong-minus-correct CE gap at one exact saved checkpoint"
PER_CHECKPOINT_STATISTIC = "all-response loss-weighted scalar CE"


@dataclass(frozen=True)
class _EndpointSpec:
    storage_class: str
    marker: Mapping[str, Any]
    marker_sha256: str
    total_checkpoint_bytes: int
    total_data_errors_by_rank: tuple[int, ...]


_LATE_ERROR_PANEL = (2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
ENDPOINTS = {
    12000: _EndpointSpec(
        storage_class="scheduled_permanent",
        marker={"ephemeral": False, "version": "2.5.0"},
        marker_sha256="77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03",
        total_checkpoint_bytes=384_970_543_909,
        total_data_errors_by_rank=_LATE_ERROR_PANEL,
    ),
    14400: _EndpointSpec(
        storage_class="retained_ephemeral",
        marker={"ephemeral": True, "version": "2.5.0"},
        marker_sha256="3c4b070a507487454f081c1bc4eac4a68ffa3b2eeec46b892efb5f0f6400762e",
        total_checkpoint_bytes=384_970_543_908,
        total_data_errors_by_rank=_LATE_ERROR_PANEL,
    ),
    16000: _EndpointSpec(
        storage_class="scheduled_permanent",
        marker={"ephemeral": False, "version": "2.5.0"},
        marker_sha256="77dfdeec42fe7990f4b3b9c4eeecd480edcf5066c110603b115920af38423d03",
        total_checkpoint_bytes=384_970_543_909,
        total_data_errors_by_rank=_LATE_ERROR_PANEL,
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Defaults to CHECKPOINT/config.json.")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--pairing-dir", required=True)
    parser.add_argument("--pairing-manifest", required=True)
    parser.add_argument("--expected-pairing-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    return parser


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _validate_args(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != scoring.WORLD_SIZE or local_world_size != scoring.LOCAL_WORLD_SIZE:
        raise ValueError("Saved-endpoint evaluation requires one node with WORLD_SIZE=8")
    if args.expected_config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError("Config pin is not the reviewed joint-v1 config")
    _step_from_root(_lexical_absolute(Path(args.checkpoint)))
    if args.examples != 512:
        raise ValueError("Saved-endpoint evaluation freezes --examples=512")
    if args.expected_pairing_manifest_sha256 != EXPECTED_PAIRING_MANIFEST_SHA256:
        raise ValueError("Saved-endpoint evaluation requires the exact shared V1 pairing manifest")
    if _lexical_absolute(Path(args.pairing_manifest)) != EXPECTED_PAIRING_MANIFEST_PATH:
        raise ValueError("Saved-endpoint pairing manifest path differs from the shared V1 path")
    if _lexical_absolute(Path(args.pairing_dir)) != EXPECTED_PAIRING_DIR:
        raise ValueError("Saved-endpoint pairing directory differs from the shared V1 directory")
    for value, name in (
        (args.checkpoint_load_threads, "--checkpoint-load-threads"),
        (args.checkpoint_hash_workers, "--checkpoint-hash-workers"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be positive")


def _step_from_root(root: Path) -> int:
    match = re.fullmatch(r"step([0-9]+)", root.name)
    step = int(match.group(1)) if match is not None else None
    if step not in ADMISSIBLE_STEPS:
        raise ValueError(
            "Only exact saved joint steps 12000, 14400, and 16000 are admissible; "
            "nearest-step substitution is forbidden"
        )
    assert step is not None
    if root != EXPECTED_CHECKPOINT_PARENT / f"step{step}":
        raise ValueError("Saved endpoint is outside the exact joint-v1 checkpoint lineage")
    return step


def _checkpoint_config_identity(
    checkpoint: Path, config_path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Any], int]:
    root = scoring._checkpoint_root(checkpoint)
    step = _step_from_root(root)
    if config_path != root / "config.json":
        raise ValueError("Saved-endpoint config must be CHECKPOINT/config.json")
    raw_config, digest = scoring._load_json_bytes(
        config_path,
        expected_sha256=expected_sha256,
        name="saved-endpoint checkpoint config",
    )
    if not isinstance(raw_config, Mapping):
        raise TypeError("Saved-endpoint checkpoint config must be an object")
    metadata = raw_config.get("vision_alignment")
    data = raw_config.get("data")
    evaluation = raw_config.get("evaluation")
    train_module = raw_config.get("train_module")
    launch = raw_config.get("launch")
    trainer = raw_config.get("trainer")
    if not all(
        isinstance(value, Mapping)
        for value in (metadata, data, evaluation, train_module, launch, trainer)
    ):
        raise ValueError("Saved-endpoint config lacks required joint sections")
    metadata = cast(Mapping[str, Any], metadata)
    data = cast(Mapping[str, Any], data)
    evaluation = cast(Mapping[str, Any], evaluation)
    train_module = cast(Mapping[str, Any], train_module)
    launch = cast(Mapping[str, Any], launch)
    trainer = cast(Mapping[str, Any], trainer)
    ep_config = train_module.get("ep_config")
    checkpointer = trainer.get("callbacks", {}).get("checkpointer")
    launch_git = launch.get("git")
    if (
        digest != EXPECTED_CONFIG_SHA256
        or raw_config.get("phase") != "joint"
        or metadata.get("phase") != "joint"
        or metadata.get("lineage_id") != scoring.EXPECTED_LINEAGE
        or raw_config.get("required_run_name") != scoring.EXPECTED_LINEAGE
        or raw_config.get("reviewed_profile_path") != scoring.EXPECTED_REVIEWED_PROFILE
        or raw_config.get("reviewed_profile_sha256") != scoring.EXPECTED_REVIEWED_PROFILE_SHA256
        or raw_config.get("reviewed_profile_allowlist_path")
        != "configs/vision_moe/vision_alignment/joint/approved_profiles.json"
        or data.get("sequence_length") != scoring.SEQUENCE_LENGTH
        or data.get("joint_visual_projection_sha256") != scoring.EXPECTED_PROJECTION_SHA256
        or data.get("source_audit_fingerprint") != scoring.EXPECTED_SOURCE_AUDIT_FINGERPRINT
        or evaluation.get("examples_per_source") != 512
        or evaluation.get("rank_batch_instances") != scoring.RANK_BATCH_INSTANCES
        or evaluation.get("seed") != scoring.PAIRING_SEED
        or raw_config.get("perception_trainability_arm") != "treatment"
        or not isinstance(ep_config, Mapping)
        or ep_config.get("degree") != scoring.EP_DEGREE
        or train_module.get("rank_microbatch_size") != scoring.SEQUENCE_LENGTH
        or train_module.get("max_sequence_length") != scoring.SEQUENCE_LENGTH
        or launch.get("workspace") != "ai2/molmofication"
        or launch.get("beaker_image") != scoring.EXPECTED_BEAKER_IMAGE
        or not isinstance(launch_git, Mapping)
        or launch_git.get("ref") != "7e42a7e3064bd944806a5cf5d351ec4f6dc24e42"
        or trainer.get("save_folder") != str(EXPECTED_CHECKPOINT_PARENT)
        or not isinstance(checkpointer, Mapping)
        or checkpointer.get("save_interval") != 4000
        or checkpointer.get("save_async") is not False
    ):
        raise ValueError("Checkpoint config differs from the reviewed joint-v1 contract")
    profile = Path(scoring.EXPECTED_REVIEWED_PROFILE).resolve()
    allowlist = Path(str(raw_config["reviewed_profile_allowlist_path"])).resolve()
    if (
        scoring._stable_file_sha256(profile, name="reviewed joint profile")
        != scoring.EXPECTED_REVIEWED_PROFILE_SHA256
        or scoring._stable_file_sha256(allowlist, name="reviewed joint profile allowlist")
        != raw_config["reviewed_profile_allowlist_sha256"]
    ):
        raise ValueError("Live reviewed profile or allowlist bytes differ")
    identity = {
        "path": str(config_path),
        "sha256": digest,
        "phase": "joint",
        "lineage_id": scoring.EXPECTED_LINEAGE,
        "run_name": scoring.EXPECTED_LINEAGE,
        "step": step,
        "reviewed_profile_path": scoring.EXPECTED_REVIEWED_PROFILE,
        "reviewed_profile_sha256": scoring.EXPECTED_REVIEWED_PROFILE_SHA256,
        "reviewed_profile_allowlist_path": raw_config["reviewed_profile_allowlist_path"],
        "reviewed_profile_allowlist_sha256": raw_config["reviewed_profile_allowlist_sha256"],
        "training_git_ref": launch_git["ref"],
        "training_beaker_image": launch["beaker_image"],
    }
    return raw_config, identity, step


def _validate_trainer_state(state: Mapping[str, Any], *, rank: int, step: int) -> dict[str, Any]:
    required = {
        "global_step",
        "global_train_tokens_seen",
        "global_train_petaflops",
        "max_steps",
        "data_loader",
        "epoch",
        "world_size",
        "rng",
        "callbacks",
    }
    spec = ENDPOINTS[step]
    loader = state.get("data_loader")
    callbacks = state.get("callbacks")
    wandb = callbacks.get("wandb") if isinstance(callbacks, Mapping) else None
    packing = loader.get("packing_state") if isinstance(loader, Mapping) else None
    expected_datasets = [
        "audited_alignment",
        "cosyn_point",
        "count_numeric",
        "native_text_replay",
        "ocr_document",
        "pixmo_caption",
        "pixmo_points_basic",
        "pixmo_points_high_frequency",
        "pixmo_transcript",
    ]
    expected_run_id = EXPECTED_WANDB_RUN_ID if rank == 0 else None
    if (
        set(state) != required
        or state.get("global_step") != step
        or state.get("global_train_tokens_seen") != step * 1_048_576
        or state.get("max_steps") != 16000
        or state.get("world_size") != scoring.TRAINING_WORLD_SIZE
        or not isinstance(loader, Mapping)
        or loader.get("batches_processed") != step
        or loader.get("consecutive_data_errors") != 0
        or loader.get("total_data_errors") != spec.total_data_errors_by_rank[rank]
        or not isinstance(packing, Mapping)
        or packing.get("dp_world_size") != scoring.TRAINING_WORLD_SIZE
        or packing.get("dp_rank") != rank
        or packing.get("rank_instances") != 8
        or packing.get("seq_len") != scoring.SEQUENCE_LENGTH
        or packing.get("dataset_names") != expected_datasets
        or not isinstance(wandb, Mapping)
        or wandb.get("step") != step
        or wandb.get("name") != scoring.EXPECTED_LINEAGE
        or wandb.get("project") != "vision-alignment"
        or wandb.get("run_id") != expected_run_id
    ):
        raise ValueError(f"Trainer rank{rank} state differs from saved joint step{step} contract")
    return {
        "global_step": step,
        "global_train_tokens_seen": state["global_train_tokens_seen"],
        "max_steps": state["max_steps"],
        "world_size": state["world_size"],
        "batches_processed": loader["batches_processed"],
        "consecutive_data_errors": loader["consecutive_data_errors"],
        "total_data_errors": loader["total_data_errors"],
        "wandb_run_id": wandb["run_id"],
        "wandb_name": wandb["name"],
    }


def _checkpoint_identity(
    checkpoint: Path,
    config_path: Path,
    *,
    step: int,
    hash_workers: int,
) -> dict[str, Any]:
    """Bind all bytes and the true storage class of one exact saved endpoint."""
    spec = ENDPOINTS[step]
    identity = scoring._model_checkpoint_identity(
        checkpoint,
        config_path,
        hash_workers=hash_workers,
    )
    model_identity = identity.pop("identity_sha256")
    root = Path(str(identity["root"]))
    if root != EXPECTED_CHECKPOINT_PARENT / f"step{step}":
        raise ValueError("Checkpoint root differs from the exact saved endpoint")
    root_entries = sorted(path.name for path in root.iterdir())
    if root_entries != [".metadata.json", "config.json", "model_and_optim", "train"]:
        raise ValueError("Saved endpoint root entries differ")
    inventory = identity["state_file_inventory"]
    if (
        len(inventory) != EXPECTED_MODEL_AND_OPTIM_FILES
        or sum(int(record["size"]) for record in inventory) != EXPECTED_MODEL_AND_OPTIM_BYTES
    ):
        raise ValueError("Saved endpoint model/optimizer inventory size differs")
    marker, marker_raw_sha = scoring._load_json_bytes(
        root / ".metadata.json", name="saved-endpoint checkpoint marker"
    )
    if marker != spec.marker or marker_raw_sha != spec.marker_sha256:
        raise ValueError("Saved endpoint marker or storage class differs")
    train_dir = scoring.perception._direct_existing_path(
        root / "train", name="saved-endpoint trainer-state directory"
    )
    expected_names = [f"rank{rank}.pt" for rank in range(scoring.TRAINING_WORLD_SIZE)]
    observed_names = sorted((path.name for path in train_dir.iterdir()), key=expected_names.index)
    if expected_names != observed_names:
        raise ValueError("Saved endpoint must contain exactly train/rank0.pt through rank15.pt")
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for rank in range(scoring.TRAINING_WORLD_SIZE):
        record, state = scoring._read_trainer_state(train_dir / f"rank{rank}.pt", root=root)
        records.append(record)
        summaries.append(_validate_trainer_state(state, rank=rank, step=step))
    leader_run_id = summaries[0]["wandb_run_id"]
    shared = [
        {
            key: value
            for key, value in summary.items()
            if key not in {"wandb_run_id", "total_data_errors"}
        }
        for summary in summaries
    ]
    if any(summary != shared[0] for summary in shared[1:]):
        raise ValueError("Saved-endpoint trainer ranks disagree on progress or run identity")
    config_record = scoring._stable_checkpoint_record(config_path, root=root)
    marker_record = scoring._stable_checkpoint_record(root / ".metadata.json", root=root)
    total_bytes = (
        sum(int(record["size"]) for record in inventory)
        + sum(int(record["size"]) for record in records)
        + int(config_record["size"])
        + int(marker_record["size"])
    )
    if (
        len(inventory) + len(records) + 2 != EXPECTED_CHECKPOINT_REGULAR_FILES
        or total_bytes != spec.total_checkpoint_bytes
    ):
        raise ValueError("Saved endpoint complete file count or byte count differs")
    identity.update(
        {
            "model_and_optim_identity_sha256": model_identity,
            "checkpoint_step": step,
            "permanent": not bool(spec.marker["ephemeral"]),
            "checkpoint_marker": dict(marker),
            "checkpoint_marker_sha256": marker_raw_sha,
            "trainer_state_rank_count": scoring.TRAINING_WORLD_SIZE,
            "trainer_state_file_inventory": records,
            "trainer_state_file_inventory_sha256": scoring._canonical_sha256(records),
            "trainer_state_summary": {**shared[0], "wandb_run_id": leader_run_id},
            "trainer_state_total_data_errors_by_rank": [
                summary["total_data_errors"] for summary in summaries
            ],
            "trainer_state_total_data_errors_sum": sum(
                int(summary["total_data_errors"]) for summary in summaries
            ),
        }
    )
    identity["identity_sha256"] = scoring._canonical_sha256(identity)
    return identity


def _checkpoint_identity_distributed(
    checkpoint: Path,
    config_path: Path,
    *,
    step: int,
    hash_workers: int,
) -> dict[str, Any]:
    packet: list[Any] = [None]
    if scoring.dist.get_rank() == 0:
        try:
            packet[0] = {
                "ok": True,
                "identity": _checkpoint_identity(
                    checkpoint,
                    config_path,
                    step=step,
                    hash_workers=hash_workers,
                ),
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    scoring.dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not identify saved joint endpoint: {detail}")
    value = result.get("identity")
    if not isinstance(value, Mapping):
        raise TypeError("Saved-endpoint checkpoint identity broadcast is malformed")
    return dict(value)


def _endpoint_identity(step: int) -> dict[str, Any]:
    spec = ENDPOINTS[step]
    return {
        "contract": ENDPOINT_CONTRACT,
        "admissible_steps": list(ADMISSIBLE_STEPS),
        "step": step,
        "storage_class": spec.storage_class,
        "nearest_step_substitution": False,
    }


def _validate_receipt_contract(
    *,
    step: int,
    checkpoint: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    pairing_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    spec = ENDPOINTS[step]
    expected_permanent = not bool(spec.marker["ephemeral"])
    if (
        checkpoint.get("checkpoint_step") != step
        or checkpoint.get("permanent") is not expected_permanent
        or checkpoint.get("checkpoint_marker") != spec.marker
        or checkpoint.get("checkpoint_marker_sha256") != spec.marker_sha256
        or checkpoint.get("trainer_state_total_data_errors_by_rank")
        != list(spec.total_data_errors_by_rank)
        or checkpoint.get("trainer_state_total_data_errors_sum")
        != sum(spec.total_data_errors_by_rank)
    ):
        raise ValueError("Checkpoint evidence differs from its exact saved-endpoint contract")
    if (
        checkpoint_config.get("step") != step
        or checkpoint_config.get("sha256") != EXPECTED_CONFIG_SHA256
    ):
        raise ValueError("Checkpoint config does not bind the exact saved endpoint")
    if (
        pairing_manifest.get("sha256") != EXPECTED_PAIRING_MANIFEST_SHA256
        or pairing_manifest.get("content_sha256") != EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256
        or _lexical_absolute(Path(str(pairing_manifest.get("path"))))
        != EXPECTED_PAIRING_MANIFEST_PATH
    ):
        raise ValueError("Receipt does not reference the exact shared V1 pairing manifest")
    if (
        protocol.get("name") != PROTOCOL_NAME
        or protocol.get("endpoint_set_version") != ENDPOINT_SET_VERSION
        or protocol.get("admissible_steps") != list(ADMISSIBLE_STEPS)
        or protocol.get("evaluated_step") != step
        or protocol.get("nearest_step_substitution") is not False
        or protocol.get("descriptive_only") is not True
        or protocol.get("promotion_eligible") is not False
    ):
        raise ValueError("Protocol does not bind the exact saved endpoint")


def _protocol(
    *,
    step: int,
    examples: int,
    pairing_sha256: Mapping[str, str],
    dp_world_size: int,
    checkpoint_config: Mapping[str, Any],
    projection: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    native_identity: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = scoring._protocol(
        examples=examples,
        pairing_sha256=pairing_sha256,
        dp_world_size=dp_world_size,
        checkpoint_config=checkpoint_config,
        projection=projection,
        source_audit=source_audit,
        native_identity=native_identity,
    )
    protocol.update(
        {
            "name": PROTOCOL_NAME,
            "primary_statistic": PRIMARY_STATISTIC,
            "per_checkpoint_statistic": PER_CHECKPOINT_STATISTIC,
            "endpoint_set_version": ENDPOINT_SET_VERSION,
            "admissible_steps": list(ADMISSIBLE_STEPS),
            "evaluated_step": step,
            "nearest_step_substitution": False,
        }
    )
    return protocol


def _producer_identity() -> dict[str, str]:
    producer = Path(__file__).resolve()
    validator = producer.with_name("vision_alignment_joint_matched_wrong_saved_steps_validate.py")
    engine = Path(str(scoring.__file__)).resolve()
    comparator_v1 = engine.with_name("vision_alignment_joint_matched_wrong_compare.py")
    perception = Path(str(scoring.perception.__file__)).resolve()
    bridge = Path(str(scoring.bridge.__file__)).resolve()
    pairing_source = inspect.getsourcefile(scoring.build_matched_wrong_image_pairing)
    training = producer.parents[1] / "train" / "Vision-Alignment.py"
    if pairing_source is None:
        raise RuntimeError("Could not locate matched/wrong pairing implementation")
    engine_sha = scoring._stable_file_sha256(engine, name="frozen V1 scoring engine")
    comparator_v1_sha = scoring._stable_file_sha256(comparator_v1, name="frozen V1 comparator")
    if (
        engine_sha != EXPECTED_SCORING_ENGINE_SHA256
        or comparator_v1_sha != EXPECTED_V1_COMPARATOR_SHA256
    ):
        raise ValueError("Frozen V1 evaluator/comparator bytes changed")
    pairing = Path(pairing_source).resolve()
    return {
        "path": str(producer),
        "sha256": scoring._stable_file_sha256(producer, name="saved-endpoints evaluator"),
        "validator_path": str(validator),
        "validator_sha256": scoring._stable_file_sha256(
            validator, name="saved-endpoints receipt validator"
        ),
        "scoring_engine_path": str(engine),
        "scoring_engine_sha256": engine_sha,
        "perception_helper_path": str(perception),
        "perception_helper_sha256": scoring._stable_file_sha256(
            perception, name="perception evaluator helper"
        ),
        "bridge_helper_path": str(bridge),
        "bridge_helper_sha256": scoring._stable_file_sha256(bridge, name="bridge evaluator helper"),
        "pairing_implementation_path": str(pairing),
        "pairing_implementation_sha256": scoring._stable_file_sha256(
            pairing, name="pairing implementation"
        ),
        "training_contract_path": str(training),
        "training_contract_sha256": scoring._stable_file_sha256(
            training, name="joint training contract"
        ),
    }


def _receipt_payload(
    *,
    step: int,
    checkpoint: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    load_coverage: Mapping[str, Any],
    projection: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    pairing_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    visual_results: Mapping[str, Any],
    blank_results: Mapping[str, Any],
    native_result: Mapping[str, Any],
    producer: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_receipt_contract(
        step=step,
        checkpoint=checkpoint,
        checkpoint_config=checkpoint_config,
        pairing_manifest=pairing_manifest,
        protocol=protocol,
    )
    payload: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "version": RECEIPT_VERSION,
        "status": "valid",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": dict(producer),
        "git": dict(git),
        "artifact_policy": {
            "output_overwrite_enabled": False,
            "pairing_manifest_requires_sha256_pin": True,
            "all_pairings_rehashed": True,
            "all_pairings_deterministically_rebuilt": True,
            "checkpoint_private_snapshot": (
                "full model_and_optim same-FD verified byte copy; load only; delete after load"
            ),
            "checkpoint_post_identity_rehashed": True,
            "native_sources_full_hash_pre_and_post": True,
            "descriptive_only": True,
            "promotion_eligible": False,
            "checkpoint_source_marker_preserved": True,
            "retained_ephemeral_not_promoted_to_permanent": True,
            "nearest_step_substitution_allowed": False,
        },
        "endpoint": _endpoint_identity(step),
        "checkpoint": dict(checkpoint),
        "checkpoint_config": dict(checkpoint_config),
        "load_coverage": dict(load_coverage),
        "projection": dict(projection),
        "source_audit": dict(source_audit),
        "tokenizer": dict(tokenizer),
        "pairing_manifest": dict(pairing_manifest),
        "protocol": dict(protocol),
        "visual_results": {
            source: visual_results[source] for source in scoring.JOINT_VISUAL_SOURCE_NAMES
        },
        "blank_results": {source: blank_results[source] for source in scoring.BLANK_SOURCE_NAMES},
        "native_result": dict(native_result),
    }
    payload["content_sha256"] = scoring._canonical_sha256(payload)
    return payload


def _load_validator_contract() -> ModuleType:
    path = (
        Path(__file__)
        .resolve()
        .with_name("vision_alignment_joint_matched_wrong_saved_steps_validate.py")
    )
    name = "_vision_alignment_joint_matched_wrong_saved_steps_validator_for_prepublication"
    cached = sys.modules.get(name)
    if isinstance(cached, ModuleType):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load saved-endpoints receipt validator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_validated_receipt(
    output: Path,
    payload: Mapping[str, Any],
    *,
    work_dir: Path,
    step: int,
) -> str:
    """Cross-validate retained candidate bytes before publishing those same bytes."""
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = scoring._artifact_path(work_dir, name="saved receipt validation work directory")
    validation_root = Path(
        tempfile.mkdtemp(prefix=".joint-saved-receipt-validation-", dir=work_dir)
    )
    validation_identity = scoring._stat_signature(validation_root.lstat())[:3]
    try:
        raw = scoring._json_output_bytes(payload)
        candidate = validation_root / "receipt.json"
        candidate_sha = scoring._write_bytes_exclusive(candidate, raw)
        validator_module = _load_validator_contract()
        validator = getattr(validator_module, "validate_evaluator_receipt", None)
        if not callable(validator):
            raise RuntimeError("Saved-endpoints validator lacks validate_evaluator_receipt")
        validator(candidate, candidate_sha, step, verify_live_checkpoint=False)
        current_producer = _producer_identity()
        current_git = scoring._validate_clean_git_identity(scoring.bridge._git_identity())
        if current_producer != payload.get("producer") or current_git != payload.get("git"):
            raise RuntimeError(
                "Saved evaluator implementation or clean git identity changed before publication"
            )
        published_sha = scoring._write_bytes_exclusive(output, raw)
        if published_sha != candidate_sha:
            raise RuntimeError("Published saved-endpoint receipt differs from validated bytes")
        loaded, reloaded_sha = scoring._load_json_bytes(
            output,
            expected_sha256=candidate_sha,
            name="published saved-endpoint evaluator receipt",
        )
        if (
            scoring._canonical_bytes(loaded) != scoring._canonical_bytes(payload)
            or reloaded_sha != candidate_sha
        ):
            raise RuntimeError("Published saved-endpoint receipt failed its strict reload")
        return published_sha
    finally:
        if (
            validation_root.name.startswith(".joint-saved-receipt-validation-")
            and validation_root.parent == work_dir
            and validation_root.exists()
            and scoring._stat_signature(validation_root.lstat())[:3] == validation_identity
        ):
            shutil.rmtree(validation_root)


def _write_receipt_distributed(
    output: Path,
    payload: Mapping[str, Any],
    *,
    work_dir: Path,
    step: int,
) -> None:
    packet: list[Any] = [None]
    if scoring.dist.get_rank() == 0:
        try:
            packet[0] = {
                "ok": True,
                "sha256": _write_validated_receipt(
                    output,
                    payload,
                    work_dir=work_dir,
                    step=step,
                ),
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    scoring.dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not persist saved-endpoint evaluation receipt: {detail}")


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate one exact saved joint endpoint with the frozen V1 scoring engine."""
    args = _parser().parse_args(argv)
    _validate_args(args)
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime_state = scoring._initialize_runtime(pairing_only=False)
    try:
        checkpoint = scoring.perception._direct_existing_path(
            Path(args.checkpoint), name="saved joint checkpoint"
        )
        root = scoring._checkpoint_root(checkpoint)
        config_path = scoring.perception._direct_existing_path(
            Path(args.config).expanduser() if args.config is not None else root / "config.json",
            name="saved joint checkpoint config",
        )
        raw_config, checkpoint_config, step = _checkpoint_config_identity(
            checkpoint,
            config_path,
            args.expected_config_sha256,
        )
        output = scoring._artifact_path(Path(args.output), name="saved joint output")
        scoring._artifact_preflight_distributed(output)
        initial_producer = _producer_identity()
        initial_git = scoring._validate_clean_git_identity(scoring.bridge._git_identity())

        artifacts = raw_config["artifacts"]
        tokenizer, token_ids = scoring.load_pinned_vision_alignment_tokenizer(
            identifier=artifacts["tokenizer_id"],
            revision=artifacts["tokenizer_revision"],
            expected_fingerprint=artifacts["tokenizer_fingerprint"],
            cache_dir=artifacts["hf_cache_dir"],
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("Pinned joint tokenizer has no pad token")
        if int(raw_config["model"]["image_patch_token_id"]) != token_ids.im_patch_id:
            raise ValueError("Saved checkpoint image-patch ID differs from tokenizer")
        tokenizer_ref = scoring._tokenizer_identity(raw_config, token_ids)
        projection = scoring._load_projection_distributed(raw_config, token_ids)
        datasets = scoring._build_visual_datasets(projection, tokenizer, token_ids)
        projection_ref = scoring._validate_visual_population_distributed(datasets, projection)
        experiment, source_audit = scoring._validate_source_audit_distributed(
            raw_config, token_ids, projection
        )
        pairing_dir = scoring._artifact_path(Path(args.pairing_dir), name="pairing directory")
        content_ids = scoring._content_ids(projection)
        pairing_manifest_path = scoring._artifact_path(
            Path(args.pairing_manifest), name="pairing manifest"
        )
        (
            pairing_payloads,
            pairing_metadata,
            pairing_manifest_ref,
        ) = scoring._load_pairing_manifest_distributed(
            pairing_manifest_path,
            args.expected_pairing_manifest_sha256,
            datasets=datasets,
            content_ids=content_ids,
            pairing_dir=pairing_dir,
            config_sha256=checkpoint_config["sha256"],
            checkpoint_config=checkpoint_config,
            projection=projection_ref,
            source_audit=source_audit,
            tokenizer=tokenizer_ref,
        )
        if (
            pairing_manifest_ref["sha256"] != EXPECTED_PAIRING_MANIFEST_SHA256
            or pairing_manifest_ref["content_sha256"] != EXPECTED_PAIRING_MANIFEST_CONTENT_SHA256
        ):
            raise RuntimeError("Loaded pairing manifest differs from the exact shared V1 artifact")
        example_counts = {int(payload["recipient_count"]) for payload in pairing_payloads.values()}
        if example_counts != {504}:
            raise ValueError("Saved-endpoint pairings must contain exactly 504 recipients/source")
        examples_per_source = example_counts.pop()
        native_dataset, native_identity = scoring._load_native_evidence_distributed(
            raw_config, tokenizer, token_ids, experiment
        )
        checkpoint_identity = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            step=step,
            hash_workers=args.checkpoint_hash_workers,
        )
        if checkpoint_identity["config_sha256"] != checkpoint_config["sha256"]:
            raise RuntimeError("Saved checkpoint inventory does not bind the expected config")
        work_dir = scoring._validate_work_dir(
            Path(args.work_dir),
            checkpoint_root=root,
            pairing_dir=pairing_dir,
            output=output,
            pairing_manifest=pairing_manifest_path,
            raw_config=raw_config,
        )
        model, module_config = scoring.bridge._build_model_and_module(
            raw_config,
            sequence_length=scoring.SEQUENCE_LENGTH,
            rank_batch_instances=scoring.RANK_BATCH_INSTANCES,
        )
        train_module = module_config.build(model, eval_only=True)
        snapshot_state_dir = scoring._materialize_checkpoint_snapshot_distributed(
            checkpoint_identity,
            base_dir=work_dir / "checkpoint-snapshots",
        )
        try:
            load_coverage = scoring.bridge._native_checkpoint_load_coverage_distributed(
                train_module, snapshot_state_dir
            )
            train_module.load_state_dict_direct(
                snapshot_state_dir,
                process_group=scoring.dist.group.WORLD,
                thread_count=args.checkpoint_load_threads,
                load_optim_state=False,
            )
        finally:
            scoring._remove_checkpoint_snapshot_distributed(snapshot_state_dir)
        load_coverage["load_completed"] = True
        load_coverage["sha256"] = scoring._canonical_sha256(
            {key: value for key, value in load_coverage.items() if key != "sha256"}
        )
        if load_coverage.get("complete") is not True:
            raise RuntimeError("Saved checkpoint load coverage is incomplete")
        dp_world_size = scoring.get_world_size(train_module.dp_process_group)
        dp_rank = scoring.get_rank(train_module.dp_process_group)
        if dp_world_size != scoring.WORLD_SIZE:
            raise ValueError("Saved-endpoint evaluation requires DP process-group size 8")
        collator = scoring.MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=scoring.SEQUENCE_LENGTH,
        )
        visual_results: dict[str, Any] = {}
        blank_results: dict[str, Any] = {}
        for source in scoring.JOINT_VISUAL_SOURCE_NAMES:
            visual, blank = scoring._evaluate_visual_source(
                train_module,
                datasets[source],
                source=source,
                pairing=pairing_payloads[source],
                pairing_sha256=pairing_metadata[source]["sha256"],
                collator=collator,
                work_dir=work_dir,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
            )
            visual_results[source] = visual
            if blank is not None:
                blank_results[source] = blank
        if set(blank_results) != set(scoring.BLANK_SOURCE_NAMES):
            raise RuntimeError("Blank controls do not cover exact caption/transcript set")
        native_result = scoring._evaluate_native_holdout(
            train_module,
            native_dataset,
            native_identity=native_identity,
            collator=collator,
            work_dir=work_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

        (
            post_payloads,
            post_metadata,
            post_manifest_ref,
        ) = scoring._load_pairing_manifest_distributed(
            pairing_manifest_path,
            args.expected_pairing_manifest_sha256,
            datasets=datasets,
            content_ids=content_ids,
            pairing_dir=pairing_dir,
            config_sha256=checkpoint_config["sha256"],
            checkpoint_config=checkpoint_config,
            projection=projection_ref,
            source_audit=source_audit,
            tokenizer=tokenizer_ref,
        )
        if (
            scoring._canonical_bytes(post_payloads) != scoring._canonical_bytes(pairing_payloads)
            or scoring._canonical_bytes(post_metadata) != scoring._canonical_bytes(pairing_metadata)
            or post_manifest_ref != pairing_manifest_ref
        ):
            raise RuntimeError("Pairing evidence changed during saved-endpoint evaluation")
        scoring._rehash_visual_images_distributed(datasets, projection_ref)
        post_checkpoint = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            step=step,
            hash_workers=args.checkpoint_hash_workers,
        )
        if post_checkpoint != checkpoint_identity:
            raise RuntimeError("Saved checkpoint changed during evaluation")
        post_projection = scoring._load_projection_distributed(raw_config, token_ids)
        _, post_audit = scoring._validate_source_audit_distributed(
            raw_config, token_ids, post_projection
        )
        if post_audit != source_audit:
            raise RuntimeError("Joint source audit changed during saved-endpoint evaluation")
        _, post_native_identity = scoring._load_native_evidence_distributed(
            raw_config, tokenizer, token_ids, experiment
        )
        if post_native_identity != native_identity:
            raise RuntimeError("Native replay evidence changed during saved-endpoint evaluation")

        pairing_sha = {
            source: pairing_metadata[source]["sha256"]
            for source in scoring.JOINT_VISUAL_SOURCE_NAMES
        }
        protocol = _protocol(
            step=step,
            examples=examples_per_source,
            pairing_sha256=pairing_sha,
            dp_world_size=dp_world_size,
            checkpoint_config=checkpoint_config,
            projection=projection_ref,
            source_audit=source_audit,
            native_identity=native_identity,
        )
        if (
            _producer_identity() != initial_producer
            or scoring._validate_clean_git_identity(scoring.bridge._git_identity()) != initial_git
        ):
            raise RuntimeError("Saved evaluator implementation or git changed during scoring")
        receipt = _receipt_payload(
            step=step,
            checkpoint=checkpoint_identity,
            checkpoint_config=checkpoint_config,
            load_coverage=load_coverage,
            projection=projection_ref,
            source_audit=source_audit,
            tokenizer=tokenizer_ref,
            pairing_manifest=pairing_manifest_ref,
            protocol=protocol,
            visual_results=visual_results,
            blank_results=blank_results,
            native_result=native_result,
            producer=initial_producer,
            git=initial_git,
        )
        _write_receipt_distributed(
            output,
            receipt,
            work_dir=work_dir,
            step=step,
        )
    finally:
        scoring._teardown_runtime(runtime_state)


if __name__ == "__main__":
    main()
