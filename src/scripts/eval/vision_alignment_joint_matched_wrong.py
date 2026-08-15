"""Build immutable, descriptive matched/wrong evidence for joint checkpoints.

This evaluator is deliberately joint-specific.  It imports the historical bridge and
perception evaluators without changing their receipt-bound bytes, then adds the exact joint
projection, source-audit, native-holdout, and permanent-checkpoint contracts needed to compare
``vision-alignment-joint-v1`` step4000 with step8000.

Pairing preparation is a single-rank CPU operation.  Model evaluation requires one eight-GPU
node (EP8), rank batch one, and 8,192-token examples.  The primary statistic is whole-response
CE only: response logits are never materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.data.multimodal import MultimodalCollator, MultimodalDataLoader
from olmo_core.data.multimodal.native_text_replay import (
    NativeTextReplayDatasetConfig,
    NativeTextReplayVerificationReceipt,
)
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JointVisualProjectionManifest,
    build_selected_joint_dataset,
    joint_alignment_runtime_registry_sha256,
    load_joint_visual_projection_manifest,
    validate_joint_live_example,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
    vision_alignment_joint_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
    serialized_example_sha256,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import (
    MultimodalFixedValidationDataset,
    MultimodalMatchedWrongImageDataset,
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    serialize_matched_wrong_image_pairing,
    validate_matched_wrong_image_pairing,
)
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device

log = logging.getLogger(__name__)


WORLD_SIZE = 8
LOCAL_WORLD_SIZE = 8
EP_DEGREE = 8
SEQUENCE_LENGTH = 8192
RANK_BATCH_INSTANCES = 1
GLOBAL_BATCH_INSTANCES = WORLD_SIZE * RANK_BATCH_INSTANCES
TRAINING_WORLD_SIZE = 16
NATIVE_HOLDOUT_EXAMPLES = 1000
PAIRING_SEED = 6198
DISTRIBUTED_TIMEOUT = timedelta(minutes=60)

RECEIPT_FORMAT = "vision_alignment_joint_matched_wrong_receipt"
RECEIPT_VERSION = 1
PAIRING_MANIFEST_FORMAT = "vision_alignment_joint_matched_wrong_pairing_manifest"
PAIRING_MANIFEST_VERSION = 1
PROTOCOL_NAME = "vision-alignment-joint-native-matched-wrong-v1"
EXPECTED_LINEAGE = "vision-alignment-joint-v1"
EXPECTED_STEPS = (4000, 8000)
EXPECTED_CONFIG_SHA256 = "64b302865831b5aaf11e86e142a85b3467a06b93d6c214fb67f7f94a45c4ddc8"
EXPECTED_PROJECTION_SHA256 = "11c1df56d7fbc270a9eff999193476c0c578c6964017d217a320b3d39305a730"
EXPECTED_SOURCE_AUDIT_FINGERPRINT = (
    "434ea76205bca8361f3291d90665af1cd36713ef238b18b1c450ff33ceab4b14"
)
EXPECTED_REVIEWED_PROFILE = "configs/vision_moe/vision_alignment/joint/joint_v1.yaml"
EXPECTED_REVIEWED_PROFILE_SHA256 = (
    "294da420f4f911fc96aad2a9eff43c59dc0831276fad5d1c0fbec37c6f78c2f5"
)
EXPECTED_BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
BLANK_SOURCE_NAMES = ("pixmo_caption", "pixmo_transcript")

_PAIRING_REQUIRED_MODEL_FIELDS = (
    "input_ids",
    "labels",
    "loss_masks",
    "position_ids",
    "token_type_ids",
    "images",
    "pooled_patches_idx",
)
_PAIRING_OPTIONAL_MODEL_FIELDS = ("subsegment_ids",)
_PAIRING_IGNORED_FIELDS = frozenset({"metadata"})


def _load_local_module(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load immutable helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


perception = _load_local_module(
    "_vision_alignment_perception_matched_wrong_for_joint",
    "vision_alignment_perception_matched_wrong.py",
)
bridge = perception.bridge


def _load_training_contract() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "train" / "Vision-Alignment.py"
    name = "_vision_alignment_training_contract_for_joint_eval"
    cached = sys.modules.get(name)
    if isinstance(cached, ModuleType):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load joint training contract {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_comparator_contract() -> ModuleType:
    path = Path(__file__).resolve().with_name("vision_alignment_joint_matched_wrong_compare.py")
    name = "_vision_alignment_joint_matched_wrong_comparator_for_prepublication"
    cached = sys.modules.get(name)
    if isinstance(cached, ModuleType):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load joint receipt comparator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Defaults to CHECKPOINT/config.json.")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--pairing-dir", required=True)
    parser.add_argument("--pairing-seed", type=int)
    parser.add_argument("--pairing-only", action="store_true")
    parser.add_argument("--pairing-manifest-output")
    parser.add_argument("--pairing-manifest")
    parser.add_argument("--expected-pairing-manifest-sha256")
    parser.add_argument("--output")
    parser.add_argument("--work-dir")
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    return parser


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


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


def _stable_file_sha256(path: Path, *, name: str) -> str:
    """Hash one regular file through the same descriptor used for stability checks."""
    path = perception._direct_existing_path(path, name=name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not a regular file")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if signature(before) != signature(after) or signature(before) != signature(current):
        raise ValueError(f"{name} changed while it was hashed")
    return digest.hexdigest()


def _project_model_input_example(example: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(example, Mapping):
        raise TypeError(f"Joint validation row {index} is not a mapping")
    fields = set(example)
    missing = sorted(set(_PAIRING_REQUIRED_MODEL_FIELDS) - fields)
    supported = (
        set(_PAIRING_REQUIRED_MODEL_FIELDS)
        | set(_PAIRING_OPTIONAL_MODEL_FIELDS)
        | _PAIRING_IGNORED_FIELDS
    )
    if missing or fields - supported:
        raise ValueError(
            f"Joint validation row {index} model fields differ: "
            f"missing={missing}, extra={sorted(fields - supported, key=str)}"
        )
    names = list(_PAIRING_REQUIRED_MODEL_FIELDS)
    names.extend(name for name in _PAIRING_OPTIONAL_MODEL_FIELDS if name in example)
    return {name: example[name] for name in names}


class _PairingModelInputDataset:
    """Expose only collator-consumed fields while retaining live image validation."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        get = getattr(self.dataset, "get", None)
        example = get(index, epoch) if callable(get) else self.dataset[index]
        return _project_model_input_example(example, index=index)

    def validate_image_content(self, indices: Sequence[int] | None = None) -> str:
        validate = getattr(self.dataset, "validate_image_content", None)
        if not callable(validate):
            raise ValueError("Joint validation dataset lacks image-content validation")
        return str(validate(indices))


class _ValidatedJointDataset:
    """Validate every joint row at the point where it is materialized."""

    def __init__(self, dataset: Any, *, source_name: str, source_kind: str, token_ids: Any):
        self.dataset = dataset
        self.source_name = source_name
        self.source_kind = source_kind
        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Mapping[str, Any]:
        get = getattr(self.dataset, "get", None)
        example = get(index, epoch) if callable(get) else self.dataset[index]
        validate_joint_live_example(
            example,
            source_name=self.source_name,
            source_kind=self.source_kind,
            token_ids=self.token_ids,
        )
        return example

    def validate_image_content(self, indices: Sequence[int] | None = None) -> str:
        validate = getattr(self.dataset, "validate_image_content", None)
        if not callable(validate):
            raise ValueError(f"{self.source_name} lacks image-content validation")
        return str(validate(indices))

    def provenance_for(self, index: int) -> Mapping[str, Any]:
        provenance = getattr(self.dataset, "provenance_for", None)
        if not callable(provenance):
            raise ValueError(f"{self.source_name} lacks row provenance")
        value = provenance(index)
        if not isinstance(value, Mapping):
            raise TypeError(f"{self.source_name} row provenance is not an object")
        return value


def _validate_args(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    expected_world_size = 1 if args.pairing_only else WORLD_SIZE
    if world_size != expected_world_size:
        purpose = "CPU pairing preparation" if args.pairing_only else "one-node EP8 evaluation"
        raise ValueError(f"Joint {purpose} requires WORLD_SIZE={expected_world_size}")
    if not args.pairing_only and local_world_size != LOCAL_WORLD_SIZE:
        raise ValueError("Joint evaluation requires LOCAL_WORLD_SIZE=8 on one node")
    if args.expected_config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError("Config pin is not the reviewed joint step4000/step8000 config")
    if args.examples != 512:
        raise ValueError("Joint pairing preparation freezes --examples=512 as the upper bound")
    for value, name in (
        (args.checkpoint_load_threads, "--checkpoint-load-threads"),
        (args.checkpoint_hash_workers, "--checkpoint-hash-workers"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.pairing_seed not in (None, PAIRING_SEED):
        raise ValueError("Joint pairing seed is frozen to the config evaluation seed 6198")
    if args.pairing_only:
        if (
            not args.pairing_manifest_output
            or args.output
            or args.pairing_manifest
            or args.expected_pairing_manifest_sha256
        ):
            raise ValueError(
                "--pairing-only requires --pairing-manifest-output and forbids evaluation "
                "manifest/output options"
            )
    elif (
        not args.output
        or not args.work_dir
        or not args.pairing_manifest
        or not _is_sha256(args.expected_pairing_manifest_sha256)
        or args.pairing_manifest_output
    ):
        raise ValueError(
            "Evaluation requires --output, an explicit --work-dir, and an exactly SHA-pinned "
            "--pairing-manifest"
        )


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(
    path: Path, *, expected_sha256: str | None = None, name: str
) -> tuple[Any, str]:
    """Read, hash, and strictly decode one regular non-symlink JSON snapshot."""
    path = perception._direct_existing_path(path, name=name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"{name} SHA-256 differs: expected {expected_sha256}, got {digest}")
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {name} from {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    current = path.lstat()
    if signature(before) != signature(after) or signature(before) != signature(current):
        raise ValueError(f"{name} changed while it was read")
    return payload, digest


def _checkpoint_root(checkpoint: Path) -> Path:
    return perception._direct_existing_path(
        bridge._checkpoint_root(checkpoint), name="joint checkpoint root"
    )


def _checkpoint_config_identity(
    checkpoint: Path, config_path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Any], int]:
    root = _checkpoint_root(checkpoint)
    if config_path != root / "config.json":
        raise ValueError("Joint evaluation config must be CHECKPOINT/config.json")
    raw_config, digest = _load_json_bytes(
        config_path,
        expected_sha256=expected_sha256,
        name="joint checkpoint config",
    )
    if not isinstance(raw_config, Mapping):
        raise TypeError("Joint checkpoint config must be an object")
    match = re.fullmatch(r"step(4000|8000)", root.name)
    if match is None:
        raise ValueError("Only permanent joint step4000 and step8000 are admissible")
    step = int(match.group(1))
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
        raise ValueError("Joint checkpoint config lacks required sections")
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
        or metadata.get("lineage_id") != EXPECTED_LINEAGE
        or raw_config.get("required_run_name") != EXPECTED_LINEAGE
        or raw_config.get("reviewed_profile_path") != EXPECTED_REVIEWED_PROFILE
        or raw_config.get("reviewed_profile_sha256") != EXPECTED_REVIEWED_PROFILE_SHA256
        or raw_config.get("reviewed_profile_allowlist_path")
        != "configs/vision_moe/vision_alignment/joint/approved_profiles.json"
        or data.get("sequence_length") != SEQUENCE_LENGTH
        or data.get("joint_visual_projection_sha256") != EXPECTED_PROJECTION_SHA256
        or data.get("source_audit_fingerprint") != EXPECTED_SOURCE_AUDIT_FINGERPRINT
        or evaluation.get("examples_per_source") != 512
        or evaluation.get("rank_batch_instances") != RANK_BATCH_INSTANCES
        or evaluation.get("seed") != PAIRING_SEED
        or raw_config.get("perception_trainability_arm") != "treatment"
        or not isinstance(ep_config, Mapping)
        or ep_config.get("degree") != EP_DEGREE
        or train_module.get("rank_microbatch_size") != SEQUENCE_LENGTH
        or train_module.get("max_sequence_length") != SEQUENCE_LENGTH
        or launch.get("workspace") != "ai2/molmofication"
        or launch.get("beaker_image") != EXPECTED_BEAKER_IMAGE
        or not isinstance(launch_git, Mapping)
        or launch_git.get("ref") != "7e42a7e3064bd944806a5cf5d351ec4f6dc24e42"
        or trainer.get("save_folder") != str(root.parent)
        or not isinstance(checkpointer, Mapping)
        or checkpointer.get("save_interval") != 4000
        or checkpointer.get("save_async") is not False
    ):
        raise ValueError("Checkpoint config differs from the reviewed joint-v1 contract")
    profile = Path(EXPECTED_REVIEWED_PROFILE).resolve()
    allowlist = Path(str(raw_config["reviewed_profile_allowlist_path"])).resolve()
    if (
        _stable_file_sha256(profile, name="reviewed joint profile")
        != EXPECTED_REVIEWED_PROFILE_SHA256
        or _stable_file_sha256(allowlist, name="reviewed joint profile allowlist")
        != raw_config["reviewed_profile_allowlist_sha256"]
    ):
        raise ValueError("Live reviewed profile or allowlist bytes differ")
    identity = {
        "path": str(config_path),
        "sha256": digest,
        "phase": "joint",
        "lineage_id": EXPECTED_LINEAGE,
        "run_name": EXPECTED_LINEAGE,
        "step": step,
        "reviewed_profile_path": EXPECTED_REVIEWED_PROFILE,
        "reviewed_profile_sha256": EXPECTED_REVIEWED_PROFILE_SHA256,
        "reviewed_profile_allowlist_path": raw_config["reviewed_profile_allowlist_path"],
        "reviewed_profile_allowlist_sha256": raw_config["reviewed_profile_allowlist_sha256"],
        "training_git_ref": launch_git["ref"],
        "training_beaker_image": launch["beaker_image"],
    }
    return raw_config, identity, step


def _initialize_runtime(*, pairing_only: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "prepared_training_environment": False,
        "created_pairing_process_group": False,
        "rendezvous": None,
    }
    if pairing_only:
        if not dist.is_initialized():
            import tempfile

            rendezvous = tempfile.TemporaryDirectory(prefix="joint-pairing-gloo-")
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{Path(rendezvous.name) / 'rendezvous'}",
                rank=0,
                world_size=1,
            )
            state["rendezvous"] = rendezvous
            state["created_pairing_process_group"] = True
        elif dist.get_world_size() != 1 or dist.get_backend() != "gloo":
            raise ValueError("Pairing-only mode requires a single-rank Gloo group")
    else:
        prepare_training_environment(timeout=DISTRIBUTED_TIMEOUT)
        state["prepared_training_environment"] = True
    return state


def _teardown_runtime(state: Mapping[str, Any]) -> None:
    if state["prepared_training_environment"]:
        teardown_training_environment()
    elif state["created_pairing_process_group"]:
        dist.destroy_process_group()
    rendezvous = state["rendezvous"]
    if rendezvous is not None:
        rendezvous.cleanup()


def _load_projection_distributed(
    raw_config: Mapping[str, Any], token_ids: Any
) -> JointVisualProjectionManifest:
    data = raw_config["data"]
    path = data.get("joint_visual_projection_path")
    expected = data.get("joint_visual_projection_sha256")
    if not isinstance(path, str) or expected != EXPECTED_PROJECTION_SHA256:
        raise ValueError("Checkpoint lacks the reviewed joint visual projection")
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            manifest = load_joint_visual_projection_manifest(
                path,
                expected_token_ids=token_ids,
                expected_sha256=expected,
                verify_finevision_materialization=False,
                load_image_path_signatures=False,
            )
            packet[0] = {
                "ok": True,
                "raw_sha256": manifest.raw_sha256,
                "content_sha256": manifest.content_sha256,
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Joint projection validation failed: {detail}")
    manifest = load_joint_visual_projection_manifest(
        path,
        expected_token_ids=token_ids,
        expected_sha256=expected,
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    if (
        manifest.raw_sha256 != result["raw_sha256"]
        or manifest.content_sha256 != result["content_sha256"]
    ):
        raise RuntimeError("Rank-local joint projection identity differs")
    return manifest


def _build_visual_datasets(
    manifest: JointVisualProjectionManifest,
    tokenizer: Any,
    token_ids: Any,
) -> dict[str, _PairingModelInputDataset]:
    datasets: dict[str, _PairingModelInputDataset] = {}
    for source in JOINT_VISUAL_SOURCE_NAMES:
        raw_dataset = build_selected_joint_dataset(
            manifest,
            tokenizer,
            token_ids,
            source,
            logical_split="validation",
            validate_required_annotations=True,
        )
        validated = _ValidatedJointDataset(
            raw_dataset,
            source_name=source,
            source_kind="visual",
            token_ids=token_ids,
        )
        datasets[source] = _PairingModelInputDataset(validated)
    if any(len(dataset) != 512 for dataset in datasets.values()):
        raise ValueError("Joint projection must expose exactly 512 validation rows per source")
    return datasets


def _validate_visual_population_distributed(
    datasets: Mapping[str, _PairingModelInputDataset],
    manifest: JointVisualProjectionManifest,
) -> dict[str, Any]:
    """Materialize and validate all 4,096 rows on rank zero and bind both image passes."""
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            pre_images = {
                source: datasets[source].validate_image_content()
                for source in JOINT_VISUAL_SOURCE_NAMES
            }
            row_hashes: dict[str, list[str]] = {}
            for source in JOINT_VISUAL_SOURCE_NAMES:
                row_hashes[source] = [
                    serialized_example_sha256(datasets[source].get(index, 0))
                    for index in range(512)
                ]
            post_images = {
                source: datasets[source].validate_image_content()
                for source in JOINT_VISUAL_SOURCE_NAMES
            }
            if pre_images != post_images:
                raise RuntimeError("Joint validation image bytes changed during row validation")
            unique_by_source = {
                source: set(manifest.selection(source, "validation").unique_image_content_sha256)
                for source in JOINT_VISUAL_SOURCE_NAMES
            }
            union = set().union(*unique_by_source.values())
            total_rows = sum(len(dataset) for dataset in datasets.values())
            if total_rows != 4096 or len(union) != 3584 or total_rows - len(union) != 512:
                raise ValueError(
                    "Joint validation union must bind 4096 rows, 3584 unique images, and "
                    "512 cross-source duplicate rows"
                )
            sources = {}
            for source in JOINT_VISUAL_SOURCE_NAMES:
                selection = manifest.selection(source, "validation")
                sources[source] = {
                    "examples": 512,
                    "runtime_dataset_fingerprint": selection.runtime_dataset_fingerprint,
                    "selection_indices_sha256": selection.selection_indices_sha256,
                    "row_image_content_sha256": bridge._content_ids_sha256(
                        selection.row_image_content_sha256
                    ),
                    "unique_image_content_count": len(selection.unique_image_content_sha256),
                    "live_image_validation_sha256": pre_images[source],
                    "live_serialized_rows_sha256": _canonical_sha256(row_hashes[source]),
                }
            packet[0] = {
                "ok": True,
                "identity": {
                    "path": str(manifest.path),
                    "raw_sha256": manifest.raw_sha256,
                    "content_sha256": manifest.content_sha256,
                    "source_spec_sha256": manifest.source_spec_sha256,
                    "visual_source_registry_sha256": (
                        vision_alignment_joint_source_registry_sha256()
                    ),
                    "runtime_registry_sha256": joint_alignment_runtime_registry_sha256(),
                    "validation_rows": total_rows,
                    "validation_unique_image_contents": len(union),
                    "validation_cross_source_duplicate_rows": total_rows - len(union),
                    "sources": sources,
                },
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Joint visual population validation failed: {detail}")
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Joint projection identity broadcast is malformed")
    return dict(identity)


def _rehash_visual_images_distributed(
    datasets: Mapping[str, _PairingModelInputDataset], expected: Mapping[str, Any]
) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            actual = {
                source: datasets[source].validate_image_content()
                for source in JOINT_VISUAL_SOURCE_NAMES
            }
            expected_images = {
                source: expected["sources"][source]["live_image_validation_sha256"]
                for source in JOINT_VISUAL_SOURCE_NAMES
            }
            if actual != expected_images:
                raise RuntimeError("Joint validation image bytes changed")
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Joint image rehash failed: {detail}")


def _cache_projection_for_training_contract(
    training: ModuleType,
    experiment: Any,
    token_ids: Any,
    projection: JointVisualProjectionManifest,
) -> None:
    key = (
        str(Path(experiment.data.joint_visual_projection_path).expanduser().resolve()),
        experiment.data.joint_visual_projection_sha256,
        training._canonical_sha256(asdict(token_ids)),
    )
    training._JOINT_PROJECTION_RUNTIME_CACHE[key] = projection


def _validate_source_audit_distributed(
    raw_config: Mapping[str, Any], token_ids: Any, projection: JointVisualProjectionManifest
) -> tuple[Any, dict[str, Any]]:
    """Run the training script's exact joint audit validator without rewalking image paths."""
    packet: list[Any] = [None]
    experiment = None
    if dist.get_rank() == 0:
        try:
            training = _load_training_contract()
            experiment = training.ExperimentConfig.from_dict(raw_config)
            _cache_projection_for_training_contract(training, experiment, token_ids, projection)
            audit = training._validated_source_audit(experiment)
            if not isinstance(audit, Mapping):
                raise TypeError("Joint source audit validation returned no object")
            path = Path(str(raw_config["data"]["source_audit_path"])).expanduser().resolve()
            _, raw_sha = _load_json_bytes(path, name="joint source audit")
            packet[0] = {
                "ok": True,
                "identity": {
                    "path": str(path),
                    "raw_sha256": raw_sha,
                    "fingerprint": audit["fingerprint"],
                    "source_registry_sha256": audit["source_registry_sha256"],
                    "runtime_registry_sha256": joint_alignment_runtime_registry_sha256(),
                    "status": audit["status"],
                    "phase": audit["phase"],
                },
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Joint source-audit validation failed: {detail}")
    if dist.get_rank() != 0:
        training = _load_training_contract()
        experiment = training.ExperimentConfig.from_dict(raw_config)
        _cache_projection_for_training_contract(training, experiment, token_ids, projection)
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Joint source-audit identity broadcast is malformed")
    return experiment, dict(identity)


def _native_config(raw: Mapping[str, Any], *, verify_source_hashes: bool) -> Any:
    values = {key: value for key, value in raw.items() if key != "_CLASS_"}
    values["verify_source_hashes"] = verify_source_hashes
    return NativeTextReplayDatasetConfig(**values)


def _native_source_inventory(manifest: Any) -> list[dict[str, Any]]:
    return [
        {
            "source_id": source.source_id,
            "path": str(source.resolved_path),
            "size": source.size_bytes,
            "sha256": source.sha256,
        }
        for source in manifest.sources
    ]


def _load_native_evidence_distributed(
    raw_config: Mapping[str, Any], tokenizer: Any, token_ids: Any, experiment: Any
) -> tuple[_ValidatedJointDataset, dict[str, Any]]:
    packet: list[Any] = [None]
    rank_zero_holdout = None
    if dist.get_rank() == 0:
        try:
            training = _load_training_contract()
            if experiment is None:
                experiment = training.ExperimentConfig.from_dict(raw_config)
            training._validate_native_replay_pair(experiment)
            train_cfg = _native_config(
                raw_config["data"]["native_text_replay"], verify_source_hashes=True
            )
            holdout_cfg = _native_config(
                raw_config["evaluation"]["native_text_holdout"], verify_source_hashes=True
            )
            train_dataset = train_cfg.build(tokenizer)
            rank_zero_holdout = holdout_cfg.build(tokenizer)
            receipt_path = Path(str(holdout_cfg.verification_receipt_path)).expanduser().resolve()
            receipt = NativeTextReplayVerificationReceipt.load(
                receipt_path,
                expected_sha256=holdout_cfg.expected_verification_receipt_sha256,
            )
            receipt.validate_pair(train_dataset.manifest, rank_zero_holdout.manifest)
            if len(rank_zero_holdout) != NATIVE_HOLDOUT_EXAMPLES:
                raise ValueError("Joint native holdout must contain exactly 1000 windows")
            validated = _ValidatedJointDataset(
                rank_zero_holdout,
                source_name="native_text_replay",
                source_kind="native_text_replay",
                token_ids=token_ids,
            )
            row_hashes = [
                serialized_example_sha256(validated.get(index, 0))
                for index in range(NATIVE_HOLDOUT_EXAMPLES)
            ]
            provenance = [
                dict(validated.provenance_for(index)) for index in range(NATIVE_HOLDOUT_EXAMPLES)
            ]
            train_inventory = _native_source_inventory(train_dataset.manifest)
            holdout_inventory = _native_source_inventory(rank_zero_holdout.manifest)
            packet[0] = {
                "ok": True,
                "identity": {
                    "train_manifest_path": str(train_dataset.manifest.path),
                    "train_manifest_sha256": _sha256_file(train_dataset.manifest.path),
                    "train_fingerprint": train_dataset.manifest.content_fingerprint,
                    "train_source_count": len(train_inventory),
                    "train_source_inventory_sha256": _canonical_sha256(train_inventory),
                    "holdout_manifest_path": str(rank_zero_holdout.manifest.path),
                    "holdout_manifest_sha256": _sha256_file(rank_zero_holdout.manifest.path),
                    "holdout_fingerprint": rank_zero_holdout.manifest.content_fingerprint,
                    "holdout_source_count": len(holdout_inventory),
                    "holdout_source_inventory_sha256": _canonical_sha256(holdout_inventory),
                    "verification_receipt_path": str(receipt_path),
                    "verification_receipt_sha256": holdout_cfg.expected_verification_receipt_sha256,
                    "full_source_hash_verification": True,
                    "train_holdout_pair_validated": True,
                    "examples": NATIVE_HOLDOUT_EXAMPLES,
                    "sequence_length": SEQUENCE_LENGTH,
                    "manifest_order_sha256": _canonical_sha256(
                        list(range(NATIVE_HOLDOUT_EXAMPLES))
                    ),
                    "row_provenance_sha256": _canonical_sha256(provenance),
                    "live_serialized_rows_sha256": _canonical_sha256(row_hashes),
                },
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Joint native replay validation failed: {detail}")
    if rank_zero_holdout is None:
        holdout_cfg = _native_config(
            raw_config["evaluation"]["native_text_holdout"], verify_source_hashes=False
        )
        rank_zero_holdout = holdout_cfg.build(tokenizer)
    dataset = _ValidatedJointDataset(
        rank_zero_holdout,
        source_name="native_text_replay",
        source_kind="native_text_replay",
        token_ids=token_ids,
    )
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Native replay identity broadcast is malformed")
    return dataset, dict(identity)


def _pairing_paths(pairing_dir: Path) -> dict[str, Path]:
    root = _artifact_path(pairing_dir, name="pairing directory")
    return {source: root / f"{source}.json" for source in JOINT_VISUAL_SOURCE_NAMES}


def _artifact_path(path: Path, *, name: str) -> Path:
    """Make a lexical absolute artifact path and reject symlinks in existing components."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (*reversed(absolute.parents), absolute):
        if component == Path(component.anchor) or not component.exists():
            continue
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} contains a symlinked component: {component}")
    return absolute


def _content_ids(
    projection: JointVisualProjectionManifest,
) -> dict[str, tuple[str, ...]]:
    return {
        source: projection.selection(source, "validation").row_image_content_sha256
        for source in JOINT_VISUAL_SOURCE_NAMES
    }


def _build_largest_common_pairings(
    datasets: Mapping[str, Any],
    *,
    content_ids: Mapping[str, Sequence[str]],
) -> tuple[int, dict[str, Mapping[str, Any]]]:
    """Deterministically derive the largest common eligible multiple of eight at most 512."""
    candidate = 512
    while candidate >= GLOBAL_BATCH_INSTANCES:
        payloads: dict[str, Mapping[str, Any]] = {}
        limit: int | None = None
        for source in JOINT_VISUAL_SOURCE_NAMES:
            try:
                payloads[source] = build_matched_wrong_image_pairing(
                    datasets[source],
                    recipient_count=candidate,
                    seed=PAIRING_SEED,
                    content_ids=content_ids[source],
                    epoch=0,
                )
            except Exception as error:
                match = re.search(r"requested \d+, found (\d+) across", str(error))
                if match is None:
                    raise
                available = int(match.group(1))
                aligned = available - available % GLOBAL_BATCH_INSTANCES
                limit = aligned if limit is None else min(limit, aligned)
        if limit is None:
            if set(payloads) != set(JOINT_VISUAL_SOURCE_NAMES):
                raise RuntimeError("Pairing construction omitted a joint source")
            return candidate, payloads
        if limit <= 0 or limit >= candidate:
            raise ValueError("Could not derive a smaller positive common pairing count")
        candidate = limit
    raise ValueError("No common batch-aligned matched/wrong population is available")


def _pairing_metadata(payload: Mapping[str, Any], *, path: Path, digest: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": digest,
        "canonical_sha256": matched_wrong_image_pairing_sha256(payload),
        "pairing_schema_version": payload["version"],
        "population": "matched_eligible_joint_validation_subset",
        "coverage": payload["coverage"],
        "recipient_indices_sha256": _canonical_sha256(
            [int(pair["recipient"]) for pair in payload["pairs"]]
        ),
        "donor_indices_sha256": _canonical_sha256(
            [int(pair["donor"]) for pair in payload["pairs"]]
        ),
    }


def _validate_pairing_payload(
    payload: Mapping[str, Any],
    *,
    dataset: Any,
    examples: int,
    content_ids: Sequence[str],
) -> str:
    validate_matched_wrong_image_pairing(
        payload,
        dataset_size=len(dataset),
        recipient_count=examples,
        seed=PAIRING_SEED,
        epoch=0,
        content_ids_sha256=bridge._content_ids_sha256(content_ids),
    )
    raw = serialize_matched_wrong_image_pairing(payload)
    canonical = matched_wrong_image_pairing_sha256(payload)
    if hashlib.sha256(raw).hexdigest() != canonical:
        raise ValueError("Pairing serializer and canonical identity differ")
    return canonical


def _json_output_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    """Publish exact bytes with an exclusive, no-follow temporary and no replacement."""
    path = _artifact_path(path, name="immutable output")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _artifact_path(path.parent, name="immutable output directory")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(parent, directory_flags)
    temporary_name = f".{path.name}.{os.getpid()}.tmp"
    temporary_fd = -1
    temporary_identity: tuple[int, int, int] | None = None
    temporary_signature: tuple[int, int, int, int, int, int] | None = None
    created = False
    try:
        directory_before = os.fstat(directory_fd)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        temporary_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        created = True
        temporary_before = os.fstat(temporary_fd)
        if not stat.S_ISREG(temporary_before.st_mode):
            raise ValueError("Immutable output temporary is not a regular file")
        temporary_identity = _stat_signature(temporary_before)[:3]
        view = memoryview(payload)
        while view:
            view = view[os.write(temporary_fd, view) :]
        os.fsync(temporary_fd)
        temporary_after = os.fstat(temporary_fd)
        temporary_signature = _stat_signature(temporary_after)
        if _stat_signature(temporary_before)[:3] != temporary_signature[
            :3
        ] or temporary_after.st_size != len(payload):
            raise RuntimeError("Immutable output temporary changed while written")
        named_temporary = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_signature(named_temporary) != temporary_signature:
            raise RuntimeError("Immutable output temporary path was replaced")
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable output {path}") from error
        temporary_signature = _stat_signature(os.fstat(temporary_fd))
        destination = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_signature(destination) != temporary_signature:
            raise RuntimeError("Immutable output hard link differs from its exact temporary")
        os.fsync(directory_fd)
        directory_after = os.fstat(directory_fd)
        directory_current = parent.lstat()
        if (
            _stat_signature(directory_before)[:3] != _stat_signature(directory_after)[:3]
            or _stat_signature(directory_before)[:3] != _stat_signature(directory_current)[:3]
        ):
            raise RuntimeError("Immutable output directory changed during publication")
    finally:
        if created:
            named: os.stat_result | None
            try:
                named = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                named = None
            if (
                named is not None
                and temporary_identity is not None
                and (_stat_signature(named)[:3] == temporary_identity)
            ):
                os.unlink(temporary_name, dir_fd=directory_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        os.close(directory_fd)
    digest = hashlib.sha256(payload).hexdigest()
    if _stable_file_sha256(path, name="published immutable output") != digest:
        raise RuntimeError("Published immutable output bytes differ")
    return digest


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish once, fsync its directory, and strictly reload the exact bytes."""
    digest = _write_bytes_exclusive(path, _json_output_bytes(payload))
    loaded, reloaded_digest = _load_json_bytes(
        path, expected_sha256=digest, name=f"persisted {path.name}"
    )
    if _canonical_bytes(loaded) != _canonical_bytes(payload):
        raise RuntimeError(f"Persisted artifact {path} differs from its payload")
    if reloaded_digest != digest:
        raise RuntimeError(f"Persisted artifact {path} digest changed")
    return digest


def _prepare_pairings_distributed(
    datasets: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    content_ids: Mapping[str, Sequence[str]],
) -> tuple[int, dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            examples, payloads = _build_largest_common_pairings(datasets, content_ids=content_ids)
            rebuilt_examples, rebuilt = _build_largest_common_pairings(
                datasets, content_ids=content_ids
            )
            if rebuilt_examples != examples or _canonical_bytes(rebuilt) != _canonical_bytes(
                payloads
            ):
                raise RuntimeError("Deterministic pairing reconstruction differed")
            metadata: dict[str, dict[str, Any]] = {}
            for source in JOINT_VISUAL_SOURCE_NAMES:
                digest = _validate_pairing_payload(
                    payloads[source],
                    dataset=datasets[source],
                    examples=examples,
                    content_ids=content_ids[source],
                )
                raw = serialize_matched_wrong_image_pairing(payloads[source])
                if paths[source].exists():
                    existing, existing_sha = _load_json_bytes(
                        paths[source],
                        expected_sha256=digest,
                        name=f"existing partial {source} pairing",
                    )
                    if _canonical_bytes(existing) != _canonical_bytes(payloads[source]):
                        raise ValueError(f"Existing partial {source} pairing differs")
                    if existing_sha != digest:
                        raise ValueError(f"Existing partial {source} pairing SHA differs")
                else:
                    _write_bytes_exclusive(paths[source], raw)
                persisted, raw_digest = _load_json_bytes(
                    paths[source], expected_sha256=digest, name=f"persisted {source} pairing"
                )
                if _canonical_bytes(persisted) != _canonical_bytes(payloads[source]):
                    raise RuntimeError(f"Persisted {source} pairing differs")
                metadata[source] = _pairing_metadata(
                    payloads[source], path=paths[source], digest=raw_digest
                )
            directory_fd = os.open(
                next(iter(paths.values())).parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            packet[0] = {
                "ok": True,
                "examples": examples,
                "payloads": payloads,
                "metadata": metadata,
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not prepare joint pairings: {detail}")
    return (
        int(result["examples"]),
        dict(result["payloads"]),
        {source: dict(result["metadata"][source]) for source in JOINT_VISUAL_SOURCE_NAMES},
    )


def _pairing_manifest_payload(
    *,
    checkpoint_config: Mapping[str, Any],
    projection: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    examples: int,
    pairings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    producer = Path(__file__).resolve()
    bridge_path = Path(str(bridge.__file__)).resolve()
    pairing_source = inspect.getsourcefile(build_matched_wrong_image_pairing)
    if pairing_source is None:
        raise RuntimeError("Could not locate pairing implementation")
    payload: dict[str, Any] = {
        "format": PAIRING_MANIFEST_FORMAT,
        "version": PAIRING_MANIFEST_VERSION,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": {
            "path": str(producer),
            "sha256": _stable_file_sha256(producer, name="joint evaluator"),
        },
        "bridge_helper": {
            "path": str(bridge_path),
            "sha256": _stable_file_sha256(bridge_path, name="bridge evaluator"),
        },
        "pairing_implementation": {
            "path": str(Path(pairing_source).resolve()),
            "sha256": _stable_file_sha256(
                Path(pairing_source).resolve(), name="pairing implementation"
            ),
        },
        "checkpoint_config": dict(checkpoint_config),
        "projection": dict(projection),
        "source_audit": dict(source_audit),
        "tokenizer": dict(tokenizer),
        "protocol": {
            "selection": "largest-common-matched-eligible-multiple-of-eight-at-most-512-v1",
            "maximum_requested_examples": 512,
            "examples_per_source": examples,
            "global_batch_instances": GLOBAL_BATCH_INSTANCES,
            "pairing_seed": PAIRING_SEED,
            "sources": list(JOINT_VISUAL_SOURCE_NAMES),
            "population": "matched_eligible_joint_validation_subset",
            "sequence_length": SEQUENCE_LENGTH,
            "source_registry_sha256": vision_alignment_joint_source_registry_sha256(),
        },
        "pairings": {source: dict(pairings[source]) for source in JOINT_VISUAL_SOURCE_NAMES},
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _write_pairing_manifest_distributed(output: Path, payload: Mapping[str, Any]) -> str:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {"ok": True, "sha256": _write_json_exclusive(output, payload)}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not persist joint pairing manifest: {detail}")
    return str(result["sha256"])


_PAIRING_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "bridge_helper",
        "pairing_implementation",
        "checkpoint_config",
        "projection",
        "source_audit",
        "tokenizer",
        "protocol",
        "pairings",
        "content_sha256",
    }
)


def _load_pairing_manifest_distributed(
    path: Path,
    expected_sha256: str,
    *,
    datasets: Mapping[str, Any],
    content_ids: Mapping[str, Sequence[str]],
    pairing_dir: Path,
    config_sha256: str,
    checkpoint_config: Mapping[str, Any],
    projection: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            manifest, raw_sha = _load_json_bytes(
                path, expected_sha256=expected_sha256, name="joint pairing manifest"
            )
            if not isinstance(manifest, Mapping) or set(manifest) != _PAIRING_MANIFEST_FIELDS:
                raise ValueError("Joint pairing manifest fields differ")
            unsigned = dict(manifest)
            content_sha = unsigned.pop("content_sha256", None)
            created_at = manifest.get("created_at")
            if not isinstance(created_at, str):
                raise ValueError("Pairing manifest created_at is not a timestamp")
            try:
                parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("Pairing manifest created_at is invalid") from error
            if parsed_created_at.tzinfo is None:
                raise ValueError("Pairing manifest created_at lacks a timezone")
            protocol = manifest.get("protocol")
            pair_meta = manifest.get("pairings")
            pairing_source = inspect.getsourcefile(build_matched_wrong_image_pairing)
            if pairing_source is None:
                raise RuntimeError("Could not locate live pairing implementation")
            implementation_refs = {
                "producer": Path(__file__).resolve(),
                "bridge_helper": Path(str(bridge.__file__)).resolve(),
                "pairing_implementation": Path(pairing_source).resolve(),
            }
            for field, live_path in implementation_refs.items():
                reference = manifest.get(field)
                if (
                    not isinstance(reference, Mapping)
                    or set(reference) != {"path", "sha256"}
                    or str(reference.get("path")) != str(live_path)
                    or reference.get("sha256")
                    != _stable_file_sha256(live_path, name=f"live {field}")
                ):
                    raise ValueError(f"Pairing manifest {field} implementation differs")
            examples = (
                protocol.get("examples_per_source") if isinstance(protocol, Mapping) else None
            )
            expected_protocol = {
                "selection": "largest-common-matched-eligible-multiple-of-eight-at-most-512-v1",
                "maximum_requested_examples": 512,
                "examples_per_source": examples,
                "global_batch_instances": GLOBAL_BATCH_INSTANCES,
                "pairing_seed": PAIRING_SEED,
                "sources": list(JOINT_VISUAL_SOURCE_NAMES),
                "population": "matched_eligible_joint_validation_subset",
                "sequence_length": SEQUENCE_LENGTH,
                "source_registry_sha256": vision_alignment_joint_source_registry_sha256(),
            }
            manifest_config = manifest.get("checkpoint_config")
            shared_fields = set(checkpoint_config) - {"path", "step"}
            if (
                manifest.get("format") != PAIRING_MANIFEST_FORMAT
                or manifest.get("version") != PAIRING_MANIFEST_VERSION
                or manifest.get("status") != "prepared"
                or content_sha != _canonical_sha256(unsigned)
                or not isinstance(manifest_config, Mapping)
                or manifest_config.get("sha256") != config_sha256
                or {field: manifest_config.get(field) for field in shared_fields}
                != {field: checkpoint_config[field] for field in shared_fields}
                or set(manifest_config) != set(checkpoint_config)
                or _canonical_bytes(manifest.get("projection")) != _canonical_bytes(projection)
                or _canonical_bytes(manifest.get("source_audit")) != _canonical_bytes(source_audit)
                or _canonical_bytes(manifest.get("tokenizer")) != _canonical_bytes(tokenizer)
                or not isinstance(protocol, Mapping)
                or protocol != expected_protocol
                or not isinstance(pair_meta, Mapping)
                or set(pair_meta) != set(JOINT_VISUAL_SOURCE_NAMES)
            ):
                raise ValueError("Joint pairing manifest identity or protocol differs")
            if type(examples) is not int or examples <= 0 or examples % GLOBAL_BATCH_INSTANCES:
                raise ValueError("Pairing manifest example count is not batch-aligned")
            rebuilt_examples, rebuilt = _build_largest_common_pairings(
                datasets, content_ids=content_ids
            )
            if rebuilt_examples != examples:
                raise ValueError("Pairing manifest is not the largest common eligible population")
            root = _artifact_path(pairing_dir, name="pairing directory")
            payloads: dict[str, Mapping[str, Any]] = {}
            metadata: dict[str, dict[str, Any]] = {}
            pins: dict[str, str] = {}
            required_meta = {
                "path",
                "sha256",
                "canonical_sha256",
                "pairing_schema_version",
                "population",
                "coverage",
                "recipient_indices_sha256",
                "donor_indices_sha256",
            }
            for source in JOINT_VISUAL_SOURCE_NAMES:
                meta = pair_meta[source]
                expected_path = root / f"{source}.json"
                if (
                    not isinstance(meta, Mapping)
                    or set(meta) != required_meta
                    or str(meta.get("path")) != str(expected_path)
                    or _artifact_path(Path(str(meta.get("path"))), name=f"{source} pairing")
                    != expected_path
                    or not _is_sha256(meta.get("sha256"))
                    or meta.get("sha256") != meta.get("canonical_sha256")
                    or meta.get("population") != "matched_eligible_joint_validation_subset"
                ):
                    raise ValueError(f"{source} pairing metadata differs")
                payload, digest = _load_json_bytes(
                    expected_path,
                    expected_sha256=meta["sha256"],
                    name=f"{source} pairing",
                )
                if not isinstance(payload, Mapping):
                    raise TypeError(f"{source} pairing is not an object")
                canonical = _validate_pairing_payload(
                    payload,
                    dataset=datasets[source],
                    examples=examples,
                    content_ids=content_ids[source],
                )
                derived = _pairing_metadata(payload, path=expected_path, digest=digest)
                if derived != meta or _canonical_bytes(payload) != _canonical_bytes(
                    rebuilt[source]
                ):
                    raise ValueError(f"{source} pairing bytes or deterministic replay differ")
                payloads[source] = payload
                metadata[source] = dict(meta)
                pins[source] = canonical
            packet[0] = {
                "ok": True,
                "payloads": payloads,
                "metadata": metadata,
                "pins": pins,
                "manifest_ref": {
                    "path": str(path),
                    "sha256": raw_sha,
                    "content_sha256": content_sha,
                },
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not load joint pairing manifest: {detail}")
    return (
        dict(result["payloads"]),
        {source: dict(result["metadata"][source]) for source in JOINT_VISUAL_SOURCE_NAMES},
        dict(result["manifest_ref"]),
    )


def _read_trainer_state(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash and weights-only decode one small trainer state from the same descriptor bytes."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    raw_parts: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Trainer-state entry is not regular: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            raw_parts.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if signature(before) != signature(after) or signature(before) != signature(current):
        raise ValueError(f"Trainer-state entry changed while read: {path}")
    safe_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
    ]
    with torch.serialization.safe_globals(safe_globals):
        state = torch.load(
            io.BytesIO(b"".join(raw_parts)),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(state, dict):
        raise TypeError(f"Trainer-state entry is not a mapping: {path}")
    record = {
        "path": path.relative_to(root).as_posix(),
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }
    return record, state


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
    if (
        set(state) != required
        or state.get("global_step") != step
        or state.get("global_train_tokens_seen") != step * 1_048_576
        or state.get("max_steps") != 16000
        or state.get("world_size") != TRAINING_WORLD_SIZE
        or not isinstance(loader, Mapping)
        or loader.get("batches_processed") != step
        or loader.get("consecutive_data_errors") != 0
        or loader.get("total_data_errors") != (1 if step == 8000 and rank in (0, 8) else 0)
        or not isinstance(packing, Mapping)
        or packing.get("dp_world_size") != TRAINING_WORLD_SIZE
        or packing.get("dp_rank") != rank
        or packing.get("rank_instances") != 8
        or packing.get("seq_len") != SEQUENCE_LENGTH
        or packing.get("dataset_names") != expected_datasets
        or not isinstance(wandb, Mapping)
        or wandb.get("step") != step
        or wandb.get("name") != EXPECTED_LINEAGE
        or wandb.get("project") != "vision-alignment"
        or (rank == 0 and (not isinstance(wandb.get("run_id"), str) or not wandb["run_id"]))
        or (rank != 0 and wandb.get("run_id") is not None)
    ):
        raise ValueError(f"Trainer rank{rank} state differs from joint step{step} contract")
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


def _stable_checkpoint_record(path: Path, *, root: Path) -> dict[str, Any]:
    """Hash one DCP entry with nonblocking, no-follow, same-FD stability checks."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint entry is not a regular file: {path}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        signature(before) != signature(after)
        or signature(before) != signature(current)
        or size != before.st_size
    ):
        raise ValueError(f"Checkpoint entry changed while hashed: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _model_checkpoint_identity(
    checkpoint: Path, config_path: Path, *, hash_workers: int
) -> dict[str, Any]:
    root = _checkpoint_root(checkpoint)
    state_dir = perception._direct_existing_path(
        bridge._checkpoint_state_dir(checkpoint), name="model checkpoint state directory"
    )
    entries = sorted(state_dir.iterdir())
    if not entries or hash_workers <= 0:
        raise ValueError("Distributed checkpoint inventory or worker count is invalid")
    if any(not stat.S_ISREG(path.lstat().st_mode) for path in entries):
        raise ValueError("Distributed checkpoint contains a non-regular entry")
    with ThreadPoolExecutor(max_workers=min(hash_workers, len(entries))) as executor:
        inventory = list(
            executor.map(lambda path: _stable_checkpoint_record(path, root=root), entries)
        )
    if entries != sorted(state_dir.iterdir()):
        raise ValueError("Distributed checkpoint entries changed during inventory")
    metadata_relative = (state_dir / ".metadata").relative_to(root).as_posix()
    records = {record["path"]: record for record in inventory}
    if metadata_relative not in records:
        raise ValueError("Distributed checkpoint metadata is missing")
    config_record = _stable_checkpoint_record(config_path, root=root)
    marker_record = _stable_checkpoint_record(root / ".metadata.json", root=root)
    identity = {
        "root": str(root),
        "state_dir": str(state_dir),
        "config_sha256": config_record["sha256"],
        "checkpoint_marker_sha256": marker_record["sha256"],
        "dcp_metadata_sha256": records[metadata_relative]["sha256"],
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": _canonical_sha256(inventory),
        "state_file_inventory": inventory,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _checkpoint_identity(
    checkpoint: Path,
    config_path: Path,
    *,
    step: int,
    hash_workers: int,
) -> dict[str, Any]:
    """Bind every DCP shard, all 16 trainer states, config, and permanent marker."""
    identity = _model_checkpoint_identity(
        checkpoint,
        config_path,
        hash_workers=hash_workers,
    )
    model_identity = identity.pop("identity_sha256")
    root = Path(str(identity["root"]))
    if root.name != f"step{step}":
        raise ValueError("Checkpoint root step differs from the requested endpoint")
    marker, marker_raw_sha = _load_json_bytes(
        root / ".metadata.json", name="permanent checkpoint marker"
    )
    if marker != {"ephemeral": False, "version": "2.5.0"}:
        raise ValueError("Checkpoint marker is not the exact permanent v2.5.0 marker")
    train_dir = perception._direct_existing_path(root / "train", name="trainer-state directory")
    expected_names = [f"rank{rank}.pt" for rank in range(TRAINING_WORLD_SIZE)]
    observed_names = sorted(path.name for path in train_dir.iterdir())
    if sorted(expected_names) != observed_names:
        raise ValueError("Checkpoint must contain exactly train/rank0.pt through rank15.pt")
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for rank in range(TRAINING_WORLD_SIZE):
        record, state = _read_trainer_state(train_dir / f"rank{rank}.pt", root=root)
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
        raise ValueError("Trainer-state ranks disagree on global progress or run identity")
    identity.update(
        {
            "model_and_optim_identity_sha256": model_identity,
            "checkpoint_step": step,
            "permanent": True,
            "checkpoint_marker": dict(marker),
            "checkpoint_marker_sha256": marker_raw_sha,
            "trainer_state_rank_count": TRAINING_WORLD_SIZE,
            "trainer_state_file_inventory": records,
            "trainer_state_file_inventory_sha256": _canonical_sha256(records),
            "trainer_state_summary": {**shared[0], "wandb_run_id": leader_run_id},
            "trainer_state_total_data_errors_by_rank": [
                summary["total_data_errors"] for summary in summaries
            ],
            "trainer_state_total_data_errors_sum": sum(
                int(summary["total_data_errors"]) for summary in summaries
            ),
        }
    )
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _checkpoint_identity_distributed(
    checkpoint: Path,
    config_path: Path,
    *,
    step: int,
    hash_workers: int,
) -> dict[str, Any]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
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
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not identify permanent joint checkpoint: {detail}")
    value = result.get("identity")
    if not isinstance(value, Mapping):
        raise TypeError("Checkpoint identity broadcast is malformed")
    return dict(value)


def _copy_checkpoint_snapshot_file(
    source: Path,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    read_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(source, read_flags)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint snapshot source is not regular: {source}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target_fd = os.open(target, write_flags, 0o400)
        digest = hashlib.sha256()
        size = 0
        try:
            while chunk := os.read(source_fd, 8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(target_fd, view) :]
            os.fsync(target_fd)
            target_info = os.fstat(target_fd)
        finally:
            os.close(target_fd)
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    current = source.lstat()

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        signature(before) != signature(after)
        or signature(before) != signature(current)
        or size != expected_size
        or target_info.st_size != expected_size
        or digest.hexdigest() != expected_sha256
    ):
        raise ValueError(f"Checkpoint source changed while snapshotting: {source}")
    target.chmod(0o400)


def _materialize_checkpoint_snapshot(identity: Mapping[str, Any], *, base_dir: Path) -> Path:
    root = perception._direct_existing_path(
        Path(str(identity["root"])), name="snapshot checkpoint root"
    )
    inventory = identity.get("state_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Checkpoint snapshot inventory is empty")
    base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    base_dir = perception._direct_existing_path(base_dir, name="checkpoint snapshot base directory")
    required_bytes = sum(int(record.get("size", -1)) for record in inventory)
    filesystem = os.statvfs(base_dir)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    if required_bytes <= 0 or available_bytes < required_bytes + 1024**3:
        raise OSError(
            "Insufficient free space for the full verified checkpoint snapshot: "
            f"required={required_bytes}, available={available_bytes}"
        )
    snapshot_root = Path(tempfile.mkdtemp(prefix=".joint-checkpoint-snapshot-", dir=base_dir))
    log.info("Materializing evaluator-owned checkpoint snapshot at %s", snapshot_root)
    try:
        for record in inventory:
            if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
                raise ValueError("Checkpoint snapshot inventory record is malformed")
            relative = Path(str(record["path"]))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "model_and_optim"
                or type(record["size"]) is not int
                or record["size"] < 0
                or not _is_sha256(record["sha256"])
            ):
                raise ValueError("Checkpoint snapshot record is invalid")
            _copy_checkpoint_snapshot_file(
                root / relative,
                snapshot_root / relative,
                expected_size=record["size"],
                expected_sha256=record["sha256"],
            )
        state_dir = snapshot_root / "model_and_optim"
        observed = sorted(
            path.relative_to(snapshot_root).as_posix() for path in state_dir.iterdir()
        )
        expected = sorted(str(record["path"]) for record in inventory)
        if observed != expected:
            raise ValueError("Private checkpoint snapshot entries differ")
        state_dir.chmod(0o500)
        snapshot_root.chmod(0o500)
        return state_dir
    except Exception:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise


def _materialize_checkpoint_snapshot_distributed(
    identity: Mapping[str, Any], *, base_dir: Path
) -> Path:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {
                "ok": True,
                "state_dir": str(_materialize_checkpoint_snapshot(identity, base_dir=base_dir)),
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not materialize private joint checkpoint: {detail}")
    return Path(str(result["state_dir"]))


def _remove_checkpoint_snapshot_distributed(state_dir: Path) -> None:
    dist.barrier()
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            root = state_dir.parent
            if not root.name.startswith(".joint-checkpoint-snapshot-"):
                raise RuntimeError(f"Refusing to remove non-joint-snapshot path {root}")
            root.chmod(0o700)
            state_dir.chmod(0o700)
            shutil.rmtree(root)
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    if not isinstance(packet[0], Mapping) or packet[0].get("ok") is not True:
        raise RuntimeError(f"Could not remove private joint checkpoint snapshot: {packet[0]}")


def _loss_identity(batch: Mapping[str, Any]) -> dict[str, Any]:
    labels = batch.get("labels")
    loss_masks = batch.get("loss_masks")
    if not isinstance(labels, torch.Tensor) or not isinstance(loss_masks, torch.Tensor):
        raise TypeError("Joint evaluation batch lacks tensor labels/loss_masks")
    if labels.ndim != 2 or loss_masks.shape != labels.shape or labels.shape[0] != 1:
        raise ValueError("Joint evaluation requires one aligned example per rank")
    # Preserve both denominators. Inline validation excludes ignore-index labels, while training
    # deliberately retains repetition-filtered native rows in its mask-weight divisor.
    mask_valid = loss_masks > 0
    labeled_valid = mask_valid & (labels != -100)
    mask_tokens = int(mask_valid.sum().item())
    labeled_tokens = int(labeled_valid.sum().item())
    mask_weight = float(loss_masks.masked_select(mask_valid).float().sum().item())
    labeled_weight = float(loss_masks.masked_select(labeled_valid).float().sum().item())
    if mask_tokens <= 0 or not math.isfinite(mask_weight) or mask_weight <= 0:
        raise ValueError("Joint evaluation example has no positive finite supervised loss mass")
    if not math.isfinite(labeled_weight) or labeled_weight < 0:
        raise ValueError("Joint evaluation example has invalid labeled loss mass")
    if labeled_tokens not in (0, mask_tokens):
        raise ValueError("Joint evaluation example is only partially ignore-index filtered")
    if labeled_tokens == mask_tokens and labeled_weight != mask_weight:
        raise ValueError("Fully labeled joint example changed its loss-weight denominator")
    return {
        "mask_tokens": mask_tokens,
        "labeled_tokens": labeled_tokens,
        "mask_loss_weight": mask_weight,
        "labeled_loss_weight": labeled_weight,
        "filtered": labeled_tokens == 0,
    }


def _forward_scalar_ce(train_module: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    identity = _loss_identity(batch)
    device_batch = move_to_device(dict(batch), train_module.device)
    output = train_module.eval_batch(device_batch, return_response_logits=False)
    if (
        not isinstance(output, LMOutputWithLoss)
        or output.logits is not None
        or not isinstance(output.ce_loss, torch.Tensor)
        or output.ce_loss.numel() != 1
    ):
        raise TypeError("Joint evaluation did not return one scalar CE sum without logits")
    summed_ce = float(output.ce_loss.detach().float().cpu().item())
    labeled_weight = float(identity["labeled_loss_weight"])
    ce = summed_ce / labeled_weight if labeled_weight > 0 else None
    if (
        not math.isfinite(summed_ce)
        or summed_ce < 0
        or (ce is not None and (not math.isfinite(ce) or ce < 0))
        or (identity["filtered"] and summed_ce != 0.0)
    ):
        raise ValueError("Joint evaluation produced invalid CE")
    del device_batch, output
    return {**identity, "summed_ce": summed_ce, "ce": ce}


def _weighted_ce(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    denominator = sum(float(row["loss_weight"]) for row in rows)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Cannot aggregate zero or invalid loss weight")
    numerator = sum(float(row[field]) * float(row["loss_weight"]) for row in rows)
    value = numerator / denominator
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Aggregate {field} is invalid")
    return value


def _visual_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty visual results")
    gaps = np.asarray([float(row["ce_gap_wrong_minus_correct"]) for row in rows])
    correct = _weighted_ce(rows, "correct_ce")
    wrong = _weighted_ce(rows, "wrong_ce")
    return {
        "examples": len(rows),
        "response_tokens": sum(int(row["response_tokens"]) for row in rows),
        "loss_weight": sum(float(row["loss_weight"]) for row in rows),
        "correct_ce": correct,
        "wrong_ce": wrong,
        "weighted_ce_gap_wrong_minus_correct": wrong - correct,
        "ce_gap_wrong_minus_correct_mean": float(gaps.mean()),
        "ce_gap_wrong_minus_correct_median": float(np.median(gaps)),
        "correct_image_win_rate": float((gaps > 0).mean()),
        "tie_rate": float((gaps == 0).mean()),
    }


def _blank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty blank-image results")
    gaps = np.asarray([float(row["ce_gap_blank_minus_correct"]) for row in rows])
    correct = _weighted_ce(rows, "correct_ce")
    blank = _weighted_ce(rows, "blank_ce")
    return {
        "examples": len(rows),
        "response_tokens": sum(int(row["response_tokens"]) for row in rows),
        "loss_weight": sum(float(row["loss_weight"]) for row in rows),
        "correct_ce": correct,
        "blank_ce": blank,
        "weighted_ce_gap_blank_minus_correct": blank - correct,
        "ce_gap_blank_minus_correct_mean": float(gaps.mean()),
        "ce_gap_blank_minus_correct_median": float(np.median(gaps)),
        "correct_image_win_rate": float((gaps > 0).mean()),
        "tie_rate": float((gaps == 0).mean()),
    }


def _native_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != NATIVE_HOLDOUT_EXAMPLES:
        raise ValueError("Native metrics require all 1000 manifest-order rows")
    expected_filtered = [334, 478, 610, 780, 792]
    filtered_indices = [int(row["dataset_index"]) for row in rows if row["filtered"]]
    if filtered_indices != expected_filtered:
        raise RuntimeError("Native repetition-filter panel differs from the reviewed holdout")
    for row in rows:
        mask_tokens = int(row["mask_tokens"])
        labeled_tokens = int(row["labeled_tokens"])
        mask_weight = float(row["mask_loss_weight"])
        labeled_weight = float(row["labeled_loss_weight"])
        summed_ce = float(row["summed_ce"])
        if row["filtered"]:
            valid = (
                labeled_tokens == 0
                and labeled_weight == 0.0
                and summed_ce == 0.0
                and row["ce"] is None
            )
        else:
            valid = (
                labeled_tokens == mask_tokens
                and labeled_weight == mask_weight
                and isinstance(row["ce"], (int, float))
                and float(row["ce"]) == summed_ce / labeled_weight
            )
        if not valid:
            raise ValueError("Native per-row dual-denominator evidence is inconsistent")
    summed_ce = sum(float(row["summed_ce"]) for row in rows)
    mask_weight = sum(float(row["mask_loss_weight"]) for row in rows)
    labeled_weight = sum(float(row["labeled_loss_weight"]) for row in rows)
    ce_loss = summed_ce / labeled_weight
    training_divisor_ce = summed_ce / mask_weight
    return {
        "examples": NATIVE_HOLDOUT_EXAMPLES,
        "filtered_examples": len(filtered_indices),
        "filtered_indices": filtered_indices,
        "mask_tokens": sum(int(row["mask_tokens"]) for row in rows),
        "labeled_tokens": sum(int(row["labeled_tokens"]) for row in rows),
        "mask_loss_weight": mask_weight,
        "labeled_loss_weight": labeled_weight,
        "summed_ce": summed_ce,
        "ce_loss": ce_loss,
        "ppl": math.exp(ce_loss),
        "training_divisor_ce": training_divisor_ce,
        "training_divisor_ppl": math.exp(training_divisor_ce),
        "ce_mean": float(np.mean([float(row["ce"]) for row in rows if row["ce"] is not None])),
    }


def _gather_ordered_records(
    local_records: Sequence[Mapping[str, Any]],
    *,
    position_field: str,
    expected: int,
    process_group: Any,
    world_size: int,
) -> list[dict[str, Any]]:
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, list(local_records), group=process_group)
    records = [dict(record) for rank_records in gathered for record in rank_records]
    records.sort(key=lambda row: row[position_field])
    if len(records) != expected or [row[position_field] for row in records] != list(
        range(expected)
    ):
        raise RuntimeError("Distributed per-example results contain a drop or duplicate")
    return records


def _evaluate_visual_source(
    train_module: Any,
    dataset: Any,
    *,
    source: str,
    pairing: Mapping[str, Any],
    pairing_sha256: str,
    collator: MultimodalCollator,
    work_dir: Path,
    dp_world_size: int,
    dp_rank: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    correct_dataset = MultimodalFixedValidationDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    wrong_dataset = MultimodalMatchedWrongImageDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    examples = len(correct_dataset)
    if examples % GLOBAL_BATCH_INSTANCES:
        raise ValueError("Joint matched/wrong population is not globally batch-aligned")

    def loader(name: str, selected: Any) -> MultimodalDataLoader:
        value = MultimodalDataLoader(
            selected,
            collator,
            work_dir=work_dir / source / name,
            global_batch_size=GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH,
            seed=PAIRING_SEED,
            shuffle=False,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
        value.reshuffle(epoch=1)
        return value

    correct_loader = loader("correct", correct_dataset)
    wrong_loader = loader("wrong", wrong_dataset)
    local_visual: list[dict[str, Any]] = []
    local_blank: list[dict[str, Any]] = []
    started = time.monotonic()
    blank_elapsed = 0.0
    pair_rows = list(pairing["pairs"])
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        zip(correct_loader, wrong_loader, strict=True)
    ):
        bridge._assert_batches_match(correct_batch, wrong_batch)
        position = batch_index * GLOBAL_BATCH_INSTANCES + dp_rank
        pair = pair_rows[position]
        correct_result = _forward_scalar_ce(train_module, correct_batch)
        wrong_result = _forward_scalar_ce(train_module, wrong_batch)
        if correct_result["filtered"] or wrong_result["filtered"]:
            raise RuntimeError("Visual evaluation rows cannot be repetition-filtered")
        if (
            correct_result["mask_tokens"] != correct_result["labeled_tokens"]
            or correct_result["mask_loss_weight"] != correct_result["labeled_loss_weight"]
            or wrong_result["mask_tokens"] != correct_result["mask_tokens"]
            or wrong_result["mask_loss_weight"] != correct_result["mask_loss_weight"]
        ):
            raise RuntimeError("Correct/wrong forwards changed response supervision")
        correct_ce = float(correct_result["ce"])
        wrong_ce = float(wrong_result["ce"])
        response_tokens = int(correct_result["labeled_tokens"])
        loss_weight = float(correct_result["labeled_loss_weight"])
        local_visual.append(
            {
                "pairing_position": position,
                "recipient_index": int(pair["recipient"]),
                "donor_index": int(pair["donor"]),
                "response_tokens": response_tokens,
                "loss_weight": loss_weight,
                "correct_ce": correct_ce,
                "wrong_ce": wrong_ce,
                "ce_gap_wrong_minus_correct": wrong_ce - correct_ce,
            }
        )
        if source in BLANK_SOURCE_NAMES:
            images = correct_batch.get("images")
            if not isinstance(images, torch.Tensor):
                raise TypeError("Blank-image control requires tensor images")
            blank_batch = dict(correct_batch)
            blank_batch["images"] = torch.zeros_like(images)
            if (
                blank_batch["images"].shape != images.shape
                or blank_batch["images"].dtype != images.dtype
                or int(torch.count_nonzero(blank_batch["images"]).item()) != 0
            ):
                raise RuntimeError("Blank-image control did not preserve zeroed geometry")
            blank_started = time.monotonic()
            blank_result = _forward_scalar_ce(train_module, blank_batch)
            blank_elapsed += time.monotonic() - blank_started
            if (
                blank_result["filtered"]
                or response_tokens != blank_result["labeled_tokens"]
                or loss_weight != blank_result["labeled_loss_weight"]
            ):
                raise RuntimeError("Blank-image forward changed response supervision")
            blank_ce = float(blank_result["ce"])
            local_blank.append(
                {
                    "pairing_position": position,
                    "recipient_index": int(pair["recipient"]),
                    "response_tokens": response_tokens,
                    "loss_weight": loss_weight,
                    "correct_ce": correct_ce,
                    "blank_ce": blank_ce,
                    "ce_gap_blank_minus_correct": blank_ce - correct_ce,
                }
            )
        gc_cuda()
    visual_rows = _gather_ordered_records(
        local_visual,
        position_field="pairing_position",
        expected=examples,
        process_group=train_module.dp_process_group,
        world_size=dp_world_size,
    )
    visual = {
        "pairing_sha256": pairing_sha256,
        "examples": examples,
        "elapsed_seconds": time.monotonic() - started,
        "population": "matched_eligible_joint_validation_subset",
        "coverage": pairing["coverage"],
        "metrics": _visual_metrics(visual_rows),
        "per_example": visual_rows,
    }
    blank = None
    if source in BLANK_SOURCE_NAMES:
        blank_rows = _gather_ordered_records(
            local_blank,
            position_field="pairing_position",
            expected=examples,
            process_group=train_module.dp_process_group,
            world_size=dp_world_size,
        )
        for visual_row, blank_row in zip(visual_rows, blank_rows, strict=True):
            if (
                visual_row["pairing_position"] != blank_row["pairing_position"]
                or visual_row["recipient_index"] != blank_row["recipient_index"]
                or visual_row["correct_ce"] != blank_row["correct_ce"]
                or visual_row["loss_weight"] != blank_row["loss_weight"]
            ):
                raise RuntimeError("Blank control does not cross-bind the visual recipients")
        blank = {
            "pairing_sha256": pairing_sha256,
            "examples": examples,
            "elapsed_seconds": blank_elapsed,
            "population": "matched_eligible_joint_validation_subset",
            "coverage": pairing["coverage"],
            "metrics": _blank_metrics(blank_rows),
            "per_example": blank_rows,
        }
    return visual, blank


def _evaluate_native_holdout(
    train_module: Any,
    dataset: _ValidatedJointDataset,
    *,
    native_identity: Mapping[str, Any],
    collator: MultimodalCollator,
    work_dir: Path,
    dp_world_size: int,
    dp_rank: int,
) -> dict[str, Any]:
    if len(dataset) != NATIVE_HOLDOUT_EXAMPLES:
        raise ValueError("Native holdout evaluation requires all 1000 windows")
    loader = MultimodalDataLoader(
        dataset,
        collator,
        work_dir=work_dir / "native_text_holdout",
        global_batch_size=GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH,
        seed=PAIRING_SEED,
        shuffle=False,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    loader.reshuffle(epoch=1)
    local: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_index, batch in enumerate(loader):
        position = batch_index * GLOBAL_BATCH_INSTANCES + dp_rank
        result = _forward_scalar_ce(train_module, batch)
        provenance = dict(dataset.provenance_for(position))
        if provenance.get("manifest_index") != position:
            raise RuntimeError("Native holdout row order differs from manifest order")
        local.append(
            {
                "evaluation_position": position,
                "dataset_index": position,
                "provenance": provenance,
                "mask_tokens": result["mask_tokens"],
                "labeled_tokens": result["labeled_tokens"],
                "mask_loss_weight": result["mask_loss_weight"],
                "labeled_loss_weight": result["labeled_loss_weight"],
                "summed_ce": result["summed_ce"],
                "filtered": result["filtered"],
                "ce": result["ce"],
            }
        )
        gc_cuda()
    rows = _gather_ordered_records(
        local,
        position_field="evaluation_position",
        expected=NATIVE_HOLDOUT_EXAMPLES,
        process_group=train_module.dp_process_group,
        world_size=dp_world_size,
    )
    provenance_sha = _canonical_sha256([row["provenance"] for row in rows])
    if provenance_sha != native_identity["row_provenance_sha256"]:
        raise RuntimeError("Evaluated native provenance differs from preflight manifest order")
    return {
        "examples": NATIVE_HOLDOUT_EXAMPLES,
        "elapsed_seconds": time.monotonic() - started,
        "dataset_order_sha256": _canonical_sha256([row["dataset_index"] for row in rows]),
        "row_provenance_sha256": provenance_sha,
        "native_identity_sha256": _canonical_sha256(native_identity),
        "metrics": _native_metrics(rows),
        "per_example": rows,
    }


def _tokenizer_identity(raw_config: Mapping[str, Any], token_ids: Any) -> dict[str, Any]:
    artifacts = raw_config["artifacts"]
    values = token_ids.as_config_dict()
    return {
        "id": artifacts["tokenizer_id"],
        "revision": artifacts["tokenizer_revision"],
        "fingerprint": artifacts["tokenizer_fingerprint"],
        "token_ids": values,
        "token_ids_sha256": _canonical_sha256(values),
    }


def _artifact_preflight_distributed(output: Path) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite immutable output {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            _artifact_path(output.parent, name="output directory")
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Output artifact preflight failed: {detail}")


def _validate_work_dir(
    work_dir: Path,
    *,
    checkpoint_root: Path,
    pairing_dir: Path,
    output: Path,
    pairing_manifest: Path,
    raw_config: Mapping[str, Any],
) -> Path:
    work = _artifact_path(work_dir, name="joint evaluation work directory")
    protected = [
        checkpoint_root,
        pairing_dir,
        output,
        pairing_manifest,
        Path(str(raw_config["data"]["joint_visual_projection_path"])).parent,
        Path(str(raw_config["data"]["source_audit_path"])).parent,
        Path(str(raw_config["data"]["native_text_replay"]["manifest_path"])).parent,
        Path(str(raw_config["evaluation"]["native_text_holdout"]["manifest_path"])).parent,
    ]
    for value in protected:
        path = _artifact_path(value, name="protected evidence directory")
        if work == path or work.is_relative_to(path) or path.is_relative_to(work):
            raise ValueError(f"Evaluation work directory {work} overlaps protected evidence {path}")
    if work.exists() and any(work.iterdir()):
        raise ValueError("Evaluation work directory must be new or empty")
    return work


def _producer_identity() -> dict[str, str]:
    producer = Path(__file__).resolve()
    comparator_path = producer.with_name("vision_alignment_joint_matched_wrong_compare.py")
    perception_path = Path(str(perception.__file__)).resolve()
    bridge_path = Path(str(bridge.__file__)).resolve()
    pairing_source = inspect.getsourcefile(build_matched_wrong_image_pairing)
    training_path = Path(__file__).resolve().parents[1] / "train" / "Vision-Alignment.py"
    if pairing_source is None:
        raise RuntimeError("Could not locate pairing implementation")
    pairing_path = Path(pairing_source).resolve()
    return {
        "path": str(producer),
        "sha256": _stable_file_sha256(producer, name="joint evaluator"),
        "comparator_path": str(comparator_path),
        "comparator_sha256": _stable_file_sha256(comparator_path, name="joint receipt comparator"),
        "perception_helper_path": str(perception_path),
        "perception_helper_sha256": _stable_file_sha256(
            perception_path, name="perception evaluator helper"
        ),
        "bridge_helper_path": str(bridge_path),
        "bridge_helper_sha256": _stable_file_sha256(bridge_path, name="bridge evaluator helper"),
        "pairing_implementation_path": str(pairing_path),
        "pairing_implementation_sha256": _stable_file_sha256(
            pairing_path, name="pairing implementation"
        ),
        "training_contract_path": str(training_path.resolve()),
        "training_contract_sha256": _stable_file_sha256(
            training_path.resolve(), name="joint training contract"
        ),
    }


def _validate_clean_git_identity(value: Any) -> dict[str, Any]:
    fields = {"revision", "dirty", "status_sha256", "tracked_diff_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Evaluator git identity fields differ")
    revision = value.get("revision")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or value.get("dirty") is not False
        or not _is_sha256(value.get("status_sha256"))
        or not _is_sha256(value.get("tracked_diff_sha256"))
    ):
        raise ValueError("Evaluator requires one exact clean git revision")
    return dict(value)


def _protocol(
    *,
    examples: int,
    pairing_sha256: Mapping[str, str],
    dp_world_size: int,
    checkpoint_config: Mapping[str, Any],
    projection: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    native_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": PROTOCOL_NAME,
        "descriptive_only": True,
        "promotion_eligible": False,
        "primary_statistic": (
            "paired source-balanced change in wrong-minus-correct CE from step4000 to step8000"
        ),
        "per_checkpoint_statistic": "all-response loss-weighted scalar CE",
        "response_logits_materialized": False,
        "sources": list(JOINT_VISUAL_SOURCE_NAMES),
        "blank_sources": list(BLANK_SOURCE_NAMES),
        "native_source": "native_text_replay",
        "visual_split": "validation",
        "visual_population": "matched_eligible_joint_validation_subset",
        "examples_per_visual_source": examples,
        "native_population": "all holdout windows in exact manifest order",
        "native_examples": NATIVE_HOLDOUT_EXAMPLES,
        "native_filtered_indices": [334, 478, 610, 780, 792],
        "pairing_seed": PAIRING_SEED,
        "pairing_sha256": dict(pairing_sha256),
        "pairing_rule": (
            "largest common multiple-of-eight; distinct pinned image content and exact "
            "collated geometry; deterministic explicit unique donors"
        ),
        "recipient_replay": "correct, wrong, and applicable blank forwards share recipients",
        "blank_rule": "zeros_like normalized image tensor; all non-image fields unchanged",
        "ce_definition": (
            "scalar summed CE divided by the one rank-local example's positive labeled loss "
            "weight; no response logits"
        ),
        "native_dual_denominator": (
            "inline CE uses labeled loss weight; training-divisor CE uses all mask loss weight"
        ),
        "sequence_length": SEQUENCE_LENGTH,
        "rank_batch_instances": RANK_BATCH_INSTANCES,
        "global_batch_instances": GLOBAL_BATCH_INSTANCES,
        "nodes": 1,
        "world_size": WORLD_SIZE,
        "local_world_size": LOCAL_WORLD_SIZE,
        "ep_degree": EP_DEGREE,
        "dp_process_group_size": dp_world_size,
        "training_beaker_image": checkpoint_config["training_beaker_image"],
        "training_git_ref": checkpoint_config["training_git_ref"],
        "checkpoint_config_sha256": checkpoint_config["sha256"],
        "projection_raw_sha256": projection["raw_sha256"],
        "source_audit_fingerprint": source_audit["fingerprint"],
        "native_holdout_fingerprint": native_identity["holdout_fingerprint"],
        "native_row_provenance_sha256": native_identity["row_provenance_sha256"],
        "native_identity": dict(native_identity),
    }


def _receipt_payload(
    *,
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
        },
        "checkpoint": dict(checkpoint),
        "checkpoint_config": dict(checkpoint_config),
        "load_coverage": dict(load_coverage),
        "projection": dict(projection),
        "source_audit": dict(source_audit),
        "tokenizer": dict(tokenizer),
        "pairing_manifest": dict(pairing_manifest),
        "protocol": dict(protocol),
        "visual_results": {source: visual_results[source] for source in JOINT_VISUAL_SOURCE_NAMES},
        "blank_results": {source: blank_results[source] for source in BLANK_SOURCE_NAMES},
        "native_result": dict(native_result),
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _write_validated_receipt(
    output: Path,
    payload: Mapping[str, Any],
    *,
    work_dir: Path,
    step: int,
) -> str:
    """Cross-validate scratch bytes, then publish those same retained bytes once."""
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = _artifact_path(work_dir, name="receipt validation work directory")
    validation_root = Path(tempfile.mkdtemp(prefix=".joint-receipt-validation-", dir=work_dir))
    validation_identity = _stat_signature(validation_root.lstat())[:3]
    try:
        raw = _json_output_bytes(payload)
        candidate = validation_root / "receipt.json"
        candidate_sha = _write_bytes_exclusive(candidate, raw)
        comparator = _load_comparator_contract()
        validator = getattr(comparator, "validate_evaluator_receipt", None)
        if not callable(validator):
            raise RuntimeError("Joint comparator lacks the frozen one-receipt validator")
        validator(candidate, candidate_sha, step, verify_live_checkpoint=False)
        current_producer = _producer_identity()
        current_git = _validate_clean_git_identity(bridge._git_identity())
        if current_producer != payload.get("producer") or current_git != payload.get("git"):
            raise RuntimeError(
                "Evaluator implementation or clean git identity changed before publication"
            )
        published_sha = _write_bytes_exclusive(output, raw)
        if published_sha != candidate_sha:
            raise RuntimeError("Published receipt differs from comparator-validated bytes")
        loaded, reloaded_sha = _load_json_bytes(
            output,
            expected_sha256=candidate_sha,
            name="published joint evaluator receipt",
        )
        if _canonical_bytes(loaded) != _canonical_bytes(payload) or reloaded_sha != candidate_sha:
            raise RuntimeError("Published joint evaluator receipt failed its strict reload")
        return published_sha
    finally:
        if (
            validation_root.name.startswith(".joint-receipt-validation-")
            and validation_root.parent == work_dir
            and validation_root.exists()
            and _stat_signature(validation_root.lstat())[:3] == validation_identity
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
    if dist.get_rank() == 0:
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
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not persist joint evaluation receipt: {detail}")


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare exact pairings or evaluate one permanent joint checkpoint."""
    args = _parser().parse_args(argv)
    _validate_args(args)
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime_state = _initialize_runtime(pairing_only=args.pairing_only)
    try:
        checkpoint = perception._direct_existing_path(
            Path(args.checkpoint), name="joint checkpoint"
        )
        root = _checkpoint_root(checkpoint)
        config_path = perception._direct_existing_path(
            Path(args.config).expanduser() if args.config is not None else root / "config.json",
            name="joint checkpoint config",
        )
        raw_config, checkpoint_config, step = _checkpoint_config_identity(
            checkpoint, config_path, args.expected_config_sha256
        )
        output = _artifact_path(
            Path(args.pairing_manifest_output if args.pairing_only else args.output),
            name="joint output",
        )
        _artifact_preflight_distributed(output)
        initial_producer = _producer_identity()
        initial_git = _validate_clean_git_identity(bridge._git_identity())

        artifacts = raw_config["artifacts"]
        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=artifacts["tokenizer_id"],
            revision=artifacts["tokenizer_revision"],
            expected_fingerprint=artifacts["tokenizer_fingerprint"],
            cache_dir=artifacts["hf_cache_dir"],
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("Pinned joint tokenizer has no pad token")
        if int(raw_config["model"]["image_patch_token_id"]) != token_ids.im_patch_id:
            raise ValueError("Joint checkpoint image-patch ID differs from tokenizer")
        tokenizer_ref = _tokenizer_identity(raw_config, token_ids)
        projection = _load_projection_distributed(raw_config, token_ids)
        datasets = _build_visual_datasets(projection, tokenizer, token_ids)
        projection_ref = _validate_visual_population_distributed(datasets, projection)
        experiment, source_audit = _validate_source_audit_distributed(
            raw_config, token_ids, projection
        )
        pairing_dir = _artifact_path(Path(args.pairing_dir), name="pairing directory")
        paths = _pairing_paths(pairing_dir)
        content_ids = _content_ids(projection)

        if args.pairing_only:
            examples, _, pairings = _prepare_pairings_distributed(
                datasets,
                paths=paths,
                content_ids=content_ids,
            )
            pairing_manifest = _pairing_manifest_payload(
                checkpoint_config=checkpoint_config,
                projection=projection_ref,
                source_audit=source_audit,
                tokenizer=tokenizer_ref,
                examples=examples,
                pairings=pairings,
            )
            if (
                _producer_identity() != initial_producer
                or _validate_clean_git_identity(bridge._git_identity()) != initial_git
            ):
                raise RuntimeError(
                    "Evaluator implementation or git worktree changed during pairing preparation"
                )
            _write_pairing_manifest_distributed(output, pairing_manifest)
            return

        (
            pairing_payloads,
            pairing_metadata,
            pairing_manifest_ref,
        ) = _load_pairing_manifest_distributed(
            _artifact_path(Path(args.pairing_manifest), name="pairing manifest"),
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
        example_counts = {int(payload["recipient_count"]) for payload in pairing_payloads.values()}
        if len(example_counts) != 1:
            raise ValueError("Joint pairings use different recipient counts")
        examples_per_source = example_counts.pop()
        native_dataset, native_identity = _load_native_evidence_distributed(
            raw_config, tokenizer, token_ids, experiment
        )
        checkpoint_identity = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            step=step,
            hash_workers=args.checkpoint_hash_workers,
        )
        if checkpoint_identity["config_sha256"] != checkpoint_config["sha256"]:
            raise RuntimeError("Checkpoint inventory does not bind the expected config")
        pairing_manifest_path = _artifact_path(Path(args.pairing_manifest), name="pairing manifest")
        work_dir = _validate_work_dir(
            Path(args.work_dir),
            checkpoint_root=root,
            pairing_dir=pairing_dir,
            output=output,
            pairing_manifest=pairing_manifest_path,
            raw_config=raw_config,
        )
        model, module_config = bridge._build_model_and_module(
            raw_config,
            sequence_length=SEQUENCE_LENGTH,
            rank_batch_instances=RANK_BATCH_INSTANCES,
        )
        train_module = module_config.build(model, eval_only=True)
        snapshot_state_dir = _materialize_checkpoint_snapshot_distributed(
            checkpoint_identity,
            base_dir=work_dir / "checkpoint-snapshots",
        )
        try:
            load_coverage = bridge._native_checkpoint_load_coverage_distributed(
                train_module, snapshot_state_dir
            )
            train_module.load_state_dict_direct(
                snapshot_state_dir,
                process_group=dist.group.WORLD,
                thread_count=args.checkpoint_load_threads,
                load_optim_state=False,
            )
        finally:
            _remove_checkpoint_snapshot_distributed(snapshot_state_dir)
        load_coverage["load_completed"] = True
        load_coverage["sha256"] = _canonical_sha256(
            {key: value for key, value in load_coverage.items() if key != "sha256"}
        )
        if load_coverage.get("complete") is not True:
            raise RuntimeError("Native checkpoint load coverage is incomplete")
        dp_world_size = get_world_size(train_module.dp_process_group)
        dp_rank = get_rank(train_module.dp_process_group)
        if dp_world_size != WORLD_SIZE:
            raise ValueError("Joint evaluation requires exact DP process-group size 8")
        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=SEQUENCE_LENGTH,
        )
        visual_results: dict[str, Any] = {}
        blank_results: dict[str, Any] = {}
        for source in JOINT_VISUAL_SOURCE_NAMES:
            visual, blank = _evaluate_visual_source(
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
        if set(blank_results) != set(BLANK_SOURCE_NAMES):
            raise RuntimeError("Blank-image controls do not cover exact caption/transcript set")
        native_result = _evaluate_native_holdout(
            train_module,
            native_dataset,
            native_identity=native_identity,
            collator=collator,
            work_dir=work_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

        post_payloads, post_metadata, post_manifest_ref = _load_pairing_manifest_distributed(
            _artifact_path(Path(args.pairing_manifest), name="pairing manifest"),
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
            _canonical_bytes(post_payloads) != _canonical_bytes(pairing_payloads)
            or _canonical_bytes(post_metadata) != _canonical_bytes(pairing_metadata)
            or post_manifest_ref != pairing_manifest_ref
        ):
            raise RuntimeError("Pairing evidence changed during joint evaluation")
        _rehash_visual_images_distributed(datasets, projection_ref)
        post_checkpoint = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            step=step,
            hash_workers=args.checkpoint_hash_workers,
        )
        if post_checkpoint != checkpoint_identity:
            raise RuntimeError("Permanent checkpoint changed during joint evaluation")
        post_projection = _load_projection_distributed(raw_config, token_ids)
        _, post_audit = _validate_source_audit_distributed(raw_config, token_ids, post_projection)
        if post_audit != source_audit:
            raise RuntimeError("Joint source audit changed during evaluation")
        _, post_native_identity = _load_native_evidence_distributed(
            raw_config, tokenizer, token_ids, experiment
        )
        if post_native_identity != native_identity:
            raise RuntimeError("Native replay evidence changed during evaluation")

        pairing_sha = {
            source: pairing_metadata[source]["sha256"] for source in JOINT_VISUAL_SOURCE_NAMES
        }
        protocol = _protocol(
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
            or _validate_clean_git_identity(bridge._git_identity()) != initial_git
        ):
            raise RuntimeError("Evaluator implementation or git worktree changed during scoring")
        receipt = _receipt_payload(
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
        _teardown_runtime(runtime_state)


if __name__ == "__main__":
    main()
