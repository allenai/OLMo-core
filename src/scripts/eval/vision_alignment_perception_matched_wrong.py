"""Evaluate perception control/treatment checkpoints with pinned wrong-image pairings.

This is intentionally separate from ``vision_alignment_matched_wrong.py``. The latter's exact
bytes are embedded in historical bridge evidence. This evaluator dynamically loads that
immutable implementation and reuses its native EP8 checkpoint and per-example CE machinery while
adding the eight-source perception provenance contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import torch.distributed as dist

from olmo_core.data.multimodal import MultimodalCollator
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    PerceptionProvenanceManifest,
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import (
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    serialize_matched_wrong_image_pairing,
    validate_matched_wrong_image_pairing,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment

WORLD_SIZE = 8
EP_DEGREE = 8
GLOBAL_BATCH_INSTANCES = 32
SCHEMA_VERSION = 4
PROTOCOL_NAME = "vision-alignment-perception-native-matched-wrong-image-v4"
PROFILE_PAIR_FORMAT = "vision_alignment_perception_profile_pair_audit"
PROFILE_PAIR_VERSION = 2
PAIRING_MANIFEST_FORMAT = "vision_alignment_perception_pairing_manifest"
PAIRING_MANIFEST_VERSION = 1
ARMS = ("frozen_vision_control", "treatment")
STEPS = (3000, 4000)
EXPECTED_FREEZE = ("lm.embedding_norm.*", "lm.blocks.*", "lm.lm_head.*")
EXPECTED_PROFILE_PAIR_RECEIPT_SHA256 = (
    "5c7d9f3b2a882ed3147ca239eaaf00e9089d8e47c552a5cd19c351fdd806ea04"
)
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


def _load_bridge_helpers() -> ModuleType:
    path = Path(__file__).resolve().with_name("vision_alignment_matched_wrong.py")
    spec = importlib.util.spec_from_file_location("_vision_alignment_bridge_matched_wrong", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load immutable bridge evaluator helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge_helpers()


def _project_model_input_example(example: Any, *, index: int) -> dict[str, Any]:
    """Drop only diagnostic metadata while validating the canonical model-input schema."""
    if not isinstance(example, Mapping):
        raise TypeError(f"Perception validation row {index} is not a mapping")
    fields = set(example)
    missing = sorted(set(_PAIRING_REQUIRED_MODEL_FIELDS) - fields)
    if missing:
        raise ValueError(f"Perception validation row {index} is missing model fields {missing}")
    supported = (
        set(_PAIRING_REQUIRED_MODEL_FIELDS)
        | set(_PAIRING_OPTIONAL_MODEL_FIELDS)
        | _PAIRING_IGNORED_FIELDS
    )
    unknown = fields - supported
    if unknown:
        raise ValueError(
            f"Perception validation row {index} has unsupported non-model fields "
            f"{sorted(unknown, key=str)}"
        )
    model_fields = list(_PAIRING_REQUIRED_MODEL_FIELDS)
    model_fields.extend(field for field in _PAIRING_OPTIONAL_MODEL_FIELDS if field in example)
    return {field: example[field] for field in model_fields}


class _PairingModelInputDataset:
    """Expose exactly the fields consumed by the canonical multimodal collator.

    Perception adapters attach a diagnostic ``metadata`` mapping. The immutable bridge pairing
    format describes array/tensor fields only, so that mapping cannot be serialized as an array.
    Projecting through the canonical serialized-example contract keeps every model-consumed field
    byte-bound by bridge pairing construction and replay, rejects any unknown field, and drops only
    metadata that :class:`MultimodalCollator` does not consume.
    """

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
        """Delegate raw image-byte validation to the provenance-selected dataset."""
        validate = getattr(self.dataset, "validate_image_content", None)
        if not callable(validate):
            raise ValueError("Perception validation dataset lacks live image-content validation")
        return str(validate(indices))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Config JSON (defaults to CHECKPOINT/config.json).")
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--profile-pair-receipt", required=True)
    parser.add_argument("--expected-profile-pair-receipt-sha256", required=True)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--pairing", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--pairing-dir")
    parser.add_argument(
        "--expected-pairing-sha256", action="append", default=[], metavar="SOURCE=SHA256"
    )
    parser.add_argument("--pairing-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--rank-batch-instances", type=int)
    parser.add_argument("--work-dir")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    parser.add_argument(
        "--pairing-only",
        action="store_true",
        help="Prepare the largest common batch-aligned pairing set and exit before model load.",
    )
    parser.add_argument("--pairing-manifest-output")
    return parser


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_args(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world_size = 1 if args.pairing_only else WORLD_SIZE
    if world_size != expected_world_size:
        purpose = "CPU pairing preparation" if args.pairing_only else "EP8 evaluation"
        raise ValueError(
            f"Perception {purpose} requires WORLD_SIZE={expected_world_size}, got {world_size}"
        )
    for value, name in (
        (args.expected_config_sha256, "config"),
        (args.expected_profile_pair_receipt_sha256, "profile-pair receipt"),
    ):
        if not _is_sha256(value):
            raise ValueError(f"Expected {name} identity must be a lowercase SHA-256")
    for value, name in (
        (args.examples, "--examples"),
        (args.bootstrap_samples, "--bootstrap-samples"),
        (args.checkpoint_load_threads, "--checkpoint-load-threads"),
        (args.checkpoint_hash_workers, "--checkpoint-hash-workers"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be positive")
    for value, name in (
        (args.pairing_seed, "--pairing-seed"),
        (args.bootstrap_seed, "--bootstrap-seed"),
    ):
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be non-negative")
    if args.rank_batch_instances is not None and args.rank_batch_instances != 4:
        raise ValueError("Perception evaluation requires --rank-batch-instances=4")
    if args.pairing_only:
        if args.output is not None or not args.pairing_manifest_output:
            raise ValueError(
                "--pairing-only requires --pairing-manifest-output and forbids --output"
            )
    elif args.pairing_manifest_output is not None:
        raise ValueError("--pairing-manifest-output requires --pairing-only")
    elif not args.output:
        raise ValueError("Promotion-grade evaluation requires an explicit new --output")


def _parse_source_values(values: Sequence[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be SOURCE=VALUE, got {value!r}")
        source, item = value.split("=", 1)
        if source not in PERCEPTION_SOURCE_NAMES or not item or source in result:
            raise ValueError(f"Invalid or duplicate {option} value {value!r}")
        result[source] = item
    return result


def _pairing_paths(args: argparse.Namespace, artifact_output: Path) -> dict[str, Path]:
    explicit = {
        source: Path(value).expanduser().resolve()
        for source, value in _parse_source_values(args.pairing, option="--pairing").items()
    }
    root = (
        Path(args.pairing_dir).expanduser().resolve()
        if args.pairing_dir
        else artifact_output.parent / f"{artifact_output.stem}-pairings"
    )
    return {
        source: explicit.get(source, root / f"{source}.json") for source in PERCEPTION_SOURCE_NAMES
    }


def _pairing_pins(args: argparse.Namespace) -> dict[str, str]:
    pins = _parse_source_values(args.expected_pairing_sha256, option="--expected-pairing-sha256")
    if any(not _is_sha256(value) for value in pins.values()):
        raise ValueError("Every expected pairing identity must be a lowercase SHA-256")
    return pins


def _canonical_sha256(value: Any) -> str:
    return bridge._canonical_sha256(value)


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
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {name} from {path}: {error}") from error
    return payload, digest


def _stable_file_record(
    path: Path, *, root: Path
) -> tuple[dict[str, Any], tuple[int, int, int, int, int, int]]:
    """Hash one regular file while proving its directory entry stayed on the same inode."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Could not open immutable checkpoint file {path}: {error}") from error
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
    try:
        current = path.lstat()
    except OSError as error:
        raise ValueError(f"Checkpoint file disappeared after hashing: {path}") from error

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    signature = identity(before)
    if signature != identity(after) or signature != identity(current) or size != before.st_size:
        raise ValueError(f"Checkpoint file changed while it was hashed: {path}")
    return (
        {
            "path": path.relative_to(root).as_posix(),
            "size": size,
            "sha256": digest.hexdigest(),
        },
        signature,
    )


def _direct_existing_path(path: Path, *, name: str) -> Path:
    """Return an absolute path while rejecting every symlinked component."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (*reversed(absolute.parents), absolute):
        if component == Path(component.anchor):
            continue
        try:
            info = component.lstat()
        except OSError as error:
            raise ValueError(f"{name} component is unavailable: {component}: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} contains a symlinked component: {component}")
    return absolute


def _checkpoint_identity(
    checkpoint: Path, config_path: Path, *, hash_workers: int
) -> dict[str, Any]:
    """Build a full checkpoint identity from stable, non-symlinked byte snapshots."""
    root = _direct_existing_path(bridge._checkpoint_root(checkpoint), name="checkpoint root")
    state_dir = _direct_existing_path(
        bridge._checkpoint_state_dir(checkpoint), name="checkpoint state directory"
    )
    if hash_workers <= 0:
        raise ValueError("Checkpoint hash worker count must be positive")
    entries = sorted(state_dir.iterdir())
    if not entries:
        raise ValueError("Distributed checkpoint does not contain state files")
    if any(not stat.S_ISREG(path.lstat().st_mode) for path in entries):
        raise ValueError("Distributed checkpoint contains a non-regular or symlinked entry")
    with ThreadPoolExecutor(max_workers=min(hash_workers, len(entries))) as executor:
        state_records = list(
            executor.map(lambda path: _stable_file_record(path, root=root), entries)
        )
    inventory = [record for record, _ in state_records]
    expected_paths = {item["path"] for item in inventory}
    dcp_metadata = state_dir / ".metadata"
    if dcp_metadata.relative_to(root).as_posix() not in expected_paths:
        raise ValueError("Distributed checkpoint metadata is missing from the state inventory")
    config_path = _direct_existing_path(config_path, name="checkpoint config")
    marker_path = _direct_existing_path(root / ".metadata.json", name="checkpoint marker")
    config_record, config_signature = _stable_file_record(config_path, root=root)
    marker_record, marker_signature = _stable_file_record(marker_path, root=root)
    for path, (_, signature) in zip(entries, state_records, strict=True):
        info = path.lstat()
        if (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ) != signature:
            raise ValueError(f"Checkpoint file changed during the full snapshot: {path}")
    for path, signature in (
        (config_path, config_signature),
        (marker_path, marker_signature),
    ):
        info = path.lstat()
        if (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ) != signature:
            raise ValueError(f"Checkpoint file changed during the full snapshot: {path}")
    if entries != sorted(state_dir.iterdir()):
        raise ValueError("Distributed checkpoint entries changed during the full snapshot")
    dcp_record = next(
        item for item in inventory if item["path"] == dcp_metadata.relative_to(root).as_posix()
    )
    identity = {
        "root": str(root),
        "state_dir": str(state_dir),
        "config_sha256": config_record["sha256"],
        "checkpoint_marker_sha256": marker_record["sha256"],
        "dcp_metadata_sha256": dcp_record["sha256"],
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": _canonical_sha256(inventory),
        "state_file_inventory": inventory,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _checkpoint_identity_distributed(
    checkpoint: Path, config_path: Path, *, hash_workers: int
) -> Mapping[str, Any]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {
                "ok": True,
                "identity": _checkpoint_identity(
                    checkpoint, config_path, hash_workers=hash_workers
                ),
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not identify a stable checkpoint snapshot: {detail}")
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Stable checkpoint identity broadcast is malformed")
    return identity


def _validate_live_checkpoint_identity_stable(
    identity: Mapping[str, Any], *, name: str, hash_workers: int = 8
) -> None:
    """Rebuild and compare an exact stable checkpoint identity."""
    actual = _checkpoint_identity(
        Path(str(identity["root"])),
        Path(str(identity["root"])) / "config.json",
        hash_workers=hash_workers,
    )
    if actual != identity:
        raise ValueError(f"Live {name} stable checkpoint identity differs")


def _copy_pinned_checkpoint_file(
    source: Path,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Copy the exact bytes read from one pinned source FD into a private snapshot."""
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(source, read_flags)
    try:
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
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
                    written = os.write(target_fd, view)
                    view = view[written:]
            os.fsync(target_fd)
            target_after = os.fstat(target_fd)
        finally:
            os.close(target_fd)
        source_after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    source_current = source.lstat()
    if (
        source_before.st_dev != source_after.st_dev
        or source_before.st_ino != source_after.st_ino
        or source_before.st_size != source_after.st_size
        or source_before.st_mtime_ns != source_after.st_mtime_ns
        or source_before.st_ctime_ns != source_after.st_ctime_ns
        or source_before.st_dev != source_current.st_dev
        or source_before.st_ino != source_current.st_ino
        or source_before.st_size != source_current.st_size
        or source_before.st_mtime_ns != source_current.st_mtime_ns
        or source_before.st_ctime_ns != source_current.st_ctime_ns
        or not stat.S_ISREG(source_current.st_mode)
        or size != expected_size
        or target_after.st_size != expected_size
        or digest.hexdigest() != expected_sha256
    ):
        raise ValueError(f"Checkpoint source changed or differed while snapshotting: {source}")
    target.chmod(0o400)


def _materialize_checkpoint_snapshot(identity: Mapping[str, Any], *, base_dir: Path) -> Path:
    """Create a private load-only copy whose bytes equal a stable checkpoint identity."""
    root = _direct_existing_path(Path(str(identity["root"])), name="snapshot checkpoint root")
    inventory = identity.get("state_file_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Checkpoint snapshot inventory is empty")
    base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    base_dir = _direct_existing_path(base_dir, name="checkpoint snapshot base directory")
    snapshot_root = Path(tempfile.mkdtemp(prefix=".perception-checkpoint-snapshot-", dir=base_dir))
    try:
        for raw in inventory:
            if not isinstance(raw, Mapping) or set(raw) != {"path", "size", "sha256"}:
                raise ValueError("Checkpoint snapshot inventory record is malformed")
            relative = Path(str(raw["path"]))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "model_and_optim"
                or type(raw["size"]) is not int
                or raw["size"] < 0
                or not _is_sha256(raw["sha256"])
            ):
                raise ValueError("Checkpoint snapshot inventory path or identity is invalid")
            _copy_pinned_checkpoint_file(
                root / relative,
                snapshot_root / relative,
                expected_size=raw["size"],
                expected_sha256=raw["sha256"],
            )
        state_dir = snapshot_root / "model_and_optim"
        observed = sorted(
            path.relative_to(snapshot_root).as_posix() for path in state_dir.iterdir()
        )
        expected = sorted(str(raw["path"]) for raw in inventory)
        if observed != expected:
            raise ValueError("Private checkpoint snapshot entries differ from the inventory")
        state_dir.chmod(0o500)
        snapshot_root.chmod(0o500)
        return state_dir
    except Exception:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise


def _materialize_checkpoint_snapshot_distributed(
    identity: Mapping[str, Any], *, base_dir: Path
) -> Path:
    """Materialize one rank-zero private snapshot and share its state path with EP ranks."""
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
        raise RuntimeError(f"Could not materialize private checkpoint snapshot: {detail}")
    return Path(str(result["state_dir"]))


def _remove_checkpoint_snapshot_distributed(state_dir: Path) -> None:
    """Remove only the evaluator-owned private snapshot after every rank finished loading."""
    dist.barrier()
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            root = state_dir.parent
            if not root.name.startswith(".perception-checkpoint-snapshot-"):
                raise RuntimeError(f"Refusing to remove non-snapshot path {root}")
            root.chmod(0o700)
            state_dir.chmod(0o700)
            shutil.rmtree(root)
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    if not isinstance(packet[0], Mapping) or packet[0].get("ok") is not True:
        raise RuntimeError(f"Could not remove private checkpoint snapshot: {packet[0]}")


def _checkpoint_root(checkpoint: Path) -> Path:
    return bridge._checkpoint_root(checkpoint)


def _vision_lr(raw_config: Mapping[str, Any]) -> Any:
    groups = raw_config.get("train_module", {}).get("optim", {}).get("group_overrides", [])
    matches = [
        group
        for group in groups
        if isinstance(group, Mapping) and group.get("params") == ["*vision.*"]
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("opts"), Mapping):
        raise ValueError("Checkpoint must contain exactly one vision optimizer group")
    return matches[0]["opts"].get("lr")


def _profile_pair_identity(
    raw_config: Mapping[str, Any],
    *,
    checkpoint: Path,
    config_path: Path,
    expected_config_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Bind a saved step3000/4000 config to the reviewed causal-pair receipt."""
    root = _checkpoint_root(checkpoint)
    if config_path != root / "config.json":
        raise ValueError("Perception evaluation config must be CHECKPOINT/config.json")
    if bridge._sha256_file(config_path) != expected_config_sha256:
        raise ValueError("Checkpoint config differs from its exact SHA-256 pin")
    if expected_receipt_sha256 != EXPECTED_PROFILE_PAIR_RECEIPT_SHA256:
        raise ValueError("Profile-pair receipt pin is not the reviewed causal-pair receipt")
    receipt, receipt_sha = _load_json_bytes(
        receipt_path,
        expected_sha256=expected_receipt_sha256,
        name="profile-pair receipt",
    )
    if not isinstance(receipt, Mapping):
        raise TypeError("Profile-pair receipt must be an object")
    if (
        receipt.get("format") != PROFILE_PAIR_FORMAT
        or receipt.get("version") != PROFILE_PAIR_VERSION
        or receipt.get("status") != "passed"
        or receipt.get("recipe_execution_module") != "__main__"
    ):
        raise ValueError("Profile-pair receipt identity or status is incompatible")

    step_match = re.fullmatch(r"step(3000|4000)", root.name)
    if step_match is None or raw_config.get("phase") != "perception":
        raise ValueError("Perception outcome evaluation requires a perception step3000/step4000")
    step = int(step_match.group(1))
    arm = raw_config.get("perception_trainability_arm")
    if arm not in ARMS:
        raise ValueError("Checkpoint perception arm is incompatible")
    sections: dict[str, Mapping[str, Any]] = {}
    for name in (
        "profiles",
        "save_folders",
        "comparison",
        "data",
        "git",
        "initialization",
        "launch_contract",
        "perception_contract",
    ):
        value = receipt.get(name)
        if not isinstance(value, Mapping):
            raise TypeError("Profile-pair receipt omits a reviewed contract section")
        sections[name] = value
    profiles = sections["profiles"]
    comparison = sections["comparison"]
    profile = profiles.get(arm)
    if not isinstance(profile, Mapping):
        raise TypeError(f"Profile-pair receipt omits {arm}")
    config_sections: dict[str, Mapping[str, Any]] = {}
    for name in (
        "vision_alignment",
        "data",
        "evaluation",
        "initialization",
        "launch",
        "train_module",
    ):
        value = raw_config.get(name)
        if not isinstance(value, Mapping):
            raise TypeError("Checkpoint config omits a reviewed contract section")
        config_sections[name] = value
    metadata = config_sections["vision_alignment"]
    data = config_sections["data"]
    evaluation = config_sections["evaluation"]
    initialization = config_sections["initialization"]
    launch = config_sections["launch"]
    train_module = config_sections["train_module"]
    trainable = comparison.get("trainable_contract_sha256")
    if not isinstance(trainable, Mapping):
        raise TypeError("Profile-pair receipt omits trainable contracts")

    exact_pairs = (
        (raw_config.get("required_run_name"), profile.get("name"), "run name"),
        (
            raw_config.get("reviewed_profile_path"),
            profile.get("repository_path"),
            "reviewed profile path",
        ),
        (
            raw_config.get("reviewed_profile_sha256"),
            profile.get("sha256"),
            "reviewed profile SHA-256",
        ),
        (
            str(root.parent),
            str(Path(str(sections["save_folders"].get(arm))).expanduser().resolve()),
            "save folder",
        ),
        (
            metadata.get("trainable_contract_sha256"),
            trainable.get(arm),
            "trainable contract",
        ),
        (
            metadata.get("data_contract_sha256"),
            sections["data"].get("data_contract_sha256"),
            "data contract",
        ),
        (
            data.get("perception_provenance_sha256"),
            sections["data"].get("perception_provenance_sha256"),
            "provenance",
        ),
        (
            data.get("source_audit_fingerprint"),
            sections["data"].get("source_audit_fingerprint"),
            "source audit",
        ),
        (
            initialization.get("checkpoint"),
            sections["initialization"].get("checkpoint"),
            "parent checkpoint",
        ),
        (
            initialization.get("parent_config_sha256"),
            sections["initialization"].get("parent_config_sha256"),
            "parent config",
        ),
        (
            initialization.get("parent_gate_sha256"),
            sections["initialization"].get("parent_gate_sha256"),
            "parent gate",
        ),
        (launch.get("workspace"), sections["launch_contract"].get("workspace"), "workspace"),
        (launch.get("budget"), sections["launch_contract"].get("budget"), "budget"),
        (launch.get("num_nodes"), 2, "node count"),
        (launch.get("num_gpus"), 8, "GPU count"),
        (evaluation.get("rank_batch_instances"), 4, "evaluation rank batch"),
        (evaluation.get("examples_per_source"), 512, "held-out source size"),
        (data.get("sequence_length"), 2560, "sequence length"),
    )
    for actual, expected, label in exact_pairs:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Checkpoint {label} differs from profile-pair receipt")
    if launch.get("clusters") != [sections["launch_contract"].get("cluster")]:
        raise ValueError("Checkpoint cluster differs from profile-pair receipt")
    git = launch.get("git")
    if not isinstance(git, Mapping) or git.get("ref") != sections["git"].get("ref"):
        raise ValueError("Checkpoint git revision differs from profile-pair receipt")

    freeze_params = train_module.get("freeze_params")
    expected_freeze = list(EXPECTED_FREEZE)
    expected_lr = 3e-6
    if arm == "frozen_vision_control":
        expected_freeze = ["vision.*", *expected_freeze]
        expected_lr = 0.0
    vision_lr = _vision_lr(raw_config)
    if (
        freeze_params != expected_freeze
        or isinstance(vision_lr, bool)
        or not isinstance(vision_lr, (int, float))
    ):
        raise ValueError("Checkpoint arm freeze/LR surface is malformed")
    if float(vision_lr) != expected_lr:
        raise ValueError("Checkpoint vision LR differs from its causal arm")

    return {
        "path": str(receipt_path),
        "sha256": receipt_sha,
        "format": receipt["format"],
        "version": receipt["version"],
        "arm": arm,
        "step": step,
        "profile": {
            "name": profile.get("name"),
            "repository_path": profile.get("repository_path"),
            "sha256": profile.get("sha256"),
        },
        "shared_config_sha256": comparison.get("shared_config_sha256"),
        "arm_config_sha256": (
            comparison["arm_config_sha256"].get(arm)
            if isinstance(comparison.get("arm_config_sha256"), Mapping)
            else None
        ),
        "data_contract_sha256": metadata.get("data_contract_sha256"),
        "trainable_contract_sha256": metadata.get("trainable_contract_sha256"),
        "git_ref": sections["git"].get("ref"),
    }


def _load_provenance_distributed(raw_config: Mapping[str, Any]) -> PerceptionProvenanceManifest:
    data = raw_config["data"]
    path = data.get("perception_provenance_path")
    expected = data.get("perception_provenance_sha256")
    if not isinstance(path, str) or not _is_sha256(expected):
        raise ValueError("Checkpoint lacks pinned perception provenance")
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            manifest = load_perception_provenance_manifest(path, expected_sha256=expected)
            manifest.validate_image_path_signatures()
            load_perception_provenance_manifest(
                path,
                expected_sha256=expected,
                load_image_path_signatures=False,
            )
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Perception provenance snapshot validation failed: {detail}")
    return load_perception_provenance_manifest(
        path,
        expected_sha256=expected,
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )


def _source_audit_identity(
    raw_config: Mapping[str, Any], manifest: PerceptionProvenanceManifest
) -> dict[str, Any]:
    data = raw_config["data"]
    path_value = data.get("source_audit_path")
    expected = data.get("source_audit_fingerprint")
    if not isinstance(path_value, str) or not _is_sha256(expected):
        raise ValueError("Checkpoint lacks a pinned perception source audit")
    path = Path(path_value).expanduser().resolve()
    audit, raw_sha256 = _load_json_bytes(path, name="perception source audit")
    if not isinstance(audit, Mapping):
        raise TypeError("Perception source audit must be an object")
    unsigned = dict(audit)
    recorded = unsigned.pop("fingerprint", None)
    computed = _canonical_sha256(unsigned)
    provenance = audit.get("image_provenance")
    if (
        recorded != expected
        or computed != expected
        or audit.get("format") != "vision_alignment_perception_source_audit"
        or audit.get("version") != 2
        or audit.get("status") != "ok"
        or audit.get("phase") != "perception"
        or audit.get("failures") != []
        or audit.get("source_registry_sha256")
        != vision_alignment_perception_source_registry_sha256()
        or audit.get("source_implementation_inventory")
        != vision_alignment_perception_implementation_inventory()
        or not isinstance(provenance, Mapping)
        or provenance.get("path") != str(manifest.path)
        or provenance.get("sha256") != manifest.raw_sha256
        or provenance.get("content_sha256") != manifest.content_sha256
        or provenance.get("source_spec_sha256") != manifest.source_spec_sha256
        or set(audit.get("sources", {})) != set(PERCEPTION_SOURCE_NAMES)
    ):
        raise ValueError("Perception source-audit identity, provenance, or status differs")
    return {
        "path": str(path),
        "raw_sha256": raw_sha256,
        "fingerprint": computed,
        "source_registry_sha256": audit["source_registry_sha256"],
    }


def _validate_image_content_distributed(datasets: Mapping[str, Any]) -> dict[str, str]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {
                "ok": True,
                "identities": {
                    source: datasets[source].validate_image_content()
                    for source in PERCEPTION_SOURCE_NAMES
                },
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Perception validation image-content check failed: {detail}")
    identities = result.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != set(PERCEPTION_SOURCE_NAMES):
        raise RuntimeError("Perception image-content identity broadcast is malformed")
    return {source: str(identities[source]) for source in PERCEPTION_SOURCE_NAMES}


def _artifact_preflight_distributed(
    *,
    output: Path,
    pairing_paths: Mapping[str, Path],
    pairing_pins: Mapping[str, str],
    pairing_only: bool,
) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite immutable output {output}")
            for source in PERCEPTION_SOURCE_NAMES:
                path = pairing_paths[source]
                expected = pairing_pins.get(source)
                if path.exists() and expected is None:
                    raise ValueError(f"Existing {source} pairing requires an exact SHA-256 pin")
                if path.exists() and bridge._sha256_file(path) != expected:
                    raise ValueError(f"Existing {source} pairing differs from its SHA-256 pin")
                if not pairing_only and (not path.is_file() or expected is None):
                    raise ValueError(
                        "Promotion-grade evaluation requires all eight existing, SHA-pinned "
                        f"pairings; missing {source}"
                    )
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Artifact preflight failed: {detail}")


def _load_pairings_distributed(
    datasets: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    pins: Mapping[str, str],
    examples: int,
    seed: int,
    content_ids: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            payloads: dict[str, Mapping[str, Any]] = {}
            metadata: dict[str, dict[str, Any]] = {}
            for source in PERCEPTION_SOURCE_NAMES:
                payload, raw_sha256 = _load_json_bytes(
                    paths[source],
                    expected_sha256=pins[source],
                    name=f"{source} pairing",
                )
                validate_matched_wrong_image_pairing(
                    payload,
                    dataset_size=len(datasets[source]),
                    recipient_count=examples,
                    seed=seed,
                    epoch=0,
                    content_ids_sha256=bridge._content_ids_sha256(content_ids[source]),
                )
                digest = matched_wrong_image_pairing_sha256(payload)
                if digest != pins[source] or raw_sha256 != digest:
                    raise ValueError(f"Pinned {source} pairing identity differs")
                payloads[source] = payload
                metadata[source] = _pairing_metadata(
                    payload, path=paths[source], digest=digest, provenance="loaded"
                )
            packet[0] = {"ok": True, "payloads": payloads, "metadata": metadata}
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not load pinned perception pairings: {detail}")
    return dict(result["payloads"]), {
        source: dict(result["metadata"][source]) for source in PERCEPTION_SOURCE_NAMES
    }


def _pairing_metadata(
    payload: Mapping[str, Any], *, path: Path, digest: str, provenance: str
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": digest,
        "expected_sha256": digest,
        "provenance": provenance,
        "population": "matched_eligible_validation_subset",
        "pairing_schema_version": payload["version"],
        "coverage": payload["coverage"],
        "recipient_indices_sha256": _canonical_sha256(
            [pair["recipient"] for pair in payload["pairs"]]
        ),
        "donor_indices_sha256": _canonical_sha256([pair["donor"] for pair in payload["pairs"]]),
    }


def _prepare_pairings_distributed(
    datasets: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    pins: Mapping[str, str],
    maximum_examples: int,
    seed: int,
    content_ids: Mapping[str, Sequence[str]],
) -> tuple[int, dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    """Build the largest common count at or below the requested maximum."""
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            candidate = maximum_examples - maximum_examples % GLOBAL_BATCH_INSTANCES
            if candidate <= 0:
                raise ValueError("--examples is smaller than one 32-instance global batch")
            loaded: dict[str, Mapping[str, Any]] = {}
            loaded_counts: set[int] = set()
            for source in PERCEPTION_SOURCE_NAMES:
                if not paths[source].exists():
                    continue
                payload, raw_sha256 = _load_json_bytes(
                    paths[source],
                    expected_sha256=pins.get(source),
                    name=f"{source} pairing",
                )
                validate_matched_wrong_image_pairing(
                    payload,
                    dataset_size=len(datasets[source]),
                    seed=seed,
                    epoch=0,
                    content_ids_sha256=bridge._content_ids_sha256(content_ids[source]),
                )
                digest = matched_wrong_image_pairing_sha256(payload)
                if pins.get(source) != digest or raw_sha256 != digest:
                    raise ValueError(f"Existing {source} pairing differs from its exact pin")
                loaded[source] = payload
                loaded_counts.add(int(payload["recipient_count"]))
            if len(loaded_counts) > 1:
                raise ValueError("Existing perception pairings use different recipient counts")
            if loaded_counts:
                candidate = loaded_counts.pop()
                if candidate > maximum_examples or candidate % GLOBAL_BATCH_INSTANCES:
                    raise ValueError("Existing pairing count is outside the requested batch bound")

            while True:
                payloads: dict[str, Mapping[str, Any]] = dict(loaded)
                limit: int | None = None
                for source in PERCEPTION_SOURCE_NAMES:
                    if source in payloads:
                        continue
                    try:
                        payloads[source] = build_matched_wrong_image_pairing(
                            datasets[source],
                            recipient_count=candidate,
                            seed=seed,
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
                    break
                if loaded:
                    raise ValueError(
                        "A pinned pairing exceeds another source's eligibility; use a fresh "
                        "pairing directory"
                    )
                if limit <= 0 or limit >= candidate:
                    raise ValueError("Could not derive a smaller positive common pairing count")
                candidate = limit

            metadata: dict[str, dict[str, Any]] = {}
            digests: dict[str, str] = {}
            for source in PERCEPTION_SOURCE_NAMES:
                payload = payloads[source]
                validate_matched_wrong_image_pairing(
                    payload,
                    dataset_size=len(datasets[source]),
                    recipient_count=candidate,
                    seed=seed,
                    epoch=0,
                    content_ids_sha256=bridge._content_ids_sha256(content_ids[source]),
                )
                digest = matched_wrong_image_pairing_sha256(payload)
                if source not in loaded:
                    expected = pins.get(source)
                    if expected is not None and expected != digest:
                        raise ValueError(
                            f"New {source} pairing differs from its supplied SHA-256 pin"
                        )
                digests[source] = digest

            # Validate the entire proposed immutable set before publishing any member. This
            # prevents a late-source pin mismatch from leaving a partially materialized set.
            for source in PERCEPTION_SOURCE_NAMES:
                payload = payloads[source]
                digest = digests[source]
                provenance = "loaded" if source in loaded else "built"
                if source not in loaded:
                    bridge._write_bytes_atomic(
                        paths[source], serialize_matched_wrong_image_pairing(payload)
                    )
                if bridge._sha256_file(paths[source]) != digest:
                    raise RuntimeError(f"Published {source} pairing bytes differ")
                metadata[source] = _pairing_metadata(
                    payload,
                    path=paths[source],
                    digest=digest,
                    provenance=provenance,
                )
            packet[0] = {
                "ok": True,
                "examples": candidate,
                "payloads": payloads,
                "metadata": metadata,
            }
        except Exception as error:  # noqa: BLE001
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Perception pairing preparation failed: {detail}")
    return (
        int(result["examples"]),
        dict(result["payloads"]),
        {source: dict(result["metadata"][source]) for source in PERCEPTION_SOURCE_NAMES},
    )


def _validation_identity(
    manifest: PerceptionProvenanceManifest,
    *,
    image_content: Mapping[str, str],
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "perception_union_provenance_v2",
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.raw_sha256,
        "content_sha256": manifest.content_sha256,
        "source_spec_sha256": manifest.source_spec_sha256,
        "validation_union_disjoint_from_train": True,
        "validation_examples_per_source": 512,
        "source_runtime_identities": {
            source: {
                "runtime_dataset_fingerprint": manifest.selection(
                    source, "validation"
                ).runtime_dataset_fingerprint,
                "selection_indices_sha256": manifest.selection(
                    source, "validation"
                ).selection_indices_sha256,
                "row_image_content_sha256": bridge._content_ids_sha256(
                    manifest.selection(source, "validation").row_image_content_sha256
                ),
                "live_image_validation_sha256": image_content[source],
            }
            for source in PERCEPTION_SOURCE_NAMES
        },
        "source_audit": dict(source_audit),
    }


def _initialize_runtime(*, pairing_only: bool) -> dict[str, Any]:
    """Initialize CPU Gloo for pairing prep or the ordinary CUDA training environment."""
    state: dict[str, Any] = {
        "prepared_training_environment": False,
        "created_pairing_process_group": False,
        "rendezvous": None,
    }
    if pairing_only:
        if dist.is_initialized():
            if dist.get_world_size() != 1 or dist.get_backend() != "gloo":
                raise ValueError("Pairing-only mode requires a single-rank Gloo process group")
        else:
            rendezvous = tempfile.TemporaryDirectory(prefix="perception-pairing-gloo-")
            init_path = Path(rendezvous.name) / "rendezvous"
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{init_path}",
                rank=0,
                world_size=1,
            )
            state["rendezvous"] = rendezvous
            state["created_pairing_process_group"] = True
    else:
        prepare_training_environment()
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


def _write_pairing_manifest_distributed(
    output: Path,
    *,
    maximum_examples: int,
    examples: int,
    seed: int,
    pairings: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    profile_pair: Mapping[str, Any],
) -> None:
    producer = Path(__file__).resolve()
    bridge_file = getattr(bridge, "__file__", None)
    if not isinstance(bridge_file, str):
        raise TypeError("Could not resolve immutable bridge evaluator source")
    bridge_path = Path(bridge_file).resolve()
    implementation_path_value = inspect.getsourcefile(build_matched_wrong_image_pairing)
    if implementation_path_value is None:
        raise RuntimeError("Could not resolve pairing implementation source")
    implementation_path = Path(implementation_path_value).resolve()
    payload: dict[str, Any] = {
        "format": PAIRING_MANIFEST_FORMAT,
        "version": PAIRING_MANIFEST_VERSION,
        "status": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"path": str(producer), "sha256": bridge._sha256_file(producer)},
        "bridge_helper": {
            "path": str(bridge_path),
            "sha256": bridge._sha256_file(bridge_path),
        },
        "pairing_implementation": {
            "path": str(implementation_path),
            "sha256": bridge._sha256_file(implementation_path),
        },
        "profile_pair": dict(profile_pair),
        "validation": dict(validation),
        "protocol": {
            "selection": "largest-common-matched-eligible-count-at-or-below-request-v1",
            "maximum_requested_examples": maximum_examples,
            "examples_per_source": examples,
            "global_batch_instances": GLOBAL_BATCH_INSTANCES,
            "pairing_seed": seed,
            "sources": list(PERCEPTION_SOURCE_NAMES),
            "source_registry_sha256": vision_alignment_perception_source_registry_sha256(),
        },
        "pairings": {source: dict(pairings[source]) for source in PERCEPTION_SOURCE_NAMES},
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    bridge._write_result_distributed(output, payload, overwrite=False)


def main(argv: Sequence[str] | None = None) -> None:
    """Run pairing preparation or one native EP8 perception evaluation."""
    args = _parser().parse_args(argv)
    _validate_args(args)
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runtime_state = _initialize_runtime(pairing_only=args.pairing_only)
    try:
        checkpoint = _direct_existing_path(Path(args.checkpoint), name="checkpoint")
        config_path = _direct_existing_path(
            Path(args.config).expanduser()
            if args.config is not None
            else bridge._checkpoint_root(checkpoint) / "config.json",
            name="checkpoint config",
        )
        raw_config, _ = _load_json_bytes(
            config_path,
            expected_sha256=args.expected_config_sha256,
            name="checkpoint config",
        )
        if not isinstance(raw_config, Mapping):
            raise TypeError("Checkpoint config must be an object")
        profile_pair = _profile_pair_identity(
            raw_config,
            checkpoint=checkpoint,
            config_path=config_path,
            expected_config_sha256=args.expected_config_sha256,
            receipt_path=Path(args.profile_pair_receipt).expanduser().resolve(),
            expected_receipt_sha256=args.expected_profile_pair_receipt_sha256,
        )
        evaluation = raw_config["evaluation"]
        data = raw_config["data"]
        artifacts = raw_config["artifacts"]
        sequence_length = int(data["sequence_length"])
        rank_batch_instances = (
            args.rank_batch_instances
            if args.rank_batch_instances is not None
            else int(evaluation["rank_batch_instances"])
        )
        if rank_batch_instances != 4 or sequence_length != 2560:
            raise ValueError("Perception evaluation requires rank batch 4 and sequence length 2560")
        pairing_seed = (
            args.pairing_seed if args.pairing_seed is not None else int(evaluation["seed"])
        )
        bootstrap_seed = (
            args.bootstrap_seed if args.bootstrap_seed is not None else pairing_seed + 1_000_003
        )
        output = (
            Path(args.pairing_manifest_output if args.pairing_only else args.output)
            .expanduser()
            .resolve()
        )
        pairing_paths = _pairing_paths(args, output)
        pairing_pins = _pairing_pins(args)
        _artifact_preflight_distributed(
            output=output,
            pairing_paths=pairing_paths,
            pairing_pins=pairing_pins,
            pairing_only=args.pairing_only,
        )

        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=artifacts["tokenizer_id"],
            revision=artifacts["tokenizer_revision"],
            expected_fingerprint=artifacts["tokenizer_fingerprint"],
            cache_dir=artifacts["hf_cache_dir"],
        )
        if int(raw_config["model"]["image_patch_token_id"]) != token_ids.im_patch_id:
            raise ValueError("Pinned tokenizer image-patch ID differs from checkpoint")
        if tokenizer.pad_token_id is None:
            raise ValueError("Pinned tokenizer has no pad token")
        manifest = _load_provenance_distributed(raw_config)
        if manifest.source_spec.pixmo_cap_path != str(
            Path(str(data["pixmo_cap_path"])).expanduser().resolve()
        ):
            raise ValueError("Perception provenance PixMoCap path differs from checkpoint")
        datasets = {
            source: _PairingModelInputDataset(
                build_selected_perception_dataset(
                    manifest,
                    tokenizer,
                    token_ids,
                    source,
                    logical_split="validation",
                    validate_required_annotations=True,
                    verify_finevision_materialization=False,
                )
            )
            for source in PERCEPTION_SOURCE_NAMES
        }
        if any(len(dataset) != 512 for dataset in datasets.values()):
            raise ValueError("Perception provenance must expose 512 held-out rows per source")
        content_ids = {
            source: manifest.selection(source, "validation").row_image_content_sha256
            for source in PERCEPTION_SOURCE_NAMES
        }
        image_content = _validate_image_content_distributed(datasets)
        source_audit = _source_audit_identity(raw_config, manifest)
        validation = _validation_identity(
            manifest,
            image_content=image_content,
            source_audit=source_audit,
        )

        if args.pairing_only:
            examples, _, pairings = _prepare_pairings_distributed(
                datasets,
                paths=pairing_paths,
                pins=pairing_pins,
                maximum_examples=args.examples,
                seed=pairing_seed,
                content_ids=content_ids,
            )
            _write_pairing_manifest_distributed(
                output,
                maximum_examples=args.examples,
                examples=examples,
                seed=pairing_seed,
                pairings=pairings,
                validation=validation,
                profile_pair=profile_pair,
            )
            return

        if args.examples % GLOBAL_BATCH_INSTANCES:
            raise ValueError("--examples must be a positive multiple of the global batch (32)")
        pairing_payloads, pairings = _load_pairings_distributed(
            datasets,
            paths=pairing_paths,
            pins=pairing_pins,
            examples=args.examples,
            seed=pairing_seed,
            content_ids=content_ids,
        )
        checkpoint_identity = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            hash_workers=args.checkpoint_hash_workers,
        )
        if checkpoint_identity["config_sha256"] != args.expected_config_sha256:
            raise RuntimeError("Checkpoint inventory does not bind the expected config")
        work_dir = (
            Path(args.work_dir).expanduser().resolve()
            if args.work_dir
            else Path(os.environ.get("RESULTS_DIR", "/tmp"))
            / "vision-alignment-perception-matched-wrong"
        )
        model, module_config = bridge._build_model_and_module(
            raw_config,
            sequence_length=sequence_length,
            rank_batch_instances=rank_batch_instances,
        )
        train_module = module_config.build(model, eval_only=True)
        snapshot_state_dir = _materialize_checkpoint_snapshot_distributed(
            checkpoint_identity,
            base_dir=work_dir / "checkpoint-snapshots",
        )
        try:
            native_load = bridge._native_checkpoint_load_coverage_distributed(
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
        native_load["load_completed"] = True
        native_load["sha256"] = _canonical_sha256(
            {key: value for key, value in native_load.items() if key != "sha256"}
        )
        dp_world_size = get_world_size(train_module.dp_process_group)
        dp_rank = get_rank(train_module.dp_process_group)
        if dp_world_size != WORLD_SIZE:
            raise ValueError("Perception outcome evaluation requires DP group size 8")

        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=sequence_length,
        )
        results: dict[str, Any] = {}
        for source_index, source in enumerate(PERCEPTION_SOURCE_NAMES):
            result = bridge._evaluate_source(
                train_module,
                datasets[source],
                source_name=source,
                pairing=pairing_payloads[source],
                pairing_sha256=pairings[source]["sha256"],
                collator=collator,
                work_dir=work_dir,
                sequence_length=sequence_length,
                rank_batch_instances=rank_batch_instances,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                bootstrap_seed=bootstrap_seed + source_index * 1_000_000,
                bootstrap_samples=args.bootstrap_samples,
            )
            result["population"] = "matched_eligible_validation_subset"
            result["coverage"] = pairing_payloads[source]["coverage"]
            results[source] = result

        post_pairing_payloads, post_pairings = _load_pairings_distributed(
            datasets,
            paths=pairing_paths,
            pins=pairing_pins,
            examples=args.examples,
            seed=pairing_seed,
            content_ids=content_ids,
        )
        if _canonical_sha256(post_pairing_payloads) != _canonical_sha256(
            pairing_payloads
        ) or _canonical_sha256(post_pairings) != _canonical_sha256(pairings):
            raise RuntimeError("Pinned pairings changed during perception evaluation")
        post_checkpoint_identity = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            hash_workers=args.checkpoint_hash_workers,
        )
        if post_checkpoint_identity != checkpoint_identity:
            raise RuntimeError("Checkpoint identity changed during perception evaluation")
        post_image_content = _validate_image_content_distributed(datasets)
        if post_image_content != image_content:
            raise RuntimeError("Perception validation image bytes changed during evaluation")
        post_manifest = _load_provenance_distributed(raw_config)
        if (
            post_manifest.raw_sha256 != manifest.raw_sha256
            or post_manifest.content_sha256 != manifest.content_sha256
            or post_manifest.source_spec_sha256 != manifest.source_spec_sha256
            or _source_audit_identity(raw_config, post_manifest) != source_audit
        ):
            raise RuntimeError("Perception provenance or source audit changed during evaluation")

        protocol: dict[str, Any] = {
            "name": PROTOCOL_NAME,
            "sources": list(PERCEPTION_SOURCE_NAMES),
            "dataset_split": "validation",
            "evaluation_population": "matched_eligible_validation_subset",
            "examples_per_source": args.examples,
            "source_epoch": 0,
            "pairing_seed": pairing_seed,
            "pairing_sha256": {
                source: pairings[source]["sha256"] for source in PERCEPTION_SOURCE_NAMES
            },
            "pairing_pin_policy": "all eight existing pairing files require exact CLI pins",
            "pairing_rule": (
                "distinct pinned content and materialized pixels; exact image tensor shape and "
                "byte-identical pooled_patches_idx; explicit unique donors"
            ),
            "recipient_replay": "correct and wrong forwards use exactly the same recipients",
            "response_logits": "only positive-loss-mask positions are materialized",
            "per_example_ce": (
                "loss-mask-weighted mean over all or first min(K,N) supervised response tokens"
            ),
            "gap_sign": "wrong_ce - correct_ce; positive is a correct-image win",
            "bootstrap": {
                "method": "deterministic iid example bootstrap percentile interval",
                "confidence": 0.95,
                "samples": args.bootstrap_samples,
                "seed": bootstrap_seed,
            },
            "windows": {name: limit for name, limit in bridge.WINDOWS},
            "message_format": data["message_format"],
            "loss_token_weighting": data["loss_token_weighting"],
            "sequence_length": sequence_length,
            "rank_batch_instances": rank_batch_instances,
            "global_batch_instances": GLOBAL_BATCH_INSTANCES,
            "world_size": WORLD_SIZE,
            "ep_degree": EP_DEGREE,
            "dp_process_group_size": dp_world_size,
            "source_registry_sha256": vision_alignment_perception_source_registry_sha256(),
            "profile_pair_receipt_sha256": profile_pair["sha256"],
            "perception_provenance_sha256": validation["manifest_sha256"],
            "source_audit_fingerprint": source_audit["fingerprint"],
            "tokenizer": {
                "id": artifacts["tokenizer_id"],
                "revision": artifacts["tokenizer_revision"],
                "fingerprint": artifacts["tokenizer_fingerprint"],
                "token_ids": token_ids.as_config_dict(),
            },
        }
        protocol["sha256"] = _canonical_sha256(protocol)
        producer = Path(__file__).resolve()
        bridge_file = getattr(bridge, "__file__", None)
        if not isinstance(bridge_file, str):
            raise TypeError("Could not resolve immutable bridge evaluator source")
        bridge_path = Path(bridge_file).resolve()
        pairing_impl_value = inspect.getsourcefile(build_matched_wrong_image_pairing)
        if pairing_impl_value is None:
            raise RuntimeError("Could not resolve pairing implementation source")
        pairing_impl = Path(pairing_impl_value).resolve()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_identity,
            "native_checkpoint_load": native_load,
            "config_path": str(config_path),
            "git": bridge._git_identity(),
            "evaluator": {
                "path": str(producer),
                "sha256": bridge._sha256_file(producer),
                "bridge_helper_path": str(bridge_path),
                "bridge_helper_sha256": bridge._sha256_file(bridge_path),
                "pairing_implementation_path": str(pairing_impl),
                "pairing_implementation_sha256": bridge._sha256_file(pairing_impl),
            },
            "profile_pair": profile_pair,
            "validation": validation,
            "pairings": pairings,
            "artifact_policy": {
                "all_pairings_require_sha256_pins": True,
                "expected_pairing_sha256": pairing_pins,
                "output_overwrite_enabled": False,
            },
            "protocol": protocol,
            "results": results,
        }
        payload["config_and_protocol_sha256"] = _canonical_sha256(
            {
                "checkpoint_config_sha256": checkpoint_identity["config_sha256"],
                "protocol_sha256": protocol["sha256"],
                "pairing_sha256": protocol["pairing_sha256"],
            }
        )
        bridge._write_result_distributed(output, payload, overwrite=False)
    finally:
        _teardown_runtime(runtime_state)


if __name__ == "__main__":
    main()
