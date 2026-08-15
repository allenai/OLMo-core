"""Compare exact joint step-4000/8000 matched-wrong evaluation receipts on CPU.

The comparator does not load model weights. It re-opens two immutable evaluator receipts and
their shared pairing manifest, verifies every referenced pairing, joins the exact same recipient
rows across checkpoints, and rederives all reported statistics from per-example values.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from olmo_core.eval.matched_wrong_image import (
    matched_wrong_image_pairing_sha256,
    validate_matched_wrong_image_pairing,
)

FORMAT = "vision_alignment_joint_matched_wrong_comparison"
VERSION = 1
EVALUATOR_FORMAT = "vision_alignment_joint_matched_wrong_receipt"
EVALUATOR_VERSION = 1
PAIRING_MANIFEST_FORMAT = "vision_alignment_joint_matched_wrong_pairing_manifest"
PAIRING_MANIFEST_VERSION = 1
EVALUATOR_PROTOCOL_NAME = "vision-alignment-joint-native-matched-wrong-v1"
PROTOCOL_NAME = "vision-alignment-joint-step4000-step8000-paired-comparison-v1"
EXPECTED_CONFIG_SHA256 = "64b302865831b5aaf11e86e142a85b3467a06b93d6c214fb67f7f94a45c4ddc8"
EXPECTED_PROJECTION_SHA256 = "11c1df56d7fbc270a9eff999193476c0c578c6964017d217a320b3d39305a730"
EXPECTED_SOURCE_AUDIT_FINGERPRINT = (
    "434ea76205bca8361f3291d90665af1cd36713ef238b18b1c450ff33ceab4b14"
)
EXPECTED_NATIVE_TRAIN_FINGERPRINT = (
    "9e4396179f003d858126da9c631622622c175635ac05f7db6c277de2eac3dc26"
)
EXPECTED_NATIVE_HOLDOUT_FINGERPRINT = (
    "6418aa4e1c1652ff4a9c504a9eed883fd5d346bdbccbda3ceae2575da29a2766"
)
EXPECTED_NATIVE_VERIFICATION_SHA256 = (
    "cc94a2387059a83221075cd48b74ad9703e9763eed800a0bdf3445221793f62d"
)
EXPECTED_REVIEWED_PROFILE = "configs/vision_moe/vision_alignment/joint/joint_v1.yaml"
EXPECTED_REVIEWED_PROFILE_SHA256 = (
    "294da420f4f911fc96aad2a9eff43c59dc0831276fad5d1c0fbec37c6f78c2f5"
)
EXPECTED_REVIEWED_PROFILE_ALLOWLIST = (
    "configs/vision_moe/vision_alignment/joint/approved_profiles.json"
)
EXPECTED_REVIEWED_PROFILE_ALLOWLIST_SHA256 = (
    "5373f9c0c6dff3430b9632c39af523003e96032f741da7c1ed4f72006fe65fe5"
)
EXPECTED_TRAINING_GIT_REF = "7e42a7e3064bd944806a5cf5d351ec4f6dc24e42"
EXPECTED_TRAINING_BEAKER_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
PAIRING_SEED = 6198
STEPS = (4000, 8000)
STEP_KEYS = ("step4000", "step8000")
SOURCE_NAMES = (
    "audited_alignment",
    "cosyn_point",
    "count_numeric",
    "ocr_document",
    "pixmo_caption",
    "pixmo_points_basic",
    "pixmo_points_high_frequency",
    "pixmo_transcript",
)
BLANK_SOURCE_NAMES = ("pixmo_caption", "pixmo_transcript")
NATIVE_EXAMPLES = 1_000
NATIVE_FILTERED_INDICES = (334, 478, 610, 780, 792)
DEFAULT_BOOTSTRAP_SEED = 8_208_2026
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD = "deterministic paired source-stratified example bootstrap percentile interval"

RECEIPT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "git",
        "artifact_policy",
        "checkpoint",
        "checkpoint_config",
        "load_coverage",
        "projection",
        "source_audit",
        "tokenizer",
        "pairing_manifest",
        "protocol",
        "visual_results",
        "blank_results",
        "native_result",
        "content_sha256",
    }
)
PAIRING_MANIFEST_FIELDS = frozenset(
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
CHECKPOINT_CONFIG_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "phase",
        "lineage_id",
        "run_name",
        "step",
        "reviewed_profile_path",
        "reviewed_profile_sha256",
        "reviewed_profile_allowlist_path",
        "reviewed_profile_allowlist_sha256",
        "training_git_ref",
        "training_beaker_image",
    }
)
PAIRING_MANIFEST_REF_FIELDS = frozenset({"path", "sha256", "content_sha256"})
PAIRING_ENTRY_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "canonical_sha256",
        "pairing_schema_version",
        "population",
        "coverage",
        "recipient_indices_sha256",
        "donor_indices_sha256",
    }
)
VISUAL_RESULT_FIELDS = frozenset(
    {
        "pairing_sha256",
        "examples",
        "elapsed_seconds",
        "metrics",
        "per_example",
        "population",
        "coverage",
    }
)
VISUAL_ROW_FIELDS = frozenset(
    {
        "pairing_position",
        "recipient_index",
        "donor_index",
        "response_tokens",
        "loss_weight",
        "correct_ce",
        "wrong_ce",
        "ce_gap_wrong_minus_correct",
    }
)
BLANK_RESULT_FIELDS = frozenset(
    {
        "pairing_sha256",
        "examples",
        "elapsed_seconds",
        "population",
        "coverage",
        "metrics",
        "per_example",
    }
)
BLANK_ROW_FIELDS = frozenset(
    {
        "pairing_position",
        "recipient_index",
        "response_tokens",
        "loss_weight",
        "correct_ce",
        "blank_ce",
        "ce_gap_blank_minus_correct",
    }
)
NATIVE_RESULT_FIELDS = frozenset(
    {
        "examples",
        "elapsed_seconds",
        "dataset_order_sha256",
        "row_provenance_sha256",
        "native_identity_sha256",
        "metrics",
        "per_example",
    }
)
NATIVE_ROW_FIELDS = frozenset(
    {
        "evaluation_position",
        "dataset_index",
        "provenance",
        "mask_tokens",
        "labeled_tokens",
        "mask_loss_weight",
        "labeled_loss_weight",
        "summed_ce",
        "filtered",
        "ce",
    }
)
OUTPUT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "created_at",
        "producer",
        "inputs",
        "policy",
        "protocol",
        "shared_inputs",
        "visual",
        "blank",
        "native",
        "count_guard",
        "correlation_disclosure",
        "content_sha256",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for step in STEPS:
        parser.add_argument(f"--step{step}", required=True)
        parser.add_argument(f"--expected-step{step}-sha256", required=True)
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


def _direct_existing_path(path: Path, *, name: str) -> Path:
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


def _safe_output_path(path: Path, *, name: str) -> Path:
    """Return a lexical absolute output path while rejecting existing symlink components."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (*reversed(absolute.parents), absolute):
        if component == Path(component.anchor):
            continue
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"{name} component is unavailable: {component}: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} contains a symlinked component: {component}")
    return absolute


def _read_regular_file(path: Path, *, name: str) -> tuple[bytes, str, Path]:
    path = _direct_existing_path(path, name=name)
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
    except OSError as error:
        raise ValueError(f"Could not read {name} from {path}: {error}") from error
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
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest(), path


def _hash_regular_file(path: Path, *, name: str) -> tuple[int, str, Path]:
    """Stream one regular file once, retaining no file-sized byte buffer."""
    path = _direct_existing_path(path, name=name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not a regular file")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"Could not hash {name} from {path}: {error}") from error
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
    if (
        signature(before) != signature(after)
        or signature(before) != signature(current)
        or size != before.st_size
    ):
        raise ValueError(f"{name} changed while it was hashed")
    return size, digest.hexdigest(), path


def _sha256_file(path: Path) -> str:
    return _hash_regular_file(path, name=str(path))[1]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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
    def reject_constant(value: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        raw, digest, _ = _read_regular_file(path, name=name)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"{name} SHA-256 differs: expected {expected_sha256}, got {digest}")
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
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


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _timestamp(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value


def _validate_content_sha256(value: Mapping[str, Any], *, name: str) -> None:
    digest = value.get("content_sha256")
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    if not _is_sha256(digest) or digest != _canonical_sha256(unsigned):
        raise ValueError(f"{name} content SHA-256 differs")


def _artifact_ref(path_value: str | Path, expected_sha256: str, *, name: str) -> dict[str, str]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{name} expected SHA-256 must be lowercase hex")
    path = _direct_existing_path(Path(path_value), name=name)
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} bytes differ from their exact SHA-256 pin")
    return {"path": str(path), "sha256": expected_sha256}


def _read_ref(value: Any, *, name: str) -> tuple[dict[str, str], Mapping[str, Any]]:
    reference = _exact(value, frozenset({"path", "sha256"}), name=f"{name} reference")
    digest = reference["sha256"]
    if not _is_sha256(digest):
        raise ValueError(f"{name} reference SHA-256 is invalid")
    path = _direct_existing_path(Path(str(reference["path"])), name=name)
    payload, _ = _load_json_bytes(path, expected_sha256=digest, name=name)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return {"path": str(path), "sha256": str(digest)}, payload


def _validate_file_ref(value: Any, *, name: str) -> dict[str, str]:
    reference = _exact(value, frozenset({"path", "sha256"}), name=name)
    return _artifact_ref(reference["path"], reference["sha256"], name=name)


def _validate_implementation_ref(value: Any, *, live_path: Path, name: str) -> dict[str, str]:
    reference = _exact(value, frozenset({"path", "sha256"}), name=name)
    if (
        Path(str(reference["path"])).name != live_path.name
        or not _is_sha256(reference["sha256"])
        or reference["sha256"] != _sha256_file(live_path)
    ):
        raise ValueError(f"{name} does not match the live reviewed implementation")
    return {"path": str(reference["path"]), "sha256": str(reference["sha256"])}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical scientific output atomically without replacing existing bytes."""
    path = _safe_output_path(path, name="comparison output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _safe_output_path(path, name="comparison output")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary_created = False
    temporary_identity: tuple[int, int] | None = None
    try:
        with temporary.open("xb") as file_handle:
            temporary_created = True
            temporary_info = os.fstat(file_handle.fileno())
            temporary_identity = (temporary_info.st_dev, temporary_info.st_ino)
            file_handle.write(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable comparison {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_created and temporary_identity is not None:
            try:
                current = temporary.lstat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == temporary_identity:
                    temporary.unlink()


def _sha256_indices(values: Sequence[int]) -> str:
    return _canonical_sha256(list(values))


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _validate_load_coverage(value: Any, *, name: str) -> Mapping[str, Any]:
    fields = frozenset(
        {
            "checkpoint_key_count",
            "complete",
            "eval_state_key_count",
            "frozen_state_key_count",
            "load_completed",
            "model_parameter_assignments_sha256",
            "model_parameter_checkpoint_key_count",
            "model_parameter_checkpoint_keys_sha256",
            "model_parameter_count",
            "persistent_buffer_count",
            "persistent_buffer_keys_sha256",
            "prepared_load_key_count",
            "sha256",
            "shadowed_frozen_key_count",
            "shadowed_frozen_keys_sha256",
            "unused_model_bearing_key_count",
        }
    )
    coverage = _exact(value, fields, name=name)
    if coverage["complete"] is not True or coverage["load_completed"] is not True:
        raise ValueError(f"{name} is incomplete")
    count_fields = fields - {
        "complete",
        "load_completed",
        "sha256",
        "model_parameter_assignments_sha256",
        "model_parameter_checkpoint_keys_sha256",
        "persistent_buffer_keys_sha256",
        "shadowed_frozen_keys_sha256",
    }
    for field in count_fields:
        _integer(coverage[field], name=f"{name} {field}")
    for field in (
        "model_parameter_assignments_sha256",
        "model_parameter_checkpoint_keys_sha256",
        "persistent_buffer_keys_sha256",
        "shadowed_frozen_keys_sha256",
    ):
        if not _is_sha256(coverage[field]):
            raise ValueError(f"{name} {field} is invalid")
    if (
        coverage["model_parameter_count"] != coverage["model_parameter_checkpoint_key_count"]
        or coverage["unused_model_bearing_key_count"] != 0
        or coverage["prepared_load_key_count"] != coverage["model_parameter_count"]
    ):
        raise ValueError(f"{name} does not cover the complete model parameter surface")
    unsigned = dict(coverage)
    digest = unsigned.pop("sha256")
    if not _is_sha256(digest) or digest != _canonical_sha256(unsigned):
        raise ValueError(f"{name} SHA-256 differs")
    return coverage


def _validate_inventory(
    value: Any,
    *,
    root: Path,
    directory: Path,
    name: str,
    verify_live_files: bool,
) -> list[Mapping[str, Any]]:
    inventory = _require_sequence(value, name=name)
    if not inventory:
        raise ValueError(f"{name} is empty")
    normalized: list[Mapping[str, Any]] = []
    paths: list[str] = []
    live_expectations: list[tuple[Path, int, str, str]] = []
    for position, item_value in enumerate(inventory):
        item = _exact(
            item_value,
            frozenset({"path", "size", "sha256"}),
            name=f"{name} item {position}",
        )
        relative = item["path"]
        if not isinstance(relative, str):
            raise TypeError(f"{name} item {position} path must be a string")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise ValueError(f"{name} item {position} path escapes its checkpoint")
        size = _integer(item["size"], name=f"{name} item {position} size")
        if not _is_sha256(item["sha256"]):
            raise ValueError(f"{name} item {position} SHA-256 is invalid")
        expected_path = Path(os.path.abspath(root / relative_path))
        try:
            expected_path.relative_to(directory)
        except ValueError as error:
            raise ValueError(f"{name} item {relative} is outside its expected directory") from error
        live_expectations.append((expected_path, size, str(item["sha256"]), relative))
        paths.append(relative)
        normalized.append(item)
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} paths are duplicated")
    if verify_live_files:

        def hash_expected(expectation: tuple[Path, int, str, str]) -> tuple[int, str, Path]:
            expected_path, _, _, relative = expectation
            return _hash_regular_file(expected_path, name=f"{name} item {relative}")

        with ThreadPoolExecutor(max_workers=min(8, len(live_expectations))) as executor:
            actual_records = list(executor.map(hash_expected, live_expectations))
        for expectation, actual in zip(live_expectations, actual_records, strict=True):
            expected_path, expected_size, expected_sha, relative = expectation
            actual_size, actual_sha, actual_path = actual
            if (
                actual_path != expected_path
                or actual_size != expected_size
                or actual_sha != expected_sha
            ):
                raise ValueError(f"{name} item {relative} live bytes differ")
        observed = sorted(path.relative_to(root).as_posix() for path in directory.iterdir())
        if observed != sorted(paths):
            raise ValueError(f"{name} does not bind the exact live directory entries")
    return normalized


def _validate_checkpoint_identity(
    value: Any,
    *,
    config: Mapping[str, Any],
    name: str,
    verify_live_files: bool,
) -> None:
    identity = _require_mapping(value, name=name)
    required = {
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
        "model_and_optim_identity_sha256",
        "checkpoint_step",
        "permanent",
        "checkpoint_marker",
        "trainer_state_rank_count",
        "trainer_state_file_inventory",
        "trainer_state_file_inventory_sha256",
        "trainer_state_summary",
        "trainer_state_total_data_errors_by_rank",
        "trainer_state_total_data_errors_sum",
        "identity_sha256",
    }
    if set(identity) != required:
        raise ValueError(
            f"{name} fields differ: missing={sorted(required - set(identity))}, "
            f"extra={sorted(set(identity) - required)}"
        )
    if identity["state_file_hash_algorithm"] != "sha256":
        raise ValueError(f"{name} uses an unsupported hash algorithm")
    if identity["config_sha256"] != config["sha256"]:
        raise ValueError(f"{name} config identity differs")
    for field in (
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_inventory_sha256",
        "model_and_optim_identity_sha256",
        "trainer_state_file_inventory_sha256",
        "identity_sha256",
    ):
        if not _is_sha256(identity[field]):
            raise ValueError(f"{name} {field} is invalid")
    root = _direct_existing_path(Path(str(identity["root"])), name=f"{name} root")
    state_dir = _direct_existing_path(
        Path(str(identity["state_dir"])), name=f"{name} state directory"
    )
    train_dir = _direct_existing_path(root / "train", name=f"{name} trainer-state directory")
    step = config["step"]
    if (
        root.name != f"step{step}"
        or state_dir.parent != root
        or identity["checkpoint_step"] != step
        or identity["permanent"] is not True
        or identity["checkpoint_marker"] != {"ephemeral": False, "version": "2.5.0"}
    ):
        raise ValueError(f"{name} root is not the declared step")
    if (
        _direct_existing_path(Path(str(config["path"])), name=f"{name} config")
        != root / "config.json"
    ):
        raise ValueError(f"{name} config is not rooted in the checkpoint")
    marker, marker_sha = _load_json_bytes(root / ".metadata.json", name=f"{name} marker")
    if (
        marker != identity["checkpoint_marker"]
        or marker_sha != identity["checkpoint_marker_sha256"]
    ):
        raise ValueError(f"{name} permanent marker differs")
    inventory = _validate_inventory(
        identity["state_file_inventory"],
        root=root,
        directory=state_dir,
        name=f"{name} state inventory",
        verify_live_files=verify_live_files,
    )
    if identity["state_file_inventory_sha256"] != _canonical_sha256(inventory):
        raise ValueError(f"{name} state inventory identity differs")
    trainer_inventory = _validate_inventory(
        identity["trainer_state_file_inventory"],
        root=root,
        directory=train_dir,
        name=f"{name} trainer-state inventory",
        verify_live_files=verify_live_files,
    )
    expected_trainer_paths = [f"train/rank{rank}.pt" for rank in range(16)]
    if (
        identity["trainer_state_rank_count"] != 16
        or [item["path"] for item in trainer_inventory] != expected_trainer_paths
        or identity["trainer_state_file_inventory_sha256"] != _canonical_sha256(trainer_inventory)
    ):
        raise ValueError(f"{name} trainer-state inventory differs")
    summary = _exact(
        identity["trainer_state_summary"],
        frozenset(
            {
                "global_step",
                "global_train_tokens_seen",
                "max_steps",
                "world_size",
                "batches_processed",
                "consecutive_data_errors",
                "wandb_run_id",
                "wandb_name",
            }
        ),
        name=f"{name} trainer-state summary",
    )
    if (
        summary["global_step"] != step
        or summary["global_train_tokens_seen"] != step * 1_048_576
        or summary["max_steps"] != 16_000
        or summary["world_size"] != 16
        or summary["batches_processed"] != step
        or summary["consecutive_data_errors"] != 0
        or summary["wandb_name"] != "vision-alignment-joint-v1"
        or not isinstance(summary["wandb_run_id"], str)
        or not summary["wandb_run_id"]
    ):
        raise ValueError(f"{name} trainer-state progress differs")
    errors_by_rank = identity["trainer_state_total_data_errors_by_rank"]
    expected_errors = [0] * 16
    if step == 8000:
        expected_errors[0] = 1
        expected_errors[8] = 1
    if (
        not isinstance(errors_by_rank, list)
        or len(errors_by_rank) != 16
        or any(type(count) is not int or count < 0 for count in errors_by_rank)
        or errors_by_rank != expected_errors
        or identity["trainer_state_total_data_errors_sum"] != sum(errors_by_rank)
    ):
        raise ValueError(f"{name} trainer-state data-error totals differ")
    model_identity_fields = (
        "root",
        "state_dir",
        "config_sha256",
        "checkpoint_marker_sha256",
        "dcp_metadata_sha256",
        "state_file_hash_algorithm",
        "state_file_inventory_sha256",
        "state_file_inventory",
    )
    model_identity = {field: identity[field] for field in model_identity_fields}
    if identity["model_and_optim_identity_sha256"] != _canonical_sha256(model_identity):
        raise ValueError(f"{name} model/optimizer identity differs")
    dcp_records = [
        item
        for item in inventory
        if item["path"] == (state_dir / ".metadata").relative_to(root).as_posix()
    ]
    if len(dcp_records) != 1 or dcp_records[0]["sha256"] != identity["dcp_metadata_sha256"]:
        raise ValueError(f"{name} DCP metadata identity differs")
    unsigned = dict(identity)
    digest = unsigned.pop("identity_sha256")
    if digest != _canonical_sha256(unsigned):
        raise ValueError(f"{name} identity SHA-256 differs")


def _validate_checkpoint_config(value: Any, *, step: int, name: str) -> Mapping[str, Any]:
    config = _exact(value, CHECKPOINT_CONFIG_FIELDS, name=name)
    if (
        config["phase"] != "joint"
        or config["lineage_id"] != "vision-alignment-joint-v1"
        or config["run_name"] != "vision-alignment-joint-v1"
        or config["step"] != step
        or config["sha256"] != EXPECTED_CONFIG_SHA256
        or config["reviewed_profile_path"] != EXPECTED_REVIEWED_PROFILE
        or config["reviewed_profile_sha256"] != EXPECTED_REVIEWED_PROFILE_SHA256
        or config["reviewed_profile_allowlist_path"] != EXPECTED_REVIEWED_PROFILE_ALLOWLIST
        or config["reviewed_profile_allowlist_sha256"] != EXPECTED_REVIEWED_PROFILE_ALLOWLIST_SHA256
        or config["training_git_ref"] != EXPECTED_TRAINING_GIT_REF
        or config["training_beaker_image"] != EXPECTED_TRAINING_BEAKER_IMAGE
    ):
        raise ValueError(f"{name} is not the locked joint step{step} checkpoint")
    _artifact_ref(config["path"], config["sha256"], name=f"{name} saved config")
    _artifact_ref(
        config["reviewed_profile_path"],
        config["reviewed_profile_sha256"],
        name=f"{name} reviewed profile",
    )
    _artifact_ref(
        config["reviewed_profile_allowlist_path"],
        config["reviewed_profile_allowlist_sha256"],
        name=f"{name} reviewed profile allowlist",
    )
    return config


def _validate_protocol(value: Any, *, name: str) -> Mapping[str, Any]:
    fields = frozenset(
        {
            "name",
            "descriptive_only",
            "promotion_eligible",
            "primary_statistic",
            "per_checkpoint_statistic",
            "response_logits_materialized",
            "sources",
            "blank_sources",
            "native_source",
            "visual_split",
            "visual_population",
            "examples_per_visual_source",
            "native_population",
            "native_examples",
            "native_filtered_indices",
            "pairing_seed",
            "pairing_sha256",
            "pairing_rule",
            "recipient_replay",
            "blank_rule",
            "ce_definition",
            "native_dual_denominator",
            "sequence_length",
            "rank_batch_instances",
            "global_batch_instances",
            "nodes",
            "world_size",
            "local_world_size",
            "ep_degree",
            "dp_process_group_size",
            "training_beaker_image",
            "training_git_ref",
            "checkpoint_config_sha256",
            "projection_raw_sha256",
            "source_audit_fingerprint",
            "native_holdout_fingerprint",
            "native_row_provenance_sha256",
            "native_identity",
        }
    )
    protocol = _exact(value, fields, name=name)
    examples = _integer(
        protocol["examples_per_visual_source"], name=f"{name} examples/source", minimum=1
    )
    pairing_sha = _require_mapping(protocol["pairing_sha256"], name=f"{name} pairings")
    if (
        protocol["name"] != EVALUATOR_PROTOCOL_NAME
        or protocol["descriptive_only"] is not True
        or protocol["promotion_eligible"] is not False
        or protocol["primary_statistic"]
        != "paired source-balanced change in wrong-minus-correct CE from step4000 to step8000"
        or protocol["per_checkpoint_statistic"] != "all-response loss-weighted scalar CE"
        or protocol["response_logits_materialized"] is not False
        or protocol["sources"] != list(SOURCE_NAMES)
        or protocol["blank_sources"] != list(BLANK_SOURCE_NAMES)
        or protocol["native_source"] != "native_text_replay"
        or protocol["visual_split"] != "validation"
        or protocol["visual_population"] != "matched_eligible_joint_validation_subset"
        or examples > 512
        or examples % 8 != 0
        or protocol["native_population"] != "all holdout windows in exact manifest order"
        or protocol["native_examples"] != NATIVE_EXAMPLES
        or protocol["native_filtered_indices"] != list(NATIVE_FILTERED_INDICES)
        or protocol["pairing_seed"] != PAIRING_SEED
        or set(pairing_sha) != set(SOURCE_NAMES)
        or any(not _is_sha256(pairing_sha[source]) for source in SOURCE_NAMES)
        or protocol["pairing_rule"]
        != (
            "largest common multiple-of-eight; distinct pinned image content and exact "
            "collated geometry; deterministic explicit unique donors"
        )
        or protocol["recipient_replay"]
        != "correct, wrong, and applicable blank forwards share recipients"
        or protocol["blank_rule"]
        != "zeros_like normalized image tensor; all non-image fields unchanged"
        or protocol["ce_definition"]
        != (
            "scalar summed CE divided by the one rank-local example's positive labeled loss "
            "weight; no response logits"
        )
        or protocol["native_dual_denominator"]
        != "inline CE uses labeled loss weight; training-divisor CE uses all mask loss weight"
        or protocol["sequence_length"] != 8192
        or protocol["rank_batch_instances"] != 1
        or protocol["global_batch_instances"] != 8
        or protocol["nodes"] != 1
        or protocol["world_size"] != 8
        or protocol["local_world_size"] != 8
        or protocol["ep_degree"] != 8
        or protocol["dp_process_group_size"] != 8
        or protocol["training_beaker_image"] != EXPECTED_TRAINING_BEAKER_IMAGE
        or protocol["training_git_ref"] != EXPECTED_TRAINING_GIT_REF
        or protocol["checkpoint_config_sha256"] != EXPECTED_CONFIG_SHA256
        or protocol["projection_raw_sha256"] != EXPECTED_PROJECTION_SHA256
        or protocol["source_audit_fingerprint"] != EXPECTED_SOURCE_AUDIT_FINGERPRINT
        or not _is_sha256(protocol["native_holdout_fingerprint"])
        or not _is_sha256(protocol["native_row_provenance_sha256"])
    ):
        raise ValueError(f"{name} differs from the locked all-response protocol")
    native_identity = _validate_native_identity(
        protocol["native_identity"], name=f"{name} native identity"
    )
    if (
        protocol["native_holdout_fingerprint"] != native_identity["holdout_fingerprint"]
        or protocol["native_row_provenance_sha256"] != native_identity["row_provenance_sha256"]
    ):
        raise ValueError(f"{name} native identity cross-binding differs")
    return protocol


def _validate_pairing_manifest(
    reference_value: Any,
    *,
    receipt_protocol: Mapping[str, Any],
    receipt_checkpoint_config: Mapping[str, Any],
    receipt_projection: Mapping[str, Any],
    receipt_source_audit: Mapping[str, Any],
    receipt_tokenizer: Mapping[str, Any],
    name: str,
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    reference = _exact(reference_value, PAIRING_MANIFEST_REF_FIELDS, name=f"{name} reference")
    if not _is_sha256(reference["sha256"]) or not _is_sha256(reference["content_sha256"]):
        raise ValueError(f"{name} reference SHA-256 is invalid")
    path = _direct_existing_path(Path(str(reference["path"])), name=name)
    payload, raw_sha = _load_json_bytes(path, expected_sha256=str(reference["sha256"]), name=name)
    manifest = _exact(payload, PAIRING_MANIFEST_FIELDS, name=name)
    if (
        manifest["format"] != PAIRING_MANIFEST_FORMAT
        or manifest["version"] != PAIRING_MANIFEST_VERSION
        or manifest["status"] != "prepared"
    ):
        raise ValueError(f"{name} identity or status differs")
    _timestamp(manifest["created_at"], name=f"{name} created_at")
    _validate_content_sha256(manifest, name=name)
    if manifest["content_sha256"] != reference["content_sha256"]:
        raise ValueError(f"{name} content SHA-256 differs from its receipt reference")
    if raw_sha != reference["sha256"]:
        raise AssertionError("Pairing manifest raw digest changed after loading")
    pairing_source = inspect.getsourcefile(validate_matched_wrong_image_pairing)
    if pairing_source is None:
        raise RuntimeError("Could not locate the live pairing implementation")
    live_implementations = {
        "producer": Path(__file__).with_name("vision_alignment_joint_matched_wrong.py"),
        "bridge_helper": Path(__file__).with_name("vision_alignment_matched_wrong.py"),
        "pairing_implementation": Path(pairing_source),
    }
    for field, live_path in live_implementations.items():
        _validate_implementation_ref(manifest[field], live_path=live_path, name=f"{name} {field}")
    manifest_config = _require_mapping(
        manifest["checkpoint_config"], name=f"{name} checkpoint config"
    )
    manifest_step = manifest_config.get("step")
    if manifest_step not in STEPS:
        raise ValueError(f"{name} checkpoint step is not an admissible endpoint")
    _validate_checkpoint_config(
        manifest_config, step=manifest_step, name=f"{name} checkpoint config"
    )
    shared_config_fields = set(CHECKPOINT_CONFIG_FIELDS) - {"path", "step"}
    if any(
        manifest_config[field] != receipt_checkpoint_config[field] for field in shared_config_fields
    ):
        raise ValueError(f"{name} checkpoint contract differs from its evaluator receipt")
    for field, expected in (
        ("projection", receipt_projection),
        ("source_audit", receipt_source_audit),
        ("tokenizer", receipt_tokenizer),
    ):
        if _canonical_bytes(manifest[field]) != _canonical_bytes(expected):
            raise ValueError(f"{name} {field} differs from its evaluator receipt")

    manifest_protocol = _exact(
        manifest["protocol"],
        frozenset(
            {
                "selection",
                "maximum_requested_examples",
                "examples_per_source",
                "global_batch_instances",
                "pairing_seed",
                "sources",
                "population",
                "sequence_length",
                "source_registry_sha256",
            }
        ),
        name=f"{name} protocol",
    )
    if (
        manifest_protocol["selection"]
        != "largest-common-matched-eligible-multiple-of-eight-at-most-512-v1"
        or manifest_protocol["maximum_requested_examples"] != 512
        or manifest_protocol["examples_per_source"]
        != receipt_protocol["examples_per_visual_source"]
        or manifest_protocol["global_batch_instances"] != 8
        or manifest_protocol["pairing_seed"] != PAIRING_SEED
        or manifest_protocol["sources"] != list(SOURCE_NAMES)
        or manifest_protocol["population"] != "matched_eligible_joint_validation_subset"
        or manifest_protocol["sequence_length"] != 8192
        or not _is_sha256(manifest_protocol["source_registry_sha256"])
        or manifest_protocol["source_registry_sha256"]
        != receipt_projection["visual_source_registry_sha256"]
    ):
        raise ValueError(f"{name} protocol differs")

    pairings = _require_mapping(manifest["pairings"], name=f"{name} pairings")
    if set(pairings) != set(SOURCE_NAMES):
        raise ValueError(f"{name} does not contain the exact visual source set")
    examples = _integer(
        receipt_protocol.get("examples_per_visual_source"),
        name=f"{name} examples/source",
        minimum=1,
    )
    pairing_seed = _integer(receipt_protocol.get("pairing_seed"), name=f"{name} pairing seed")
    normalized: dict[str, Mapping[str, Any]] = {}
    for source in SOURCE_NAMES:
        entry = _exact(pairings[source], PAIRING_ENTRY_FIELDS, name=f"{name} {source}")
        for field in (
            "sha256",
            "canonical_sha256",
            "recipient_indices_sha256",
            "donor_indices_sha256",
        ):
            if not _is_sha256(entry[field]):
                raise ValueError(f"{name} {source} {field} is invalid")
        pairing_path = _direct_existing_path(
            Path(str(entry["path"])), name=f"{name} {source} pairing"
        )
        pairing_payload, _ = _load_json_bytes(
            pairing_path,
            expected_sha256=str(entry["sha256"]),
            name=f"{name} {source} pairing",
        )
        pairing = _require_mapping(pairing_payload, name=f"{name} {source} pairing")
        validate_matched_wrong_image_pairing(
            pairing,
            dataset_size=512,
            recipient_count=examples,
            seed=pairing_seed,
            epoch=0,
        )
        if (
            entry["sha256"] != entry["canonical_sha256"]
            or entry["canonical_sha256"] != matched_wrong_image_pairing_sha256(pairing)
            or receipt_protocol["pairing_sha256"][source] != entry["canonical_sha256"]
            or entry["pairing_schema_version"] != pairing["version"]
            or entry["population"] != "matched_eligible_joint_validation_subset"
            or pairing["content_ids_sha256"]
            != receipt_projection["sources"][source]["row_image_content_sha256"]
            or _canonical_bytes(entry["coverage"]) != _canonical_bytes(pairing["coverage"])
        ):
            raise ValueError(f"{name} {source} pairing metadata differs from its bytes")
        recipients = [pair["recipient"] for pair in pairing["pairs"]]
        donors = [pair["donor"] for pair in pairing["pairs"]]
        if entry["recipient_indices_sha256"] != _sha256_indices(recipients) or entry[
            "donor_indices_sha256"
        ] != _sha256_indices(donors):
            raise ValueError(f"{name} {source} pairing index identities differ")
        normalized[source] = pairing
    return (
        {
            "path": str(path),
            "sha256": str(reference["sha256"]),
            "content_sha256": str(reference["content_sha256"]),
        },
        manifest,
        normalized,
    )


def _receipt_weighted_ce(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    denominator = sum(float(row["loss_weight"]) for row in rows)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Evaluator aggregate loss weight is invalid")
    value = sum(float(row[field]) * float(row["loss_weight"]) for row in rows) / denominator
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Evaluator aggregate {field} is invalid")
    return value


def _expected_visual_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps = np.asarray([float(row["ce_gap_wrong_minus_correct"]) for row in rows])
    correct = _receipt_weighted_ce(rows, "correct_ce")
    wrong = _receipt_weighted_ce(rows, "wrong_ce")
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


def _expected_blank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps = np.asarray([float(row["ce_gap_blank_minus_correct"]) for row in rows])
    correct = _receipt_weighted_ce(rows, "correct_ce")
    blank = _receipt_weighted_ce(rows, "blank_ce")
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


def _validate_visual_rows(
    value: Any,
    *,
    source: str,
    pairing: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    result = _exact(value, VISUAL_RESULT_FIELDS, name=name)
    pairs = pairing["pairs"]
    examples = _integer(result["examples"], name=f"{name} examples", minimum=1)
    if examples != len(pairs):
        raise ValueError(f"{name} row count differs from the pairing")
    _finite(result["elapsed_seconds"], name=f"{name} elapsed seconds", minimum=0)
    if result["population"] != "matched_eligible_joint_validation_subset":
        raise ValueError(f"{name} population differs")
    if _canonical_bytes(result["coverage"]) != _canonical_bytes(pairing["coverage"]):
        raise ValueError(f"{name} coverage differs from its pairing")
    if result["pairing_sha256"] != matched_wrong_image_pairing_sha256(pairing):
        raise ValueError(f"{name} pairing SHA-256 differs")
    rows = _require_sequence(result["per_example"], name=f"{name} rows")
    if len(rows) != examples:
        raise ValueError(f"{name} does not contain every example")
    normalized: list[dict[str, Any]] = []
    row_by_index = {row["index"]: row for row in pairing["rows"]}
    for position, (row_value, pair) in enumerate(zip(rows, pairs, strict=True)):
        row = _exact(row_value, VISUAL_ROW_FIELDS, name=f"{name} row {position}")
        if (
            row["pairing_position"] != position
            or row["recipient_index"] != pair["recipient"]
            or row["donor_index"] != pair["donor"]
        ):
            raise ValueError(f"{name} row {position} differs from the pinned pairing")
        response_tokens = _integer(
            row["response_tokens"], name=f"{name} row {position} response tokens", minimum=1
        )
        loss_weight = _finite(
            row["loss_weight"], name=f"{name} row {position} loss weight", minimum=0
        )
        if loss_weight <= 0:
            raise ValueError(f"{name} row {position} loss weight must be positive")
        correct = _finite(row["correct_ce"], name=f"{name} row {position} correct CE", minimum=0)
        wrong = _finite(row["wrong_ce"], name=f"{name} row {position} wrong CE", minimum=0)
        gap = _finite(row["ce_gap_wrong_minus_correct"], name=f"{name} row {position} gap")
        if gap != wrong - correct:
            raise ValueError(f"{name} row {position} gap arithmetic differs")
        normalized.append(
            {
                "pairing_position": position,
                "recipient_index": pair["recipient"],
                "donor_index": pair["donor"],
                "recipient_content_id": row_by_index[pair["recipient"]]["content_id"],
                "donor_content_id": row_by_index[pair["donor"]]["content_id"],
                "response_tokens": response_tokens,
                "loss_weight": loss_weight,
                "correct_ce": correct,
                "wrong_ce": wrong,
                "wrong_gap": gap,
            }
        )
    if _canonical_bytes(result["metrics"]) != _canonical_bytes(_expected_visual_metrics(rows)):
        raise ValueError(f"{name} metrics differ from per-example recomputation")
    return normalized


def _validate_blank_rows(
    value: Any,
    *,
    source: str,
    visual_rows: Sequence[Mapping[str, Any]],
    pairing_sha256: str,
    pairing_coverage: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    result = _exact(value, BLANK_RESULT_FIELDS, name=name)
    if result["pairing_sha256"] != pairing_sha256:
        raise ValueError(f"{name} pairing SHA-256 differs")
    examples = _integer(result["examples"], name=f"{name} examples", minimum=1)
    _finite(result["elapsed_seconds"], name=f"{name} elapsed seconds", minimum=0)
    if result["population"] != "matched_eligible_joint_validation_subset":
        raise ValueError(f"{name} population differs")
    if _canonical_bytes(result["coverage"]) != _canonical_bytes(pairing_coverage):
        raise ValueError(f"{name} coverage differs from its pairing")
    rows = _require_sequence(result["per_example"], name=f"{name} rows")
    if examples != len(visual_rows) or len(rows) != examples:
        raise ValueError(f"{name} does not contain the matched visual population")
    normalized: list[dict[str, Any]] = []
    for position, (row_value, visual) in enumerate(zip(rows, visual_rows, strict=True)):
        row = _exact(row_value, BLANK_ROW_FIELDS, name=f"{name} row {position}")
        identity = ("pairing_position", "recipient_index", "response_tokens", "loss_weight")
        if any(row[field] != visual[field] for field in identity):
            raise ValueError(f"{name} row {position} identity differs from correct-image scoring")
        correct = _finite(row["correct_ce"], name=f"{name} row {position} correct CE", minimum=0)
        blank = _finite(row["blank_ce"], name=f"{name} row {position} blank CE", minimum=0)
        gap = _finite(row["ce_gap_blank_minus_correct"], name=f"{name} row {position} gap")
        if correct != visual["correct_ce"]:
            raise ValueError(f"{name} row {position} reuses a different correct CE")
        if gap != blank - correct:
            raise ValueError(f"{name} row {position} gap arithmetic differs")
        normalized.append(
            {
                "pairing_position": position,
                "recipient_index": visual["recipient_index"],
                "recipient_content_id": visual["recipient_content_id"],
                "response_tokens": visual["response_tokens"],
                "loss_weight": visual["loss_weight"],
                "correct_ce": correct,
                "blank_ce": blank,
                "blank_gap": gap,
            }
        )
    if _canonical_bytes(result["metrics"]) != _canonical_bytes(_expected_blank_metrics(rows)):
        raise ValueError(f"{name} metrics differ from per-example recomputation")
    return normalized


def _expected_native_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    filtered_indices = [int(row["dataset_index"]) for row in rows if row["filtered"] is True]
    mask_loss_weight = sum(float(row["mask_loss_weight"]) for row in rows)
    labeled_loss_weight = sum(float(row["labeled_loss_weight"]) for row in rows)
    summed_ce = sum(float(row["summed_ce"]) for row in rows)
    if mask_loss_weight <= 0 or labeled_loss_weight <= 0:
        raise ValueError("Native aggregate has a zero loss divisor")
    ce_loss = summed_ce / labeled_loss_weight
    training_divisor_ce = summed_ce / mask_loss_weight
    ces = [float(row["ce"]) for row in rows if row["ce"] is not None]
    return {
        "examples": len(rows),
        "filtered_examples": len(filtered_indices),
        "filtered_indices": filtered_indices,
        "mask_tokens": sum(int(row["mask_tokens"]) for row in rows),
        "labeled_tokens": sum(int(row["labeled_tokens"]) for row in rows),
        "mask_loss_weight": mask_loss_weight,
        "labeled_loss_weight": labeled_loss_weight,
        "summed_ce": summed_ce,
        "ce_loss": ce_loss,
        "ppl": math.exp(ce_loss),
        "training_divisor_ce": training_divisor_ce,
        "training_divisor_ppl": math.exp(training_divisor_ce),
        "ce_mean": float(np.mean(ces)),
    }


def _validate_native_rows(value: Any, *, name: str) -> tuple[list[dict[str, Any]], str, str, str]:
    result = _exact(value, NATIVE_RESULT_FIELDS, name=name)
    examples = _integer(result["examples"], name=f"{name} examples", minimum=1)
    if examples != NATIVE_EXAMPLES:
        raise ValueError(f"{name} must contain the complete {NATIVE_EXAMPLES}-row holdout")
    _finite(result["elapsed_seconds"], name=f"{name} elapsed seconds", minimum=0)
    if not _is_sha256(result["dataset_order_sha256"]):
        raise ValueError(f"{name} dataset order SHA-256 is invalid")
    for field in ("dataset_order_sha256", "row_provenance_sha256", "native_identity_sha256"):
        if not _is_sha256(result[field]):
            raise ValueError(f"{name} {field} is invalid")
    rows = _require_sequence(result["per_example"], name=f"{name} rows")
    if len(rows) != NATIVE_EXAMPLES:
        raise ValueError(f"{name} does not contain exactly {NATIVE_EXAMPLES} rows")
    normalized: list[dict[str, Any]] = []
    indices: list[int] = []
    provenance_rows: list[Mapping[str, Any]] = []
    for position, row_value in enumerate(rows):
        row = _exact(row_value, NATIVE_ROW_FIELDS, name=f"{name} row {position}")
        dataset_index = _integer(row["dataset_index"], name=f"{name} row {position} dataset index")
        if row["evaluation_position"] != position or dataset_index != position:
            raise ValueError(f"{name} rows are not in manifest order")
        provenance = _require_mapping(row["provenance"], name=f"{name} row {position} provenance")
        if provenance.get("manifest_index") != position:
            raise ValueError(f"{name} row {position} provenance is not manifest aligned")
        mask_tokens = _integer(
            row["mask_tokens"], name=f"{name} row {position} mask tokens", minimum=1
        )
        labeled_tokens = _integer(
            row["labeled_tokens"], name=f"{name} row {position} labeled tokens"
        )
        mask_weight = _finite(
            row["mask_loss_weight"], name=f"{name} row {position} mask loss weight", minimum=0
        )
        labeled_weight = _finite(
            row["labeled_loss_weight"],
            name=f"{name} row {position} labeled loss weight",
            minimum=0,
        )
        summed_ce = _finite(row["summed_ce"], name=f"{name} row {position} summed CE", minimum=0)
        if mask_weight <= 0 or type(row["filtered"]) is not bool:
            raise ValueError(f"{name} row {position} mask/filter identity is invalid")
        filtered = row["filtered"]
        if filtered:
            if (
                labeled_tokens != 0
                or labeled_weight != 0
                or summed_ce != 0
                or row["ce"] is not None
            ):
                raise ValueError(f"{name} row {position} filtered evidence is inconsistent")
            ce: float | None = None
        else:
            if labeled_tokens != mask_tokens or labeled_weight != mask_weight:
                raise ValueError(f"{name} row {position} is only partially labeled")
            ce = _finite(row["ce"], name=f"{name} row {position} CE", minimum=0)
            if ce != summed_ce / labeled_weight:
                raise ValueError(f"{name} row {position} CE arithmetic differs")
        indices.append(dataset_index)
        provenance_rows.append(provenance)
        normalized.append(
            {
                "evaluation_position": position,
                "dataset_index": dataset_index,
                "provenance": dict(provenance),
                "mask_tokens": mask_tokens,
                "labeled_tokens": labeled_tokens,
                "mask_loss_weight": mask_weight,
                "labeled_loss_weight": labeled_weight,
                "summed_ce": summed_ce,
                "filtered": filtered,
                "ce": ce,
            }
        )
    if len(set(indices)) != NATIVE_EXAMPLES:
        raise ValueError(f"{name} repeats a native holdout dataset index")
    if result["dataset_order_sha256"] != _canonical_sha256(indices):
        raise ValueError(f"{name} dataset order SHA-256 differs from its rows")
    if result["row_provenance_sha256"] != _canonical_sha256(provenance_rows):
        raise ValueError(f"{name} provenance SHA-256 differs from its rows")
    filtered_indices = tuple(row["dataset_index"] for row in normalized if row["filtered"])
    if filtered_indices != NATIVE_FILTERED_INDICES:
        raise ValueError(f"{name} filtered indices differ from the audited holdout")
    if _canonical_bytes(result["metrics"]) != _canonical_bytes(_expected_native_metrics(rows)):
        raise ValueError(f"{name} metrics differ from per-example recomputation")
    return (
        normalized,
        str(result["dataset_order_sha256"]),
        str(result["row_provenance_sha256"]),
        str(result["native_identity_sha256"]),
    )


def _validate_semantic_policy(value: Any, *, name: str) -> Mapping[str, Any]:
    policy = _exact(
        value,
        frozenset(
            {
                "output_overwrite_enabled",
                "pairing_manifest_requires_sha256_pin",
                "all_pairings_rehashed",
                "all_pairings_deterministically_rebuilt",
                "checkpoint_private_snapshot",
                "checkpoint_post_identity_rehashed",
                "native_sources_full_hash_pre_and_post",
                "descriptive_only",
                "promotion_eligible",
            }
        ),
        name=name,
    )
    if (
        policy["output_overwrite_enabled"] is not False
        or policy["pairing_manifest_requires_sha256_pin"] is not True
        or policy["all_pairings_rehashed"] is not True
        or policy["all_pairings_deterministically_rebuilt"] is not True
        or policy["checkpoint_private_snapshot"]
        != "full model_and_optim same-FD verified byte copy; load only; delete after load"
        or policy["checkpoint_post_identity_rehashed"] is not True
        or policy["native_sources_full_hash_pre_and_post"] is not True
        or policy["descriptive_only"] is not True
        or policy["promotion_eligible"] is not False
    ):
        raise ValueError(f"{name} differs from the exact immutable descriptive policy")
    return policy


def _validate_git(value: Any, *, name: str) -> Mapping[str, Any]:
    git = _exact(
        value,
        frozenset({"revision", "dirty", "status_sha256", "tracked_diff_sha256"}),
        name=name,
    )
    if (
        not isinstance(git["revision"], str)
        or len(git["revision"]) != 40
        or any(character not in "0123456789abcdef" for character in git["revision"])
        or git["dirty"] is not False
        or not _is_sha256(git["status_sha256"])
        or not _is_sha256(git["tracked_diff_sha256"])
    ):
        raise ValueError(f"{name} is not one exact clean git revision")
    return git


def _validate_producer(value: Any, *, name: str) -> Mapping[str, Any]:
    producer = _exact(
        value,
        frozenset(
            {
                "path",
                "sha256",
                "comparator_path",
                "comparator_sha256",
                "perception_helper_path",
                "perception_helper_sha256",
                "bridge_helper_path",
                "bridge_helper_sha256",
                "pairing_implementation_path",
                "pairing_implementation_sha256",
                "training_contract_path",
                "training_contract_sha256",
            }
        ),
        name=name,
    )
    pairing_source = inspect.getsourcefile(validate_matched_wrong_image_pairing)
    if pairing_source is None:
        raise RuntimeError("Could not locate live matched-wrong pairing source")
    live = {
        "": Path(__file__).with_name("vision_alignment_joint_matched_wrong.py"),
        "comparator_": Path(__file__),
        "perception_helper_": Path(__file__).with_name(
            "vision_alignment_perception_matched_wrong.py"
        ),
        "bridge_helper_": Path(__file__).with_name("vision_alignment_matched_wrong.py"),
        "pairing_implementation_": Path(pairing_source),
        "training_contract_": Path(__file__).resolve().parents[1] / "train" / "Vision-Alignment.py",
    }
    for prefix, live_path in live.items():
        _validate_implementation_ref(
            {"path": producer[f"{prefix}path"], "sha256": producer[f"{prefix}sha256"]},
            live_path=live_path,
            name=f"{name} {prefix or 'evaluator'}",
        )
    return producer


def _validate_tokenizer(value: Any, *, name: str) -> Mapping[str, Any]:
    tokenizer = _exact(
        value,
        frozenset({"id", "revision", "fingerprint", "token_ids", "token_ids_sha256"}),
        name=name,
    )
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
    if (
        tokenizer["id"] != "allenai/dolma2-tokenizer"
        or tokenizer["revision"] != "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
        or tokenizer["fingerprint"]
        != "8fec2af8c372f4c72a1a665ad8e70517625f94f041dbfcb7db4932071380f9a7"
        or tokenizer["token_ids"] != expected_token_ids
        or tokenizer["token_ids_sha256"] != _canonical_sha256(expected_token_ids)
    ):
        raise ValueError(f"{name} differs from the reviewed joint tokenizer")
    return tokenizer


def _validate_projection(value: Any, *, name: str) -> Mapping[str, Any]:
    projection = _exact(
        value,
        frozenset(
            {
                "path",
                "raw_sha256",
                "content_sha256",
                "source_spec_sha256",
                "visual_source_registry_sha256",
                "runtime_registry_sha256",
                "validation_rows",
                "validation_unique_image_contents",
                "validation_cross_source_duplicate_rows",
                "sources",
            }
        ),
        name=name,
    )
    if (
        projection["raw_sha256"] != EXPECTED_PROJECTION_SHA256
        or projection["validation_rows"] != 4096
        or projection["validation_unique_image_contents"] != 3584
        or projection["validation_cross_source_duplicate_rows"] != 512
    ):
        raise ValueError(f"{name} identity or population differs")
    _artifact_ref(projection["path"], projection["raw_sha256"], name=f"{name} manifest")
    for field in (
        "content_sha256",
        "source_spec_sha256",
        "visual_source_registry_sha256",
        "runtime_registry_sha256",
    ):
        if not _is_sha256(projection[field]):
            raise ValueError(f"{name} {field} is invalid")
    sources = _require_mapping(projection["sources"], name=f"{name} sources")
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError(f"{name} source set differs")
    source_fields = frozenset(
        {
            "examples",
            "runtime_dataset_fingerprint",
            "selection_indices_sha256",
            "row_image_content_sha256",
            "unique_image_content_count",
            "live_image_validation_sha256",
            "live_serialized_rows_sha256",
        }
    )
    for source in SOURCE_NAMES:
        source_identity = _exact(sources[source], source_fields, name=f"{name} {source}")
        if (
            source_identity["examples"] != 512
            or type(source_identity["unique_image_content_count"]) is not int
            or not 1 <= source_identity["unique_image_content_count"] <= 512
        ):
            raise ValueError(f"{name} {source} population differs")
        for field in source_fields - {"examples", "unique_image_content_count"}:
            if not _is_sha256(source_identity[field]):
                raise ValueError(f"{name} {source} {field} is invalid")
    return projection


def _validate_source_audit(value: Any, *, name: str) -> Mapping[str, Any]:
    audit = _exact(
        value,
        frozenset(
            {
                "path",
                "raw_sha256",
                "fingerprint",
                "source_registry_sha256",
                "runtime_registry_sha256",
                "status",
                "phase",
            }
        ),
        name=name,
    )
    if (
        audit["fingerprint"] != EXPECTED_SOURCE_AUDIT_FINGERPRINT
        or audit["status"] != "ok"
        or audit["phase"] != "joint"
    ):
        raise ValueError(f"{name} status, phase, or fingerprint differs")
    _artifact_ref(audit["path"], audit["raw_sha256"], name=f"{name} artifact")
    for field in ("raw_sha256", "source_registry_sha256", "runtime_registry_sha256"):
        if not _is_sha256(audit[field]):
            raise ValueError(f"{name} {field} is invalid")
    return audit


def _validate_native_identity(value: Any, *, name: str) -> Mapping[str, Any]:
    fields = frozenset(
        {
            "train_manifest_path",
            "train_manifest_sha256",
            "train_fingerprint",
            "train_source_count",
            "train_source_inventory_sha256",
            "holdout_manifest_path",
            "holdout_manifest_sha256",
            "holdout_fingerprint",
            "holdout_source_count",
            "holdout_source_inventory_sha256",
            "verification_receipt_path",
            "verification_receipt_sha256",
            "full_source_hash_verification",
            "train_holdout_pair_validated",
            "examples",
            "sequence_length",
            "manifest_order_sha256",
            "row_provenance_sha256",
            "live_serialized_rows_sha256",
        }
    )
    identity = _exact(value, fields, name=name)
    if (
        identity["full_source_hash_verification"] is not True
        or identity["train_holdout_pair_validated"] is not True
        or identity["examples"] != NATIVE_EXAMPLES
        or identity["sequence_length"] != 8192
        or identity["manifest_order_sha256"] != _canonical_sha256(list(range(NATIVE_EXAMPLES)))
        or identity["train_source_count"] != 947
        or identity["holdout_source_count"] != 560
        or identity["train_fingerprint"] != EXPECTED_NATIVE_TRAIN_FINGERPRINT
        or identity["holdout_fingerprint"] != EXPECTED_NATIVE_HOLDOUT_FINGERPRINT
        or identity["verification_receipt_sha256"] != EXPECTED_NATIVE_VERIFICATION_SHA256
    ):
        raise ValueError(f"{name} population or validation status differs")
    for prefix in ("train", "holdout"):
        _artifact_ref(
            identity[f"{prefix}_manifest_path"],
            identity[f"{prefix}_manifest_sha256"],
            name=f"{name} {prefix} manifest",
        )
    _artifact_ref(
        identity["verification_receipt_path"],
        identity["verification_receipt_sha256"],
        name=f"{name} verification receipt",
    )
    for field in fields:
        if field.endswith(("_sha256", "_fingerprint")) and not _is_sha256(identity[field]):
            raise ValueError(f"{name} {field} is invalid")
    return identity


def _load_evaluator_receipt(
    path_value: str | Path,
    *,
    expected_sha256: str,
    step: int,
    verify_live_checkpoint: bool = True,
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"step{step} expected receipt SHA-256 must be lowercase hex")
    path = _direct_existing_path(Path(path_value), name=f"step{step} evaluator receipt")
    payload, raw_sha = _load_json_bytes(
        path,
        expected_sha256=expected_sha256,
        name=f"step{step} evaluator receipt",
    )
    receipt = _exact(payload, RECEIPT_FIELDS, name=f"step{step} evaluator receipt")
    if (
        receipt["format"] != EVALUATOR_FORMAT
        or receipt["version"] != EVALUATOR_VERSION
        or receipt["status"] != "valid"
    ):
        raise ValueError(f"step{step} evaluator receipt identity or validity differs")
    _timestamp(receipt["created_at"], name=f"step{step} evaluator receipt created_at")
    _validate_content_sha256(receipt, name=f"step{step} evaluator receipt")
    config = _validate_checkpoint_config(
        receipt["checkpoint_config"], step=step, name=f"step{step} checkpoint config"
    )
    _validate_checkpoint_identity(
        receipt["checkpoint"],
        config=config,
        name=f"step{step} checkpoint identity",
        verify_live_files=verify_live_checkpoint,
    )
    load_coverage = _validate_load_coverage(
        receipt["load_coverage"], name=f"step{step} load coverage"
    )
    _validate_producer(receipt["producer"], name=f"step{step} producer")
    _validate_git(receipt["git"], name=f"step{step} git")
    projection = _validate_projection(receipt["projection"], name=f"step{step} projection")
    source_audit = _validate_source_audit(receipt["source_audit"], name=f"step{step} source audit")
    if (
        projection["visual_source_registry_sha256"] != source_audit["source_registry_sha256"]
        or projection["runtime_registry_sha256"] != source_audit["runtime_registry_sha256"]
    ):
        raise ValueError(f"step{step} projection/source-audit registries differ")
    tokenizer = _validate_tokenizer(receipt["tokenizer"], name=f"step{step} tokenizer")
    protocol = _validate_protocol(receipt["protocol"], name=f"step{step} protocol")
    _validate_semantic_policy(receipt["artifact_policy"], name=f"step{step} artifact policy")
    if (
        protocol.get("descriptive_only") is not True
        or protocol.get("promotion_eligible") is not False
    ):
        raise ValueError(f"step{step} protocol does not lock descriptive-only semantics")

    manifest_ref, manifest, pairings = _validate_pairing_manifest(
        receipt["pairing_manifest"],
        receipt_protocol=protocol,
        receipt_checkpoint_config=config,
        receipt_projection=projection,
        receipt_source_audit=source_audit,
        receipt_tokenizer=tokenizer,
        name=f"step{step} pairing manifest",
    )
    visual_results = _require_mapping(receipt["visual_results"], name=f"step{step} visual results")
    if set(visual_results) != set(SOURCE_NAMES):
        raise ValueError(f"step{step} visual results lack the exact source set")
    visual_rows: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_NAMES:
        visual_rows[source] = _validate_visual_rows(
            visual_results[source],
            source=source,
            pairing=pairings[source],
            name=f"step{step} {source} visual result",
        )

    blank_results = _require_mapping(receipt["blank_results"], name=f"step{step} blank results")
    if set(blank_results) != set(BLANK_SOURCE_NAMES):
        raise ValueError(f"step{step} blank results lack the exact caption/transcript set")
    blank_rows: dict[str, list[dict[str, Any]]] = {}
    for source in BLANK_SOURCE_NAMES:
        blank_rows[source] = _validate_blank_rows(
            blank_results[source],
            source=source,
            visual_rows=visual_rows[source],
            pairing_sha256=matched_wrong_image_pairing_sha256(pairings[source]),
            pairing_coverage=pairings[source]["coverage"],
            name=f"step{step} {source} blank result",
        )

    (
        native_rows,
        native_order_sha256,
        native_provenance_sha256,
        native_identity_sha256,
    ) = _validate_native_rows(receipt["native_result"], name=f"step{step} native result")
    native_identity = _validate_native_identity(
        protocol.get("native_identity"), name=f"step{step} native identity"
    )
    if (
        native_identity_sha256 != _canonical_sha256(native_identity)
        or native_provenance_sha256 != native_identity["row_provenance_sha256"]
        or protocol.get("native_row_provenance_sha256") != native_provenance_sha256
        or native_order_sha256 != native_identity["manifest_order_sha256"]
    ):
        raise ValueError(f"step{step} native result does not bind the protocol identity")
    return {
        "input": {"path": str(path), "sha256": raw_sha},
        "receipt": receipt,
        "checkpoint": receipt["checkpoint"],
        "checkpoint_config": config,
        "load_coverage": load_coverage,
        "protocol": protocol,
        "manifest_ref": manifest_ref,
        "manifest": manifest,
        "pairings": pairings,
        "visual_rows": visual_rows,
        "blank_rows": blank_rows,
        "native_rows": native_rows,
        "native_order_sha256": native_order_sha256,
        "native_provenance_sha256": native_provenance_sha256,
        "native_identity_sha256": native_identity_sha256,
    }


def _assert_shared_inputs(step4000: Mapping[str, Any], step8000: Mapping[str, Any]) -> None:
    receipt4000 = step4000["receipt"]
    receipt8000 = step8000["receipt"]
    for field in (
        "producer",
        "git",
        "artifact_policy",
        "projection",
        "source_audit",
        "tokenizer",
        "protocol",
        "pairing_manifest",
    ):
        if _canonical_bytes(receipt4000[field]) != _canonical_bytes(receipt8000[field]):
            raise ValueError(f"step4000/8000 evaluator receipts use different {field}")
    for field in (
        "phase",
        "lineage_id",
        "run_name",
        "sha256",
        "reviewed_profile_path",
        "reviewed_profile_sha256",
        "reviewed_profile_allowlist_path",
        "reviewed_profile_allowlist_sha256",
        "training_git_ref",
        "training_beaker_image",
    ):
        if step4000["checkpoint_config"][field] != step8000["checkpoint_config"][field]:
            raise ValueError(f"step4000/8000 checkpoint config {field} differs")
    if _canonical_bytes(step4000["load_coverage"]) != _canonical_bytes(step8000["load_coverage"]):
        raise ValueError("step4000/8000 checkpoint load surfaces differ")
    root4000 = Path(str(step4000["checkpoint"]["root"]))
    root8000 = Path(str(step8000["checkpoint"]["root"]))
    if (
        root4000 == root8000
        or root4000.parent != root8000.parent
        or step4000["checkpoint"]["identity_sha256"] == step8000["checkpoint"]["identity_sha256"]
        or step4000["checkpoint"]["trainer_state_summary"]["wandb_run_id"]
        != step8000["checkpoint"]["trainer_state_summary"]["wandb_run_id"]
    ):
        raise ValueError("step4000/8000 are not distinct permanent endpoints of one run")
    if _canonical_bytes(step4000["manifest"]) != _canonical_bytes(step8000["manifest"]):
        raise ValueError("step4000/8000 receipts do not reference one exact pairing manifest")
    for source in SOURCE_NAMES:
        if _canonical_bytes(step4000["pairings"][source]) != _canonical_bytes(
            step8000["pairings"][source]
        ):
            raise ValueError(f"step4000/8000 {source} pairing bytes differ")


def _assert_row_identity(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
    name: str,
) -> None:
    if len(before) != len(after):
        raise ValueError(f"{name} row counts differ")
    for position, (left, right) in enumerate(zip(before, after, strict=True)):
        if any(left[field] != right[field] for field in fields):
            raise ValueError(f"{name} row {position} identity differs")


def _validate_cross_checkpoint_rows(
    step4000: Mapping[str, Any], step8000: Mapping[str, Any]
) -> None:
    visual_identity = (
        "pairing_position",
        "recipient_index",
        "donor_index",
        "recipient_content_id",
        "donor_content_id",
        "response_tokens",
        "loss_weight",
    )
    for source in SOURCE_NAMES:
        _assert_row_identity(
            step4000["visual_rows"][source],
            step8000["visual_rows"][source],
            fields=visual_identity,
            name=f"{source} visual step4000/8000",
        )
    blank_identity = (
        "pairing_position",
        "recipient_index",
        "recipient_content_id",
        "response_tokens",
        "loss_weight",
    )
    for source in BLANK_SOURCE_NAMES:
        _assert_row_identity(
            step4000["blank_rows"][source],
            step8000["blank_rows"][source],
            fields=blank_identity,
            name=f"{source} blank step4000/8000",
        )
    native_identity = (
        "evaluation_position",
        "dataset_index",
        "provenance",
        "mask_tokens",
        "labeled_tokens",
        "mask_loss_weight",
        "labeled_loss_weight",
        "filtered",
    )
    _assert_row_identity(
        step4000["native_rows"],
        step8000["native_rows"],
        fields=native_identity,
        name="native step4000/8000",
    )
    if step4000["native_order_sha256"] != step8000["native_order_sha256"]:
        raise ValueError("step4000/8000 native manifest order identities differ")
    if (
        step4000["native_provenance_sha256"] != step8000["native_provenance_sha256"]
        or step4000["native_identity_sha256"] != step8000["native_identity_sha256"]
    ):
        raise ValueError("step4000/8000 native holdout provenance identities differ")


def _array(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    result = np.asarray([row[field] for row in rows], dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"Could not form finite non-empty {field} vector")
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.shape != weights.shape or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Weighted mean inputs differ or contain invalid weights")
    return float(np.sum(values * weights, dtype=np.float64) / np.sum(weights, dtype=np.float64))


def _percentile_interval(samples: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"confidence": BOOTSTRAP_CONFIDENCE, "low": float(low), "high": float(high)}


def _paired_bootstrap(
    values: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, np.ndarray | None],
    seed: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap paired rows, reusing every sampled index across all reported fields."""
    if set(values) != set(weights) or not values:
        raise ValueError("Paired bootstrap values/weights differ or are empty")
    lengths = {len(array) for array in values.values()}
    if len(lengths) != 1:
        raise ValueError("Paired bootstrap fields use different row counts")
    row_count = lengths.pop()
    if row_count <= 0 or samples <= 0:
        raise ValueError("Paired bootstrap row/sample counts must be positive")
    for field, array in values.items():
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"Paired bootstrap {field} is invalid")
        field_weights = weights[field]
        if field_weights is not None and (
            field_weights.shape != array.shape
            or not np.isfinite(field_weights).all()
            or np.any(field_weights <= 0)
        ):
            raise ValueError(f"Paired bootstrap {field} weights are invalid")
    rng = np.random.RandomState(seed)
    estimates = {field: np.empty(samples, dtype=np.float64) for field in values}
    chunk = min(samples, 512)
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        indices = rng.randint(0, row_count, size=(end - start, row_count))
        for field, array in values.items():
            selected = array[indices]
            field_weights = weights[field]
            if field_weights is None:
                estimates[field][start:end] = selected.mean(axis=1)
            else:
                selected_weights = field_weights[indices]
                estimates[field][start:end] = np.sum(
                    selected * selected_weights, axis=1, dtype=np.float64
                ) / np.sum(selected_weights, axis=1, dtype=np.float64)
    return {field: _percentile_interval(array) for field, array in estimates.items()}


def _source_stratified_bootstrap(
    values: Mapping[str, Mapping[str, np.ndarray]],
    *,
    weights: Mapping[str, Mapping[str, np.ndarray | None]],
    sources: Sequence[str],
    seed: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    """Resample rows in each source and equally average sources with shared index draws."""
    if set(values) != set(weights) or not values or samples <= 0:
        raise ValueError("Source-stratified bootstrap inputs are invalid")
    if any(set(values[field]) != set(sources) for field in values):
        raise ValueError("Source-stratified bootstrap lacks the exact source set")
    if any(set(weights[field]) != set(sources) for field in weights):
        raise ValueError("Source-stratified bootstrap weights lack the exact source set")
    rng = np.random.RandomState(seed)
    estimates = {field: np.zeros(samples, dtype=np.float64) for field in values}
    chunk = min(samples, 256)
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        for source in sources:
            reference = next(iter(values.values()))[source]
            row_count = len(reference)
            if row_count <= 0:
                raise ValueError(f"Source-stratified bootstrap {source} is empty")
            indices = rng.randint(0, row_count, size=(end - start, row_count))
            for field in values:
                array = values[field][source]
                field_weights = weights[field][source]
                if (
                    array.shape != reference.shape
                    or array.ndim != 1
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"Source-stratified bootstrap {source}/{field} is invalid")
                selected = array[indices]
                if field_weights is None:
                    means = selected.mean(axis=1)
                else:
                    if (
                        field_weights.shape != array.shape
                        or not np.isfinite(field_weights).all()
                        or np.any(field_weights <= 0)
                    ):
                        raise ValueError(
                            f"Source-stratified bootstrap {source}/{field} weights are invalid"
                        )
                    selected_weights = field_weights[indices]
                    means = np.sum(selected * selected_weights, axis=1, dtype=np.float64) / np.sum(
                        selected_weights, axis=1, dtype=np.float64
                    )
                estimates[field][start:end] += means / len(sources)
    return {field: _percentile_interval(array) for field, array in estimates.items()}


def _step_visual_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    weights = _array(rows, "loss_weight")
    correct = _array(rows, "correct_ce")
    wrong = _array(rows, "wrong_ce")
    gap = _array(rows, "wrong_gap")
    return {
        "examples": len(rows),
        "response_tokens": int(sum(int(row["response_tokens"]) for row in rows)),
        "loss_weight": float(weights.sum()),
        "correct_ce": _weighted_mean(correct, weights),
        "wrong_ce": _weighted_mean(wrong, weights),
        "wrong_gap_equal_example_mean": float(gap.mean()),
        "correct_image_win_rate": float((gap > 0).mean()),
    }


def _compare_visual_source(
    rows4000: Sequence[Mapping[str, Any]],
    rows8000: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    weights = _array(rows4000, "loss_weight")
    correct4000 = _array(rows4000, "correct_ce")
    correct8000 = _array(rows8000, "correct_ce")
    wrong4000 = _array(rows4000, "wrong_ce")
    wrong8000 = _array(rows8000, "wrong_ce")
    gap4000 = _array(rows4000, "wrong_gap")
    gap8000 = _array(rows8000, "wrong_gap")
    changes = {
        "correct_ce_change_8000_minus_4000": correct8000 - correct4000,
        "wrong_ce_change_8000_minus_4000": wrong8000 - wrong4000,
        "gap_change_8000_minus_4000": gap8000 - gap4000,
    }
    intervals = _paired_bootstrap(
        changes,
        weights={
            "correct_ce_change_8000_minus_4000": weights,
            "wrong_ce_change_8000_minus_4000": weights,
            "gap_change_8000_minus_4000": None,
        },
        seed=seed,
        samples=samples,
    )
    output = {
        "step4000": _step_visual_summary(rows4000),
        "step8000": _step_visual_summary(rows8000),
        "paired_changes": {
            "correct_ce_change_8000_minus_4000": {
                "mean": _weighted_mean(changes["correct_ce_change_8000_minus_4000"], weights),
                "bootstrap_ci": intervals["correct_ce_change_8000_minus_4000"],
            },
            "wrong_ce_change_8000_minus_4000": {
                "mean": _weighted_mean(changes["wrong_ce_change_8000_minus_4000"], weights),
                "bootstrap_ci": intervals["wrong_ce_change_8000_minus_4000"],
            },
            "gap_change_8000_minus_4000": {
                "mean": float(changes["gap_change_8000_minus_4000"].mean()),
                "bootstrap_ci": intervals["gap_change_8000_minus_4000"],
                "positive_row_rate": float((changes["gap_change_8000_minus_4000"] > 0).mean()),
            },
        },
    }
    return output, changes, weights


def _macro_visual(
    source_outputs: Mapping[str, Mapping[str, Any]],
    source_changes: Mapping[str, Mapping[str, np.ndarray]],
    source_weights: Mapping[str, np.ndarray],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    metric_names = (
        "correct_ce_change_8000_minus_4000",
        "wrong_ce_change_8000_minus_4000",
        "gap_change_8000_minus_4000",
    )
    values = {
        metric: {source: source_changes[source][metric] for source in SOURCE_NAMES}
        for metric in metric_names
    }
    weights = {
        metric: {
            source: (None if metric.startswith("gap_change") else source_weights[source])
            for source in SOURCE_NAMES
        }
        for metric in metric_names
    }
    intervals = _source_stratified_bootstrap(
        values,
        weights=weights,
        sources=SOURCE_NAMES,
        seed=seed,
        samples=samples,
    )

    def macro_step(step_key: str, metric: str) -> float:
        return float(np.mean([source_outputs[source][step_key][metric] for source in SOURCE_NAMES]))

    def macro_change(metric: str) -> float:
        means = []
        for source in SOURCE_NAMES:
            array = source_changes[source][metric]
            if metric.startswith("gap_change"):
                means.append(float(array.mean()))
            else:
                means.append(_weighted_mean(array, source_weights[source]))
        return float(np.mean(means))

    return {
        "source_weighting": "equal_weight_per_source",
        "source_count": len(SOURCE_NAMES),
        "step4000": {
            "correct_ce": macro_step("step4000", "correct_ce"),
            "wrong_ce": macro_step("step4000", "wrong_ce"),
            "wrong_gap_equal_example_mean": macro_step("step4000", "wrong_gap_equal_example_mean"),
        },
        "step8000": {
            "correct_ce": macro_step("step8000", "correct_ce"),
            "wrong_ce": macro_step("step8000", "wrong_ce"),
            "wrong_gap_equal_example_mean": macro_step("step8000", "wrong_gap_equal_example_mean"),
        },
        "paired_changes": {
            metric: {"mean": macro_change(metric), "bootstrap_ci": intervals[metric]}
            for metric in metric_names
        },
        "source_signs": {
            "positive_gap_change": sum(
                source_outputs[source]["paired_changes"]["gap_change_8000_minus_4000"]["mean"] > 0
                for source in SOURCE_NAMES
            ),
            "lower_correct_ce": sum(
                source_outputs[source]["paired_changes"]["correct_ce_change_8000_minus_4000"][
                    "mean"
                ]
                < 0
                for source in SOURCE_NAMES
            ),
            "source_count": len(SOURCE_NAMES),
        },
    }


def _step_blank_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    weights = _array(rows, "loss_weight")
    correct = _array(rows, "correct_ce")
    blank = _array(rows, "blank_ce")
    gap = _array(rows, "blank_gap")
    return {
        "examples": len(rows),
        "response_tokens": int(sum(int(row["response_tokens"]) for row in rows)),
        "loss_weight": float(weights.sum()),
        "correct_ce": _weighted_mean(correct, weights),
        "blank_ce": _weighted_mean(blank, weights),
        "blank_gap_equal_example_mean": float(gap.mean()),
        "correct_beats_blank_rate": float((gap > 0).mean()),
    }


def _compare_blank_source(
    rows4000: Sequence[Mapping[str, Any]],
    rows8000: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    weights = _array(rows4000, "loss_weight")
    correct4000 = _array(rows4000, "correct_ce")
    correct8000 = _array(rows8000, "correct_ce")
    blank4000 = _array(rows4000, "blank_ce")
    blank8000 = _array(rows8000, "blank_ce")
    gap4000 = _array(rows4000, "blank_gap")
    gap8000 = _array(rows8000, "blank_gap")
    changes = {
        "correct_ce_change_8000_minus_4000": correct8000 - correct4000,
        "blank_ce_change_8000_minus_4000": blank8000 - blank4000,
        "blank_gap_change_8000_minus_4000": gap8000 - gap4000,
    }
    intervals = _paired_bootstrap(
        changes,
        weights={
            "correct_ce_change_8000_minus_4000": weights,
            "blank_ce_change_8000_minus_4000": weights,
            "blank_gap_change_8000_minus_4000": None,
        },
        seed=seed,
        samples=samples,
    )
    output = {
        "step4000": _step_blank_summary(rows4000),
        "step8000": _step_blank_summary(rows8000),
        "paired_changes": {
            "correct_ce_change_8000_minus_4000": {
                "mean": _weighted_mean(changes["correct_ce_change_8000_minus_4000"], weights),
                "bootstrap_ci": intervals["correct_ce_change_8000_minus_4000"],
            },
            "blank_ce_change_8000_minus_4000": {
                "mean": _weighted_mean(changes["blank_ce_change_8000_minus_4000"], weights),
                "bootstrap_ci": intervals["blank_ce_change_8000_minus_4000"],
            },
            "blank_gap_change_8000_minus_4000": {
                "mean": float(changes["blank_gap_change_8000_minus_4000"].mean()),
                "bootstrap_ci": intervals["blank_gap_change_8000_minus_4000"],
            },
        },
    }
    return output, changes, weights


def _macro_blank(
    source_outputs: Mapping[str, Mapping[str, Any]],
    source_changes: Mapping[str, Mapping[str, np.ndarray]],
    source_weights: Mapping[str, np.ndarray],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    metric_names = (
        "correct_ce_change_8000_minus_4000",
        "blank_ce_change_8000_minus_4000",
        "blank_gap_change_8000_minus_4000",
    )
    values = {
        metric: {source: source_changes[source][metric] for source in BLANK_SOURCE_NAMES}
        for metric in metric_names
    }
    weights = {
        metric: {
            source: (None if metric.startswith("blank_gap") else source_weights[source])
            for source in BLANK_SOURCE_NAMES
        }
        for metric in metric_names
    }
    intervals = _source_stratified_bootstrap(
        values,
        weights=weights,
        sources=BLANK_SOURCE_NAMES,
        seed=seed,
        samples=samples,
    )

    def macro_step(step_key: str, metric: str) -> float:
        return float(
            np.mean([source_outputs[source][step_key][metric] for source in BLANK_SOURCE_NAMES])
        )

    def macro_change(metric: str) -> float:
        means = []
        for source in BLANK_SOURCE_NAMES:
            values_for_source = source_changes[source][metric]
            if metric.startswith("blank_gap"):
                means.append(float(values_for_source.mean()))
            else:
                means.append(_weighted_mean(values_for_source, source_weights[source]))
        return float(np.mean(means))

    return {
        "source_weighting": "equal_weight_per_source",
        "source_count": len(BLANK_SOURCE_NAMES),
        "step4000": {
            "correct_ce": macro_step("step4000", "correct_ce"),
            "blank_ce": macro_step("step4000", "blank_ce"),
            "blank_gap_equal_example_mean": macro_step("step4000", "blank_gap_equal_example_mean"),
        },
        "step8000": {
            "correct_ce": macro_step("step8000", "correct_ce"),
            "blank_ce": macro_step("step8000", "blank_ce"),
            "blank_gap_equal_example_mean": macro_step("step8000", "blank_gap_equal_example_mean"),
        },
        "paired_changes": {
            metric: {"mean": macro_change(metric), "bootstrap_ci": intervals[metric]}
            for metric in metric_names
        },
    }


def _native_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _expected_native_metrics(rows)
    return {
        "examples": metrics["examples"],
        "filtered_examples": metrics["filtered_examples"],
        "filtered_indices": metrics["filtered_indices"],
        "mask_tokens": metrics["mask_tokens"],
        "labeled_tokens": metrics["labeled_tokens"],
        "mask_loss_weight": metrics["mask_loss_weight"],
        "labeled_loss_weight": metrics["labeled_loss_weight"],
        "summed_ce": metrics["summed_ce"],
        "inline_compatible_ce_loss": metrics["ce_loss"],
        "inline_compatible_perplexity": metrics["ppl"],
        "training_divisor_ce": metrics["training_divisor_ce"],
        "training_divisor_perplexity": metrics["training_divisor_ppl"],
        "unfiltered_equal_example_ce_mean": metrics["ce_mean"],
    }


def _native_bootstrap(
    rows4000: Sequence[Mapping[str, Any]],
    rows8000: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    summed_reduction = _array(rows4000, "summed_ce") - _array(rows8000, "summed_ce")
    labeled_weights = _array(rows4000, "labeled_loss_weight")
    mask_weights = _array(rows4000, "mask_loss_weight")
    unfiltered = labeled_weights > 0
    ce_reduction = np.asarray(
        [
            float(before["ce"]) - float(after["ce"])
            for before, after in zip(rows4000, rows8000, strict=True)
            if before["ce"] is not None and after["ce"] is not None
        ],
        dtype=np.float64,
    )
    if unfiltered.sum() != len(ce_reduction):
        raise ValueError("Native filtered membership differs during bootstrap")
    rng = np.random.RandomState(seed)
    inline = np.empty(samples, dtype=np.float64)
    training = np.empty(samples, dtype=np.float64)
    unfiltered_mean = np.empty(samples, dtype=np.float64)
    chunk = min(samples, 256)
    for start in range(0, samples, chunk):
        end = min(samples, start + chunk)
        indices = rng.randint(0, len(rows4000), size=(end - start, len(rows4000)))
        selected_summed = summed_reduction[indices]
        selected_labeled = labeled_weights[indices]
        selected_mask = mask_weights[indices]
        labeled_divisors = selected_labeled.sum(axis=1, dtype=np.float64)
        if np.any(labeled_divisors <= 0):
            raise ValueError("Native bootstrap drew no labeled loss mass")
        numerators = selected_summed.sum(axis=1, dtype=np.float64)
        inline[start:end] = numerators / labeled_divisors
        training[start:end] = numerators / selected_mask.sum(axis=1, dtype=np.float64)

        # Use the same 1,000-row draws, dropping filtered occurrences for the optional CE mean.
        selected_unfiltered = unfiltered[indices]
        ce_by_row = np.zeros(len(rows4000), dtype=np.float64)
        ce_by_row[unfiltered] = ce_reduction
        selected_ce = ce_by_row[indices]
        counts = selected_unfiltered.sum(axis=1)
        if np.any(counts <= 0):
            raise ValueError("Native bootstrap drew no unfiltered examples")
        unfiltered_mean[start:end] = (selected_ce * selected_unfiltered).sum(axis=1) / counts
    return {
        "inline_compatible_ce_reduction": _percentile_interval(inline),
        "training_divisor_ce_reduction": _percentile_interval(training),
        "unfiltered_equal_example_ce_reduction": _percentile_interval(unfiltered_mean),
    }


def _compare_native(
    rows4000: Sequence[Mapping[str, Any]],
    rows8000: Sequence[Mapping[str, Any]],
    *,
    dataset_order_sha256: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    summary4000 = _native_summary(rows4000)
    summary8000 = _native_summary(rows8000)
    intervals = _native_bootstrap(rows4000, rows8000, seed=seed, samples=samples)
    summed_reduction = sum(float(row["summed_ce"]) for row in rows4000) - sum(
        float(row["summed_ce"]) for row in rows8000
    )
    unfiltered_ce_reductions = [
        float(before["ce"]) - float(after["ce"])
        for before, after in zip(rows4000, rows8000, strict=True)
        if before["ce"] is not None and after["ce"] is not None
    ]
    return {
        "population": "complete_manifest_order_native_holdout",
        "dataset_order_sha256": dataset_order_sha256,
        "all_rows_joined": True,
        "filtered_indices": list(NATIVE_FILTERED_INDICES),
        "step4000": summary4000,
        "step8000": summary8000,
        "paired_inline_compatible_ce_reduction_4000_minus_8000": {
            "mean": summed_reduction / float(summary4000["labeled_loss_weight"]),
            "bootstrap_ci": intervals["inline_compatible_ce_reduction"],
        },
        "paired_training_divisor_ce_reduction_4000_minus_8000": {
            "mean": summed_reduction / float(summary4000["mask_loss_weight"]),
            "bootstrap_ci": intervals["training_divisor_ce_reduction"],
        },
        "paired_unfiltered_equal_example_ce_reduction_4000_minus_8000": {
            "mean": float(np.mean(unfiltered_ce_reductions)),
            "bootstrap_ci": intervals["unfiltered_equal_example_ce_reduction"],
        },
        "inline_compatible_ce_retention_fraction_step8000_over_step4000": (
            float(summary8000["inline_compatible_ce_loss"])
            / float(summary4000["inline_compatible_ce_loss"])
            if summary4000["inline_compatible_ce_loss"] != 0
            else None
        ),
        "training_divisor_ce_retention_fraction_step8000_over_step4000": (
            float(summary8000["training_divisor_ce"]) / float(summary4000["training_divisor_ce"])
            if summary4000["training_divisor_ce"] != 0
            else None
        ),
    }


def _correlation_disclosure(
    pairings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    recipient_ids: dict[str, list[str]] = {}
    donor_ids: dict[str, list[str]] = {}
    for source in SOURCE_NAMES:
        row_by_index = {row["index"]: row for row in pairings[source]["rows"]}
        recipient_ids[source] = [
            row_by_index[pair["recipient"]]["content_id"] for pair in pairings[source]["pairs"]
        ]
        donor_ids[source] = [
            row_by_index[pair["donor"]]["content_id"] for pair in pairings[source]["pairs"]
        ]
    all_recipients = [identity for source in SOURCE_NAMES for identity in recipient_ids[source]]
    all_donors = [identity for source in SOURCE_NAMES for identity in donor_ids[source]]
    within_source_recipient_donor_reuse = sum(
        len(recipient_ids[source])
        + len(donor_ids[source])
        - len(set(recipient_ids[source]) | set(donor_ids[source]))
        for source in SOURCE_NAMES
    )
    cross_source_recipient_duplicates = len(all_recipients) - len(set(all_recipients))
    all_source_recipient_or_donor_duplicate_uses = (
        len(all_recipients) + len(all_donors) - len(set(all_recipients) | set(all_donors))
    )
    return {
        "row_independence_assumed": False,
        "paired_checkpoint_scores": True,
        "within_source_recipient_donor_content_reuse": within_source_recipient_donor_reuse,
        "cross_source_duplicate_recipient_rows": cross_source_recipient_duplicates,
        "all_source_duplicate_recipient_or_donor_uses": (
            all_source_recipient_or_donor_duplicate_uses
        ),
        "primary_bootstrap_unit": "example_within_source",
        "image_cluster_sensitivity": {
            "performed": False,
            "reason": (
                "The locked pairing guarantees unique recipient and donor content within each "
                "source; cross-source image clusters cross the equal-source strata and are "
                "disclosed rather than treated as independent promotion evidence."
            ),
        },
    }


def _build_components(
    evaluations: Mapping[int, Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if set(evaluations) != set(STEPS):
        raise ValueError("Comparison requires exact step4000 and step8000 evaluations")
    if bootstrap_seed != DEFAULT_BOOTSTRAP_SEED or bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError("Comparison bootstrap seed/sample count differs from locked policy")
    step4000 = evaluations[4000]
    step8000 = evaluations[8000]
    _assert_shared_inputs(step4000, step8000)
    _validate_cross_checkpoint_rows(step4000, step8000)

    visual_sources: dict[str, Any] = {}
    visual_changes: dict[str, dict[str, np.ndarray]] = {}
    visual_weights: dict[str, np.ndarray] = {}
    for source_index, source in enumerate(SOURCE_NAMES):
        output, changes, weights = _compare_visual_source(
            step4000["visual_rows"][source],
            step8000["visual_rows"][source],
            seed=bootstrap_seed + 1_000_000 + source_index * 10_000,
            samples=bootstrap_samples,
        )
        visual_sources[source] = output
        visual_changes[source] = changes
        visual_weights[source] = weights
    visual_macro = _macro_visual(
        visual_sources,
        visual_changes,
        visual_weights,
        seed=bootstrap_seed + 20_000_000,
        samples=bootstrap_samples,
    )

    blank_sources: dict[str, Any] = {}
    blank_changes: dict[str, dict[str, np.ndarray]] = {}
    blank_weights: dict[str, np.ndarray] = {}
    for source_index, source in enumerate(BLANK_SOURCE_NAMES):
        output, changes, weights = _compare_blank_source(
            step4000["blank_rows"][source],
            step8000["blank_rows"][source],
            seed=bootstrap_seed + 30_000_000 + source_index * 10_000,
            samples=bootstrap_samples,
        )
        blank_sources[source] = output
        blank_changes[source] = changes
        blank_weights[source] = weights
    blank_macro = _macro_blank(
        blank_sources,
        blank_changes,
        blank_weights,
        seed=bootstrap_seed + 40_000_000,
        samples=bootstrap_samples,
    )

    native = _compare_native(
        step4000["native_rows"],
        step8000["native_rows"],
        dataset_order_sha256=step4000["native_order_sha256"],
        seed=bootstrap_seed + 50_000_000,
        samples=bootstrap_samples,
    )
    count_source = visual_sources["count_numeric"]
    count_guard = {
        "source": "count_numeric",
        "complete_pair_count": count_source["step4000"]["examples"],
        "same_rows_at_both_checkpoints": True,
        "step4000": {
            "correct_ce": count_source["step4000"]["correct_ce"],
            "wrong_gap_equal_example_mean": count_source["step4000"][
                "wrong_gap_equal_example_mean"
            ],
        },
        "step8000": {
            "correct_ce": count_source["step8000"]["correct_ce"],
            "wrong_gap_equal_example_mean": count_source["step8000"][
                "wrong_gap_equal_example_mean"
            ],
        },
        "paired_changes": {
            "correct_ce_change_8000_minus_4000": count_source["paired_changes"][
                "correct_ce_change_8000_minus_4000"
            ],
            "gap_change_8000_minus_4000": count_source["paired_changes"][
                "gap_change_8000_minus_4000"
            ],
        },
        "interpretation": "descriptive_count_specific_guard_without_threshold",
    }
    return {
        "visual": {
            "estimand": ("all-response wrong-minus-correct CE gap change, step8000 minus step4000"),
            "sources": visual_sources,
            "equal_source_macro": visual_macro,
        },
        "blank": {
            "estimand": ("all-response blank-minus-correct CE gap change, step8000 minus step4000"),
            "sources": blank_sources,
            "equal_source_macro": blank_macro,
        },
        "native": native,
        "count_guard": count_guard,
        "correlation_disclosure": _correlation_disclosure(step4000["pairings"]),
    }


def _build_comparison_receipt(
    evaluations: Mapping[int, Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    components = _build_components(
        evaluations,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    first = evaluations[4000]
    second = evaluations[8000]
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    _timestamp(timestamp, name="comparison created_at")
    script_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "status": "valid",
        "created_at": timestamp,
        "producer": {
            "script": str(script_path),
            "script_sha256": _sha256_file(script_path),
        },
        "inputs": {
            "step4000": dict(first["input"]),
            "step8000": dict(second["input"]),
        },
        "policy": {
            "conclusion": "descriptive_only",
            "descriptive_only": True,
            "promotion_eligible": False,
            "promotion_decision": None,
            "margins_defined": False,
            "reason": "No user-approved promotion margins or automatic decision rule exist.",
        },
        "protocol": {
            "name": PROTOCOL_NAME,
            "response_window": "all",
            "bootstrap": {
                "method": BOOTSTRAP_METHOD,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "protocol_fixed_pre_results": True,
                "same_resample_indices_across_conditions_and_metrics": True,
            },
            "visual_source_weighting": "equal_weight_per_source",
            "ce_aggregation": "loss_weight_weighted_within_source",
            "gap_aggregation": "equal_example_within_source",
            "native_population": NATIVE_EXAMPLES,
        },
        "shared_inputs": {
            "evaluator_protocol_sha256": _canonical_sha256(first["protocol"]),
            "pairing_manifest": dict(first["manifest_ref"]),
            "pairing_payloads_sha256": {
                source: matched_wrong_image_pairing_sha256(first["pairings"][source])
                for source in SOURCE_NAMES
            },
            "checkpoint_config_sha256": first["checkpoint_config"]["sha256"],
            "reviewed_profile_sha256": first["checkpoint_config"]["reviewed_profile_sha256"],
            "projection_sha256": _canonical_sha256(first["receipt"]["projection"]),
            "source_audit_sha256": _canonical_sha256(first["receipt"]["source_audit"]),
            "tokenizer_sha256": _canonical_sha256(first["receipt"]["tokenizer"]),
            "git_sha256": _canonical_sha256(first["receipt"]["git"]),
            "native_dataset_order_sha256": first["native_order_sha256"],
            "native_row_provenance_sha256": first["native_provenance_sha256"],
            "native_identity_sha256": first["native_identity_sha256"],
            "step4000_checkpoint_identity_sha256": first["checkpoint"]["identity_sha256"],
            "step8000_checkpoint_identity_sha256": second["checkpoint"]["identity_sha256"],
        },
        **components,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    _validate_comparison_structure(payload)
    return payload


def _validate_comparison_structure(value: Any) -> None:
    """Validate output schema and immutable policy before or after publication."""
    receipt = _exact(value, OUTPUT_FIELDS, name="comparison receipt")
    if receipt["format"] != FORMAT or receipt["version"] != VERSION or receipt["status"] != "valid":
        raise ValueError("Comparison receipt identity or validity differs")
    _timestamp(receipt["created_at"], name="comparison receipt created_at")
    _validate_content_sha256(receipt, name="comparison receipt")
    policy = _exact(
        receipt["policy"],
        frozenset(
            {
                "conclusion",
                "descriptive_only",
                "promotion_eligible",
                "promotion_decision",
                "margins_defined",
                "reason",
            }
        ),
        name="comparison policy",
    )
    if (
        policy["conclusion"] != "descriptive_only"
        or policy["descriptive_only"] is not True
        or policy["promotion_eligible"] is not False
        or policy["promotion_decision"] is not None
        or policy["margins_defined"] is not False
    ):
        raise ValueError("Comparison receipt violates descriptive-only policy")
    inputs = _exact(receipt["inputs"], frozenset(STEP_KEYS), name="comparison receipt inputs")
    for key in STEP_KEYS:
        reference = _exact(
            inputs[key], frozenset({"path", "sha256"}), name=f"comparison {key} input"
        )
        if not _is_sha256(reference["sha256"]):
            raise ValueError(f"Comparison {key} input SHA-256 is invalid")
    protocol = _require_mapping(receipt["protocol"], name="comparison protocol")
    if protocol.get("name") != PROTOCOL_NAME or protocol.get("response_window") != "all":
        raise ValueError("Comparison protocol differs")
    bootstrap = _require_mapping(protocol.get("bootstrap"), name="comparison bootstrap")
    if (
        bootstrap.get("method") != BOOTSTRAP_METHOD
        or bootstrap.get("confidence") != BOOTSTRAP_CONFIDENCE
        or bootstrap.get("samples") != DEFAULT_BOOTSTRAP_SAMPLES
        or bootstrap.get("seed") != DEFAULT_BOOTSTRAP_SEED
        or bootstrap.get("protocol_fixed_pre_results") is not True
        or bootstrap.get("same_resample_indices_across_conditions_and_metrics") is not True
    ):
        raise ValueError("Comparison bootstrap policy differs")
    if receipt["visual"]["equal_source_macro"]["source_count"] != len(SOURCE_NAMES):
        raise ValueError("Comparison visual source count guard differs")
    if receipt["native"]["step4000"]["examples"] != NATIVE_EXAMPLES:
        raise ValueError("Comparison native population guard differs")


def validate_comparison_receipt(value: Any, *, verify_inputs: bool = True) -> None:
    """Strictly re-open inputs and rederive every field in one comparison receipt."""
    _validate_comparison_structure(value)
    if not verify_inputs:
        return
    receipt = _require_mapping(value, name="comparison receipt")
    inputs = receipt["inputs"]
    evaluations = {
        4000: _load_evaluator_receipt(
            inputs["step4000"]["path"],
            expected_sha256=inputs["step4000"]["sha256"],
            step=4000,
            verify_live_checkpoint=True,
        ),
        8000: _load_evaluator_receipt(
            inputs["step8000"]["path"],
            expected_sha256=inputs["step8000"]["sha256"],
            step=8000,
            verify_live_checkpoint=True,
        ),
    }
    expected = _build_comparison_receipt(
        evaluations,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
        created_at=receipt["created_at"],
    )
    if _canonical_bytes(receipt) != _canonical_bytes(expected):
        raise ValueError("Comparison receipt fields differ from full input rederivation")


def build_comparison_receipt(
    *,
    step4000_path: str | Path,
    step4000_sha256: str,
    step8000_path: str | Path,
    step8000_sha256: str,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Load two evaluator receipts and build a strict CPU-only comparison receipt."""
    evaluations = {
        4000: _load_evaluator_receipt(
            step4000_path,
            expected_sha256=step4000_sha256,
            step=4000,
            verify_live_checkpoint=False,
        ),
        8000: _load_evaluator_receipt(
            step8000_path,
            expected_sha256=step8000_sha256,
            step=8000,
            verify_live_checkpoint=False,
        ),
    }
    return _build_comparison_receipt(
        evaluations,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )


def validate_evaluator_receipt(
    path: str | Path,
    expected_sha256: str,
    step: int,
    verify_live_checkpoint: bool = False,
) -> None:
    """Validate one exact evaluator receipt against the frozen joint schema.

    This public entry point is also used by the evaluator's prepublication cross-schema check.
    Full checkpoint bytes can be deferred there because the evaluator has just authenticated the
    live endpoint; the comparator always enables it in its own closing prepublication check.
    """
    if step not in STEPS:
        raise ValueError(f"Evaluator receipt step must be one of {STEPS}")
    _load_evaluator_receipt(
        path,
        expected_sha256=expected_sha256,
        step=step,
        verify_live_checkpoint=verify_live_checkpoint,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the strict CPU-only step4000/8000 comparison."""
    args = _parser().parse_args(argv)
    if args.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError("--bootstrap-seed differs from the frozen comparison policy")
    if args.bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError("--bootstrap-samples differs from the frozen comparison policy")
    receipt = build_comparison_receipt(
        step4000_path=args.step4000,
        step4000_sha256=args.expected_step4000_sha256,
        step8000_path=args.step8000,
        step8000_sha256=args.expected_step8000_sha256,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    # A second complete open/hash/rederivation is the closing input predicate.  It must succeed
    # before an immutable output path is published.
    validate_comparison_receipt(receipt, verify_inputs=True)
    output = _safe_output_path(Path(args.output), name="comparison output")
    _write_json_atomic(output, receipt)
    written, _ = _load_json_bytes(output, name="written comparison receipt")
    validate_comparison_receipt(written, verify_inputs=False)
    if _canonical_bytes(written) != _canonical_bytes(receipt):
        raise RuntimeError("Written comparison receipt bytes decode differently")
    print(
        json.dumps(
            {"status": "valid", "output": str(output), "content_sha256": receipt["content_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
