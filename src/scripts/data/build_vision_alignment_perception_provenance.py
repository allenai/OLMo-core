#!/usr/bin/env python
"""Build immutable image-byte provenance for Vision Alignment perception data.

All eight raw adapters are constructed from one exact production source specification and the
current transitive implementation inventory. Seven sources draw candidates from native
validation. ``audited_alignment`` draws candidates from physical training. Every source then
selects exactly 512 distinct image-content hashes by deterministic ranking and retains one
deterministic representative row per hash. Every training row in every source whose encoded
image bytes occur anywhere in validation is removed.

The builder writes into a sibling ``.<name>.building`` directory, retains a content-aware
SQLite image-hash cache for exact resume, validates the finished manifest with the runtime
loader, and renames the directory into place only after all checks pass. Existing output is
never overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_FORMAT,
    PERCEPTION_PROVENANCE_VERSION,
    PERCEPTION_SOURCE_NAMES,
    image_reference_sha256,
    load_perception_provenance_manifest,
    perception_annotation_content_sha256,
    selected_dataset_fingerprint,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset,
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
)

BUILDER_NAME = "build_vision_alignment_perception_provenance"
BUILDER_VERSION = 1
MANIFEST_NAME = "vision-alignment-perception-provenance.json"
STATE_FORMAT = "vision_alignment_perception_provenance_build_state"
STATE_VERSION = 1
HASH_CACHE_VERSION = 1
VALIDATION_IMAGE_CONTENTS_PER_SOURCE = 512
VALIDATION_SELECTION_DOMAIN = b"vision-alignment-perception-validation-v1\0"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
HASH_COMMIT_BATCH = 2_048
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
FINEVISION_MATERIALIZATION_FORMAT = "vision_alignment_finevision_materialization"
FINEVISION_MATERIALIZATION_VERSION = 1
FINEVISION_MATERIALIZATION_MANIFEST = "vision-alignment-finevision-materialization.json"
FINEVISION_MATERIALIZER_SCRIPT = "materialize_vision_alignment_finevision.py"
CANONICAL_FINEVISION_SOURCE_ROOT = Path(
    "/weka/oe-training-default/mm-olmo/hf_datasets/HuggingFaceM4___FineVision"
)
CANONICAL_FINEVISION_SOURCES = (
    ("visualwebinstruct(filtered)", "visualwebinstruct-filtered", 73, 263_581),
    ("geo170k(align)", "geo170k-align", 1, 35_297),
)

_DATASET_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

log = logging.getLogger(__name__)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON file {path}: {error}") from error


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as file_handle:
            file_handle.write(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _exact_fields(value: Any, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _lower_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _relative_artifact_path(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a non-empty relative path")
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{name} escapes its artifact root")
    return path


@dataclass(frozen=True)
class FineVisionMaterialization:
    """Strictly verified FineVision Arrow materialization consumed by perception."""

    manifest_path: Path
    raw_sha256: str
    content_sha256: str
    source_root: Path
    visualweb_path: Path
    geo170k_path: Path
    visualweb_fingerprint: str
    geo170k_fingerprint: str


def _validate_finevision_materialization(
    manifest_path: str | Path,
    expected_sha256: str,
) -> FineVisionMaterialization:
    """Validate materializer bytes, canonical raw shards, and immutable Arrow outputs."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest_value = json.loads(
            manifest_raw,
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid FineVision materialization manifest {manifest_path}") from error
    raw_sha = hashlib.sha256(manifest_raw).hexdigest()
    if raw_sha != _lower_sha256(
        expected_sha256,
        name="expected FineVision materialization manifest SHA-256",
    ):
        raise ValueError(
            "FineVision materialization manifest SHA-256 differs: "
            f"expected {expected_sha256}, got {raw_sha}"
        )
    root = _exact_fields(
        manifest_value,
        {
            "format",
            "version",
            "builder_sha256",
            "source_root",
            "sources",
            "status",
            "created_at",
            "outputs",
            "content_sha256",
        },
        name="FineVision materialization manifest",
    )
    if (
        root["format"] != FINEVISION_MATERIALIZATION_FORMAT
        or root["version"] != FINEVISION_MATERIALIZATION_VERSION
        or root["status"] != "verified"
    ):
        raise ValueError("FineVision materialization identity or status differs")
    unsigned = dict(root)
    content_sha = _lower_sha256(
        unsigned.pop("content_sha256"),
        name="FineVision materialization content_sha256",
    )
    if _canonical_sha256(unsigned) != content_sha:
        raise ValueError("FineVision materialization content_sha256 differs")
    if not isinstance(root["created_at"], str):
        raise TypeError("FineVision materialization created_at must be ISO-8601")
    try:
        created = datetime.fromisoformat(root["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("FineVision materialization created_at is not ISO-8601") from error
    if created.tzinfo is None:
        raise ValueError("FineVision materialization created_at must include a timezone")

    materializer_path = Path(__file__).resolve().with_name(FINEVISION_MATERIALIZER_SCRIPT)
    builder_sha = _lower_sha256(
        root["builder_sha256"],
        name="FineVision materializer builder_sha256",
    )
    if not materializer_path.is_file() or _sha256_file(materializer_path) != builder_sha:
        raise ValueError("FineVision materializer bytes differ from the current builder")
    source_root = Path(str(root["source_root"])).expanduser().resolve()
    if source_root != CANONICAL_FINEVISION_SOURCE_ROOT.resolve():
        raise ValueError("FineVision materialization source_root is not canonical")

    raw_sources = root["sources"]
    outputs = root["outputs"]
    if (
        not isinstance(raw_sources, list)
        or not isinstance(outputs, list)
        or len(raw_sources) != len(CANONICAL_FINEVISION_SOURCES)
        or len(outputs) != len(CANONICAL_FINEVISION_SOURCES)
    ):
        raise ValueError("FineVision materialization must contain the exact two sources")
    artifact_root = manifest_path.parent
    resolved_outputs: dict[str, Path] = {}
    for position, (name, output_name, expected_shards, expected_rows) in enumerate(
        CANONICAL_FINEVISION_SOURCES
    ):
        source = _exact_fields(
            raw_sources[position],
            {
                "name",
                "output_name",
                "shards",
                "shard_count",
                "rows",
                "physical_schema_sha256",
                "source_metadata_sha256",
            },
            name=f"FineVision sources[{position}]",
        )
        output = _exact_fields(
            outputs[position],
            {
                "name",
                "path",
                "rows",
                "dataset_fingerprint",
                "dataset_info_sha256",
                "physical_schema_sha256",
                "shards",
            },
            name=f"FineVision outputs[{position}]",
        )
        if (
            source["name"] != name
            or source["output_name"] != output_name
            or output["name"] != name
            or output["path"] != output_name
            or source["shard_count"] != expected_shards
            or source["rows"] != expected_rows
            or output["rows"] != expected_rows
        ):
            raise ValueError(f"FineVision materialization pins differ for {name!r}")
        source_schema = _lower_sha256(
            source["physical_schema_sha256"],
            name=f"FineVision {name} source schema",
        )
        if (
            output["physical_schema_sha256"] != source_schema
            or _SHA256_RE.fullmatch(str(source["source_metadata_sha256"])) is None
        ):
            raise ValueError(f"FineVision materialization schema identity differs for {name!r}")
        fingerprint = output["dataset_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or _DATASET_FINGERPRINT_RE.fullmatch(fingerprint) is None
        ):
            raise ValueError(f"FineVision output fingerprint is invalid for {name!r}")

        source_shards = source["shards"]
        output_shards = output["shards"]
        actual_source_paths = tuple(
            path.relative_to(source_root).as_posix()
            for path in sorted((source_root / name).glob("train-*.parquet"))
        )
        if (
            not isinstance(source_shards, list)
            or not isinstance(output_shards, list)
            or len(source_shards) != expected_shards
            or len(output_shards) != expected_shards
            or tuple(item.get("path") for item in source_shards) != actual_source_paths
        ):
            raise ValueError(f"FineVision shard inventory differs for {name!r}")
        source_row_total = 0
        output_row_total = 0
        for shard_index, (source_shard, output_shard) in enumerate(
            zip(source_shards, output_shards)
        ):
            source_shard = _exact_fields(
                source_shard,
                {"path", "bytes", "rows", "sha256"},
                name=f"FineVision {name} source shard {shard_index}",
            )
            output_shard = _exact_fields(
                output_shard,
                {"path", "bytes", "rows", "sha256"},
                name=f"FineVision {name} output shard {shard_index}",
            )
            expected_output_path = (
                f"{output_name}/data-{shard_index:05d}-of-{expected_shards:05d}.arrow"
            )
            if output_shard["path"] != expected_output_path:
                raise ValueError(f"FineVision output shard path differs for {name!r}")
            source_path = _relative_artifact_path(
                source_root,
                source_shard["path"],
                name=f"FineVision {name} source shard path",
            )
            output_path = _relative_artifact_path(
                artifact_root,
                output_shard["path"],
                name=f"FineVision {name} output shard path",
            )
            source_rows = _positive_integer(
                source_shard["rows"], name=f"FineVision {name} source shard rows"
            )
            output_rows = _positive_integer(
                output_shard["rows"], name=f"FineVision {name} output shard rows"
            )
            if source_rows != output_rows:
                raise ValueError(f"FineVision source/output row counts differ for {name!r}")
            for path, entry, entry_name in (
                (source_path, source_shard, "source"),
                (output_path, output_shard, "output"),
            ):
                expected_bytes = _positive_integer(
                    entry["bytes"],
                    name=f"FineVision {name} {entry_name} shard bytes",
                )
                expected_digest = _lower_sha256(
                    entry["sha256"],
                    name=f"FineVision {name} {entry_name} shard SHA-256",
                )
                if (
                    not path.is_file()
                    or path.stat().st_size != expected_bytes
                    or _sha256_file(path) != expected_digest
                ):
                    raise ValueError(
                        f"FineVision {entry_name} shard bytes differ for {name!r}: {path}"
                    )
            receipt_path = output_path.with_suffix(".receipt.json")
            expected_receipt = {
                "source_sha256": source_shard["sha256"],
                "output_sha256": output_shard["sha256"],
                "rows": source_rows,
            }
            if not receipt_path.is_file() or _read_json(receipt_path) != expected_receipt:
                raise ValueError(f"FineVision output receipt differs for {output_path}")
            source_row_total += source_rows
            output_row_total += output_rows
        if source_row_total != expected_rows or output_row_total != expected_rows:
            raise ValueError(f"FineVision total row counts differ for {name!r}")

        output_path = (artifact_root / output_name).resolve()
        dataset_info = output_path / "dataset_info.json"
        if (
            not dataset_info.is_file()
            or _sha256_file(dataset_info) != output["dataset_info_sha256"]
        ):
            raise ValueError(f"FineVision output lacks dataset_info.json: {output_path}")
        from scripts.data.materialize_vision_alignment_finevision import (
            output_dataset_fingerprint,
        )

        expected_fingerprint = output_dataset_fingerprint(
            source_name=name,
            rows=expected_rows,
            physical_schema_sha256=source_schema,
            shards=output_shards,
            dataset_info_sha256=output["dataset_info_sha256"],
        )
        if fingerprint != expected_fingerprint:
            raise ValueError(f"FineVision content fingerprint differs for {name!r}")
        from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

        live = load_from_disk_compat(output_path)
        if len(live) != expected_rows:
            raise ValueError(f"FineVision live Arrow identity differs for {name!r}")
        resolved_outputs[name] = output_path

    complete = artifact_root / "COMPLETE"
    if not complete.is_file() or complete.read_text().strip() != raw_sha:
        raise ValueError("FineVision materialization COMPLETE receipt differs")
    return FineVisionMaterialization(
        manifest_path=manifest_path,
        raw_sha256=raw_sha,
        content_sha256=content_sha,
        source_root=source_root,
        visualweb_path=resolved_outputs["visualwebinstruct(filtered)"],
        geo170k_path=resolved_outputs["geo170k(align)"],
        visualweb_fingerprint=next(
            value["dataset_fingerprint"]
            for value in outputs
            if value["name"] == "visualwebinstruct(filtered)"
        ),
        geo170k_fingerprint=next(
            value["dataset_fingerprint"] for value in outputs if value["name"] == "geo170k(align)"
        ),
    )


def _write_lines(path: Path, values: Sequence[str], *, allow_empty: bool = False) -> dict[str, Any]:
    if (not values and not allow_empty) or any("\n" in value or "\r" in value for value in values):
        raise ValueError(f"Refusing to write empty or malformed line artifact {path}")
    raw = (("\n".join(values) + "\n") if values else "").encode("ascii")
    _atomic_write(path, raw)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(values),
    }


@dataclass(frozen=True)
class FileSignature:
    """Filesystem identity for a resumable path-backed image hash."""

    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        """Return lossless SQLite text fields in schema order."""
        return (
            str(self.size_bytes),
            str(self.mtime_ns),
            str(self.ctime_ns),
            str(self.inode),
            str(self.device),
        )


def _file_signature(path: Path) -> FileSignature:
    try:
        info = path.stat()
    except OSError as error:
        raise ValueError(f"Could not stat perception image {path}: {error}") from error
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"Perception image must be a non-empty regular file: {path}")
    return FileSignature(
        size_bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        inode=info.st_ino,
        device=info.st_dev,
    )


def _sha256_path_stable(path: Path, expected: FileSignature) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            opened = os.fstat(file_handle.fileno())
            opened_signature = FileSignature(
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_ino,
                opened.st_dev,
            )
            if opened_signature != expected:
                raise ValueError(f"Perception image changed before hashing: {path}")
            while chunk := file_handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Could not hash perception image {path}: {error}") from error
    if _file_signature(path) != expected:
        raise ValueError(f"Perception image changed while hashing: {path}")
    return digest.hexdigest()


class ImageHashCache:
    """Single-writer durable cache keyed by path plus complete file identity."""

    def __init__(self, path: Path, plan_sha256: str):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS hashes ("
            "path TEXT PRIMARY KEY, size_bytes TEXT NOT NULL, mtime_ns TEXT NOT NULL, "
            "ctime_ns TEXT NOT NULL, inode TEXT NOT NULL, device TEXT NOT NULL, "
            "sha256 TEXT NOT NULL)"
        )
        expected = {
            "schema_version": str(HASH_CACHE_VERSION),
            "plan_sha256": plan_sha256,
        }
        current = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if current and current != expected:
            raise ValueError(f"Hash cache {path} belongs to a different build plan")
        if not current:
            self.connection.executemany("INSERT INTO metadata VALUES (?, ?)", expected.items())
            self.connection.commit()
        self._pending: list[tuple[Any, ...]] = []

    def lookup(self, path: str, signature: FileSignature) -> str | None:
        """Return a digest only when every cached filesystem field still matches."""
        row = self.connection.execute(
            "SELECT size_bytes, mtime_ns, ctime_ns, inode, device, sha256 "
            "FROM hashes WHERE path=?",
            (path,),
        ).fetchone()
        if row is None or tuple(row[:5]) != signature.as_tuple():
            return None
        digest = row[5]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Hash cache {self.path} contains an invalid digest for {path}")
        return digest

    def store(self, path: str, signature: FileSignature, digest: str) -> None:
        """Queue one newly verified path hash and durably flush bounded batches."""
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Invalid image digest {digest!r}")
        self._pending.append((path, *signature.as_tuple(), digest))
        if len(self._pending) >= HASH_COMMIT_BATCH:
            self.flush()

    def flush(self) -> None:
        """Commit all queued hashes or roll the transaction back."""
        if not self._pending:
            return
        try:
            self.connection.executemany(
                "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?, ?, ?, ?)", self._pending
            )
            self.connection.commit()
            self._pending.clear()
        except Exception:
            self.connection.rollback()
            raise

    def close(self) -> None:
        """Flush, checkpoint, and close the durable cache."""
        self.flush()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


class ImageHasher:
    """Hash raw adapter image references with path and digest interning."""

    def __init__(self, cache: ImageHashCache):
        self.cache = cache
        self.path_results: dict[str, tuple[FileSignature, str]] = {}
        self.digest_pool: dict[str, str] = {}

    def _intern(self, digest: str) -> str:
        return self.digest_pool.setdefault(digest, digest)

    def hash_reference(self, reference: Any) -> str:
        """Return the encoded-byte hash for a raw adapter reference."""
        if isinstance(reference, Mapping):
            embedded = reference.get("bytes")
            if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
                return self._intern(image_reference_sha256(reference))
            reference = reference.get("path")
        if isinstance(reference, (bytes, bytearray, memoryview)):
            return self._intern(image_reference_sha256(reference))
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"Unsupported perception image reference {type(reference)!r}")
        path = Path(reference).expanduser()
        if not path.is_absolute():
            raise ValueError(f"Perception image path must be absolute: {reference!r}")
        path_key = str(path)
        signature = _file_signature(path)
        in_memory = self.path_results.get(path_key)
        if in_memory is not None:
            if in_memory[0] != signature:
                raise ValueError(f"Perception image changed during source scan: {path}")
            return in_memory[1]
        digest = self.cache.lookup(path_key, signature)
        if digest is None:
            digest = _sha256_path_stable(path, signature)
            self.cache.store(path_key, signature, digest)
        digest = self._intern(digest)
        self.path_results[path_key] = (signature, digest)
        return digest

    def validate_paths_unchanged(self) -> None:
        """Restat every path before publication to detect post-hash mutation."""
        for path_value, (expected, _) in self.path_results.items():
            if _file_signature(Path(path_value)) != expected:
                raise ValueError(f"Perception image changed after hashing: {path_value}")


@dataclass(frozen=True)
class RawSplitScan:
    """One exact physical adapter split and its ordered row-byte hashes."""

    dataset: Any
    physical_split: str
    base_fingerprint: str
    base_annotation_sha256: str
    row_hashes: tuple[str, ...]

    @property
    def examples(self) -> int:
        """Return the raw logical example count."""
        return len(self.row_hashes)


def _dataset_fingerprint(dataset: Any, *, source_name: str, split: str) -> str:
    value = runtime_dataset_fingerprint(dataset)
    if not isinstance(value, str) or _DATASET_FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(
            f"Raw {source_name}/{split} lacks a stable lowercase 16- to 64-hex fingerprint"
        )
    return value


def _validate_dataset_annotations(dataset: Any, *, source_name: str, split: str) -> None:
    validate = getattr(dataset, "validate_required_annotations", None)
    if not callable(validate):
        raise TypeError(f"Raw {source_name}/{split} lacks required-annotation validation")
    validate()


def _dataset_snapshot(
    dataset: Any,
    *,
    source_name: str,
    split: str,
) -> tuple[int, str, str]:
    """Capture the exact row, runtime-fingerprint, and annotation identity of a raw split."""
    size = len(dataset)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"Raw {source_name}/{split} dataset is empty")
    return (
        size,
        _dataset_fingerprint(dataset, source_name=source_name, split=split),
        perception_annotation_content_sha256(dataset),
    )


def _scan_dataset(
    dataset: Any,
    *,
    source_name: str,
    physical_split: str,
    hasher: ImageHasher,
) -> RawSplitScan:
    before = _dataset_snapshot(
        dataset,
        source_name=source_name,
        split=physical_split,
    )
    size, base_fingerprint, base_annotation_sha256 = before
    raw_images = getattr(dataset, "raw_image_references", None)
    if not callable(raw_images):
        raise TypeError(f"Raw {source_name}/{physical_split} lacks raw_image_references")
    row_hashes: list[str] = []
    for index in range(size):
        references = tuple(raw_images(index))
        if len(references) != 1:
            raise ValueError(
                f"Raw {source_name}/{physical_split} row {index} must expose exactly one "
                f"image reference, got {len(references)}"
            )
        row_hashes.append(hasher.hash_reference(references[0]))
    after = _dataset_snapshot(
        dataset,
        source_name=source_name,
        split=physical_split,
    )
    if after != before:
        raise ValueError(
            f"Raw {source_name}/{physical_split} identity changed during image scan: "
            f"before={before!r}, after={after!r}"
        )
    return RawSplitScan(
        dataset=dataset,
        physical_split=physical_split,
        base_fingerprint=base_fingerprint,
        base_annotation_sha256=base_annotation_sha256,
        row_hashes=tuple(row_hashes),
    )


def _validate_raw_scans_unchanged(
    raw_scans: Mapping[tuple[str, str], RawSplitScan],
    *,
    source_spec: VisionAlignmentPerceptionSourceSpec,
    tokenizer: Any,
    token_ids: Any,
) -> None:
    """Rebuild and revalidate every raw adapter immediately before publication."""
    for (source_name, physical_split), scan in sorted(raw_scans.items()):
        if scan.physical_split != physical_split:
            raise ValueError(
                f"Raw scan key {source_name}/{physical_split} differs from its split identity"
            )
        dataset = build_vision_alignment_perception_dataset(
            source_spec,
            tokenizer,
            token_ids,
            source_name,
            split=physical_split,
            validate_required_annotations=False,
        )
        expected = (scan.examples, scan.base_fingerprint, scan.base_annotation_sha256)
        before = _dataset_snapshot(
            dataset,
            source_name=source_name,
            split=physical_split,
        )
        _validate_dataset_annotations(
            dataset,
            source_name=source_name,
            split=physical_split,
        )
        after = _dataset_snapshot(
            dataset,
            source_name=source_name,
            split=physical_split,
        )
        if before != expected or after != expected:
            raise ValueError(
                f"Raw {source_name}/{physical_split} identity changed before publication: "
                f"expected={expected!r}, before={before!r}, after={after!r}"
            )


def _validation_representative_indices(
    row_hashes: Sequence[str],
    *,
    source_name: str,
    source_spec_sha256: str,
    target_image_contents: int,
) -> tuple[int, ...]:
    """Select one deterministic row for exactly ``target_image_contents`` hashes."""
    if (
        isinstance(target_image_contents, bool)
        or not isinstance(target_image_contents, int)
        or target_image_contents < 1
    ):
        raise ValueError("validation target_image_contents must be positive")
    groups: dict[str, list[int]] = {}
    for index, digest in enumerate(row_hashes):
        groups.setdefault(digest, []).append(index)
    if len(groups) < target_image_contents:
        raise ValueError(
            f"{source_name} validation has only {len(groups)} distinct image contents; "
            f"exactly {target_image_contents} are required"
        )
    ranked = sorted(
        groups,
        key=lambda digest: hashlib.sha256(
            VALIDATION_SELECTION_DOMAIN
            + source_spec_sha256.encode("ascii")
            + digest.encode("ascii")
        ).digest(),
    )
    representatives = []
    for digest in ranked[:target_image_contents]:
        representatives.append(
            min(
                groups[digest],
                key=lambda index: hashlib.sha256(
                    VALIDATION_SELECTION_DOMAIN
                    + b"representative\0"
                    + source_spec_sha256.encode("ascii")
                    + source_name.encode("utf-8")
                    + digest.encode("ascii")
                    + str(index).encode("ascii")
                ).digest(),
            )
        )
    indices = tuple(sorted(representatives))
    if (
        len(indices) != target_image_contents
        or len({row_hashes[index] for index in indices}) != target_image_contents
    ):
        raise ValueError(f"{source_name} could not form exact content-distinct validation")
    return indices


def _source_components(
    source_name: str, source_spec: VisionAlignmentPerceptionSourceSpec
) -> list[str]:
    if source_name == "audited_alignment":
        return ["visualwebinstruct(filtered)", "geo170k(align)"]
    if source_name == "ocr_document":
        return list(source_spec.ocr_source_names)
    return [source_name]


def _prepare_staging(
    output_dir: Path,
    state: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[Path, int]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building"
    state_path = staging / "build-state.json"
    if staging.exists():
        if not resume:
            raise FileExistsError(
                f"Recoverable staging directory {staging} exists; rerun with --resume"
            )
    else:
        if resume:
            raise FileNotFoundError(f"No recoverable staging directory exists at {staging}")
        staging.mkdir()
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(directory_fd)
        raise RuntimeError(
            f"Another provenance builder owns staging directory {staging}"
        ) from error
    try:
        if state_path.is_file():
            if _read_json(state_path) != state:
                raise ValueError(f"Staging directory {staging} belongs to a different build plan")
        else:
            # A process can die after the exclusive staging mkdir but before committing
            # build-state. Only that provably empty state is safe to initialize on resume.
            if any(staging.iterdir()):
                raise ValueError(f"Staging directory {staging} lacks build-state but is not empty")
            _atomic_write_json(state_path, state)
            _fsync_directory(staging)
    except BaseException:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)
        raise
    return staging, directory_fd


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing every existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic immutable publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _publish_staging(staging: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact {output_dir}")
    _rename_directory_no_replace(staging, output_dir)
    parent_fd = os.open(output_dir.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _load_source_spec(path: Path) -> VisionAlignmentPerceptionSourceSpec:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise TypeError("Perception source spec must be a JSON object")
    mapping = dict(value)
    registry_version = mapping.pop("source_registry_version", None)
    expected_fields = {field.name for field in fields(VisionAlignmentPerceptionSourceSpec)}
    if (
        registry_version != VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
        or set(mapping) != expected_fields
    ):
        raise ValueError("Perception source spec fields or registry version differ")
    if not isinstance(mapping.get("ocr_source_names"), list):
        raise TypeError("Perception source spec ocr_source_names must be a JSON list")
    mapping["ocr_source_names"] = tuple(mapping["ocr_source_names"])
    spec = VisionAlignmentPerceptionSourceSpec(**mapping)
    if spec.as_canonical_dict() != value:
        raise ValueError("Perception source spec is not its exact canonical representation")
    return spec


def _bind_finevision_materialization(
    source_spec: VisionAlignmentPerceptionSourceSpec,
    materialization: FineVisionMaterialization,
) -> VisionAlignmentPerceptionSourceSpec:
    """Derive audited-alignment paths exclusively from the verified Arrow artifact."""
    expected_root = str(materialization.source_root)
    expected_visualweb = str(materialization.visualweb_path)
    expected_geo170k = str(materialization.geo170k_path)
    for field_name, expected in (
        ("finevision_root", expected_root),
        ("finevision_visualweb_path", expected_visualweb),
        ("finevision_geo170k_path", expected_geo170k),
    ):
        current = getattr(source_spec, field_name)
        if current is not None and str(Path(current).expanduser().resolve()) != expected:
            raise ValueError(
                f"Perception source spec {field_name} differs from the FineVision artifact"
            )
    return replace(
        source_spec,
        finevision_root=expected_root,
        finevision_visualweb_path=expected_visualweb,
        finevision_geo170k_path=expected_geo170k,
        finevision_visualweb_fingerprint=materialization.visualweb_fingerprint,
        finevision_geo170k_fingerprint=materialization.geo170k_fingerprint,
    )


def _resolve_created_at(
    destination: Path,
    *,
    resume: bool,
    created_at: str | None,
) -> str:
    """Choose one stable timestamp, recovering it from an exact resumed plan."""
    if created_at is None and resume:
        state_path = destination.parent / f".{destination.name}.building" / "build-state.json"
        if state_path.is_file():
            prior_state = _read_json(state_path)
            if isinstance(prior_state, Mapping):
                prior_created_at = prior_state.get("created_at")
                if isinstance(prior_created_at, str):
                    created_at = prior_created_at
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return created_at


def build_vision_alignment_perception_provenance(
    *,
    source_spec: VisionAlignmentPerceptionSourceSpec,
    expected_source_spec_sha256: str,
    expected_source_registry_sha256: str,
    expected_implementation_inventory: Mapping[str, Any],
    expected_implementation_inventory_sha256: str,
    output_dir: str | Path,
    tokenizer: Any,
    token_ids: Any,
    finevision_materialization: FineVisionMaterialization,
    resume: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one strict perception provenance artifact.

    :param source_spec: Exact production source and serialization specification.
    :param expected_source_spec_sha256: Pinned canonical source-spec digest.
    :param expected_source_registry_sha256: Pinned transitive registry digest.
    :param expected_implementation_inventory: Pinned transitive file inventory.
    :param expected_implementation_inventory_sha256: Canonical digest of that inventory.
    :param output_dir: New immutable artifact directory.
    :param tokenizer: Pinned prepared tokenizer used to construct all raw adapters.
    :param token_ids: Prepared model-specific image token identities.
    :param finevision_materialization: Strictly verified upstream Arrow artifact. Its paths
        must exactly match the bound source spec; raw-root fallback is forbidden.
    :param resume: Resume the exact plan from its sibling staging directory.
    :param created_at: Optional timezone-aware ISO timestamp, primarily for reproducible tests.
    :returns: The exact manifest mapping written to disk.

    :raises ValueError: If any pin, annotation, image, split, or manifest contract differs.
    :raises FileExistsError: If output exists or staging requires explicit resume.
    """
    destination = Path(output_dir).expanduser().resolve()
    created_at = _resolve_created_at(destination, resume=resume, created_at=created_at)
    source_spec.validate_production_contract()
    canonical_spec = source_spec.as_canonical_dict()
    expected_materialization_binding = (
        str(finevision_materialization.source_root.resolve()),
        str(finevision_materialization.visualweb_path.resolve()),
        str(finevision_materialization.geo170k_path.resolve()),
        finevision_materialization.visualweb_fingerprint,
        finevision_materialization.geo170k_fingerprint,
    )
    actual_materialization_binding = (
        canonical_spec["finevision_root"],
        canonical_spec["finevision_visualweb_path"],
        canonical_spec["finevision_geo170k_path"],
        canonical_spec["finevision_visualweb_fingerprint"],
        canonical_spec["finevision_geo170k_fingerprint"],
    )
    if actual_materialization_binding != expected_materialization_binding:
        raise ValueError("Perception source spec does not bind the verified FineVision artifact")
    if (
        _SHA256_RE.fullmatch(expected_source_spec_sha256) is None
        or source_spec.preprocessing_sha256 != expected_source_spec_sha256
    ):
        raise ValueError("Pinned perception source-spec SHA-256 differs")
    current_inventory = vision_alignment_perception_implementation_inventory()
    if (
        _SHA256_RE.fullmatch(expected_implementation_inventory_sha256) is None
        or _canonical_sha256(expected_implementation_inventory)
        != expected_implementation_inventory_sha256
        or dict(expected_implementation_inventory) != current_inventory
    ):
        raise ValueError("Pinned perception implementation inventory differs")
    current_registry_sha = vision_alignment_perception_source_registry_sha256()
    if (
        _SHA256_RE.fullmatch(expected_source_registry_sha256) is None
        or expected_source_registry_sha256 != current_registry_sha
    ):
        raise ValueError("Pinned perception source-registry SHA-256 differs")

    script_path = Path(__file__).resolve()
    builder_sha = _sha256_file(script_path)
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "builder_sha256": builder_sha,
        "source_spec": source_spec.as_canonical_dict(),
        "source_spec_sha256": expected_source_spec_sha256,
        "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": current_registry_sha,
        "source_implementation_inventory": current_inventory,
        "source_implementation_inventory_sha256": expected_implementation_inventory_sha256,
        "finevision_materialization": {
            "manifest": str(finevision_materialization.manifest_path),
            "sha256": finevision_materialization.raw_sha256,
            "content_sha256": finevision_materialization.content_sha256,
        },
        "validation_selection": {
            "algorithm": "sha256-ranked-distinct-content-representatives-v1",
            "image_contents_per_source": VALIDATION_IMAGE_CONTENTS_PER_SOURCE,
        },
        "created_at": created_at,
    }
    plan_sha = _canonical_sha256(state)
    staging, staging_fd = _prepare_staging(destination, state, resume=resume)
    try:
        copied_materialization_path = staging / "upstream" / FINEVISION_MATERIALIZATION_MANIFEST
        try:
            materialization_raw = finevision_materialization.manifest_path.read_bytes()
        except OSError as error:
            raise ValueError("FineVision materialization manifest became unreadable") from error
        if hashlib.sha256(materialization_raw).hexdigest() != finevision_materialization.raw_sha256:
            raise ValueError("FineVision materialization manifest changed before provenance build")
        _atomic_write(copied_materialization_path, materialization_raw)
        materialization_reference = {
            "path": copied_materialization_path.relative_to(staging).as_posix(),
            "sha256": finevision_materialization.raw_sha256,
            "content_sha256": finevision_materialization.content_sha256,
            "visualweb_fingerprint": finevision_materialization.visualweb_fingerprint,
            "geo170k_fingerprint": finevision_materialization.geo170k_fingerprint,
        }
        raw_scans: dict[tuple[str, str], RawSplitScan] = {}
        cache = ImageHashCache(staging / "image-hash-cache.sqlite3", plan_sha)
        hasher = ImageHasher(cache)
        try:
            for source_name in PERCEPTION_SOURCE_NAMES:
                train_dataset = build_vision_alignment_perception_dataset(
                    source_spec,
                    tokenizer,
                    token_ids,
                    source_name,
                    split="train",
                    validate_required_annotations=False,
                )
                _validate_dataset_annotations(
                    train_dataset,
                    source_name=source_name,
                    split="train",
                )
                raw_scans[(source_name, "train")] = _scan_dataset(
                    train_dataset,
                    source_name=source_name,
                    physical_split="train",
                    hasher=hasher,
                )
                if source_name != "audited_alignment":
                    validation_dataset = build_vision_alignment_perception_dataset(
                        source_spec,
                        tokenizer,
                        token_ids,
                        source_name,
                        split="validation",
                        validate_required_annotations=False,
                    )
                    _validate_dataset_annotations(
                        validation_dataset,
                        source_name=source_name,
                        split="validation",
                    )
                    raw_scans[(source_name, "validation")] = _scan_dataset(
                        validation_dataset,
                        source_name=source_name,
                        physical_split="validation",
                        hasher=hasher,
                    )
            hasher.validate_paths_unchanged()
        finally:
            cache.close()

        audited_scan = raw_scans[("audited_alignment", "train")]
        validation_indices: dict[str, tuple[int, ...]] = {}
        for source_name in PERCEPTION_SOURCE_NAMES:
            validation_scan = (
                audited_scan
                if source_name == "audited_alignment"
                else raw_scans[(source_name, "validation")]
            )
            validation_indices[source_name] = _validation_representative_indices(
                validation_scan.row_hashes,
                source_name=source_name,
                source_spec_sha256=expected_source_spec_sha256,
                target_image_contents=VALIDATION_IMAGE_CONTENTS_PER_SOURCE,
            )

        validation_union = {
            (
                audited_scan.row_hashes[index]
                if source_name == "audited_alignment"
                else raw_scans[(source_name, "validation")].row_hashes[index]
            )
            for source_name in PERCEPTION_SOURCE_NAMES
            for index in validation_indices[source_name]
        }
        train_indices: dict[str, tuple[int, ...]] = {}
        filtering: dict[str, Any] = {}
        for source_name in PERCEPTION_SOURCE_NAMES:
            train_scan = raw_scans[(source_name, "train")]
            kept = tuple(
                index
                for index, digest in enumerate(train_scan.row_hashes)
                if digest not in validation_union
            )
            if not kept:
                raise ValueError(
                    f"Validation content filtering removed every {source_name} training row"
                )
            train_indices[source_name] = kept
            filtering[source_name] = {
                "candidate_train_examples": train_scan.examples,
                "removed_train_examples": train_scan.examples - len(kept),
                "output_train_examples": len(kept),
            }

        sources: dict[str, Any] = {}
        logical_hashes: dict[tuple[str, str], tuple[str, ...]] = {}
        for source_name in PERCEPTION_SOURCE_NAMES:
            split_entries: dict[str, Any] = {}
            for logical_split in ("train", "validation"):
                if logical_split == "train":
                    scan = raw_scans[(source_name, "train")]
                    indices = train_indices[source_name]
                elif source_name == "audited_alignment":
                    scan = audited_scan
                    indices = validation_indices[source_name]
                else:
                    scan = raw_scans[(source_name, "validation")]
                    indices = validation_indices[source_name]
                row_hashes = tuple(scan.row_hashes[index] for index in indices)
                unique_hashes = tuple(sorted(set(row_hashes)))
                logical_hashes[(source_name, logical_split)] = row_hashes
                prefix = f"{source_name}-{logical_split}"
                index_ref = _write_lines(
                    staging / "selections" / f"{prefix}.indices",
                    tuple(str(index) for index in indices),
                )
                index_ref["indices_sha256"] = _canonical_sha256(list(indices))
                row_ref = _write_lines(
                    staging / "inventories" / f"{prefix}-rows.sha256",
                    row_hashes,
                )
                unique_ref = _write_lines(
                    staging / "inventories" / f"{prefix}-unique.sha256",
                    unique_hashes,
                )
                for reference in (index_ref, row_ref, unique_ref):
                    reference["path"] = str(
                        (Path(reference["path"]).relative_to(staging)).as_posix()
                    )
                split_entries[logical_split] = {
                    "physical_split": scan.physical_split,
                    "base_annotation_sha256": scan.base_annotation_sha256,
                    "base_dataset_fingerprint": scan.base_fingerprint,
                    "base_examples": scan.examples,
                    "selection": index_ref,
                    "runtime_dataset_fingerprint": selected_dataset_fingerprint(
                        source_name=source_name,
                        logical_split=logical_split,
                        physical_split=scan.physical_split,
                        base_fingerprint=scan.base_fingerprint,
                        selection_indices_sha256=index_ref["indices_sha256"],
                        source_spec_sha256=expected_source_spec_sha256,
                    ),
                    "runtime_examples": len(indices),
                    "row_image_content": row_ref,
                    "unique_image_content": unique_ref,
                }
            sources[source_name] = {
                "components": _source_components(source_name, source_spec),
                **split_entries,
            }

        train_union = tuple(
            sorted(
                {
                    digest
                    for source_name in PERCEPTION_SOURCE_NAMES
                    for digest in logical_hashes[(source_name, "train")]
                }
            )
        )
        output_validation_union = tuple(
            sorted(
                {
                    digest
                    for source_name in PERCEPTION_SOURCE_NAMES
                    for digest in logical_hashes[(source_name, "validation")]
                }
            )
        )
        overlap = set(train_union).intersection(output_validation_union)
        if overlap or set(output_validation_union) != validation_union:
            raise ValueError("Perception train/validation union accounting differs")
        train_union_ref = _write_lines(
            staging / "inventories" / "train-union-unique.sha256",
            train_union,
        )
        validation_union_ref = _write_lines(
            staging / "inventories" / "validation-union-unique.sha256",
            output_validation_union,
        )
        for reference in (train_union_ref, validation_union_ref):
            reference["path"] = str(Path(reference["path"]).relative_to(staging).as_posix())

        image_path_signature_rows = []
        for path_value, (signature, digest) in sorted(hasher.path_results.items()):
            image_path_signature_rows.append(
                _canonical_bytes(
                    {
                        "path": path_value,
                        "size_bytes": signature.size_bytes,
                        "mtime_ns": signature.mtime_ns,
                        "ctime_ns": signature.ctime_ns,
                        "inode": signature.inode,
                        "device": signature.device,
                        "sha256": digest,
                    }
                ).decode("ascii")
            )
        image_path_signatures_ref = _write_lines(
            staging / "inventories" / "image-path-signatures.jsonl",
            image_path_signature_rows,
            allow_empty=True,
        )
        image_path_signatures_ref["path"] = str(
            Path(image_path_signatures_ref["path"]).relative_to(staging).as_posix()
        )

        manifest: dict[str, Any] = {
            "format": PERCEPTION_PROVENANCE_FORMAT,
            "version": PERCEPTION_PROVENANCE_VERSION,
            "status": "verified",
            "phase": "perception",
            "created_at": created_at,
            "builder": {
                "name": BUILDER_NAME,
                "version": BUILDER_VERSION,
                "script_sha256": builder_sha,
            },
            "source_spec": source_spec.as_canonical_dict(),
            "source_spec_sha256": expected_source_spec_sha256,
            "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
            "source_registry_sha256": current_registry_sha,
            "source_implementation_inventory": current_inventory,
            "finevision_materialization": materialization_reference,
            "image_path_signatures": image_path_signatures_ref,
            "validation_selection": state["validation_selection"],
            "sources": sources,
            "unions": {
                "train_unique_image_content": train_union_ref,
                "validation_unique_image_content": validation_union_ref,
                "overlap_count": 0,
            },
            "filtering": filtering,
        }
        manifest["content_sha256"] = _canonical_sha256(manifest)
        # Inventories bind the bytes observed above; reject any path-backed source that
        # changed while selections and manifest metadata were being materialized.
        hasher.validate_paths_unchanged()
        if _sha256_file(script_path) != builder_sha:
            raise ValueError("Provenance builder changed during artifact construction")
        manifest_path = staging / MANIFEST_NAME
        _atomic_write_json(manifest_path, manifest)
        validated = load_perception_provenance_manifest(manifest_path, require_complete=False)
        if validated.content_sha256 != manifest["content_sha256"]:
            raise ValueError("Runtime loader returned a different provenance content identity")
        # Close the raw-source snapshot window after all manifest construction and loader
        # validation. Annotation scans are rerun, identities are checked on both sides of
        # those scans, and every path-backed image is restatted immediately before COMPLETE
        # and the atomic no-replace publish.
        _validate_raw_scans_unchanged(
            raw_scans,
            source_spec=source_spec,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
        hasher.validate_paths_unchanged()
        _atomic_write(staging / "COMPLETE", (_sha256_file(manifest_path) + "\n").encode("ascii"))
        _fsync_directory(staging)
        _publish_staging(staging, destination)
        return manifest
    finally:
        fcntl.flock(staging_fd, fcntl.LOCK_UN)
        os.close(staging_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", required=True, help="Canonical source-spec JSON")
    parser.add_argument("--expected-source-spec-sha256", required=True)
    parser.add_argument("--expected-source-registry-sha256", required=True)
    parser.add_argument("--implementation-inventory", required=True, help="Pinned inventory JSON")
    parser.add_argument("--expected-implementation-inventory-sha256", required=True)
    parser.add_argument("--finevision-materialization-manifest", required=True)
    parser.add_argument("--expected-finevision-materialization-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline provenance builder CLI and return a process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source_spec = _load_source_spec(Path(args.source_spec).expanduser().resolve())
    materialization = _validate_finevision_materialization(
        args.finevision_materialization_manifest,
        args.expected_finevision_materialization_sha256,
    )
    source_spec = _bind_finevision_materialization(source_spec, materialization)
    # Pins are checked after artifact path derivation; a raw-root fallback can therefore
    # never retain the caller's source-spec identity.
    if source_spec.preprocessing_sha256 != args.expected_source_spec_sha256:
        raise ValueError("Pinned source-spec SHA-256 does not bind FineVision materialization")
    inventory = _read_json(Path(args.implementation_inventory).expanduser().resolve())
    if not isinstance(inventory, Mapping):
        raise TypeError("Implementation inventory must be a JSON object")
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=source_spec.tokenizer_id,
        revision=source_spec.tokenizer_revision,
        expected_fingerprint=source_spec.tokenizer_fingerprint,
        cache_dir=args.hf_cache_dir,
    )
    build_vision_alignment_perception_provenance(
        source_spec=source_spec,
        expected_source_spec_sha256=args.expected_source_spec_sha256,
        expected_source_registry_sha256=args.expected_source_registry_sha256,
        expected_implementation_inventory=inventory,
        expected_implementation_inventory_sha256=args.expected_implementation_inventory_sha256,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        token_ids=token_ids,
        finevision_materialization=materialization,
        resume=args.resume,
    )
    manifest_path = Path(args.output_dir).expanduser().resolve() / MANIFEST_NAME
    log.info("Published perception provenance at %s", manifest_path)
    log.info("Manifest SHA-256: %s", _sha256_file(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
