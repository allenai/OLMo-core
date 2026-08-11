#!/usr/bin/env python
"""Build the content-disjoint PixMoCap artifact for vision alignment.

The builder hashes the bytes of every unique image path, preserves the complete validation
split, and removes every training row whose image content occurs in validation. Work is written
to a resumable sibling staging directory and renamed to the requested output only after the
saved DatasetDict and all inventories have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MANIFEST_FORMAT = "vision_alignment_validation_manifest"
MANIFEST_VERSION = 3
BUILDER_FORMAT = "vision_alignment_pixmo_cap_builder"
BUILDER_VERSION = 1
BUILDER_SCRIPT = "src/scripts/data/build_vision_alignment_pixmo_cap.py"
FILTER_ALGORITHM = "preserve-validation-drop-train-content-overlap-v1"
IMAGE_HASH_ALGORITHM = "sha256"
ROW_PATH_ALGORITHM = "sha256-jsonl-index-image-path-v1"
ROW_CONTENT_ALGORITHM = "sha256-lines-v1"

CANONICAL_SOURCE_DATASET = "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/cap"
CANONICAL_TRAIN_FINGERPRINT = "db8d55b1f2bbb62e"
CANONICAL_TRAIN_EXAMPLES = 714_985
CANONICAL_VALIDATION_FINGERPRINT = "502dc5bb570bab20"
CANONICAL_VALIDATION_EXAMPLES = 2_048

_SPLITS = ("train", "validation")
_STATE_FORMAT = "vision_alignment_pixmo_cap_build_state"
_STATE_VERSION = 1
_CACHE_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_HASH_BATCH_SIZE = 2_048

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitScan:
    """Validated source rows and their deterministic path identity."""

    fingerprint: str
    paths: Tuple[str, ...]
    row_image_paths_sha256: str
    unique_image_paths: int

    @property
    def examples(self) -> int:
        """Return the number of rows in the split."""
        return len(self.paths)


@dataclass(frozen=True)
class FileSignature:
    """Filesystem identity used to validate a resumable hash-cache entry."""

    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int

    def as_tuple(self) -> Tuple[int, int, int, int, int]:
        """Return the signature in SQLite column order."""
        return (self.size_bytes, self.mtime_ns, self.ctime_ns, self.inode, self.device)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_image_paths_sha256(paths: Iterable[str]) -> str:
    """Hash indexed image paths using ``sha256-jsonl-index-image-path-v1``.

    Each row is compact, key-sorted JSON containing exactly ``index`` and ``image``, followed
    by one newline. The index makes order and row multiplicity part of the identity.
    """
    digest = hashlib.sha256()
    for index, image in enumerate(paths):
        digest.update((_canonical_json({"index": index, "image": image}) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON file {path}: {error}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_write_lines(path: Path, values: Iterable[str]) -> Tuple[str, int]:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with temporary.open("xb") as handle:
            for value in values:
                if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ValueError(f"Invalid SHA-256 inventory value {value!r}")
                handle.write(value.encode("ascii") + b"\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise ValueError(f"Refusing to write empty image inventory {path}")
        digest = _sha256_file(temporary)
        os.replace(temporary, path)
        return digest, count
    finally:
        if temporary.exists():
            temporary.unlink()


def _fingerprint(dataset: Any, split: str) -> str:
    value = getattr(dataset, "_fingerprint", None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"PixMoCap {split} split has no stable HuggingFace fingerprint")
    return value


def _validate_image_path(value: Any, split: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PixMoCap {split} row {index} has a blank or non-string image path")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(
            f"PixMoCap {split} row {index} image path must be canonical and absolute: {value!r}"
        )
    return value


def _validate_annotations(caption: Any, transcripts: Any, split: str, index: int) -> None:
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"PixMoCap {split} row {index} has a blank or non-string caption")
    if not isinstance(transcripts, (list, tuple)) or not transcripts:
        raise ValueError(f"PixMoCap {split} row {index} has no transcripts")
    for transcript_index, transcript in enumerate(transcripts):
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(
                f"PixMoCap {split} row {index} transcript {transcript_index} is blank or non-string"
            )


def _scan_split(dataset: Any, split: str, batch_size: int) -> SplitScan:
    required = {"image", "caption", "transcripts"}
    columns = set(getattr(dataset, "column_names", ()))
    if not required.issubset(columns):
        raise ValueError(f"PixMoCap {split} split lacks columns {sorted(required - columns)}")
    paths: List[str] = []
    selected = dataset.select_columns(sorted(required))
    for batch in selected.iter(batch_size=batch_size):
        sizes = {len(batch[field]) for field in required}
        if len(sizes) != 1:
            raise ValueError(f"PixMoCap {split} yielded inconsistent column batch sizes")
        for image, caption, transcripts in zip(
            batch["image"], batch["caption"], batch["transcripts"]
        ):
            index = len(paths)
            path = _validate_image_path(image, split, index)
            _validate_annotations(caption, transcripts, split, index)
            paths.append(path)
    if len(paths) != len(dataset) or not paths:
        raise ValueError(
            f"PixMoCap {split} scan returned {len(paths)} rows for dataset length {len(dataset)}"
        )
    return SplitScan(
        fingerprint=_fingerprint(dataset, split),
        paths=tuple(paths),
        row_image_paths_sha256=row_image_paths_sha256(paths),
        unique_image_paths=len(set(paths)),
    )


def _file_signature(path: Path) -> FileSignature:
    try:
        resolved = path.resolve(strict=True)
        info = path.stat()
    except OSError as error:
        raise ValueError(f"Could not stat PixMoCap image {path}: {error}") from error
    if resolved != path:
        raise ValueError(f"PixMoCap image path is not canonical: {path} resolves to {resolved}")
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"PixMoCap image is not a non-empty regular file: {path}")
    return FileSignature(
        size_bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        inode=info.st_ino,
        device=info.st_dev,
    )


def _sha256_file_stable(path: Path, expected: FileSignature) -> Tuple[str, FileSignature]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            opened_signature = FileSignature(
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_ino,
                opened.st_dev,
            )
            if opened_signature != expected:
                raise ValueError(f"PixMoCap image changed before hashing: {path}")
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        current = _file_signature(path)
    except OSError as error:
        raise ValueError(f"Could not hash PixMoCap image {path}: {error}") from error
    if current != expected:
        raise ValueError(f"PixMoCap image changed while hashing: {path}")
    return digest.hexdigest(), current


class ImageHashCache:
    """Durable path-deduplicated SHA-256 cache for resumable production builds."""

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
            "path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, "
            "ctime_ns INTEGER NOT NULL, inode INTEGER NOT NULL, device INTEGER NOT NULL, "
            "sha256 TEXT NOT NULL)"
        )
        expected = {
            "schema_version": str(_CACHE_SCHEMA_VERSION),
            "plan_sha256": plan_sha256,
        }
        current = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if current and current != expected:
            raise ValueError(f"Hash cache {path} belongs to a different build plan")
        if not current:
            self.connection.executemany("INSERT INTO metadata VALUES (?, ?)", expected.items())
            self.connection.commit()

    def lookup(self, path: str, signature: FileSignature) -> Optional[str]:
        """Return a cache hit only when all current filesystem identity fields match."""
        row = self.connection.execute(
            "SELECT size_bytes, mtime_ns, ctime_ns, inode, device, sha256 FROM hashes WHERE path=?",
            (path,),
        ).fetchone()
        if row is None or tuple(row[:5]) != signature.as_tuple():
            return None
        digest = row[5]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Hash cache {self.path} contains an invalid digest for {path}")
        return digest

    def store_many(self, rows: Sequence[Tuple[str, FileSignature, str]]) -> None:
        """Persist a batch of freshly computed file hashes."""
        self.connection.executemany(
            "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(path, *signature.as_tuple(), digest) for path, signature, digest in rows],
        )
        self.connection.commit()

    def close(self) -> None:
        """Checkpoint and close the cache before its staging directory is renamed."""
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def _hash_paths(
    paths: Sequence[str], cache_path: Path, plan_sha256: str, workers: int
) -> Dict[str, str]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    unique_paths = sorted(set(paths))
    hashes: Dict[str, str] = {}
    cache = ImageHashCache(cache_path, plan_sha256)
    try:
        missing: List[Tuple[str, FileSignature]] = []
        for index, value in enumerate(unique_paths, start=1):
            signature = _file_signature(Path(value))
            cached = cache.lookup(value, signature)
            if cached is None:
                missing.append((value, signature))
            else:
                hashes[value] = cached
            if index % 50_000 == 0:
                log.info(
                    "Validated hash-cache metadata for %d/%d unique paths", index, len(unique_paths)
                )
        log.info(
            "Hashing %d unique image files with %d workers (%d cache hits)",
            len(missing),
            workers,
            len(hashes),
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pixmo-sha256") as pool:
            for offset in range(0, len(missing), _HASH_BATCH_SIZE):
                batch = missing[offset : offset + _HASH_BATCH_SIZE]
                results = list(
                    pool.map(lambda item: _sha256_file_stable(Path(item[0]), item[1]), batch)
                )
                stored = [
                    (path, signature, result[0])
                    for (path, signature), result in zip(batch, results)
                ]
                cache.store_many(stored)
                hashes.update((path, digest) for path, _, digest in stored)
                log.info(
                    "Hashed %d/%d uncached image files",
                    min(offset + len(batch), len(missing)),
                    len(missing),
                )
    finally:
        cache.close()
    if set(hashes) != set(unique_paths):
        raise ValueError("Image hash cache did not produce an exhaustive unique-path inventory")
    return hashes


def _content_summary(paths: Iterable[str], image_hashes: Mapping[str, str]) -> Tuple[str, int]:
    digest = hashlib.sha256()
    unique = set()
    for path in paths:
        try:
            image_hash = image_hashes[path]
        except KeyError:
            raise ValueError(f"No actual-byte image hash was recorded for {path}") from None
        digest.update(image_hash.encode("ascii") + b"\n")
        unique.add(image_hash)
    return digest.hexdigest(), len(unique)


def _split_manifest(scan: SplitScan, image_hashes: Mapping[str, str]) -> Dict[str, Any]:
    content_digest, unique_content = _content_summary(scan.paths, image_hashes)
    return {
        "dataset_fingerprint": scan.fingerprint,
        "examples": scan.examples,
        "row_image_paths_sha256": scan.row_image_paths_sha256,
        "row_image_content_sha256": content_digest,
        "unique_image_paths": scan.unique_image_paths,
        "unique_image_content": unique_content,
    }


def _prepare_staging(output_dir: Path, state: Mapping[str, Any], resume: bool) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite immutable artifact {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building"
    state_path = staging / "build-state.json"
    if staging.exists():
        if not resume:
            raise FileExistsError(
                f"Recoverable staging directory {staging} exists; inspect it and rerun with --resume"
            )
        if not state_path.is_file() or _read_json(state_path) != state:
            raise ValueError(f"Staging directory {staging} does not match the requested build")
    else:
        if resume:
            raise FileNotFoundError(f"No recoverable staging directory exists at {staging}")
        staging.mkdir()
        _atomic_write_json(state_path, state)
    return staging


def _recover_incomplete_dataset(staging: Path) -> None:
    incomplete = staging / "dataset.incomplete"
    if not incomplete.exists():
        return
    recovery = staging / "recovery"
    recovery.mkdir(exist_ok=True)
    counter = 0
    while True:
        destination = recovery / f"dataset.incomplete.{counter:04d}"
        if not destination.exists():
            os.replace(incomplete, destination)
            log.warning("Preserved partial DatasetDict at %s", destination)
            return
        counter += 1


def _load_dataset_dict(path: Path) -> Any:
    from olmo_core.data.multimodal.dataset_compat import load_from_disk_compat

    dataset = load_from_disk_compat(str(path))
    try:
        splits = set(dataset.keys())
    except AttributeError:
        raise ValueError(f"PixMoCap path {path} is not a DatasetDict") from None
    if splits != set(_SPLITS):
        raise ValueError(
            f"PixMoCap DatasetDict {path} must contain exactly {_SPLITS}; found {sorted(splits)}"
        )
    return dataset


def _validate_source_pins(
    dataset: Any,
    expected_train_fingerprint: str,
    expected_train_examples: int,
    expected_validation_fingerprint: str,
    expected_validation_examples: int,
) -> None:
    expected = {
        "train": (expected_train_fingerprint, expected_train_examples),
        "validation": (expected_validation_fingerprint, expected_validation_examples),
    }
    for split, (fingerprint, examples) in expected.items():
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"Expected {split} fingerprint must be non-empty")
        if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
            raise ValueError(f"Expected {split} examples must be a positive integer")
        actual = (_fingerprint(dataset[split], split), len(dataset[split]))
        if actual != (fingerprint, examples):
            raise ValueError(
                f"PixMoCap {split} source pin mismatch: expected {(fingerprint, examples)!r}, "
                f"found {actual!r}"
            )


def _save_dataset_dict(
    source: Any,
    keep_train_indices: Sequence[int],
    staging: Path,
    max_shard_size: str,
    save_num_proc: Optional[int],
) -> None:
    from datasets import DatasetDict

    final = staging / "dataset"
    if final.exists():
        return
    _recover_incomplete_dataset(staging)
    incomplete = staging / "dataset.incomplete"
    filtered = DatasetDict(
        {
            "train": source["train"].select(keep_train_indices),
            "validation": source["validation"],
        }
    )
    kwargs: Dict[str, Any] = {"max_shard_size": max_shard_size}
    if save_num_proc is not None:
        if save_num_proc < 1:
            raise ValueError("save_num_proc must be at least 1")
        kwargs["num_proc"] = save_num_proc
    filtered.save_to_disk(str(incomplete), **kwargs)
    os.replace(incomplete, final)


def _ordered_hashes(paths: Iterable[str], image_hashes: Mapping[str, str]) -> Iterable[str]:
    for path in paths:
        yield image_hashes[path]


def build_pixmo_cap_artifact(
    *,
    source_dataset_path: str,
    output_dir: str,
    expected_train_fingerprint: str,
    expected_train_examples: int,
    expected_validation_fingerprint: str,
    expected_validation_examples: int,
    workers: int = 32,
    scan_batch_size: int = 4_096,
    max_shard_size: str = "2GB",
    save_num_proc: Optional[int] = None,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Build and atomically publish a content-disjoint PixMoCap DatasetDict.

    :param source_dataset_path: Pinned source DatasetDict path.
    :param output_dir: New immutable artifact root. The DatasetDict is written below
        ``dataset/`` and the v3 validation manifest beside it.
    :param expected_train_fingerprint: Required live source train fingerprint.
    :param expected_train_examples: Required live source train row count.
    :param expected_validation_fingerprint: Required live validation fingerprint.
    :param expected_validation_examples: Required live validation row count.
    :param workers: Number of image-file hashing threads.
    :param scan_batch_size: Arrow annotation scan batch size.
    :param max_shard_size: HuggingFace output shard-size limit.
    :param save_num_proc: Optional HuggingFace save worker count.
    :param resume: Resume an exactly matching sibling ``.building`` directory.

    :returns: The strict v3 manifest that was published.

    :raises ValueError: If source pins, annotations, images, inventories, or saved output fail
        validation.
    :raises FileExistsError: If output exists or unapproved partial staging exists.
    """
    if workers < 1 or scan_batch_size < 1:
        raise ValueError("workers and scan_batch_size must be positive")
    source_path = Path(source_dataset_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source_path == destination or source_path in destination.parents:
        raise ValueError("Output must not equal or be nested inside the source DatasetDict")
    script_path = Path(__file__).resolve()
    builder_sha256 = _sha256_file(script_path)
    state: Dict[str, Any] = {
        "format": _STATE_FORMAT,
        "version": _STATE_VERSION,
        "builder_script_sha256": builder_sha256,
        "source_dataset_path": str(source_path),
        "output_dir": str(destination),
        "materialization": {
            "max_shard_size": max_shard_size,
            "save_num_proc": save_num_proc,
        },
        "expected_splits": {
            "train": {
                "dataset_fingerprint": expected_train_fingerprint,
                "examples": expected_train_examples,
            },
            "validation": {
                "dataset_fingerprint": expected_validation_fingerprint,
                "examples": expected_validation_examples,
            },
        },
    }
    plan_sha256 = hashlib.sha256(_canonical_json(state).encode("utf-8")).hexdigest()

    source = _load_dataset_dict(source_path)
    _validate_source_pins(
        source,
        expected_train_fingerprint,
        expected_train_examples,
        expected_validation_fingerprint,
        expected_validation_examples,
    )
    staging = _prepare_staging(destination, state, resume)

    log.info("Scanning and validating PixMoCap annotations")
    source_scans = {split: _scan_split(source[split], split, scan_batch_size) for split in _SPLITS}
    all_paths = source_scans["train"].paths + source_scans["validation"].paths
    image_hashes = _hash_paths(
        all_paths, staging / "image-hash-cache.sqlite3", plan_sha256, workers
    )

    validation_hash_rows = tuple(_ordered_hashes(source_scans["validation"].paths, image_hashes))
    validation_hash_set = set(validation_hash_rows)
    validation_duplicate_examples = len(validation_hash_rows) - len(validation_hash_set)
    if validation_duplicate_examples:
        log.warning(
            "Preserving %d repeated validation rows with duplicate actual image content",
            validation_duplicate_examples,
        )

    train_hash_set = set(_ordered_hashes(source_scans["train"].paths, image_hashes))
    source_overlap = train_hash_set & validation_hash_set
    keep_train_indices = [
        index
        for index, path in enumerate(source_scans["train"].paths)
        if image_hashes[path] not in validation_hash_set
    ]
    removed_train_examples = source_scans["train"].examples - len(keep_train_indices)
    if not keep_train_indices:
        raise ValueError("Content-disjoint filtering removed every PixMoCap training row")
    if bool(source_overlap) != bool(removed_train_examples):
        raise ValueError("Image-overlap accounting is internally inconsistent")

    _save_dataset_dict(
        source,
        keep_train_indices,
        staging,
        max_shard_size=max_shard_size,
        save_num_proc=save_num_proc,
    )
    output = _load_dataset_dict(staging / "dataset")
    output_scans = {split: _scan_split(output[split], split, scan_batch_size) for split in _SPLITS}
    expected_output_paths = {
        "train": tuple(source_scans["train"].paths[index] for index in keep_train_indices),
        "validation": source_scans["validation"].paths,
    }
    for split in _SPLITS:
        if output_scans[split].paths != expected_output_paths[split]:
            raise ValueError(f"Saved output {split} rows differ from the intended filtered split")
    output_hash_sets = {
        split: set(_ordered_hashes(output_scans[split].paths, image_hashes)) for split in _SPLITS
    }
    output_overlap = output_hash_sets["train"] & output_hash_sets["validation"]
    if output_overlap:
        raise ValueError("Filtered output still contains train/validation image-content overlap")

    inventory_meta: Dict[str, Dict[str, Any]] = {}
    output_split_meta: Dict[str, Dict[str, Any]] = {}
    for split in _SPLITS:
        inventory_name = f"{split}-images.sha256"
        inventory_digest, inventory_count = _atomic_write_lines(
            staging / inventory_name, sorted(output_hash_sets[split])
        )
        row_name = f"{split}-row-images.sha256"
        row_digest, row_count = _atomic_write_lines(
            staging / row_name,
            _ordered_hashes(output_scans[split].paths, image_hashes),
        )
        if row_count != output_scans[split].examples:
            raise ValueError(f"Ordered {split} row-content inventory is not exhaustive")
        split_meta = _split_manifest(output_scans[split], image_hashes)
        if row_digest != split_meta["row_image_content_sha256"]:
            raise ValueError(f"Ordered {split} row-content inventory digest mismatch")
        split_meta["row_image_content_path"] = row_name
        output_split_meta[split] = split_meta
        inventory_meta[split] = {
            "path": inventory_name,
            "sha256": inventory_digest,
            "count": inventory_count,
        }

    source_split_meta = {
        split: _split_manifest(source_scans[split], image_hashes) for split in _SPLITS
    }
    if (
        source_split_meta["validation"]["row_image_paths_sha256"]
        != output_split_meta["validation"]["row_image_paths_sha256"]
        or source_split_meta["validation"]["row_image_content_sha256"]
        != output_split_meta["validation"]["row_image_content_sha256"]
    ):
        raise ValueError("Saved output changed validation path or content order")

    manifest: Dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "builder": {
            "format": BUILDER_FORMAT,
            "version": BUILDER_VERSION,
            "script": BUILDER_SCRIPT,
            "script_sha256": builder_sha256,
            "filter_algorithm": FILTER_ALGORITHM,
            "image_hash_algorithm": IMAGE_HASH_ALGORITHM,
            "row_image_paths_algorithm": ROW_PATH_ALGORITHM,
            "row_image_content_algorithm": ROW_CONTENT_ALGORITHM,
        },
        "source": {
            "dataset_path": str(source_path),
            "splits": source_split_meta,
        },
        "output": {
            "dataset_path": "dataset",
            "splits": output_split_meta,
        },
        "inventories": inventory_meta,
        "filtering": {
            "source_overlap_unique_images": len(source_overlap),
            "removed_train_examples": removed_train_examples,
            "validation_duplicate_examples": validation_duplicate_examples,
            "output_overlap_unique_images": len(output_overlap),
        },
    }
    if _sha256_file(script_path) != builder_sha256:
        raise ValueError("Builder script changed during artifact construction")
    manifest_path = staging / "vision-alignment-validation-manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)
    _atomic_write(staging / "COMPLETE", (manifest_sha256 + "\n").encode("ascii"))
    os.replace(staging, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New immutable artifact root")
    parser.add_argument("--source-dataset", default=CANONICAL_SOURCE_DATASET)
    parser.add_argument("--expected-train-fingerprint", default=CANONICAL_TRAIN_FINGERPRINT)
    parser.add_argument("--expected-train-examples", type=int, default=CANONICAL_TRAIN_EXAMPLES)
    parser.add_argument(
        "--expected-validation-fingerprint", default=CANONICAL_VALIDATION_FINGERPRINT
    )
    parser.add_argument(
        "--expected-validation-examples", type=int, default=CANONICAL_VALIDATION_EXAMPLES
    )
    parser.add_argument(
        "--workers", type=int, default=min(32, os.cpu_count() or 1), help="Image hash threads"
    )
    parser.add_argument("--scan-batch-size", type=int, default=4_096)
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--save-num-proc", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact build plan from the sibling .building directory",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the canonical PixMoCap artifact builder CLI."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = build_pixmo_cap_artifact(
        source_dataset_path=args.source_dataset,
        output_dir=args.output_dir,
        expected_train_fingerprint=args.expected_train_fingerprint,
        expected_train_examples=args.expected_train_examples,
        expected_validation_fingerprint=args.expected_validation_fingerprint,
        expected_validation_examples=args.expected_validation_examples,
        workers=args.workers,
        scan_batch_size=args.scan_batch_size,
        max_shard_size=args.max_shard_size,
        save_num_proc=args.save_num_proc,
        resume=args.resume,
    )
    output = Path(args.output_dir).expanduser().resolve()
    manifest_path = output / "vision-alignment-validation-manifest.json"
    log.info("Published content-disjoint PixMoCap artifact at %s", output)
    log.info("Validation manifest: %s (sha256=%s)", manifest_path, _sha256_file(manifest_path))
    log.info(
        "Removed %d train examples across %d overlapping image contents",
        manifest["filtering"]["removed_train_examples"],
        manifest["filtering"]["source_overlap_unique_images"],
    )


if __name__ == "__main__":
    main()
