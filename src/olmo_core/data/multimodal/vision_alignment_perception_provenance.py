"""Strict runtime binding for perception train/validation image provenance.

The offline builder hashes encoded source image bytes, constructs a union-disjoint validation
population, and writes explicit selected-index files. This module verifies that immutable
artifact and wraps the exact raw adapters with those selections. It intentionally does not
rehash the full image store at every training startup; the manifest instead pins the complete
byte inventories and requires their files to remain immutable.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .finevision import FINEVISION_ROOT
from .vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset,
)
from .vision_alignment_sources import runtime_dataset_fingerprint

__all__ = [
    "PERCEPTION_PROVENANCE_FORMAT",
    "PERCEPTION_PROVENANCE_MANIFEST",
    "PERCEPTION_PROVENANCE_VERSION",
    "PERCEPTION_SOURCE_NAMES",
    "FineVisionMaterializationReference",
    "PerceptionProvenanceManifest",
    "SelectedVisionAlignmentDataset",
    "build_selected_perception_dataset",
    "image_reference_sha256",
    "load_perception_provenance_manifest",
    "perception_annotation_content_sha256",
    "selected_dataset_fingerprint",
    "validate_finevision_materialization",
]

PERCEPTION_PROVENANCE_FORMAT = "vision_alignment_perception_image_provenance"
PERCEPTION_PROVENANCE_MANIFEST = "vision-alignment-perception-provenance.json"
PERCEPTION_PROVENANCE_VERSION = 2
VALIDATION_IMAGE_CONTENTS_PER_SOURCE = 512
PERCEPTION_SOURCE_NAMES = (
    "audited_alignment",
    "cosyn_point",
    "ocr_document",
    "pixmo_caption",
    "pixmo_points_basic",
    "pixmo_points_high_frequency",
    "pixmo_transcript",
    "scalar_count",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATASET_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16,64}")
_ROOT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "phase",
        "created_at",
        "builder",
        "source_spec",
        "source_spec_sha256",
        "source_registry_version",
        "source_registry_sha256",
        "source_implementation_inventory",
        "finevision_materialization",
        "image_path_signatures",
        "validation_selection",
        "sources",
        "unions",
        "filtering",
        "content_sha256",
    }
)
_SOURCE_FIELDS = frozenset({"components", "train", "validation"})
_SPLIT_FIELDS = frozenset(
    {
        "physical_split",
        "base_annotation_sha256",
        "base_dataset_fingerprint",
        "base_examples",
        "selection",
        "runtime_dataset_fingerprint",
        "runtime_examples",
        "row_image_content",
        "unique_image_content",
    }
)
_FILE_REF_FIELDS = frozenset({"path", "sha256", "count"})
_PATH_SIGNATURE_FIELDS = frozenset(
    {"path", "size_bytes", "mtime_ns", "ctime_ns", "inode", "device", "sha256"}
)
_SELECTION_FIELDS = _FILE_REF_FIELDS | {"indices_sha256"}
_UNION_FIELDS = frozenset(
    {"train_unique_image_content", "validation_unique_image_content", "overlap_count"}
)
_FILTER_FIELDS = frozenset(
    {"candidate_train_examples", "removed_train_examples", "output_train_examples"}
)
_BUILDER_FIELDS = frozenset({"name", "version", "script_sha256"})
_FINEVISION_MATERIALIZATION_FORMAT = "vision_alignment_finevision_materialization"
_FINEVISION_MATERIALIZATION_VERSION = 1
_FINEVISION_MATERIALIZATION_MANIFEST = "vision-alignment-finevision-materialization.json"
_FINEVISION_REFERENCE_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "content_sha256",
        "visualweb_fingerprint",
        "geo170k_fingerprint",
    }
)
_FINEVISION_MANIFEST_FIELDS = frozenset(
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
    }
)
_FINEVISION_SOURCE_FIELDS = frozenset(
    {
        "name",
        "output_name",
        "shards",
        "shard_count",
        "rows",
        "physical_schema_sha256",
        "source_metadata_sha256",
    }
)
_FINEVISION_OUTPUT_FIELDS = frozenset(
    {
        "name",
        "path",
        "rows",
        "dataset_fingerprint",
        "dataset_info_sha256",
        "physical_schema_sha256",
        "shards",
    }
)
_FINEVISION_SHARD_FIELDS = frozenset({"path", "bytes", "rows", "sha256"})
_VALIDATION_SELECTION_FIELDS = frozenset({"algorithm", "image_contents_per_source"})
_VALIDATION_SELECTION_ALGORITHM = "sha256-ranked-distinct-content-representatives-v1"
_FINEVISION_CANONICAL_SOURCES = (
    ("visualwebinstruct(filtered)", "visualwebinstruct-filtered", 73, 263_581),
    ("geo170k(align)", "geo170k-align", 1, 35_297),
)
_FINEVISION_VALIDATION_CACHE: Dict[
    tuple[str, str, str, str, str],
    tuple["FineVisionMaterializationReference", Tuple[Tuple[str, int, int, int, int], ...]],
] = {}


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
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def image_reference_sha256(reference: Any) -> str:
    """Hash the encoded bytes consumed by one strict perception image reference.

    :param reference: An absolute path, raw bytes, or Hugging Face image struct. Image
        structs follow runtime decoding semantics by preferring non-empty embedded bytes.
    :returns: The SHA-256 of the encoded image bytes.
    :raises ValueError: If the reference has no immutable supported byte representation.
    """
    if isinstance(reference, Mapping):
        embedded = reference.get("bytes")
        if isinstance(embedded, (bytes, bytearray, memoryview)) and embedded:
            return hashlib.sha256(bytes(embedded)).hexdigest()
        reference = reference.get("path")
    if isinstance(reference, (bytes, bytearray, memoryview)):
        if not reference:
            raise ValueError("Image byte reference is empty")
        return hashlib.sha256(bytes(reference)).hexdigest()
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Unsupported image reference type {type(reference)!r}")
    path = Path(reference).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Image reference must be an existing absolute file: {reference!r}")
    return _sha256_file(path)


def perception_annotation_content_sha256(dataset: Any) -> str:
    """Return a full ordered non-image annotation identity for a perception adapter.

    Native PixMo adapters implement an explicit annotation-only scan. OCR computes the same
    identity while loading its source JSON, and audited FineVision relies on the fully hashed
    materialized Arrow receipt. All results are required to be lowercase SHA-256 values.

    :param dataset: One raw perception dataset before provenance row selection.
    :returns: Its full ordered annotation/config SHA-256.
    :raises ValueError: If the adapter exposes no approved full annotation identity.
    """
    method = getattr(dataset, "annotation_content_sha256", None)
    value = method() if callable(method) else None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("Perception adapter lacks a full annotation-content SHA-256")
    return value


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON repeats key {key!r}")
        value[key] = item
    return value


def _exact_mapping(value: Any, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _dataset_fingerprint(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DATASET_FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 16- to 64-hex dataset fingerprint")
    return value


def _count(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _artifact_path(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a non-empty relative path")
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{name} escapes the provenance artifact root")
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _validate_file_reference(
    root: Path,
    value: Any,
    *,
    name: str,
    selection: bool = False,
    minimum_count: int = 1,
) -> tuple[Mapping[str, Any], bytes]:
    fields = _SELECTION_FIELDS if selection else _FILE_REF_FIELDS
    reference = _exact_mapping(value, fields, name=name)
    path = _artifact_path(root, reference["path"], name=f"{name}.path")
    expected_sha = _sha256(reference["sha256"], name=f"{name}.sha256")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{name} is not readable: {path}") from error
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    _count(reference["count"], name=f"{name}.count", minimum=minimum_count)
    if selection:
        _sha256(reference["indices_sha256"], name=f"{name}.indices_sha256")
    return reference, raw


def _read_indices(
    root: Path,
    value: Any,
    *,
    name: str,
    base_size: int,
) -> Tuple[int, ...]:
    reference, raw = _validate_file_reference(root, value, name=name, selection=True)
    try:
        lines = raw.decode("utf-8").splitlines()
        indices = tuple(int(line) for line in lines)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not a newline-delimited integer selection") from error
    if (
        len(indices) != reference["count"]
        or len(set(indices)) != len(indices)
        or tuple(sorted(indices)) != indices
        or any(index < 0 or index >= base_size for index in indices)
    ):
        raise ValueError(f"{name} must contain sorted unique, in-bounds indices matching its count")
    if _canonical_sha256(list(indices)) != reference["indices_sha256"]:
        raise ValueError(f"{name} index identity differs")
    return indices


def _read_hash_inventory(
    root: Path,
    value: Any,
    *,
    name: str,
    unique: bool,
) -> Tuple[str, ...]:
    reference, raw = _validate_file_reference(root, value, name=name)
    try:
        hashes = tuple(raw.decode("utf-8").splitlines())
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not readable UTF-8") from error
    if len(hashes) != reference["count"] or any(
        _SHA256_RE.fullmatch(value) is None for value in hashes
    ):
        raise ValueError(f"{name} contains invalid hash rows or count")
    if unique and tuple(sorted(set(hashes))) != hashes:
        raise ValueError(f"{name} must contain sorted unique hashes")
    return hashes


@dataclass(frozen=True)
class PerceptionPathSignature:
    """Builder-time identity and content digest for one path-backed image."""

    path: str
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int
    sha256: str


def _read_path_signatures(root: Path, value: Any) -> Tuple[PerceptionPathSignature, ...]:
    reference, raw = _validate_file_reference(
        root, value, name="image_path_signatures", minimum_count=0
    )
    records: list[PerceptionPathSignature] = []
    for ordinal, raw_line in enumerate(raw.splitlines()):
        try:
            row = json.loads(raw_line, object_pairs_hook=_strict_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"image_path_signatures row {ordinal} is invalid JSON") from error
        row = _exact_mapping(
            row,
            _PATH_SIGNATURE_FIELDS,
            name=f"image_path_signatures[{ordinal}]",
        )
        path_value = row["path"]
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError("Every image-path signature must identify an absolute path")
        records.append(
            PerceptionPathSignature(
                path=path_value,
                size_bytes=_count(
                    row["size_bytes"], name=f"image_path_signatures[{ordinal}].size", minimum=1
                ),
                mtime_ns=_count(row["mtime_ns"], name=f"image_path_signatures[{ordinal}].mtime_ns"),
                ctime_ns=_count(row["ctime_ns"], name=f"image_path_signatures[{ordinal}].ctime_ns"),
                inode=_count(row["inode"], name=f"image_path_signatures[{ordinal}].inode"),
                device=_count(row["device"], name=f"image_path_signatures[{ordinal}].device"),
                sha256=_sha256(row["sha256"], name=f"image_path_signatures[{ordinal}].sha256"),
            )
        )
    if len(records) != reference["count"] or tuple(record.path for record in records) != tuple(
        sorted({record.path for record in records})
    ):
        raise ValueError("Image-path signatures must be sorted, unique, and match their count")
    return tuple(records)


def _validate_path_signature_reference(root: Path, value: Any) -> None:
    """Validate only the manifest-level identity of the path-signature inventory."""
    reference = _exact_mapping(value, _FILE_REF_FIELDS, name="image_path_signatures")
    _artifact_path(root, reference["path"], name="image_path_signatures.path")
    _sha256(reference["sha256"], name="image_path_signatures.sha256")
    _count(reference["count"], name="image_path_signatures.count")


def selected_dataset_fingerprint(
    *,
    source_name: str,
    logical_split: str,
    physical_split: str,
    base_fingerprint: str,
    selection_indices_sha256: str,
    source_spec_sha256: str,
) -> str:
    """Return the canonical fingerprint for one selected runtime dataset."""
    return _canonical_sha256(
        {
            "version": "vision-alignment-perception-selected-v1",
            "source_name": source_name,
            "logical_split": logical_split,
            "physical_split": physical_split,
            "base_fingerprint": base_fingerprint,
            "selection_indices_sha256": selection_indices_sha256,
            "source_spec_sha256": source_spec_sha256,
        }
    )


def _finevision_output_fingerprint(
    *,
    source_name: str,
    rows: int,
    physical_schema_sha256: str,
    shards: Sequence[Mapping[str, Any]],
    dataset_info_sha256: str,
) -> str:
    """Recompute the materializer's path-independent output identity locally."""
    return _canonical_sha256(
        {
            "version": "vision-alignment-finevision-arrow-content-v1",
            "source_name": source_name,
            "rows": rows,
            "physical_schema_sha256": physical_schema_sha256,
            "shards": [{"rows": shard["rows"], "sha256": shard["sha256"]} for shard in shards],
            "dataset_info_sha256": dataset_info_sha256,
        }
    )


@dataclass(frozen=True)
class FineVisionMaterializationReference:
    """Validated upstream receipt and live FineVision output identities."""

    path: Path
    sha256: str
    content_sha256: str
    visualweb_fingerprint: str
    geo170k_fingerprint: str


def validate_finevision_materialization(
    provenance_root: Path,
    value: Any,
    source_spec: VisionAlignmentPerceptionSourceSpec,
) -> FineVisionMaterializationReference:
    """Validate copied materialization receipt and rehash both live Arrow outputs.

    The implementation reproduces the materializer's output fingerprint without importing
    producer code. The copied receipt binds its recorded producer metadata, while the source
    spec supplies the absolute paths whose shard, receipt, and ``dataset_info.json`` bytes are
    checked before an audited-alignment adapter can be built.

    :param provenance_root: Root directory of the perception provenance artifact.
    :param value: Strict ``finevision_materialization`` manifest field.
    :param source_spec: Canonical perception source specification containing live paths.
    :returns: The fully validated immutable materialization reference.
    :raises ValueError: If any receipt, path, byte, count, or fingerprint differs.
    """
    reference = _exact_mapping(
        value,
        _FINEVISION_REFERENCE_FIELDS,
        name="finevision_materialization",
    )
    if reference["path"] != f"upstream/{_FINEVISION_MATERIALIZATION_MANIFEST}":
        raise ValueError("FineVision materialization receipt path is not canonical")
    copied_path = _artifact_path(
        provenance_root,
        reference["path"],
        name="finevision_materialization.path",
    )
    expected_raw_sha = _sha256(reference["sha256"], name="finevision_materialization.sha256")
    try:
        copied_raw = copied_path.read_bytes()
        manifest = json.loads(copied_raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Copied FineVision materialization receipt is invalid") from error
    if hashlib.sha256(copied_raw).hexdigest() != expected_raw_sha:
        raise ValueError("Copied FineVision materialization raw SHA-256 differs")
    manifest = _exact_mapping(
        manifest,
        _FINEVISION_MANIFEST_FIELDS,
        name="FineVision materialization receipt",
    )
    if (
        manifest["format"] != _FINEVISION_MATERIALIZATION_FORMAT
        or _count(manifest["version"], name="FineVision materialization version", minimum=1)
        != _FINEVISION_MATERIALIZATION_VERSION
        or manifest["status"] != "verified"
    ):
        raise ValueError("FineVision materialization receipt identity or status differs")
    expected_content_sha = _sha256(
        reference["content_sha256"],
        name="finevision_materialization.content_sha256",
    )
    manifest_content_sha = _sha256(
        manifest["content_sha256"],
        name="FineVision receipt content_sha256",
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("content_sha256")
    if (
        manifest_content_sha != expected_content_sha
        or _canonical_sha256(unsigned_manifest) != manifest_content_sha
    ):
        raise ValueError("FineVision materialization content SHA-256 differs")
    if not isinstance(manifest["created_at"], str):
        raise ValueError("FineVision materialization created_at must be ISO-8601")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("FineVision materialization created_at is not ISO-8601") from error
    if created_at.tzinfo is None:
        raise ValueError("FineVision materialization created_at must include a timezone")
    materializer_sha = _sha256(
        manifest["builder_sha256"],
        name="FineVision materializer builder_sha256",
    )

    source_root = Path(str(manifest["source_root"])).expanduser().resolve()
    if (
        source_root != Path(FINEVISION_ROOT).expanduser().resolve()
        or Path(source_spec.finevision_root).expanduser().resolve() != source_root
    ):
        raise ValueError("FineVision materialization source root differs from the source spec")
    visualweb_path = Path(str(source_spec.finevision_visualweb_path)).expanduser().resolve()
    geo170k_path = Path(str(source_spec.finevision_geo170k_path)).expanduser().resolve()
    if visualweb_path.parent != geo170k_path.parent:
        raise ValueError("FineVision materialized outputs must share one artifact root")
    artifact_root = visualweb_path.parent
    external_manifest = artifact_root / _FINEVISION_MATERIALIZATION_MANIFEST
    complete_path = artifact_root / "COMPLETE"
    if (
        not external_manifest.is_file()
        or external_manifest.read_bytes() != copied_raw
        or not complete_path.is_file()
        or complete_path.read_text().strip() != expected_raw_sha
    ):
        raise ValueError("Live FineVision materialization receipt or COMPLETE marker differs")

    visualweb_fingerprint = _sha256(
        reference["visualweb_fingerprint"],
        name="finevision_materialization.visualweb_fingerprint",
    )
    geo170k_fingerprint = _sha256(
        reference["geo170k_fingerprint"],
        name="finevision_materialization.geo170k_fingerprint",
    )
    if (
        visualweb_fingerprint != source_spec.finevision_visualweb_fingerprint
        or geo170k_fingerprint != source_spec.finevision_geo170k_fingerprint
    ):
        raise ValueError("FineVision materialization fingerprints differ from the source spec")

    def stat_signature(paths: Sequence[Path]) -> Tuple[Tuple[str, int, int, int, int], ...]:
        """Return stable stat signatures for materialized FineVision files."""
        rows = []
        for path in paths:
            try:
                info = path.stat()
            except OSError as error:
                raise ValueError(
                    f"FineVision materialization path is unavailable: {path}"
                ) from error
            if not path.is_file():
                raise ValueError(f"FineVision materialization path is not a file: {path}")
            rows.append((str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino))
        return tuple(rows)

    cache_paths = [copied_path, external_manifest, complete_path]
    for output_root in (visualweb_path, geo170k_path):
        cache_paths.extend(sorted(output_root.glob("data-*.arrow")))
        cache_paths.extend(sorted(output_root.glob("data-*.receipt.json")))
        cache_paths.append(output_root / "dataset_info.json")
    cache_key = (
        str(copied_path),
        expected_raw_sha,
        str(visualweb_path),
        str(geo170k_path),
        materializer_sha,
    )
    current_signature = stat_signature(cache_paths)
    cached = _FINEVISION_VALIDATION_CACHE.get(cache_key)
    if cached is not None and cached[1] == current_signature:
        return cached[0]

    raw_sources = manifest["sources"]
    outputs = manifest["outputs"]
    if (
        not isinstance(raw_sources, list)
        or not isinstance(outputs, list)
        or len(raw_sources) != len(_FINEVISION_CANONICAL_SOURCES)
        or len(outputs) != len(_FINEVISION_CANONICAL_SOURCES)
    ):
        raise ValueError("FineVision materialization must contain the exact two sources")
    output_roots = (visualweb_path, geo170k_path)
    expected_fingerprints = (visualweb_fingerprint, geo170k_fingerprint)
    for position, (
        (name, output_name, shard_count, rows),
        output_root,
        expected_fingerprint,
    ) in enumerate(zip(_FINEVISION_CANONICAL_SOURCES, output_roots, expected_fingerprints)):
        source = _exact_mapping(
            raw_sources[position],
            _FINEVISION_SOURCE_FIELDS,
            name=f"FineVision sources[{position}]",
        )
        output = _exact_mapping(
            outputs[position],
            _FINEVISION_OUTPUT_FIELDS,
            name=f"FineVision outputs[{position}]",
        )
        if (
            source["name"] != name
            or source["output_name"] != output_name
            or source["shard_count"] != shard_count
            or source["rows"] != rows
            or output["name"] != name
            or output["path"] != output_name
            or output["rows"] != rows
            or output_root.name != output_name
        ):
            raise ValueError(f"FineVision materialization pins differ for {name!r}")
        source_schema = _sha256(
            source["physical_schema_sha256"],
            name=f"FineVision {name} source schema",
        )
        _sha256(
            source["source_metadata_sha256"],
            name=f"FineVision {name} source metadata",
        )
        if output["physical_schema_sha256"] != source_schema:
            raise ValueError(f"FineVision materialization schema differs for {name!r}")
        if output["dataset_fingerprint"] != expected_fingerprint:
            raise ValueError(f"FineVision materialization fingerprint differs for {name!r}")

        source_shards = source["shards"]
        output_shards = output["shards"]
        if (
            not isinstance(source_shards, list)
            or not isinstance(output_shards, list)
            or len(source_shards) != shard_count
            or len(output_shards) != shard_count
        ):
            raise ValueError(f"FineVision shard inventory differs for {name!r}")
        source_rows = 0
        output_rows = 0
        expected_live_names = []
        for shard_index, (source_shard_value, output_shard_value) in enumerate(
            zip(source_shards, output_shards)
        ):
            source_shard = _exact_mapping(
                source_shard_value,
                _FINEVISION_SHARD_FIELDS,
                name=f"FineVision {name} source shard {shard_index}",
            )
            output_shard = _exact_mapping(
                output_shard_value,
                _FINEVISION_SHARD_FIELDS,
                name=f"FineVision {name} output shard {shard_index}",
            )
            expected_output_relative = (
                f"{output_name}/data-{shard_index:05d}-of-{shard_count:05d}.arrow"
            )
            source_relative = Path(str(source_shard["path"]))
            if (
                source_relative.is_absolute()
                or ".." in source_relative.parts
                or output_shard["path"] != expected_output_relative
            ):
                raise ValueError(f"FineVision shard path differs for {name!r}")
            source_shard_rows = _count(
                source_shard["rows"],
                name=f"FineVision {name} source shard rows",
                minimum=1,
            )
            output_shard_rows = _count(
                output_shard["rows"],
                name=f"FineVision {name} output shard rows",
                minimum=1,
            )
            _count(
                source_shard["bytes"],
                name=f"FineVision {name} source shard bytes",
                minimum=1,
            )
            _sha256(
                source_shard["sha256"],
                name=f"FineVision {name} source shard SHA-256",
            )
            if source_shard_rows != output_shard_rows:
                raise ValueError(f"FineVision source/output shard rows differ for {name!r}")
            live_shard = output_root / Path(expected_output_relative).name
            expected_live_names.append(live_shard.name)
            expected_bytes = _count(
                output_shard["bytes"],
                name=f"FineVision {name} output shard bytes",
                minimum=1,
            )
            expected_sha = _sha256(
                output_shard["sha256"],
                name=f"FineVision {name} output shard SHA-256",
            )
            if (
                not live_shard.is_file()
                or live_shard.stat().st_size != expected_bytes
                or _sha256_file(live_shard) != expected_sha
            ):
                raise ValueError(f"FineVision live output shard differs: {live_shard}")
            receipt_path = live_shard.with_suffix(".receipt.json")
            try:
                receipt = json.loads(
                    receipt_path.read_bytes(),
                    object_pairs_hook=_strict_json_object,
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"FineVision live receipt is invalid: {receipt_path}") from error
            if receipt != {
                "source_sha256": source_shard["sha256"],
                "output_sha256": expected_sha,
                "rows": source_shard_rows,
            }:
                raise ValueError(f"FineVision live receipt differs: {receipt_path}")
            source_rows += source_shard_rows
            output_rows += output_shard_rows
        actual_live_names = sorted(path.name for path in output_root.glob("data-*.arrow"))
        if actual_live_names != expected_live_names or source_rows != rows or output_rows != rows:
            raise ValueError(f"FineVision live shard inventory differs for {name!r}")

        dataset_info_sha = _sha256(
            output["dataset_info_sha256"],
            name=f"FineVision {name} dataset_info_sha256",
        )
        dataset_info_path = output_root / "dataset_info.json"
        if not dataset_info_path.is_file() or _sha256_file(dataset_info_path) != dataset_info_sha:
            raise ValueError(f"FineVision live dataset_info.json differs for {name!r}")
        recomputed_fingerprint = _finevision_output_fingerprint(
            source_name=name,
            rows=rows,
            physical_schema_sha256=source_schema,
            shards=output_shards,
            dataset_info_sha256=dataset_info_sha,
        )
        if recomputed_fingerprint != expected_fingerprint:
            raise ValueError(f"FineVision output fingerprint arithmetic differs for {name!r}")
        from .dataset_compat import load_from_disk_compat

        live_dataset = load_from_disk_compat(output_root)
        if len(live_dataset) != rows:
            raise ValueError(f"FineVision live dataset row count differs for {name!r}")

    # Close the validation window by checking the external receipt after all output reads.
    if external_manifest.read_bytes() != copied_raw:
        raise ValueError("Live FineVision materialization receipt changed during validation")
    validated = FineVisionMaterializationReference(
        path=copied_path,
        sha256=expected_raw_sha,
        content_sha256=expected_content_sha,
        visualweb_fingerprint=visualweb_fingerprint,
        geo170k_fingerprint=geo170k_fingerprint,
    )
    final_signature = stat_signature(cache_paths)
    if final_signature != current_signature:
        raise ValueError("FineVision materialization files changed during validation")
    _FINEVISION_VALIDATION_CACHE[cache_key] = (validated, final_signature)
    return validated


def _load_finevision_materialization_reference(
    provenance_root: Path,
    value: Any,
    source_spec: VisionAlignmentPerceptionSourceSpec,
) -> FineVisionMaterializationReference:
    """Validate the copied receipt identity without rehashing the live Arrow payload."""
    reference = _exact_mapping(
        value,
        _FINEVISION_REFERENCE_FIELDS,
        name="finevision_materialization",
    )
    if reference["path"] != f"upstream/{_FINEVISION_MATERIALIZATION_MANIFEST}":
        raise ValueError("FineVision materialization receipt path is not canonical")
    copied_path = _artifact_path(
        provenance_root,
        reference["path"],
        name="finevision_materialization.path",
    )
    raw_sha = _sha256(reference["sha256"], name="finevision_materialization.sha256")
    if _sha256_file(copied_path) != raw_sha:
        raise ValueError("Copied FineVision materialization raw SHA-256 differs")
    visualweb_fingerprint = _sha256(
        reference["visualweb_fingerprint"],
        name="finevision_materialization.visualweb_fingerprint",
    )
    geo170k_fingerprint = _sha256(
        reference["geo170k_fingerprint"],
        name="finevision_materialization.geo170k_fingerprint",
    )
    if (
        visualweb_fingerprint != source_spec.finevision_visualweb_fingerprint
        or geo170k_fingerprint != source_spec.finevision_geo170k_fingerprint
    ):
        raise ValueError("FineVision materialization fingerprints differ from the source spec")
    return FineVisionMaterializationReference(
        path=copied_path,
        sha256=raw_sha,
        content_sha256=_sha256(
            reference["content_sha256"],
            name="finevision_materialization.content_sha256",
        ),
        visualweb_fingerprint=visualweb_fingerprint,
        geo170k_fingerprint=geo170k_fingerprint,
    )


@dataclass(frozen=True)
class PerceptionSplitSelection:
    """Validated selected rows and image inventories for one logical split."""

    physical_split: str
    base_annotation_sha256: str
    base_dataset_fingerprint: str
    base_examples: int
    indices: Tuple[int, ...]
    selection_indices_sha256: str
    runtime_dataset_fingerprint: str
    row_image_content_sha256: Tuple[str, ...]
    unique_image_content_sha256: Tuple[str, ...]


@dataclass(frozen=True)
class PerceptionProvenanceManifest:
    """Validated perception provenance artifact and selected source rows."""

    path: Path
    raw_sha256: str
    content_sha256: str
    source_spec: VisionAlignmentPerceptionSourceSpec
    source_spec_sha256: str
    finevision_materialization: FineVisionMaterializationReference
    image_path_signatures: Tuple[PerceptionPathSignature, ...]
    selections: Mapping[tuple[str, str], PerceptionSplitSelection]

    def selection(self, source_name: str, logical_split: str) -> PerceptionSplitSelection:
        """Return one source/split selection or raise for an unknown key."""
        try:
            return self.selections[(source_name, logical_split)]
        except KeyError as error:
            raise ValueError(
                f"Provenance has no selection for {source_name!r}/{logical_split!r}"
            ) from error

    def validate_image_path_signatures(self, *, workers: int = 64, max_pending: int = 256) -> None:
        """Restat every path-backed image and reject any post-build filesystem drift.

        Encoded bytes were fully hashed by the provenance builder. Exact size, timestamps,
        inode, and device make the production startup check metadata-only while still failing
        closed if the shared image store changes after that hash pass.

        :param workers: Maximum number of concurrent metadata reads.
        :param max_pending: Maximum metadata reads submitted at once.
        """

        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("Image-signature workers must be an integer in [1, 64]")
        if type(max_pending) is not int or not workers <= max_pending <= 256:
            raise ValueError("Image-signature pending reads must be between workers and 256")

        def restat(record: PerceptionPathSignature) -> tuple[Path, tuple[int, ...]]:
            """Return the current filesystem signature for one pinned image."""
            path = Path(record.path)
            try:
                info = path.stat()
            except OSError as error:
                raise ValueError(f"Pinned perception image is unavailable: {path}") from error
            return path, (
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                info.st_ino,
                info.st_dev,
            )

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="vision-image-signature"
        ) as executor:
            for offset in range(0, len(self.image_path_signatures), max_pending):
                records = self.image_path_signatures[offset : offset + max_pending]
                futures = [executor.submit(restat, record) for record in records]
                for record, future in zip(records, futures):
                    path, actual = future.result()
                    expected = (
                        record.size_bytes,
                        record.mtime_ns,
                        record.ctime_ns,
                        record.inode,
                        record.device,
                    )
                    if not stat.S_ISREG(actual[0]) or actual[1:] != expected:
                        raise ValueError(f"Pinned perception image signature changed: {path}")


def load_perception_provenance_manifest(
    path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
    verify_finevision_materialization: bool = True,
    load_image_path_signatures: bool = True,
) -> PerceptionProvenanceManifest:
    """Load and validate a perception image-provenance artifact.

    ``load_image_path_signatures=False`` is reserved for nonzero training ranks after rank 0
    has loaded and restatted the exact inventory and broadcast success. Offline callers use
    the exhaustive default.
    """
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.name != PERCEPTION_PROVENANCE_MANIFEST:
        raise ValueError(
            f"Perception provenance must use canonical name {PERCEPTION_PROVENANCE_MANIFEST!r}"
        )
    root_dir = manifest_path.parent
    try:
        raw = manifest_path.read_bytes()
        root = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid perception provenance {manifest_path}: {error}") from error
    root = _exact_mapping(root, _ROOT_FIELDS, name="perception provenance")
    raw_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and raw_sha != _sha256(
        expected_sha256, name="expected provenance SHA-256"
    ):
        raise ValueError(
            f"Perception provenance raw SHA mismatch: expected {expected_sha256}, got {raw_sha}"
        )
    complete_path = root_dir / "COMPLETE"
    try:
        complete_stat = complete_path.lstat()
        complete_raw = complete_path.read_bytes()
    except OSError as error:
        raise ValueError("Perception provenance lacks its COMPLETE marker") from error
    if (
        not stat.S_ISREG(complete_stat.st_mode)
        or complete_path.is_symlink()
        or complete_raw != f"{raw_sha}\n".encode("ascii")
    ):
        raise ValueError("Perception provenance COMPLETE marker differs from the manifest")
    if (
        root["format"] != PERCEPTION_PROVENANCE_FORMAT
        or _count(root["version"], name="provenance version", minimum=1)
        != PERCEPTION_PROVENANCE_VERSION
        or root["status"] != "verified"
        or root["phase"] != "perception"
    ):
        raise ValueError("Perception provenance identity or status is incompatible")
    if not isinstance(root["created_at"], str):
        raise ValueError("Perception provenance created_at must be an ISO-8601 string")
    try:
        created_at = datetime.fromisoformat(root["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Perception provenance created_at is not ISO-8601") from error
    if created_at.tzinfo is None:
        raise ValueError("Perception provenance created_at must include a timezone")
    builder = _exact_mapping(root["builder"], _BUILDER_FIELDS, name="provenance builder")
    if not isinstance(builder["name"], str) or not builder["name"]:
        raise ValueError("Perception provenance builder.name must be a non-empty string")
    _count(builder["version"], name="provenance builder.version", minimum=1)
    _sha256(builder["script_sha256"], name="provenance builder.script_sha256")
    content_sha = _sha256(root["content_sha256"], name="provenance content SHA-256")
    unsigned = dict(root)
    unsigned.pop("content_sha256")
    if _canonical_sha256(unsigned) != content_sha:
        raise ValueError("Perception provenance content SHA-256 differs")

    source_spec_value = root["source_spec"]
    if not isinstance(source_spec_value, Mapping):
        raise ValueError("Perception provenance source_spec must be an object")
    source_spec_mapping = dict(source_spec_value)
    source_spec_registry_version = source_spec_mapping.pop("source_registry_version", None)
    expected_spec_fields = {field.name for field in fields(VisionAlignmentPerceptionSourceSpec)}
    if (
        _count(
            source_spec_registry_version,
            name="source_spec.source_registry_version",
            minimum=1,
        )
        != VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
        or set(source_spec_mapping) != expected_spec_fields
    ):
        raise ValueError("Perception provenance source_spec fields or registry version differ")
    ocr_source_names = source_spec_mapping["ocr_source_names"]
    if not isinstance(ocr_source_names, list) or any(
        not isinstance(value, str) for value in ocr_source_names
    ):
        raise ValueError("Perception source_spec.ocr_source_names must be a string list")
    source_spec_mapping["ocr_source_names"] = tuple(ocr_source_names)
    try:
        source_spec = VisionAlignmentPerceptionSourceSpec(**source_spec_mapping)
    except TypeError as error:
        raise ValueError(f"Invalid perception source_spec: {error}") from error
    source_spec_sha = _sha256(root["source_spec_sha256"], name="source_spec_sha256")
    if (
        source_spec.as_canonical_dict() != source_spec_value
        or source_spec.preprocessing_sha256 != source_spec_sha
    ):
        raise ValueError("Perception source_spec SHA-256 differs")
    if (
        _count(root["source_registry_version"], name="source_registry_version", minimum=1)
        != VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
    ):
        raise ValueError("Perception source registry version differs")
    source_spec.validate_production_contract()
    finevision_materialization = (
        validate_finevision_materialization(
            root_dir,
            root["finevision_materialization"],
            source_spec,
        )
        if verify_finevision_materialization
        else _load_finevision_materialization_reference(
            root_dir,
            root["finevision_materialization"],
            source_spec,
        )
    )
    if load_image_path_signatures:
        image_path_signatures = _read_path_signatures(root_dir, root["image_path_signatures"])
    else:
        _validate_path_signature_reference(root_dir, root["image_path_signatures"])
        image_path_signatures = ()
    validation_selection = _exact_mapping(
        root["validation_selection"],
        _VALIDATION_SELECTION_FIELDS,
        name="validation_selection",
    )
    if (
        validation_selection["algorithm"] != _VALIDATION_SELECTION_ALGORITHM
        or _count(
            validation_selection["image_contents_per_source"],
            name="validation_selection.image_contents_per_source",
            minimum=1,
        )
        != VALIDATION_IMAGE_CONTENTS_PER_SOURCE
    ):
        raise ValueError("Perception validation selection policy differs")

    raw_sources = root["sources"]
    if (
        not isinstance(raw_sources, Mapping)
        or tuple(sorted(raw_sources)) != PERCEPTION_SOURCE_NAMES
    ):
        raise ValueError("Perception provenance must contain the exact eight-source set")
    selections: Dict[tuple[str, str], PerceptionSplitSelection] = {}
    for source_name in PERCEPTION_SOURCE_NAMES:
        source = _exact_mapping(
            raw_sources[source_name], _SOURCE_FIELDS, name=f"sources.{source_name}"
        )
        components = source["components"]
        if (
            not isinstance(components, list)
            or not components
            or any(not isinstance(component, str) or not component for component in components)
            or len(set(components)) != len(components)
        ):
            raise ValueError(f"sources.{source_name}.components must be unique non-empty names")
        expected_components = {
            "audited_alignment": ["visualwebinstruct(filtered)", "geo170k(align)"],
            "ocr_document": list(source_spec.ocr_source_names),
        }.get(source_name, [source_name])
        if components != expected_components:
            raise ValueError(f"sources.{source_name}.components differ from the source spec")
        for logical_split in ("train", "validation"):
            split = _exact_mapping(
                source[logical_split],
                _SPLIT_FIELDS,
                name=f"sources.{source_name}.{logical_split}",
            )
            physical_split = split["physical_split"]
            if not isinstance(physical_split, str) or not physical_split:
                raise ValueError("physical_split must be a non-empty string")
            expected_physical_split = (
                "train"
                if logical_split == "train" or source_name == "audited_alignment"
                else "validation"
            )
            if physical_split != expected_physical_split:
                raise ValueError(
                    f"{source_name}.{logical_split} must use physical split "
                    f"{expected_physical_split!r}"
                )
            base_fp = _dataset_fingerprint(
                split["base_dataset_fingerprint"],
                name=f"{source_name}.{logical_split}.base_dataset_fingerprint",
            )
            base_annotation_sha256 = _sha256(
                split["base_annotation_sha256"],
                name=f"{source_name}.{logical_split}.base_annotation_sha256",
            )
            base_examples = _count(
                split["base_examples"],
                name=f"{source_name}.{logical_split}.base_examples",
                minimum=1,
            )
            indices = _read_indices(
                root_dir,
                split["selection"],
                name=f"{source_name}.{logical_split}.selection",
                base_size=base_examples,
            )
            selection_ref = split["selection"]
            expected_runtime_fp = selected_dataset_fingerprint(
                source_name=source_name,
                logical_split=logical_split,
                physical_split=physical_split,
                base_fingerprint=base_fp,
                selection_indices_sha256=selection_ref["indices_sha256"],
                source_spec_sha256=source_spec_sha,
            )
            if split["runtime_dataset_fingerprint"] != expected_runtime_fp or split[
                "runtime_examples"
            ] != len(indices):
                raise ValueError(f"{source_name}.{logical_split} runtime identity differs")
            row_hashes = _read_hash_inventory(
                root_dir,
                split["row_image_content"],
                name=f"{source_name}.{logical_split}.row_image_content",
                unique=False,
            )
            unique_hashes = _read_hash_inventory(
                root_dir,
                split["unique_image_content"],
                name=f"{source_name}.{logical_split}.unique_image_content",
                unique=True,
            )
            if len(row_hashes) != len(indices) or tuple(sorted(set(row_hashes))) != unique_hashes:
                raise ValueError(f"{source_name}.{logical_split} image inventories disagree")
            if logical_split == "validation" and (
                len(indices) != VALIDATION_IMAGE_CONTENTS_PER_SOURCE
                or len(unique_hashes) != VALIDATION_IMAGE_CONTENTS_PER_SOURCE
            ):
                raise ValueError(
                    f"{source_name}.validation must contain exactly "
                    f"{VALIDATION_IMAGE_CONTENTS_PER_SOURCE} distinct image contents"
                )
            selections[(source_name, logical_split)] = PerceptionSplitSelection(
                physical_split=physical_split,
                base_annotation_sha256=base_annotation_sha256,
                base_dataset_fingerprint=base_fp,
                base_examples=base_examples,
                indices=indices,
                selection_indices_sha256=selection_ref["indices_sha256"],
                runtime_dataset_fingerprint=expected_runtime_fp,
                row_image_content_sha256=row_hashes,
                unique_image_content_sha256=unique_hashes,
            )

    audited_train = selections[("audited_alignment", "train")]
    audited_validation = selections[("audited_alignment", "validation")]
    if (
        audited_train.base_dataset_fingerprint != audited_validation.base_dataset_fingerprint
        or audited_train.base_annotation_sha256 != audited_validation.base_annotation_sha256
        or audited_train.base_examples != audited_validation.base_examples
    ):
        raise ValueError(
            "audited_alignment train/validation must share one physical base dataset identity"
        )

    unions = _exact_mapping(root["unions"], _UNION_FIELDS, name="provenance unions")
    train_union = _read_hash_inventory(
        root_dir,
        unions["train_unique_image_content"],
        name="unions.train_unique_image_content",
        unique=True,
    )
    validation_union = _read_hash_inventory(
        root_dir,
        unions["validation_unique_image_content"],
        name="unions.validation_unique_image_content",
        unique=True,
    )
    overlap = set(train_union).intersection(validation_union)
    if _count(unions["overlap_count"], name="unions.overlap_count") != 0 or overlap:
        raise ValueError("Perception train and validation image-content unions overlap")
    expected_train_union = tuple(
        sorted(
            {
                value
                for source_name in PERCEPTION_SOURCE_NAMES
                for value in selections[(source_name, "train")].unique_image_content_sha256
            }
        )
    )
    expected_validation_union = tuple(
        sorted(
            {
                value
                for source_name in PERCEPTION_SOURCE_NAMES
                for value in selections[(source_name, "validation")].unique_image_content_sha256
            }
        )
    )
    if train_union != expected_train_union or validation_union != expected_validation_union:
        raise ValueError("Perception union inventories differ from their source inventories")

    filtering = root["filtering"]
    if not isinstance(filtering, Mapping) or tuple(sorted(filtering)) != PERCEPTION_SOURCE_NAMES:
        raise ValueError("Perception filtering must cover the exact eight-source set")
    for source_name in PERCEPTION_SOURCE_NAMES:
        values = _exact_mapping(
            filtering[source_name], _FILTER_FIELDS, name=f"filtering.{source_name}"
        )
        candidate = _count(
            values["candidate_train_examples"], name=f"filtering.{source_name}.candidate"
        )
        removed = _count(values["removed_train_examples"], name=f"filtering.{source_name}.removed")
        output = _count(
            values["output_train_examples"], name=f"filtering.{source_name}.output", minimum=1
        )
        if candidate - removed != output or output != len(
            selections[(source_name, "train")].indices
        ):
            raise ValueError(f"Filtering arithmetic differs for {source_name}")
        if candidate != selections[(source_name, "train")].base_examples:
            raise ValueError(f"Filtering candidate size differs for {source_name}")

    return PerceptionProvenanceManifest(
        path=manifest_path,
        raw_sha256=raw_sha,
        content_sha256=content_sha,
        source_spec=source_spec,
        source_spec_sha256=source_spec_sha,
        finevision_materialization=finevision_materialization,
        image_path_signatures=image_path_signatures,
        selections=selections,
    )


class SelectedVisionAlignmentDataset:
    """Map a provenance-selected logical split onto an exact raw perception dataset."""

    content_fingerprint_version = "vision-alignment-perception-selected-v1"

    def __init__(
        self,
        dataset: Any,
        *,
        source_name: str,
        logical_split: str,
        selection: PerceptionSplitSelection,
    ):
        base_fingerprint = runtime_dataset_fingerprint(dataset)
        if (
            base_fingerprint != selection.base_dataset_fingerprint
            or len(dataset) != selection.base_examples
            or perception_annotation_content_sha256(dataset) != selection.base_annotation_sha256
        ):
            raise ValueError(
                f"Raw {source_name}/{logical_split} dataset differs from perception provenance"
            )
        self._dataset = dataset
        self._selection = selection
        self.source_name = source_name
        self.logical_split = logical_split
        self.indices = selection.indices
        self.content_fingerprint = selection.runtime_dataset_fingerprint

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> Any:
        """Build one selected row without substituting another logical index."""
        raw_index = self.indices[index]
        get = getattr(self._dataset, "get", None)
        return get(raw_index, epoch) if callable(get) else self._dataset[raw_index]

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return raw image references for one selected logical row."""
        raw_images = getattr(self._dataset, "raw_image_references", None)
        if not callable(raw_images):
            raise ValueError(f"Selected source {self.source_name!r} lacks raw image access")
        return tuple(raw_images(self.indices[index]))

    def validate_image_content(self, indices: Optional[Sequence[int]] = None) -> str:
        """Rehash selected image bytes and compare them with the provenance inventory.

        :param indices: Logical selected-row indices. ``None`` validates the full selection.
        :returns: A canonical SHA-256 over the validated ordered ``(index, image_sha)`` rows.
        :raises ValueError: If any row has multiple images or its current bytes drifted.
        """
        selected_indices = tuple(range(len(self))) if indices is None else tuple(indices)
        rows = []
        for index in selected_indices:
            if index < 0 or index >= len(self):
                raise ValueError(f"Selected image-validation index is out of bounds: {index}")
            references = self.raw_image_references(index)
            if len(references) != 1:
                raise ValueError(
                    f"Selected source {self.source_name!r} row {index} has "
                    f"{len(references)} raw images; expected exactly one"
                )
            actual = image_reference_sha256(references[0])
            expected = self._selection.row_image_content_sha256[index]
            if actual != expected:
                raise ValueError(
                    f"Selected source {self.source_name!r} row {index} image bytes differ"
                )
            rows.append({"index": index, "image_sha256": actual})
        return _canonical_sha256(rows)

    def validate_required_annotations(self) -> None:
        """Run the raw adapter's fail-closed annotation validator."""
        validate = getattr(self._dataset, "validate_required_annotations", None)
        if not callable(validate):
            raise ValueError(f"Selected source {self.source_name!r} lacks annotation validation")
        validate()


def build_selected_perception_dataset(
    manifest: PerceptionProvenanceManifest,
    tokenizer: Any,
    token_ids: Any,
    source_name: str,
    *,
    logical_split: str,
    validate_required_annotations: bool = True,
    verify_finevision_materialization: bool = True,
) -> SelectedVisionAlignmentDataset:
    """Build one provenance-selected perception dataset from its physical raw split."""
    selection = manifest.selection(source_name, logical_split)
    if source_name == "audited_alignment" and verify_finevision_materialization:
        # Revalidate the copied upstream receipt and every live Arrow byte immediately
        # before the adapter trusts its externally supplied content fingerprints.
        validate_finevision_materialization(
            manifest.path.parent,
            {
                "path": str(
                    manifest.finevision_materialization.path.relative_to(manifest.path.parent)
                ),
                "sha256": manifest.finevision_materialization.sha256,
                "content_sha256": manifest.finevision_materialization.content_sha256,
                "visualweb_fingerprint": manifest.finevision_materialization.visualweb_fingerprint,
                "geo170k_fingerprint": manifest.finevision_materialization.geo170k_fingerprint,
            },
            manifest.source_spec,
        )
    raw_dataset = build_vision_alignment_perception_dataset(
        manifest.source_spec,
        tokenizer,
        token_ids,
        source_name,
        split=selection.physical_split,
        validate_required_annotations=validate_required_annotations,
    )
    return SelectedVisionAlignmentDataset(
        raw_dataset,
        source_name=source_name,
        logical_split=logical_split,
        selection=selection,
    )
