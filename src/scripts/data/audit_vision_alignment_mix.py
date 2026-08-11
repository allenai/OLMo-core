#!/usr/bin/env python
"""Audit serialized vision-alignment examples and calibrate source sampling.

This command is deliberately offline: it reads exact runtime probes from
``export_vision_alignment_probe.py`` and never constructs a model, downloads a dataset, or
launches a job. Production input is a strict version-2 JSON catalog::

    {
      "format": "vision_alignment_preprocessed_source_catalog",
      "version": 2,
      "recipe_version": 1,
      "formatter_version": "vision-alignment-document-v1",
      "source_registry_version": 1,
      "source_registry_sha256": "<exact registry SHA-256>",
      "exporter_sha256": "<exact exporter SHA-256>",
      "image_manifest_sha256": "<64 lowercase hex characters>",
      "preprocessing_config": {"...": "..."},
      "preprocessing_config_sha256": "<64 lowercase hex characters>",
      "probe": {
        "format": "vision_alignment_serialized_probe",
        "version": 1,
        "selection_algorithm": "sha256-affine-permutation-v1",
        "seed": 6198,
        "epoch": 0,
        "examples_per_source": 1024
      },
      "sources": [
        {
          "name": "pixmo_caption",
          "format": "jsonl",
          "path": "samples/pixmo_caption.jsonl",
          "dataset_fingerprint": "<runtime source fingerprint>",
          "dataset_size": 714985,
          "sha256": "<expected SHA-256>",
          "probe_indices": [12, 34],
          "probe_indices_sha256": "<indices digest>",
          "serialized_row_hashes_sha256": "<ordered row-hash digest>"
        }
      ]
    }

Each canonical JSONL record pins its live source/index/epoch and descriptors for every
model-consumed array. Token, label, position, type, and loss arrays are also embedded so this
command can validate structure and measure loss mass. Version-1 JSONL/NPZ catalogs remain
readable for offline diagnostics, but the training launcher rejects their unbound reports.

The output is canonical JSON. ``input_content_sha256`` pins source names, formats, and exact
file bytes; ``fingerprint`` pins the complete report excluding the fingerprint field itself.
An artifact with malformed records, zero-loss sources, or target/catalog mismatches is emitted
with ``status: failed`` and the command exits non-zero, so it cannot silently become a training
calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VISION_ALIGNMENT_PHASES,
    VisionAlignmentMixtureConfig,
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_PROBE_FORMAT,
    VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_PROBE_VERSION,
    VISION_ALIGNMENT_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION,
    array_content_descriptor,
    select_deterministic_probe_indices,
    serialized_descriptor_sha256,
    vision_alignment_source_registry_sha256,
)

SOURCE_CATALOG_FORMAT = "vision_alignment_preprocessed_source_catalog"
SOURCE_CATALOG_VERSION = VISION_ALIGNMENT_SOURCE_CATALOG_VERSION
LEGACY_SOURCE_CATALOG_VERSION = 1
AUDIT_FORMAT = "vision_alignment_source_audit"
AUDIT_VERSION = 2

_LEGACY_ROOT_FIELDS = frozenset(
    {
        "format",
        "version",
        "recipe_version",
        "formatter_version",
        "image_manifest_sha256",
        "preprocessing_config_sha256",
        "sources",
    }
)
_ROOT_FIELDS = _LEGACY_ROOT_FIELDS | {
    "source_registry_version",
    "source_registry_sha256",
    "exporter_sha256",
    "preprocessing_config",
    "probe",
}
_SOURCE_REQUIRED_FIELDS = frozenset(
    {"name", "format", "path", "dataset_fingerprint", "dataset_size"}
)
_SOURCE_FIELDS = _SOURCE_REQUIRED_FIELDS | {
    "sha256",
    "probe_indices",
    "probe_indices_sha256",
    "serialized_row_hashes_sha256",
}
_PROBE_FIELDS = frozenset(
    {
        "format",
        "version",
        "selection_algorithm",
        "seed",
        "epoch",
        "examples_per_source",
    }
)
_SOURCE_FORMATS = frozenset({"jsonl", "npz"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_REPORTED_ERRORS = 20


@dataclass(frozen=True)
class CatalogSource:
    """One serialized preprocessed-example source.

    :param name: Stable source name matching the loss-mass target key.
    :param serialization: Either ``jsonl`` or fixed-shape ``npz``.
    :param catalog_path: Path exactly as represented in the catalog.
    :param resolved_path: Absolute path used for reading.
    :param expected_sha256: Optional catalog-pinned digest.
    :param sha256: Digest of the exact bytes read by this audit.
    """

    name: str
    serialization: str
    catalog_path: str
    resolved_path: Path
    expected_sha256: Optional[str]
    sha256: str
    dataset_fingerprint: str
    dataset_size: int
    probe_indices: Optional[Tuple[int, ...]] = None
    probe_indices_sha256: Optional[str] = None
    serialized_row_hashes_sha256: Optional[str] = None


@dataclass(frozen=True)
class SourceCatalog:
    """Validated source catalog and exact input fingerprints.

    :param sources: Sources in deterministic name order.
    :param catalog_sha256: Digest of the catalog's exact bytes.
    :param input_content_sha256: Canonical digest of source names, formats, and byte digests.
    """

    sources: Tuple[CatalogSource, ...]
    catalog_sha256: str
    input_content_sha256: str
    recipe_version: int
    formatter_version: str
    image_manifest_sha256: str
    preprocessing_config_sha256: str
    version: int
    source_registry_version: Optional[int] = None
    source_registry_sha256: Optional[str] = None
    exporter_sha256: Optional[str] = None
    preprocessing_config: Optional[Mapping[str, Any]] = None
    probe: Optional[Mapping[str, Any]] = None


@dataclass
class MetricAccumulator:
    """Accumulate a deterministic scalar count distribution."""

    count: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def add(self, value: float) -> None:
        """Add one finite scalar value."""
        if not math.isfinite(value):
            raise ValueError(f"Metric value must be finite, got {value}")
        self.count += 1
        self.total = math.fsum((self.total, value))
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self, *, integral: bool) -> Dict[str, Any]:
        """Return the canonical summary representation."""
        if self.count == 0:
            return {"total": 0, "mean": None, "min": None, "max": None}
        total: int | float = int(self.total) if integral else float(self.total)
        minimum: int | float = int(self.minimum or 0) if integral else float(self.minimum or 0)
        maximum: int | float = int(self.maximum or 0) if integral else float(self.maximum or 0)
        return {
            "total": total,
            "mean": float(self.total / self.count),
            "min": minimum,
            "max": maximum,
        }


@dataclass
class SourceAccumulator:
    """Accumulate example-level audit diagnostics for one source."""

    seen: int = 0
    valid: int = 0
    truncated: int = 0
    zero_loss: int = 0
    input_tokens: MetricAccumulator = field(default_factory=MetricAccumulator)
    supervised_tokens: MetricAccumulator = field(default_factory=MetricAccumulator)
    loss_weight: MetricAccumulator = field(default_factory=MetricAccumulator)
    image_crops: MetricAccumulator = field(default_factory=MetricAccumulator)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0

    def add_error(self, index: int, message: str) -> None:
        """Record one malformed example while bounding report size."""
        self.seen += 1
        self.error_count += 1
        if len(self.errors) < _MAX_REPORTED_ERRORS:
            self.errors.append({"example_index": index, "message": message})

    def add_example(self, record: Mapping[str, Any]) -> None:
        """Validate and accumulate one preprocessed example."""
        input_ids = _validate_input_ids(record.get("input_ids"))
        loss_masks = _validate_loss_masks(record.get("loss_masks"), len(input_ids))
        image_crops = _extract_image_crops(record)
        truncated = _extract_truncated(record)

        positive_tokens = sum(weight > 0.0 for weight in loss_masks)
        summed_loss_weight = math.fsum(loss_masks)
        self.seen += 1
        self.valid += 1
        self.truncated += int(truncated)
        self.zero_loss += int(summed_loss_weight == 0.0)
        self.input_tokens.add(float(len(input_ids)))
        self.supervised_tokens.add(float(positive_tokens))
        self.loss_weight.add(summed_loss_weight)
        self.image_crops.add(float(image_crops))

    def as_dict(self) -> Dict[str, Any]:
        """Return canonical per-source diagnostics."""
        return {
            "examples": {
                "seen": self.seen,
                "valid": self.valid,
                "errors": self.error_count,
            },
            "raw_input_tokens": self.input_tokens.as_dict(integral=True),
            "positive_supervised_tokens": self.supervised_tokens.as_dict(integral=True),
            "summed_loss_weight": self.loss_weight.as_dict(integral=False),
            "mean_sum_loss_masks": (
                None if self.valid == 0 else float(self.loss_weight.total / self.valid)
            ),
            "image_crops": self.image_crops.as_dict(integral=True),
            "truncated_examples": self.truncated,
            "zero_loss_examples": self.zero_loss,
            "error_samples": self.errors,
        }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Could not read source file {path}: {error}") from error
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def load_source_catalog(path: os.PathLike[str] | str) -> SourceCatalog:
    """Load a strict catalog and hash every serialized source.

    Paths must be local and resolve relative to the catalog. An optional declared source digest
    is always verified when present.

    :param path: Catalog JSON path.
    :returns: Validated catalog in deterministic source-name order.
    :raises ValueError: If the schema, paths, or declared hashes are invalid.
    """
    catalog_path = Path(path).expanduser().resolve()
    try:
        raw = catalog_path.read_bytes()
        root = _require_mapping(
            json.loads(raw, object_pairs_hook=_strict_json_object), "catalog root"
        )
    except OSError as error:
        raise ValueError(f"Could not read source catalog {catalog_path}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid source catalog JSON in {catalog_path}: {error}") from error

    if root.get("format") != SOURCE_CATALOG_FORMAT:
        raise ValueError(f"Catalog format must be {SOURCE_CATALOG_FORMAT!r}")
    version = _require_int(root.get("version"), "catalog version", minimum=1)
    if version not in (LEGACY_SOURCE_CATALOG_VERSION, SOURCE_CATALOG_VERSION):
        raise ValueError(
            f"Unsupported source catalog version {version}; expected "
            f"{LEGACY_SOURCE_CATALOG_VERSION} or {SOURCE_CATALOG_VERSION}"
        )
    expected_root_fields = (
        _ROOT_FIELDS if version == SOURCE_CATALOG_VERSION else _LEGACY_ROOT_FIELDS
    )
    _reject_unknown_fields(
        root,
        allowed=expected_root_fields,
        required=expected_root_fields,
        name="catalog root",
    )
    recipe_version = _require_int(root["recipe_version"], "recipe_version", minimum=1)
    formatter_version = _require_nonempty_string(root["formatter_version"], "formatter_version")
    image_manifest_sha256 = _require_nonempty_string(
        root["image_manifest_sha256"], "image_manifest_sha256"
    )
    preprocessing_config_sha256 = _require_nonempty_string(
        root["preprocessing_config_sha256"], "preprocessing_config_sha256"
    )
    for name, value in (
        ("image_manifest_sha256", image_manifest_sha256),
        ("preprocessing_config_sha256", preprocessing_config_sha256),
    ):
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")

    source_registry_version: Optional[int] = None
    source_registry_sha256: Optional[str] = None
    exporter_sha256: Optional[str] = None
    preprocessing_config: Optional[Mapping[str, Any]] = None
    probe: Optional[Mapping[str, Any]] = None
    if version == SOURCE_CATALOG_VERSION:
        source_registry_version = _require_int(
            root["source_registry_version"], "source_registry_version", minimum=1
        )
        if source_registry_version != VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION:
            raise ValueError(
                "Source catalog registry version differs from the runtime registry: "
                f"{source_registry_version} != {VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION}"
            )
        source_registry_sha256 = _require_nonempty_string(
            root["source_registry_sha256"], "source_registry_sha256"
        )
        if source_registry_sha256 != vision_alignment_source_registry_sha256():
            raise ValueError("Source catalog was not produced from this exact source registry")
        exporter_sha256 = _require_nonempty_string(root["exporter_sha256"], "exporter_sha256")
        exporter_path = Path(__file__).resolve().parent / "export_vision_alignment_probe.py"
        if exporter_sha256 != _sha256_file(exporter_path):
            raise ValueError("Source catalog was not produced by this exact canonical exporter")
        preprocessing_config = dict(
            _require_mapping(root["preprocessing_config"], "preprocessing_config")
        )
        if _canonical_sha256(preprocessing_config) != preprocessing_config_sha256:
            raise ValueError("preprocessing_config does not match preprocessing_config_sha256")
        raw_probe = _require_mapping(root["probe"], "probe")
        _reject_unknown_fields(
            raw_probe,
            allowed=_PROBE_FIELDS,
            required=_PROBE_FIELDS,
            name="probe",
        )
        probe_format = _require_nonempty_string(raw_probe["format"], "probe.format")
        probe_version = _require_int(raw_probe["version"], "probe.version", minimum=1)
        selection_algorithm = _require_nonempty_string(
            raw_probe["selection_algorithm"], "probe.selection_algorithm"
        )
        probe_epoch = _require_int(raw_probe["epoch"], "probe.epoch", minimum=0)
        if (
            probe_format != VISION_ALIGNMENT_PROBE_FORMAT
            or probe_version != VISION_ALIGNMENT_PROBE_VERSION
            or selection_algorithm != VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM
            or probe_epoch != 0
        ):
            raise ValueError("Canonical probe identity is incompatible")
        _require_int(raw_probe["seed"], "probe.seed", minimum=0)
        _require_int(
            raw_probe["examples_per_source"],
            "probe.examples_per_source",
            minimum=1,
        )
        probe = dict(raw_probe)
    raw_sources = root["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("catalog sources must be a non-empty list")

    names = set()
    resolved_paths = set()
    sources = []
    for index, raw_source in enumerate(raw_sources):
        prefix = f"catalog sources[{index}]"
        source = _require_mapping(raw_source, prefix)
        source_fields = (
            _SOURCE_FIELDS
            if version == SOURCE_CATALOG_VERSION
            else _SOURCE_REQUIRED_FIELDS | {"sha256"}
        )
        source_required_fields = (
            _SOURCE_FIELDS if version == SOURCE_CATALOG_VERSION else _SOURCE_REQUIRED_FIELDS
        )
        _reject_unknown_fields(
            source,
            allowed=source_fields,
            required=source_required_fields,
            name=prefix,
        )
        name = _require_nonempty_string(source["name"], f"{prefix}.name")
        if name in names:
            raise ValueError(f"Duplicate source name {name!r}")
        names.add(name)
        serialization = _require_nonempty_string(source["format"], f"{prefix}.format")
        if serialization not in _SOURCE_FORMATS:
            raise ValueError(
                f"{prefix}.format must be one of {sorted(_SOURCE_FORMATS)}, got {serialization!r}"
            )
        if version == SOURCE_CATALOG_VERSION and serialization != "jsonl":
            raise ValueError("Canonical version-2 probes must use JSONL serialization")
        source_path = _require_nonempty_string(source["path"], f"{prefix}.path")
        dataset_fingerprint = _require_nonempty_string(
            source["dataset_fingerprint"], f"{prefix}.dataset_fingerprint"
        )
        dataset_size = _require_int(source["dataset_size"], f"{prefix}.dataset_size", minimum=1)
        probe_indices: Optional[Tuple[int, ...]] = None
        probe_indices_sha256: Optional[str] = None
        serialized_row_hashes_sha256: Optional[str] = None
        if version == SOURCE_CATALOG_VERSION:
            assert probe is not None
            raw_indices = source["probe_indices"]
            if not isinstance(raw_indices, list):
                raise ValueError(f"{prefix}.probe_indices must be a list")
            probe_indices = tuple(
                _require_int(value, f"{prefix}.probe_indices[{ordinal}]", minimum=0)
                for ordinal, value in enumerate(raw_indices)
            )
            expected_examples = int(probe["examples_per_source"])
            if (
                len(probe_indices) != expected_examples
                or len(set(probe_indices)) != len(probe_indices)
                or any(value >= dataset_size for value in probe_indices)
            ):
                raise ValueError(
                    f"{prefix}.probe_indices must contain exactly {expected_examples} unique "
                    "in-bounds rows"
                )
            expected_indices = select_deterministic_probe_indices(
                dataset_size,
                expected_examples,
                seed=int(probe["seed"]),
                dataset_fingerprint=dataset_fingerprint,
            )
            if probe_indices != expected_indices:
                raise ValueError(
                    f"{prefix}.probe_indices differ from the canonical deterministic selection"
                )
            probe_indices_sha256 = _require_nonempty_string(
                source["probe_indices_sha256"], f"{prefix}.probe_indices_sha256"
            )
            serialized_row_hashes_sha256 = _require_nonempty_string(
                source["serialized_row_hashes_sha256"],
                f"{prefix}.serialized_row_hashes_sha256",
            )
            for digest_name, digest_value in (
                ("probe_indices_sha256", probe_indices_sha256),
                ("serialized_row_hashes_sha256", serialized_row_hashes_sha256),
            ):
                if _SHA256_RE.fullmatch(digest_value) is None:
                    raise ValueError(f"{prefix}.{digest_name} must be a lowercase SHA-256")
            if _canonical_sha256(list(probe_indices)) != probe_indices_sha256:
                raise ValueError(f"{prefix}.probe_indices_sha256 does not match its rows")
        if "://" in source_path:
            raise ValueError(f"{prefix}.path must point to a materialized local file")
        unresolved = Path(source_path).expanduser()
        resolved = (
            unresolved.resolve()
            if unresolved.is_absolute()
            else (catalog_path.parent / unresolved).resolve()
        )
        if resolved in resolved_paths:
            raise ValueError(f"Multiple source names resolve to the same file {resolved}")
        resolved_paths.add(resolved)
        expected_sha256 = source.get("sha256")
        if expected_sha256 is not None:
            expected_sha256 = _require_nonempty_string(expected_sha256, f"{prefix}.sha256")
            if _SHA256_RE.fullmatch(expected_sha256) is None:
                raise ValueError(f"{prefix}.sha256 must be 64 lowercase hexadecimal characters")
        actual_sha256 = _sha256_file(resolved)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                f"Source {name!r} SHA-256 mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        sources.append(
            CatalogSource(
                name=name,
                serialization=serialization,
                catalog_path=source_path,
                resolved_path=resolved,
                expected_sha256=expected_sha256,
                sha256=actual_sha256,
                dataset_fingerprint=dataset_fingerprint,
                dataset_size=dataset_size,
                probe_indices=probe_indices,
                probe_indices_sha256=probe_indices_sha256,
                serialized_row_hashes_sha256=serialized_row_hashes_sha256,
            )
        )

    sources.sort(key=lambda item: item.name)
    content_descriptor = [
        {
            "dataset_fingerprint": source.dataset_fingerprint,
            "dataset_size": source.dataset_size,
            "format": source.serialization,
            "name": source.name,
            "probe_indices_sha256": source.probe_indices_sha256,
            "serialized_row_hashes_sha256": source.serialized_row_hashes_sha256,
            "sha256": source.sha256,
        }
        for source in sources
    ]
    return SourceCatalog(
        sources=tuple(sources),
        catalog_sha256=hashlib.sha256(raw).hexdigest(),
        input_content_sha256=hashlib.sha256(_canonical_bytes(content_descriptor)).hexdigest(),
        recipe_version=recipe_version,
        formatter_version=formatter_version,
        image_manifest_sha256=image_manifest_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
        version=version,
        source_registry_version=source_registry_version,
        source_registry_sha256=source_registry_sha256,
        exporter_sha256=exporter_sha256,
        preprocessing_config=preprocessing_config,
        probe=probe,
    )


def _validate_input_ids(value: Any) -> List[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("input_ids must be a non-empty list")
    output = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"input_ids[{index}] must be a non-negative integer")
        output.append(token_id)
    return output


def _validate_loss_masks(value: Any, input_length: int) -> List[float]:
    if not isinstance(value, list):
        raise ValueError("loss_masks must be a list")
    if len(value) != input_length:
        raise ValueError(
            f"loss_masks length {len(value)} does not match input_ids length {input_length}"
        )
    output = []
    for index, weight in enumerate(value):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"loss_masks[{index}] must be numeric")
        float_weight = float(weight)
        if not math.isfinite(float_weight) or float_weight < 0:
            raise ValueError(f"loss_masks[{index}] must be finite and non-negative")
        output.append(float_weight)
    return output


_SERIALIZED_DESCRIPTOR_REQUIRED_FIELDS = frozenset(
    {
        "input_ids",
        "labels",
        "loss_masks",
        "position_ids",
        "token_type_ids",
        "images",
        "pooled_patches_idx",
    }
)
_SERIALIZED_DESCRIPTOR_OPTIONAL_FIELDS = frozenset({"subsegment_ids"})
_ARRAY_DESCRIPTOR_FIELDS = frozenset({"dtype", "shape", "sha256"})
_INLINE_SERIALIZED_FIELDS = frozenset(
    {"input_ids", "labels", "loss_masks", "position_ids", "token_type_ids"}
)
_PROBE_RECORD_BASE_FIELDS = frozenset(
    {
        "source",
        "probe_index",
        "probe_epoch",
        "serialized_fields",
        "serialized_row_sha256",
        "image_crops",
        "pooled_tokens",
    }
)


def _validate_canonical_probe_record(
    record: Mapping[str, Any],
    *,
    source_name: str,
    expected_index: int,
) -> str:
    expected_record_fields = _PROBE_RECORD_BASE_FIELDS | _INLINE_SERIALIZED_FIELDS
    has_subsegments = "subsegment_ids" in record
    if has_subsegments:
        expected_record_fields = expected_record_fields | {"subsegment_ids"}
    _reject_unknown_fields(
        record,
        allowed=expected_record_fields,
        required=expected_record_fields,
        name="canonical probe record",
    )
    if record["source"] != source_name:
        raise ValueError(
            f"probe source is {record['source']!r}, expected canonical source {source_name!r}"
        )
    probe_index = _require_int(record["probe_index"], "probe_index", minimum=0)
    probe_epoch = _require_int(record["probe_epoch"], "probe_epoch", minimum=0)
    if probe_index != expected_index or probe_epoch != 0:
        raise ValueError(
            f"probe identity differs: index={probe_index!r}, epoch={probe_epoch!r}, "
            f"expected index={expected_index}, epoch=0"
        )
    descriptor = _require_mapping(record["serialized_fields"], "serialized_fields")
    expected_descriptor_fields = set(_SERIALIZED_DESCRIPTOR_REQUIRED_FIELDS)
    if "subsegment_ids" in descriptor:
        expected_descriptor_fields.add("subsegment_ids")
    if set(descriptor) != expected_descriptor_fields:
        raise ValueError("serialized_fields differ from the versioned model-input schema")
    if has_subsegments != ("subsegment_ids" in descriptor):
        raise ValueError("subsegment_ids inline data and descriptor presence disagree")

    for field_name, raw_field_descriptor in descriptor.items():
        field_descriptor = _require_mapping(raw_field_descriptor, f"serialized_fields.{field_name}")
        _reject_unknown_fields(
            field_descriptor,
            allowed=_ARRAY_DESCRIPTOR_FIELDS,
            required=_ARRAY_DESCRIPTOR_FIELDS,
            name=f"serialized_fields.{field_name}",
        )
        dtype_name = _require_nonempty_string(
            field_descriptor["dtype"], f"serialized_fields.{field_name}.dtype"
        )
        try:
            dtype = np.dtype(dtype_name)
        except TypeError as error:
            raise ValueError(f"serialized_fields.{field_name}.dtype is invalid") from error
        if dtype.hasobject:
            raise ValueError(f"serialized_fields.{field_name}.dtype may not be object")
        shape = field_descriptor["shape"]
        if not isinstance(shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        ):
            raise ValueError(f"serialized_fields.{field_name}.shape is invalid")
        digest = _require_nonempty_string(
            field_descriptor["sha256"], f"serialized_fields.{field_name}.sha256"
        )
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"serialized_fields.{field_name}.sha256 is invalid")
        if field_name in _INLINE_SERIALIZED_FIELDS or field_name == "subsegment_ids":
            inline = record[field_name]
            actual_descriptor = array_content_descriptor(
                np.asarray(inline, dtype=dtype), field_name=field_name
            )
            if actual_descriptor != field_descriptor:
                raise ValueError(f"inline {field_name} differs from its serialized descriptor")

    sequence_length = descriptor["input_ids"]["shape"]
    if len(sequence_length) != 1 or any(
        descriptor[field]["shape"] != sequence_length
        for field in ("labels", "loss_masks", "position_ids", "token_type_ids")
    ):
        raise ValueError("serialized token, label, position, type, and loss arrays must align")
    images_shape = descriptor["images"]["shape"]
    pooled_shape = descriptor["pooled_patches_idx"]["shape"]
    if len(images_shape) != 3 or len(pooled_shape) != 2:
        raise ValueError("serialized image and pooled-patch arrays have invalid rank")
    image_crops = _require_int(record["image_crops"], "image_crops", minimum=0)
    pooled_tokens = _require_int(record["pooled_tokens"], "pooled_tokens", minimum=0)
    if image_crops != images_shape[0] or pooled_tokens != pooled_shape[0]:
        raise ValueError("probe image geometry differs from serialized field descriptors")

    row_sha256 = _require_nonempty_string(record["serialized_row_sha256"], "serialized_row_sha256")
    if _SHA256_RE.fullmatch(row_sha256) is None or row_sha256 != serialized_descriptor_sha256(
        descriptor
    ):
        raise ValueError("serialized_row_sha256 does not match serialized_fields")
    return row_sha256


def _extract_image_crops(record: Mapping[str, Any]) -> int:
    if "image_crops" in record:
        value = record["image_crops"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("image_crops must be a non-negative integer")
        return value
    images = record.get("images", [])
    if images is None:
        return 0
    if not isinstance(images, list):
        raise ValueError("images must be a list when image_crops is absent")
    return len(images)


def _extract_truncated(record: Mapping[str, Any]) -> bool:
    top_level = record.get("truncated")
    metadata = record.get("metadata")
    nested: Any = None
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        nested = metadata.get("truncated")
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError("top-level and metadata truncation flags disagree")
    value = top_level if top_level is not None else nested
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("truncated must be boolean")
    return value


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Mapping[str, Any] | str]]:
    try:
        with path.open("rb") as file_handle:
            for index, raw_line in enumerate(file_handle):
                try:
                    value = json.loads(raw_line, object_pairs_hook=_strict_json_object)
                    yield index, _require_mapping(value, "example")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    yield index, f"invalid JSONL record: {error}"
    except OSError as error:
        raise ValueError(f"Could not read JSONL source {path}: {error}") from error


def _iter_npz(path: Path) -> Iterator[Tuple[int, Mapping[str, Any] | str]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "input_ids" not in archive or "loss_masks" not in archive:
                raise ValueError("NPZ source requires input_ids and loss_masks arrays")
            input_ids = archive["input_ids"]
            loss_masks = archive["loss_masks"]
            if input_ids.ndim != 2 or loss_masks.ndim != 2:
                raise ValueError("NPZ input_ids and loss_masks must both be rank two")
            if input_ids.shape != loss_masks.shape:
                raise ValueError("NPZ input_ids and loss_masks shapes must match")
            example_count = input_ids.shape[0]
            image_crops = archive["image_crops"] if "image_crops" in archive else None
            truncated = archive["truncated"] if "truncated" in archive else None
            for name, value in (("image_crops", image_crops), ("truncated", truncated)):
                if value is not None and (value.ndim != 1 or value.shape[0] != example_count):
                    raise ValueError(f"NPZ {name} must be rank one with one value per example")
            for index in range(example_count):
                record: Dict[str, Any] = {
                    "input_ids": input_ids[index].tolist(),
                    "loss_masks": loss_masks[index].tolist(),
                }
                if image_crops is not None:
                    record["image_crops"] = image_crops[index].item()
                if truncated is not None:
                    record["truncated"] = truncated[index].item()
                yield index, record
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid NPZ source {path}: {error}") from error


def _iter_source(source: CatalogSource) -> Iterator[Tuple[int, Mapping[str, Any] | str]]:
    if source.serialization == "jsonl":
        yield from _iter_jsonl(source.resolved_path)
    elif source.serialization == "npz":
        yield from _iter_npz(source.resolved_path)
    else:  # pragma: no cover - catalog validation makes this unreachable.
        raise ValueError(f"Unsupported source format {source.serialization!r}")


def _normalize_targets(targets: Mapping[str, float]) -> Dict[str, float]:
    if not targets:
        raise ValueError("target_loss_mass must not be empty")
    normalized: Dict[str, float] = {}
    for source, value in targets.items():
        if not isinstance(source, str) or not source:
            raise ValueError("target_loss_mass keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Target loss mass for {source!r} must be numeric")
        float_value = float(value)
        if not math.isfinite(float_value) or float_value <= 0:
            raise ValueError(f"Target loss mass for {source!r} must be finite and positive")
        normalized[source] = float_value
    total = math.fsum(normalized.values())
    return {source: value / total for source, value in sorted(normalized.items())}


def audit_source_catalog(
    catalog_path: os.PathLike[str] | str,
    target_loss_mass: Mapping[str, float],
    *,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit exact preprocessed examples and derive example-sampling probabilities.

    Invalid individual JSONL records are included in the report instead of aborting the scan.
    Any such record still makes the overall artifact fail calibration. Catalog and NPZ structural
    failures raise immediately because their example boundaries cannot be interpreted safely.

    :param catalog_path: Strict source catalog path.
    :param target_loss_mass: Desired effective loss mass by source.
    :param phase: Optional named recipe phase recorded as provenance.
    :returns: Canonicalizable audit dictionary, including its stable ``fingerprint``.
    :raises ValueError: If the catalog or requested targets are structurally invalid.
    """
    if phase is not None and phase not in VISION_ALIGNMENT_PHASES:
        raise ValueError(f"phase must be one of {VISION_ALIGNMENT_PHASES}, got {phase!r}")
    catalog = load_source_catalog(catalog_path)
    targets = _normalize_targets(target_loss_mass)

    source_reports: Dict[str, Any] = {}
    mean_loss_weight: Dict[str, float] = {}
    verified_row_hashes: Dict[str, List[str]] = {}
    failures = []
    for source in catalog.sources:
        accumulator = SourceAccumulator()
        row_hashes: List[str] = []
        for ordinal, (index, value) in enumerate(_iter_source(source)):
            if isinstance(value, str):
                accumulator.add_error(index, value)
                continue
            try:
                if source.probe_indices is not None:
                    if ordinal >= len(source.probe_indices):
                        raise ValueError("canonical probe contains more rows than pinned indices")
                    row_hashes.append(
                        _validate_canonical_probe_record(
                            value,
                            source_name=source.name,
                            expected_index=source.probe_indices[ordinal],
                        )
                    )
                accumulator.add_example(value)
            except ValueError as error:
                accumulator.add_error(index, str(error))
        if source.probe_indices is not None:
            if accumulator.seen != len(source.probe_indices):
                failures.append(
                    f"{source.name}: canonical probe contains {accumulator.seen} rows, expected "
                    f"{len(source.probe_indices)}"
                )
            actual_row_hashes_sha256 = _canonical_sha256(row_hashes)
            if actual_row_hashes_sha256 != source.serialized_row_hashes_sha256:
                failures.append(
                    f"{source.name}: serialized row hashes differ from the pinned runtime probe"
                )
        verified_row_hashes[source.name] = row_hashes
        post_scan_sha256 = _sha256_file(source.resolved_path)
        if post_scan_sha256 != source.sha256:
            raise ValueError(
                f"Source {source.name!r} changed while it was being audited: "
                f"started with {source.sha256}, finished with {post_scan_sha256}"
            )
        source_reports[source.name] = accumulator.as_dict()
        if accumulator.error_count:
            failures.append(f"{source.name}: {accumulator.error_count} malformed examples")
        if accumulator.valid == 0:
            failures.append(f"{source.name}: no valid examples")
        elif accumulator.loss_weight.total <= 0:
            failures.append(f"{source.name}: mean sum(loss_masks) is not positive")
        else:
            mean_loss_weight[source.name] = accumulator.loss_weight.total / accumulator.valid

    catalog_names = {source.name for source in catalog.sources}
    target_names = set(targets)
    if catalog_names != target_names:
        failures.append(
            "target/catalog source mismatch: "
            f"missing targets for {sorted(catalog_names - target_names)}, "
            f"missing catalog sources for {sorted(target_names - catalog_names)}"
        )

    sampling_probabilities: Optional[Dict[str, float]] = None
    realized_loss_mass: Optional[Dict[str, float]] = None
    if not failures:
        sampling_probabilities = sampling_weights_from_loss_mass(targets, mean_loss_weight)
        realized_loss_mass = expected_loss_mass(sampling_probabilities, mean_loss_weight)

    source_inputs = {
        source.name: {
            "format": source.serialization,
            "path": source.catalog_path,
            "sha256": source.sha256,
            "dataset_fingerprint": source.dataset_fingerprint,
            "dataset_size": source.dataset_size,
            "probe_indices": (
                list(source.probe_indices) if source.probe_indices is not None else None
            ),
            "probe_indices_sha256": source.probe_indices_sha256,
            "serialized_row_hashes": (
                verified_row_hashes[source.name] if source.probe_indices is not None else None
            ),
            "serialized_row_hashes_sha256": source.serialized_row_hashes_sha256,
        }
        for source in catalog.sources
    }
    report: Dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "version": AUDIT_VERSION,
        "auditor_sha256": _sha256_file(Path(__file__).resolve()),
        "status": "ok" if not failures else "failed",
        "phase": phase,
        "recipe_version": catalog.recipe_version,
        "formatter_version": catalog.formatter_version,
        "source_catalog_version": catalog.version,
        "source_registry_version": catalog.source_registry_version,
        "source_registry_sha256": catalog.source_registry_sha256,
        "exporter_sha256": catalog.exporter_sha256,
        "image_manifest_sha256": catalog.image_manifest_sha256,
        "preprocessing_config": catalog.preprocessing_config,
        "preprocessing_config_sha256": catalog.preprocessing_config_sha256,
        "probe": catalog.probe,
        "catalog_sha256": catalog.catalog_sha256,
        "input_content_sha256": catalog.input_content_sha256,
        "inputs": source_inputs,
        "target_loss_mass": targets,
        "sources": source_reports,
        "mean_loss_weight": mean_loss_weight,
        "sampling_probabilities": sampling_probabilities,
        "expected_loss_mass": realized_loss_mass,
        "failures": failures,
    }
    report["fingerprint"] = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    return report


def _parse_target_assignments(values: Sequence[str]) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for value in values:
        source, separator, raw_weight = value.partition("=")
        if not separator or not source or not raw_weight:
            raise ValueError(
                f"Invalid target {value!r}; expected repeated SOURCE=POSITIVE_WEIGHT values"
            )
        if source in targets:
            raise ValueError(f"Duplicate target source {source!r}")
        try:
            targets[source] = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"Invalid target weight in {value!r}") from error
    return targets


def _write_canonical_json(report: Mapping[str, Any], output: str) -> None:
    payload = _canonical_bytes(report) + b"\n"
    if output == "-":
        sys.stdout.buffer.write(payload)
        return
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(payload)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", help="Strict JSON source catalog")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--phase",
        choices=VISION_ALIGNMENT_PHASES,
        help="Use the checked-in effective-loss targets for this recipe phase",
    )
    target_group.add_argument(
        "--target-loss-mass",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help="Custom target; repeat once for every catalog source",
    )
    parser.add_argument("--output", required=True, help="Output JSON path, or '-' for stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line audit and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.phase is not None:
            mixture = VisionAlignmentMixtureConfig(phase=args.phase)
            targets = mixture.resolved_targets()
        else:
            targets = _parse_target_assignments(args.target_loss_mass)
        report = audit_source_catalog(args.catalog, targets, phase=args.phase)
        _write_canonical_json(report, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    if report["status"] != "ok":
        print("error: audit failed; inspect the emitted artifact's failures", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
