#!/usr/bin/env python
"""Audit exact perception runtime probes and calibrate the eight-source loss-mass mix.

This phase-specific auditor deliberately leaves the completed bridge auditor byte-identical.
It accepts only the immutable catalog emitted by
``export_vision_alignment_perception_probe.py`` and revalidates its source implementation,
image-provenance manifest, deterministic row selection, serialized model inputs, and exact file
bytes before deriving example-sampling probabilities.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VisionAlignmentMixtureConfig,
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_PROBE_EPOCHS,
    VISION_ALIGNMENT_PERCEPTION_PROBE_EXAMPLES,
    VISION_ALIGNMENT_PERCEPTION_PROBE_FORMAT,
    VISION_ALIGNMENT_PERCEPTION_PROBE_SEED,
    VISION_ALIGNMENT_PERCEPTION_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_PERCEPTION_PROBE_VERSION,
    VISION_ALIGNMENT_PERCEPTION_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_RECIPE_VERSION,
    select_deterministic_probe_indices,
)
from scripts.data import audit_vision_alignment_mix as shared_audit

CATALOG_FORMAT = "vision_alignment_perception_preprocessed_source_catalog"
CATALOG_VERSION = VISION_ALIGNMENT_PERCEPTION_SOURCE_CATALOG_VERSION
AUDIT_FORMAT = "vision_alignment_perception_source_audit"
AUDIT_VERSION = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROOT_FIELDS = frozenset(
    {
        "format",
        "version",
        "phase",
        "recipe_version",
        "formatter_version",
        "source_registry_version",
        "source_registry_sha256",
        "source_implementation_inventory",
        "exporter_sha256",
        "image_provenance",
        "preprocessing_config",
        "preprocessing_config_sha256",
        "probe",
        "sources",
    }
)
_PROVENANCE_FIELDS = frozenset({"path", "sha256", "content_sha256", "source_spec_sha256"})
_PROBE_FIELDS = frozenset(
    {"format", "version", "selection_algorithm", "seed", "epochs", "examples_per_source"}
)
_SOURCE_FIELDS = frozenset(
    {
        "name",
        "format",
        "path",
        "dataset_fingerprint",
        "dataset_size",
        "sha256",
        "probe_indices",
        "probe_indices_sha256",
        "probe_epochs",
        "serialized_row_hashes_sha256",
        "probe_image_content_sha256",
    }
)


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
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
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


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _catalog_local_path(catalog: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or "://" in value:
        raise ValueError(f"{name} must be a non-empty local relative path")
    path = (catalog.parent / value).resolve()
    if catalog.parent != path and catalog.parent not in path.parents:
        raise ValueError(f"{name} escapes the catalog directory")
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _load_catalog(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        root = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid perception source catalog {path}: {error}") from error
    return raw, _exact_mapping(root, _ROOT_FIELDS, name="perception source catalog")


def _iter_jsonl_bytes(raw: bytes):
    """Yield strict JSONL records parsed from the exact bytes whose digest was pinned."""
    for index, raw_line in enumerate(raw.splitlines()):
        try:
            value = json.loads(raw_line, object_pairs_hook=_strict_json_object)
            if not isinstance(value, Mapping):
                raise ValueError("example must be an object")
            yield index, value
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            yield index, f"invalid JSONL record: {error}"


def audit_perception_catalog(
    path: str | Path, *, expected_catalog_sha256: Optional[str] = None
) -> dict[str, Any]:
    """Audit a strict perception probe catalog and return its canonical report."""
    catalog_path = Path(path).expanduser().resolve()
    auditor_path = Path(__file__).resolve()
    auditor_sha256 = _sha256_file(auditor_path)
    catalog_raw, catalog = _load_catalog(catalog_path)
    catalog_sha256 = hashlib.sha256(catalog_raw).hexdigest()
    if expected_catalog_sha256 is not None and catalog_sha256 != _sha256(
        expected_catalog_sha256, name="expected catalog SHA-256"
    ):
        raise ValueError("Perception catalog differs from its external SHA-256 pin")
    if (
        catalog["format"] != CATALOG_FORMAT
        or _integer(catalog["version"], name="catalog.version", minimum=1) != CATALOG_VERSION
        or catalog["phase"] != "perception"
    ):
        raise ValueError("Perception catalog identity is incompatible")
    if (
        _integer(catalog["recipe_version"], name="catalog.recipe_version", minimum=1)
        != VISION_ALIGNMENT_RECIPE_VERSION
        or catalog["formatter_version"] != VISION_ALIGNMENT_FORMATTER_VERSION
    ):
        raise ValueError("Perception catalog recipe or formatter identity differs")

    implementation_inventory = vision_alignment_perception_implementation_inventory()
    registry_sha256 = vision_alignment_perception_source_registry_sha256()
    if (
        _integer(
            catalog["source_registry_version"],
            name="catalog.source_registry_version",
            minimum=1,
        )
        != VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
        or catalog["source_registry_sha256"] != registry_sha256
        or catalog["source_implementation_inventory"] != implementation_inventory
    ):
        raise ValueError("Perception catalog source implementation differs")
    exporter_path = Path(__file__).resolve().parent / "export_vision_alignment_perception_probe.py"
    exporter_sha256 = _sha256(catalog["exporter_sha256"], name="catalog.exporter_sha256")
    if _sha256_file(exporter_path) != exporter_sha256:
        raise ValueError("Perception catalog exporter bytes differ")

    provenance_ref = _exact_mapping(
        catalog["image_provenance"], _PROVENANCE_FIELDS, name="catalog.image_provenance"
    )
    provenance_path = Path(provenance_ref["path"]).expanduser().resolve()
    provenance_sha256 = _sha256(provenance_ref["sha256"], name="provenance raw SHA-256")
    provenance = load_perception_provenance_manifest(
        provenance_path, expected_sha256=provenance_sha256
    )
    if (
        provenance.content_sha256 != provenance_ref["content_sha256"]
        or provenance.source_spec_sha256 != provenance_ref["source_spec_sha256"]
        or catalog["preprocessing_config"] != provenance.source_spec.as_canonical_dict()
        or catalog["preprocessing_config_sha256"] != provenance.source_spec_sha256
    ):
        raise ValueError("Perception catalog preprocessing/provenance identity differs")

    probe = _exact_mapping(catalog["probe"], _PROBE_FIELDS, name="catalog.probe")
    examples_per_source = _integer(
        probe["examples_per_source"], name="probe.examples_per_source", minimum=1024
    )
    probe_seed = _integer(probe["seed"], name="probe.seed", minimum=0)
    probe_epochs = _integer(probe["epochs"], name="probe.epochs", minimum=1)
    if (
        probe["format"] != VISION_ALIGNMENT_PERCEPTION_PROBE_FORMAT
        or _integer(probe["version"], name="probe.version", minimum=1)
        != VISION_ALIGNMENT_PERCEPTION_PROBE_VERSION
        or probe["selection_algorithm"] != VISION_ALIGNMENT_PERCEPTION_PROBE_SELECTION_ALGORITHM
        or examples_per_source != VISION_ALIGNMENT_PERCEPTION_PROBE_EXAMPLES
        or probe_epochs != VISION_ALIGNMENT_PERCEPTION_PROBE_EPOCHS
        or probe_seed != VISION_ALIGNMENT_PERCEPTION_PROBE_SEED
    ):
        raise ValueError("Perception probe identity is incompatible")

    raw_sources = catalog["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(PERCEPTION_SOURCE_NAMES):
        raise ValueError("Perception catalog must contain exactly eight sources")
    names = [value.get("name") if isinstance(value, Mapping) else None for value in raw_sources]
    if tuple(names) != PERCEPTION_SOURCE_NAMES:
        raise ValueError("Perception catalog sources are not the exact canonical ordering")

    targets = VisionAlignmentMixtureConfig(phase="perception").resolved_targets()
    source_reports: dict[str, Any] = {}
    source_inputs: dict[str, Any] = {}
    mean_loss_weight: dict[str, float] = {}
    failures: list[str] = []
    input_descriptor = []
    for ordinal, raw_source in enumerate(raw_sources):
        source = _exact_mapping(raw_source, _SOURCE_FIELDS, name=f"catalog.sources[{ordinal}]")
        source_name = names[ordinal]
        assert isinstance(source_name, str)
        if source["format"] != "jsonl":
            raise ValueError(f"Perception source {source_name!r} must be JSONL")
        if source["path"] != f"{source_name}.jsonl":
            raise ValueError(f"Perception source {source_name!r} filename is not canonical")
        selected = provenance.selection(source_name, "train")
        if source["dataset_fingerprint"] != selected.runtime_dataset_fingerprint or _integer(
            source["dataset_size"], name=f"{source_name}.dataset_size", minimum=1
        ) != len(selected.indices):
            raise ValueError(f"Perception source {source_name!r} runtime identity differs")
        indices_raw = source["probe_indices"]
        if not isinstance(indices_raw, list):
            raise ValueError(f"Perception source {source_name!r} probe_indices must be a list")
        indices = tuple(
            _integer(value, name=f"{source_name}.probe_indices", minimum=0) for value in indices_raw
        )
        expected_indices = select_deterministic_probe_indices(
            len(selected.indices),
            examples_per_source // probe_epochs,
            seed=probe_seed,
            dataset_fingerprint=selected.runtime_dataset_fingerprint,
        )
        if indices != expected_indices:
            raise ValueError(f"Perception source {source_name!r} probe selection differs")
        if _canonical_sha256(list(indices)) != _sha256(
            source["probe_indices_sha256"], name=f"{source_name}.probe_indices_sha256"
        ):
            raise ValueError(f"Perception source {source_name!r} probe-index digest differs")
        if source["probe_epochs"] != probe_epochs:
            raise ValueError(f"Perception source {source_name!r} epoch panel differs")
        expected_image_digest = _canonical_sha256(
            [
                {
                    "index": index,
                    "image_sha256": selected.row_image_content_sha256[index],
                }
                for index in indices
            ]
        )
        if source["probe_image_content_sha256"] != expected_image_digest:
            raise ValueError(f"Perception source {source_name!r} image-content probe differs")

        source_path = _catalog_local_path(catalog_path, source["path"], name=f"{source_name}.path")
        expected_source_sha = _sha256(source["sha256"], name=f"{source_name}.sha256")
        try:
            source_raw = source_path.read_bytes()
        except OSError as error:
            raise ValueError(f"Could not read perception source {source_path}: {error}") from error
        source_sha = hashlib.sha256(source_raw).hexdigest()
        if source_sha != expected_source_sha:
            raise ValueError(f"Perception source {source_name!r} file SHA-256 differs")
        accumulator = shared_audit.SourceAccumulator()
        row_hashes = []
        expected_pairs = tuple((index, epoch) for epoch in range(probe_epochs) for index in indices)
        for row_ordinal, (row_index, value) in enumerate(_iter_jsonl_bytes(source_raw)):
            if isinstance(value, str):
                accumulator.add_error(row_index, value)
                continue
            try:
                if row_ordinal >= len(expected_pairs):
                    raise ValueError("probe contains more rows than its pinned epoch panel")
                if not isinstance(value.get("truncated"), bool):
                    raise ValueError("perception probe requires a boolean truncation flag")
                canonical_value = dict(value)
                canonical_value.pop("truncated")
                expected_index, expected_epoch = expected_pairs[row_ordinal]
                if canonical_value.get("probe_epoch") != expected_epoch:
                    raise ValueError("probe epoch differs from its canonical panel")
                # The shared bridge validator owns an epoch-zero schema. We already
                # checked the perception epoch above, then reuse its structural checks.
                canonical_value["probe_epoch"] = 0
                row_hashes.append(
                    shared_audit._validate_canonical_probe_record(
                        canonical_value,
                        source_name=source_name,
                        expected_index=expected_index,
                    )
                )
                accumulator.add_example(value)
            except ValueError as error:
                accumulator.add_error(row_index, str(error))
        if accumulator.seen != examples_per_source:
            failures.append(
                f"{source_name}: saw {accumulator.seen} probe rows, expected {examples_per_source}"
            )
        expected_row_hashes_sha = _sha256(
            source["serialized_row_hashes_sha256"],
            name=f"{source_name}.serialized_row_hashes_sha256",
        )
        if _canonical_sha256(row_hashes) != expected_row_hashes_sha:
            failures.append(f"{source_name}: serialized row hashes differ")
        if source_path.read_bytes() != source_raw:
            raise ValueError(f"Perception source {source_name!r} changed during audit")
        source_reports[source_name] = accumulator.as_dict()
        if accumulator.error_count or accumulator.valid != examples_per_source:
            failures.append(f"{source_name}: {accumulator.error_count} malformed probe rows")
        if accumulator.truncated:
            failures.append(
                f"{source_name}: {accumulator.truncated} probe rows were truncated at 2560 tokens"
            )
        if accumulator.loss_weight.total <= 0 or accumulator.zero_loss:
            failures.append(f"{source_name}: has non-positive supervised loss mass")
        else:
            mean_loss_weight[source_name] = accumulator.loss_weight.total / accumulator.valid
        source_inputs[source_name] = {
            **dict(source),
            "path": str(source["path"]),
            "serialized_row_hashes": row_hashes,
        }
        input_descriptor.append(
            {
                "name": source_name,
                "sha256": source_sha,
                "dataset_fingerprint": selected.runtime_dataset_fingerprint,
                "probe_indices_sha256": source["probe_indices_sha256"],
                "probe_epochs": probe_epochs,
                "serialized_row_hashes_sha256": expected_row_hashes_sha,
                "probe_image_content_sha256": expected_image_digest,
            }
        )

    sampling_probabilities = None
    expected_mass = None
    if not failures:
        sampling_probabilities = sampling_weights_from_loss_mass(targets, mean_loss_weight)
        expected_mass = expected_loss_mass(sampling_probabilities, mean_loss_weight)
        if any(
            not math.isclose(expected_mass[name], targets[name], abs_tol=1e-12) for name in targets
        ):
            failures.append("calibrated expected loss mass differs from its exact targets")

    shared_path = Path(shared_audit.__file__).resolve()
    report: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "version": AUDIT_VERSION,
        "status": "ok" if not failures else "failed",
        "phase": "perception",
        "recipe_version": VISION_ALIGNMENT_RECIPE_VERSION,
        "formatter_version": VISION_ALIGNMENT_FORMATTER_VERSION,
        "source_catalog_version": CATALOG_VERSION,
        "auditor_sha256": auditor_sha256,
        "shared_auditor_sha256": _sha256_file(shared_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "input_content_sha256": _canonical_sha256(input_descriptor),
        "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": registry_sha256,
        "source_implementation_inventory": implementation_inventory,
        "exporter_sha256": exporter_sha256,
        "image_provenance": dict(provenance_ref),
        "preprocessing_config": provenance.source_spec.as_canonical_dict(),
        "preprocessing_config_sha256": provenance.source_spec_sha256,
        "probe": dict(probe),
        "inputs": source_inputs,
        "target_loss_mass": targets,
        "sources": source_reports,
        "mean_loss_weight": mean_loss_weight,
        "sampling_probabilities": sampling_probabilities,
        "expected_loss_mass": expected_mass,
        "failures": failures,
    }
    report["fingerprint"] = _canonical_sha256(report)
    if (
        vision_alignment_perception_implementation_inventory() != implementation_inventory
        or vision_alignment_perception_source_registry_sha256() != registry_sha256
        or _sha256_file(exporter_path) != exporter_sha256
        or _sha256_file(auditor_path) != auditor_sha256
        or _sha256_file(shared_path) != report["shared_auditor_sha256"]
    ):
        raise ValueError("Perception source/audit implementation changed during audit")
    return report


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable perception audit {path}")
    raw = _canonical_bytes(value) + b"\n"
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Audit the exact perception catalog and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        catalog_path = Path(args.catalog).expanduser().resolve()
        report = audit_perception_catalog(
            catalog_path,
            expected_catalog_sha256=args.expected_catalog_sha256,
        )
        _write_once(Path(args.output), report)
    except (FileExistsError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    if report["status"] != "ok":
        print("error: perception audit failed; inspect failures", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
