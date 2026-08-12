#!/usr/bin/env python
"""Export exact-runtime probes for the Vision Alignment perception mixture.

The bridge exporter remains byte-pinned by completed runs and is intentionally untouched.
This phase-specific exporter builds all eight perception adapters from the separate reviewed
registry, validates their annotations, serializes a deterministic multi-epoch probe panel, and emits
a strict catalog for loss-mass calibration. A separately produced image-provenance manifest
must be supplied; its raw digest is bound into the catalog but this command does not create or
approve train/evaluation splits.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VisionAlignmentMixtureConfig,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_SOURCE_NAMES,
    PerceptionProvenanceManifest,
    build_selected_perception_dataset,
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
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset,
    vision_alignment_perception_implementation_inventory,
    vision_alignment_perception_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
    select_deterministic_probe_indices,
    serialized_probe_record,
)

SOURCE_CATALOG_FORMAT = "vision_alignment_perception_preprocessed_source_catalog"
SOURCE_CATALOG_VERSION = VISION_ALIGNMENT_PERCEPTION_SOURCE_CATALOG_VERSION
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
DEFAULT_PROBE_EXAMPLES = VISION_ALIGNMENT_PERCEPTION_PROBE_EXAMPLES
DEFAULT_PROBE_SEED = VISION_ALIGNMENT_PERCEPTION_PROBE_SEED
DEFAULT_PROBE_EPOCHS = VISION_ALIGNMENT_PERCEPTION_PROBE_EPOCHS
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite perception artifact {destination}")
        os.rename(source, destination)
        return
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        == 0  # RENAME_NOREPLACE
    ):
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Refusing to overwrite perception artifact {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def export_source_probe(
    dataset: Any,
    *,
    source_name: str,
    output_path: Path,
    num_examples: int,
    seed: int,
    epochs: int,
    unbounded_dataset: Optional[Any] = None,
    max_sequence_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Write a deterministic canonical JSONL probe for one perception source."""
    validate = getattr(dataset, "validate_required_annotations", None)
    if not callable(validate):
        raise ValueError(f"Perception source {source_name!r} has no annotation validator")
    validate()
    fingerprint = runtime_dataset_fingerprint(dataset)
    if fingerprint is None or _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"Perception source {source_name!r} lacks a 64-hex fingerprint")
    if epochs < 1 or num_examples % epochs:
        raise ValueError("Perception probe examples must divide its positive epoch count")
    unique_indices = num_examples // epochs
    indices = select_deterministic_probe_indices(
        len(dataset),
        unique_indices,
        seed=seed,
        dataset_fingerprint=fingerprint,
    )
    validate_image_content = getattr(dataset, "validate_image_content", None)
    if not callable(validate_image_content):
        raise ValueError(f"Perception source {source_name!r} lacks image-content validation")
    probe_pairs = tuple((index, epoch) for epoch in range(epochs) for index in indices)
    probe_image_content_sha256 = validate_image_content(indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable probe {output_path}")
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_value)
    file_digest = hashlib.sha256()
    row_hashes = []
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            for dataset_index, probe_epoch in probe_pairs:
                example = dataset.get(dataset_index, probe_epoch)
                record = serialized_probe_record(
                    example,
                    source_name=source_name,
                    dataset_index=dataset_index,
                    epoch=probe_epoch,
                )
                if unbounded_dataset is not None:
                    if max_sequence_length is None or max_sequence_length <= 0:
                        raise ValueError("Truncation evidence requires a positive sequence limit")
                    raw_indices = getattr(dataset, "indices", None)
                    if not isinstance(raw_indices, tuple):
                        raise ValueError(
                            f"Perception source {source_name!r} lacks selected raw indices"
                        )
                    raw_index = raw_indices[dataset_index]
                    metadata = example.get("metadata")
                    if isinstance(metadata, Mapping) and "original_length" in metadata:
                        unbounded_length = int(metadata["original_length"])
                    else:
                        unbounded_example = unbounded_dataset.get(raw_index, probe_epoch)
                        unbounded_length = len(unbounded_example["input_ids"])
                    if unbounded_length >= 2**31 - 1:
                        raise ValueError(
                            f"Perception source {source_name!r} exceeds the audit length bound"
                        )
                    record["truncated"] = unbounded_length > max_sequence_length
                else:
                    raise ValueError(f"Perception source {source_name!r} lacks truncation evidence")
                row_hashes.append(record["serialized_row_sha256"])
                raw = _canonical_bytes(record) + b"\n"
                file_handle.write(raw)
                file_digest.update(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        if validate_image_content(indices) != probe_image_content_sha256:
            raise ValueError(f"Perception source {source_name!r} image bytes changed during export")
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "name": source_name,
        "format": "jsonl",
        "path": output_path.name,
        "dataset_fingerprint": fingerprint,
        "dataset_size": len(dataset),
        "sha256": file_digest.hexdigest(),
        "probe_indices": list(indices),
        "probe_indices_sha256": _canonical_sha256(list(indices)),
        "probe_epochs": epochs,
        "serialized_row_hashes_sha256": _canonical_sha256(row_hashes),
        "probe_image_content_sha256": probe_image_content_sha256,
    }


def build_probe_catalog(
    *,
    spec: VisionAlignmentPerceptionSourceSpec,
    source_entries: Sequence[Mapping[str, Any]],
    image_provenance: PerceptionProvenanceManifest,
    implementation_inventory: Mapping[str, Any],
    source_registry_sha256: str,
    exporter_sha256: str,
    probe_seed: int,
    examples_per_source: int,
    epochs: int,
) -> Dict[str, Any]:
    """Build the strict perception source catalog consumed by its phase auditor."""
    names = [str(entry.get("name")) for entry in source_entries]
    if names != sorted(set(names)):
        raise ValueError("Perception source entries must be unique and canonically ordered")
    return {
        "format": SOURCE_CATALOG_FORMAT,
        "version": SOURCE_CATALOG_VERSION,
        "phase": "perception",
        "recipe_version": VISION_ALIGNMENT_RECIPE_VERSION,
        "formatter_version": VISION_ALIGNMENT_FORMATTER_VERSION,
        "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": source_registry_sha256,
        "source_implementation_inventory": dict(implementation_inventory),
        "exporter_sha256": exporter_sha256,
        "image_provenance": {
            "path": str(image_provenance.path),
            "sha256": image_provenance.raw_sha256,
            "content_sha256": image_provenance.content_sha256,
            "source_spec_sha256": image_provenance.source_spec_sha256,
        },
        "preprocessing_config": spec.as_canonical_dict(),
        "preprocessing_config_sha256": spec.preprocessing_sha256,
        "probe": {
            "format": VISION_ALIGNMENT_PERCEPTION_PROBE_FORMAT,
            "version": VISION_ALIGNMENT_PERCEPTION_PROBE_VERSION,
            "selection_algorithm": VISION_ALIGNMENT_PERCEPTION_PROBE_SELECTION_ALGORITHM,
            "seed": probe_seed,
            "epochs": epochs,
            "examples_per_source": examples_per_source,
        },
        "sources": [dict(entry) for entry in source_entries],
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable catalog {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-provenance-manifest", required=True)
    parser.add_argument("--expected-image-provenance-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-per-source", type=int, default=DEFAULT_PROBE_EXAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROBE_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_PROBE_EPOCHS)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Export all canonical perception probes and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.examples_per_source != DEFAULT_PROBE_EXAMPLES:
            raise ValueError(
                f"Production perception probes require exactly {DEFAULT_PROBE_EXAMPLES} rows"
            )
        if args.epochs != DEFAULT_PROBE_EPOCHS:
            raise ValueError(
                f"Production perception probes require exactly {DEFAULT_PROBE_EPOCHS} epochs"
            )
        if args.seed != DEFAULT_PROBE_SEED:
            raise ValueError(f"Production perception probes require seed {DEFAULT_PROBE_SEED}")
        provenance = load_perception_provenance_manifest(
            args.image_provenance_manifest,
            expected_sha256=args.expected_image_provenance_sha256,
        )
        spec = provenance.source_spec
        implementation_inventory = vision_alignment_perception_implementation_inventory()
        source_registry_sha256 = vision_alignment_perception_source_registry_sha256()
        exporter_sha256 = _sha256_file(Path(__file__).resolve())
        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=VISION_ALIGNMENT_TOKENIZER_ID,
            revision=VISION_ALIGNMENT_TOKENIZER_REVISION,
            expected_fingerprint=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
            cache_dir=args.hf_cache_dir,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        source_names = tuple(
            sorted(VisionAlignmentMixtureConfig(phase="perception").resolved_targets())
        )
        if source_names != PERCEPTION_SOURCE_NAMES:
            raise ValueError("Perception mix and provenance source sets differ")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite perception artifact {output_dir}")
        lock_path = output_dir.with_name(f".{output_dir.name}.lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(lock_fd)
            raise RuntimeError(f"Another perception exporter holds {lock_path}") from error
        staging_dir: Optional[Path] = None
        entries = []
        try:
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_dir.name}.", suffix=".building", dir=output_dir.parent
                )
            )
            for source_name in source_names:
                dataset = build_selected_perception_dataset(
                    provenance,
                    tokenizer,
                    token_ids,
                    source_name,
                    logical_split="train",
                    validate_required_annotations=False,
                )
                selection = provenance.selection(source_name, "train")
                if source_name in ("pixmo_caption", "pixmo_transcript"):
                    # PixMoCap is shared with the completed bridge, so build a separate
                    # perception-only unbounded adapter rather than changing that class.
                    unbounded_spec = replace(spec, sequence_length=2**31 - 1)
                    unbounded_dataset = build_vision_alignment_perception_dataset(
                        unbounded_spec,
                        tokenizer,
                        token_ids,
                        source_name,
                        split=selection.physical_split,
                        validate_required_annotations=False,
                    )
                else:
                    unbounded_dataset = dataset._dataset
                if len(unbounded_dataset) != selection.base_examples:
                    raise ValueError(
                        f"Perception source {source_name!r} changes row identity when "
                        "measuring truncation"
                    )
                entries.append(
                    export_source_probe(
                        dataset,
                        source_name=source_name,
                        output_path=staging_dir / f"{source_name}.jsonl",
                        num_examples=args.examples_per_source,
                        seed=args.seed,
                        epochs=args.epochs,
                        unbounded_dataset=unbounded_dataset,
                        max_sequence_length=spec.sequence_length,
                    )
                )
            catalog = build_probe_catalog(
                spec=spec,
                source_entries=entries,
                image_provenance=provenance,
                implementation_inventory=implementation_inventory,
                source_registry_sha256=source_registry_sha256,
                exporter_sha256=exporter_sha256,
                probe_seed=args.seed,
                examples_per_source=args.examples_per_source,
                epochs=args.epochs,
            )
            catalog_path = staging_dir / "vision-alignment-perception-source-catalog.json"
            _write_once(catalog_path, catalog)
            if (
                vision_alignment_perception_implementation_inventory() != implementation_inventory
                or vision_alignment_perception_source_registry_sha256() != source_registry_sha256
                or _sha256_file(Path(__file__).resolve()) != exporter_sha256
                or _sha256_file(provenance.path) != provenance.raw_sha256
            ):
                raise ValueError("Perception exporter/source implementation changed during export")
            _fsync_tree(staging_dir)
            _publish_no_replace(staging_dir, output_dir)
            _fsync_directory(output_dir.parent)
            catalog_path = output_dir / catalog_path.name
        except BaseException:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "catalog": str(catalog_path),
                "preprocessing_config_sha256": spec.preprocessing_sha256,
                "sources": {entry["name"]: args.examples_per_source for entry in entries},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
