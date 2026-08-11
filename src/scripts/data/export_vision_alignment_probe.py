#!/usr/bin/env python
"""Export deterministic probes from the exact Vision Alignment runtime adapters.

This command is the canonical producer for
``audit_vision_alignment_mix.py``. It builds the same source configs and tokenizer as the
training launcher, materializes exact ``dataset.get(index, epoch=0)`` examples, and records
both their deterministic live indices and hashes of every model-consumed array. Large image
arrays are hashed rather than embedded in JSONL.

Only the bridge source set is currently complete. Future phases fail closed until every
checked-in source target has a canonical adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from olmo_core.data.multimodal.mixtures.vision_alignment import (
    VisionAlignmentMixtureConfig,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_PROBE_FORMAT,
    VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_PROBE_VERSION,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    VisionAlignmentSourceSpec,
    build_vision_alignment_dataset,
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
    select_deterministic_probe_indices,
    serialized_probe_record,
    vision_alignment_source_registry_sha256,
)

SOURCE_CATALOG_FORMAT = "vision_alignment_preprocessed_source_catalog"
SOURCE_CATALOG_VERSION = VISION_ALIGNMENT_SOURCE_CATALOG_VERSION
EXPORT_FORMAT = VISION_ALIGNMENT_PROBE_FORMAT
EXPORT_VERSION = VISION_ALIGNMENT_PROBE_VERSION
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
DEFAULT_PROBE_EXAMPLES = 1024
DEFAULT_PROBE_SEED = 6198
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_image_hash_manifest(path: Path) -> str:
    try:
        raw = path.read_bytes()
        hashes = raw.decode("utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read training image-hash manifest {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"Training image-hash manifest {path} is not UTF-8") from error
    if not hashes or any(_SHA256_RE.fullmatch(value) is None for value in hashes):
        raise ValueError("Training image-hash manifest contains an invalid SHA-256 row")
    if hashes != sorted(set(hashes)):
        raise ValueError("Training image-hash manifest must contain sorted unique SHA-256 rows")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(payload)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_source_probe(
    dataset: Any,
    *,
    source_name: str,
    output_path: os.PathLike[str] | str,
    num_examples: int,
    seed: int,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Export one deterministic runtime source probe as canonical JSONL.

    :param dataset: Exact built map-style runtime dataset.
    :param source_name: Canonical mixture source name.
    :param output_path: JSONL output path.
    :param num_examples: Number of deterministic probe examples.
    :param seed: Probe selection seed.
    :param overwrite: Replace an existing output when true.
    :returns: Strict source-catalog entry for the exported file.
    :raises ValueError: If source identity or serialized examples are invalid.
    """
    if re.fullmatch(r"[a-z0-9][a-z0-9_]{0,127}", source_name) is None:
        raise ValueError(f"Invalid canonical source name {source_name!r}")
    validate = getattr(dataset, "validate_required_annotations", None)
    if callable(validate):
        validate()
    dataset_fingerprint = runtime_dataset_fingerprint(dataset)
    if dataset_fingerprint is None:
        raise ValueError(f"Runtime source {source_name!r} does not expose a stable fingerprint")
    dataset_size = len(dataset)
    indices = select_deterministic_probe_indices(
        dataset_size,
        num_examples,
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
    )
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary_value)
    file_digest = hashlib.sha256()
    row_hashes = []
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            for dataset_index in indices:
                get = getattr(dataset, "get", None)
                example = get(dataset_index, 0) if callable(get) else dataset[dataset_index]
                record = serialized_probe_record(
                    example,
                    source_name=source_name,
                    dataset_index=dataset_index,
                    epoch=0,
                )
                row_hashes.append(record["serialized_row_sha256"])
                payload = _canonical_bytes(record) + b"\n"
                file_handle.write(payload)
                file_digest.update(payload)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "name": source_name,
        "format": "jsonl",
        "path": output.name,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_size": dataset_size,
        "sha256": file_digest.hexdigest(),
        "probe_indices": list(indices),
        "probe_indices_sha256": _canonical_sha256(list(indices)),
        "serialized_row_hashes_sha256": _canonical_sha256(row_hashes),
    }


def build_probe_catalog(
    *,
    spec: VisionAlignmentSourceSpec,
    source_entries: Sequence[Mapping[str, Any]],
    image_manifest_sha256: str,
    probe_seed: int,
    examples_per_source: int,
) -> Dict[str, Any]:
    """Build the strict canonical catalog consumed by the audit command.

    :param spec: Exact source preprocessing specification.
    :param source_entries: Entries returned by :func:`export_source_probe`.
    :param image_manifest_sha256: Digest of the full training image-hash manifest.
    :param probe_seed: Deterministic probe selection seed.
    :param examples_per_source: Required row count per source.
    :returns: Canonicalizable version-2 source catalog.
    """
    if _SHA256_RE.fullmatch(image_manifest_sha256) is None:
        raise ValueError("image_manifest_sha256 must be a lowercase SHA-256")
    names = [str(entry.get("name")) for entry in source_entries]
    if len(names) != len(set(names)) or names != sorted(names):
        raise ValueError("source_entries must have unique names in sorted order")
    return {
        "format": SOURCE_CATALOG_FORMAT,
        "version": SOURCE_CATALOG_VERSION,
        "recipe_version": VISION_ALIGNMENT_RECIPE_VERSION,
        "formatter_version": VISION_ALIGNMENT_FORMATTER_VERSION,
        "source_registry_version": VISION_ALIGNMENT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": vision_alignment_source_registry_sha256(),
        "exporter_sha256": _sha256_file(Path(__file__).resolve()),
        "image_manifest_sha256": image_manifest_sha256,
        "preprocessing_config": spec.as_canonical_dict(),
        "preprocessing_config_sha256": spec.preprocessing_sha256,
        "probe": {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "selection_algorithm": VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
            "seed": probe_seed,
            "epoch": 0,
            "examples_per_source": examples_per_source,
        },
        "sources": [dict(entry) for entry in source_entries],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("bridge", "perception", "joint"), required=True)
    parser.add_argument("--dataset-path", required=True, help="PixMoCap dataset root")
    parser.add_argument("--image-hashes", required=True, help="Sorted training image SHA file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--catalog-name", default="vision-alignment-source-catalog.json")
    parser.add_argument("--examples-per-source", type=int, default=DEFAULT_PROBE_EXAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROBE_SEED)
    parser.add_argument("--tokenizer-id", default=VISION_ALIGNMENT_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=VISION_ALIGNMENT_TOKENIZER_REVISION)
    parser.add_argument("--tokenizer-fingerprint", default=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run canonical Vision Alignment probe export and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.phase != "bridge":
            raise ValueError(
                "Canonical probe export is currently complete only for bridge; future phases "
                "remain closed until every checked-in source has an audited adapter"
            )
        if args.examples_per_source < DEFAULT_PROBE_EXAMPLES:
            raise ValueError(
                f"Production probes require at least {DEFAULT_PROBE_EXAMPLES} examples per source"
            )
        sequence_length = 2560
        spec = VisionAlignmentSourceSpec(
            phase=args.phase,
            pixmo_cap_path=str(Path(args.dataset_path).expanduser().resolve()),
            sequence_length=sequence_length,
            max_crops=8,
            message_format="document",
            loss_token_weighting="root_subsegments_root_tokens",
            caption_prompt="Description:",
            transcript_prompt="Transcript:",
            require_transcript=True,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            tokenizer_fingerprint=args.tokenizer_fingerprint,
        )
        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=spec.tokenizer_id,
            revision=spec.tokenizer_revision,
            expected_fingerprint=spec.tokenizer_fingerprint,
            cache_dir=args.hf_cache_dir,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        catalog_path = output_dir / args.catalog_name
        source_names = sorted(VisionAlignmentMixtureConfig(phase=args.phase).resolved_targets())
        expected_outputs = [catalog_path, *(output_dir / f"{name}.jsonl" for name in source_names)]
        existing = [path for path in expected_outputs if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing probe artifacts: {existing}")

        entries = []
        for source_name in source_names:
            dataset = build_vision_alignment_dataset(
                spec,
                tokenizer,
                token_ids,
                source_name,
                split="train",
                # export_source_probe() performs this gate immediately before materialization.
                validate_required_annotations=False,
            )
            entry = export_source_probe(
                dataset,
                source_name=source_name,
                output_path=output_dir / f"{source_name}.jsonl",
                num_examples=args.examples_per_source,
                seed=args.seed,
                overwrite=args.overwrite,
            )
            entries.append(entry)
        image_manifest_sha256 = _validate_image_hash_manifest(
            Path(args.image_hashes).expanduser().resolve()
        )
        catalog = build_probe_catalog(
            spec=spec,
            source_entries=entries,
            image_manifest_sha256=image_manifest_sha256,
            probe_seed=args.seed,
            examples_per_source=args.examples_per_source,
        )
        _atomic_write(
            catalog_path,
            _canonical_bytes(catalog) + b"\n",
            overwrite=args.overwrite,
        )
    except (FileExistsError, OSError, ValueError) as error:
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
