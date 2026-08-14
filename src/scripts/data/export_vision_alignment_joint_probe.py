#!/usr/bin/env python
"""Export the exact nine-source Vision Alignment joint runtime probe catalog.

The eight visual sources are the reviewed perception row selections instantiated with the
joint phase's 8,192-token adapters.  Every probed visual row is independently rebuilt through
an otherwise identical unbounded adapter.  Export fails if the two serializations differ or
if the raw serialization would exceed 8,192 tokens; a joint calibration artifact therefore
cannot silently admit truncation.  Native replay contributes a separate 1,024-row epoch-zero
panel and must be exactly 8,192 tokens per row.

The catalog records the exporter's file digest as data.  It deliberately does not embed an
expected digest in this source file, avoiding a self-referential hash contract.
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

from olmo_core.data.multimodal.native_text_replay import (
    NativeTextReplayDataset,
    NativeTextReplayManifest,
    NativeTextReplayVerificationReceipt,
)
from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JOINT_VISUAL_UNBOUNDED_SEQUENCE_LENGTH,
    JointVisualProjectionManifest,
    build_selected_joint_dataset,
    joint_alignment_runtime_implementation_inventory,
    joint_alignment_runtime_registry_sha256,
    load_joint_visual_projection_manifest,
    validate_joint_live_example,
    validate_joint_probe_record,
    validate_joint_unbounded_dataset_identity,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
    build_vision_alignment_joint_dataset_config,
    vision_alignment_joint_adapter_projection_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
    select_deterministic_probe_indices,
    serialized_probe_record,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision import Molmo2TokenIds

SOURCE_CATALOG_FORMAT = "vision_alignment_joint_preprocessed_source_catalog"
SOURCE_CATALOG_VERSION = VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION
JOINT_PROBE_FORMAT = "vision_alignment_joint_runtime_probe"
JOINT_PROBE_VERSION = 1
JOINT_PROBE_SEED = 6198
JOINT_VISUAL_PROBE_INDICES = 256
JOINT_VISUAL_PROBE_EPOCHS = (0, 1, 2, 3)
JOINT_NATIVE_PROBE_INDICES = 1024
JOINT_NATIVE_PROBE_EPOCHS = (0,)
JOINT_SEQUENCE_LENGTH = 8192
UNBOUNDED_SEQUENCE_LENGTH = JOINT_VISUAL_UNBOUNDED_SEQUENCE_LENGTH
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
CATALOG_NAME = "vision-alignment-joint-source-catalog.json"
EXPORTER_IMPLEMENTATION_PATH = "src/scripts/data/export_vision_alignment_joint_probe.py"
JOINT_SOURCE_NAMES = tuple(sorted((*JOINT_VISUAL_SOURCE_NAMES, "native_text_replay")))

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_ENTRY_FIELDS = frozenset(
    {
        "name",
        "kind",
        "format",
        "path",
        "dataset_fingerprint",
        "dataset_size",
        "sha256",
        "probe_epochs",
        "probe_indices",
        "probe_indices_sha256",
        "serialized_row_hashes_sha256",
        "probe_image_content_sha256",
        "max_observed_sequence_length",
        "truncated_rows",
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
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_native_manifest_pinned(
    path: Path, *, expected_raw_sha256: str, expected_content_fingerprint: str
) -> NativeTextReplayManifest:
    """Load native replay while binding the parser's bytes to both external identities."""
    before_sha256 = _sha256_file(path)
    if before_sha256 != expected_raw_sha256:
        raise ValueError("Native train manifest differs from its external raw SHA-256 pin")
    manifest = NativeTextReplayManifest.load(path)
    if (
        manifest.manifest_sha256 != expected_raw_sha256
        or manifest.content_fingerprint != expected_content_fingerprint
    ):
        raise ValueError("Native train manifest runtime identity differs from its external pins")
    return manifest


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
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Refusing to overwrite joint artifact {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _dataset_get(dataset: Any, index: int, epoch: int) -> Mapping[str, Any]:
    get = getattr(dataset, "get", None)
    value = get(index, epoch) if callable(get) else dataset[index]
    if not isinstance(value, Mapping):
        raise ValueError(f"Joint probe row {index} at epoch {epoch} must be an object")
    return value


def _sequence_length(example: Mapping[str, Any], *, name: str) -> int:
    input_ids = example.get("input_ids")
    if input_ids is None:
        raise ValueError(f"{name} is missing input_ids")
    try:
        length = len(input_ids)
    except TypeError as error:
        raise ValueError(f"{name} input_ids must be one-dimensional array-like data") from error
    if isinstance(length, bool) or length < 1:
        raise ValueError(f"{name} input_ids must not be empty")
    return length


def _probe_image_digest(dataset: Any, indices: Sequence[int], *, source_name: str) -> str:
    validate_image_content = getattr(dataset, "validate_image_content", None)
    if not callable(validate_image_content):
        raise ValueError(f"Joint visual source {source_name!r} lacks image-content validation")
    digest = validate_image_content(indices)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"Joint visual source {source_name!r} returned an invalid image digest")
    return digest


def _live_probe_record(
    dataset: Any,
    *,
    source_name: str,
    kind: str,
    dataset_index: int,
    epoch: int,
    unbounded_dataset: Optional[Any],
    token_ids: Molmo2TokenIds,
) -> Dict[str, Any]:
    example = _dataset_get(dataset, dataset_index, epoch)
    validate_joint_live_example(
        example,
        source_name=source_name,
        source_kind=kind,
        token_ids=token_ids,
    )
    record = serialized_probe_record(
        example,
        source_name=source_name,
        dataset_index=dataset_index,
        epoch=epoch,
    )
    if kind == "visual":
        if unbounded_dataset is None:
            raise ValueError(f"Joint visual source {source_name!r} lacks an unbounded adapter")
        raw_indices = getattr(dataset, "indices", None)
        if not isinstance(raw_indices, tuple) or len(raw_indices) != len(dataset):
            raise ValueError(f"Joint visual source {source_name!r} lacks exact projected indices")
        raw_index = raw_indices[dataset_index]
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ValueError(f"Joint visual source {source_name!r} has an invalid projected row")
        unbounded_example = _dataset_get(unbounded_dataset, raw_index, epoch)
        validate_joint_live_example(
            unbounded_example,
            source_name=source_name,
            source_kind=kind,
            token_ids=token_ids,
        )
        raw_length = _sequence_length(
            unbounded_example, name=f"unbounded {source_name}/{dataset_index}/{epoch}"
        )
        if raw_length > JOINT_SEQUENCE_LENGTH:
            raise ValueError(
                f"Joint visual source {source_name!r} row {dataset_index} epoch {epoch} "
                f"requires {raw_length} tokens, exceeding {JOINT_SEQUENCE_LENGTH}; truncation "
                "is forbidden"
            )
        unbounded_record = serialized_probe_record(
            unbounded_example,
            source_name=source_name,
            dataset_index=dataset_index,
            epoch=epoch,
        )
        if _canonical_bytes(unbounded_record) != _canonical_bytes(record):
            raise ValueError(
                f"Joint visual source {source_name!r} row {dataset_index} epoch {epoch} "
                "differs from its unbounded serialization"
            )
    elif kind == "native_text_replay":
        if unbounded_dataset is not None:
            raise ValueError("Native replay must not name an unbounded visual adapter")
        raw_length = _sequence_length(example, name=f"native_text_replay/{dataset_index}/{epoch}")
        if raw_length != JOINT_SEQUENCE_LENGTH:
            raise ValueError(
                f"Native replay row {dataset_index} must contain exactly "
                f"{JOINT_SEQUENCE_LENGTH} tokens, got {raw_length}"
            )
        if record["image_crops"] != 0 or record["pooled_tokens"] != 0:
            raise ValueError("Native replay rows must not contain visual inputs")
    else:
        raise ValueError(f"Unknown joint probe source kind {kind!r}")
    record["raw_sequence_length"] = raw_length
    record["truncated"] = False
    validate_joint_probe_record(
        record,
        source_name=source_name,
        source_kind=kind,
        expected_index=dataset_index,
        expected_epoch=epoch,
        sequence_length=JOINT_SEQUENCE_LENGTH,
        token_ids=token_ids,
    )
    return record


def export_source_probe(
    dataset: Any,
    *,
    source_name: str,
    kind: str,
    output_path: Path,
    unique_indices: int,
    epochs: Sequence[int],
    seed: int,
    token_ids: Molmo2TokenIds,
    unbounded_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write one immutable deterministic joint probe and return its catalog entry.

    :param dataset: Exact selected visual adapter or native replay dataset.
    :param source_name: Canonical nine-way mixture source name.
    :param kind: ``visual`` or ``native_text_replay``.
    :param output_path: Destination for canonical JSONL bytes.
    :param unique_indices: Number of deterministic logical indices.
    :param epochs: Exact ordered epoch panel applied to every selected index.
    :param seed: Deterministic affine selection seed.
    :param token_ids: Exact tokenizer-adapted Molmo2 IDs for this joint runtime.
    :param unbounded_dataset: Required raw unbounded adapter for visual sources.
    :returns: Canonical source-catalog entry.
    :raises ValueError: If runtime identity, serialization, images, or length evidence differs.
    """
    if source_name not in JOINT_SOURCE_NAMES:
        raise ValueError(f"Unknown joint source {source_name!r}")
    expected_kind = "native_text_replay" if source_name == "native_text_replay" else "visual"
    if kind != expected_kind:
        raise ValueError(f"Joint source {source_name!r} requires kind {expected_kind!r}")
    epoch_panel = tuple(epochs)
    if not epoch_panel or any(
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 for epoch in epoch_panel
    ):
        raise ValueError("Joint probe epochs must be a non-empty sequence of non-negative integers")
    if epoch_panel != tuple(sorted(set(epoch_panel))):
        raise ValueError("Joint probe epochs must be unique and canonically ordered")
    if (
        isinstance(unique_indices, bool)
        or not isinstance(unique_indices, int)
        or unique_indices < 1
    ):
        raise ValueError("Joint probe unique_indices must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Joint probe seed must be non-negative")

    validate = getattr(dataset, "validate_required_annotations", None)
    if kind == "visual":
        if not callable(validate):
            raise ValueError(f"Joint visual source {source_name!r} has no annotation validator")
        validate()
    fingerprint = runtime_dataset_fingerprint(dataset)
    if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"Joint source {source_name!r} lacks a lowercase SHA-256 fingerprint")
    dataset_size = len(dataset)
    indices = select_deterministic_probe_indices(
        dataset_size,
        unique_indices,
        seed=seed,
        dataset_fingerprint=fingerprint,
    )
    image_digest = (
        _probe_image_digest(dataset, indices, source_name=source_name)
        if kind == "visual"
        else _canonical_sha256([])
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable joint probe {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_value)
    file_digest = hashlib.sha256()
    row_hashes = []
    maximum_length = 0
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            for epoch in epoch_panel:
                for dataset_index in indices:
                    record = _live_probe_record(
                        dataset,
                        source_name=source_name,
                        kind=kind,
                        dataset_index=dataset_index,
                        epoch=epoch,
                        unbounded_dataset=unbounded_dataset,
                        token_ids=token_ids,
                    )
                    maximum_length = max(maximum_length, int(record["raw_sequence_length"]))
                    row_hashes.append(record["serialized_row_sha256"])
                    raw = _canonical_bytes(record) + b"\n"
                    file_handle.write(raw)
                    file_digest.update(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        if kind == "visual" and (
            _probe_image_digest(dataset, indices, source_name=source_name) != image_digest
        ):
            raise ValueError(f"Joint source {source_name!r} image bytes changed during export")
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "name": source_name,
        "kind": kind,
        "format": "jsonl",
        "path": output_path.name,
        "dataset_fingerprint": fingerprint,
        "dataset_size": dataset_size,
        "sha256": file_digest.hexdigest(),
        "probe_epochs": list(epoch_panel),
        "probe_indices": list(indices),
        "probe_indices_sha256": _canonical_sha256(list(indices)),
        "serialized_row_hashes_sha256": _canonical_sha256(row_hashes),
        "probe_image_content_sha256": image_digest,
        "max_observed_sequence_length": maximum_length,
        "truncated_rows": 0,
    }


def _preprocessing_descriptor(
    projection: JointVisualProjectionManifest,
    native_manifest: NativeTextReplayManifest,
) -> Dict[str, Any]:
    return {
        "visual": projection.source_spec.as_canonical_dict(),
        "native_text_replay_fingerprint": native_manifest.content_fingerprint,
    }


def _probe_policy(seed: int) -> Dict[str, Any]:
    return {
        "format": JOINT_PROBE_FORMAT,
        "version": JOINT_PROBE_VERSION,
        "selection_algorithm": VISION_ALIGNMENT_PROBE_SELECTION_ALGORITHM,
        "seed": seed,
        "visual": {
            "unique_indices": JOINT_VISUAL_PROBE_INDICES,
            "epochs": list(JOINT_VISUAL_PROBE_EPOCHS),
            "rows_per_source": JOINT_VISUAL_PROBE_INDICES * len(JOINT_VISUAL_PROBE_EPOCHS),
        },
        "native_text_replay": {
            "unique_indices": JOINT_NATIVE_PROBE_INDICES,
            "epochs": list(JOINT_NATIVE_PROBE_EPOCHS),
            "rows_per_source": JOINT_NATIVE_PROBE_INDICES * len(JOINT_NATIVE_PROBE_EPOCHS),
        },
        "sequence_length": JOINT_SEQUENCE_LENGTH,
        "truncation_policy": "forbid-raw-length-above-sequence-length-v1",
    }


def build_probe_catalog(
    *,
    projection: JointVisualProjectionManifest,
    native_manifest: NativeTextReplayManifest,
    verification_receipt_path: Path,
    verification_receipt_sha256: str,
    source_entries: Sequence[Mapping[str, Any]],
    exporter_sha256: str,
    probe_seed: int,
) -> Dict[str, Any]:
    """Build the canonical version-1 joint source catalog.

    The schema intentionally has no optional fields.  The native and visual preprocessing
    identities remain separate inside one aggregate descriptor so that a change to either
    invalidates the catalog-level preprocessing digest.
    """
    names = tuple(str(entry.get("name")) for entry in source_entries)
    if names != JOINT_SOURCE_NAMES:
        raise ValueError("Joint source entries must be the exact canonical nine-source ordering")
    if isinstance(probe_seed, bool) or not isinstance(probe_seed, int) or probe_seed < 0:
        raise ValueError("Joint probe seed must be a non-negative integer")
    if _SHA256_RE.fullmatch(exporter_sha256) is None:
        raise ValueError("Joint exporter SHA-256 must be lowercase hexadecimal")
    if _SHA256_RE.fullmatch(verification_receipt_sha256) is None:
        raise ValueError("Native verification receipt SHA-256 must be lowercase hexadecimal")
    for value, name in (
        (projection.raw_sha256, "visual projection raw SHA-256"),
        (projection.content_sha256, "visual projection content SHA-256"),
        (native_manifest.manifest_sha256, "native manifest raw SHA-256"),
        (native_manifest.content_fingerprint, "native manifest content fingerprint"),
    ):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be lowercase hexadecimal")
    for source_name, raw_entry in zip(JOINT_SOURCE_NAMES, source_entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != _SOURCE_ENTRY_FIELDS:
            raise ValueError(f"Joint source entry {source_name!r} has a noncanonical schema")
        expected_kind = "native_text_replay" if source_name == "native_text_replay" else "visual"
        expected_epochs = (
            JOINT_NATIVE_PROBE_EPOCHS
            if expected_kind == "native_text_replay"
            else JOINT_VISUAL_PROBE_EPOCHS
        )
        expected_count = (
            JOINT_NATIVE_PROBE_INDICES
            if expected_kind == "native_text_replay"
            else JOINT_VISUAL_PROBE_INDICES
        )
        fingerprint = raw_entry["dataset_fingerprint"]
        dataset_size = raw_entry["dataset_size"]
        indices = raw_entry["probe_indices"]
        raw_epochs = raw_entry["probe_epochs"]
        if (
            raw_entry["name"] != source_name
            or raw_entry["kind"] != expected_kind
            or raw_entry["format"] != "jsonl"
            or raw_entry["path"] != f"{source_name}.jsonl"
            or not isinstance(fingerprint, str)
            or _SHA256_RE.fullmatch(fingerprint) is None
            or isinstance(dataset_size, bool)
            or not isinstance(dataset_size, int)
            or dataset_size < expected_count
            or not isinstance(indices, list)
            or len(indices) != expected_count
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= dataset_size
                for index in indices
            )
            or len(set(indices)) != len(indices)
            or not isinstance(raw_epochs, list)
            or any(type(epoch) is not int for epoch in raw_epochs)
            or tuple(raw_epochs) != expected_epochs
            or type(raw_entry["truncated_rows"]) is not int
            or raw_entry["truncated_rows"] != 0
        ):
            raise ValueError(f"Joint source entry {source_name!r} identity differs")
        if tuple(indices) != select_deterministic_probe_indices(
            dataset_size,
            expected_count,
            seed=probe_seed,
            dataset_fingerprint=fingerprint,
        ) or raw_entry["probe_indices_sha256"] != _canonical_sha256(indices):
            raise ValueError(f"Joint source entry {source_name!r} probe selection differs")
        for field_name in (
            "sha256",
            "serialized_row_hashes_sha256",
            "probe_image_content_sha256",
        ):
            value = raw_entry[field_name]
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"Joint source entry {source_name!r} {field_name} is invalid")
        maximum = raw_entry["max_observed_sequence_length"]
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < 1
            or maximum > JOINT_SEQUENCE_LENGTH
            or (expected_kind == "native_text_replay" and maximum != JOINT_SEQUENCE_LENGTH)
        ):
            raise ValueError(f"Joint source entry {source_name!r} length evidence differs")
    preprocessing = _preprocessing_descriptor(projection, native_manifest)
    catalog: Dict[str, Any] = {
        "format": SOURCE_CATALOG_FORMAT,
        "version": SOURCE_CATALOG_VERSION,
        "phase": "joint",
        "recipe_version": VISION_ALIGNMENT_RECIPE_VERSION,
        "formatter_version": VISION_ALIGNMENT_FORMATTER_VERSION,
        "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": joint_alignment_runtime_registry_sha256(),
        "source_implementation_inventory": joint_alignment_runtime_implementation_inventory(),
        "exporter_implementation": {
            "path": EXPORTER_IMPLEMENTATION_PATH,
            "sha256": exporter_sha256,
        },
        "visual_projection": {
            "path": str(projection.path),
            "raw_sha256": projection.raw_sha256,
            "content_sha256": projection.content_sha256,
        },
        "native_train_manifest": {
            "path": str(native_manifest.path),
            "raw_sha256": native_manifest.manifest_sha256,
            "content_fingerprint": native_manifest.content_fingerprint,
        },
        "native_verification_receipt": {
            "path": str(verification_receipt_path),
            "sha256": verification_receipt_sha256,
        },
        "preprocessing": preprocessing,
        "preprocessing_sha256": _canonical_sha256(preprocessing),
        "probe": _probe_policy(probe_seed),
        "sources": [dict(entry) for entry in source_entries],
    }
    catalog["content_sha256"] = _canonical_sha256(catalog)
    return catalog


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
            raise FileExistsError(
                f"Refusing to overwrite immutable joint catalog {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _build_unbounded_visual_dataset(
    projection: JointVisualProjectionManifest,
    tokenizer: Any,
    token_ids: Any,
    source_name: str,
) -> Any:
    selection = projection.selection(source_name, "train")
    config = build_vision_alignment_joint_dataset_config(
        projection.source_spec,
        token_ids,
        source_name,
        split=selection.physical_split,
    )
    try:
        unbounded_config = replace(config, max_sequence_length=UNBOUNDED_SEQUENCE_LENGTH)
    except TypeError as error:
        raise ValueError(f"Joint visual config {source_name!r} is not replaceable") from error
    if (
        vision_alignment_joint_adapter_projection_sha256(unbounded_config)
        != selection.adapter_projection_sha256
    ):
        raise ValueError(f"Unbounded joint adapter projection differs for {source_name!r}")
    dataset = unbounded_config.build(tokenizer)
    validate = getattr(dataset, "validate_required_annotations", None)
    if not callable(validate):
        raise ValueError(f"Unbounded joint source {source_name!r} lacks annotation validation")
    validate()
    validate_joint_unbounded_dataset_identity(
        dataset,
        source_name=source_name,
        selection=selection,
        max_sequence_length=UNBOUNDED_SEQUENCE_LENGTH,
    )
    return dataset


def _fresh_native_runtime_evidence(
    native_manifest: NativeTextReplayManifest,
    receipt: NativeTextReplayVerificationReceipt,
    *,
    expected_size: int,
    tokenizer: Optional[Any] = None,
) -> NativeTextReplayDataset:
    """Rebuild native replay so closing validation repeats every source-file stat check."""
    dataset = NativeTextReplayDataset(
        native_manifest.path,
        expected_fingerprint=native_manifest.content_fingerprint,
        verification_receipt_path=receipt.path,
        expected_verification_receipt_sha256=receipt.receipt_sha256,
        validate_source_files=True,
    )
    if tokenizer is not None:
        dataset.validate_tokenizer(tokenizer)
    if (
        len(dataset) != expected_size
        or len(dataset) != native_manifest.num_windows
        or dataset.sequence_length != JOINT_SEQUENCE_LENGTH
        or runtime_dataset_fingerprint(dataset) != native_manifest.content_fingerprint
        or sum(dataset.source_counts.values()) != expected_size
    ):
        raise ValueError("Fresh native replay runtime size or identity evidence differs")
    return dataset


def _closing_validate_inputs(
    *,
    projection: JointVisualProjectionManifest,
    native_manifest: NativeTextReplayManifest,
    receipt: NativeTextReplayVerificationReceipt,
    expected_native_size: int,
    tokenizer: Any,
    token_ids: Molmo2TokenIds,
) -> None:
    """Re-load every external identity and fresh-stat native replay before publication."""
    closing_native = _load_native_manifest_pinned(
        native_manifest.path,
        expected_raw_sha256=native_manifest.manifest_sha256,
        expected_content_fingerprint=native_manifest.content_fingerprint,
    )
    closing_receipt = NativeTextReplayVerificationReceipt.load(
        receipt.path, expected_sha256=receipt.receipt_sha256
    )
    closing_receipt.validate_manifest(closing_native)
    if (
        closing_native.provenance.get("verification_receipt_sha256")
        != closing_receipt.receipt_sha256
    ):
        raise ValueError("Closing native manifest does not bind its verification receipt")
    _fresh_native_runtime_evidence(
        closing_native,
        closing_receipt,
        expected_size=expected_native_size,
        tokenizer=tokenizer,
    )
    # Re-open the receipt after the potentially long source-stat pass so its reviewed builder
    # identity is the last native-replay dependency observed before publication.
    closing_receipt = NativeTextReplayVerificationReceipt.load(
        receipt.path, expected_sha256=receipt.receipt_sha256
    )
    closing_receipt.validate_manifest(closing_native)
    closing_projection = load_joint_visual_projection_manifest(
        projection.path,
        expected_token_ids=token_ids,
        expected_sha256=projection.raw_sha256,
    )
    if (
        closing_projection.raw_sha256 != projection.raw_sha256
        or closing_projection.content_sha256 != projection.content_sha256
        or _canonical_bytes(closing_projection.source_spec.as_canonical_dict())
        != _canonical_bytes(projection.source_spec.as_canonical_dict())
    ):
        raise ValueError("Joint visual projection identity changed during closing validation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-projection-manifest", required=True)
    parser.add_argument("--expected-visual-projection-sha256", required=True)
    parser.add_argument("--native-train-manifest", required=True)
    parser.add_argument("--expected-native-train-manifest-sha256", required=True)
    parser.add_argument("--expected-native-content-fingerprint", required=True)
    parser.add_argument("--native-verification-receipt", required=True)
    parser.add_argument("--expected-native-verification-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=JOINT_PROBE_SEED)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Export the immutable nine-source joint catalog and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.seed != JOINT_PROBE_SEED:
            raise ValueError(f"Production joint probes require seed {JOINT_PROBE_SEED}")
        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=VISION_ALIGNMENT_TOKENIZER_ID,
            revision=VISION_ALIGNMENT_TOKENIZER_REVISION,
            expected_fingerprint=VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
            cache_dir=args.hf_cache_dir,
        )
        projection = load_joint_visual_projection_manifest(
            args.visual_projection_manifest,
            expected_token_ids=token_ids,
            expected_sha256=args.expected_visual_projection_sha256,
        )
        native_path = Path(args.native_train_manifest).expanduser().resolve()
        native_manifest = _load_native_manifest_pinned(
            native_path,
            expected_raw_sha256=args.expected_native_train_manifest_sha256,
            expected_content_fingerprint=args.expected_native_content_fingerprint,
        )
        if (
            native_manifest.sequence_length != JOINT_SEQUENCE_LENGTH
            or native_manifest.provenance.get("split") != "train"
        ):
            raise ValueError("Joint native replay must be the exact 8,192-token train manifest")
        receipt_path = Path(args.native_verification_receipt).expanduser().resolve()
        receipt = NativeTextReplayVerificationReceipt.load(
            receipt_path,
            expected_sha256=args.expected_native_verification_receipt_sha256,
        )
        receipt.validate_manifest(native_manifest)
        if native_manifest.provenance.get("verification_receipt_sha256") != receipt.receipt_sha256:
            raise ValueError(
                "Native train manifest does not bind the supplied verification receipt"
            )

        if (
            projection.source_spec.perception_spec.tokenizer_id != VISION_ALIGNMENT_TOKENIZER_ID
            or projection.source_spec.perception_spec.tokenizer_revision
            != VISION_ALIGNMENT_TOKENIZER_REVISION
            or projection.source_spec.perception_spec.tokenizer_fingerprint
            != VISION_ALIGNMENT_TOKENIZER_FINGERPRINT
        ):
            raise ValueError("Joint visual projection does not bind the pinned runtime tokenizer")

        exporter_path = Path(__file__).resolve()
        exporter_sha256 = _sha256_file(exporter_path)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite joint artifact {output_dir}")
        lock_path = output_dir.with_name(f".{output_dir.name}.lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(lock_fd)
            raise RuntimeError(f"Another joint exporter holds {lock_path}") from error

        staging_dir: Optional[Path] = None
        entries = []
        try:
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_dir.name}.", suffix=".building", dir=output_dir.parent
                )
            )
            native_dataset: Optional[NativeTextReplayDataset] = None
            for source_name in JOINT_SOURCE_NAMES:
                if source_name == "native_text_replay":
                    native_dataset = NativeTextReplayDataset(
                        native_path,
                        expected_fingerprint=native_manifest.content_fingerprint,
                        verification_receipt_path=receipt_path,
                        expected_verification_receipt_sha256=receipt.receipt_sha256,
                    )
                    native_dataset.validate_tokenizer(tokenizer)
                    entry = export_source_probe(
                        native_dataset,
                        source_name=source_name,
                        kind="native_text_replay",
                        output_path=staging_dir / f"{source_name}.jsonl",
                        unique_indices=JOINT_NATIVE_PROBE_INDICES,
                        epochs=JOINT_NATIVE_PROBE_EPOCHS,
                        seed=args.seed,
                        token_ids=token_ids,
                    )
                else:
                    selected = build_selected_joint_dataset(
                        projection,
                        tokenizer,
                        token_ids,
                        source_name,
                        logical_split="train",
                        validate_required_annotations=False,
                    )
                    unbounded = _build_unbounded_visual_dataset(
                        projection, tokenizer, token_ids, source_name
                    )
                    entry = export_source_probe(
                        selected,
                        source_name=source_name,
                        kind="visual",
                        output_path=staging_dir / f"{source_name}.jsonl",
                        unique_indices=JOINT_VISUAL_PROBE_INDICES,
                        epochs=JOINT_VISUAL_PROBE_EPOCHS,
                        seed=args.seed,
                        unbounded_dataset=unbounded,
                        token_ids=token_ids,
                    )
                entries.append(entry)
            catalog = build_probe_catalog(
                projection=projection,
                native_manifest=native_manifest,
                verification_receipt_path=receipt_path,
                verification_receipt_sha256=receipt.receipt_sha256,
                source_entries=entries,
                exporter_sha256=exporter_sha256,
                probe_seed=args.seed,
            )
            catalog_path = staging_dir / CATALOG_NAME
            _write_once(catalog_path, catalog)
            native_entry = next(entry for entry in entries if entry["name"] == "native_text_replay")
            _closing_validate_inputs(
                projection=projection,
                native_manifest=native_manifest,
                receipt=receipt,
                expected_native_size=native_entry["dataset_size"],
                tokenizer=tokenizer,
                token_ids=token_ids,
            )
            if (
                _sha256_file(exporter_path) != exporter_sha256
                or _sha256_file(projection.path) != projection.raw_sha256
                or _sha256_file(native_path) != native_manifest.manifest_sha256
                or _sha256_file(receipt_path) != receipt.receipt_sha256
                or joint_alignment_runtime_registry_sha256() != catalog["source_registry_sha256"]
                or joint_alignment_runtime_implementation_inventory()
                != catalog["source_implementation_inventory"]
            ):
                raise ValueError("Joint source or implementation identity changed during export")
            _fsync_tree(staging_dir)
            _publish_no_replace(staging_dir, output_dir)
            _fsync_directory(output_dir.parent)
            catalog_path = output_dir / CATALOG_NAME
        except BaseException:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    except (FileExistsError, OLMoConfigurationError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "catalog": str(catalog_path),
                "content_sha256": catalog["content_sha256"],
                "sources": {
                    entry["name"]: len(entry["probe_indices"]) * len(entry["probe_epochs"])
                    for entry in entries
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
