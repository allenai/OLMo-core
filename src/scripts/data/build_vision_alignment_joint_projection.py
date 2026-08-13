#!/usr/bin/env python
"""Build the immutable 8,192-token visual projection for joint alignment.

The joint phase does not select or rank visual examples. This builder loads one externally
SHA-pinned, complete perception provenance artifact and instantiates the same eight adapters
with only ``max_sequence_length`` changed. Every parent logical index and image-content
identity is then replayed through the joint adapter before a compact projection manifest is
published atomically. Existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.data.multimodal.vision_alignment_joint_provenance import (
    JOINT_VISUAL_PROJECTION_FORMAT,
    JOINT_VISUAL_PROJECTION_MANIFEST,
    JOINT_VISUAL_PROJECTION_VERSION,
    joint_selected_dataset_fingerprint,
    load_joint_visual_projection_manifest,
)
from olmo_core.data.multimodal.vision_alignment_joint_sources import (
    JOINT_TO_PERCEPTION_SOURCE,
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
    VisionAlignmentJointSourceSpec,
    build_vision_alignment_joint_dataset,
    build_vision_alignment_joint_dataset_config,
    vision_alignment_joint_adapter_projection_sha256,
    vision_alignment_joint_implementation_inventory,
    vision_alignment_joint_source_registry_sha256,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_MANIFEST,
    PerceptionProvenanceManifest,
    image_reference_sha256,
    load_perception_provenance_manifest,
    perception_annotation_content_sha256,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
    runtime_dataset_fingerprint,
)

BUILDER_NAME = "build_vision_alignment_joint_projection"
BUILDER_VERSION = 1
PROJECTION_ALGORITHM = "exact-parent-logical-row-selection-v1"
PARENT_SEQUENCE_LENGTH = 2560
JOINT_SEQUENCE_LENGTH = 8192
DEFAULT_HF_CACHE_DIR = "/weka/oe-training-default/rustin/hf-cache/hub"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATASET_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16,64}")

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
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON repeats key {key!r}")
        value[key] = item
    return value


def _lower_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _dataset_fingerprint(value: Any, *, name: str) -> str:
    if type(value) is not str or _DATASET_FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 16- to 64-hex fingerprint")
    return value


def _normalized_output_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value):
        raise ValueError("output_dir must be a non-empty path")
    unresolved = Path(value).expanduser()
    if os.path.lexists(unresolved):
        raise FileExistsError(f"Refusing to overwrite immutable artifact {unresolved}")
    path = unresolved.resolve()
    if path == path.parent or not path.name:
        raise ValueError("output_dir must identify a new artifact directory")
    return path


def _resolve_created_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if type(value) is not str or not value:
        raise ValueError("created_at must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return value


def _read_parent_root(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid parent perception provenance {path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Parent perception provenance raw SHA-256 differs")
    if not isinstance(value, Mapping):
        raise ValueError("Parent perception provenance must be a JSON object")
    return raw, value


def _load_parent(
    path: str | Path,
    expected_sha256: str,
) -> tuple[PerceptionProvenanceManifest, bytes, Mapping[str, Any]]:
    expected_sha256 = _lower_sha256(
        expected_sha256,
        name="expected parent perception provenance SHA-256",
    )
    if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path):
        raise ValueError("parent perception provenance must be a non-empty path")
    parent_path = Path(path).expanduser().resolve()
    if parent_path.name != PERCEPTION_PROVENANCE_MANIFEST:
        raise ValueError(
            "Parent perception provenance must use canonical manifest name "
            f"{PERCEPTION_PROVENANCE_MANIFEST!r}"
        )
    parent = load_perception_provenance_manifest(
        parent_path,
        expected_sha256=expected_sha256,
        verify_finevision_materialization=True,
        load_image_path_signatures=True,
        require_complete=True,
    )
    raw, root = _read_parent_root(parent_path, expected_sha256=expected_sha256)
    if parent.path != parent_path or parent.raw_sha256 != expected_sha256:
        raise ValueError("Loaded parent perception provenance identity differs")
    return parent, raw, root


def _parent_components(root: Mapping[str, Any], parent_source_name: str) -> list[str]:
    sources = root.get("sources")
    source = sources.get(parent_source_name) if isinstance(sources, Mapping) else None
    components = source.get("components") if isinstance(source, Mapping) else None
    if (
        not isinstance(components, list)
        or not components
        or any(type(component) is not str or not component for component in components)
        or len(set(components)) != len(components)
    ):
        raise ValueError(f"Parent perception components differ for source {parent_source_name!r}")
    return components


@dataclass(frozen=True)
class _RawDatasetIdentity:
    examples: int
    fingerprint: str
    annotation_sha256: str
    adapter_projection_sha256: str


def _raw_dataset_identity(
    dataset: Any,
    *,
    source_name: str,
    physical_split: str,
    expected_adapter_sha256: str,
) -> _RawDatasetIdentity:
    try:
        examples = len(dataset)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Joint adapter {source_name}/{physical_split} has no stable length"
        ) from error
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 1:
        raise ValueError(f"Joint adapter {source_name}/{physical_split} is empty")
    fingerprint = _dataset_fingerprint(
        runtime_dataset_fingerprint(dataset),
        name=f"{source_name}/{physical_split} base dataset fingerprint",
    )
    annotation_sha = perception_annotation_content_sha256(dataset)
    config = getattr(dataset, "config", None)
    if config is None or getattr(config, "max_sequence_length", None) != JOINT_SEQUENCE_LENGTH:
        raise ValueError(f"Joint adapter {source_name}/{physical_split} has the wrong config")
    try:
        adapter_sha = vision_alignment_joint_adapter_projection_sha256(config)
    except ValueError as error:
        raise ValueError(
            f"Joint adapter {source_name}/{physical_split} config cannot be projected"
        ) from error
    if adapter_sha != expected_adapter_sha256:
        raise ValueError(
            f"Joint adapter {source_name}/{physical_split} projection differs from its parent"
        )
    return _RawDatasetIdentity(
        examples=examples,
        fingerprint=fingerprint,
        annotation_sha256=annotation_sha,
        adapter_projection_sha256=adapter_sha,
    )


def _validate_annotations(dataset: Any, *, source_name: str, physical_split: str) -> None:
    validate = getattr(dataset, "validate_required_annotations", None)
    if not callable(validate):
        raise ValueError(
            f"Joint adapter {source_name}/{physical_split} lacks annotation validation"
        )
    validate()


def _validate_selected_images(
    dataset: Any,
    *,
    source_name: str,
    logical_split: str,
    indices: Sequence[int],
    expected_row_sha256: Sequence[str],
) -> None:
    if len(indices) != len(expected_row_sha256):
        raise ValueError(f"Parent image inventory count differs for {source_name}/{logical_split}")
    raw_images = getattr(dataset, "raw_image_references", None)
    if not callable(raw_images):
        raise ValueError(f"Joint adapter {source_name!r} lacks raw image access")
    for ordinal, (index, expected_sha) in enumerate(zip(indices, expected_row_sha256)):
        try:
            references = tuple(raw_images(index))
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                f"Joint adapter row differs for {source_name}/{logical_split} at {ordinal}"
            ) from error
        if len(references) != 1:
            raise ValueError(
                f"Joint adapter {source_name}/{logical_split} row {ordinal} has "
                f"{len(references)} images; expected exactly one"
            )
        if image_reference_sha256(references[0]) != expected_sha:
            raise ValueError(
                f"Joint image bytes differ from parent for {source_name}/{logical_split} "
                f"at row {ordinal}"
            )


def _atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as file_handle:
            file_handle.write(raw)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
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


def _publish_staging(staging: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}")
    _rename_directory_no_replace(staging, destination)
    _fsync_directory(destination.parent)


def _remove_owned_staging(staging: Path, destination: Path) -> None:
    if (
        staging.parent == destination.parent
        and staging.name.startswith(f".{destination.name}.building-")
        and os.path.lexists(staging)
    ):
        shutil.rmtree(staging)


def build_vision_alignment_joint_projection(
    *,
    parent_perception_provenance: str | Path,
    expected_parent_perception_sha256: str,
    output_dir: str | Path,
    tokenizer: Any,
    token_ids: Any,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one exact joint visual projection.

    :param parent_perception_provenance: Complete reviewed perception provenance manifest.
    :param expected_parent_perception_sha256: External raw SHA-256 pin for that manifest.
    :param output_dir: New immutable output directory.
    :param tokenizer: Pinned prepared tokenizer used by the parent source specification.
    :param token_ids: Prepared model image-token identities.
    :param created_at: Optional timezone-aware ISO timestamp for reproducible builds.
    :returns: The exact projection manifest mapping written to disk.
    :raises ValueError: If parent, adapter, row, image, or implementation identity differs.
    :raises FileExistsError: If the immutable output already exists.
    """
    destination = _normalized_output_path(output_dir)
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}")
    expected_parent_sha = _lower_sha256(
        expected_parent_perception_sha256,
        name="expected parent perception provenance SHA-256",
    )
    created_at_value = _resolve_created_at(created_at)
    parent, parent_raw, parent_root = _load_parent(
        parent_perception_provenance,
        expected_parent_sha,
    )
    parent.validate_image_path_signatures()
    script_path = Path(__file__).resolve()
    builder_sha = _sha256_file(script_path)
    joint_spec = VisionAlignmentJointSourceSpec.from_perception(parent.source_spec)
    joint_spec_sha = joint_spec.preprocessing_sha256
    registry_sha = vision_alignment_joint_source_registry_sha256()
    implementation_inventory = vision_alignment_joint_implementation_inventory()

    parent_registry_sha = _lower_sha256(
        parent_root.get("source_registry_sha256"),
        name="parent perception source_registry_sha256",
    )
    datasets: dict[tuple[str, str], tuple[Any, _RawDatasetIdentity]] = {}
    sources: dict[str, Any] = {}
    for source_name in JOINT_VISUAL_SOURCE_NAMES:
        parent_source_name = JOINT_TO_PERCEPTION_SOURCE[source_name]
        source: dict[str, Any] = {
            "parent_source_name": parent_source_name,
            "components": _parent_components(parent_root, parent_source_name),
        }
        for logical_split in ("train", "validation"):
            parent_selection = parent.selection(parent_source_name, logical_split)
            physical_split = parent_selection.physical_split
            cache_key = (source_name, physical_split)
            cached = datasets.get(cache_key)
            if cached is None:
                expected_config = build_vision_alignment_joint_dataset_config(
                    joint_spec,
                    token_ids,
                    source_name,
                    split=physical_split,
                )
                expected_adapter_sha = vision_alignment_joint_adapter_projection_sha256(
                    expected_config
                )
                dataset = build_vision_alignment_joint_dataset(
                    joint_spec,
                    tokenizer,
                    token_ids,
                    source_name,
                    split=physical_split,
                    validate_required_annotations=False,
                )
                _validate_annotations(
                    dataset,
                    source_name=source_name,
                    physical_split=physical_split,
                )
                identity = _raw_dataset_identity(
                    dataset,
                    source_name=source_name,
                    physical_split=physical_split,
                    expected_adapter_sha256=expected_adapter_sha,
                )
                cached = (dataset, identity)
                datasets[cache_key] = cached
            dataset, identity = cached
            if identity.examples != parent_selection.base_examples:
                raise ValueError(
                    f"Joint base example count differs from parent for "
                    f"{source_name}/{logical_split}"
                )
            if identity.annotation_sha256 != parent_selection.base_annotation_sha256:
                raise ValueError(
                    f"Joint base annotation identity differs from parent for "
                    f"{source_name}/{logical_split}"
                )
            _validate_selected_images(
                dataset,
                source_name=source_name,
                logical_split=logical_split,
                indices=parent_selection.indices,
                expected_row_sha256=parent_selection.row_image_content_sha256,
            )
            source[logical_split] = {
                "physical_split": physical_split,
                "base_examples": identity.examples,
                "joint_base_dataset_fingerprint": identity.fingerprint,
                "joint_base_annotation_sha256": identity.annotation_sha256,
                "adapter_projection_sha256": identity.adapter_projection_sha256,
                "selection_indices_sha256": parent_selection.selection_indices_sha256,
                "runtime_examples": len(parent_selection.indices),
                "row_image_content_sha256": _canonical_sha256(
                    list(parent_selection.row_image_content_sha256)
                ),
                "unique_image_content_sha256": _canonical_sha256(
                    list(parent_selection.unique_image_content_sha256)
                ),
                "runtime_dataset_fingerprint": joint_selected_dataset_fingerprint(
                    source_name=source_name,
                    parent_source_name=parent_source_name,
                    logical_split=logical_split,
                    physical_split=physical_split,
                    joint_base_fingerprint=identity.fingerprint,
                    selection_indices_sha256=parent_selection.selection_indices_sha256,
                    joint_source_spec_sha256=joint_spec_sha,
                    parent_provenance_sha256=parent.raw_sha256,
                    parent_provenance_content_sha256=parent.content_sha256,
                ),
            }
        sources[source_name] = source

    # Close each raw-adapter snapshot after all selected rows have been rehashed.
    for (source_name, physical_split), (dataset, original_identity) in datasets.items():
        _validate_annotations(
            dataset,
            source_name=source_name,
            physical_split=physical_split,
        )
        final_identity = _raw_dataset_identity(
            dataset,
            source_name=source_name,
            physical_split=physical_split,
            expected_adapter_sha256=original_identity.adapter_projection_sha256,
        )
        if final_identity != original_identity:
            raise ValueError(
                f"Joint base dataset identity changed during projection for "
                f"{source_name}/{physical_split}"
            )

    train_union = tuple(
        sorted(
            {
                digest
                for source_name in JOINT_VISUAL_SOURCE_NAMES
                for digest in parent.selection(
                    JOINT_TO_PERCEPTION_SOURCE[source_name], "train"
                ).unique_image_content_sha256
            }
        )
    )
    validation_union = tuple(
        sorted(
            {
                digest
                for source_name in JOINT_VISUAL_SOURCE_NAMES
                for digest in parent.selection(
                    JOINT_TO_PERCEPTION_SOURCE[source_name], "validation"
                ).unique_image_content_sha256
            }
        )
    )
    overlap = set(train_union).intersection(validation_union)
    if not train_union or not validation_union or overlap:
        raise ValueError("Joint train/validation image unions are not non-empty and disjoint")

    # Revalidate the parent receipt, materialization, path signatures, and exact raw bytes at
    # the end of all adapter and image reads.
    parent.validate_image_path_signatures()
    final_parent, final_parent_raw, final_parent_root = _load_parent(
        parent.path,
        expected_parent_sha,
    )
    if (
        final_parent_raw != parent_raw
        or _canonical_bytes(final_parent_root) != _canonical_bytes(parent_root)
        or final_parent.content_sha256 != parent.content_sha256
        or final_parent.source_spec_sha256 != parent.source_spec_sha256
        or final_parent.selections != parent.selections
    ):
        raise ValueError("Parent perception provenance changed during joint projection")
    final_parent.validate_image_path_signatures()
    if _sha256_file(script_path) != builder_sha:
        raise ValueError("Joint projection builder changed during artifact construction")

    manifest: dict[str, Any] = {
        "format": JOINT_VISUAL_PROJECTION_FORMAT,
        "version": JOINT_VISUAL_PROJECTION_VERSION,
        "status": "verified",
        "phase": "joint",
        "created_at": created_at_value,
        "builder": {
            "name": BUILDER_NAME,
            "version": BUILDER_VERSION,
            "script_sha256": builder_sha,
        },
        "parent_perception_provenance": {
            "path": str(parent.path.resolve()),
            "sha256": parent.raw_sha256,
            "content_sha256": parent.content_sha256,
            "source_spec_sha256": parent.source_spec_sha256,
            "source_registry_sha256": parent_registry_sha,
        },
        "source_name_projection": dict(JOINT_TO_PERCEPTION_SOURCE),
        "source_spec": joint_spec.as_canonical_dict(),
        "source_spec_sha256": joint_spec_sha,
        "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
        "source_registry_sha256": registry_sha,
        "source_implementation_inventory": implementation_inventory,
        "projection_policy": {
            "algorithm": PROJECTION_ALGORITHM,
            "parent_sequence_length": PARENT_SEQUENCE_LENGTH,
            "sequence_length": JOINT_SEQUENCE_LENGTH,
            "allowed_adapter_config_delta": ["max_sequence_length"],
        },
        "sources": sources,
        "unions": {
            "train_unique_image_content_sha256": _canonical_sha256(train_union),
            "train_count": len(train_union),
            "validation_unique_image_content_sha256": _canonical_sha256(validation_union),
            "validation_count": len(validation_union),
            "overlap_count": 0,
        },
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    ).resolve()
    try:
        manifest_path = staging / JOINT_VISUAL_PROJECTION_MANIFEST
        manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write(manifest_path, manifest_raw)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        validated = load_joint_visual_projection_manifest(
            manifest_path,
            expected_sha256=manifest_sha,
            verify_finevision_materialization=True,
            load_image_path_signatures=True,
            require_complete=False,
        )
        if (
            validated.content_sha256 != manifest["content_sha256"]
            or validated.parent_provenance.raw_sha256 != expected_parent_sha
        ):
            raise ValueError("Runtime loader returned a different joint projection identity")
        if _sha256_file(script_path) != builder_sha:
            raise ValueError("Joint projection builder changed before publication")
        if parent.path.read_bytes() != parent_raw:
            raise ValueError("Parent perception provenance changed before publication")
        parent.validate_image_path_signatures()
        _atomic_write(staging / "COMPLETE", f"{manifest_sha}\n".encode("ascii"))
        _fsync_directory(staging)
        load_joint_visual_projection_manifest(
            manifest_path,
            expected_sha256=manifest_sha,
            verify_finevision_materialization=True,
            load_image_path_signatures=True,
            require_complete=True,
        )
        _publish_staging(staging, destination)
        return manifest
    finally:
        _remove_owned_staging(staging, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-perception-provenance", required=True)
    parser.add_argument("--expected-parent-perception-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--created-at")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the joint visual projection builder CLI and return a process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parent, _, _ = _load_parent(
        args.parent_perception_provenance,
        args.expected_parent_perception_sha256,
    )
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=parent.source_spec.tokenizer_id,
        revision=parent.source_spec.tokenizer_revision,
        expected_fingerprint=parent.source_spec.tokenizer_fingerprint,
        cache_dir=args.hf_cache_dir,
    )
    build_vision_alignment_joint_projection(
        parent_perception_provenance=args.parent_perception_provenance,
        expected_parent_perception_sha256=args.expected_parent_perception_sha256,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        token_ids=token_ids,
        created_at=args.created_at,
    )
    manifest_path = Path(args.output_dir).expanduser().resolve() / JOINT_VISUAL_PROJECTION_MANIFEST
    log.info("Published joint visual projection at %s", manifest_path)
    log.info("Manifest SHA-256: %s", _sha256_file(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
