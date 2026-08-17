"""Strict runtime binding for the Vision Alignment joint visual projection.

The joint phase does not choose a new visual population.  Its projection artifact binds an
8,192-token instantiation of each adapter to the exact logical rows and image-byte inventories
selected by the parent perception provenance. Loading validates both data artifacts and every
mechanically derived identity before a dataset can be constructed.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from olmo_core.nn.vision import Molmo2TokenIds

from .vision_alignment_joint_sources import (
    JOINT_SEQUENCE_SENSITIVE_ANNOTATION_SOURCES,
    JOINT_TO_PERCEPTION_SOURCE,
    JOINT_VISUAL_SOURCE_NAMES,
    VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
    VisionAlignmentJointSourceSpec,
    build_vision_alignment_joint_dataset,
    build_vision_alignment_joint_dataset_config,
    vision_alignment_joint_adapter_projection_sha256,
    vision_alignment_joint_annotation_replay_sha256,
)
from .vision_alignment_perception_provenance import (
    PERCEPTION_PROVENANCE_MANIFEST,
    VALIDATION_IMAGE_CONTENTS_PER_SOURCE,
    PerceptionProvenanceManifest,
    image_reference_sha256,
    load_perception_provenance_manifest,
    perception_annotation_content_sha256,
)
from .vision_alignment_sources import runtime_dataset_fingerprint

__all__ = [
    "JOINT_VISUAL_PROJECTION_FORMAT",
    "JOINT_VISUAL_PROJECTION_ALGORITHM",
    "JOINT_VISUAL_PROJECTION_MANIFEST",
    "JOINT_VISUAL_PROJECTION_VERSION",
    "JointVisualProjectionManifest",
    "JointVisualSplitProjection",
    "SelectedVisionAlignmentJointDataset",
    "build_selected_joint_dataset",
    "joint_selected_dataset_fingerprint",
    "load_joint_visual_projection_manifest",
]

JOINT_VISUAL_PROJECTION_FORMAT = "vision_alignment_joint_visual_projection"
JOINT_VISUAL_PROJECTION_MANIFEST = "vision-alignment-joint-visual-projection.json"
JOINT_VISUAL_PROJECTION_VERSION = 1
JOINT_MODEL_VOCAB_SIZE = 100352

JOINT_VISUAL_PROJECTION_ALGORITHM = (
    "exact-parent-logical-row-selection-with-source-aware-annotation-projection-v1"
)
_PARENT_SEQUENCE_LENGTH = 2560
_JOINT_SEQUENCE_LENGTH = 8192
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATASET_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16,64}")

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
_INLINE_SERIALIZED_FIELDS = frozenset(
    {"input_ids", "labels", "loss_masks", "position_ids", "token_type_ids"}
)
_ARRAY_DESCRIPTOR_FIELDS = frozenset({"dtype", "shape", "sha256"})
_INT64_SERIALIZED_FIELDS = frozenset(
    {"input_ids", "labels", "position_ids", "token_type_ids", "subsegment_ids"}
)
_FLOAT32_SERIALIZED_FIELDS = frozenset({"loss_masks", "images"})
_PROBE_RECORD_BASE_FIELDS = frozenset(
    {
        "source",
        "probe_index",
        "probe_epoch",
        "serialized_fields",
        "serialized_row_sha256",
        "image_crops",
        "pooled_tokens",
        "raw_sequence_length",
        "truncated",
    }
)

_ROOT_FIELDS = frozenset(
    {
        "format",
        "version",
        "status",
        "phase",
        "created_at",
        "builder",
        "parent_perception_provenance",
        "token_ids",
        "source_name_projection",
        "source_spec",
        "source_spec_sha256",
        "source_registry_version",
        "source_registry_sha256",
        "source_implementation_inventory",
        "projection_policy",
        "sources",
        "unions",
        "content_sha256",
    }
)
_BUILDER_FIELDS = frozenset({"name", "version", "script_sha256"})
_TOKEN_ID_FIELDS = frozenset(
    {
        "im_start_id",
        "im_end_id",
        "im_patch_id",
        "im_col_id",
        "low_res_im_start_id",
        "image_placeholder_id",
        "im_end_turn_id",
    }
)
_PARENT_FIELDS = frozenset(
    {"path", "sha256", "content_sha256", "source_spec_sha256", "source_registry_sha256"}
)
_POLICY_FIELDS = frozenset(
    {"algorithm", "parent_sequence_length", "sequence_length", "allowed_adapter_config_delta"}
)
_SOURCE_FIELDS = frozenset({"parent_source_name", "components", "train", "validation"})
_SPLIT_FIELDS = frozenset(
    {
        "physical_split",
        "base_examples",
        "joint_base_dataset_fingerprint",
        "joint_base_annotation_sha256",
        "adapter_projection_sha256",
        "selection_indices_sha256",
        "runtime_examples",
        "row_image_content_sha256",
        "unique_image_content_sha256",
        "runtime_dataset_fingerprint",
    }
)
_UNION_FIELDS = frozenset(
    {
        "train_unique_image_content_sha256",
        "train_count",
        "validation_unique_image_content_sha256",
        "validation_count",
        "overlap_count",
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


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON repeats key {key!r}")
        value[key] = item
    return value


def _exact_mapping(value: Any, expected: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _fingerprint(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DATASET_FINGERPRINT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 16- to 64-hex fingerprint")
    return value


def _count(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _projection_token_ids(value: Any) -> Molmo2TokenIds:
    token_ids = _exact_mapping(value, _TOKEN_ID_FIELDS, name="token_ids")
    parsed = {
        field_name: _count(token_ids[field_name], name=f"token_ids.{field_name}")
        for field_name in sorted(_TOKEN_ID_FIELDS)
    }
    if len(set(parsed.values())) != len(parsed) or any(
        token_id >= JOINT_MODEL_VOCAB_SIZE for token_id in parsed.values()
    ):
        raise ValueError("Joint projection token IDs must be distinct model-vocabulary rows")
    return Molmo2TokenIds(**parsed)


def _parent_manifest_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("parent_perception_provenance.path must be a non-empty path")
    unresolved = Path(value).expanduser()
    if ".." in unresolved.parts:
        raise ValueError("parent_perception_provenance.path contains path traversal")
    if unresolved.is_absolute():
        path = unresolved.resolve()
        if str(path) != value:
            raise ValueError("parent_perception_provenance.path must be normalized")
    else:
        path = (root / unresolved).resolve()
        if root != path and root not in path.parents:
            raise ValueError("parent_perception_provenance.path escapes the projection root")
    if path.name != PERCEPTION_PROVENANCE_MANIFEST or not path.is_file():
        raise ValueError("parent_perception_provenance.path is not a canonical manifest")
    return path


def _parent_components(parent_root: Mapping[str, Any], parent_source_name: str) -> list[str]:
    sources = parent_root.get("sources")
    if not isinstance(sources, Mapping) or parent_source_name not in sources:
        raise ValueError("Parent perception provenance source inventory differs")
    source = sources[parent_source_name]
    if not isinstance(source, Mapping):
        raise ValueError("Parent perception provenance source entry differs")
    components = source.get("components")
    if not isinstance(components, list) or any(
        not isinstance(component, str) or not component for component in components
    ):
        raise ValueError("Parent perception provenance components differ")
    return components


def _inventory_sha256(values: Sequence[str]) -> str:
    return _canonical_sha256(list(values))


def joint_selected_dataset_fingerprint(
    *,
    source_name: str,
    parent_source_name: str,
    logical_split: str,
    physical_split: str,
    joint_base_fingerprint: str,
    selection_indices_sha256: str,
    joint_source_spec_sha256: str,
    parent_provenance_sha256: str,
    parent_provenance_content_sha256: str,
) -> str:
    """Return the selected joint dataset identity derived from its exact parent rows."""
    return _canonical_sha256(
        {
            "version": "vision-alignment-joint-selected-v1",
            "source_name": source_name,
            "parent_source_name": parent_source_name,
            "logical_split": logical_split,
            "physical_split": physical_split,
            "joint_base_fingerprint": joint_base_fingerprint,
            "selection_indices_sha256": selection_indices_sha256,
            "joint_source_spec_sha256": joint_source_spec_sha256,
            "parent_perception_provenance_sha256": parent_provenance_sha256,
            "parent_perception_provenance_content_sha256": (parent_provenance_content_sha256),
        }
    )


@dataclass(frozen=True)
class JointVisualSplitProjection:
    """Validated joint adapter identity and exact parent selection for one split."""

    physical_split: str
    base_examples: int
    joint_base_dataset_fingerprint: str
    joint_base_annotation_sha256: str
    adapter_projection_sha256: str
    indices: Tuple[int, ...]
    selection_indices_sha256: str
    runtime_dataset_fingerprint: str
    row_image_content_sha256: Tuple[str, ...]
    unique_image_content_sha256: Tuple[str, ...]


@dataclass(frozen=True)
class JointVisualProjectionManifest:
    """Fully validated joint visual projection and its parent perception provenance."""

    path: Path
    raw_sha256: str
    content_sha256: str
    parent_provenance: PerceptionProvenanceManifest
    token_ids: Molmo2TokenIds
    source_spec: VisionAlignmentJointSourceSpec
    source_spec_sha256: str
    selections: Mapping[tuple[str, str], JointVisualSplitProjection]

    def selection(self, source_name: str, logical_split: str) -> JointVisualSplitProjection:
        """Return one canonical source/split projection or reject an unknown key."""
        if source_name not in JOINT_TO_PERCEPTION_SOURCE:
            raise ValueError(f"Unknown joint visual source {source_name!r}")
        if logical_split not in ("train", "validation"):
            raise ValueError(f"Unknown joint logical split {logical_split!r}")
        return self.selections[(source_name, logical_split)]


def load_joint_visual_projection_manifest(
    path: str | Path,
    *,
    expected_token_ids: Molmo2TokenIds,
    expected_sha256: Optional[str] = None,
    verify_finevision_materialization: bool = True,
    load_image_path_signatures: bool = True,
) -> JointVisualProjectionManifest:
    """Load a joint projection and prove it is an exact 8,192-token parent projection.

    :param path: Canonically named projection manifest.
    :param expected_token_ids: Exact tokenizer-adapted IDs required by the caller.
    :param expected_sha256: Optional externally pinned raw manifest SHA-256.
    :param verify_finevision_materialization: Forwarded to the parent provenance loader.
    :param load_image_path_signatures: Forwarded to the parent provenance loader.
    :returns: The validated projection and exact parent selections.
    :raises ValueError: If any schema, artifact, parent, adapter, row, or image identity differs.
    """
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.name != JOINT_VISUAL_PROJECTION_MANIFEST:
        raise ValueError(
            f"Joint visual projection must use canonical name {JOINT_VISUAL_PROJECTION_MANIFEST!r}"
        )
    try:
        raw = manifest_path.read_bytes()
        root_value = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid joint visual projection {manifest_path}: {error}") from error
    root = _exact_mapping(root_value, _ROOT_FIELDS, name="joint visual projection")
    raw_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and raw_sha != _sha256(
        expected_sha256, name="expected joint projection SHA-256"
    ):
        raise ValueError(
            f"Joint visual projection raw SHA mismatch: expected {expected_sha256}, got {raw_sha}"
        )
    complete_path = manifest_path.parent / "COMPLETE"
    try:
        complete_stat = complete_path.lstat()
        complete_raw = complete_path.read_bytes()
    except OSError as error:
        raise ValueError("Joint visual projection lacks its COMPLETE marker") from error
    if (
        not stat.S_ISREG(complete_stat.st_mode)
        or complete_path.is_symlink()
        or complete_raw != f"{raw_sha}\n".encode("ascii")
    ):
        raise ValueError("Joint visual projection COMPLETE marker differs")
    if (
        root["format"] != JOINT_VISUAL_PROJECTION_FORMAT
        or _count(root["version"], name="joint projection version", minimum=1)
        != JOINT_VISUAL_PROJECTION_VERSION
        or root["status"] != "verified"
        or root["phase"] != "joint"
    ):
        raise ValueError("Joint visual projection identity or status is incompatible")
    if not isinstance(root["created_at"], str):
        raise ValueError("Joint visual projection created_at must be ISO-8601")
    try:
        created_at = datetime.fromisoformat(root["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Joint visual projection created_at is not ISO-8601") from error
    if created_at.tzinfo is None:
        raise ValueError("Joint visual projection created_at must include a timezone")

    builder = _exact_mapping(root["builder"], _BUILDER_FIELDS, name="joint projection builder")
    _sha256(builder["script_sha256"], name="joint builder.script_sha256")
    if not isinstance(builder["name"], str) or not builder["name"]:
        raise ValueError("Joint visual projection builder.name must be a non-empty string")
    _count(builder["version"], name="joint builder.version", minimum=1)
    content_sha = _sha256(root["content_sha256"], name="joint projection content_sha256")
    unsigned = dict(root)
    unsigned.pop("content_sha256")
    if _canonical_sha256(unsigned) != content_sha:
        raise ValueError("Joint visual projection content SHA-256 differs")

    parent_ref = _exact_mapping(
        root["parent_perception_provenance"],
        _PARENT_FIELDS,
        name="parent_perception_provenance",
    )
    parent_path = _parent_manifest_path(manifest_path.parent, parent_ref["path"])
    parent_expected_sha = _sha256(parent_ref["sha256"], name="parent provenance.sha256")
    try:
        parent_raw = parent_path.read_bytes()
        parent_root = json.loads(parent_raw, object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Parent perception provenance is invalid") from error
    if hashlib.sha256(parent_raw).hexdigest() != parent_expected_sha:
        raise ValueError("Parent perception provenance raw SHA-256 differs")
    if not isinstance(parent_root, Mapping):
        raise ValueError("Parent perception provenance must be an object")
    parent = load_perception_provenance_manifest(
        parent_path,
        expected_sha256=parent_expected_sha,
        verify_finevision_materialization=verify_finevision_materialization,
        load_image_path_signatures=load_image_path_signatures,
    )
    if (
        parent.raw_sha256 != parent_expected_sha
        or parent.content_sha256
        != _sha256(parent_ref["content_sha256"], name="parent provenance.content_sha256")
        or parent.source_spec_sha256
        != _sha256(parent_ref["source_spec_sha256"], name="parent provenance.source_spec_sha256")
        or parent_root.get("source_registry_sha256")
        != _sha256(
            parent_ref["source_registry_sha256"],
            name="parent provenance.source_registry_sha256",
        )
    ):
        raise ValueError("Parent perception provenance reference differs")

    token_ids = _projection_token_ids(root["token_ids"])
    if type(expected_token_ids) is not Molmo2TokenIds or token_ids != expected_token_ids:
        raise ValueError("Joint projection token IDs differ from the expected runtime")
    if root["source_name_projection"] != dict(JOINT_TO_PERCEPTION_SOURCE):
        raise ValueError("Joint source-name projection differs")
    joint_spec = VisionAlignmentJointSourceSpec.from_perception(parent.source_spec)
    joint_spec_sha = _sha256(root["source_spec_sha256"], name="joint source_spec_sha256")
    if _canonical_bytes(root["source_spec"]) != _canonical_bytes(
        joint_spec.as_canonical_dict()
    ) or (joint_spec_sha != joint_spec.preprocessing_sha256):
        raise ValueError("Joint source specification differs from its parent projection")
    if (
        _count(root["source_registry_version"], name="joint source_registry_version", minimum=1)
        != VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION
    ):
        raise ValueError("Joint source registry version differs")

    policy = _exact_mapping(root["projection_policy"], _POLICY_FIELDS, name="projection_policy")
    if (
        policy["algorithm"] != JOINT_VISUAL_PROJECTION_ALGORITHM
        or _count(policy["parent_sequence_length"], name="parent sequence length", minimum=1)
        != _PARENT_SEQUENCE_LENGTH
        or _count(policy["sequence_length"], name="joint sequence length", minimum=1)
        != _JOINT_SEQUENCE_LENGTH
        or policy["allowed_adapter_config_delta"] != ["max_sequence_length"]
    ):
        raise ValueError("Joint projection policy differs")

    raw_sources = root["sources"]
    if not isinstance(raw_sources, Mapping) or tuple(sorted(raw_sources)) != tuple(
        JOINT_VISUAL_SOURCE_NAMES
    ):
        raise ValueError("Joint projection must contain the exact eight-source set")
    selections: Dict[tuple[str, str], JointVisualSplitProjection] = {}
    for source_name in JOINT_VISUAL_SOURCE_NAMES:
        parent_source_name = JOINT_TO_PERCEPTION_SOURCE[source_name]
        source = _exact_mapping(
            raw_sources[source_name], _SOURCE_FIELDS, name=f"sources.{source_name}"
        )
        if source["parent_source_name"] != parent_source_name or source[
            "components"
        ] != _parent_components(parent_root, parent_source_name):
            raise ValueError(f"Joint projection mapping or components differ for {source_name}")
        for logical_split in ("train", "validation"):
            split = _exact_mapping(
                source[logical_split],
                _SPLIT_FIELDS,
                name=f"sources.{source_name}.{logical_split}",
            )
            parent_selection = parent.selection(parent_source_name, logical_split)
            physical_split = split["physical_split"]
            if not isinstance(physical_split, str) or physical_split != (
                parent_selection.physical_split
            ):
                raise ValueError(f"{source_name}.{logical_split} physical split differs")
            base_examples = _count(
                split["base_examples"],
                name=f"{source_name}.{logical_split}.base_examples",
                minimum=1,
            )
            if base_examples != parent_selection.base_examples:
                raise ValueError(f"{source_name}.{logical_split} base example count differs")
            base_fingerprint = _fingerprint(
                split["joint_base_dataset_fingerprint"],
                name=f"{source_name}.{logical_split}.joint_base_dataset_fingerprint",
            )
            base_annotation = _sha256(
                split["joint_base_annotation_sha256"],
                name=f"{source_name}.{logical_split}.joint_base_annotation_sha256",
            )
            if (source_name in JOINT_SEQUENCE_SENSITIVE_ANNOTATION_SOURCES) == (
                base_annotation == parent_selection.base_annotation_sha256
            ):
                raise ValueError(
                    f"{source_name}.{logical_split} annotation projection relation differs"
                )
            selection_sha = _sha256(
                split["selection_indices_sha256"],
                name=f"{source_name}.{logical_split}.selection_indices_sha256",
            )
            if selection_sha != parent_selection.selection_indices_sha256:
                raise ValueError(f"{source_name}.{logical_split} parent selection differs")
            expected_adapter_sha = vision_alignment_joint_adapter_projection_sha256(
                build_vision_alignment_joint_dataset_config(
                    joint_spec,
                    token_ids,
                    source_name,
                    split=physical_split,
                )
            )
            adapter_sha = _sha256(
                split["adapter_projection_sha256"],
                name=f"{source_name}.{logical_split}.adapter_projection_sha256",
            )
            if adapter_sha != expected_adapter_sha:
                raise ValueError(f"{source_name}.{logical_split} adapter projection differs")
            runtime_examples = _count(
                split["runtime_examples"],
                name=f"{source_name}.{logical_split}.runtime_examples",
                minimum=1,
            )
            if runtime_examples != len(parent_selection.indices):
                raise ValueError(f"{source_name}.{logical_split} runtime row count differs")
            row_hashes = parent_selection.row_image_content_sha256
            unique_hashes = parent_selection.unique_image_content_sha256
            if logical_split == "validation" and (
                runtime_examples != VALIDATION_IMAGE_CONTENTS_PER_SOURCE
                or len(unique_hashes) != VALIDATION_IMAGE_CONTENTS_PER_SOURCE
            ):
                raise ValueError(
                    f"{source_name}.validation must retain exactly "
                    f"{VALIDATION_IMAGE_CONTENTS_PER_SOURCE} distinct image contents"
                )
            if split["row_image_content_sha256"] != _inventory_sha256(row_hashes) or split[
                "unique_image_content_sha256"
            ] != _inventory_sha256(unique_hashes):
                raise ValueError(f"{source_name}.{logical_split} image inventories differ")
            expected_runtime_fingerprint = joint_selected_dataset_fingerprint(
                source_name=source_name,
                parent_source_name=parent_source_name,
                logical_split=logical_split,
                physical_split=physical_split,
                joint_base_fingerprint=base_fingerprint,
                selection_indices_sha256=selection_sha,
                joint_source_spec_sha256=joint_spec_sha,
                parent_provenance_sha256=parent.raw_sha256,
                parent_provenance_content_sha256=parent.content_sha256,
            )
            if split["runtime_dataset_fingerprint"] != expected_runtime_fingerprint:
                raise ValueError(f"{source_name}.{logical_split} runtime fingerprint differs")
            selections[(source_name, logical_split)] = JointVisualSplitProjection(
                physical_split=physical_split,
                base_examples=base_examples,
                joint_base_dataset_fingerprint=base_fingerprint,
                joint_base_annotation_sha256=base_annotation,
                adapter_projection_sha256=adapter_sha,
                indices=parent_selection.indices,
                selection_indices_sha256=selection_sha,
                runtime_dataset_fingerprint=expected_runtime_fingerprint,
                row_image_content_sha256=row_hashes,
                unique_image_content_sha256=unique_hashes,
            )

    train_union = tuple(
        sorted(
            {
                value
                for source_name in JOINT_VISUAL_SOURCE_NAMES
                for value in selections[(source_name, "train")].unique_image_content_sha256
            }
        )
    )
    validation_union = tuple(
        sorted(
            {
                value
                for source_name in JOINT_VISUAL_SOURCE_NAMES
                for value in selections[(source_name, "validation")].unique_image_content_sha256
            }
        )
    )
    parent_train_union = tuple(
        sorted(
            {
                value
                for source_name in JOINT_VISUAL_SOURCE_NAMES
                for value in parent.selection(
                    JOINT_TO_PERCEPTION_SOURCE[source_name], "train"
                ).unique_image_content_sha256
            }
        )
    )
    parent_validation_union = tuple(
        sorted(
            {
                value
                for source_name in JOINT_VISUAL_SOURCE_NAMES
                for value in parent.selection(
                    JOINT_TO_PERCEPTION_SOURCE[source_name], "validation"
                ).unique_image_content_sha256
            }
        )
    )
    overlap = set(train_union).intersection(validation_union)
    unions = _exact_mapping(root["unions"], _UNION_FIELDS, name="unions")
    if (
        train_union != parent_train_union
        or validation_union != parent_validation_union
        or overlap
        or unions["train_unique_image_content_sha256"] != _inventory_sha256(train_union)
        or _count(unions["train_count"], name="unions.train_count", minimum=1) != len(train_union)
        or unions["validation_unique_image_content_sha256"] != _inventory_sha256(validation_union)
        or _count(unions["validation_count"], name="unions.validation_count", minimum=1)
        != len(validation_union)
        or _count(unions["overlap_count"], name="unions.overlap_count") != 0
    ):
        raise ValueError("Joint image unions differ from the disjoint parent unions")

    return JointVisualProjectionManifest(
        path=manifest_path,
        raw_sha256=raw_sha,
        content_sha256=content_sha,
        parent_provenance=parent,
        token_ids=token_ids,
        source_spec=joint_spec,
        source_spec_sha256=joint_spec_sha,
        selections=selections,
    )


class SelectedVisionAlignmentJointDataset:
    """Apply an exact parent row selection to one validated 8,192-token joint adapter."""

    content_fingerprint_version = "vision-alignment-joint-selected-v1"

    def __init__(
        self,
        dataset: Any,
        *,
        source_name: str,
        logical_split: str,
        selection: JointVisualSplitProjection,
        parent_annotation_sha256: str,
    ):
        base_fingerprint = runtime_dataset_fingerprint(dataset)
        if (
            base_fingerprint != selection.joint_base_dataset_fingerprint
            or len(dataset) != selection.base_examples
            or perception_annotation_content_sha256(dataset)
            != selection.joint_base_annotation_sha256
        ):
            raise ValueError(f"Raw joint {source_name}/{logical_split} dataset differs")
        config = getattr(dataset, "config", None)
        if (
            config is None
            or getattr(config, "max_sequence_length", None) != _JOINT_SEQUENCE_LENGTH
            or vision_alignment_joint_adapter_projection_sha256(config)
            != selection.adapter_projection_sha256
        ):
            raise ValueError(f"Raw joint {source_name}/{logical_split} adapter differs")
        parent_annotation_sha256 = _sha256(
            parent_annotation_sha256,
            name=f"{source_name}/{logical_split} parent annotation SHA-256",
        )
        if source_name in JOINT_SEQUENCE_SENSITIVE_ANNOTATION_SOURCES:
            joint_replay = vision_alignment_joint_annotation_replay_sha256(
                dataset,
                source_name,
                sequence_length=_JOINT_SEQUENCE_LENGTH,
            )
            parent_replay = vision_alignment_joint_annotation_replay_sha256(
                dataset,
                source_name,
                sequence_length=_PARENT_SEQUENCE_LENGTH,
            )
            if (
                joint_replay != selection.joint_base_annotation_sha256
                or parent_replay != parent_annotation_sha256
            ):
                raise ValueError(
                    f"Raw joint {source_name}/{logical_split} annotation projection differs"
                )
        elif selection.joint_base_annotation_sha256 != parent_annotation_sha256:
            raise ValueError(
                f"Raw joint {source_name}/{logical_split} stable annotation projection differs"
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

    def _raw_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(f"Selected joint row index is out of bounds: {index!r}")
        return self.indices[index]

    def get(self, index: int, epoch: int = 0) -> Any:
        """Build one selected joint row without substituting its parent raw index."""
        raw_index = self._raw_index(index)
        get = getattr(self._dataset, "get", None)
        return get(raw_index, epoch) if callable(get) else self._dataset[raw_index]

    def raw_image_references(self, index: int) -> Tuple[Any, ...]:
        """Return the exact raw image references for one selected logical row."""
        raw_images = getattr(self._dataset, "raw_image_references", None)
        if not callable(raw_images):
            raise ValueError(f"Selected joint source {self.source_name!r} lacks raw image access")
        return tuple(raw_images(self._raw_index(index)))

    def validate_image_content(self, indices: Optional[Sequence[int]] = None) -> str:
        """Rehash selected image bytes against the exact parent image inventory."""
        selected_indices = tuple(range(len(self))) if indices is None else tuple(indices)
        rows = []
        for index in selected_indices:
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
                raise ValueError(f"Selected joint image index is out of bounds: {index!r}")
            references = self.raw_image_references(index)
            if len(references) != 1:
                raise ValueError(
                    f"Selected joint source {self.source_name!r} row {index} has "
                    f"{len(references)} raw images; expected exactly one"
                )
            actual = image_reference_sha256(references[0])
            expected = self._selection.row_image_content_sha256[index]
            if actual != expected:
                raise ValueError(
                    f"Selected joint source {self.source_name!r} row {index} image bytes differ"
                )
            rows.append({"index": index, "image_sha256": actual})
        return _canonical_sha256(rows)

    def validate_required_annotations(self) -> None:
        """Run the raw joint adapter's fail-closed annotation validator."""
        validate = getattr(self._dataset, "validate_required_annotations", None)
        if not callable(validate):
            raise ValueError(
                f"Selected joint source {self.source_name!r} lacks annotation validation"
            )
        validate()


def build_selected_joint_dataset(
    manifest: JointVisualProjectionManifest,
    tokenizer: Any,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    logical_split: str,
    validate_required_annotations: bool = True,
) -> SelectedVisionAlignmentJointDataset:
    """Build one 8,192-token joint adapter and apply its exact parent row selection."""
    if type(token_ids) is not Molmo2TokenIds or token_ids != manifest.token_ids:
        raise ValueError("Joint runtime token IDs differ from the projection artifact")
    selection = manifest.selection(source_name, logical_split)
    parent_selection = manifest.parent_provenance.selection(
        JOINT_TO_PERCEPTION_SOURCE[source_name], logical_split
    )
    dataset = build_vision_alignment_joint_dataset(
        manifest.source_spec,
        tokenizer,
        token_ids,
        source_name,
        split=selection.physical_split,
        validate_required_annotations=validate_required_annotations,
    )
    return SelectedVisionAlignmentJointDataset(
        dataset,
        source_name=source_name,
        logical_split=logical_split,
        selection=selection,
        parent_annotation_sha256=parent_selection.base_annotation_sha256,
    )
