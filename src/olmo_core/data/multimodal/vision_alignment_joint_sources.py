"""Canonical visual-source registry for the Vision Alignment joint phase.

Joint training deliberately reuses the reviewed perception adapters and serialization
contract. Its only adapter-config change is the larger sequence bound; the public mixture
name ``count_numeric`` maps to perception's internal ``scalar_count`` adapter name. This
module keeps that derivation explicit and fail-closed without changing the immutable
perception registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping

import numpy as np

from olmo_core.nn.vision import Molmo2TokenIds

from .finevision import FineVisionDataset, FineVisionDatasetConfig
from .pixmo_cap import PixMoCapDataset, PixMoCapDatasetConfig
from .pixmo_points import (
    CoSynPointDataset,
    CoSynPointDatasetConfig,
    PixMoCountDataset,
    PixMoCountDatasetConfig,
    PixMoPointsDataset,
    PixMoPointsDatasetConfig,
)
from .vision_alignment_perception import (
    VisionAlignmentAuditedAlignmentDataset,
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDataset,
    VisionAlignmentOcrDocumentDatasetConfig,
)
from .vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset_config,
    vision_alignment_perception_implementation_inventory,
)

__all__ = [
    "JOINT_SEQUENCE_SENSITIVE_ANNOTATION_SOURCES",
    "JOINT_TO_PERCEPTION_SOURCE",
    "JOINT_VISUAL_SOURCE_NAMES",
    "VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM",
    "VISION_ALIGNMENT_JOINT_ANNOTATION_PROJECTION_ALGORITHM",
    "VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256",
    "VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION",
    "VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION",
    "VisionAlignmentJointSourceSpec",
    "build_vision_alignment_joint_dataset",
    "build_vision_alignment_joint_dataset_config",
    "vision_alignment_joint_adapter_projection_sha256",
    "vision_alignment_joint_annotation_replay_sha256",
    "vision_alignment_joint_implementation_inventory",
    "vision_alignment_joint_source_registry_sha256",
]

VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION = 1
VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION = 1
VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM = (
    "sha256-json-config-class-and-asdict-without-max-sequence-length-v1"
)
VISION_ALIGNMENT_JOINT_ANNOTATION_PROJECTION_ALGORITHM = (
    "source-aware-full-annotation-max-sequence-length-projection-v1"
)
VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256 = (
    "15db21181c9751a0ed14a7d44a3f58ca53377447ab346e30979d6a37b32900a8"
)

JOINT_VISUAL_SOURCE_NAMES = tuple(
    sorted(
        (
            "audited_alignment",
            "cosyn_point",
            "count_numeric",
            "ocr_document",
            "pixmo_caption",
            "pixmo_points_basic",
            "pixmo_points_high_frequency",
            "pixmo_transcript",
        )
    )
)
"""Canonical public visual-source names for joint training and evaluation."""

JOINT_TO_PERCEPTION_SOURCE: Mapping[str, str] = MappingProxyType(
    {
        source_name: "scalar_count" if source_name == "count_numeric" else source_name
        for source_name in JOINT_VISUAL_SOURCE_NAMES
    }
)
"""Exact joint-to-perception adapter-name projection."""

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
JOINT_SEQUENCE_SENSITIVE_ANNOTATION_SOURCES = frozenset({"audited_alignment", "ocr_document"})
"""Sources whose legacy annotation identity includes ``max_sequence_length``."""


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


def _perception_source_module_path() -> Path:
    return Path(__file__).resolve().with_name("vision_alignment_perception_sources.py")


def _validate_pinned_perception_registry() -> None:
    actual = _sha256_file(_perception_source_module_path())
    if actual != VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256:
        raise ValueError(
            "Joint visual sources require the pinned perception source module: "
            f"expected {VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256}, got {actual}"
        )


@dataclass(frozen=True)
class VisionAlignmentJointSourceSpec:
    """Joint serialization derived from one reviewed perception source spec.

    :param perception_spec: Exact production perception specification selected by the parent
        checkpoint's provenance.
    :param phase: Must remain ``joint``.
    :param sequence_length: Must remain 8,192. Every other preprocessing field is read from
        ``perception_spec`` and cannot be independently overridden.
    """

    perception_spec: VisionAlignmentPerceptionSourceSpec
    phase: str = "joint"
    sequence_length: int = 8192

    @classmethod
    def from_perception(
        cls, perception_spec: VisionAlignmentPerceptionSourceSpec
    ) -> "VisionAlignmentJointSourceSpec":
        """Derive the unique production joint specification from ``perception_spec``."""
        spec = cls(perception_spec=perception_spec)
        spec.validate_production_contract()
        return spec

    def validate_production_contract(self) -> None:
        """Validate the pinned parent and the two allowed phase-level substitutions."""
        _validate_pinned_perception_registry()
        if type(self.perception_spec) is not VisionAlignmentPerceptionSourceSpec:
            raise ValueError(
                "Joint source spec requires an exact VisionAlignmentPerceptionSourceSpec"
            )
        self.perception_spec.validate_production_contract()
        if self.phase != "joint":
            raise ValueError(f"Joint source spec phase must be 'joint', got {self.phase!r}")
        if type(self.sequence_length) is not int or self.sequence_length != 8192:
            raise ValueError(
                "Joint source spec sequence_length must be exactly 8192, got "
                f"{self.sequence_length!r}"
            )

    def as_canonical_dict(self) -> Dict[str, Any]:
        """Return the joint descriptor mechanically projected from the parent spec."""
        self.validate_production_contract()
        parent = dict(self.perception_spec.as_canonical_dict())
        parent_registry_version = parent.pop("source_registry_version")
        parent["phase"] = self.phase
        parent["sequence_length"] = self.sequence_length
        return {
            "source_registry_version": VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION,
            "parent_perception_source_registry_version": parent_registry_version,
            "parent_perception_preprocessing_sha256": (self.perception_spec.preprocessing_sha256),
            **parent,
        }

    @property
    def preprocessing_sha256(self) -> str:
        """Return the SHA-256 of the canonical joint preprocessing descriptor."""
        return hashlib.sha256(_canonical_bytes(self.as_canonical_dict())).hexdigest()


def _config_projection(config: Any) -> Dict[str, Any]:
    try:
        values = asdict(config)
    except TypeError as error:
        raise ValueError("Joint adapter projection requires a dataclass config") from error
    if "max_sequence_length" not in values:
        raise ValueError("Joint adapter config must declare max_sequence_length")
    del values["max_sequence_length"]
    config_type = type(config)
    return {
        "algorithm": VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM,
        "config_class": f"{config_type.__module__}.{config_type.__qualname__}",
        "config": values,
    }


def vision_alignment_joint_adapter_projection_sha256(config: Any) -> str:
    """Hash a fully-qualified adapter config after removing only its sequence bound.

    :param config: A dataclass dataset configuration with ``max_sequence_length``.
    :returns: SHA-256 of the canonical class-and-config projection.
    :raises ValueError: If ``config`` is not a compatible dataclass configuration.
    """
    return hashlib.sha256(_canonical_bytes(_config_projection(config))).hexdigest()


def _annotation_sha256(dataset: Any, *, source_name: str) -> str:
    annotation_identity = getattr(dataset, "annotation_content_sha256", None)
    if not callable(annotation_identity):
        raise ValueError(f"Joint visual source {source_name!r} has no annotation identity")
    digest = annotation_identity()
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(
            f"Joint visual source {source_name!r} returned an invalid annotation SHA-256"
        )
    return digest


def _require_source_dataset(dataset: Any, source_name: str) -> None:
    expected: type[Any]
    expected_config: type[Any]
    if source_name == "audited_alignment":
        expected = VisionAlignmentAuditedAlignmentDataset
        expected_config = VisionAlignmentAuditedAlignmentDatasetConfig
    elif source_name == "ocr_document":
        expected = VisionAlignmentOcrDocumentDataset
        expected_config = VisionAlignmentOcrDocumentDatasetConfig
    elif source_name in {"pixmo_caption", "pixmo_transcript"}:
        expected = PixMoCapDataset
        expected_config = PixMoCapDatasetConfig
    elif source_name in {"pixmo_points_basic", "pixmo_points_high_frequency"}:
        expected = PixMoPointsDataset
        expected_config = PixMoPointsDatasetConfig
    elif source_name == "count_numeric":
        expected = PixMoCountDataset
        expected_config = PixMoCountDatasetConfig
    elif source_name == "cosyn_point":
        expected = CoSynPointDataset
        expected_config = CoSynPointDatasetConfig
    else:
        raise KeyError(source_name)
    if type(dataset) is not expected:
        raise ValueError(
            f"Joint visual source {source_name!r} requires exact dataset type "
            f"{expected.__module__}.{expected.__qualname__}"
        )

    config = getattr(dataset, "config", None)
    if type(config) is not expected_config:
        raise ValueError(
            f"Joint visual source {source_name!r} requires exact config type "
            f"{expected_config.__module__}.{expected_config.__qualname__}"
        )
    if source_name == "pixmo_caption" and getattr(config, "mode", None) != "caption":
        raise ValueError("pixmo_caption annotation replay requires caption mode")
    if source_name == "pixmo_transcript" and getattr(config, "mode", None) != "transcript":
        raise ValueError("pixmo_transcript annotation replay requires transcript mode")
    if source_name == "pixmo_points_basic" and getattr(config, "kind", None) != "basic":
        raise ValueError("pixmo_points_basic annotation replay requires basic kind")
    if source_name == "pixmo_points_high_frequency" and (
        getattr(config, "kind", None) != "high_frequency"
    ):
        raise ValueError(
            "pixmo_points_high_frequency annotation replay requires high_frequency kind"
        )
    if source_name == "count_numeric" and getattr(config, "mode", None) != "scalar_count":
        raise ValueError("count_numeric annotation replay requires scalar_count mode")


def _ocr_annotation_replay_sha256(dataset: Any, *, sequence_length: int) -> str:
    config = dataset.config
    ordered = hashlib.sha256()
    for index in range(len(dataset)):
        source_name, row = dataset._row(index)
        ordered.update(
            _canonical_bytes({"index": index, **dataset._descriptor(source_name, row)}) + b"\n"
        )
    return hashlib.sha256(
        _canonical_bytes(
            {
                "version": dataset.content_fingerprint_version,
                "source_names": list(config.source_names),
                "split": config.split,
                "rows": len(dataset),
                "ordered_annotations_sha256": ordered.hexdigest(),
                "max_crops": config.max_crops,
                "max_sequence_length": sequence_length,
                "loss_token_weighting": config.loss_token_weighting,
                "message_format": config.message_format,
                "prompt_prefix": config.prompt_prefix,
                "answer_prefix": config.answer_prefix,
                "seed": config.seed,
                "skip_bad_rows": config.skip_bad_rows,
            }
        )
    ).hexdigest()


def _finevision_annotation_replay_sha256(dataset: Any, *, sequence_length: int) -> str:
    if type(dataset) is not FineVisionDataset:
        raise ValueError("Audited-alignment annotation replay requires exact FineVision children")
    config = dataset.config
    if type(config) is not FineVisionDatasetConfig:
        raise ValueError(
            "Audited-alignment annotation replay requires exact FineVision child configs"
        )
    arrow_fingerprint = (
        config.expected_materialized_fingerprint
        if config.expected_materialized_fingerprint is not None
        else getattr(dataset._data, "_fingerprint", None)
    )
    if type(arrow_fingerprint) is not str or not arrow_fingerprint:
        raise ValueError("Audited-alignment child lacks a stable Arrow fingerprint")
    selected = (
        np.arange(len(dataset._data), dtype="<i8")
        if dataset._index is None
        else np.asarray(dataset._index, dtype="<i8")
    )
    if selected.ndim != 1:
        raise ValueError("Audited-alignment child selection must be one-dimensional")
    return hashlib.sha256(
        _canonical_bytes(
            {
                "version": dataset.content_fingerprint_version,
                "arrow_fingerprint": arrow_fingerprint,
                "config_name": config.config_name,
                "dataset_path": os.path.realpath(config.resolved_path()),
                "split": config.split,
                "texts_column": config.texts_column,
                "images_column": config.images_column,
                "quality_filters": {
                    name: getattr(config, name)
                    for name in (
                        "min_formatting",
                        "min_visual_dependency",
                        "min_image_correspondence",
                        "min_relevance",
                    )
                },
                "require_quality_columns": config.require_quality_columns,
                "strict_annotations": config.strict_annotations,
                "skip_bad_rows": config.skip_bad_rows,
                "max_crops": config.max_crops,
                "max_images": config.max_images,
                "max_sequence_length": sequence_length,
                "loss_token_weighting": config.loss_token_weighting,
                "message_format": config.message_format,
                "seed": config.seed,
                "selected_rows": len(selected),
                "selection_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            }
        )
    ).hexdigest()


def _audited_alignment_annotation_replay_sha256(dataset: Any, *, sequence_length: int) -> str:
    config = dataset.config
    children = getattr(dataset, "_datasets", None)
    if type(children) is not list or len(children) != 2:
        raise ValueError("Audited-alignment annotation replay requires exactly two children")
    if any(
        type(getattr(child, "config", None)) is not FineVisionDatasetConfig for child in children
    ):
        raise ValueError(
            "Audited-alignment annotation replay requires exact FineVision child configs"
        )
    if any(
        getattr(getattr(child, "config", None), "max_sequence_length", None)
        != config.max_sequence_length
        for child in children
    ):
        raise ValueError("Audited-alignment child sequence bounds must match the outer adapter")
    expected_offsets = [0]
    for child in children:
        expected_offsets.append(expected_offsets[-1] + len(child))
    if getattr(dataset, "_offsets", None) != expected_offsets:
        raise ValueError("Audited-alignment child offsets differ from child lengths")
    child_fingerprints = [
        _finevision_annotation_replay_sha256(child, sequence_length=sequence_length)
        for child in children
    ]
    return hashlib.sha256(
        _canonical_bytes(
            {
                "version": dataset.content_fingerprint_version,
                "split": config.split,
                "child_fingerprints": child_fingerprints,
                "child_lengths": [len(child) for child in children],
                "max_crops": config.max_crops,
                "max_sequence_length": sequence_length,
                "loss_token_weighting": config.loss_token_weighting,
                "message_format": config.message_format,
                "seed": config.seed,
            }
        )
    ).hexdigest()


def vision_alignment_joint_annotation_replay_sha256(
    dataset: Any,
    source_name: str,
    *,
    sequence_length: int,
) -> str:
    """Replay a visual adapter's annotation identity at an explicit sequence bound.

    Most visual adapters hash only ordered source annotations and are therefore invariant to
    serialization length. The reviewed OCR and audited-alignment adapters historically fold
    ``max_sequence_length`` into that identity. For those two sources this function rebuilds
    their exact pinned payload while substituting only the requested length. Replaying both
    the perception and joint bounds proves that no annotation or adapter field changed behind
    the permitted 2,560-to-8,192 projection.

    :param dataset: An exact canonical raw visual adapter instance.
    :param source_name: Canonical public joint source name.
    :param sequence_length: Positive sequence bound to substitute into the identity.
    :returns: The canonical annotation SHA-256 at ``sequence_length``.
    :raises KeyError: If ``source_name`` is not a canonical joint visual name.
    :raises ValueError: If the dataset type, mode, payload, or native replay has drifted.
    """
    if source_name not in JOINT_TO_PERCEPTION_SOURCE:
        raise KeyError(source_name)
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("annotation replay sequence_length must be a positive integer")
    _require_source_dataset(dataset, source_name)
    config = dataset.config
    native_sequence_length = getattr(config, "max_sequence_length", None)
    if type(native_sequence_length) is not int or native_sequence_length <= 0:
        raise ValueError(f"Joint visual source {source_name!r} has an invalid sequence bound")

    if source_name == "ocr_document":
        native_replay = _ocr_annotation_replay_sha256(
            dataset, sequence_length=native_sequence_length
        )
        if native_replay != _annotation_sha256(dataset, source_name=source_name):
            raise ValueError(
                f"Joint visual source {source_name!r} native annotation replay differs"
            )
        replay = _ocr_annotation_replay_sha256(dataset, sequence_length=sequence_length)
    elif source_name == "audited_alignment":
        native_replay = _audited_alignment_annotation_replay_sha256(
            dataset, sequence_length=native_sequence_length
        )
        if native_replay != _annotation_sha256(dataset, source_name=source_name):
            raise ValueError(
                f"Joint visual source {source_name!r} native annotation replay differs"
            )
        replay = _audited_alignment_annotation_replay_sha256(
            dataset, sequence_length=sequence_length
        )
    else:
        replay = _annotation_sha256(dataset, source_name=source_name)

    return replay


def _build_joint_dataset_config_unchecked(
    spec: VisionAlignmentJointSourceSpec,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str,
) -> Any:
    parent = spec.perception_spec
    common: Dict[str, Any] = {
        "max_crops": parent.max_crops,
        "max_sequence_length": spec.sequence_length,
        "loss_token_weighting": parent.loss_token_weighting,
        "token_ids": token_ids,
        "message_format": parent.message_format,
        "seed": 0,
    }
    if source_name == "pixmo_caption":
        return PixMoCapDatasetConfig(
            dataset_path=parent.pixmo_cap_path,
            split=split,
            require_split=True,
            mode="caption",
            fixed_prompt=parent.caption_prompt,
            style_length_conditioning=False,
            **common,
        )
    if source_name == "pixmo_transcript":
        return PixMoCapDatasetConfig(
            dataset_path=parent.pixmo_cap_path,
            split=split,
            require_split=True,
            mode="transcript",
            require_transcript=parent.require_transcript,
            fixed_prompt=parent.transcript_prompt,
            style_length_conditioning=False,
            **common,
        )
    if source_name == "pixmo_points_basic":
        return PixMoPointsDatasetConfig(
            split=split,
            require_split=True,
            kind="basic",
            counting=False,
            both_mode="per_annotation",
            **common,
        )
    if source_name == "pixmo_points_high_frequency":
        return PixMoPointsDatasetConfig(
            split=split,
            require_split=True,
            kind="high_frequency",
            counting=False,
            both_mode="per_annotation",
            **common,
        )
    if source_name == "cosyn_point":
        return CoSynPointDatasetConfig(split=split, require_split=True, **common)
    if source_name == "count_numeric":
        return PixMoCountDatasetConfig(
            split=split,
            require_split=True,
            mode="scalar_count",
            counting="both",
            **common,
        )
    if source_name == "ocr_document":
        return VisionAlignmentOcrDocumentDatasetConfig(
            source_names=parent.ocr_source_names,
            split=split,
            **common,
        )
    if source_name == "audited_alignment":
        return VisionAlignmentAuditedAlignmentDatasetConfig(
            root=parent.finevision_root,
            visualweb_path=parent.finevision_visualweb_path,
            geo170k_path=parent.finevision_geo170k_path,
            visualweb_fingerprint=parent.finevision_visualweb_fingerprint,
            geo170k_fingerprint=parent.finevision_geo170k_fingerprint,
            split=split,
            **common,
        )
    raise KeyError(source_name)


def build_vision_alignment_joint_dataset_config(
    spec: VisionAlignmentJointSourceSpec,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
) -> Any:
    """Build one joint visual config and prove parity with its perception adapter.

    :param spec: Joint specification derived from the selected perception provenance.
    :param token_ids: Prepared model-specific image token identities.
    :param source_name: Canonical public joint visual-source name.
    :param split: Required source split.
    :returns: The matching 8,192-token adapter configuration.
    :raises KeyError: If ``source_name`` is not a canonical joint public name. In
        particular, callers must use ``count_numeric`` rather than ``scalar_count``.
    :raises ValueError: If the parent registry, spec, or adapter projection has drifted.
    """
    spec.validate_production_contract()
    if source_name not in JOINT_TO_PERCEPTION_SOURCE:
        raise KeyError(source_name)
    joint_config = _build_joint_dataset_config_unchecked(spec, token_ids, source_name, split=split)
    parent_name = JOINT_TO_PERCEPTION_SOURCE[source_name]
    parent_config = build_vision_alignment_perception_dataset_config(
        spec.perception_spec,
        token_ids,
        parent_name,
        split=split,
    )
    joint_digest = vision_alignment_joint_adapter_projection_sha256(joint_config)
    parent_digest = vision_alignment_joint_adapter_projection_sha256(parent_config)
    if joint_digest != parent_digest:
        raise ValueError(
            "Joint adapter config differs from its pinned perception projection for "
            f"{source_name!r} ({joint_digest} != {parent_digest})"
        )
    if getattr(joint_config, "max_sequence_length", None) != spec.sequence_length:
        raise ValueError(f"Joint adapter {source_name!r} did not retain sequence length 8192")
    if getattr(parent_config, "max_sequence_length", None) != 2560:
        raise ValueError(f"Perception adapter {parent_name!r} did not retain sequence length 2560")
    return joint_config


def build_vision_alignment_joint_dataset(
    spec: VisionAlignmentJointSourceSpec,
    tokenizer: Any,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
    validate_required_annotations: bool = True,
) -> Any:
    """Build and optionally validate one canonical joint visual dataset.

    :param spec: Joint specification derived from perception provenance.
    :param tokenizer: Prepared runtime tokenizer.
    :param token_ids: Prepared model-specific image token identities.
    :param source_name: Canonical public joint visual-source name.
    :param split: Required source split.
    :param validate_required_annotations: Enforce the adapter's strict annotation scan.
    :returns: The built map-style dataset.
    :raises ValueError: If strict validation is requested but unavailable.
    """
    dataset = build_vision_alignment_joint_dataset_config(
        spec, token_ids, source_name, split=split
    ).build(tokenizer)
    if validate_required_annotations:
        validate = getattr(dataset, "validate_required_annotations", None)
        if not callable(validate):
            raise ValueError(f"Joint visual source {source_name!r} has no annotation validator")
        validate()
    return dataset


def vision_alignment_joint_implementation_inventory() -> Dict[str, Any]:
    """Return hashes for the joint registry and its transitive perception adapters."""
    _validate_pinned_perception_registry()
    parent_inventory = vision_alignment_perception_implementation_inventory()
    files = dict(parent_inventory["files"])
    files["vision_alignment_joint_sources.py"] = _sha256_file(Path(__file__).resolve())
    return {
        "version": 1,
        "parent_perception_source_registry_version": (
            VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION
        ),
        "parent_implementation_inventory_version": parent_inventory["version"],
        "source_names": list(JOINT_VISUAL_SOURCE_NAMES),
        "source_mapping": dict(JOINT_TO_PERCEPTION_SOURCE),
        "adapter_projection_algorithm": VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM,
        "annotation_projection_algorithm": (VISION_ALIGNMENT_JOINT_ANNOTATION_PROJECTION_ALGORITHM),
        "files": {name: files[name] for name in sorted(files)},
    }


def vision_alignment_joint_source_registry_sha256() -> str:
    """Return a digest of the exact transitive joint visual-source implementation."""
    return hashlib.sha256(
        _canonical_bytes(vision_alignment_joint_implementation_inventory())
    ).hexdigest()
