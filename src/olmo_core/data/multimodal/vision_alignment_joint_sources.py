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
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping

from olmo_core.nn.vision import Molmo2TokenIds

from .pixmo_cap import PixMoCapDatasetConfig
from .pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from .vision_alignment_perception import (
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDatasetConfig,
)
from .vision_alignment_perception_sources import (
    VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
    VisionAlignmentPerceptionSourceSpec,
    build_vision_alignment_perception_dataset_config,
    vision_alignment_perception_implementation_inventory,
)

__all__ = [
    "JOINT_TO_PERCEPTION_SOURCE",
    "JOINT_VISUAL_SOURCE_NAMES",
    "VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM",
    "VISION_ALIGNMENT_JOINT_PARENT_SOURCE_MODULE_SHA256",
    "VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION",
    "VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION",
    "VisionAlignmentJointSourceSpec",
    "build_vision_alignment_joint_dataset",
    "build_vision_alignment_joint_dataset_config",
    "vision_alignment_joint_adapter_projection_sha256",
    "vision_alignment_joint_implementation_inventory",
    "vision_alignment_joint_source_registry_sha256",
]

VISION_ALIGNMENT_JOINT_SOURCE_REGISTRY_VERSION = 1
VISION_ALIGNMENT_JOINT_SOURCE_CATALOG_VERSION = 1
VISION_ALIGNMENT_JOINT_ADAPTER_PROJECTION_ALGORITHM = (
    "sha256-json-config-class-and-asdict-without-max-sequence-length-v1"
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
        "files": {name: files[name] for name in sorted(files)},
    }


def vision_alignment_joint_source_registry_sha256() -> str:
    """Return a digest of the exact transitive joint visual-source implementation."""
    return hashlib.sha256(
        _canonical_bytes(vision_alignment_joint_implementation_inventory())
    ).hexdigest()
