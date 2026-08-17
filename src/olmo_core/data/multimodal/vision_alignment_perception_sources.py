"""Canonical source construction for the Vision Alignment perception phase.

Perception reuses the bridge serialization contract while owning its phase-specific source
specification and adapter dispatch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from olmo_core.nn.vision import Molmo2TokenIds

from .finevision import FINEVISION_ROOT
from .pixmo_cap import PixMoCapDatasetConfig
from .pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from .vision_alignment_perception import (
    VISION_ALIGNMENT_OCR_SOURCES,
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDatasetConfig,
)
from .vision_alignment_sources import (
    VISION_ALIGNMENT_FORMATTER_VERSION,
    VISION_ALIGNMENT_RECIPE_VERSION,
    VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
    VISION_ALIGNMENT_TOKENIZER_ID,
    VISION_ALIGNMENT_TOKENIZER_REVISION,
)

__all__ = [
    "VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION",
    "VisionAlignmentPerceptionSourceSpec",
    "build_vision_alignment_perception_dataset",
    "build_vision_alignment_perception_dataset_config",
]

VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class VisionAlignmentPerceptionSourceSpec:
    """All checked fields that determine perception-source serialization.

    :param phase: Must be ``perception``.
    :param pixmo_cap_path: Reviewed PixMoCap ``DatasetDict`` artifact path.
    :param sequence_length: Maximum serialized token length before packing.
    :param max_crops: Maximum image crops per example.
    :param message_format: Must be the native ``document`` layout.
    :param loss_token_weighting: Per-response weighting policy.
    :param caption_prompt: Fixed caption document prompt.
    :param transcript_prompt: Fixed transcript document prompt.
    :param require_transcript: Require every transcript row to be non-blank.
    :param ocr_source_names: Reviewed OCR/document sources in concatenation order.
    :param finevision_root: Root of the raw reviewed FineVision copies.
    :param finevision_visualweb_path: Optional materialized VisualWeb ``DatasetDict``.
    :param finevision_geo170k_path: Optional materialized Geo170K ``DatasetDict``.
    :param tokenizer_id: Dolma2 tokenizer identifier.
    :param tokenizer_revision: Immutable tokenizer revision.
    :param tokenizer_fingerprint: Digest of the pinned tokenizer files.
    :param recipe_version: Vision Alignment recipe version.
    :param formatter_version: Native document formatter identity.
    """

    phase: str
    pixmo_cap_path: str
    sequence_length: int
    max_crops: int
    message_format: str
    loss_token_weighting: str
    caption_prompt: str
    transcript_prompt: str
    require_transcript: bool
    ocr_source_names: Tuple[str, ...] = VISION_ALIGNMENT_OCR_SOURCES
    finevision_root: str = FINEVISION_ROOT
    finevision_visualweb_path: Optional[str] = None
    finevision_geo170k_path: Optional[str] = None
    finevision_visualweb_fingerprint: Optional[str] = None
    finevision_geo170k_fingerprint: Optional[str] = None
    tokenizer_id: str = VISION_ALIGNMENT_TOKENIZER_ID
    tokenizer_revision: str = VISION_ALIGNMENT_TOKENIZER_REVISION
    tokenizer_fingerprint: str = VISION_ALIGNMENT_TOKENIZER_FINGERPRINT
    recipe_version: int = VISION_ALIGNMENT_RECIPE_VERSION
    formatter_version: str = VISION_ALIGNMENT_FORMATTER_VERSION

    def as_canonical_dict(self) -> Dict[str, Any]:
        """Return the versioned, path-normalized preprocessing descriptor."""
        values = asdict(self)
        values["ocr_source_names"] = list(self.ocr_source_names)
        values["pixmo_cap_path"] = str(Path(self.pixmo_cap_path).expanduser().resolve())
        values["finevision_root"] = str(Path(self.finevision_root).expanduser().resolve())
        for field_name in ("finevision_visualweb_path", "finevision_geo170k_path"):
            path = values[field_name]
            if path is not None:
                values[field_name] = str(Path(path).expanduser().resolve())
        return {
            "source_registry_version": VISION_ALIGNMENT_PERCEPTION_SOURCE_REGISTRY_VERSION,
            **values,
        }

    def validate_production_contract(self) -> None:
        """Validate the fail-closed production perception serialization contract."""
        expected = {
            "phase": "perception",
            "sequence_length": 2560,
            "max_crops": 8,
            "message_format": "document",
            "loss_token_weighting": "root_subsegments_root_tokens",
            "caption_prompt": "Description:",
            "transcript_prompt": "Transcript:",
            "require_transcript": True,
            "ocr_source_names": VISION_ALIGNMENT_OCR_SOURCES,
            "finevision_root": FINEVISION_ROOT,
            "tokenizer_id": VISION_ALIGNMENT_TOKENIZER_ID,
            "tokenizer_revision": VISION_ALIGNMENT_TOKENIZER_REVISION,
            "tokenizer_fingerprint": VISION_ALIGNMENT_TOKENIZER_FINGERPRINT,
            "recipe_version": VISION_ALIGNMENT_RECIPE_VERSION,
            "formatter_version": VISION_ALIGNMENT_FORMATTER_VERSION,
        }
        for field_name, expected_value in expected.items():
            actual_value = getattr(self, field_name)
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise ValueError(
                    f"Perception source spec {field_name} differs: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
        for field_name in (
            "pixmo_cap_path",
            "finevision_visualweb_path",
            "finevision_geo170k_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise ValueError(f"Perception production requires an absolute {field_name}")
        for field_name in (
            "finevision_visualweb_fingerprint",
            "finevision_geo170k_fingerprint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"Perception production requires a SHA-256 {field_name}")

    @property
    def preprocessing_sha256(self) -> str:
        """SHA-256 of the canonical perception preprocessing descriptor."""
        return hashlib.sha256(_canonical_bytes(self.as_canonical_dict())).hexdigest()


def build_vision_alignment_perception_dataset_config(
    spec: VisionAlignmentPerceptionSourceSpec,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
) -> Any:
    """Build one canonical perception source configuration.

    :param spec: Exact perception preprocessing specification.
    :param token_ids: Prepared model-specific image token identities.
    :param source_name: Canonical perception mixture source.
    :param split: Required source split.
    :returns: The matching source configuration.
    :raises KeyError: If ``source_name`` has no reviewed perception adapter.
    """
    if spec.phase != "perception":
        raise ValueError(f"Invalid perception source phase {spec.phase!r}")
    common: Dict[str, Any] = {
        "max_crops": spec.max_crops,
        "max_sequence_length": spec.sequence_length,
        "loss_token_weighting": spec.loss_token_weighting,
        "token_ids": token_ids,
        "message_format": spec.message_format,
        "seed": 0,
    }
    if source_name == "pixmo_caption":
        return PixMoCapDatasetConfig(
            dataset_path=spec.pixmo_cap_path,
            split=split,
            require_split=True,
            mode="caption",
            fixed_prompt=spec.caption_prompt,
            style_length_conditioning=False,
            **common,
        )
    if source_name == "pixmo_transcript":
        return PixMoCapDatasetConfig(
            dataset_path=spec.pixmo_cap_path,
            split=split,
            require_split=True,
            mode="transcript",
            require_transcript=spec.require_transcript,
            fixed_prompt=spec.transcript_prompt,
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
    if source_name == "scalar_count":
        return PixMoCountDatasetConfig(
            split=split,
            require_split=True,
            mode="scalar_count",
            counting="both",
            **common,
        )
    if source_name == "ocr_document":
        return VisionAlignmentOcrDocumentDatasetConfig(
            source_names=spec.ocr_source_names,
            split=split,
            **common,
        )
    if source_name == "audited_alignment":
        return VisionAlignmentAuditedAlignmentDatasetConfig(
            root=spec.finevision_root,
            visualweb_path=spec.finevision_visualweb_path,
            geo170k_path=spec.finevision_geo170k_path,
            visualweb_fingerprint=spec.finevision_visualweb_fingerprint,
            geo170k_fingerprint=spec.finevision_geo170k_fingerprint,
            split=split,
            **common,
        )
    raise KeyError(source_name)


def build_vision_alignment_perception_dataset(
    spec: VisionAlignmentPerceptionSourceSpec,
    tokenizer: Any,
    token_ids: Molmo2TokenIds,
    source_name: str,
    *,
    split: str = "train",
    validate_required_annotations: bool = True,
) -> Any:
    """Build and optionally validate one canonical perception dataset."""
    dataset = build_vision_alignment_perception_dataset_config(
        spec,
        token_ids,
        source_name,
        split=split,
    ).build(tokenizer)
    if validate_required_annotations:
        validate = getattr(dataset, "validate_required_annotations", None)
        if not callable(validate):
            raise ValueError(f"Perception source {source_name!r} has no annotation validator")
        validate()
    return dataset
