"""Default visual sources for bridge, perception, and joint alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.vision_alignment_perception import (
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDatasetConfig,
)

DEFAULT_ALIGNMENT_ARTIFACT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts"
)

ALIGNMENT_LOSS_TARGETS: dict[str, dict[str, float]] = {
    "bridge": {"pixmo_caption": 0.70, "pixmo_transcript": 0.30},
    "perception": {
        "pixmo_caption": 0.45,
        "pixmo_transcript": 0.20,
        "pixmo_points_basic": 0.10,
        "pixmo_points_high_frequency": 0.02,
        "cosyn_point": 0.03,
        "ocr_document": 0.10,
        "scalar_count": 0.05,
        "audited_alignment": 0.05,
    },
    "joint": {
        "native_text_replay": 0.35,
        "pixmo_caption": 0.28,
        "pixmo_transcript": 0.12,
        "pixmo_points_basic": 0.05,
        "pixmo_points_high_frequency": 0.01,
        "cosyn_point": 0.02,
        "ocr_document": 0.08,
        "count_numeric": 0.04,
        "audited_alignment": 0.05,
    },
}

ALIGNMENT_MEAN_LOSS_WEIGHTS: dict[str, dict[str, float]] = {
    "bridge": {
        "pixmo_caption": 28.829104933305643,
        "pixmo_transcript": 29.77422616124386,
    },
    "perception": {
        "audited_alignment": 18.58728085551411,
        "cosyn_point": 20.31768759561237,
        "ocr_document": 4.165374373900704,
        "pixmo_caption": 29.374187365814578,
        "pixmo_points_basic": 33.16318166248675,
        "pixmo_points_high_frequency": 27.629776440531714,
        "pixmo_transcript": 29.7867612372429,
        "scalar_count": 3.464101552963257,
    },
    "joint": {
        "audited_alignment": 18.266647445358103,
        "cosyn_point": 20.201689065172104,
        "count_numeric": 3.464101552963257,
        "native_text_replay": 8191.0,
        "ocr_document": 4.189212302095257,
        "pixmo_caption": 29.160138869367074,
        "pixmo_points_basic": 33.28980797799886,
        "pixmo_points_high_frequency": 28.820370947258198,
        "pixmo_transcript": 29.65188992714684,
    },
}
"""Loss-weight calibration at 8,192 tokens for bridge/joint and 2,560 for perception."""


def build_visual_sources(
    phase: str,
    sequence_length: int,
    artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    *,
    split: str = "train",
) -> dict[str, Config]:
    """Construct the default visual mixture without opening images or scanning annotations.

    Prepared selection files retain the image-disjoint train/validation populations. The
    resulting ordinary dataset configs can be replaced or extended by a recipe or CLI
    overrides; the mixture loader has no phase-specific source restrictions.

    :param phase: ``bridge``, ``perception``, or ``joint``.
    :param sequence_length: Maximum serialized example length.
    :param artifact_root: Directory containing the prepared alignment datasets and selections.
    :param split: Logical ``train`` or ``validation`` split.
    """
    if phase not in ALIGNMENT_LOSS_TARGETS:
        raise ValueError(f"Unknown alignment phase {phase!r}")
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown alignment split {split!r}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    root = Path(artifact_root)
    common: dict[str, Any] = {
        "max_crops": 8,
        "max_sequence_length": sequence_length,
        "loss_token_weighting": "root_subsegments_root_tokens",
        "message_format": "document",
        "seed": 0,
    }

    sources: dict[str, Config] = {
        "pixmo_caption": PixMoCapDatasetConfig(
            dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
            split=split,
            require_split=True,
            mode="caption",
            fixed_prompt="Description:",
            style_length_conditioning=False,
            **common,
        ),
        "pixmo_transcript": PixMoCapDatasetConfig(
            dataset_path=str(root / "pixmo-cap-content-disjoint-v1/dataset"),
            split=split,
            require_split=True,
            mode="transcript",
            require_transcript=True,
            fixed_prompt="Transcript:",
            style_length_conditioning=False,
            **common,
        ),
    }
    if phase == "bridge":
        return sources

    selection_root = root / "perception-provenance-v2"
    with (selection_root / "vision-alignment-perception-provenance.json").open() as stream:
        manifest = json.load(stream)
    spec = manifest["source_spec"]
    count_name = "count_numeric" if phase == "joint" else "scalar_count"
    sources.update(
        {
            "pixmo_points_basic": PixMoPointsDatasetConfig(
                require_split=True,
                kind="basic",
                counting=False,
                both_mode="per_annotation",
                **common,
            ),
            "pixmo_points_high_frequency": PixMoPointsDatasetConfig(
                require_split=True,
                kind="high_frequency",
                counting=False,
                both_mode="per_annotation",
                **common,
            ),
            "cosyn_point": CoSynPointDatasetConfig(require_split=True, **common),
            count_name: PixMoCountDatasetConfig(
                require_split=True, mode="scalar_count", counting="both", **common
            ),
            "ocr_document": VisionAlignmentOcrDocumentDatasetConfig(
                source_names=tuple(spec["ocr_source_names"]), **common
            ),
            "audited_alignment": VisionAlignmentAuditedAlignmentDatasetConfig(
                root=spec["finevision_root"],
                visualweb_path=str(
                    root / "finevision-materialization-v1/visualwebinstruct-filtered"
                ),
                geo170k_path=str(root / "finevision-materialization-v1/geo170k-align"),
                visualweb_fingerprint=spec["finevision_visualweb_fingerprint"],
                geo170k_fingerprint=spec["finevision_geo170k_fingerprint"],
                **common,
            ),
        }
    )
    selected: dict[str, Config] = {}
    for name, config in sorted(sources.items()):
        source_name = "scalar_count" if name == "count_numeric" else name
        splits = manifest["sources"][source_name]
        selection = splits[split]
        config.split = selection["physical_split"]  # type: ignore[attr-defined]
        other_split = "validation" if split == "train" else "train"
        other = splits[other_split]
        excluded = []
        if selection["physical_split"] == other["physical_split"]:
            excluded.append(str(selection_root / other["selection"]["path"]))
        selected[name] = MultimodalSourceConfig(
            dataset=config,
            selection_path=str(selection_root / selection["selection"]["path"]),
            excluded_selection_paths=excluded,
        )
    return selected
