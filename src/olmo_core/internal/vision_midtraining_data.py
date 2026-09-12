"""Visual replay and text/vision allocation for mixed midtraining."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from olmo_core.config import Config
from olmo_core.data import NumpyFSLDatasetConfig
from olmo_core.data.multimodal.alignment import (
    MultimodalMixtureConfig,
    MultimodalSourceConfig,
)
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pixmo_points import (
    CoSynPointDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig

from .vision_alignment_data import DEFAULT_ALIGNMENT_ARTIFACT_ROOT
from .vision_alignment_data import build_visual_sources as build_alignment_sources

DEFAULT_MIDTRAINING_ARTIFACT_ROOT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts"
)
TEXT_SOURCE_NAME = "text_midtraining"

DEFAULT_VISUAL_EXAMPLE_WEIGHTS = {
    "pixmo_cap": 0.666666666667,
    "pixmo_points_basic": 0.122201237231,
    "pixmo_count": 0.058384427650,
    "pixmo_points_high_frequency": 0.096695558356,
    "cosyn_point": 0.056052110097,
}
"""Conditional example weights within the caption, pointing and counting group."""

DEFAULT_VISUAL_LOSS_SHARES = {"ocr_document": 8 / 65, "audited_alignment": 5 / 65}
"""Initial replay targets within visual loss mass, retaining alignment's relative shares."""

DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS = {
    "audited_alignment": 104.0703125,
    "cosyn_point": 100.984375,
    "ocr_document": 4.65625,
    "pixmo_cap": 442.71875,
    "pixmo_count": 39.078125,
    "pixmo_points_basic": 345.6015625,
    "pixmo_points_high_frequency": 250.4921875,
}
"""128-example estimates (seed 6198), documented in ``visual_calibration_v1.json``.

Valid for the default 8K/eight-crop document sources with unweighted response loss and
Dolma2 revision ``5292e5d6c0f40b67cc765fe41bec991cf4345b5c``. These are bounded estimates,
not corpus-exact means; recalibrate after changing the tokenizer or source serialization.
"""


def build_visual_sources(
    sequence_length: int = 8192,
    max_crops: int = 8,
    alignment_artifact_root: str = DEFAULT_ALIGNMENT_ARTIFACT_ROOT,
    midtraining_artifact_root: str = DEFAULT_MIDTRAINING_ARTIFACT_ROOT,
    *,
    split: str = "train",
) -> dict[str, Config]:
    """Build caption, pointing/counting, OCR and audited alignment source configs.

    This constructs configs only. OCR and audited sources retain alignment's physical splits,
    quality filters, and train/validation exclusions, but use midtraining's unweighted
    response-token loss. Model-specific image token IDs are supplied by the mixture builder.

    :param sequence_length: Maximum serialized example length.
    :param max_crops: Maximum local image crops.
    :param alignment_artifact_root: Prepared alignment datasets and selection manifests.
    :param midtraining_artifact_root: Prepared grounded-count dataset directory.
    :param split: Logical ``train`` or ``validation`` split.
    """
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be an integer of at least two")
    if type(max_crops) is not int or max_crops < 1:
        raise ValueError("max_crops must be a positive integer")
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown visual split {split!r}")

    common: dict[str, Any] = {
        "split": split,
        "require_split": True,
        "max_crops": max_crops,
        "max_sequence_length": sequence_length,
        "loss_token_weighting": "none",
        "message_format": "document",
        "seed": 0,
    }
    count_common = common.copy()
    count_common["split"] = "grounded_validation" if split == "validation" else "train"
    sources: dict[str, Config] = {
        "pixmo_cap": PixMoCapDatasetConfig(
            dataset_path=str(
                Path(alignment_artifact_root) / "pixmo-cap-content-disjoint-v1/dataset"
            ),
            mode="transcript_and_caption",
            **common,
        ),
        "pixmo_points_basic": PixMoPointsDatasetConfig(
            kind="basic", counting="both", both_mode="duplicate", **common
        ),
        "pixmo_count": PixMoCountDatasetConfig(
            dataset_path=str(
                Path(midtraining_artifact_root) / "pixmo-count-grounded-holdout-v1/dataset"
            ),
            mode="grounded",
            counting="both",
            scalar_count_replay=True,
            **count_common,
        ),
        "pixmo_points_high_frequency": PixMoPointsDatasetConfig(
            kind="high_frequency", counting="both", both_mode="duplicate", **common
        ),
        "cosyn_point": CoSynPointDatasetConfig(**common),
    }
    alignment = build_alignment_sources(
        "joint", sequence_length, alignment_artifact_root, split=split
    )
    for name in DEFAULT_VISUAL_LOSS_SHARES:
        selected = alignment[name]
        assert isinstance(selected, MultimodalSourceConfig)
        source_config: Any = selected.dataset
        source_config.max_crops = max_crops
        source_config.loss_token_weighting = "none"
        sources[name] = selected
    return dict(sorted(sources.items()))


def _validate_text_share(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Text loss mass must be finite and between zero and one")


def _validate_positive_mapping(values: Mapping[str, float], name: str) -> None:
    for source, value in values.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"{name} must use nonempty source names")
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} for {source!r} must be finite and positive")


def loss_mass_targets(
    mean_loss_weight: Mapping[str, float] | None = None,
    *,
    target_text_loss_mass: float = 0.9,
    visual_example_weights: Mapping[str, float] | None = None,
    visual_loss_shares: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Allocate text/vision loss while retaining configurable visual example ratios.

    ``visual_loss_shares`` reserves conditional visual loss for named sources. The remainder
    goes to ``visual_example_weights`` sources, retaining their relative example frequencies.
    These disjoint groups can contain any source names. Returned targets work with the shared
    mixture's ``target / mean`` sampler; zero-mass modalities are omitted.

    :param mean_loss_weight: Mean summed loss weights for the active sources.
    :param target_text_loss_mass: Text's share of expected loss mass, including endpoints.
    :param visual_example_weights: Relative example weights within the remaining visual mass.
    :param visual_loss_shares: Named sources' shares of aggregate visual loss mass.
    """
    _validate_text_share(target_text_loss_mass)
    if target_text_loss_mass == 1:
        return {TEXT_SOURCE_NAME: 1.0}

    examples = dict(
        DEFAULT_VISUAL_EXAMPLE_WEIGHTS if visual_example_weights is None else visual_example_weights
    )
    shares = dict(DEFAULT_VISUAL_LOSS_SHARES if visual_loss_shares is None else visual_loss_shares)
    _validate_positive_mapping(examples, "Visual example weight")
    _validate_positive_mapping(shares, "Visual loss share")
    if set(examples) & set(shares) or TEXT_SOURCE_NAME in examples or TEXT_SOURCE_NAME in shares:
        raise ValueError(
            "Visual weight groups must be disjoint and cannot contain text_midtraining"
        )
    reserved = sum(shares.values())
    if examples:
        if reserved >= 1:
            raise ValueError(
                "Visual loss shares must leave positive mass for example-weighted sources"
            )
    elif not math.isclose(reserved, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Without example-weighted sources, visual loss shares must sum to one")

    means = {} if mean_loss_weight is None else dict(mean_loss_weight)
    expected = set(examples) | set(shares)
    if target_text_loss_mass > 0:
        expected.add(TEXT_SOURCE_NAME)
    if set(means) != expected:
        raise ValueError("Mean loss weights must cover exactly the active weighted sources")
    _validate_positive_mapping(means, "Mean loss weight")

    visual_mass = 1 - target_text_loss_mass
    example_mass = visual_mass * (1 - reserved)
    weighted_means = {name: weight * means[name] for name, weight in examples.items()}
    total = sum(weighted_means.values())
    if examples and (not math.isfinite(total) or total <= 0):
        raise ValueError("Example-weighted visual loss must have a finite positive total")
    targets = {name: example_mass * value / total for name, value in weighted_means.items()}
    targets.update({name: visual_mass * share for name, share in shares.items()})
    if target_text_loss_mass > 0:
        targets[TEXT_SOURCE_NAME] = target_text_loss_mass
    return dict(sorted(targets.items()))


def build_dataset(
    text_dataset: NumpyFSLDatasetConfig,
    mean_loss_weight: Mapping[str, float] | None = None,
    *,
    visual_sources: dict[str, Config] | None = None,
    target_text_loss_mass: float = 0.9,
    visual_example_weights: Mapping[str, float] | None = None,
    visual_loss_shares: Mapping[str, float] | None = None,
    tokenizer_revision: str | None = None,
    tokenizer_cache_dir: str | None = None,
    model_vocab_size: int | None = None,
) -> MultimodalMixtureConfig:
    """Build a calibrated mixed-midtraining data config without preparing datasets.

    Supply the existing 61-source text config explicitly: alignment's pretraining replay
    is not a substitute. No horizon, optimizer, router, or launch policy is set here.
    Calibration must describe these exact source configs, including unweighted OCR loss.

    :param text_dataset: Unchanged fixed-length midtraining text recipe.
    :param mean_loss_weight: Calibrated mean summed loss weight by source. Unmasked text's
        mean defaults to ``sequence_length - 1``; masked text requires a mean in mixed data.
    :param visual_sources: Optional replacement for the default seven visual configs.
    :param target_text_loss_mass: Outer text allocation, defaulting to 90%. One selects
        text-only data without constructing visual sources or requiring visual calibration.
    :param visual_example_weights: Relative example weights within the remaining visual mass.
    :param visual_loss_shares: Named sources' shares of aggregate visual loss mass.
    :param tokenizer_revision: Parent model's tokenizer revision.
    :param tokenizer_cache_dir: Optional tokenizer cache.
    :param model_vocab_size: Model vocabulary including reserved image-token rows.
    """
    if type(text_dataset) is not NumpyFSLDatasetConfig:
        raise TypeError("Mixed midtraining requires an explicit NumpyFSLDatasetConfig")
    _validate_text_share(target_text_loss_mass)
    means = {} if mean_loss_weight is None else dict(mean_loss_weight)
    sources: dict[str, Config] = {}
    if target_text_loss_mass > 0:
        replay = PretrainingReplayConfig(dataset=text_dataset.copy(), split="all")
        replay.resolve_dataset()
        sources[TEXT_SOURCE_NAME] = replay
        if TEXT_SOURCE_NAME not in means:
            if text_dataset.label_mask_paths is None:
                means[TEXT_SOURCE_NAME] = float(text_dataset.sequence_length - 1)
            elif target_text_loss_mass == 1:
                # Singleton sampling needs no calibration; this is only a normalization unit.
                means[TEXT_SOURCE_NAME] = 1.0
            else:
                raise ValueError("Masked mixed text requires an explicit mean loss weight")

    if target_text_loss_mass == 1:
        means = {TEXT_SOURCE_NAME: means[TEXT_SOURCE_NAME]}
    else:
        visual = (
            build_visual_sources(sequence_length=text_dataset.sequence_length)
            if visual_sources is None
            else {name: config.copy() for name, config in visual_sources.items()}
        )
        if TEXT_SOURCE_NAME in visual:
            raise ValueError("Visual sources cannot replace text_midtraining")

        def validate_length(config: Config):
            length = getattr(config, "max_sequence_length", text_dataset.sequence_length)
            if length != text_dataset.sequence_length:
                raise ValueError("Visual and text sequence lengths must agree")

        for source in visual.values():
            source.apply(validate_length)
        sources.update(visual)
    _validate_positive_mapping(means, "Mean loss weight")

    dataset = MultimodalMixtureConfig(
        tokenizer=text_dataset.tokenizer.copy(),
        sources=dict(sorted(sources.items())),
        target_loss_mass=loss_mass_targets(
            means,
            target_text_loss_mass=target_text_loss_mass,
            visual_example_weights=visual_example_weights,
            visual_loss_shares=visual_loss_shares,
        ),
        mean_loss_weight=means,
        tokenizer_revision=tokenizer_revision,
        tokenizer_cache_dir=tokenizer_cache_dir,
        model_vocab_size=model_vocab_size,
    )
    dataset.sampling_weights()
    return dataset
