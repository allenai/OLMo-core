"""image-only-v10 mixture registry for Molmo2 SFT.

Extends the richer OLMo-core ``image-only-v9`` recipe (includes mantis, multidoc,
``correction_qa``, ``pixmo_multi_points``) with hub-backed FineVision (10%) and
local DynaMath (5%). Existing v9 buckets are scaled by 0.85 to keep the total at 1.0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from olmo_core.data.multimodal.dynamath import (
    DYNAMATH_TRAINING_VARIANTS,
    DynaMathDatasetConfig,
    dynamath_variant_from_name,
)
from olmo_core.data.multimodal.finevision import (
    FINEVISION_V10_DATASET_NAMES,
    build_finevision_v10_config,
    finevision_v10_hub_name,
)
from olmo_core.data.multimodal.mixture_weights import DatasetSource, SubMixture
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    IMAGE_ONLY_V9_SUBMIXTURES,
    build_image_only_v9_dataset,
    filter_submixtures_single_image,
)
from olmo_core.data.multimodal.mixtures.registry import (
    LazyDatasetMap,
    build_mixture,
    mixture_dataset_names,
    source_lookup,
)

__all__ = [
    "IMAGE_ONLY_V10_BASE_SCALE",
    "IMAGE_ONLY_V10_SUBMIXTURES",
    "SINGLE_IMAGE_ONLY_V10_SUBMIXTURES",
    "VALIDATION_MIXTURES_V10",
    "build_image_only_v10_dataset",
    "build_image_only_v10_datasets",
    "build_image_only_v10_mixture",
    "build_single_image_only_v10_mixture",
    "image_only_v10_dataset_names",
]

# 1 - finevision(0.10) - dynamath(0.05)
IMAGE_ONLY_V10_BASE_SCALE = 0.85

FINEVISION_V10_RATE = 0.10
DYNAMATH_V10_RATE = 0.05

IMAGE_ONLY_V10_SUBMIXTURES: List[SubMixture] = [
    SubMixture(
        group.name,
        group.rate * IMAGE_ONLY_V10_BASE_SCALE,
        list(group.datasets),
    )
    for group in IMAGE_ONLY_V9_SUBMIXTURES
] + [
    SubMixture(
        "finevision",
        FINEVISION_V10_RATE,
        [DatasetSource(name) for name in FINEVISION_V10_DATASET_NAMES],
    ),
    SubMixture(
        "dynamath",
        DYNAMATH_V10_RATE,
        [DatasetSource(name, root_size_factor=0) for name in DYNAMATH_TRAINING_VARIANTS],
    ),
]

SINGLE_IMAGE_ONLY_V10_SUBMIXTURES: List[SubMixture] = filter_submixtures_single_image(
    IMAGE_ONLY_V10_SUBMIXTURES
)

VALIDATION_MIXTURES_V10: Dict[str, Optional[Tuple[str, ...]]] = {
    "image-only-v10": None,
    "single-image-only-v10": None,
    "finevision": FINEVISION_V10_DATASET_NAMES,
    "dynamath": DYNAMATH_TRAINING_VARIANTS,
    "finevision-dynamath": FINEVISION_V10_DATASET_NAMES + DYNAMATH_TRAINING_VARIANTS,
}


def _source_lookup() -> Dict[str, DatasetSource]:
    return source_lookup(IMAGE_ONLY_V10_SUBMIXTURES)


def image_only_v10_dataset_names() -> List[str]:
    """All dataset names in the image-only-v10 mixture."""
    return mixture_dataset_names(IMAGE_ONLY_V10_SUBMIXTURES)


def build_image_only_v10_dataset(
    name: str,
    tokenizer,
    seed: int = 0,
    *,
    max_sequence_length: Optional[int] = None,
    finevision_cache_dir: Optional[str] = None,
):
    """Build a single image-only-v10 dataset by mixture source name."""
    if name in FINEVISION_V10_DATASET_NAMES:
        hub_name = finevision_v10_hub_name(name)
        kw = dict(seed=seed, cache_dir=finevision_cache_dir)
        if max_sequence_length is not None:
            kw["max_sequence_length"] = max_sequence_length
        return build_finevision_v10_config(hub_name, **kw).build(tokenizer)

    if name in DYNAMATH_TRAINING_VARIANTS:
        variant = dynamath_variant_from_name(name)
        kw = dict(variant=variant, seed=seed)
        if max_sequence_length is not None:
            kw["max_sequence_length"] = max_sequence_length
        return DynaMathDatasetConfig(**kw).build(tokenizer)

    return build_image_only_v9_dataset(
        name,
        tokenizer,
        seed,
        max_sequence_length=max_sequence_length,
    )


def _v10_dataset_builder(
    tokenizer,
    seed: int,
    max_sequence_length: Optional[int],
    finevision_cache_dir: Optional[str],
):
    """Bind ``build_image_only_v10_dataset`` to everything but the source name."""

    def build(name: str):
        return build_image_only_v10_dataset(
            name,
            tokenizer,
            seed,
            max_sequence_length=max_sequence_length,
            finevision_cache_dir=finevision_cache_dir,
        )

    return build


def build_image_only_v10_datasets(
    tokenizer,
    seed: int = 0,
    *,
    max_sequence_length: Optional[int] = None,
    finevision_cache_dir: Optional[str] = None,
) -> LazyDatasetMap:
    """Lazy registry of all image-only-v10 datasets keyed by mixture source name."""
    return LazyDatasetMap(
        IMAGE_ONLY_V10_SUBMIXTURES,
        _v10_dataset_builder(tokenizer, seed, max_sequence_length, finevision_cache_dir),
    )


def build_image_only_v10_mixture(
    tokenizer,
    seed: int = 0,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    max_sequence_length: int = 16384,
    finevision_cache_dir: Optional[str] = None,
    submixtures: Optional[Sequence[SubMixture]] = None,
) -> Tuple[List, List[float], List[str]]:
    """Build weighted datasets for :class:`~olmo_core.data.multimodal.MixtureDataLoader`.

    ``dataset_names`` restricts the submixtures *before* any dataset is built, so a tier
    such as ``--mixture=finevision`` constructs 5 sources rather than all 54. See
    :func:`~olmo_core.data.multimodal.mixtures.registry.build_mixture`.
    """
    return build_mixture(
        IMAGE_ONLY_V10_SUBMIXTURES,
        _v10_dataset_builder(tokenizer, seed, max_sequence_length, finevision_cache_dir),
        submixtures=submixtures,
        dataset_names=dataset_names,
    )


def build_single_image_only_v10_mixture(
    tokenizer,
    seed: int = 0,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    max_sequence_length: int = 16384,
    finevision_cache_dir: Optional[str] = None,
) -> Tuple[List, List[float], List[str]]:
    """Build image-only-v10 with multi-image v9 sources removed."""
    return build_image_only_v10_mixture(
        tokenizer,
        seed,
        dataset_names=dataset_names,
        max_sequence_length=max_sequence_length,
        finevision_cache_dir=finevision_cache_dir,
        submixtures=SINGLE_IMAGE_ONLY_V10_SUBMIXTURES,
    )
