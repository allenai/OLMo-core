"""Resolving a mixture tier name to the sources it actually trains on.

``--mixture=<tier>`` selects both *which* registry to draw from (image-only-v9 or -v10,
full or single-image-only) and *which* sources within it. Several places need that
answer — the training script picks a builder, the pack profile picks a crop budget — and
when each derived it independently they drifted apart.

The concrete symptom: ``mixture_pack_profiles`` was a hand-maintained table that gave
``demo``, ``demo-pointing``, ``pointing``, ``nlp-demo`` and ``academic`` the single-image
25-crop budget even though every one of them contains at least one source from
:data:`MULTI_IMAGE_MIXTURE_DATASETS`. ``DynamicPacker`` emits an over-capacity example as
its own pack rather than enforcing the cap, so those tiers silently exceeded the memory
bound the profile was chosen to provide.

Resolving tiers here once means that class of mistake can't recur.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from olmo_core.data.multimodal.mixture_weights import SubMixture, restrict_submixtures
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    IMAGE_ONLY_V9_SUBMIXTURES,
    SINGLE_IMAGE_ONLY_V9_SUBMIXTURES,
    VALIDATION_MIXTURES,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    IMAGE_ONLY_V10_SUBMIXTURES,
    SINGLE_IMAGE_ONLY_V10_SUBMIXTURES,
    VALIDATION_MIXTURES_V10,
)

__all__ = [
    "all_validation_mixtures",
    "is_v10_mixture",
    "mixture_registry_submixtures",
    "mixture_source_names",
    "mixture_tier_names",
]


def all_validation_mixtures() -> Dict[str, Optional[Tuple[str, ...]]]:
    """Every known tier → its source allowlist (``None`` means the whole registry)."""
    return {**VALIDATION_MIXTURES, **VALIDATION_MIXTURES_V10}


def mixture_tier_names() -> Tuple[str, ...]:
    return tuple(sorted(all_validation_mixtures()))


def is_v10_mixture(mixture: str) -> bool:
    """Whether ``mixture`` draws from the image-only-v10 registry.

    v10 is a superset of v9 (same groups scaled by 0.85, plus FineVision and DynaMath),
    so a tier is only v9 if it isn't named in the v10 table.
    """
    return mixture in VALIDATION_MIXTURES_V10


def mixture_registry_submixtures(mixture: str) -> List[SubMixture]:
    """The unrestricted submixture groups a tier draws from.

    Mirrors the four-way choice the training script makes: v10 vs v9, and full vs
    single-image-only. Both call this rather than branching independently.
    """
    if mixture == "single-image-only-v10":
        return list(SINGLE_IMAGE_ONLY_V10_SUBMIXTURES)
    if is_v10_mixture(mixture):
        return list(IMAGE_ONLY_V10_SUBMIXTURES)
    if mixture == "single-image-only-v9":
        return list(SINGLE_IMAGE_ONLY_V9_SUBMIXTURES)
    return list(IMAGE_ONLY_V9_SUBMIXTURES)


def mixture_source_names(mixture: str) -> Tuple[str, ...]:
    """Source names a tier actually trains on, from metadata only.

    Builds nothing — this reads the submixture structures, so it is safe to call before
    any dataset exists (the pack profile is chosen at config time).

    :raises ValueError: If ``mixture`` is not a known tier.
    """
    tiers = all_validation_mixtures()
    if mixture not in tiers:
        raise ValueError(f"Unknown mixture {mixture!r}; use one of: {', '.join(sorted(tiers))}")

    allowlist: Optional[Sequence[str]] = tiers[mixture]
    groups = restrict_submixtures(mixture_registry_submixtures(mixture), allowlist)
    return tuple(src.name for group in groups for src in group.datasets)
