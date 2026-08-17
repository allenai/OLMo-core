"""Default stage-2 packing settings per mixture tier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = [
    "MixturePackProfile",
    "MULTI_IMAGE_PACK_MAX_CROPS",
    "SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS",
    "MIXTURE_PACK_PROFILES",
    "get_mixture_pack_profile",
]

# One high-res image: 1 global + up to 24 local crops (mm_olmo pointing/high-res budget).
SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS = 1 + 24

# Worst case for multi-image sources in image-only-v9: 5 images × 25 crops each.
MULTI_IMAGE_PACK_MAX_CROPS = 5 * (1 + 24)


@dataclass(frozen=True)
class MixturePackProfile:
    pack_max_crops: int
    pack_shortcut_max_len_images: bool
    description: str = ""


MIXTURE_PACK_PROFILES: Dict[str, MixturePackProfile] = {
    # Full mixtures include multi-image sources — keep a generous crop ceiling.
    "image-only-v9": MixturePackProfile(
        pack_max_crops=MULTI_IMAGE_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=False,
        description="Full image-only-v9 (includes multi-image sources).",
    ),
    "image-only-v10": MixturePackProfile(
        pack_max_crops=MULTI_IMAGE_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=False,
        description="Full image-only-v10 (v9 + FineVision + DynaMath).",
    ),
    # Single-image tiers match mm_olmo's effective SFT packing (≈25 crops + shortcut).
    "single-image-only-v9": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
        description="image-only-v9 with multi-image sources removed.",
    ),
    "single-image-only-v10": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
        description="image-only-v10 with multi-image v9 sources removed.",
    ),
    # Small debug/demo slices are single-image — use the throughput-friendly profile.
    "debug": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "demo": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "demo-pointing": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "pointing": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "nlp-demo": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "academic": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "finevision": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "dynamath": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "finevision-dynamath": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
}

_DEFAULT_PROFILE = MixturePackProfile(
    pack_max_crops=MULTI_IMAGE_PACK_MAX_CROPS,
    pack_shortcut_max_len_images=False,
    description="Unknown mixture — use conservative multi-image crop ceiling.",
)


def get_mixture_pack_profile(mixture: str) -> MixturePackProfile:
    """Return the recommended pack settings for a mixture tier."""
    return MIXTURE_PACK_PROFILES.get(mixture, _DEFAULT_PROFILE)
