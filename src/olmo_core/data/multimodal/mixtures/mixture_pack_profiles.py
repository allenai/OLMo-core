"""Default stage-2 packing settings per mixture tier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = [
    "MixturePackProfile",
    "MULTI_IMAGE_PACK_MAX_CROPS",
    "SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS",
    "MIXTURE_PACK_PROFILES",
    "PER_SOURCE_ABLATION_TIERS",
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


# One tier per dataset added since image-only-v9 (5 v10 FineVision configs + the 11
# v11-only sources). Mirrors FINEVISION_V10_DATASET_NAMES and
# IMAGE_ONLY_V11_NEW_DATASET_NAMES; DynaMath is deliberately absent because its six
# seed variants are ablated together via the existing "dynamath" tier (each variant on
# its own is 339 rows, less than one global batch, i.e. an empty epoch).
PER_SOURCE_ABLATION_TIERS: tuple = (
    # v10-new
    "finevision_densefusion_1m",
    "finevision_objects365_qa",
    "finevision_arxivqa",
    "finevision_geomverse",
    "finevision_doclingmatix",
    # v11-new
    "chartverse",
    "arxivcap",
    "omniscience",
    "vistext",
    "chart2text",
    "finevision_visualwebinstruct",
    "finevision_mavis_math_rule_geo",
    "finevision_mavis_math_metagen",
    "finevision_geo170k_align",
    "finevision_geo170k_qa",
    "mmfinereason",
)


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
    "image-only-v11": MixturePackProfile(
        pack_max_crops=MULTI_IMAGE_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=False,
        description="Full image-only-v11 (v10 + ChartVerse + captions + web reasoning).",
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
    "single-image-only-v11": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
        description="image-only-v11 with multi-image v9 sources removed.",
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
    # v11 sub-tiers: every v11-only source is single-image.
    "chartverse": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "figure-captions": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "web-reasoning": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    "v11-new": MixturePackProfile(
        pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
        pack_shortcut_max_len_images=True,
    ),
    # Per-source ablation tiers (one key per dataset added since image-only-v9), used to
    # measure each source's marginal contribution from a fixed v9 checkpoint. Every one is
    # single-image, so they all take the throughput-friendly profile. Listing the names
    # literally keeps this module dependency-free; `test_ablation_tiers_have_pack_profiles`
    # cross-checks them against the real name tuples so the two cannot drift.
    **{
        name: MixturePackProfile(
            pack_max_crops=SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
            pack_shortcut_max_len_images=True,
            description=f"Single-source ablation: {name} only.",
        )
        for name in PER_SOURCE_ABLATION_TIERS
    },
}

_DEFAULT_PROFILE = MixturePackProfile(
    pack_max_crops=MULTI_IMAGE_PACK_MAX_CROPS,
    pack_shortcut_max_len_images=False,
    description="Unknown mixture — use conservative multi-image crop ceiling.",
)


def get_mixture_pack_profile(mixture: str) -> MixturePackProfile:
    """Return the recommended pack settings for a mixture tier."""
    return MIXTURE_PACK_PROFILES.get(mixture, _DEFAULT_PROFILE)
