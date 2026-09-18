"""Default stage-2 packing settings per mixture tier.

Derived, not tabulated. A tier's crop budget follows from whether the sources it
actually trains on include any multi-image source: those rows cost several images' worth
of crops each, so they need the larger ceiling.

This used to be a hand-maintained ``name -> profile`` dict, and it had already drifted —
``demo``, ``demo-pointing``, ``pointing``, ``nlp-demo`` and ``academic`` were all labelled
single-image while containing sources from :data:`MULTI_IMAGE_MIXTURE_DATASETS`. That
matters because :class:`~olmo_core.data.multimodal.packing.DynamicPacker` emits an example
that alone exceeds a capacity as its own pack rather than enforcing the cap (a documented
deviation from mm_olmo), so an over-budget row passes straight through and the collator
sizes the batch to it. Deriving the profile makes that class of mistake impossible.

Deriving also moves four tiers the *other* way. The single-source pointing bisect tiers
(``pixmo_points_train``, ``pixmo_points_high_freq_train``, ``pixmo_count_train``,
``cosyn_point``) were absent from the old table and fell through to its conservative
default, so they packed at 125 crops with ``shortcut_max_len_images=False`` despite being
single-image. They now get the single-image profile, which is both cheaper and correct —
but it does change how those bisect runs pack, so a before/after comparison of a bisect
result is not apples-to-apples across this change. ``pixmo_multi_points`` stays on the
multi-image profile.

The single-image budget is additionally *tuned*, not just derived: it is the measured
balance point where token and crop occupancy bind at the same time (see
:data:`SINGLE_IMAGE_PACK_MAX_CROPS`). The multi-image budget is deliberately left at its
conservative derived ceiling -- it has never been swept, and a multi-image row costs
several images' worth of crops, so the single-image measurement does not transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    MULTI_IMAGE_MIXTURE_DATASETS,
)
from olmo_core.data.multimodal.mixtures.tiers import mixture_source_names

__all__ = [
    "MixturePackProfile",
    "MULTI_IMAGE_PACK_MAX_CROPS",
    "SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS",
    "SINGLE_IMAGE_PACK_MAX_CROPS",
    "SINGLE_IMAGE_PACK_PROFILE",
    "MULTI_IMAGE_PACK_PROFILE",
    "get_mixture_pack_profile",
    "mixture_is_multi_image",
]

# The *floor*: one high-res image is 1 global + up to 24 local crops (mm_olmo
# pointing/high-res budget). A per-pack budget below this would force a high-res row into
# its own mostly-padding 16k pack, so it is the smallest defensible capacity — not a
# recommendation. This is the value the single-image profile shipped with until the
# crop-budget sweep below.
SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS = 1 + 24

# The shipped single-image budget: the measured balance point, not one example's cost.
#
# A 16,384-token pack at the 25-crop floor is ~69% padding -- the pack fills its crop
# budget (97.5% crop occupancy) long before it fills its token budget (30.6%), so the LM
# runs 3.2x more padded sequences than it needs to for the same ViT work. Total ViT crops
# are exactly invariant across the sweep; only the number of 16k LM sequences changes.
#
# Swept on 8xB300 against `single-image-only-v10` (100-step medians, pad-attention fix in,
# `rank_microbatch_instances=2`, `dl_num_workers=8`), measured in *useful* (non-pad) TPS:
#
#     crops |  25  |  32  |  40  |  50  |  64  |  80  | 100
#     useful TPS | 8,400 | 9,527 | 11,164 | 12,140 | 14,815 | 15,729 | 14,624
#
# 80 is the peak: token occupancy saturates at 98.7% there, and 100 buys no extra
# occupancy while costing step time (6.64s vs 6.17s) -- crop budget with no tokens to
# spend it on is pure cost. Over 2,000 steps this is **1.85x more rows/sec** (100.0 vs
# 54.0 rows/sec at 8 GPUs), convergence-neutral against examples consumed.
#
# TWO THINGS TO KNOW BEFORE CHANGING THIS:
#   * `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is REQUIRED at this budget. See
#     `SHIP_STACK_ENV` in `src/scripts/train/Molmo2-Stage2.py`.
#   * `GLOBAL_BATCH_INSTANCES` counts *packs*, so raising the crop budget raises
#     examples/step at a fixed pack count (~4.1 -> ~13.1 examples/pack). Pick the pack
#     count deliberately; see STAGE2_FAST_CONFIG.md.
#
# `crops=64` (+69.9%) is the lower-risk alternative if you want memory margin.
SINGLE_IMAGE_PACK_MAX_CROPS = 80

# Worst case for multi-image sources in image-only-v9: 5 images × 25 crops each.
#
# NOTE: this is the mm_olmo high-res budget, which is more generous than the per-image
# budget stage 2 actually runs (``MAX_CROPS = 8`` in ``Molmo2-Stage2.py``); a 5-image row
# there costs ~40 crops, not 125. Kept as a conservative ceiling — the packer only needs
# it to be an upper bound — but do not read 125 as a measured figure.
MULTI_IMAGE_PACK_MAX_CROPS = 5 * (1 + 24)


@dataclass(frozen=True)
class MixturePackProfile:
    pack_max_crops: int
    pack_shortcut_max_len_images: bool
    description: str = ""


SINGLE_IMAGE_PACK_PROFILE = MixturePackProfile(
    pack_max_crops=SINGLE_IMAGE_PACK_MAX_CROPS,
    pack_shortcut_max_len_images=True,
    description="single-image sources only (throughput-tuned crop budget)",
)

MULTI_IMAGE_PACK_PROFILE = MixturePackProfile(
    pack_max_crops=MULTI_IMAGE_PACK_MAX_CROPS,
    pack_shortcut_max_len_images=False,
    description="includes multi-image sources — conservative crop ceiling",
)


def mixture_is_multi_image(mixture: str) -> bool:
    """Whether any source the tier trains on is a multi-image source.

    Unknown tiers are treated as multi-image: the conservative ceiling costs throughput,
    the permissive one risks an OOM, so an unrecognized name should fail safe.
    """
    try:
        sources = mixture_source_names(mixture)
    except ValueError:
        return True
    return bool(set(sources) & set(MULTI_IMAGE_MIXTURE_DATASETS))


def get_mixture_pack_profile(mixture: str) -> MixturePackProfile:
    """Return the recommended pack settings for a mixture tier."""
    return (
        MULTI_IMAGE_PACK_PROFILE if mixture_is_multi_image(mixture) else SINGLE_IMAGE_PACK_PROFILE
    )
