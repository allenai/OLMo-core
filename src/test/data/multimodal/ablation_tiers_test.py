"""Per-source ablation tiers must resolve to one dataset and use single-image packing.

These tiers exist to measure what each dataset added since ``image-only-v9`` contributes
on its own, by fine-tuning a fixed v9 checkpoint on exactly one source. Two mistakes are
easy to make and both fail silently rather than loudly, hence these tests:

* registering a v11 source's tier in ``VALIDATION_MIXTURES`` instead of
  ``VALIDATION_MIXTURES_V11`` — ``_build_mixture`` dispatches by dict membership, so it
  would route to the v9 builder and raise at data-build time, deep into a launched job;
* forgetting a ``MIXTURE_PACK_PROFILES`` entry — the tier then silently falls back to the
  *multi-image* profile (125 crops), changing packing density and steps-per-epoch.
"""

from olmo_core.data.multimodal.finevision import (
    FINEVISION_V10_DATASET_NAMES,
    FINEVISION_V11_DATASET_NAMES,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import VALIDATION_MIXTURES_V10
from olmo_core.data.multimodal.mixtures.image_only_v11 import (
    IMAGE_ONLY_V11_NEW_DATASET_NAMES,
    VALIDATION_MIXTURES_V11,
)
from olmo_core.data.multimodal.mixtures.mixture_pack_profiles import (
    PER_SOURCE_ABLATION_TIERS,
    SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
    get_mixture_pack_profile,
)


def test_ablation_tiers_match_the_real_dataset_names():
    # PER_SOURCE_ABLATION_TIERS is a literal list (so mixture_pack_profiles stays
    # dependency-free); this is the guard that keeps it in sync with the source of truth.
    expected = set(FINEVISION_V10_DATASET_NAMES) | set(IMAGE_ONLY_V11_NEW_DATASET_NAMES)
    assert set(PER_SOURCE_ABLATION_TIERS) == expected
    assert len(PER_SOURCE_ABLATION_TIERS) == len(set(PER_SOURCE_ABLATION_TIERS))


def test_every_ablation_tier_is_registered_in_the_right_dict():
    for name in FINEVISION_V10_DATASET_NAMES:
        assert VALIDATION_MIXTURES_V10.get(name) == (name,)
    for name in IMAGE_ONLY_V11_NEW_DATASET_NAMES:
        # Must be in the V11 dict specifically: _build_mixture checks membership, and a
        # v11 name found only in the v9 dict would fall through to the v9 builder.
        assert VALIDATION_MIXTURES_V11.get(name) == (name,)


def test_v11_finevision_tiers_are_not_shadowed_by_v10():
    # v11 adds more finevision_* configs; none may collide with a v10 key, or the v10
    # builder (checked later in the dispatch chain) would win for that name.
    assert not set(FINEVISION_V11_DATASET_NAMES) & set(FINEVISION_V10_DATASET_NAMES)


def test_ablation_tiers_have_single_image_pack_profiles():
    for name in PER_SOURCE_ABLATION_TIERS:
        profile = get_mixture_pack_profile(name)
        assert profile.pack_max_crops == SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS, name
        assert profile.pack_shortcut_max_len_images is True, name


def test_dynamath_is_ablated_as_a_group_not_per_variant():
    # Each dynamath_seed_* variant is 339 rows, fewer than one global batch, so a
    # per-variant tier would yield total_batches == 0 (an empty epoch).
    from olmo_core.data.multimodal.dynamath import DYNAMATH_TRAINING_VARIANTS

    assert VALIDATION_MIXTURES_V10["dynamath"] == DYNAMATH_TRAINING_VARIANTS
    for variant in DYNAMATH_TRAINING_VARIANTS:
        assert variant not in PER_SOURCE_ABLATION_TIERS
