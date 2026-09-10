"""Tests for mixture tiers and pack profiles."""

from __future__ import annotations

import pytest

from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    IMAGE_ONLY_V9_SUBMIXTURES,
    MULTI_IMAGE_MIXTURE_DATASETS,
    SINGLE_IMAGE_ONLY_V9_SUBMIXTURES,
    VALIDATION_MIXTURES,
    filter_submixtures_single_image,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    SINGLE_IMAGE_ONLY_V10_SUBMIXTURES,
    VALIDATION_MIXTURES_V10,
    image_only_v10_dataset_names,
)
from olmo_core.data.multimodal.mixtures.mixture_pack_profiles import (
    MULTI_IMAGE_PACK_MAX_CROPS,
    SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS,
    get_mixture_pack_profile,
    mixture_is_multi_image,
)
from olmo_core.data.multimodal.mixtures.tiers import (
    all_validation_mixtures,
    mixture_source_names,
)


def test_filter_submixtures_single_image_removes_multi_image_sources():
    filtered = filter_submixtures_single_image(IMAGE_ONLY_V9_SUBMIXTURES)
    names = {src.name for group in filtered for src in group.datasets}
    assert not names.intersection(MULTI_IMAGE_MIXTURE_DATASETS)
    assert "tulu4" in names
    assert "text_vqa" in names


def test_single_image_only_v9_has_fewer_sources_than_full_v9():
    full = {src.name for group in IMAGE_ONLY_V9_SUBMIXTURES for src in group.datasets}
    single = {src.name for group in SINGLE_IMAGE_ONLY_V9_SUBMIXTURES for src in group.datasets}
    assert single < full
    assert len(full) - len(single) == len(MULTI_IMAGE_MIXTURE_DATASETS)


def test_single_image_only_v10_keeps_finevision_and_dynamath():
    names = {src.name for group in SINGLE_IMAGE_ONLY_V10_SUBMIXTURES for src in group.datasets}
    assert any(name.startswith("finevision_") for name in names)
    assert any(name.startswith("dynamath_") for name in names)
    assert len(names) == len(image_only_v10_dataset_names()) - len(MULTI_IMAGE_MIXTURE_DATASETS)


def test_validation_mixtures_include_single_image_tiers():
    assert VALIDATION_MIXTURES["single-image-only-v9"] is None
    assert VALIDATION_MIXTURES_V10["single-image-only-v10"] is None


@pytest.mark.parametrize(
    ("mixture", "pack_max_crops", "shortcut"),
    [
        ("image-only-v9", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("single-image-only-v9", SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS, True),
        ("image-only-v10", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("single-image-only-v10", SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS, True),
        ("debug", SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS, True),
        # Tiers the old hand-maintained table got wrong: each contains at least one
        # source from MULTI_IMAGE_MIXTURE_DATASETS but was labelled single-image.
        ("demo", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("demo-pointing", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("pointing", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("nlp-demo", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("academic", MULTI_IMAGE_PACK_MAX_CROPS, False),
        # Single-source bisect tiers: previously fell through to the conservative
        # default, now correctly identified.
        ("pixmo_multi_points", MULTI_IMAGE_PACK_MAX_CROPS, False),
        ("pixmo_points_train", SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS, True),
        ("cosyn_point", SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS, True),
    ],
)
def test_mixture_pack_profiles(mixture, pack_max_crops, shortcut):
    profile = get_mixture_pack_profile(mixture)
    assert profile.pack_max_crops == pack_max_crops
    assert profile.pack_shortcut_max_len_images is shortcut


def test_every_tier_profile_matches_its_actual_sources():
    """The property the old table violated, checked across every tier at once.

    A tier gets the multi-image crop ceiling if and only if the sources it actually
    trains on include a multi-image source. Deriving this is the point of J-15: a new
    tier, or a source moved between groups, cannot silently get the wrong budget.
    """
    multi_image = set(MULTI_IMAGE_MIXTURE_DATASETS)
    for tier in all_validation_mixtures():
        sources = set(mixture_source_names(tier))
        assert sources, tier
        expected_multi = bool(sources & multi_image)
        profile = get_mixture_pack_profile(tier)
        assert profile.pack_max_crops == (
            MULTI_IMAGE_PACK_MAX_CROPS if expected_multi else SINGLE_IMAGE_HIGH_RES_PACK_MAX_CROPS
        ), f"{tier}: multi-image sources present={sorted(sources & multi_image)}"
        # The shortcut is only safe when every row fits the single-image budget.
        assert profile.pack_shortcut_max_len_images is not expected_multi, tier


def test_unknown_mixture_fails_safe_to_the_conservative_profile():
    """An unrecognized tier must not get the permissive budget: the conservative one
    costs throughput, the permissive one risks an OOM."""
    assert get_mixture_pack_profile("not-a-real-tier").pack_max_crops == MULTI_IMAGE_PACK_MAX_CROPS
    assert mixture_is_multi_image("not-a-real-tier") is True
