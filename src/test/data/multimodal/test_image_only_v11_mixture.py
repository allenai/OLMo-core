"""Tests for the image-only-v11 mixture registry."""

from __future__ import annotations

import pytest

from olmo_core.data.multimodal.caption_datasets import CAPTION_DATASET_NAMES
from olmo_core.data.multimodal.chartverse import CHARTVERSE_DATASET_NAME
from olmo_core.data.multimodal.finevision import FINEVISION_V11_DATASET_NAMES
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    MULTI_IMAGE_MIXTURE_DATASETS,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    IMAGE_ONLY_V10_SUBMIXTURES,
    image_only_v10_dataset_names,
)
from olmo_core.data.multimodal.mixtures.image_only_v11 import (
    CHART_REASONING_V11_RATE,
    FIGURE_CAPTIONS_V11_RATE,
    IMAGE_ONLY_V11_BASE_SCALE,
    IMAGE_ONLY_V11_SUBMIXTURES,
    MMFINEREASON_DATASET_NAME,
    SINGLE_IMAGE_ONLY_V11_SUBMIXTURES,
    VALIDATION_MIXTURES_V11,
    WEB_REASONING_V11_RATE,
    image_only_v11_dataset_names,
    image_only_v11_new_dataset_names,
)


def test_image_only_v11_submixture_structure():
    names = [group.name for group in IMAGE_ONLY_V11_SUBMIXTURES]
    assert names == [
        "demo",
        "image_academic",
        "image_pointing",
        "nlp",
        "finevision",
        "dynamath",
        "chart_reasoning",
        "figure_captions",
        "web_reasoning",
    ]


def test_image_only_v11_rates_sum_to_one():
    rates = {group.name: group.rate for group in IMAGE_ONLY_V11_SUBMIXTURES}
    for group in IMAGE_ONLY_V10_SUBMIXTURES:
        assert rates[group.name] == pytest.approx(group.rate * IMAGE_ONLY_V11_BASE_SCALE)
    assert rates["chart_reasoning"] == CHART_REASONING_V11_RATE
    assert rates["figure_captions"] == FIGURE_CAPTIONS_V11_RATE
    assert rates["web_reasoning"] == WEB_REASONING_V11_RATE
    assert sum(rates.values()) == pytest.approx(1.0)


def test_base_scale_matches_new_group_mass():
    new_mass = CHART_REASONING_V11_RATE + FIGURE_CAPTIONS_V11_RATE + WEB_REASONING_V11_RATE
    assert IMAGE_ONLY_V11_BASE_SCALE == pytest.approx(1.0 - new_mass)


def test_image_only_v11_new_groups_membership():
    groups = {group.name: group for group in IMAGE_ONLY_V11_SUBMIXTURES}
    assert [src.name for src in groups["chart_reasoning"].datasets] == [CHARTVERSE_DATASET_NAME]
    assert [src.name for src in groups["figure_captions"].datasets] == list(CAPTION_DATASET_NAMES)
    web = [src.name for src in groups["web_reasoning"].datasets]
    assert web == list(FINEVISION_V11_DATASET_NAMES) + [MMFINEREASON_DATASET_NAME]


def test_mmfinereason_is_size_capped():
    """586k rows would take ~37% of its group under plain sqrt(len) weighting."""
    groups = {group.name: group for group in IMAGE_ONLY_V11_SUBMIXTURES}
    mmfr = next(
        src for src in groups["web_reasoning"].datasets if src.name == MMFINEREASON_DATASET_NAME
    )
    assert mmfr.root_size_factor == 50_000


def test_image_only_v11_dataset_name_count():
    # v10 has 54 sources; +1 chartverse +4 captions +5 finevision-v11 +1 mmfinereason
    assert len(image_only_v11_dataset_names()) == 65
    assert len(image_only_v11_dataset_names()) == len(image_only_v10_dataset_names()) + 11


def test_image_only_v11_new_dataset_names():
    names = image_only_v11_new_dataset_names()
    assert len(names) == 11
    assert CHARTVERSE_DATASET_NAME in names
    assert set(CAPTION_DATASET_NAMES) <= set(names)
    assert set(FINEVISION_V11_DATASET_NAMES) <= set(names)
    assert MMFINEREASON_DATASET_NAME in names
    # v11-only means exactly that: no overlap with the v10 source set.
    assert not (set(names) & set(image_only_v10_dataset_names()))


def test_single_image_tier_drops_only_the_v9_multi_image_sources():
    """Every v11-only source is single-image, so the delta is unchanged from v10."""
    names = {src.name for group in SINGLE_IMAGE_ONLY_V11_SUBMIXTURES for src in group.datasets}
    assert len(names) == len(image_only_v11_dataset_names()) - len(MULTI_IMAGE_MIXTURE_DATASETS)
    assert set(image_only_v11_new_dataset_names()) <= names


def test_validation_mixtures_v11_keys_do_not_collide():
    from olmo_core.data.multimodal.mixtures.image_only_v9 import VALIDATION_MIXTURES
    from olmo_core.data.multimodal.mixtures.image_only_v10 import (
        VALIDATION_MIXTURES_V10,
    )

    # Molmo2-Stage2 merges these three dicts; a collision would silently mis-route a
    # tier to the wrong builder.
    assert not (set(VALIDATION_MIXTURES_V11) & set(VALIDATION_MIXTURES_V10))
    assert not (set(VALIDATION_MIXTURES_V11) & set(VALIDATION_MIXTURES))

    assert VALIDATION_MIXTURES_V11["image-only-v11"] is None
    assert VALIDATION_MIXTURES_V11["single-image-only-v11"] is None
    assert VALIDATION_MIXTURES_V11["figure-captions"] == CAPTION_DATASET_NAMES
