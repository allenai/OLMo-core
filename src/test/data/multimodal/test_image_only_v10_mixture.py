"""Tests for the image-only-v10 mixture registry."""

from __future__ import annotations

import pytest

from olmo_core.data.multimodal.dynamath import DYNAMATH_TRAINING_VARIANTS
from olmo_core.data.multimodal.finevision import FINEVISION_V10_DATASET_NAMES
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    DYNAMATH_V10_RATE,
    FINEVISION_V10_RATE,
    IMAGE_ONLY_V10_BASE_SCALE,
    IMAGE_ONLY_V10_SUBMIXTURES,
    image_only_v10_dataset_names,
)
from olmo_core.data.multimodal.mixtures.image_only_v9 import IMAGE_ONLY_V9_SUBMIXTURES


def test_image_only_v10_submixture_structure():
    names = [group.name for group in IMAGE_ONLY_V10_SUBMIXTURES]
    assert names == [
        "demo",
        "image_academic",
        "image_pointing",
        "nlp",
        "finevision",
        "dynamath",
    ]


def test_image_only_v10_rates_sum_to_one():
    rates = {group.name: group.rate for group in IMAGE_ONLY_V10_SUBMIXTURES}
    assert rates == {
        "demo": pytest.approx(0.25 * IMAGE_ONLY_V10_BASE_SCALE),
        "image_academic": pytest.approx(0.418 * IMAGE_ONLY_V10_BASE_SCALE),
        "image_pointing": pytest.approx(0.166 * IMAGE_ONLY_V10_BASE_SCALE),
        "nlp": pytest.approx(0.166 * IMAGE_ONLY_V10_BASE_SCALE),
        "finevision": FINEVISION_V10_RATE,
        "dynamath": DYNAMATH_V10_RATE,
    }
    assert sum(rates.values()) == pytest.approx(1.0)


def test_image_only_v10_extends_v9_with_finevision_and_dynamath():
    v9_demo = IMAGE_ONLY_V9_SUBMIXTURES[0].datasets
    v10_demo = IMAGE_ONLY_V10_SUBMIXTURES[0].datasets
    assert v10_demo == v9_demo

    finevision = IMAGE_ONLY_V10_SUBMIXTURES[-2]
    dynamath = IMAGE_ONLY_V10_SUBMIXTURES[-1]
    assert [src.name for src in finevision.datasets] == list(FINEVISION_V10_DATASET_NAMES)
    assert [src.name for src in dynamath.datasets] == list(DYNAMATH_TRAINING_VARIANTS)
    assert all(src.root_size_factor == 0 for src in dynamath.datasets)


def test_image_only_v10_dataset_name_count():
    # v9 has 43 sources; +5 finevision +6 dynamath
    assert len(image_only_v10_dataset_names()) == 54
