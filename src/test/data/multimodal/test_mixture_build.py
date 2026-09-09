"""Tests for mixture construction startup behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    SINGLE_IMAGE_ONLY_V10_SUBMIXTURES,
    build_single_image_only_v10_mixture,
)


def test_build_single_image_only_v10_mixture_builds_each_source_once(monkeypatch):
    build_calls: list[str] = []

    def fake_build(name, tokenizer, seed, **kwargs):
        build_calls.append(name)
        ds = MagicMock()
        ds.__len__.return_value = 100
        return ds

    monkeypatch.setattr(
        "olmo_core.data.multimodal.mixtures.image_only_v10.build_image_only_v10_dataset",
        fake_build,
    )

    tokenizer = MagicMock()
    datasets, weights, names = build_single_image_only_v10_mixture(tokenizer, seed=0)

    expected = {src.name for group in SINGLE_IMAGE_ONLY_V10_SUBMIXTURES for src in group.datasets}
    assert set(build_calls) == expected
    assert len(build_calls) == len(expected)
    assert len(datasets) == len(weights) == len(names)
    assert set(names) == set(expected)


def test_build_single_image_only_v10_mixture_names_match_weights():
    from olmo_core.data.multimodal.mixtures.image_only_v10 import compute_flat_mixture_weights

    needed = {
        src.name
        for group in SINGLE_IMAGE_ONLY_V10_SUBMIXTURES
        for src in group.datasets
    }
    lengths = {name: 100 for name in needed}
    flat = compute_flat_mixture_weights(SINGLE_IMAGE_ONLY_V10_SUBMIXTURES, lengths)
    assert pytest.approx(sum(w for _, w in flat)) == 1.0
