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


def test_build_single_image_only_v11_mixture_builds_each_source_once(monkeypatch):
    from olmo_core.data.multimodal.mixtures.image_only_v11 import (
        SINGLE_IMAGE_ONLY_V11_SUBMIXTURES,
        build_single_image_only_v11_mixture,
    )

    build_calls: list[str] = []

    def fake_build(name, tokenizer, seed, **kwargs):
        build_calls.append(name)
        ds = MagicMock()
        ds.__len__.return_value = 100
        return ds

    monkeypatch.setattr(
        "olmo_core.data.multimodal.mixtures.image_only_v11.build_image_only_v11_dataset",
        fake_build,
    )

    tokenizer = MagicMock()
    datasets, weights, names = build_single_image_only_v11_mixture(tokenizer, seed=0)

    expected = {
        src.name for group in SINGLE_IMAGE_ONLY_V11_SUBMIXTURES for src in group.datasets
    }
    assert set(build_calls) == expected
    assert len(build_calls) == len(expected)
    assert len(datasets) == len(weights) == len(names)
    assert pytest.approx(sum(weights)) == 1.0


def test_v11_group_shares_are_sane():
    """Guard the within-group weighting knobs (sampling_rate / root_size_factor)."""
    from olmo_core.data.multimodal.mixture_weights import compute_flat_mixture_weights
    from olmo_core.data.multimodal.mixtures.image_only_v11 import (
        IMAGE_ONLY_V11_SUBMIXTURES,
    )

    # Realistic row counts for the v11-only sources; 100 elsewhere.
    lengths = {
        src.name: 100 for group in IMAGE_ONLY_V11_SUBMIXTURES for src in group.datasets
    }
    lengths.update(
        {
            "chartverse": 250_000,
            "arxivcap": 200_000,
            "omniscience": 150_000,
            "vistext": 9_969,
            "chart2text": 26_961,
            "finevision_visualwebinstruct": 150_000,
            "finevision_mavis_math_rule_geo": 99_986,
            "finevision_mavis_math_metagen": 87_348,
            "finevision_geo170k_align": 35_297,
            "finevision_geo170k_qa": 12_101,
            "mmfinereason": 586_000,
        }
    )
    weights = dict(compute_flat_mixture_weights(IMAGE_ONLY_V11_SUBMIXTURES, lengths))

    web_reasoning = next(g for g in IMAGE_ONLY_V11_SUBMIXTURES if g.name == "web_reasoning")
    group_weights = {src.name: weights[src.name] for src in web_reasoning.datasets}
    web_total = sum(group_weights.values())
    assert web_total == pytest.approx(web_reasoning.rate)

    # MMFineReason is 586k rows: uncapped sqrt(len) would give it ~37% of the group.
    assert group_weights["mmfinereason"] / web_total < 0.25

    # visualwebinstruct is the best-evidenced MMMU-Pro source: it should lead its group.
    assert max(group_weights, key=group_weights.__getitem__) == "finevision_visualwebinstruct"

    # VisText is boosted so its tiny row count still buys real exposure.
    assert weights["vistext"] > weights["chart2text"]
