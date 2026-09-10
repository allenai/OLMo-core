"""Tests for restricting a mixture to a subset of its sources.

Covers :func:`restrict_submixtures` and the two mixture builders that use it. The
central property is *when* the ``dataset_names`` allowlist is applied: before the
datasets are built (so excluded sources are never constructed) rather than after
the weights are computed (which also skewed the weights — see the module docstring
on ``restrict_submixtures``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import MagicMock

import pytest

from olmo_core.data.multimodal.mixture_weights import (
    SubMixture,
    compute_flat_mixture_weights,
    restrict_submixtures,
)
from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    ACADEMIC_MIXTURE_DATASETS,
    DEBUG_MIXTURE_DATASETS,
    DEMO_MIXTURE_DATASETS,
    IMAGE_ONLY_V9_SUBMIXTURES,
    NLP_MIXTURE_DATASETS,
    POINTING_MIXTURE_DATASETS,
    build_image_only_v9_mixture,
)
from olmo_core.data.multimodal.mixtures.image_only_v10 import (
    IMAGE_ONLY_V10_SUBMIXTURES,
    VALIDATION_MIXTURES_V10,
    build_image_only_v10_mixture,
)

UNIFORM_LEN = 100


def _weights_filter_after(
    groups: Sequence[SubMixture],
    lengths: Dict[str, int],
    allowed: Sequence[str],
) -> Dict[str, float]:
    """The pre-fix ordering: weigh the whole mixture, then drop and renormalize.

    Kept here as a reference implementation so the equivalence assertions below
    test a property rather than a table of magic numbers.
    """
    flat = compute_flat_mixture_weights(list(groups), lengths)
    keep = set(allowed)
    flat = [(name, w) for name, w in flat if name in keep]
    norm = sum(w for _, w in flat)
    return {name: w / norm for name, w in flat}


def _weights_filter_first(
    groups: Sequence[SubMixture],
    lengths: Dict[str, int],
    allowed: Sequence[str],
) -> Dict[str, float]:
    restricted = restrict_submixtures(groups, allowed)
    needed = {src.name for group in restricted for src in group.datasets}
    return dict(compute_flat_mixture_weights(restricted, {n: lengths[n] for n in needed}))


def _uniform_lengths(groups: Sequence[SubMixture]) -> Dict[str, int]:
    return {src.name: UNIFORM_LEN for group in groups for src in group.datasets}


def _groups_touched(groups: Sequence[SubMixture], allowed: Sequence[str]) -> List[SubMixture]:
    keep = set(allowed)
    return [g for g in groups if {s.name for s in g.datasets} & keep]


# --------------------------------------------------------------------------------------
# restrict_submixtures
# --------------------------------------------------------------------------------------


def test_restrict_submixtures_none_is_identity():
    assert restrict_submixtures(IMAGE_ONLY_V9_SUBMIXTURES, None) == list(IMAGE_ONLY_V9_SUBMIXTURES)


def test_restrict_submixtures_drops_emptied_groups_and_keeps_rates():
    restricted = restrict_submixtures(IMAGE_ONLY_V9_SUBMIXTURES, NLP_MIXTURE_DATASETS)
    assert [g.name for g in restricted] == ["nlp"]
    assert restricted[0].rate == 0.166
    assert [s.name for s in restricted[0].datasets] == ["tulu4"]


def test_restrict_submixtures_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="No mixture sources matched"):
        restrict_submixtures(IMAGE_ONLY_V9_SUBMIXTURES, ["not_a_real_source"])


def test_restrict_submixtures_logs_absent_requests(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="olmo_core.data.multimodal.mixture_weights"):
        restrict_submixtures(IMAGE_ONLY_V9_SUBMIXTURES, ["tulu4", "not_a_real_source"])
    assert "not_a_real_source" in caplog.text


# --------------------------------------------------------------------------------------
# Weight equivalence: filtering first must not move weights for whole-group tiers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier,allowed",
    [
        ("demo", DEMO_MIXTURE_DATASETS),
        ("pointing", POINTING_MIXTURE_DATASETS),
        ("academic", ACADEMIC_MIXTURE_DATASETS),
        ("demo-pointing", DEMO_MIXTURE_DATASETS + POINTING_MIXTURE_DATASETS),
        ("nlp-demo", NLP_MIXTURE_DATASETS + DEMO_MIXTURE_DATASETS),
    ],
)
def test_v9_whole_group_tiers_have_unchanged_weights(tier, allowed):
    """A tier that keeps every group it touches whole is unaffected by the reorder.

    ``group.rate * (surviving mass / total mass)`` equals ``group.rate`` when nothing
    in the group was dropped, so both orderings agree exactly.
    """
    del tier
    lengths = _uniform_lengths(IMAGE_ONLY_V9_SUBMIXTURES)
    before = _weights_filter_after(IMAGE_ONLY_V9_SUBMIXTURES, lengths, allowed)
    after = _weights_filter_first(IMAGE_ONLY_V9_SUBMIXTURES, lengths, allowed)

    assert set(before) == set(after)
    for name in before:
        assert after[name] == pytest.approx(before[name]), name

    # Guard the precondition, so this test can't silently start passing vacuously
    # if a tier is later redefined to take only part of a group.
    for group in _groups_touched(IMAGE_ONLY_V9_SUBMIXTURES, allowed):
        assert {s.name for s in group.datasets} <= set(allowed), group.name


@pytest.mark.parametrize("tier", ["finevision", "dynamath", "finevision-dynamath"])
def test_v10_tiers_have_unchanged_weights(tier):
    """The v10 validation tiers each take whole groups, so weights are untouched."""
    allowed = VALIDATION_MIXTURES_V10[tier]
    assert allowed is not None
    lengths = _uniform_lengths(IMAGE_ONLY_V10_SUBMIXTURES)
    before = _weights_filter_after(IMAGE_ONLY_V10_SUBMIXTURES, lengths, allowed)
    after = _weights_filter_first(IMAGE_ONLY_V10_SUBMIXTURES, lengths, allowed)

    assert set(before) == set(after)
    for name in before:
        assert after[name] == pytest.approx(before[name]), name


# --------------------------------------------------------------------------------------
# Weight change: `debug` deliberately moves, and is pinned so it can't drift back
# --------------------------------------------------------------------------------------


def test_debug_tier_is_no_longer_dominated_by_tulu4():
    """``debug`` keeps 2 of 33 ``image_academic`` sources but all of ``nlp``.

    Under the old ordering ``image_academic`` was scaled by its surviving size-mass
    and the tier collapsed onto the one text source. Filtering first gives each
    surviving group its nominal rate.

    The post-fix weights here are length-independent: both surviving ``image_academic``
    sources have no ``root_size_factor``, so their size factors are equal and the group's
    ``0.418`` splits evenly (``0.209`` each) whatever the row counts are, against
    ``nlp``'s ``0.166`` — normalized over ``0.584``.

    The pre-fix weight is *not* length-independent (it depends on how much of the
    academic group's size-mass the two kept sources represent), so it is asserted only
    loosely: the point is that the tier was overwhelmingly ``tulu4``.
    """
    lengths = _uniform_lengths(IMAGE_ONLY_V9_SUBMIXTURES)
    before = _weights_filter_after(IMAGE_ONLY_V9_SUBMIXTURES, lengths, DEBUG_MIXTURE_DATASETS)
    after = _weights_filter_first(IMAGE_ONLY_V9_SUBMIXTURES, lengths, DEBUG_MIXTURE_DATASETS)

    assert before["tulu4"] > 0.85
    assert after["tulu4"] == pytest.approx(0.166 / 0.584, abs=1e-4)
    assert after["tulu4"] == pytest.approx(0.2842, abs=1e-4)
    assert after["text_vqa"] == pytest.approx(after["chart_qa_weighted"])
    assert after["text_vqa"] == pytest.approx(0.3579, abs=1e-4)
    assert sum(after.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Builders: excluded sources must never be constructed
# --------------------------------------------------------------------------------------


def _patch_v10_builder(monkeypatch) -> List[str]:
    built: List[str] = []

    def fake_build(name, tokenizer, seed, **kwargs):
        built.append(name)
        ds = MagicMock()
        ds.__len__.return_value = UNIFORM_LEN
        return ds

    monkeypatch.setattr(
        "olmo_core.data.multimodal.mixtures.image_only_v10.build_image_only_v10_dataset",
        fake_build,
    )
    return built


@pytest.mark.parametrize(
    "tier,expected_sources",
    [("finevision", 5), ("dynamath", 6), ("finevision-dynamath", 11)],
)
def test_v10_tier_builds_only_requested_sources(monkeypatch, tier, expected_sources):
    built = _patch_v10_builder(monkeypatch)
    allowed = VALIDATION_MIXTURES_V10[tier]
    assert allowed is not None

    datasets, weights, names = build_image_only_v10_mixture(
        MagicMock(), seed=0, dataset_names=allowed
    )

    total_sources = len(
        {src.name for group in IMAGE_ONLY_V10_SUBMIXTURES for src in group.datasets}
    )
    assert expected_sources < total_sources  # the fix is meaningful for this tier
    assert set(built) == set(allowed)
    assert len(built) == expected_sources
    assert set(names) == set(allowed)
    assert len(datasets) == len(weights) == len(names) == expected_sources
    assert sum(weights) == pytest.approx(1.0)


def test_v9_tier_builds_only_requested_sources(monkeypatch):
    built: List[str] = []

    def fake_build(name, tokenizer, seed, **kwargs):
        built.append(name)
        ds = MagicMock()
        ds.__len__.return_value = UNIFORM_LEN
        return ds

    monkeypatch.setattr(
        "olmo_core.data.multimodal.mixtures.image_only_v9.build_image_only_v9_dataset",
        fake_build,
    )

    _, weights, names = build_image_only_v9_mixture(
        MagicMock(), seed=0, dataset_names=DEBUG_MIXTURE_DATASETS
    )

    assert set(built) == set(DEBUG_MIXTURE_DATASETS)
    assert set(names) == set(DEBUG_MIXTURE_DATASETS)
    assert sum(weights) == pytest.approx(1.0)


def test_builder_raises_on_unmatched_dataset_names(monkeypatch):
    _patch_v10_builder(monkeypatch)
    with pytest.raises(ValueError, match="No mixture sources matched"):
        build_image_only_v10_mixture(MagicMock(), seed=0, dataset_names=["not_a_real_source"])


def _tier_weight_map(tier_allowed: Sequence[str]) -> List[Tuple[str, float]]:
    lengths = _uniform_lengths(IMAGE_ONLY_V10_SUBMIXTURES)
    restricted = restrict_submixtures(IMAGE_ONLY_V10_SUBMIXTURES, tier_allowed)
    needed = {src.name for group in restricted for src in group.datasets}
    return compute_flat_mixture_weights(restricted, {n: lengths[n] for n in needed})


def test_full_v10_mixture_is_unaffected_by_the_reorder():
    """``dataset_names=None`` must be a pure no-op on the production mixture."""
    lengths = _uniform_lengths(IMAGE_ONLY_V10_SUBMIXTURES)
    direct = dict(compute_flat_mixture_weights(IMAGE_ONLY_V10_SUBMIXTURES, lengths))
    via_restrict = dict(_tier_weight_map(list(direct)))
    assert set(direct) == set(via_restrict)
    for name, weight in direct.items():
        assert via_restrict[name] == pytest.approx(weight), name


# --------------------------------------------------------------------------------------
# Tier resolution must agree with what the builders actually produce
# --------------------------------------------------------------------------------------


def test_tier_source_names_match_what_the_builders_return():
    """`mixture_source_names` must equal the sources a tier really trains on.

    This is the guarantee that makes deriving the pack profile safe: the profile is
    chosen at config time from metadata alone, before any dataset exists, so if that
    metadata view disagreed with the builders the profile would describe a different
    mixture than the one being trained — exactly the drift the old hand-maintained table
    suffered from.
    """
    from unittest.mock import patch

    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        build_single_image_only_v9_mixture,
    )
    from olmo_core.data.multimodal.mixtures.image_only_v10 import (
        build_single_image_only_v10_mixture,
    )
    from olmo_core.data.multimodal.mixtures.tiers import (
        all_validation_mixtures,
        is_v10_mixture,
        mixture_source_names,
    )

    def fake_build(name, tokenizer, seed, **kwargs):
        ds = MagicMock()
        ds.__len__.return_value = UNIFORM_LEN
        return ds

    tiers = all_validation_mixtures()
    with patch(
        "olmo_core.data.multimodal.mixtures.image_only_v10.build_image_only_v10_dataset",
        fake_build,
    ), patch(
        "olmo_core.data.multimodal.mixtures.image_only_v9.build_image_only_v9_dataset",
        fake_build,
    ):
        for tier, allowlist in tiers.items():
            single = tier in ("single-image-only-v9", "single-image-only-v10")
            build: Any
            if is_v10_mixture(tier):
                build = (
                    build_single_image_only_v10_mixture if single else build_image_only_v10_mixture
                )
            else:
                build = (
                    build_single_image_only_v9_mixture if single else build_image_only_v9_mixture
                )
            _, _, names = build(MagicMock(), seed=0, dataset_names=allowlist)
            assert set(names) == set(mixture_source_names(tier)), tier


def test_multi_image_tier_weights_are_pinned():
    """The other tier whose weights the filter-first reorder deliberately moved.

    ``multi-image`` takes a partial slice of three different groups, so unlike the
    whole-group tiers its cross-group balance shifts. Pinned so the change stays
    deliberate; `debug` is pinned separately above.
    """
    from olmo_core.data.multimodal.mixtures.image_only_v9 import (
        MULTI_IMAGE_MIXTURE_DATASETS,
    )

    lengths = _uniform_lengths(IMAGE_ONLY_V9_SUBMIXTURES)
    before = _weights_filter_after(IMAGE_ONLY_V9_SUBMIXTURES, lengths, MULTI_IMAGE_MIXTURE_DATASETS)
    after = _weights_filter_first(IMAGE_ONLY_V9_SUBMIXTURES, lengths, MULTI_IMAGE_MIXTURE_DATASETS)

    assert set(before) == set(after) == set(MULTI_IMAGE_MIXTURE_DATASETS)
    assert sum(after.values()) == pytest.approx(1.0)
    # The post-fix weights are length-independent for this tier: each surviving source is
    # alone in its group, so its within-group share is 1.0 and only the group rates matter.
    assert after["correction_qa_multi_only_max5"] == pytest.approx(0.2998, abs=1e-3)
    assert after["pixmo_multi_points"] == pytest.approx(0.1990, abs=1e-3)

    # The pre-fix weights are *not* length-independent — they depend on how much of each
    # group's size-mass survived — so assert only the qualitative defect: `pixmo_multi_points`
    # carries `root_size_factor=200_000`, which let it dominate the whole tier.
    assert before["pixmo_multi_points"] > 0.7
    assert after["pixmo_multi_points"] < before["pixmo_multi_points"]
    assert after["correction_qa_multi_only_max5"] > before["correction_qa_multi_only_max5"]


def test_production_tiers_are_bit_identical_under_the_reorder():
    """No production mixture changed. Only `debug` and `multi-image` move."""
    from olmo_core.data.multimodal.mixtures.tiers import (
        all_validation_mixtures,
        mixture_registry_submixtures,
    )

    moved = []
    for tier, allowlist in all_validation_mixtures().items():
        if allowlist is None:
            continue  # full registry: the reorder is a no-op by construction
        groups = mixture_registry_submixtures(tier)
        lengths = _uniform_lengths(groups)
        before = _weights_filter_after(groups, lengths, allowlist)
        after = _weights_filter_first(groups, lengths, allowlist)
        if any(abs(after[k] - before[k]) > 1e-9 for k in before):
            moved.append(tier)

    assert sorted(moved) == ["debug", "multi-image"], moved
