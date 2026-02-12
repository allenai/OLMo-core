"""Unit tests for ModelMergeCallback."""

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import ModelMergeCallback
from olmo_core.train.callbacks.model_merger import (
    compute_merge_steps_from_wsds,
    compute_merge_window_starts,
)


# ============================================================================
# merge_step validation
# ============================================================================


@pytest.mark.parametrize("value", [0, -10])
def test_merge_last_n_steps_non_positive_raises_error(value):
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=100, merge_last_n_steps=value)


@pytest.mark.parametrize("value", [0, -5])
def test_merge_step_non_positive_raises_error(value):
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=value, merge_last_n_steps=10)


def test_merge_step_list_with_invalid_raises_error():
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=[100, -5, 200], merge_last_n_steps=10)


def test_empty_merge_step_raises_error():
    with pytest.raises(OLMoConfigurationError, match="merge_step or merge_interval must be set"):
        ModelMergeCallback(merge_step=[], merge_last_n_steps=10)


@pytest.mark.parametrize(
    "merge_step, merge_last_n_steps",
    [
        ([100, 250], 100),  # non-overlapping
        ([100, 200], 100),  # exact boundary
        (100, 100),         # single step
        ([100, 150], 100),  # overlapping windows (50 step gap < 100 window)
        ([100, 110, 120], 50),  # multiple overlapping windows
    ],
)
def test_merge_windows_ok(merge_step, merge_last_n_steps):
    ModelMergeCallback(merge_step=merge_step, merge_last_n_steps=merge_last_n_steps)


# ============================================================================
# merge_interval validation
# ============================================================================


@pytest.mark.parametrize("value", [0, -5])
def test_merge_interval_non_positive_raises_error(value):
    with pytest.raises(OLMoConfigurationError, match="merge_interval must be positive"):
        ModelMergeCallback(merge_interval=value, merge_last_n_steps=10)


def test_merge_interval_and_merge_step_both_set_raises_error():
    with pytest.raises(OLMoConfigurationError, match="Cannot set both"):
        ModelMergeCallback(merge_step=100, merge_interval=500, merge_last_n_steps=10)


def test_merge_interval_valid():
    cb = ModelMergeCallback(merge_interval=500, merge_last_n_steps=100)
    assert cb.merge_interval == 500
    assert cb._merge_steps == []  # deferred to pre_train


# ============================================================================
# Helper functions
# ============================================================================


def test_compute_merge_window_starts():
    # Non-overlapping: each gets its own checkpoint
    assert compute_merge_window_starts([500, 1000], 100) == [401, 901]
    # Clamp to 0
    assert compute_merge_window_starts([50], 100) == [0]
    # Overlapping: only the earliest start in the group
    # merge at 5000 (window 4501-5000) and 5200 (window 4701-5200)
    # 4701 <= 5000, so only 4501 is needed
    assert compute_merge_window_starts([5000, 5200], 500) == [4501]
    # Three overlapping windows
    assert compute_merge_window_starts([100, 150, 200], 100) == [1]
    # Mixed: first two overlap, third is separate
    assert compute_merge_window_starts([500, 550, 1000], 100) == [401, 901]


def test_compute_merge_steps_from_wsds_with_fixed_decay():
    # 2 periods of 10000 tokens, batch size 100, decay 1000 tokens
    steps = compute_merge_steps_from_wsds(
        period_lengths=[10000, 10000],
        tokens_per_step=100,
        decay=1000,
    )
    # Period 1: 10000 tokens, pre-decay at 10000-1000=9000, step=90
    # Period 2: 20000 tokens, pre-decay at 20000-1000=19000, step=190
    assert steps == [90, 190]


def test_compute_merge_steps_from_wsds_with_decay_fraction():
    steps = compute_merge_steps_from_wsds(
        period_lengths=[10000],
        tokens_per_step=100,
        decay_fraction=0.1,
    )
    # 10000 tokens, decay=10000*0.1=1000, pre-decay at 9000, step=90
    assert steps == [90]


def test_compute_merge_steps_from_wsds_requires_decay():
    with pytest.raises(ValueError, match="Either decay or decay_fraction"):
        compute_merge_steps_from_wsds(period_lengths=[10000], tokens_per_step=100)
