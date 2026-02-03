"""Unit tests for ModelMergeCallback."""

import logging

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import ModelMergeCallback


def test_merge_last_n_steps_zero_raises_error():
    """Test that merge_last_n_steps=0 raises OLMoConfigurationError."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=100, merge_last_n_steps=0)


def test_merge_last_n_steps_negative_raises_error():
    """Test that negative merge_last_n_steps raises OLMoConfigurationError."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=100, merge_last_n_steps=-10)


def test_merge_step_zero_raises_error():
    """Test that merge_step=0 raises OLMoConfigurationError."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=0, merge_last_n_steps=10)


def test_merge_step_negative_raises_error():
    """Test that negative merge_step raises OLMoConfigurationError."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=-5, merge_last_n_steps=10)


def test_merge_step_list_with_invalid_raises_error():
    """Test that merge_step list containing invalid values raises OLMoConfigurationError."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        ModelMergeCallback(merge_step=[100, -5, 200], merge_last_n_steps=10)


def test_warn_for_truncated_windows(caplog):
    """Test that truncated windows produce a warning."""
    callback = ModelMergeCallback(merge_step=100, merge_last_n_steps=50)
    callback._merge_steps = [30, 100]  # 30 < 50, so window will be truncated

    with caplog.at_level(logging.WARNING):
        callback._warn_for_truncated_windows()

    assert "truncated" in caplog.text.lower()
    assert "30" in caplog.text


def test_overlapping_merge_windows_raises_error():
    """Test that overlapping merge windows raise OLMoConfigurationError."""
    callback = ModelMergeCallback(
        merge_step=[100, 150],  # 50 step gap
        merge_last_n_steps=100,  # 100 step window (does overlap)
    )
    # Manually set merge_steps as if _compute_merge_steps had run
    callback._merge_steps = [100, 150]

    with pytest.raises(OLMoConfigurationError, match="Merge windows would overlap"):
        callback._check_for_overlapping_merge_windows()


def test_non_overlapping_merge_windows_ok():
    """Test that non-overlapping merge windows don't raise error."""
    callback = ModelMergeCallback(
        merge_step=[100, 250],  # 150 step gap
        merge_last_n_steps=100,  # 100 step window (no overlap)
    )
    # Manually set merge_steps as if _compute_merge_steps had run
    callback._merge_steps = [100, 250]

    # Should not raise
    callback._check_for_overlapping_merge_windows()


def test_merge_windows_exact_boundary_ok():
    """Test that merge windows at exact boundary (gap == merge_last_n_steps) don't raise error."""
    callback = ModelMergeCallback(
        merge_step=[100, 200],  # 100 step gap
        merge_last_n_steps=100,  # 100 step window - exactly at boundary
    )
    # Manually set merge_steps as if _compute_merge_steps had run
    callback._merge_steps = [100, 200]

    # Should not raise - gap equals merge_last_n_steps is ok
    callback._check_for_overlapping_merge_windows()


def test_single_merge_step_no_overlap_check():
    """Test that single merge step doesn't trigger overlap check."""
    callback = ModelMergeCallback(merge_step=100, merge_last_n_steps=100)
    callback._merge_steps = [100]

    # Should not raise - nothing to compare
    callback._check_for_overlapping_merge_windows()
