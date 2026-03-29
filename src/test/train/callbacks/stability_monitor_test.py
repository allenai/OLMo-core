from unittest.mock import Mock

import pytest

from olmo_core.train.callbacks import StabilityMonitorCallback


def test_spike_detection_and_scoring():
    """Test end-to-end spike detection and score calculation."""
    callback = StabilityMonitorCallback(window_size=5, rolling_window=5, threshold_std=2.0)
    callback._trainer = Mock()

    # Fill window with values that have some variance (needed for spike detection)
    for i in range(5):
        metrics = {"train/CE loss": 1.0 + i * 0.1, "optim/total grad norm": 0.5 + i * 0.01}
        callback.pre_log_metrics(i, metrics)

    # Running SpikeScore should be 0 (no spikes in window)
    assert metrics["spike/SpikeScore"] == 0.0

    # Now trigger a loss spike (100.0 is way above mean ~1.2 + 2*std)
    metrics = {"train/CE loss": 100.0, "optim/total grad norm": 0.5}
    callback.pre_log_metrics(5, metrics)

    # Should have 1 spike out of 6 steps
    assert callback._total_spike_count == 1
    assert metrics["spike/SpikeScore (total)"] == pytest.approx(1 / 6)

    # Trigger a grad norm spike
    metrics = {"train/CE loss": 1.0, "optim/total grad norm": 100.0}
    callback.pre_log_metrics(6, metrics)

    # Should have 2 spikes out of 7 steps
    assert callback._total_spike_count == 2


def test_checkpoint_persistence():
    """Test state survives checkpoint save/load cycle."""
    callback1 = StabilityMonitorCallback(window_size=5, rolling_window=5, threshold_std=2.0)
    callback1._trainer = Mock()

    # Generate some history with variance, then a spike
    for i in range(5):
        callback1.pre_log_metrics(i, {"train/CE loss": 1.0 + i * 0.1})
    callback1.pre_log_metrics(5, {"train/CE loss": 100.0})  # spike

    # Save and restore state
    state = callback1.state_dict()
    callback2 = StabilityMonitorCallback(window_size=5, rolling_window=5, threshold_std=2.0)
    callback2.load_state_dict(state)

    # Verify counts preserved
    assert callback2._total_spike_count == 1
    assert callback2._total_step_count == 6
    assert list(callback2._loss_history) == list(callback1._loss_history)


def test_disabled_callback():
    """Test callback does nothing when disabled."""
    callback = StabilityMonitorCallback(enabled=False)
    callback._trainer = Mock()

    metrics = {"train/CE loss": 100.0}
    callback.pre_log_metrics(0, metrics)

    assert callback._total_step_count == 0
    assert "spike/SpikeScore" not in metrics
    assert "spike/SpikeScore (total)" not in metrics


def test_identical_values_no_crash():
    """Test callback handles identical values (std=0) without dividing by zero."""
    callback = StabilityMonitorCallback(window_size=5, rolling_window=5, threshold_std=2.0)
    callback._trainer = Mock()

    # All identical values - std will be 0
    for i in range(10):
        callback.pre_log_metrics(i, {"train/CE loss": 1.0, "optim/total grad norm": 0.5})

    # Should not crash, and no spikes detected (can't detect with std=0)
    assert callback._total_step_count == 10
    assert callback._total_spike_count == 0
