"""StabilityMonitor callback for detecting training instability."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class StabilityMonitorCallback(Callback):
    """
    Monitors training stability by tracking "spikes" in loss and gradient norm.

    A spike is detected when a value exceeds the running mean of the last ``window_size`` values
    by more than ``threshold_std`` standard deviations. This helps identify training instability.

    Metrics recorded:

    - ``spike/SpikeScore``: Running spike rate over the last ``rolling_window`` steps.
      Only recorded once the rolling window is full.
    - ``spike/SpikeScore (total)``: Cumulative spike rate (total spikes / total steps).
    """

    window_size: int = 128
    """Number of recent values to use for computing mean and std for spike detection."""

    rolling_window: int = 10000
    """Number of recent steps to use for computing running SpikeScore."""

    threshold_std: float = 6.0
    """Number of standard deviations above the mean to consider a spike."""

    enabled: bool = True
    """Whether this callback is enabled."""

    loss_metric_name: str = TRAIN_CE_LOSS_METRIC
    grad_norm_metric_name: str = OPTIM_GRAD_NORM_METRIC

    # Internal state
    _loss_history: List[float] = field(default_factory=list, repr=False)
    _grad_norm_history: List[float] = field(default_factory=list, repr=False)
    _spike_history: List[bool] = field(default_factory=list, repr=False)
    _total_spike_count: int = 0
    _total_step_count: int = 0

    def state_dict(self) -> Dict[str, Any]:
        """Save state for checkpoint resumption."""
        return {
            "loss_history": self._loss_history,
            "grad_norm_history": self._grad_norm_history,
            "spike_history": self._spike_history,
            "total_spike_count": self._total_spike_count,
            "total_step_count": self._total_step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore state from checkpoint."""
        self._loss_history = state_dict.get("loss_history", [])
        self._grad_norm_history = state_dict.get("grad_norm_history", [])
        self._spike_history = state_dict.get("spike_history", [])
        self._total_spike_count = state_dict.get("total_spike_count", 0)
        self._total_step_count = state_dict.get("total_step_count", 0)

    def _append_to_history(self, history: List, value, max_size: int) -> None:
        """Append value to history, removing oldest if over max_size."""
        history.append(value)
        if len(history) > max_size:
            history.pop(0)

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        """Check for spikes and record spike score metrics."""
        if not self.enabled:
            return

        del step  # unused but part of interface

        loss_spike = False
        grad_norm_spike = False

        # Check loss spike (only if we have a full window)
        if self.loss_metric_name in metrics:
            loss_value = metrics[self.loss_metric_name]
            loss_spike = self._is_spike(loss_value, self._loss_history)
            self._append_to_history(self._loss_history, loss_value, self.window_size)

        # Check grad norm spike (only if we have a full window)
        if self.grad_norm_metric_name in metrics:
            grad_norm_value = metrics[self.grad_norm_metric_name]
            grad_norm_spike = self._is_spike(grad_norm_value, self._grad_norm_history)
            self._append_to_history(self._grad_norm_history, grad_norm_value, self.window_size)

        # Determine if this step had any spike
        any_spike = loss_spike or grad_norm_spike
        self._append_to_history(self._spike_history, any_spike, self.rolling_window)
        self._total_step_count += 1
        if any_spike:
            self._total_spike_count += 1
            log.debug(
                f"Spike detected at step: loss_spike={loss_spike}, grad_norm_spike={grad_norm_spike}"
            )

        # Record running SpikeScore (only when rolling window is full)
        if len(self._spike_history) >= self.rolling_window:
            running_spike_score = sum(self._spike_history) / self.rolling_window
            metrics["spike/SpikeScore"] = running_spike_score

        # Record cumulative SpikeScore
        if self._total_step_count >= self.window_size:
            cumulative_spike_score = self._total_spike_count / self._total_step_count
            metrics["spike/SpikeScore (total)"] = cumulative_spike_score

    def _is_spike(self, value: float, history: Sequence[float]) -> bool:
        """
        Check if value is a spike relative to history.

        Returns True if value exceeds mean + threshold_std * std.
        Only checks if history has window_size values.
        """
        if len(history) < self.window_size:
            return False

        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)

        # Avoid numerical issues when std is very small
        if std < 1e-10:
            return False

        threshold = mean + self.threshold_std * std
        return value > threshold
