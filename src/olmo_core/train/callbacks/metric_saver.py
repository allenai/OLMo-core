import dataclasses
import json
import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MetricSaverCallback(Callback):
    """
    A callback that captures the latest metrics on rank 0 and saves to a JSON file in the trainer's
    ``save_folder``.
    """

    step_metrics_fname: str = "metrics_step{step}.json"
    """The filename to save the step metrics to, with ``{step}`` as a placeholder for the step number."""
    final_metrics_fname: str = "metrics.json"
    """The filename to save the final metrics to."""
    metrics_to_capture: Optional[List[str]] = None
    """
    An optional list of glob patterns to filter which metrics to capture.
    If ``None``, all metrics are captured.
    """
    save_interval: Optional[int] = None
    """An optional interval (in steps) at which to save the metrics."""
    fixed_steps: Optional[List[int]] = None
    """An optional list of fixed steps at which to save the metrics."""
    enabled: bool = True

    _metrics: Optional[Dict[str, Any]] = dataclasses.field(default=None, repr=False)
    _metrics_step: int = dataclasses.field(default=0, repr=False)

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """
        The latest metrics recorded.
        """
        return self._metrics

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self.enabled or get_rank() != 0:
            return

        if self._metrics is None:
            self._metrics = {}

        if step >= self._metrics_step:
            if self.metrics_to_capture is not None:
                metrics = {
                    k: v
                    for k, v in metrics.items()
                    if any(fnmatch(k, pattern) for pattern in self.metrics_to_capture)
                }
            self._metrics.update(metrics)
            self._metrics_step = step

        if (self.save_interval is not None and step % self.save_interval == 0) or (
            self.fixed_steps is not None and step in self.fixed_steps
        ):
            dest_path = self._write_metrics(
                self.step_metrics_fname.format(step=step), self._metrics
            )
            log.info(f"Metrics for step {step} saved to '{dest_path}'")

    def close(self):
        if not self.enabled or get_rank() != 0:
            return

        if self.metrics is not None:
            dest_path = self._write_metrics(self.final_metrics_fname, self.metrics)
            log.info(f"Final metrics from step {self._metrics_step} saved to '{dest_path}'")

        self._metrics = None
        self._metrics_step = 0

    def _write_metrics(self, fname: str, metrics: Dict[str, float]) -> PathOrStr:
        return self.trainer.write_file(fname, json.dumps(metrics))
