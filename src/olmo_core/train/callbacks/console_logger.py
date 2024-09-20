import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Dict, List, Optional

from olmo_core.utils import format_float

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ConsoleLoggerCallback(Callback):
    """
    Logs progress and a subset of metrics to the console.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    log_interval: int = 1
    """
    How often, in steps, to log progress to the console.
    """

    metrics_log_interval: Optional[int] = None
    """
    How often, in steps, to log metrics to the console. If not set, defaults to :data:`log_interval`.
    """

    metrics: List[str] = field(
        default_factory=lambda: [
            "train/*",
            "system/*",
            "optim/total grad norm",
            "optim/step skipped",
            "optim/LR*",
            "throughput/*",
        ]
    )
    """
    Metrics to log to the console. Wildcards are supported.
    """

    def post_step(self):
        if self._should_log_metrics(self.step):
            # Will log to console from `self.log_metrics()`.
            return

        if self.step % self.log_interval != 0:
            return

        log.info(self._get_progress_marker(self.step))

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self._should_log_metrics(step):
            return

        prefix = self._get_progress_marker(step)
        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if any(fnmatch(name, pat) for pat in self.metrics)
                ]
            )
        )

    def _get_progress_marker(self, step: int) -> str:
        return f"[step={step}/{self.trainer.max_steps},epoch={self.trainer.epoch}]"

    def _should_log_metrics(self, step: int) -> bool:
        metrics_log_interval = self.metrics_log_interval or self.log_interval
        if step > 1 and step % metrics_log_interval != 0:
            return False
        else:
            return True
