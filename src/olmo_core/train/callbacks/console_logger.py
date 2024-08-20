import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Dict, List

from .callback import Callback

log = logging.getLogger(__name__)


def format_float(value: float) -> str:
    if value < 0.0001:
        return str(value)  # scientific notation
    elif value > 1000:
        return f"{int(value):,d}"
    elif value > 100:
        return f"{value:.1f}"
    elif value > 10:
        return f"{value:.2f}"
    elif value > 1:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}"


@dataclass
class ConsoleLoggerCallback(Callback):
    log_interval: int = 1
    metrics: List[str] = field(
        default_factory=lambda: ["train/*", "sys/*", "optim/total_grad_norm"]
    )

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if step % self.log_interval != 0:
            return

        prefix = f"[step={step}/{self.trainer.max_steps},epoch={self.trainer.epoch}]"
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
