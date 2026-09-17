"""Restore a metric saver's same-step snapshot before resumed startup evaluation."""

import json
from dataclasses import dataclass
from typing import ClassVar

from olmo_core.distributed.utils import get_rank
from olmo_core.io import file_exists, resource_path

from .callback import Callback


@dataclass
class RestoreMetricsCallback(Callback):
    """Preserve existing metrics when a resumed run only needs to finish evaluation.

    Runs before startup evaluation can overwrite a metric saver's step snapshot. Only
    the restored checkpoint's exact step is loaded; later, unsaved training is excluded.
    """

    metrics_callback: str
    """Name of the attached :class:`MetricSaverCallback` receiving the saved metrics."""

    priority: ClassVar[int] = 4

    def pre_train(self):
        """Merge the existing same-step file into the metric saver on rank zero."""
        if get_rank() != 0 or not self.trainer.checkpoint_loaded:
            return
        callback = self.trainer.callbacks[self.metrics_callback]
        filename = callback.step_metrics_fname.format(step=self.step)
        if file_exists(f"{self.trainer.save_folder}/{filename}"):
            path = resource_path(self.trainer.save_folder, filename)
            callback.log_metrics(self.step, json.loads(path.read_text()))
