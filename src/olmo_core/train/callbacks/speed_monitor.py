import time
from dataclasses import dataclass
from typing import Any, Dict

import torch

from .callback import Callback


@dataclass
class SpeedMonitorCallback(Callback):
    _total_steps: int = 0
    _total_tokens: int = 0
    _start_time: float = 0.0

    _step_start_time: float = 0.0
    _step_tokens: int = 0

    def pre_train(self):
        self._total_steps = 0
        self._total_tokens = 0
        self._start_time = time.monotonic()

    def pre_step(self, batch: Dict[str, Any]):
        self._step_start_time = time.monotonic()
        self._step_tokens = batch["input_ids"].numel()
        self._total_steps += 1
        self._total_tokens += self._step_tokens

    def post_step(self):
        step_time = time.monotonic() - self._step_start_time
        total_time = time.monotonic() - self._start_time
        self.trainer.record_metric(
            "throughput/total_tokens", torch.tensor(self.trainer.global_train_tokens_seen)
        )
        self.trainer.record_metric(
            "throughput/device/tokens_per_second", torch.tensor(self._step_tokens / step_time)
        )
        self.trainer.record_metric(
            "throughput/device/tokens_per_second.avg", torch.tensor(self._total_tokens / total_time)
        )
        self.trainer.record_metric(
            "throughput/device/batches_per_second", torch.tensor(1 / step_time)
        )
        self.trainer.record_metric(
            "throughput/device/batches_per_second.avg", torch.tensor(self._total_steps / total_time)
        )
