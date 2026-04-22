"""
Callback for capturing a CUDA memory snapshot at a specific training step.
Useful for diagnosing OOM errors with the PyTorch memory visualizer:
https://pytorch.org/memory_viz
"""

from typing import Any, Dict

import torch

from olmo_core.train.callbacks import Callback


class MemorySnapshotCallback(Callback):
    """Captures a CUDA memory snapshot at a given training step."""

    def __init__(self, out_path: str = "/tmp/memory_snapshot.pickle", capture_step: int = 10):
        self.out_path = out_path
        self.capture_step = capture_step
        self._done = False

    def pre_step(self, batch: Dict[str, Any]):
        del batch
        if not self._done and self.trainer.global_step == self.capture_step:
            torch.cuda.memory._record_memory_history(max_entries=100_000)

    def post_train_batch(self):
        if not self._done and self.trainer.global_step == self.capture_step:
            torch.cuda.memory._dump_snapshot(self.out_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            self._done = True
            print(f"[MemorySnapshotCallback] Snapshot saved to {self.out_path}")
