import logging
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GPUMemorySnapshotCallback(Callback):
    """
    Takes a snapshot of the GPU memory usage during the first few training steps.
    To vizualize the snapshot, use https://pytorch.org/memory_viz
    """

    priority: ClassVar[int] = -1
    device_id: Optional[int] = None
    enabled: bool = False
    """
    Set to ``True`` to enable memory snapshots.
    """
    snapshot_steps: int = 3
    """
    Number of steps to record before taking a snapshot.
    """
    max_entries: int = 100_000
    """
    Maximum number of memory events (alloc/free) to record per snapshot.
    """
    record_on_rank: Optional[int] = None
    """
    Which rank to record memory snapshots on. Defaults to rank 0 if not specified.
    """
    _step_count: int = 0
    _recording: bool = False

    @property
    def device(self) -> torch.device:
        return (
            torch.device("cuda")
            if self.device_id is None
            else torch.device(f"cuda:{self.device_id}")
        )

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)

    @property
    def device_capacity(self) -> int:
        return torch.cuda.get_device_properties(self.device).total_memory

    @property
    def recording_rank(self) -> int:
        """The rank that should perform memory recording."""
        return 0 if self.record_on_rank is None else self.record_on_rank

    def pre_train(self):
        if not self.enabled or get_rank() != self.recording_rank:
            return

        if self.trainer.device.type != "cuda":
            log.warning("GPU memory snapshots are only available on CUDA devices")
            self.enabled = False
            return

        self._start_recording()

    def post_step(self):
        if not self.enabled or get_rank() != self.recording_rank or not self._recording:
            return

        self._step_count += 1

        if self._step_count >= self.snapshot_steps:
            self._save_snapshot()
            self._stop_recording()

    def post_train(self):
        if get_rank() == self.recording_rank:
            self._stop_recording()

    def _start_recording(self):
        try:
            torch.cuda.memory._record_memory_history(enabled="all", max_entries=self.max_entries)
            self._recording = True
            self._step_count = 0
            log.info(
                f"Started recording GPU memory history on rank {get_rank()} with max_entries={self.max_entries}"
            )
        except Exception as e:
            log.error(f"Failed to start memory recording: {e}")
            self.enabled = False

    def _save_snapshot(self):
        log.info("Saving GPU memory snapshot...")
        output_dir = self.trainer.work_dir / "memory_snapshots"
        output_dir.mkdir(exist_ok=True, parents=True)
        snapshot_path = (
            output_dir / f"memory_snapshot_rank_{get_rank()}_step_{self.trainer.global_step}.pickle"
        )
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            final_path = self.trainer.persist_working_file(snapshot_path)
            log.info(f"GPU memory snapshot saved to '{final_path}'")
        except Exception as e:
            log.error(f"Failed to capture memory snapshot: {e}")

    def _stop_recording(self):
        if self._recording:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
                self._recording = False
                log.info("Stopped recording GPU memory history")
            except Exception as e:
                log.error(f"Failed to stop memory recording: {e}")
