import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GPUMemoryMonitorCallback(Callback):
    """
    Adds metrics for GPU memory statistics.
    """

    device_id: Optional[int] = None
    _num_alloc_retries: int = 0

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

    def pre_train(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        log.info(
            f"GPU capacity: {self.device_name} with {self._to_gib(self.device_capacity):.2f}GiB memory"
        )

    def post_step(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)
        self.trainer.record_metric("system/GPU active mem (GiB)", max_active_gib)
        self.trainer.record_metric("system/GPU active mem (%)", max_active_pct)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)
        self.trainer.record_metric("system/GPU reserved mem (GiB)", max_reserved_gib)
        self.trainer.record_metric("system/GPU reserved mem (%)", max_reserved_pct)

        num_retries = cuda_info["num_alloc_retries"]
        if num_retries > self._num_alloc_retries:
            log.warning(f"{num_retries} CUDA memory allocation retries.")
            self._num_alloc_retries = num_retries

        num_ooms = cuda_info["num_ooms"]
        if num_ooms > 0:
            log.warning(f"{num_ooms} CUDA OOM errors thrown.")

        torch.cuda.reset_peak_memory_stats()

    def _to_pct(self, memory: float) -> float:
        return 100 * memory / self.device_capacity

    def _to_gib(self, memory_in_bytes: int) -> float:
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib
