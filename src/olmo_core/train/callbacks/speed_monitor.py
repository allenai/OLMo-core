import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from olmo_core.distributed.utils import get_world_size

from ..common import ReduceType
from ..train_module import TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitorCallback(Callback):
    """
    Monitors throughput.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    priority: ClassVar[int] = -2

    num_flops_per_token: Optional[int] = None
    device_peak_flops: Optional[int] = None

    _total_steps: int = 0
    _total_tokens: int = 0
    _start_time: float = 0.0
    _first_step: bool = True
    _step_last_logged: float = 0.0
    _batch_load_start: float = 0.0
    _batch_load_time: float = 0.0
    _step_tokens: int = 0
    _step_seq_len: int = 0
    _parallel_degree: int = 1
    _bps_avg: Optional[float] = None

    def reset(self):
        self._first_step = True
        self._bps_avg = None

    @property
    def bps_avg(self) -> Optional[float]:
        return self._bps_avg

    def _get_num_flops_per_token(self, seq_len: int) -> Optional[int]:
        if self.num_flops_per_token is not None:
            return self.num_flops_per_token
        elif isinstance(self.trainer.train_module, TransformerTrainModule):
            return self.trainer.train_module.num_flops_per_token(seq_len)
        else:
            return None

    def pre_train(self):
        self._first_step = True

        if self.trainer.dp_process_group is not None:
            self._parallel_degree = get_world_size() // get_world_size(
                self.trainer.dp_process_group
            )

        if (
            self.device_peak_flops is None
            and self.trainer.device.type == "cuda"
            and isinstance(self.trainer.train_module, TransformerTrainModule)
        ):
            device_name = torch.cuda.get_device_name(self.trainer.device)
            if self.trainer.train_module.autocast_precision == torch.bfloat16:
                if "A100" in device_name:
                    self.device_peak_flops = int(312e12)
                elif "H100" in device_name:
                    # data from https://www.nvidia.com/en-us/data-center/h100/
                    # NOTE: Specifications are one-half lower without sparsity.
                    if "NVL" in device_name:
                        self.device_peak_flops = int(1979e12)
                    elif "PCIe" in device_name:
                        self.device_peak_flops = int(756e12)
                    else:  # for SXM and other variants
                        self.device_peak_flops = int(989e12)
                elif "B200" in device_name:
                    self.device_peak_flops = int(2200e12)
                else:  # for other GPU types, assume A100
                    self.device_peak_flops = int(312e12)

    def pre_load_batch(self):
        self._batch_load_start = time.perf_counter()

    def pre_step(self, batch: Dict[str, Any]):
        self._batch_load_time = time.perf_counter() - self._batch_load_start

        if self._first_step:
            # We don't record the first batch since the first one tends to take
            # unusually long.
            return

        self._total_steps += 1
        if "input_ids" in batch:
            self._step_tokens = batch["input_ids"].numel() // self._parallel_degree
            self._step_seq_len = batch["input_ids"].shape[1]
            self._total_tokens += self._step_tokens

    def post_step(self):
        counter = time.perf_counter()
        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time, reduce_type=ReduceType.max
        )

        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._start_time = counter
            self._first_step = False
            self._step_last_logged = counter
            return

        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        device_tps: Optional[float] = None
        device_tps_avg: Optional[float] = None
        if self._step_tokens and self._total_tokens:
            device_tps = self._step_tokens / step_time
            device_tps_avg = self._total_tokens / total_time
            self.trainer.record_metric("throughput/device/TPS", device_tps)
            self.trainer.record_metric("throughput/device/TPS (actual avg)", device_tps_avg)

        if self.trainer.global_train_tokens_seen is not None:
            self.trainer.record_metric(
                "throughput/total tokens", self.trainer.global_train_tokens_seen
            )
            global_tps = self.trainer.global_train_tokens_seen / total_time
            self.trainer.record_metric("throughput/TPS (actual avg)", global_tps)

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        self._bps_avg = bps_avg
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

        data_pct = 100 * self._batch_load_time / step_time
        self.trainer.record_metric(
            "throughput/device/data loading (%)", data_pct, reduce_type=ReduceType.max
        )

        if (
            (num_flops_per_token := self._get_num_flops_per_token(self._step_seq_len)) is not None
            and self.device_peak_flops is not None
            and device_tps is not None
            and device_tps_avg is not None
        ):
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            mfu = 100 * num_flops_per_token * device_tps / self.device_peak_flops
            mfu_avg = 100 * num_flops_per_token * device_tps_avg / self.device_peak_flops
            self.trainer.record_metric("throughput/device/MFU", mfu)
            self.trainer.record_metric("throughput/device/MFU (actual avg)", mfu_avg)
        else:
            # Log why MFU is not being recorded
            if num_flops_per_token is None:
                log.info("MFU not recorded: num_flops_per_token is None")
            if self.device_peak_flops is None:
                log.info("MFU not recorded: device_peak_flops is None")
            if device_tps is None:
                log.info("MFU not recorded: device_tps is None")
            if device_tps_avg is None:
                log.info("MFU not recorded: device_tps_avg is None")
