import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from olmo_core.config import DType
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
    device_peak_flops_per_second: Optional[int] = None

    _total_steps: int = 0
    _total_tokens: int = 0
    _start_time: float = 0.0
    _first_step: bool = True
    _step_last_logged: float = 0.0
    _batch_load_start: float = 0.0
    _batch_load_time: float = 0.0
    _step_tokens: int = 0
    _step_seq_len: int = 0
    _step_flops: int = 0
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
            self.device_peak_flops_per_second is None
            and self.trainer.device.type == "cuda"
            and isinstance(self.trainer.train_module, TransformerTrainModule)
        ):
            device_name = torch.cuda.get_device_name(self.trainer.device)

            tm = self.trainer.train_module
            using_half_precision = tm.autocast_precision == torch.bfloat16 or (
                tm.dp_config is not None and tm.dp_config.param_dtype == DType.bfloat16
            )
            if using_half_precision:
                dense_correction = 0.5  # listed specs are one-half lower without sparsity
                if "H100" in device_name:
                    # data from https://www.nvidia.com/en-us/data-center/h100/
                    if "NVL" in device_name:
                        self.device_peak_flops_per_second = int(1671e12 * dense_correction)
                    elif "PCIe" in device_name:
                        self.device_peak_flops_per_second = int(1513e12 * dense_correction)
                    else:  # for SXM and other variants
                        self.device_peak_flops_per_second = int(1979e12 * dense_correction)
                elif "B200" in device_name:
                    # data from https://www.nvidia.com/en-us/data-center/hgx/
                    self.device_peak_flops_per_second = int(4.5e15 * dense_correction)
                else:  # for other GPU types, assume A100
                    # data from https://www.nvidia.com/en-us/data-center/a100/
                    self.device_peak_flops_per_second = int(312e12 * dense_correction)
            log.info(
                f"Device: {device_name}, Device peak Flops/s: {self.device_peak_flops_per_second}"
            )

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
            tokens_in_batch = batch["input_ids"].numel()
            self._step_tokens = tokens_in_batch // self._parallel_degree
            self._step_seq_len = batch["input_ids"].shape[1]
            self._total_tokens += self._step_tokens

            self._step_flops = 0
            if (
                num_flops_per_token := self._get_num_flops_per_token(self._step_seq_len)
            ) is not None:
                self._step_flops = num_flops_per_token * tokens_in_batch

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

        if self._step_tokens and self._total_tokens:
            tps = self._step_tokens / step_time
            tps_avg = self._total_tokens / total_time
            self.trainer.record_metric("throughput/device/TPS", tps)
            self.trainer.record_metric("throughput/device/TPS (actual avg)", tps_avg)

        if self.trainer.global_train_tokens_seen is not None:
            self.trainer.record_metric(
                "throughput/total tokens", self.trainer.global_train_tokens_seen
            )

        flops_ps: Optional[float] = None
        flops_ps_avg: Optional[float] = None
        if self._step_flops and self.trainer.global_train_flops:
            flops_ps = self._step_flops / step_time
            flops_ps_avg = self.trainer.global_train_flops / total_time
            self.trainer.record_metric("throughput/device/flopsPS", flops_ps)
            self.trainer.record_metric("throughput/device/flopsPS (actual avg)", flops_ps_avg)
            self.trainer.record_metric("throughput/total flops", self.trainer.global_train_flops)

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
            self.device_peak_flops_per_second is not None
            and flops_ps is not None
            and flops_ps_avg is not None
        ):
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            # MFU is computed from FLOPs/sec. This stays correct even if sequence length changes.
            mfu = 100 * flops_ps / self.device_peak_flops_per_second
            mfu_avg = 100 * flops_ps_avg / self.device_peak_flops_per_second
            self.trainer.record_metric("throughput/device/MFU", mfu)
            self.trainer.record_metric("throughput/device/MFU (actual avg)", mfu_avg)
