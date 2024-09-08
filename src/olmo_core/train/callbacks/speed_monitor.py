import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from olmo_core.nn.transformer import Transformer

from .callback import Callback


@dataclass
class SpeedMonitorCallback(Callback):
    """
    Monitors throughput.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    priority: ClassVar[int] = -1

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

    def _get_num_flops_per_token(self, seq_len: int) -> Optional[int]:
        if self.num_flops_per_token is not None:
            return self.num_flops_per_token
        elif isinstance(self.trainer.model, Transformer):
            return self.trainer.model.num_flops_per_token(seq_len)
        else:
            return None

    def pre_train(self):
        self._first_step = True

        if self.device_peak_flops is None and self.trainer.device.type == "cuda":
            device_name = torch.cuda.get_device_name(self.trainer.device)
            if self.trainer.autocast_precision == torch.bfloat16:
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

        self._step_tokens = batch["input_ids"].numel()
        self._step_seq_len = batch["input_ids"].shape[1]
        self._total_steps += 1
        self._total_tokens += self._step_tokens

    def post_step(self):
        counter = time.perf_counter()
        self.trainer.record_metric("throughput/device/data loading (s)", self._batch_load_time)

        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._start_time = counter
            self._step_last_logged = counter
            self._first_step = False
            return

        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        tps = self._step_tokens / step_time
        tps_avg = self._total_tokens / total_time
        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        data_pct = 100 * self._batch_load_time / step_time

        self.trainer.record_metric("throughput/total tokens", self.trainer.global_train_tokens_seen)
        self.trainer.record_metric("throughput/device/data loading (%)", data_pct)
        self.trainer.record_metric("throughput/device/TPS", tps)
        self.trainer.record_metric("throughput/device/TPS (actual avg)", tps_avg)
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

        if (
            num_flops_per_token := self._get_num_flops_per_token(self._step_seq_len)
        ) is not None and self.device_peak_flops is not None:
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            mfu = 100 * num_flops_per_token * tps / self.device_peak_flops
            mfu_avg = 100 * num_flops_per_token * tps_avg / self.device_peak_flops
            self.trainer.record_metric("throughput/device/MFU", mfu)
            self.trainer.record_metric("throughput/device/MFU (actual avg)", mfu_avg)
