from dataclasses import dataclass
import logging
import torch
from torch import nn
from xlstm.xlstm_large import mLSTMLayer, mLSTMLayerConfig
from mlstm_kernels.torch.backend_module import mLSTMBackendConfig

from olmo_core.config import Config, DType

log = logging.getLogger(__name__)


class XLSTM(mLSTMLayer):
    def __init__(self, config):
        super().__init__(config)  # type: ignore

        self.xlstm_cache_manager = None

    def init_xlstm_cache_manager(self, batch_size: int):
        self.xlstm_cache_manager = XLSTMCacheManager()

    def forward(self, x: torch.Tensor):  # type: ignore
        if self.training:
            self.mlstm_backend.config.mode = "train"
        else:
            self.mlstm_backend.config.mode = "inference"

        if self.xlstm_cache_manager is not None:
            prev_mode = self.mlstm_backend.config.mode
            state = self.xlstm_cache_manager.state

            h, state = super().forward(x, state)

            self.xlstm_cache_manager.state = state  # type: ignore
            self.mlstm_backend.config.mode = prev_mode

            return h
        else:
            h, _ = super().forward(x)
            return h


@dataclass
class XLSTMConfig(Config):
    num_heads: int
    dtype: DType = DType.float32

    def build(self, d_model: int, init_device) -> XLSTM:
        return XLSTM(mLSTMLayerConfig(
            embedding_dim=d_model,
            num_heads=self.num_heads,
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--triton_limit_chunk",
                sequence_kernel="native_sequence__triton",
                step_kernel="triton",
                mode="train",
                return_last_states=True,
                autocast_kernel_dtype="float32",
            )
        )).to(device=init_device, dtype=self.dtype.as_pt())

    def num_params(self):
        raise NotImplementedError()


class XLSTMCacheManager(nn.Module):
    def __init__(self):
        super().__init__()

        # not designed to be managed externally - cant easily allocate beforehand
        # so we just init to none and let the prefill allocate the state
        self.state = None

    def zero_cache(self):
        raise NotImplementedError()

    def reallocate(self, batch_size: int):
        self.state = None

    def is_reusable(self, batch_size: int) -> bool:
        # not implemented
        return False

    def reset(self, batch_size: int):
        if self.is_reusable(batch_size):
            self.zero_cache()
        else:
            log.debug("Unreusable XLSTM cache, reallocating")
            self.reallocate(batch_size)