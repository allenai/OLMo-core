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

    def forward(self, x: torch.Tensor):  # type: ignore
        return super().forward(x)


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
        )).to(init_device)