from dataclasses import dataclass
import logging
import torch
from torch import nn
from xlstm.xlstm_large import mLSTMLayer, mLSTMLayerConfig

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
        )).to(init_device)