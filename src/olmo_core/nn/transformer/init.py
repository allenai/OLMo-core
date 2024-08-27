from typing import Union

import torch.nn as nn

from olmo_core.config import StrEnum

from ..attention import Attention, FusedAttention
from ..feed_forward import FeedForward


class InitMethod(StrEnum):
    normal = "normal"
    """
    Every linear and embedding layer and initialized from a truncated normal distributed
    with standard deviation 0.02.
    """

    llama = "llama"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the number of layers.
    """

    llama_depth = "llama_depth"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the layer index.
    """

    def _init_linear(self, m: nn.Linear, *, std: float = 0.02):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def init_embeddings(self, m: nn.Embedding):
        if self in (InitMethod.llama, InitMethod.llama_depth):
            nn.init.normal_(m.weight)
        else:
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    def init_final_w_out(self, m: nn.Linear, *, d_model: int):
        std = 0.02
        if self in (InitMethod.llama, InitMethod.llama_depth):
            std = d_model**-0.05
        self._init_linear(m, std=std)

    def init_attention(
        self, m: Union[Attention, FusedAttention], *, block_idx: int, num_blocks: int
    ):
        std = 0.02
        if self == InitMethod.llama:
            std = 0.02 / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = 0.02 / (2 * (block_idx + 1)) ** 0.5

        if isinstance(m, Attention):
            for w in (m.w_q, m.w_k, m.w_v):
                self._init_linear(w, std=0.02)
        elif isinstance(m, FusedAttention):
            self._init_linear(m.w_qkv, std=0.02)
        else:
            raise NotImplementedError(m)

        self._init_linear(m.w_out, std=std)

    def init_feed_forward(self, m: FeedForward, *, block_idx: int, num_blocks: int):
        std = 0.02
        if self == InitMethod.llama:
            std = 0.02 / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = 0.02 / (2 * (block_idx + 1)) ** 0.5

        self._init_linear(m.w1, std=0.02)
        self._init_linear(m.w2, std=std)
        self._init_linear(m.w3, std=std)
