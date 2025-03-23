from typing import Optional, Union

import torch
import torch.nn as nn

from olmo_core.config import StrEnum

from ..attention import Attention, FusedAttention
from ..feed_forward import FeedForward
from ..moe import MoE


class InitMethod(StrEnum):
    normal = "normal"
    """
    Every linear and embedding layer and initialized from a truncated normal distributed
    with standard deviation 0.02.
    """

    normalized = "normalized"
    """
    Follow the nGPT initialization scheme.
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

    mup = "mup"
    """
    Apply muP
    """

    def _init_linear(
        self, m: nn.Linear, *, std: float = 0.02, generator: Optional[torch.Generator] = None
    ):
        nn.init.trunc_normal_(
            m.weight, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def init_embeddings(
        self, m: nn.Embedding, *, d_model: int, generator: Optional[torch.Generator] = None
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth):
            nn.init.normal_(m.weight, generator=generator)
        elif self in InitMethod.normalized:
            nn.init.normal_(m.weight, std=d_model**-0.5) 
        elif self == InitMethod.mup:
            # nn.init.normal_(m.weight, std=d_model ** -0.5) 
            nn.init.normal_(m.weight, std=1)       
        else:
            nn.init.trunc_normal_(
                m.weight, mean=0.0, std=0.02, a=-3 * 0.02, b=3 * 0.02, generator=generator
            )

    def init_final_w_out(
        self, m: nn.Linear, *, d_model: int, generator: Optional[torch.Generator] = None
    ):
        if self == InitMethod.mup:
            # std = d_model**-0.5
            std = 1.0
        
        elif self in (InitMethod.llama, InitMethod.llama_depth, InitMethod.normalized):
            std = d_model**-0.5
        
        else:
            std = 0.02

        self._init_linear(m, std=std, generator=generator)

    def init_attention(
        self,
        m: Union[Attention, FusedAttention],
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.mup:
            std_qkv = d_model ** -0.5
            std_out = d_model ** -0.5

            if isinstance(m, Attention):
                for w in (m.w_q, m.w_k, m.w_v):
                    self._init_linear(w, std=std_qkv, generator=generator)
                self._init_linear(m.w_out, std=std_out, generator=generator)

            elif isinstance(m, FusedAttention):
                self._init_linear(m.w_qkv, std=std_qkv, generator=generator)
                self._init_linear(m.w_out, std=std_out, generator=generator)

            else:
                raise NotImplementedError(m)

        else:
            std = 0.02
            if self in InitMethod.normalized:
                std = d_model ** -0.5
                if isinstance(m, Attention):
                    for w in (m.w_q, m.w_k, m.w_v):
                        self._init_linear(w, std=std, generator=generator)

                elif isinstance(m, FusedAttention):
                    self._init_linear(m.w_qkv, std=std, generator=generator)

                else:
                    raise NotImplementedError(m)

            if self == InitMethod.llama:
                std = std / (2 * num_blocks) ** 0.5
            elif self == InitMethod.llama_depth:
                std = std / (2 * (block_idx + 1)) ** 0.5
            elif self == InitMethod.normalized:
                std = std / (2 * num_blocks) ** 0.5

            self._init_linear(m.w_out, std=std, generator=generator)

    def init_feed_forward(
        self,
        m: FeedForward,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.mup:
            std_in = d_model ** -0.5
            std_out = m.hidden_size ** -0.5

            self._init_linear(m.w1, std=std_in, generator=generator)
            self._init_linear(m.w3, std=std_in, generator=generator)
            self._init_linear(m.w2, std=std_out, generator=generator)

        else:
            std = 0.02
            if self in InitMethod.normalized:
                std = d_model**-0.5

            self._init_linear(m.w1, std=std, generator=generator)

            std = 0.02
            if self == InitMethod.llama:
                std = 0.02 / (2 * num_blocks) ** 0.5
            elif self == InitMethod.llama_depth:
                std = 0.02 / (2 * (block_idx + 1)) ** 0.5
            elif self in InitMethod.normalized:
                std = d_model**-0.5

            self._init_linear(m.w3, std=std, generator=generator)

            if self == InitMethod.normalized:
                std = std / (2 * num_blocks) ** 0.5

            self._init_linear(m.w2, std=std, generator=generator)

    def init_feed_forward_moe(
        self,
        m: MoE,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        generator: Optional[torch.Generator] = None,
    ):
        del d_model

        std = 0.02
        if self == InitMethod.llama:
            std = 0.02 / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = 0.02 / (2 * (block_idx + 1)) ** 0.5

        self._init_linear(m.inner.router.layer, std=0.02, generator=generator)
        nn.init.trunc_normal_(
            m.inner.experts.mlp.w1, mean=0.0, std=0.02, a=-3 * std, b=3 * std, generator=generator
        )
        nn.init.trunc_normal_(
            m.inner.experts.mlp.w2, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator
        )
        if hasattr(m.inner.experts.mlp, "v1"):
            nn.init.trunc_normal_(
                m.inner.experts.mlp.v1,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
        if (bias := getattr(m.inner.experts, "bias", None)) is not None:
            nn.init.zeros_(bias)
