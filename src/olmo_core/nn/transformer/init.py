from typing import Dict, Optional, Union, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

from ..attention import Attention, AttentionBase, FusedAttention
from ..feed_forward import FeedForward
from ..moe import DroplessMoEMLP, MoEBase, MoELinearRouter, MoEMLP
from ..mup import MuP


def _apply_init(init_fun, x: torch.Tensor, *args, **kwargs):
    if not isinstance(x, DTensor):
        init_fun(x, *args, **kwargs)

    # Initialize full version of x locally, then apply init to that.
    full_x = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    init_fun(full_x, *args, **kwargs)
    full_x = distribute_like(x, full_x)

    # Now copy over the corresponding shard of `full_x` into `x`.
    get_local_tensor(x).copy_(get_local_tensor(full_x))


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

    def _init_weight(
        self,
        weight: torch.Tensor,
        *,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
        mup: Optional[MuP] = None,
    ):
        std = MuP.scale_init_std(mup, std)
        _apply_init(
            nn.init.trunc_normal_,
            weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

    def _init_linear(
        self,
        m: nn.Linear,
        *,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
        mup: Optional[MuP] = None,
    ):
        self._init_weight(m.weight, std=std, generator=generator, mup=mup)

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def init_embeddings(
        self,
        m: nn.Embedding,
        *,
        d_model: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth):
            _apply_init(nn.init.normal_, m.weight, generator=generator)
        elif self == InitMethod.normalized:
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=d_model**-0.5)
        else:
            _apply_init(
                nn.init.trunc_normal_,
                m.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )

    def init_final_w_out(
        self,
        m: nn.Linear,
        *,
        d_model: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
        mup: Optional[MuP] = None,
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth, InitMethod.normalized):
            std = d_model**-0.5
        self._init_linear(m, std=std, generator=generator, mup=mup)

    def init_attention(
        self,
        m: AttentionBase,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.normalized:
            std = d_model**-0.5

        # NOTE: isinstance checks could fail with AC wrappers
        if isinstance(m, Attention) or hasattr(m, "w_q"):
            m = cast(Attention, m)
            for name, w in zip(("w_q.weight", "w_k.weight", "w_v.weight"), (m.w_q, m.w_k, m.w_v)):
                self._init_linear(w, std=std, generator=generator, mup=m.mups.get(name))
        elif isinstance(m, FusedAttention) or hasattr(m, "w_qkv"):
            m = cast(FusedAttention, m)
            # FusedAttention does not support muP, so no mup scaling
            self._init_linear(m.w_qkv, std=std, generator=generator)
        else:
            raise NotImplementedError(m)

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        mups: Optional[Dict[str, MuP]] = getattr(m, "mups", None)
        self._init_linear(
            m.w_out, std=std, generator=generator, mup=mups.get("w_out.weight") if mups else None
        )

    def init_feed_forward(
        self,
        m: FeedForward,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.normalized:
            std = d_model**-0.5

        self._init_linear(m.w1, std=std, generator=generator, mup=m.mups.get("w1.weight"))

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        self._init_linear(m.w3, std=std, generator=generator, mup=m.mups.get("w3.weight"))

        if self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        self._init_linear(m.w2, std=std, generator=generator, mup=m.mups.get("w2.weight"))

    def init_feed_forward_moe(
        self,
        m: MoEBase,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        del d_model

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        router = cast(MoELinearRouter, m.router)
        self._init_weight(
            router.weight, std=std, generator=generator, mup=router.mups.get("router")
        )

        mlp = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp)
        self._init_weight(mlp.w1, std=std, generator=generator, mup=mlp.mups.get("w1"))
        self._init_weight(mlp.w2, std=std, generator=generator, mup=mlp.mups.get("w2"))
        self._init_weight(mlp.w3, std=std, generator=generator, mup=mlp.mups.get("w3"))
