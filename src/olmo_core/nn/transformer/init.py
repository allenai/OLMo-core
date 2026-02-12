from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

if TYPE_CHECKING:
    from ..attention import SequenceMixer
    from ..feed_forward import FeedForward
    from ..moe import MoEBase


def _apply_init(init_fun, x: torch.Tensor, *args, **kwargs):
    if not isinstance(x, DTensor):
        init_fun(x, *args, **kwargs)
        return

    # Initialize full version of x locally, then apply init to that.
    full_x = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    init_fun(full_x, *args, **kwargs)
    full_x = distribute_like(x, full_x)

    # Now copy over the corresponding shard of `full_x` into `x`.
    get_local_tensor(x).copy_(get_local_tensor(full_x))


def init_linear(
    m: nn.Linear | nn.Conv1d, *, std: float = 0.02, generator: Optional[torch.Generator] = None
):
    _apply_init(
        nn.init.trunc_normal_,
        m.weight,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )
    if m.bias is not None:
        nn.init.zeros_(m.bias)


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
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth, InitMethod.normalized):
            std = d_model**-0.5
        init_linear(m, std=std, generator=generator)

    def init_attention(
        self,
        m: "SequenceMixer",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        m.init_weights(
            init_method=self,
            d_model=d_model,
            block_idx=block_idx,
            num_blocks=num_blocks,
            std=std,
            generator=generator,
        )

    def init_feed_forward(
        self,
        m: "FeedForward",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(m.w1, std=std, generator=generator)

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        init_linear(m.w3, std=std, generator=generator)

        if self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(m.w2, std=std, generator=generator)

    def init_feed_forward_moe(
        self,
        m: "MoEBase",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        from ..moe import DroplessMoEMLP, MoELinearRouter, MoEMLP

        del d_model

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(MoELinearRouter, m.router).weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
        _apply_init(
            nn.init.trunc_normal_,
            cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w1,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
        _apply_init(
            nn.init.trunc_normal_,
            cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w2,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
        _apply_init(
            nn.init.trunc_normal_,
            cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w3,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
