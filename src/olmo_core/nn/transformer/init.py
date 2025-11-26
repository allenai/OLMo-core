from typing import Optional, Union, cast
import math
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

from ..attention import Attention, AttentionBase, FusedAttention, MultiheadLatentAttention
from ..feed_forward import FeedForward
from ..moe import DroplessMoEMLP, MoEBase, MoELinearRouter, MoEMLP, MoEOrthogonalRouter


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

def kaiming_fan_in_uniform_(
    tensor: torch.Tensor,
    in_features: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    # Kaiming uniform with a=sqrt(5)
    # same as uniform(-1/sqrt(in_features), 1/sqrt(in_features))

    nn.init.uniform_(
        tensor,
        a=-1.0 / math.sqrt(in_features),
        b=1.0 / math.sqrt(in_features),
        generator=generator,
    )
    return tensor


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

    def _init_linear(
        self, m: nn.Linear, *, std: float = 0.02, generator: Optional[torch.Generator] = None
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
        self._init_linear(m, std=std, generator=generator)

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
            for w in (m.w_q, m.w_k, m.w_v):
                self._init_linear(w, std=std, generator=generator)
        elif isinstance(m, FusedAttention) or hasattr(m, "w_qkv"):
            m = cast(FusedAttention, m)
            self._init_linear(m.w_qkv, std=std, generator=generator)
        elif isinstance(m, MultiheadLatentAttention):
            m = cast(MultiheadLatentAttention, m)
            if hasattr(m, "wq"):
                self._init_linear(m.wq, std=std, generator=generator)
            else:
                self._init_linear(m.wq_a, std=std, generator=generator)
                self._init_linear(m.wq_b, std=std, generator=generator)
                
            self._init_linear(m.wkv_a, std=std, generator=generator)
            self._init_linear(m.wkv_b, std=std, generator=generator)
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
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self == InitMethod.normalized:
            std = d_model**-0.5

        self._init_linear(m.w1, std=std, generator=generator)

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        self._init_linear(m.w3, std=std, generator=generator)

        if self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        self._init_linear(m.w2, std=std, generator=generator)

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
    def init_moe_v2(
        self,
        b,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
        ep_generator: Optional[torch.Generator] = None,
    ):
        from ..moe.v2.block import MoEFusedV2TransformerBlock
        b = cast(MoEFusedV2TransformerBlock, b)
        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        if ep_generator is None:
            assert b.ep_enabled is False, "ep_generator should be provided when ep_enabled is True"
            # use default generator for ep_generator
            # which means (EP is not used)
            ep_generator = generator

        # router
        if b.shared_experts_router:
            _apply_init(
                nn.init.trunc_normal_,
                b.shared_experts_router.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
        if b.routed_experts_router:
            _apply_init(
                nn.init.trunc_normal_,
                b.routed_experts_router.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
        # routed experts
        if b.routed_experts:
            _apply_init(
                nn.init.trunc_normal_,
                b.routed_experts.w_up_gate,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=ep_generator, # might be sharded, use ep_generator
            )
            # _apply_init(
            #     kaiming_fan_in_uniform_,
            #     b.routed_experts.w_up_gate,
            #     in_features=b.routed_experts.d_model, # fan_in = d_model
            #     generator=generator,
            # )
            _apply_init(
                nn.init.trunc_normal_,
                b.routed_experts.w_down,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=ep_generator, # might be sharded, use ep_generator
            )
            # assert b.routed_experts_router is not None
            # _apply_init(
            #     kaiming_fan_in_uniform_,
            #     b.routed_experts.w_down,
            #     in_features=b.routed_experts.hidden_size * b.routed_experts_router.top_k, # fan_in = moe hidden size
            #     generator=generator,
            # )

        # shared experts
        if b.shared_experts:
            _apply_init(
                nn.init.trunc_normal_,
                b.shared_experts.w_up_gate,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
            # _apply_init(
            #     kaiming_fan_in_uniform_,
            #     b.shared_experts.w_up_gate,
            #     in_features=b.shared_experts.d_model, 
            #     generator=generator,
            # )
            _apply_init(
                nn.init.trunc_normal_,
                b.shared_experts.w_down,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )
            # _apply_init(
            #     kaiming_fan_in_uniform_,
            #     b.shared_experts.w_down,
            #     in_features=b.shared_experts.hidden_size, 
            #     generator=generator,
            # )
