from typing import Optional, Union, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

from ..attention import Attention, FusedAttention, SequenceMixer
from ..feed_forward import FeedForward
from ..moe import DroplessMoEMLP, MoEBase, MoELinearRouter, MoEMLP


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

    fan_in = "fan_in"
    """
    Per-layer fan-in initialization where each weight matrix is initialized with
    ``std = 1/√d_in`` where ``d_in`` is the fan-in (number of input features) of that
    specific layer. Embeddings use ``std = 1.0`` with normal distribution.
    This provides forward-pass variance-preserving initialization adapted to each layer's
    specific dimensions, with no depth scaling.
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
        elif self == InitMethod.fan_in:
            # Fan-in init uses std = 1.0 for embeddings with normal distribution
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=1.0)
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
        if self in (
            InitMethod.llama,
            InitMethod.llama_depth,
            InitMethod.normalized,
            InitMethod.fan_in,
        ):
            std = d_model**-0.5
        self._init_linear(m, std=std, generator=generator)

    def init_attention(
        self,
        m: SequenceMixer,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        # NOTE: isinstance checks could fail with AC wrappers
        if isinstance(m, Attention) or hasattr(m, "w_q"):
            m = cast(Attention, m)

            # Compute std for Q/K/V initialization
            if self == InitMethod.fan_in:
                # For fan_in, use 1/√d_in based on actual weight shape (ignores base std parameter)
                # Each projection may have different output dims (n_heads * head_dim vs n_kv_heads * head_dim)
                # but they all have the same input dim
                for w in (m.w_q, m.w_k, m.w_v):
                    w_std = w.in_features**-0.5
                    self._init_linear(w, std=w_std, generator=generator)
            else:
                if self == InitMethod.normalized:
                    std = d_model**-0.5
                for w in (m.w_q, m.w_k, m.w_v):
                    self._init_linear(w, std=std, generator=generator)
        elif isinstance(m, FusedAttention) or hasattr(m, "w_qkv"):
            m = cast(FusedAttention, m)

            # Compute std for fused QKV initialization
            if self == InitMethod.fan_in:
                # For fan_in, use 1/√d_in based on actual weight shape
                std = m.w_qkv.in_features**-0.5
            elif self == InitMethod.normalized:
                std = d_model**-0.5

            self._init_linear(m.w_qkv, std=std, generator=generator)
        else:
            raise NotImplementedError(m)

        # Compute std for w_out initialization
        if self == InitMethod.fan_in:
            # For fan_in, w_out uses 1/√d_in based on actual weight shape
            std = m.w_out.in_features**-0.5
        elif self == InitMethod.llama:
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
        # Compute std for w1 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w1 uses 1/√d_in where d_in = d_model (ignores base std parameter)
            std = m.w1.in_features**-0.5
        elif self == InitMethod.normalized:
            std = d_model**-0.5

        self._init_linear(m.w1, std=std, generator=generator)

        # Compute std for w3 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w3 uses 1/√d_in where d_in = d_model
            std = m.w3.in_features**-0.5
        elif self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        self._init_linear(m.w3, std=std, generator=generator)

        # Compute std for w2 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w2 uses 1/√d_in where d_in = hidden_size
            std = m.w2.in_features**-0.5
        elif self == InitMethod.normalized:
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
        elif self == InitMethod.fan_in:
            # For fan_in, router weight uses 1/√d_model (d_in = d_model)
            router_weight = cast(MoELinearRouter, m.router).weight
            # Router weight is flattened (num_experts * d_model,) -> (num_experts, d_model)
            d_in = router_weight.numel() // cast(MoELinearRouter, m.router).num_experts
            std = d_in**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(MoELinearRouter, m.router).weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w1
        if self == InitMethod.fan_in:
            # w1 has shape (num_experts * d_model, hidden_size)
            # d_in for each expert is d_model
            w1 = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w1
            d_in = w1.shape[0] // cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).num_experts
            std = d_in**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w1,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w2
        if self == InitMethod.fan_in:
            # w2 has shape (num_experts * hidden_size, d_model)
            # d_in for each expert is hidden_size
            w2 = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w2
            d_in = w2.shape[0] // cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).num_experts
            std = d_in**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w2,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w3
        if self == InitMethod.fan_in:
            # w3 has shape (num_experts * d_model, hidden_size)
            # d_in for each expert is d_model
            w3 = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w3
            d_in = w3.shape[0] // cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).num_experts
            std = d_in**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp).w3,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
