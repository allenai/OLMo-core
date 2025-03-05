import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

from olmo_core.config import Config, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError

from ..attention import AttentionConfig
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


class TransformerBlockType(StrEnum):
    """
    An enumeration of the different transformer block implementations.
    """

    default = "default"
    """
    ➡️ :class:`TransformerBlock`
    """

    reordered_norm = "reordered_norm"
    """
    ➡️ :class:`ReorderedNormTransformerBlock`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformerBlock`
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformerBlock`
    """

    moe_reordered_norm = "moe_reordered_norm"
    """
    ➡️ :class:`MoEReorderedNormTransformerBlock`
    """

    moe_parallel = "moe_parallel"
    """
    ➡️ :class:`MoEParallelTransformerBlock`
    """

    moe_parallel_reordered_norm = "moe_parallel_reordered_norm"
    """
    ➡️ :class:`MoEParallelReorderedNormTransformerBlock`
    """


@dataclass
class TransformerBlockConfig(Config):
    """
    A configuration class for easily building transformer blocks.
    """

    attention: AttentionConfig
    """
    The attention config.
    """
    layer_norm: Optional[LayerNormConfig] = None
    """
    The layer norm config.
    """
    feed_forward: Optional[FeedForwardConfig] = None
    """
    The feed-forward config, required for non-MoE blocks.
    """
    feed_forward_moe: Optional[MoEConfig] = None
    """
    The config for the MoE feed-forward layer. Required for MoE blocks.
    """
    name: TransformerBlockType = TransformerBlockType.default
    """
    The block type.
    """
    dropout: Optional[float] = None
    """
    Dropout probability.
    """

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlockBase":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == TransformerBlockType.default:
                return TransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.reordered_norm:
                return ReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.normalized:
                return NormalizedTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe:
                return MoETransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_reordered_norm:
                return MoEReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_parallel:
                return MoEParallelTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_parallel_reordered_norm:
                return MoEParallelReorderedNormTransformerBlock(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class TransformerBlockBase(nn.Module):
    """
    Base class for transformer block implementations.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        raise NotImplementedError


class TransformerBlock(TransformerBlockBase):
    """
    A typical "Llama-style" transformer block implementation.

    :param d_model: The model dimensionality.
    :param block_idx: The index/position of the block within the model. Ranges from 0 to ``n_layers - 1``.
    :param attention: The attention module config.
    :param feed_forward: The feed forward module config.
    :param layer_norm: The layer norm config for both the attention LN and the feed forward LN.
    :param dropout: Dropout probability.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(d_model, init_device=init_device, cache=cache)
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.feed_forward.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())


class ReorderedNormTransformerBlock(TransformerBlock):
    """
    Like :class:`TransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the output
    of the feed-forward instead of the input.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))


@beta_feature
class NormalizedTransformerBlock(TransformerBlockBase):
    """
    An nGPT block implementation to be used with the :class:`~olmo_core.nn.attention.NormalizedAttention`
    attention type and :class:`~olmo_core.nn.feed_forward.NormalizedFeedForward` feed-forward type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(d_model, init_device=init_device, cache=cache)
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.attn_alpha = nn.Parameter(
            self.attn_alpha_init_scaling
            * torch.ones(d_model, dtype=torch.float32, device=init_device)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.mlp_alpha = nn.Parameter(
            self.mlp_alpha_init_scaling
            * torch.ones(d_model, dtype=torch.float32, device=init_device)
        )

    def reset_parameters(self):
        nn.init.ones_(self.attn_alpha)
        self.attn_alpha.mul_(self.attn_alpha_init_scaling)
        nn.init.ones_(self.mlp_alpha)
        self.mlp_alpha.mul_(self.mlp_alpha_init_scaling)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = l2_normalize(
            torch.lerp(
                x,
                l2_normalize(self.attention(x, **kwargs)),
                (
                    self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
                ).abs(),
            )
        )

        return l2_normalize(
            torch.lerp(
                h,
                l2_normalize(self.feed_forward(h)),
                (self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)).abs(),
            )
        )

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer block variant"
        )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if hasattr(self.attention, "normalize_matrices"):
            self.attention.normalize_matrices()  # type: ignore

        if hasattr(self.feed_forward, "normalize_matrices"):
            self.feed_forward.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))


class MoETransformerBlock(TransformerBlockBase):
    """
    Like :class:`TransformerBlock` except that the dense :class:`~olmo_core.nn.feed_forward.FeedForward`
    module is replaced with a mixture-of-experts (MoE).
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        attention: AttentionConfig,
        feed_forward_moe: MoEConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_idx = block_idx
        self.attention = attention.build(d_model, init_device=init_device, cache=cache)
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward_moe = feed_forward_moe.build(
            d_model=d_model, init_device=init_device, cache=cache
        )
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def compute_losses(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, torch.Tensor]:
        return self.feed_forward_moe.compute_losses(total_bz, reset=reset)

    def reset_losses(self):
        self.feed_forward_moe.reset_losses()

    def compute_metrics(
        self, total_bz: Union[int, torch.Tensor], reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.feed_forward_moe.compute_metrics(total_bz, reset=reset)

    def reset_metrics(self):
        self.feed_forward_moe.reset_metrics()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        Parameters are the same as :meth:`TransformerBlock.forward()`.
        """
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward_moe(self.feed_forward_norm(h)))

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        self.feed_forward_moe.apply_ep(ep_mesh, **kwargs)

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.feed_forward_moe.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())


class MoEReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the
    output of the feed-forward MoE instead of the input.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward_moe(h)))


class MoEParallelTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention and MLP are done in parallel
    like PaLM instead of in sequence.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return (
            x
            + self.dropout(self.feed_forward_moe(self.feed_forward_norm(x)))
            + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        )


class MoEParallelReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoEReorderedNormTransformerBlock` except that the attention and MLP are done in parallel
    like PaLM instead of in sequence.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return (
            x
            + self.dropout(self.feed_forward_norm(self.feed_forward_moe(x)))
            + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        )
