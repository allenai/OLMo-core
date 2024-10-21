from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from ..attention import AttentionConfig
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig


class TransformerBlockType(StrEnum):
    """
    An enumeration of the different transformer block implementations.
    """

    default = "default"
    """
    :class:`TransformerBlock`
    """

    reordered_norm = "reordered_norm"
    """
    :class:`ReorderedNormTransformerBlock`
    """

    moe = "moe"
    """
    :class:`MoETransformerBlock`
    """

    moe_reordered_norm = "moe"
    """
    :class:`MoEReorderedNormTransformerBlock`
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
    layer_norm: LayerNormConfig
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
    dropout: float = 0.0
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
        if self.name == TransformerBlockType.default:
            if self.feed_forward is None:
                raise OLMoConfigurationError("'feed_forward' config is required")
            return TransformerBlock(
                d_model=d_model,
                block_idx=block_idx,
                attention=self.attention,
                feed_forward=self.feed_forward,
                layer_norm=self.layer_norm,
                dropout=self.dropout,
                init_device=init_device,
                cache=cache,
            )
        elif self.name == TransformerBlockType.reordered_norm:
            if self.feed_forward is None:
                raise OLMoConfigurationError("'feed_forward' config is required")
            return ReorderedNormTransformerBlock(
                d_model=d_model,
                block_idx=block_idx,
                attention=self.attention,
                feed_forward=self.feed_forward,
                layer_norm=self.layer_norm,
                dropout=self.dropout,
                init_device=init_device,
                cache=cache,
            )
        elif self.name == TransformerBlockType.moe:
            if self.feed_forward_moe is None:
                raise OLMoConfigurationError("'feed_forward_moe' config is required for MoE blocks")
            return MoETransformerBlock(
                d_model=d_model,
                block_idx=block_idx,
                attention=self.attention,
                feed_forward_moe=self.feed_forward_moe,
                layer_norm=self.layer_norm,
                dropout=self.dropout,
                init_device=init_device,
                cache=cache,
            )
        elif self.name == TransformerBlockType.moe_reordered_norm:
            if self.feed_forward_moe is None:
                raise OLMoConfigurationError("'feed_forward_moe' config is required for MoE blocks")
            return MoEReorderedNormTransformerBlock(
                d_model=d_model,
                block_idx=block_idx,
                attention=self.attention,
                feed_forward_moe=self.feed_forward_moe,
                layer_norm=self.layer_norm,
                dropout=self.dropout,
                init_device=init_device,
                cache=cache,
            )
        else:
            raise NotImplementedError(self.name)


class TransformerBlockBase(nn.Module):
    """
    Base class for transformer block implementations.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.dropout(
            self.attention(self.attention_norm(x), max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)
        )
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))


class ReorderedNormTransformerBlock(TransformerBlock):
    """
    Like :class:`TransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the output
    of the feed-forward instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.dropout(
            self.attention_norm(self.attention(x, max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens))
        )
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))


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
        self.feed_forward_moe = feed_forward_moe.build(d_model=d_model, init_device=init_device)
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        Parameters are the same as :meth:`TransformerBlock.forward()`.
        """
        h = x + self.dropout(
            self.attention(self.attention_norm(x), max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)
        )
        return h + self.dropout(self.feed_forward_moe(self.feed_forward_norm(h)))


class MoEReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the
    output of the feed-forward MoE instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.dropout(
            self.attention_norm(self.attention(x, max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens))
        )
        return h + self.dropout(self.feed_forward_norm(self.feed_forward_moe(h)))
