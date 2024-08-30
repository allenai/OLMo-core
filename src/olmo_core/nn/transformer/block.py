from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from olmo_core.config import Config, StrEnum

from ..attention import AttentionConfig
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig


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


@dataclass
class TransformerBlockConfig(Config):
    """
    A configuration class for easily building transformer blocks.
    """

    attention: AttentionConfig
    feed_forward: FeedForwardConfig
    layer_norm: LayerNormConfig
    name: TransformerBlockType = TransformerBlockType.default
    dropout: float = 0.0

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlock":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            dict(
                d_model=d_model,
                block_idx=block_idx,
                init_device=init_device,
                cache=cache,
            )
        )

        if self.name == TransformerBlockType.default:
            return TransformerBlock(**kwargs)
        elif self.name == TransformerBlockType.reordered_norm:
            return ReorderedNormTransformerBlock(**kwargs)
        else:
            raise NotImplementedError(self.name)


class TransformerBlock(nn.Module):
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
