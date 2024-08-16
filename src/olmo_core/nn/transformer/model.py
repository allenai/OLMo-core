from typing import Optional, Sequence

import torch
import torch.nn as nn

from olmo_core.utils import get_cumulative_document_lengths

from ..attention import AttentionConfig
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig
from .block import TransformerBlock


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param n_layers: The number of transformer layers/blocks.
    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param attention: The attention module config for each block.
    :param feed_forward: The feed forward module config for each block.
    :param layer_norm: The layer norm config for both the attention LN and the feed forward LN
        in each block.
    :param dropout: Dropout probability in each block.
    :param bias: Whether to use biases in the linear output layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        d_model: int,
        vocab_size: int,
        attention: AttentionConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        cache = cache or BufferCache()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    attention=attention,
                    feed_forward=feed_forward,
                    layer_norm=layer_norm,
                    dropout=dropout,
                    init_device=init_device,
                    cache=cache,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = layer_norm.build(d_model, init_device=init_device)
        self.w_out = nn.Linear(d_model, vocab_size, bias=bias, dtype=dtype, device=init_device)
        self._cache = cache

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embeddings.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w_out.weight, mean=0.0, std=0.02)
        if self.w_out.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape ``(batch_size, max_docs)``.
            Required together with ``max_doc_lens`` when using intra-document masking.
        :param max_doc_lens: Maximum document length for each instance in the batch.
            Required together with ``doc_lens`` when using intra-document masking.

        :returns: The output logits.
        """
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        h = self.embeddings(input_ids)

        for block in self.blocks:
            h = block(h, max_doc_len=max_doc_len, cu_doc_lens=cu_doc_lens)

        h = self.norm(h)
        return self.w_out(h).float()
