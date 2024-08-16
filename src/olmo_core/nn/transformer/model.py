from typing import Optional

import torch
import torch.nn as nn

from ..attention import AttentionConfig
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig
from .block import TransformerBlock


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.

        :returns: The output logits.
        """
        h = self.embeddings(input_ids)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        return self.w_out(h).float()
