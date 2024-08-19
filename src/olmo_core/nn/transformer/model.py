from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from olmo_core.config import Config
from olmo_core.utils import get_cumulative_document_lengths, has_flash_attn

from ..attention import AttentionConfig, AttentionType
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig, LayerNormType
from ..rope import RoPEConfig, RoPEType
from .block import TransformerBlockConfig, TransformerBlockType

__all__ = ["TransformerConfig", "Transformer"]


@dataclass
class TransformerConfig(Config):
    """
    A config for easily building transformer models.

    See :class:`Transformer` for a description of the parameters.
    """

    d_model: int
    vocab_size: int
    n_layers: int
    block: TransformerBlockConfig
    layer_norm: LayerNormConfig
    bias: bool = True
    dtype: torch.dtype = torch.float32

    def build(self, init_device: str = "cpu") -> "Transformer":
        """
        Build the model corresponding to this config.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        return Transformer(init_device=init_device, **kwargs)

    @property
    def num_params(self) -> int:
        """
        The number of parameters that a model from this config would have.
        """

        def layer_norm_params(layer_norm: LayerNormConfig) -> int:
            ln_params = 0
            if layer_norm.elementwise_affine:
                ln_params += self.d_model
                if layer_norm.bias:
                    ln_params += self.d_model
            return ln_params

        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size

        block_params = 0

        n_heads = self.block.attention.n_heads
        n_kv_heads = self.block.attention.n_kv_heads or n_heads
        head_dim = self.d_model // n_heads

        # Block attention Q projection.
        block_params += self.d_model * self.d_model
        if self.block.attention.bias:
            block_params += self.d_model

        # Block attention KV projections.
        block_params += 2 * self.d_model * n_kv_heads * head_dim
        if self.block.attention.bias:
            block_params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.block.attention.qk_norm is not None:
            block_params += 2 * layer_norm_params(self.block.attention.qk_norm)

        # Block attention out.
        block_params += self.d_model * self.d_model
        if self.block.attention.bias:
            block_params += self.d_model

        # Block attention norm.
        block_params += layer_norm_params(self.block.layer_norm)

        # Block feed forward.
        block_params += 3 * self.d_model * self.block.feed_forward.hidden_size
        if self.block.feed_forward.bias:
            block_params += 2 * self.block.feed_forward.hidden_size + self.d_model

        # Block feed forward norm.
        block_params += layer_norm_params(self.block.layer_norm)

        # All block params.
        num_params += self.n_layers * block_params

        # Final layer norm.
        num_params += layer_norm_params(self.layer_norm)

        # Final FF out.
        num_params += self.d_model * self.vocab_size
        if self.bias:
            num_params += self.vocab_size

        return num_params

    @classmethod
    def llama2_271M(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 271M Llama2 model config.
        """
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=16,
            n_heads=8,
            rope_theta=10_000,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama2_1B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 1B Llama2 model config.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=18,
            n_heads=16,
            rope_theta=10_000,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama2_7B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 7B Llama2 model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=32,
            n_heads=32,
            rope_theta=10_000,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama2_13B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 7B Llama2 model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=40,
            n_heads=40,
            rope_theta=10_000,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama2_26B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 26B Llama2 model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=80,
            n_heads=40,
            rope_theta=10_000,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama2_70B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 70B Llama2 model config.
        """
        return cls.llama_like(
            d_model=8192,
            vocab_size=vocab_size,
            n_layers=80,
            n_heads=64,
            n_kv_heads=8,
            rope_theta=10_000,
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama3_8B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        An 8B Llama3 model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=500_000,
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=1024,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama3_70B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 70B Llama3 model config.
        """
        return cls.llama_like(
            d_model=8196,
            vocab_size=vocab_size,
            n_layers=80,
            n_heads=64,
            n_kv_heads=8,
            rope_theta=500_000,
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama3_405B(cls, vocab_size: int, fused_ops: bool = False) -> "TransformerConfig":
        """
        A 405B Llama3 model config.
        """
        return cls.llama_like(
            d_model=16384,
            vocab_size=vocab_size,
            n_layers=126,
            n_heads=128,
            n_kv_heads=8,
            rope_theta=500_000,
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=4096,
            fused_ops=fused_ops,
        )

    @classmethod
    def llama_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        rope_theta: int = 500_000,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        fused_ops: bool = False,
    ) -> "TransformerConfig":
        """
        Create a Llama-like configuration.

        :param hidden_size_multiple_of: Ensure the FFN hidden size is a multiple of this value.
        :param hidden_size_multiplier: Custom multiplier for the FFN hidden size.
        :param fused_ops: Use fused operations where possible.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = hidden_size_multiple_of * (
            (hidden_size + hidden_size_multiple_of - 1) // hidden_size_multiple_of
        )

        # Configure global layer norm.
        layer_norm = LayerNormConfig(
            name=LayerNormType.fused_rms if fused_ops else LayerNormType.rms, eps=1e-5, bias=False
        )

        # Decide on attention/rope implementations.
        att_type = AttentionType.default
        rope_type = RoPEType.complex
        if fused_ops and n_kv_heads is None:  # fused attention not compatible with MQA/GQA.
            att_type = AttentionType.fused
            rope_type = RoPEType.fused

        # Configure blocks.
        block = TransformerBlockConfig(
            name=TransformerBlockType.default,
            attention=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                bias=False,
                rope=RoPEConfig(name=rope_type, theta=rope_theta),
                use_flash=has_flash_attn(),
            ),
            feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
            layer_norm=layer_norm,
        )

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            layer_norm=layer_norm,
            bias=False,
        )


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param n_layers: The number of transformer layers/blocks.
    :param block: The block configuration.
    :param layer_norm: The layer norm config for the final layer norm.
    :param bias: Whether to use a bias in the final linear layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig,
        layer_norm: LayerNormConfig,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        cache = BufferCache()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.blocks = nn.ModuleList(
            [
                block.build(
                    d_model,
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
