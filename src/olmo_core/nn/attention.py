from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config, DType, StrEnum
from ..exceptions import OLMoConfigurationError
from .buffer_cache import BufferCache
from .layer_norm import LayerNorm, LayerNormConfig
from .rope import (
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    RoPEConfig,
    RotaryEmbedding,
)

__all__ = ["AttentionType", "AttentionConfig", "Attention", "FusedAttention"]


class AttentionType(StrEnum):
    """
    An enumeration of the different attention implementations.

    - "default" ➡️ :class:`Attention`
    - "fused" ➡️ :class:`FusedAttention`
    """

    default = "default"
    fused = "fused"


@dataclass
class AttentionConfig(Config):
    """
    A configuration class for easily building any of the different attention modules.

    See :class:`Attention` for a description of the parameters.
    """

    name: AttentionType = AttentionType.default
    """
    - "default" ➡️ :class:`Attention`
    - "fused" ➡️ :class:`FusedAttention`
    """
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    bias: bool = True
    rope: Optional[RoPEConfig] = None
    clip_qkv: Optional[float] = None
    qk_norm: Optional[LayerNormConfig] = None
    dropout: float = 0.0
    use_flash: Optional[bool] = None
    dtype: DType = DType.float32

    def build(
        self,
        d_model: int,
        *,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> Union["Attention", "FusedAttention"]:
        """
        Build the corresponding attention module.

        See :class:`Attention` for a description of the parameters.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs["dtype"] = kwargs["dtype"].as_pt()
        kwargs.update(
            dict(
                d_model=d_model,
                init_device=init_device,
                cache=cache,
            )
        )

        if self.name == "default":
            return Attention(**kwargs)
        elif self.name == "fused":
            kwargs.pop("use_flash", None)
            return FusedAttention(**kwargs)
        else:
            raise NotImplementedError(self.name)


class Attention(nn.Module):
    """
    An implementation of multi-head self-attention with support for multi-query (MQA)
    and grouped-query (GQA) attention.

    Intra-document masking is also supported by passing in the
    ``max_doc_len`` and ``cu_doc_lens`` parameters to :meth:`forward()`. Currently this requires
    `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ (``use_flash=True``).

    .. seealso::
        :class:`FusedAttention` if you have flash-attn installed and you're not using MQA or GQA.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_kv_heads: The number of key and value heads, if different.
    :param bias: Include biases with linear layers.
    :param rope: The config for RoPE, if RoPE should be used.
    :param clip_qkv: Clip QKV to this value, if set.
    :param qk_norm: Configuration a layer norm for queries and keys.
    :param dropout: Dropout probability.
    :param use_flash: Use flash attention.
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        bias: bool = True,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        use_flash: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_v = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.clip_qkv = clip_qkv
        self.dropout_p = dropout

        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if qk_norm is not None:
            self.q_norm = qk_norm.build(size=d_model, init_device=init_device)
            self.k_norm = qk_norm.build(
                size=self.n_kv_heads * self.head_dim, init_device=init_device
            )

        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_class

        self._flash_attn_func = None
        if use_flash:
            from flash_attn import (  # type: ignore
                flash_attn_func,
                flash_attn_varlen_func,
            )

            self._flash_attn_func = flash_attn_func
            self._flash_attn_varlen_func = flash_attn_varlen_func

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.clip_qkv is not None:
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            k.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            v.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        if self.rope is not None:
            q, k = self.rope(q, k, head_first=False)

        if max_doc_len is not None and cu_doc_lens is not None:
            if self._flash_attn_varlen_func is None:
                raise RuntimeError(
                    "flash-attn (use_flash=True) is required for intra-document masking"
                )
            # shape: (batch_size * seq_len, n_heads, head_dim)
            att = self._flash_attn_varlen_func(
                q.view(B * T, self.n_heads, self.head_dim),
                k.view(B * T, self.n_kv_heads, self.head_dim),
                v.view(B * T, self.n_kv_heads, self.head_dim),
                cu_doc_lens,
                cu_doc_lens,
                max_doc_len,
                max_doc_len,
                dropout_p=self.dropout_p,
                causal=True,
            )
        elif self._flash_attn_func is not None:
            # shape: (batch_size, seq_len, n_heads, head_dim)
            att = self._flash_attn_func(q, k, v, dropout_p=self.dropout_p, causal=True)
        else:
            # Fall back to PyTorch's SDPA...

            # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
            # shape: (batch_size, n_heads, seq_len, head_dim),
            #        (batch_size, n_kv_heads, seq_len, head_dim),
            #        (batch_size, n_kv_heads, seq_len, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # PyTorch's SDPA doesn't support GQA, so we have to do this.
            if self.n_heads != self.n_kv_heads and self.n_kv_heads > 1:
                k = k.repeat_interleave(
                    self.n_heads // self.n_kv_heads, dim=1, output_size=self.n_heads
                )
                v = v.repeat_interleave(
                    self.n_heads // self.n_kv_heads, dim=1, output_size=self.n_heads
                )

            # shape: (batch_size, n_heads, seq_len, head_dim)
            att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True)
            # shape: (batch_size, seq_len, n_heads, head_dim)
            att = att.transpose(1, 2).contiguous()

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)


class FusedAttention(nn.Module):
    """
    An "fused" implementation of multi-head self-attention.

    Intra-document masking is supported by passing in the ``max_doc_len`` and ``cu_doc_lens``
    parameters to :meth:`forward()`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    .. warning::
        If using RoPE, this requires that you use the "fused" RoPE implementation
        (:class:`~olmo_core.nn.rope.FusedRotaryEmbedding`).

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param bias: Include biases with linear layers.
    :param rope: The config for RoPE, if RoPE should be used.
    :param clip_qkv: Clip QKV to this value, if set.
    :param dropout: Dropout probability.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        from flash_attn import (  # type: ignore
            flash_attn_qkvpacked_func,
            flash_attn_varlen_qkvpacked_func,
        )

        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=bias, dtype=dtype, device=init_device)
        self.w_out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.clip_qkv = clip_qkv
        self.dropout_p = dropout
        self.rope: Optional[FusedRotaryEmbedding] = None
        if rope is not None:
            if rope.name != "fused":
                raise OLMoConfigurationError(f"{self.__class__.__name__} requires fused RoPE")
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, FusedRotaryEmbedding)
            self.rope = rope_class

        self._flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
        self._flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        B, T, _ = x.shape

        # shape: (batch_size, seq_len, 3, n_heads, head_dim)
        qkv = self.w_qkv(x).view(B, T, 3, self.n_heads, self.head_dim)

        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        if self.rope is not None:
            qkv = self.rope(qkv)

        if max_doc_len is not None and cu_doc_lens is not None:
            # shape: (batch_size * seq_len, n_heads, head_dim)
            att = self._flash_attn_varlen_qkvpacked_func(
                qkv.view(B * T, 3, self.n_heads, self.head_dim),
                cu_doc_lens,
                max_doc_len,
                dropout_p=self.dropout_p,
                causal=True,
            )
        else:
            # shape: (batch_size, seq_len, n_heads, head_dim)
            att = self._flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout_p, causal=True)

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)
