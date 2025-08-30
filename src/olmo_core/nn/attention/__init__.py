import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError

from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..layer_norm import LayerNorm, LayerNormConfig
from ..rope import (
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    RoPEConfig,
    RotaryEmbedding,
)
from ..utils import get_tp_wrappers
from .flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_qkvpacked,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
)
from .ring import (
    RingAttentionLlama3LoadBalancer,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
)

__all__ = [
    "AttentionType",
    "AttentionConfig",
    "AttentionBase",
    "Attention",
    "FusedAttention",
    "NormalizedAttention",
    "RingAttentionLoadBalancerType",
    "RingAttentionLoadBalancer",
    "RingAttentionZigZagLoadBalancer",
    "RingAttentionLlama3LoadBalancer",
]


@dataclass
class SlidingWindowAttentionConfig(Config):
    pattern: List[int]
    """
    The pattern of window sizes to use for attention, repeated to cover all layers.
    A value of -1 indicates full attention. For example, a pattern of ``[4096, 4096, 4096, -1]``
    means that for each set of 4 layers, the first 3 will use a window size of 4096,
    and the last layer will use full attention.
    """

    force_full_attention_on_first_layer: bool = True
    """
    If `True`, the first transformer layer will always use full attention, regardless of the pattern.
    """

    force_full_attention_on_last_layer: bool = True
    """
    If `True`, the last transformer layer will always use full attention, regardless of the pattern.
    """

    def _get_window_size(self, layer_idx: int, n_layers: int) -> int:
        """
        Get the window size for a given layer, returning -1 for full attention.
        """
        if self.force_full_attention_on_first_layer and layer_idx == 0:
            return -1
        if self.force_full_attention_on_last_layer and layer_idx == (n_layers - 1):
            return -1

        # Adjust the layer index if the first layer is special-cased to full attention
        # (in which case the pattern is applied starting from the second layer)
        effective_layer_idx = layer_idx
        if self.force_full_attention_on_first_layer:
            effective_layer_idx -= 1

        window_size = self.pattern[effective_layer_idx % len(self.pattern)]
        if window_size <= 0 and window_size != -1:
            raise OLMoConfigurationError(
                f"Sliding window size must be positive or -1 (got {window_size})"
            )
        return window_size

    def should_use_swa(self, layer_idx: int, n_layers: int) -> bool:
        """
        Returns `True` if the given layer uses sliding window attention.
        """
        return self._get_window_size(layer_idx, n_layers) != -1

    def get_window_size(self, layer_idx: int, n_layers: int) -> int:
        """
        Get the sliding window size for a given layer.
        """
        window_size = self._get_window_size(layer_idx, n_layers)
        if window_size == -1:
            raise ValueError(f"Layer {layer_idx} is not configured for sliding window attention.")
        return window_size


class AttentionType(StrEnum):
    """
    An enumeration of the different attention implementations.
    """

    default = "default"
    """
    ➡️ :class:`Attention`
    """
    fused = "fused"
    """
    ➡️ :class:`FusedAttention`
    """
    normalized = "normalized"
    """
    ➡️ :class:`NormalizedAttention`
    """


@dataclass
class AttentionConfig(Config):
    """
    A configuration class for easily building any of the different attention modules.

    See the individual :class:`Attention` subclasses for a description of the configuration options.
    """

    name: AttentionType = AttentionType.default
    """
    The name of the implementation.
    """
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    bias: Optional[bool] = None
    rope: Optional[RoPEConfig] = None
    clip_qkv: Optional[float] = None
    qk_norm: Optional[LayerNormConfig] = None
    dropout: Optional[float] = None
    use_flash: Optional[bool] = None
    use_flex: Optional[bool] = None
    dtype: DType = DType.float32
    sliding_window: Optional[SlidingWindowAttentionConfig] = None
    use_head_qk_norm: Optional[bool] = None
    use_sinks: Optional[bool] = None

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the attention implementation will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = d_model // n_heads
        bias = self.bias if self.bias is not None else self.name != AttentionType.normalized

        params = 0

        # Block attention Q projection.
        params += d_model * d_model
        if bias:
            params += d_model

        # Block attention KV projections.
        params += 2 * d_model * n_kv_heads * head_dim
        if bias:
            params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.qk_norm is not None:
            if self.use_head_qk_norm:
                params += 2 * self.qk_norm.num_params(head_dim)
            else:
                params += 2 * self.qk_norm.num_params(d_model)

        # Block attention out.
        params += d_model * d_model
        if bias:
            params += d_model

        # Block QK scaling factors.
        if self.name == AttentionType.normalized:
            head_dim = d_model // n_heads
            params += n_heads * head_dim
            params += n_kv_heads * head_dim

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "AttentionBase":
        """
        Build the corresponding attention module.

        :param d_model: The model dimensionality.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")

        sliding_window_config: Optional[SlidingWindowAttentionConfig] = kwargs.pop(
            "sliding_window", None
        )
        if sliding_window_config is not None and sliding_window_config.should_use_swa(
            layer_idx, n_layers
        ):
            kwargs["window_size"] = sliding_window_config.get_window_size(layer_idx, n_layers)

        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == "default":
                return Attention(**kwargs)
            elif self.name == "fused":
                kwargs.pop("use_flash", None)
                if kwargs.get("use_flex"):
                    raise OLMoConfigurationError(
                        "Flex attention is not supported with fused attention"
                    )
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with fused attention"
                    )
                return FusedAttention(**kwargs)
            elif self.name == "normalized":
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with normalized attention"
                    )
                return NormalizedAttention(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class AttentionBase(nn.Module):
    """
    Base class for attention modules.
    """

    @abstractmethod
    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        raise NotImplementedError


class Attention(AttentionBase):
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
        use_flex: bool = False,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        use_head_qk_norm: bool = False,
        use_sinks: Optional[bool] = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
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
        self.use_head_qk_norm = use_head_qk_norm
        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if qk_norm is not None:
            if use_head_qk_norm:
                self.q_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
                self.k_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
            else:
                self.q_norm = qk_norm.build(size=d_model, init_device=init_device)
                self.k_norm = qk_norm.build(
                    size=self.n_kv_heads * self.head_dim, init_device=init_device
                )

        self.use_sinks = use_sinks
        if use_sinks is not None and use_sinks:
            self.sinks = nn.Parameter(torch.empty(self.n_heads, dtype=dtype, device=init_device))
        else:
            self.sinks = None

        # Translate window size so that we only look left, not right.
        if window_size is not None:
            if not use_flash and not use_flex:
                raise OLMoConfigurationError(
                    f"'window_size' is only supported with 'use_flash=True' or 'use_flex=True' (got {use_flash=}, {use_flex=})"
                )
            if window_size <= 0:
                raise OLMoConfigurationError(f"'window_size' must be positive (got {window_size})")
            # Flash attn window is [i - window_size[0], i + window_size[1]] inclusive
            self.window_size = (window_size - 1, 0)
        else:
            self.window_size = (-1, -1)

        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )

            # On layers with sliding windows, we don't do rope extension.
            uses_full_attention = self.window_size == (-1, -1)
            uses_sliding_window = not uses_full_attention
            if uses_sliding_window and rope.scaling is not None:
                rope = rope.replace(scaling=None)
            assert not (uses_sliding_window and rope.scaling is not None)

            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_class

        self.use_flash = use_flash
        self.use_flex = use_flex

        self._cp_pg: Optional[dist.ProcessGroup] = None
        self._cp_enabled = False
        self._cp_load_balancer: Optional[RingAttentionLoadBalancerType] = None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        scale: Optional[float] = None,
        block_mask: Optional[BlockMask] = None,
        sinks: Optional[torch.Tensor] = None,
        mask_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        att: torch.Tensor
        if self.cp_enabled:
            assert self._cp_pg is not None and self._cp_load_balancer is not None
            if not self.use_flash:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' requires flash (use_flash=True) for context parallelism"
                )
            if self.use_flex:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' cannot use flex attention for context parallelism"
                )
            att = dispatch_ring_flash_attn(
                q,
                k,
                v,
                group=self._cp_pg,
                strategy=self._cp_load_balancer,
                cu_seqlens=cu_doc_lens,
                cu_seqlens_q=cu_doc_lens_q,
                cu_seqlens_k=cu_doc_lens_k,
                max_seqlen=max_doc_len,
                max_seqlen_q=max_doc_len_q,
                max_seqlen_k=max_doc_len_k,
                heads_k_stride=self._cp_head_stride,
                local_k_slice=local_k_slice,
                dropout_p=self.dropout_p,
                causal=True,
                softmax_scale=scale,
                window_size=(self.window_size, 0) if self.window_size is not None else (-1, -1),
            )
        elif self.use_flash:
            if sinks is not None:
                raise OLMoConfigurationError("Sinks with flash attention is not yet implemented.")
            else:
                att = dispatch_flash_attn(
                    q,
                    k,
                    v,
                    cu_seqlens=cu_doc_lens,
                    cu_seqlens_q=cu_doc_lens_q,
                    cu_seqlens_k=cu_doc_lens_k,
                    max_seqlen=max_doc_len,
                    max_seqlen_q=max_doc_len_q,
                    max_seqlen_k=max_doc_len_k,
                    dropout_p=self.dropout_p,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(self.window_size, 0) if self.window_size is not None else (-1, -1),
                )
        elif self.use_flex:
            if self.dropout_p != 0:
                raise NotImplementedError("Our flex attention does not yet support dropout.")

            if cu_doc_lens is not None or max_doc_len is not None:
                raise NotImplementedError("Intra-document masking not supported in simplified flex.")

            B, S_q, n_heads, head_dim = q.shape
            _, S_kv, n_kv_heads, _ = k.shape
            
            # Transpose to (B, H, S, D) for flex_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            score_mod_fn = None
            mask_mod = None
            if self.sinks is not None:
                num_sink_tokens = 1
                sink_idx = S_kv  # Sink at end
                # Now tensors are (B, H, S, D), so create sinks accordingly
                sink_k = k.new_zeros(B, n_kv_heads, num_sink_tokens, head_dim)
                sink_v = v.new_zeros(B, n_kv_heads, num_sink_tokens, head_dim)
                # Concatenate along sequence dimension (dim=2 now)
                k = torch.cat([k, sink_k], dim=2)
                v = torch.cat([v, sink_v], dim=2)
                kv_len = S_kv + num_sink_tokens

                if self.window_size is not None and self.window_size != (-1, -1):
                    # window_size is a tuple (left, right), we use the left window
                    window = self.window_size[0] if isinstance(self.window_size, tuple) else self.window_size
                    mask_mod = self._get_sliding_window_with_sink_mask_mod(window, sink_idx)
                else:
                    mask_mod = self._get_causal_with_sink_mask_mod(sink_idx)
                block_mask = create_block_mask(
                    mask_mod, B, n_heads, S_q, kv_len, device=q.device.type
                )

                if hasattr(self.sinks, 'to_local'):
                    local_sinks = self.sinks.to_local()
                elif hasattr(self.sinks, '_local_tensor'):
                    local_sinks = self.sinks._local_tensor
                else:
                    local_sinks = self.sinks

                def score_mod_fn(score, batch_idx, head_idx, q_idx, kv_idx):
                    is_sink = kv_idx == sink_idx
                    sink_logit = local_sinks[head_idx]
                    return torch.where(is_sink, sink_logit + 0.0, score)

            else:
                num_sink_tokens = 0
                kv_len = S_kv
                if self.window_size is not None and self.window_size != (-1, -1):
                    # window_size is a tuple (left, right), we use the left window
                    window = self.window_size[0] if isinstance(self.window_size, tuple) else self.window_size
                    mask_mod = self._get_sliding_window_mask_mod(window)
                else:
                    mask_mod = self._get_causal_mask_mod()

                block_mask = create_block_mask(
                    mask_mod, B, n_heads, S_q, kv_len, device=q.device.type
                )

            with torch.autocast(enabled=False, device_type=q.device.type):
                # q, k, v are already (B, H, S, D)
                flex_att = flex_attention(
                    q, k, v, block_mask=block_mask, scale=scale, score_mod=score_mod_fn, enable_gqa=True
                )
                assert isinstance(flex_att, torch.Tensor)
                att = flex_att

            # Transpose back to (B, S, H, D)
            att = att.transpose(1, 2).contiguous()

        else:
            # Fall back to PyTorch's SDPA...
            if any(
                opt is not None
                for opt in (
                    cu_doc_lens,
                    cu_doc_lens_q,
                    cu_doc_lens_k,
                    max_doc_len,
                    max_doc_len_q,
                    max_doc_len_k,
                )
            ):
                raise RuntimeError(
                    f"{self.__class__.__name__} requires flash-attn (use_flash=True) for intra-document masking"
                )
            # NOTE: PyTorch's SDPA doesn't support GQA, so we have to do this.
            # shape: (batch_size, n_heads, seq_len, head_dim)
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
            # shape: (batch_size, n_heads, seq_len, head_dim),
            # (batch_size, n_kv_heads, seq_len, head_dim),
            # (batch_size, n_kv_heads, seq_len, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if sinks is None:
                # shape: (batch_size, n_heads, seq_len, head_dim)
                att = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout_p, is_causal=True, scale=scale
                )
            else:
                batch_size, n_heads, seq_len, _ = q.shape
                attn_logits = torch.matmul(
                    q, k.transpose(2, 3)
                )  # (batch_size, n_heads, seq_len, seq_len)
                if scale is not None:
                    attn_logits *= scale
                if mask_fn is not None:
                    attention_mask = materialize_dense_mask(
                        mask_fn, seq_len, q.device, batch_size, n_heads
                    )
                    attn_logits = attn_logits.masked_fill(~attention_mask, -float("inf"))
                else:
                    causal_mask = torch.triu(
                        q.new_full((seq_len, seq_len), -float("inf")), diagonal=1
                    )
                    attn_logits = attn_logits + causal_mask[None, None, :, :]
                if hasattr(sinks, 'to_local'):
                    local_sinks = sinks.to_local()
                elif hasattr(sinks, '_local_tensor'):
                    local_sinks = sinks._local_tensor
                else:
                    local_sinks = sinks
                if local_sinks.ndim == 1:
                    S = local_sinks.numel()
                    sink_logits = (
                        local_sinks.view(1, 1, 1, S)
                        .to(attn_logits)
                        .expand(batch_size, n_heads, seq_len, S)
                    )
                elif local_sinks.ndim == 2:
                    assert local_sinks.size(0) == n_heads, "Sinks first dim must equal n_heads"
                    S = local_sinks.size(1)
                    sink_logits = (
                        local_sinks.view(1, n_heads, 1, S)
                        .to(attn_logits)
                        .expand(batch_size, n_heads, seq_len, S)
                    )
                else:
                    raise ValueError("Sinks must have shape (S) or (n_heads, S)")
                combined_logits = torch.cat(
                    [attn_logits, sink_logits], dim=-1
                )  # (batch_size, n_heads, head_dim, head_dim + seq_len)
                combined_probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32).to(
                    attn_logits.dtype
                )
                probs = combined_probs[..., :seq_len]
                probs = F.dropout(probs, p=self.dropout_p)
                att = torch.matmul(probs, v)
            # shape: (batch_size, seq_len, n_heads, head_dim)
            att = att.transpose(1, 2).contiguous()
        return att
    
    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        mask_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.

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

        if not self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        # NOTE: use -1 instead of `n_heads` / `n_kv_heads` to infer actual local size when
        # using tensor parallelism.
        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            q, k = self.rope(
                q, k, head_first=False, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis
            )
        assert (
            not self.use_flex or block_mask is not None
        ), "Block mask cannot be null for flex attention layer"

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            block_mask=block_mask,
            sinks=self.sinks if self.use_sinks else None,
            mask_fn=mask_fn,
        )

        # shape: (batch_size, seq_len, d_model)
        att = att.reshape(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return causal_mask

    @staticmethod
    def _get_sliding_window_mask_mod(window: int) -> _mask_mod_signature:
        def sliding_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx - kv_idx <= window)
        return sliding_mod

    @staticmethod
    def _get_causal_with_sink_mask_mod(sink_idx: int) -> _mask_mod_signature:
        orig = Attention._get_causal_mask_mod()
        def causal_with_sink(b, h, q_idx, kv_idx):
            return orig(b, h, q_idx, kv_idx) | (kv_idx == sink_idx)
        return causal_with_sink

    @staticmethod
    def _get_sliding_window_with_sink_mask_mod(window: int, sink_idx: int) -> _mask_mod_signature:
        def sliding_mod(b, h, q_idx, kv_idx):
            keep = (kv_idx <= q_idx) & (q_idx - kv_idx <= window)
            return keep | (kv_idx == sink_idx)
        return sliding_mod

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        plan = {
            "w_q": colwise_parallel(
                output_layouts=None if self.q_norm is None else Shard(1),
                use_local_output=self.q_norm is None,
            ),
            "w_k": colwise_parallel(
                output_layouts=None if self.k_norm is None else Shard(1),
                use_local_output=self.k_norm is None,
            ),
            "w_v": colwise_parallel(),
            "w_out": rowwise_parallel(
                output_layouts=output_layout, use_local_output=use_local_output
            ),
        }
        if self.q_norm is not None:
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))
        if self.k_norm is not None:
            plan["k_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))
        if self.sinks is not None:
            from torch.distributed.tensor import distribute_tensor
            self.sinks = nn.Parameter(distribute_tensor(self.sinks.data, tp_mesh, [Shard(0)]))
        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        """
        Prepare the module for context-parallelism (ring attention).

        .. important::
            This requires flash-attn and ring-flash-attn (``use_flash=True``).

        :param cp_mesh: The context parallel device sub-mesh.
        :param load_balancer: The load balancer type.
        """
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True
        self._cp_head_stride = head_stride


@beta_feature
class NormalizedAttention(Attention):
    """
    An nGPT attention implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        rope: Optional[RoPEConfig] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        use_flash: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope=rope,
            qk_norm=qk_norm,
            use_flash=use_flash,
            bias=False,
            dtype=dtype,
            init_device=init_device,
            cache=cache,
        )

        self.sq_init_value = 1.0
        self.sq_init_scaling = 1.0 / math.sqrt(d_model)
        self.sq = nn.Parameter(
            torch.empty(self.head_dim * self.n_heads, dtype=dtype, device=init_device)
        )

        self.sk_init_value = 1.0
        self.sk_init_scaling = 1.0 / math.sqrt(d_model)
        self.sk = nn.Parameter(
            torch.empty(self.head_dim * self.n_kv_heads, dtype=dtype, device=init_device)
        )

        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sq)
        nn.init.ones_(self.sk)
        with torch.no_grad():
            self.sq.mul_(self.sq_init_scaling)
            self.sk.mul_(self.sk_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        sq = (self.sq * (self.sq_init_value / self.sq_init_scaling)).view(1, 1, -1)
        q = sq * q

        sk = (self.sk * (self.sk_init_value / self.sk_init_scaling)).view(1, 1, -1)
        k = sk * k

        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )
            q, k = self.rope(
                q, k, head_first=False, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis
            )

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            scale=self.sqrt_head_dim,
        )

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError("TP is not implemented yet for the normalized attention variant")

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        self._normalize_matrix(self.w_q.weight)
        self._normalize_matrix(self.w_k.weight)
        self._normalize_matrix(self.w_v.weight)
        self._normalize_matrix(self.w_out.weight, dim=0)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))


class FusedAttention(AttentionBase):
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

        self._cp_pg: Optional[dist.ProcessGroup] = None
        self._cp_enabled = False
        self._cp_load_balancer: Optional[RingAttentionLoadBalancerType] = None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
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
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )
            qkv = self.rope(qkv, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis)

        if self.cp_enabled:
            assert self._cp_pg is not None and self._cp_load_balancer is not None
            att = dispatch_ring_flash_attn_qkvpacked(
                qkv,
                group=self._cp_pg,
                strategy=self._cp_load_balancer,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                dropout_p=self.dropout_p,
                causal=True,
            )
        else:
            att = dispatch_flash_attn_qkvpacked(
                qkv,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                dropout_p=self.dropout_p,
                causal=True,
            )

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)  # type: ignore

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError("TP is not implemented yet for the fused attention variant")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True
        self._cp_head_stride = head_stride


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# def _get_flex_attn_mask_mod(
#     window_size: Optional[Tuple[int, int]] = None,
#     doc_lens: Optional[Tuple[int, ...]] = None,
#     device: Optional[torch.device] = None,
#     num_sink_tokens: int = 0,
# ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
#     if device is None:
#         raise ValueError("Device is required")

#     has_window = window_size is not None and window_size != (-1, -1)
#     has_docs = doc_lens is not None

#     if has_docs:
#         document_ids = torch.cat(
#             [torch.full((int(doc_len),), i, device=device, dtype=torch.long) for i, doc_len in enumerate(doc_lens)]
#         )

#     def total_mask_mod(B: torch.Tensor, H: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
#         is_sink = kv_idx < num_sink_tokens
#         adjusted_kv_idx = kv_idx - num_sink_tokens
#         is_regular = kv_idx >= num_sink_tokens
#         causal_mask = q_idx >= adjusted_kv_idx
        
#         if has_window:
#             window_mask = (q_idx - adjusted_kv_idx <= window_size[0]) & (adjusted_kv_idx - q_idx <= window_size[1])  # type: ignore
#         else:
#             window_mask = torch.ones_like(causal_mask, dtype=torch.bool)
        
#         if has_docs:
#             clamped_idx = torch.clamp(adjusted_kv_idx, min=0, max=len(document_ids) - 1)
#             doc_mask = document_ids[q_idx] == document_ids[clamped_idx]
#         else:
#             doc_mask = torch.ones_like(causal_mask, dtype=torch.bool)
        
#         regular_mask = causal_mask & window_mask & doc_mask
#         return is_sink | (is_regular & regular_mask)

#     return total_mask_mod



def _get_flex_attn_mask_mod(
    window_size: Optional[Tuple[int, int]] = None,
    doc_lens: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,
    num_sink_tokens: int = 0,
    seq_len: Optional[int] = None,  # Need seq_len to know where sinks start
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if device is None:
        raise ValueError("Device is required")

    has_window = window_size is not None and window_size != (-1, -1)
    has_docs = doc_lens is not None

    document_ids = None
    if has_docs:
        document_ids = torch.cat(
            [torch.full((int(doc_len),), i, device=device, dtype=torch.long) for i, doc_len in enumerate(doc_lens)]
        )

    def total_mask_mod(B: torch.Tensor, H: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        # Sinks are at the BEGINNING (first num_sink_tokens positions)
        is_sink = kv_idx < num_sink_tokens
        
        # For regular tokens, adjust kv_idx 
        adjusted_kv_idx = kv_idx - num_sink_tokens
        
        # Basic causal mask for regular tokens
        is_regular = kv_idx >= num_sink_tokens
        causal = adjusted_kv_idx <= q_idx
        
        # Build mask step by step
        regular_mask = is_regular & causal
        
        # Apply sliding window if specified
        if has_window and window_size is not None:
            # Only keep tokens within window
            within_window = (q_idx - adjusted_kv_idx) <= window_size[0]
            regular_mask = regular_mask & within_window
        
        # Apply document boundaries if specified
        if has_docs and document_ids is not None:
            # Clamp indices to valid range
            max_idx = len(document_ids) - 1
            q_clamped = torch.clamp(q_idx, min=0, max=max_idx)
            kv_clamped = torch.clamp(adjusted_kv_idx, min=0, max=max_idx)
            
            # Check if in same document (only for regular tokens)
            same_doc = document_ids[q_clamped] == document_ids[kv_clamped]
            
            # Apply doc constraint only to regular tokens
            regular_mask = regular_mask & ((kv_idx < num_sink_tokens) | same_doc)
        
        # Final mask: always allow sinks OR regular tokens that pass all constraints
        return is_sink | regular_mask
    return total_mask_mod


def _get_flex_attn_causal_block_mask(
    seq_len: int,
    device: torch.device,
    window_size: Optional[Tuple[int, int]] = None,
    doc_lens: Optional[Tuple[int, ...]] = None,
    block_size: int = 128,
    num_sink_tokens: int = 0,
) -> BlockMask:
    if doc_lens is not None:
        token_count = int(sum(doc_lens))
        if token_count % seq_len != 0:
            raise ValueError("Sum of document lengths is not a multiple of sequence length")

        # For intra-document masking, we merge the batch size dimension into the sequence dimension.
        return create_block_mask(
            _get_flex_attn_mask_mod(window_size, doc_lens=doc_lens, device=device, 
                                    num_sink_tokens=num_sink_tokens, seq_len=token_count),
            B=1,
            H=None,
            Q_LEN=token_count,
            KV_LEN=token_count + num_sink_tokens,
            device=device.type,
            BLOCK_SIZE=block_size,
        )

    else:
        return create_block_mask(
            _get_flex_attn_mask_mod(window_size, device=device, num_sink_tokens=num_sink_tokens, seq_len=seq_len),
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len + num_sink_tokens,
            device=device.type,
            BLOCK_SIZE=block_size,
        )


def get_flex_attn_causal_block_mask(
    seq_len: int,
    device: torch.device,
    window_size: Optional[Tuple[int, int]] = None,
    doc_lens: Optional[torch.Tensor] = None,
    block_size: int = 128,
    return_mask_fn: bool = False,
    num_sink_tokens: int = 0,
) -> Union[BlockMask, Tuple[BlockMask, Callable]]:
    if doc_lens is not None:
        doc_lens_list = tuple(doc_lens.flatten().tolist())
        mask_fn = _get_flex_attn_mask_mod(window_size, doc_lens=doc_lens_list, device=device, 
                                          num_sink_tokens=num_sink_tokens, seq_len=seq_len)
        block_mask = _get_flex_attn_causal_block_mask(
            seq_len, device, window_size, doc_lens_list, block_size, num_sink_tokens=num_sink_tokens
        )
    else:
        mask_fn = _get_flex_attn_mask_mod(window_size, device=device, num_sink_tokens=num_sink_tokens, seq_len=seq_len)
        block_mask = _get_flex_attn_causal_block_mask(
            seq_len, device, window_size, doc_lens=None, block_size=block_size, num_sink_tokens=num_sink_tokens
        )

    if return_mask_fn:
        return block_mask, mask_fn
    return block_mask


def materialize_dense_mask(
    mask_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    seq_len: int,
    device: torch.device,
    batch_size: int = 1,
    n_heads: int = 1,
) -> torch.Tensor:
    q_indices = torch.arange(seq_len, device=device)
    kv_indices = torch.arange(seq_len, device=device)
    q_idx, kv_idx = torch.meshgrid(q_indices, kv_indices, indexing="ij")
    B = torch.zeros(1, device=device, dtype=torch.long)
    H = torch.zeros(1, device=device, dtype=torch.long)
    dense_mask = mask_fn(B, H, q_idx, kv_idx)
    dense_mask = dense_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)
    return dense_mask
