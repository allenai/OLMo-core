import logging
import math
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.kv_cache import KVCacheManager

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
from . import flash_attn_api
from .backend import (
    AttentionBackend,
    AttentionBackendName,
    FlashAttention2Backend,
    FlashAttention3Backend,
    TEAttentionBackend,
    TorchAttentionBackend,
)
from .ring import (
    RingAttentionLlama3LoadBalancer,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
)
import nvtx


__all__ = [
    "SlidingWindowAttentionConfig",
    "AttentionType",
    "AttentionBackendName",
    "AttentionBackend",
    "TorchAttentionBackend",
    "FlashAttention2Backend",
    "FlashAttention3Backend",
    "TEAttentionBackend",
    "AttentionConfig",
    "AttentionBase",
    "Attention",
    "FusedAttention",
    "NormalizedAttention",
    "RingAttentionLoadBalancerType",
    "RingAttentionLoadBalancer",
    "RingAttentionZigZagLoadBalancer",
    "RingAttentionLlama3LoadBalancer",
    "MultiheadLatentAttentionConfig",
    "MultiheadLatentAttention",
]

log = logging.getLogger(__name__)


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
    mla = "mla"
    """
    ➡️ :class:`MultiheadLatentAttention`
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
    backend: Optional[AttentionBackendName] = None
    dtype: DType = DType.float32
    sliding_window: Optional[SlidingWindowAttentionConfig] = None
    d_attn: Optional[int] = None

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the attention implementation will have once built.

        :param d_model: The model dimensionality.
        """

        if self.d_attn is None:
            d_attn = d_model
        else:
            d_attn = self.d_attn

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = d_attn // n_heads # not assuming d_model here
        bias = self.bias if self.bias is not None else self.name != AttentionType.normalized

        params = 0

        # Block attention Q projection. (input = d_model, output = d_attn)
        params += d_model * d_attn
        if bias:
            params += d_attn

        # Block attention KV projections. (input = d_model, output = n_kv_heads * head_dim)
        params += 2 * d_model * n_kv_heads * head_dim
        if bias:
            params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.qk_norm is not None:
            params += 2 * self.qk_norm.num_params(d_model)

        # Block attention out.
        params += d_attn * d_model # input = d_attn, output = d_model
        if bias:
            params += d_model

        # Block QK scaling factors.
        if self.name == AttentionType.normalized:
            head_dim = d_attn // n_heads
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

    def flops_per_seq(self, d_model: int, seq_length: int) -> int:
        """
        Estimate the number of FLOPs for a single forward + backward pass through attention
        for a sequence of the given length.

        :param d_model: The model dimensionality.
        :param seq_length: The sequence length.
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        d_attn = self.d_attn if self.d_attn is not None else d_model

        head_dim = d_attn // n_heads

        # 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        flops_factor = 2 * 3

        flops = 0

        # Q projection (seq_length, d_model) x (d_model, d_attn)
        flops += flops_factor * (d_model * d_attn * seq_length)

        # K, V projections (seq_length, d_model) x (d_model, n_kv_heads * head_dim) x 2
        flops += flops_factor * (d_model * n_kv_heads * head_dim * seq_length) * 2


        # QK^T scores:
        # Treat as n_heads independent GEMMs:
        # (seq_length, head_dim) x (head_dim, seq_length) for each head.
        # mnk = seq_length * seq_length * head_dim, repeated n_heads times.
        flops += flops_factor * (n_heads * seq_length * seq_length * head_dim) / 2 # divide by 2 for causal attention

        # Attention @ V:
        # Again n_heads independent GEMMs:
        # (seq_length, seq_length) x (seq_length, head_dim) for each head.
        flops += flops_factor * (n_heads * seq_length * seq_length * head_dim) / 2 # divide by 2 for causal attention

        # Output projection (seq_length, d_attn) x (d_attn, d_model)
        flops += flops_factor * (d_attn * d_model * seq_length)

        # NOTE: We ignore non-matmul ops (softmax, masking, RoPE, qk_norm, dropout, etc.)
        # since they are relatively small compared to the GEMMs above.

        return flops


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
    ``max_doc_len`` and ``cu_doc_lens`` parameters to :meth:`forward()`. This requires
    a backend that supports it, like the flash backend.

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
    :param use_flash: Deprecated, use ``backend="flash_2"`` instead.
    :param backend: The attention backend to use. If not set, it will be chosen automatically.
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
        softmax_scale: Optional[float] = None,
        use_flash: Optional[bool] = None,
        backend: Optional[AttentionBackendName] = None,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        use_head_qk_norm: bool = False,
        d_attn: Optional[int] = None,
    ):
        super().__init__()

        if d_attn is None:
            d_attn = d_model

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
        self.w_out = nn.Linear(d_attn, d_model, bias=bias, dtype=dtype, device=init_device)
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

        if backend is not None:
            backend = AttentionBackendName(backend)

        if use_flash:
            if backend is not None and backend != AttentionBackendName.flash_2:
                raise OLMoConfigurationError(
                    f"'use_flash' is only compatible with 'flash_2' backend (got '{backend}')"
                )
            elif backend is None:
                warnings.warn(
                    "'use_flash' is deprecated, use 'backend=flash_2' instead", DeprecationWarning
                )
                backend = AttentionBackendName.flash_2

        # Translate window size so that we only look left, not right.
        window_size_tuple: Tuple[int, int] = (-1, -1)
        if window_size is not None:
            if window_size <= 0:
                raise OLMoConfigurationError(f"'window_size' must be positive (got {window_size})")

            if backend is None and flash_attn_api.has_flash_attn_2():
                # note: flash_3 and te backends are faster than flash_2 and also support SWA
                backend = AttentionBackendName.flash_2

            # Window size is [i - window_size[0], i + window_size[1]] inclusive
            window_size_tuple = (window_size - 1, 0)

        if backend is None:
            backend = AttentionBackendName.torch

        backend.assert_supported()
        log.info(f"Using attention backend '{backend}'")
        self.backend = backend.build(
            head_dim=self.head_dim,
            n_heads=n_heads,
            n_kv_heads=self.n_kv_heads,
            scale=softmax_scale,
            dropout_p=dropout,
            window_size=window_size_tuple,
            cache=cache,
        )
        self.kv_cache_manager: Optional[KVCacheManager] = None

    @property
    def cp_enabled(self) -> bool:
        return self.backend.cp_enabled

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
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.record_leftpad(cache_leftpad)
        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = self.backend(
            (q, k, v),
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            kv_cache_manager=self.kv_cache_manager,
        )
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.update_seqlen(q.shape[1])
        return att

    @nvtx.annotate("Attention.forward", color='red')
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
        cache_leftpad: Optional[torch.Tensor] = None,
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
        # shape: (batch_size, seq_len, n_heads (local), head_dim)
        q = q.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        k = k.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.rope is not None:
            # In context-parallel mode we must be given pre-sharded buffers
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None
            q, k = self.rope(
                q,
                k,
                head_first=False,
                start_pos=start_pos,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
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
            cache_leftpad=cache_leftpad,
        )

        # shape: (batch_size, seq_len, d_attn)
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
            # if full-dim norm: output is sharded on the embedding dimension (B, T, E [sharded])
            #    which will be reshaped into (B, T, H [sharded], D)
            # if head-wise norm: output is sharded on the head dimension (B, T, H [sharded], D)
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))
        if self.k_norm is not None:
            plan["k_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))

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
            This requires a backend that supports CP, such as "flash_2" or "te".

        :param cp_mesh: The context parallel device sub-mesh.
        :param load_balancer: The load balancer type.
        """
        self.backend.apply_cp(cp_mesh, load_balancer, head_stride=head_stride)

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        """
        Initialize the kv cache manager for attention. When the kv cache manager exists,
        kv caching will be used during the forward pass. This should only be called during inference.

        :param batch_size: The batch size for the cache.
        :param max_seq_len: The maximum sequence length for the cache.
        """
        self.backend.assert_supports_kv_cache()
        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.weight.device,
        )


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
        use_flash: Optional[bool] = None,
        backend: Optional[AttentionBackendName] = None,
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
            backend=backend,
            softmax_scale=math.sqrt(d_model // n_heads),
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
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache_leftpad:
            raise NotImplementedError(
                "cache_leftpad is not supported for the normalized attention variant"
            )

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

            start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None
            q, k = self.rope(
                q,
                k,
                head_first=False,
                start_pos=start_pos,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
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
            cache_leftpad=cache_leftpad,
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
        Currently this is only supported with the "flash_2" backend.

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
        backend: Optional[AttentionBackendName] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=bias, dtype=dtype, device=init_device)
        self.w_out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.clip_qkv = clip_qkv
        self.rope: Optional[FusedRotaryEmbedding] = None
        if rope is not None:
            if rope.name != "fused":
                raise OLMoConfigurationError(f"{self.__class__.__name__} requires fused RoPE")
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, FusedRotaryEmbedding)
            self.rope = rope_class

        if backend is not None:
            backend = AttentionBackendName(backend)
        elif backend is None:
            backend = AttentionBackendName.flash_2

        backend.assert_supported()
        backend.assert_supports_packed_qkv()
        log.info(f"Using attention backend '{backend}'")
        self.backend = backend.build(
            head_dim=self.head_dim, n_heads=self.n_heads, dropout_p=dropout, cache=cache
        )

    @property
    def cp_enabled(self) -> bool:
        return self.backend.cp_enabled

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
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
        if cache_leftpad:
            raise NotImplementedError(
                "cache_leftpad is not supported for the fused attention variant"
            )

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

        att = self.backend(
            qkv,
            cu_doc_lens=cu_doc_lens,
            max_doc_len=max_doc_len,
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
        self.backend.apply_cp(cp_mesh, load_balancer, head_stride=head_stride)



@dataclass
class MultiheadLatentAttentionConfig(AttentionConfig):
    """
    A configuration class for Multi-head Latent Attention (MLA).

    This attention mechanism uses low-rank projections for queries and key/values,
    splitting the queries into non-positional and rotary components.
    """
    name: str = "mla"
    # n_heads: int = 16
    # bias: Optional[bool] = None
    # dropout: Optional[float] = 0.0
    # dtype: DType = DType.float32
    # rope: Optional[RoPEConfig] = None
    # use_flash: Optional[bool] = None

    # MLA specific parameters
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # rename qk_norm -> qkv_norm
    qkv_norm: Optional[LayerNormConfig] = None

    def num_params(self, d_model: int) -> int:
        params = 0
        has_bias = self.bias is not False

        # Queries
        if self.q_lora_rank == 0:
            # Direct query projection
            params += d_model * self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            if has_bias:
                params += self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        else:
            # Low-rank query projections
            # Down-projection wq_a
            params += d_model * self.q_lora_rank
            if has_bias:
                params += self.q_lora_rank
            # LayerNorm on latent query
            if self.qkv_norm is not None:
                params += self.qkv_norm.num_params(self.q_lora_rank)
            # Up-projection wq_b
            params += self.q_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            if has_bias:
                params += self.n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)

        # Key/value projections
        # Down-projection wkv_a
        params += d_model * (self.kv_lora_rank + self.qk_rope_head_dim)
        if has_bias:
            params += (self.kv_lora_rank + self.qk_rope_head_dim)
        # LayerNorm on latent key/value
        if self.qkv_norm is not None:
            params += self.qkv_norm.num_params(self.kv_lora_rank)
        # Up-projection wkv_b
        params += self.kv_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        if has_bias:
            params += self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)

        # Output projection
        params += self.n_heads * self.v_head_dim * d_model
        if has_bias:
            params += d_model

        return params

    def build(
        self,
        d_model: int,
        *,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "MultiheadLatentAttention":
        assert self.qk_norm is None, "qk_norm is not used in MLA, use qkv_norm instead"
        assert self.clip_qkv is None, "clip_qkv is not supported"
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )
        return MultiheadLatentAttention(**kwargs)


class MultiheadLatentAttention(AttentionBase):
    """ 
    Multi-head Latent Attention (MLA) implementation.

    This attention mechanism uses low-rank projections for queries and key/values,
    splitting the queries into non-positional and rotary components.
    
    #TODO: Update docstring to include all parameters
    """
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        rope: Optional[RoPEConfig] = None,
        dropout: float = 0.0,
        use_flash: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        qkv_norm: Optional[LayerNormConfig] = None,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
    ):
        super().__init__()
        assert qkv_norm is not None, "qkv_norm need be provided for MLA" # TODO: support MLA without norm at latent Q and KV
        
        self.n_rep = 1 # no GQA support for MLA
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.d_model = d_model
        self.bias = bias
        self.qk_nope_head_dim = qk_nope_head_dim # non-positional query/key dim
        self.qk_rope_head_dim = qk_rope_head_dim # rotary query/key dim
        
        self.v_head_dim = v_head_dim # value head dim
        
        self.softmax_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

        if q_lora_rank == 0: # no low-rank projection for queries
            self.wq = nn.Linear(d_model, n_heads * (qk_nope_head_dim + qk_rope_head_dim), bias=bias, dtype=dtype, device=init_device)
        else:
            # the low rank down-projection
            self.wq_a = nn.Linear(d_model, q_lora_rank, bias=bias, dtype=dtype, device=init_device)
            
            # the layer norm applied to the latent query
            self.q_norm = qkv_norm.build(size=q_lora_rank, init_device=init_device)
            
            # the low rank up-projection, including the non-positional and positional parts. We merge two weight matrices into one to improve performance.
            self.wq_b = nn.Linear(q_lora_rank, n_heads * (qk_nope_head_dim + qk_rope_head_dim), bias=bias, dtype=dtype, device=init_device)

        # the low rank down-projection for key/value + rotary key
        # Note that the rotary key is not low-rank, 
        # and for this part it does not have a concept of head (it will be expanded to the number of heads later)
        # we merge the two weight matrices into one to improve performance.
        self.wkv_a = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=bias, dtype=dtype, device=init_device)
        
        # the layer norm applied to the latent key/value
        self.kv_norm = qkv_norm.build(size=kv_lora_rank, init_device=init_device)
        
        # the low rank up-projection, including the non-positional key and value parts.
        self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=bias, dtype=dtype, device=init_device)
        
        self.w_out = nn.Linear(n_heads * v_head_dim, d_model, bias=bias, dtype=dtype, device=init_device) # output projection

        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )
            rope_class = rope.build(self.qk_rope_head_dim, cache=cache)
            assert isinstance(rope_class, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_class
            
        self.use_flash = use_flash
        self._cp_pg: Optional[dist.ProcessGroup] = None
        self._cp_enabled = False
        self._cp_load_balancer: Optional[RingAttentionLoadBalancerType] = None

    # @nvtx.annotate("MLA.forward", color=NVTX_COLOR)
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
        """
        Apply multi-head latent attention.

        :param x: Input tensor of shape (B, T, d_model).
        :param start_pos: Starting position in the sequence.
        :param freqs_cis: Precomputed rotary embeddings.
        :param mask: Optional attention mask.
        :returns: Tensor of shape (B, T, d_model).
        """

        B, T, _ = x.shape

        # Compute query projections
        q: torch.Tensor
        if self.q_lora_rank == 0:
            # No low-rank projection for queries
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(B, T, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Compute key and value projections
        kv_out = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv_out, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # rotary part of the key and query
        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            q_pe, k_pe = self.rope(
                q_pe, 
                k_pe.unsqueeze(2),  # shape: (B, T, 1, qk_rope_head_dim)
                head_first=False, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis
            )
        else:
            raise RuntimeError("RoPE is required for MLA")
        
        # shape: (B, T, 1, qk_rope_head_dim) -> (B, T, n_heads, qk_rope_head_dim)
        k_pe_expanded = k_pe.expand(-1, -1, self.n_heads, -1)
        
        k_nope_and_v = self.wkv_b(self.kv_norm(kv_latent)) # the up-projection of the latent key (nope part) and value
        k_nope_and_v = k_nope_and_v.view(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(k_nope_and_v, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Combine key parts
        k_combined = torch.cat([k_nope, k_pe_expanded], dim=-1)  # shape: (B, T, n_heads, qk_nope_head_dim + qk_rope_head_dim)
        
        # Combine query parts
        q_combined = torch.cat([q_nope, q_pe], dim=-1) # shape: (B, T, n_heads, qk_nope_head_dim + qk_rope_head_dim)
        
        # Q and K are now of shape (B, T, n_heads, qk_nope_head_dim + qk_rope_head_dim)
        # V is of shape (B, T, n_heads, v_head_dim)
        assert q_combined.shape == (B, T, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        assert k_combined.shape == (B, T, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        assert v.shape == (B, T, self.n_heads, self.v_head_dim)
        
        
        # Compute attention scores
        attn = self.sdpa(
            q_combined,
            k_combined,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            scale=self.softmax_scale,
        )
        
        # shape: (B, T, n_heads, v_head_dim)
        attn = attn.view(B, T, -1)
        
        # shape: (B, T, d_model)
        return self.w_out(attn)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(float8_enabled=float8_enabled)
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )
        plan = {
            "wq" if hasattr(self, "wq") else "wq_b": colwise_parallel(),
            "wkv_b": colwise_parallel(),
            "w_out": rowwise_parallel(output_layout=output_layout, use_local_output=use_local_output),
        }
        if hasattr(self, "wq_a"):
            plan["wq_a"] = colwise_parallel()
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(-1))
        if hasattr(self, "qkv_norm"):
            plan["qkv_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(-1))
        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True

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
    ) -> torch.Tensor:
        """
        Apply scaled dot-product attention. Copies the Attention sdpa.
        """
        
        att: torch.Tensor
        if self.cp_enabled:
            assert self._cp_pg is not None and self._cp_load_balancer is not None
            if not self.use_flash:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' requires flash (use_flash=True) for context parallelism"
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
                heads_k_stride=1,  # TODO: should this ever not be 1?
                local_k_slice=local_k_slice,
                dropout_p=self.dropout_p,
                causal=True,
                softmax_scale=scale,
            )
        elif self.use_flash:
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
            )
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
            #        (batch_size, n_kv_heads, seq_len, head_dim),
            #        (batch_size, n_kv_heads, seq_len, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # shape: (batch_size, n_heads, seq_len, head_dim)
            att = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p, is_causal=True, scale=scale
            )

            # shape: (batch_size, seq_len, n_heads, head_dim)
            att = att.transpose(1, 2).contiguous()

        return att