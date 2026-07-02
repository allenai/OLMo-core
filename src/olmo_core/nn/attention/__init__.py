import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNetConfig

from ..buffer_cache import BufferCache
from ..config import ModuleConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNorm, LayerNormConfig
from ..rope import ComplexRotaryEmbedding, FusedRotaryEmbedding, RoPEConfig, RotaryEmbedding
from ..utils import get_tp_wrappers
from . import flash_attn_api
from .backend import (
    AttentionBackend,
    AttentionBackendName,
    FlashAttention2Backend,
    FlashAttention3Backend,
    FlashAttention4Backend,
    TEAttentionBackend,
    TorchAttentionBackend,
)
from .landmark import build_landmark_masks, landmark_grouped_softmax, repeat_kv
from .landmark_kernel import fused_landmark_attention, has_landmark_kernel
from .ring import (
    RingAttentionLlama3LoadBalancer,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
    UlyssesLoadBalancer,
)

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = [
    "SlidingWindowAttentionConfig",
    "AttentionTypePatternConfig",
    "GateGranularity",
    "GateConfig",
    "AttentionType",
    "AttentionBackendName",
    "AttentionBackend",
    "TorchAttentionBackend",
    "FlashAttention2Backend",
    "FlashAttention3Backend",
    "FlashAttention4Backend",
    "TEAttentionBackend",
    "AttentionConfig",
    "Attention",
    "FusedAttention",
    "NormalizedAttention",
    "LandmarkAttention",
    "FastLandmarkAttention",
    "FastCompressiveLandmarkAttention",
    "SparseLandmarkAttention",
    "RingAttentionLoadBalancerType",
    "RingAttentionLoadBalancer",
    "RingAttentionZigZagLoadBalancer",
    "RingAttentionLlama3LoadBalancer",
    "UlyssesLoadBalancer",
    "RingContextParallelStyle",
    "UlyssesContextParallelStyle",
    "GatedDeltaNetConfig",
    "GatedDeltaNet",
]

log = logging.getLogger(__name__)


class GateGranularity(StrEnum):
    headwise = "headwise"
    """Head-wise gating: one gate value per attention head, broadcast across head dimension."""
    elementwise = "elementwise"
    """Element-wise gating: one gate value per output element."""


@dataclass
class GateConfig(Config):
    granularity: GateGranularity = GateGranularity.headwise
    """The granularity of gating to use."""
    full_precision: bool = True
    """Whether to always apply gating in full precision regardless of the input data type."""


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
    landmark = "landmark"
    """
    ➡️ :class:`LandmarkAttention`
    """

    fast_landmark = "fast_landmark"
    """
    ➡️ :class:`FastLandmarkAttention` (landmark attention with the optimized FA2-style kernel)
    """

    fast_compressive_landmark = "fast_compressive_landmark"
    """
    ➡️ :class:`FastCompressiveLandmarkAttention` (fast landmark attention that also folds each
    block's landmark token into the attention output -- a compressed summary of the block)
    """

    sparse_landmark = "sparse_landmark"
    """
    ➡️ :class:`SparseLandmarkAttention` (sparse landmark-only-across-chunks attention)
    """

    shared_vector_landmark = "shared_vector_landmark"
    """
    ➡️ :class:`SharedVectorLandmarkAttention` (fast landmark attention that appends a learned,
    per-block positional vector to every value before the attention aggregation)
    """


@dataclass
class AttentionTypePatternConfig(Config):
    """
    Specifies the :class:`AttentionType` to use on a per-layer basis, mirroring the format of
    :class:`SlidingWindowAttentionConfig`. The ``pattern`` is repeated to cover all layers, so a
    single model can mix attention implementations (e.g. full attention with landmark variants).

    The shared attention parameters (``mem_freq``, ``num_landmarks``, etc.) are taken from the
    enclosing :class:`AttentionConfig`; this pattern only selects the *type* per layer.
    """

    pattern: List[AttentionType]
    """
    The pattern of attention types to use, repeated to cover all layers. For example, a pattern of
    ``[fast_landmark, fast_landmark, fast_landmark, sparse_landmark]`` means that for each set of 4
    layers, the first 3 use fast landmark attention and the last uses sparse landmark attention.
    """

    force_full_attention_on_first_layer: bool = False
    """
    If `True`, the first transformer layer will always use :data:`AttentionType.default` (full
    attention), regardless of the pattern.
    """

    force_full_attention_on_last_layer: bool = False
    """
    If `True`, the last transformer layer will always use :data:`AttentionType.default` (full
    attention), regardless of the pattern.
    """

    def get_type(self, layer_idx: int, n_layers: int) -> AttentionType:
        """
        Get the :class:`AttentionType` for a given layer.
        """
        if self.force_full_attention_on_first_layer and layer_idx == 0:
            return AttentionType.default
        if self.force_full_attention_on_last_layer and layer_idx == (n_layers - 1):
            return AttentionType.default

        # Adjust the layer index if the first layer is special-cased to full attention
        # (in which case the pattern is applied starting from the second layer).
        effective_layer_idx = layer_idx
        if self.force_full_attention_on_first_layer:
            effective_layer_idx -= 1

        return AttentionType(self.pattern[effective_layer_idx % len(self.pattern)])

    def landmark_types(self) -> Set[AttentionType]:
        """
        The set of landmark attention types that appear anywhere in the pattern.
        """
        return {t for t in self.pattern if t in _LANDMARK_ATTENTION_TYPES}


_LANDMARK_ATTENTION_TYPES = (
    AttentionType.landmark,
    AttentionType.fast_landmark,
    AttentionType.fast_compressive_landmark,
    AttentionType.sparse_landmark,
    AttentionType.shared_vector_landmark,
)


@SequenceMixerConfig.register("attention")
@dataclass
class AttentionConfig(SequenceMixerConfig["SequenceMixer"]):
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
    head_dim: Optional[int] = None
    bias: Optional[bool] = None
    gate: Optional[GateConfig] = None
    rope: Optional[RoPEConfig] = None
    clip_qkv: Optional[float] = None
    qk_norm: Optional[LayerNormConfig] = None
    dropout: Optional[float] = None
    use_flash: Optional[bool] = None
    backend: Optional[AttentionBackendName] = None
    dtype: DType = DType.float32
    sliding_window: Optional[SlidingWindowAttentionConfig] = None
    layer_types: Optional[AttentionTypePatternConfig] = None
    """
    Optionally select the :class:`AttentionType` on a per-layer basis. When set, this overrides
    :data:`name` for each layer (which is then only used as a fallback). The shared landmark
    parameters below (``mem_freq``, ``num_landmarks``, etc.) apply to whichever layers resolve to a
    landmark variant.
    """
    use_head_qk_norm: Optional[bool] = None
    mem_freq: Optional[int] = None
    """
    The number of regular tokens between landmark tokens, used only by
    :class:`LandmarkAttention` (``name="landmark"``). The landmark block size is ``mem_freq + 1``.
    """
    landmark_use_kernel: Optional[bool] = None
    """
    For :class:`LandmarkAttention` only: use the fused Triton kernel instead of the eager path.
    Defaults to ``False`` (eager). See :class:`LandmarkAttention` for the caveats.
    """
    num_landmarks: Optional[int] = None
    """
    For :class:`SparseLandmarkAttention` (``name="sparse_landmark"``) only: number of landmark
    tokens per chunk (the last ``num_landmarks`` tokens of each chunk). Defaults to 1.
    """
    nonselected_landmark_mass: Optional[float] = None
    """
    For :class:`FastCompressiveLandmarkAttention` (``name="fast_compressive_landmark"``) only: the
    fraction of attention mass reserved at top-k decode time for the landmark tokens of the
    non-selected blocks. Defaults to 0.1. See :class:`FastCompressiveLandmarkAttention`.
    """
    vec_dim: Optional[int] = None
    """
    For :class:`SharedVectorLandmarkAttention` (``name="shared_vector_landmark"``) only: the length
    of the learned per-block positional vector appended to every value. Defaults to 32. The per-head
    attention output becomes ``head_dim + vec_dim`` and ``w_out`` is widened to match.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the attention implementation will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        bias = self.bias if self.bias is not None else self.name != AttentionType.normalized

        params = 0

        # Block attention Q projection.
        params += d_model * n_heads * head_dim
        if bias:
            params += n_heads * head_dim

        # Block attention KV projections.
        params += 2 * d_model * n_kv_heads * head_dim
        if bias:
            params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.qk_norm is not None:
            if self.use_head_qk_norm:
                params += 2 * self.qk_norm.num_params(head_dim)
            else:
                params += self.qk_norm.num_params(n_heads * head_dim)  # q_norm
                params += self.qk_norm.num_params(n_kv_heads * head_dim)  # k_norm

        # Block attention out.
        params += n_heads * head_dim * d_model
        if bias:
            params += d_model

        # Block attention gate projection.
        if self.gate is not None:
            if self.gate.granularity == GateGranularity.headwise:
                params += d_model * n_heads
                if bias:
                    params += n_heads
            elif self.gate.granularity == GateGranularity.elementwise:
                params += d_model * (n_heads * head_dim)
                if bias:
                    params += n_heads * head_dim

        # Block QK scaling factors.
        if self.name == AttentionType.normalized:
            params += n_heads * head_dim
            params += n_kv_heads * head_dim

        # Shared-vector landmark: the separate w_out_vec branch (n_heads * vec_dim -> d_model) plus
        # the per-head weight_landmark map and base vector.
        if self.name == AttentionType.shared_vector_landmark:
            vec_dim = self.vec_dim if self.vec_dim is not None else 32
            params += n_heads * vec_dim * d_model  # w_out_vec
            if bias:
                params += d_model  # w_out_vec bias
            params += n_heads * head_dim * vec_dim  # weight_landmark
            params += n_heads * vec_dim  # base

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "SequenceMixer":
        """
        Build the corresponding attention module.

        :param d_model: The model dimensionality.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")

        # Resolve the effective attention type for this layer. When a per-layer pattern is
        # configured it overrides ``name``; otherwise ``name`` applies to every layer.
        layer_types_config: Optional[AttentionTypePatternConfig] = kwargs.pop("layer_types", None)
        if layer_types_config is not None:
            effective_name = layer_types_config.get_type(layer_idx, n_layers)
            # The full set of attention types this config may produce across all layers, used for
            # validating the shared landmark parameters below.
            possible_types: Set[AttentionType] = {
                AttentionType(t) for t in layer_types_config.pattern
            }
            if (
                layer_types_config.force_full_attention_on_first_layer
                or layer_types_config.force_full_attention_on_last_layer
            ):
                possible_types.add(AttentionType.default)
        else:
            effective_name = self.name
            possible_types = {self.name}

        is_landmark = effective_name in _LANDMARK_ATTENTION_TYPES

        sliding_window_config: Optional[SlidingWindowAttentionConfig] = kwargs.pop(
            "sliding_window", None
        )
        use_swa = sliding_window_config is not None and sliding_window_config.should_use_swa(
            layer_idx, n_layers
        )
        if use_swa and is_landmark:
            raise OLMoConfigurationError(
                f"layer {layer_idx} resolves to landmark attention ('{effective_name}') but sliding "
                "window attention is also configured for it; these are mutually exclusive"
            )
        if use_swa:
            assert sliding_window_config is not None
            kwargs["window_size"] = sliding_window_config.get_window_size(layer_idx, n_layers)
        else:  # global (non-SWA) layer
            rope_config: Optional[RoPEConfig] = kwargs.get("rope")
            if rope_config is not None and rope_config.no_global_rope:
                kwargs["rope"] = None

        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )

        mem_freq = kwargs.pop("mem_freq", None)
        landmark_use_kernel = kwargs.pop("landmark_use_kernel", None)
        num_landmarks = kwargs.pop("num_landmarks", None)
        nonselected_landmark_mass = kwargs.pop("nonselected_landmark_mass", None)
        vec_dim = kwargs.pop("vec_dim", None)
        if mem_freq is not None and not (possible_types & set(_LANDMARK_ATTENTION_TYPES)):
            raise OLMoConfigurationError(
                "'mem_freq' is only supported with landmark attention variants "
                f"(no landmark layer is configured; got name='{self.name}')"
            )
        if landmark_use_kernel is not None and AttentionType.landmark not in possible_types:
            raise OLMoConfigurationError(
                "'landmark_use_kernel' is only supported with landmark attention "
                f"(got name='{self.name}')"
            )
        if num_landmarks is not None and AttentionType.sparse_landmark not in possible_types:
            raise OLMoConfigurationError(
                "'num_landmarks' is only supported with sparse_landmark attention "
                f"(got name='{self.name}')"
            )
        if (
            nonselected_landmark_mass is not None
            and AttentionType.fast_compressive_landmark not in possible_types
        ):
            raise OLMoConfigurationError(
                "'nonselected_landmark_mass' is only supported with fast_compressive_landmark "
                f"attention (got name='{self.name}')"
            )
        if vec_dim is not None and AttentionType.shared_vector_landmark not in possible_types:
            raise OLMoConfigurationError(
                "'vec_dim' is only supported with shared_vector_landmark attention "
                f"(got name='{self.name}')"
            )

        try:
            if effective_name == "default":
                return Attention(**kwargs)
            elif effective_name == "fused":
                kwargs.pop("use_flash", None)
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with fused attention"
                    )
                return FusedAttention(**kwargs)
            elif effective_name == "normalized":
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with normalized attention"
                    )
                return NormalizedAttention(**kwargs)
            elif effective_name == "landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError("landmark attention requires 'mem_freq' to be set")
                if landmark_use_kernel is not None:
                    kwargs["use_kernel"] = landmark_use_kernel
                return LandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "fast_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "fast_landmark attention requires 'mem_freq' to be set"
                    )
                return FastLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "fast_compressive_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "fast_compressive_landmark attention requires 'mem_freq' to be set"
                    )
                if nonselected_landmark_mass is not None:
                    kwargs["nonselected_landmark_mass"] = nonselected_landmark_mass
                return FastCompressiveLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "sparse_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "sparse_landmark attention requires 'mem_freq' to be set"
                    )
                if num_landmarks is not None:
                    kwargs["num_landmarks"] = num_landmarks
                return SparseLandmarkAttention(mem_freq=mem_freq, **kwargs)
            elif effective_name == "shared_vector_landmark":
                if mem_freq is None:
                    raise OLMoConfigurationError(
                        "shared_vector_landmark attention requires 'mem_freq' to be set"
                    )
                if vec_dim is not None:
                    kwargs["vec_dim"] = vec_dim
                return SharedVectorLandmarkAttention(mem_freq=mem_freq, **kwargs)
            else:
                raise NotImplementedError(effective_name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{effective_name}' {self.__class__.__name__}, {e}"
            ) from e


class Attention(SequenceMixer):
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
    :param gate: Configuration for attention gating. If None, no gating is applied.
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
        head_dim: Optional[int] = None,
        bias: bool = True,
        gate: Optional[GateConfig] = None,
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
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_model = d_model
        # Some models (e.g. Qwen3) use explicit head_dim that differs from d_model // n_heads.
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(
            d_model, n_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_k = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_v = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_out = nn.Linear(
            n_heads * self.head_dim, d_model, bias=bias, dtype=dtype, device=init_device
        )

        self.gate = gate
        self.w_g: Optional[nn.Linear] = None
        if gate is not None:
            if gate.granularity == GateGranularity.headwise:
                self.w_g = nn.Linear(
                    d_model, self.n_heads, bias=bias, dtype=dtype, device=init_device
                )
            elif gate.granularity == GateGranularity.elementwise:
                self.w_g = nn.Linear(
                    d_model,
                    self.n_heads * self.head_dim,
                    bias=bias,
                    dtype=dtype,
                    device=init_device,
                )

        self.clip_qkv = clip_qkv
        self.use_head_qk_norm = use_head_qk_norm

        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if qk_norm is not None:
            if use_head_qk_norm:
                self.q_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
                self.k_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
            else:
                self.q_norm = qk_norm.build(size=n_heads * self.head_dim, init_device=init_device)
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
        self.window_size = window_size
        window_size_tuple: Tuple[int, int] = (-1, -1)
        if window_size is not None:
            if window_size <= 0:
                raise OLMoConfigurationError(f"'window_size' must be positive (got {window_size})")

            if backend is None and flash_attn_api.has_flash_attn_2():
                # note: flash_3, flash_4, and te backends are faster than flash_2 and also support SWA
                backend = AttentionBackendName.flash_2

            # Window size is [i - window_size[0], i + window_size[1]] inclusive
            window_size_tuple = (window_size - 1, 0)

        if backend is None:
            backend = AttentionBackendName.torch

        if not torch.cuda.is_available() and backend != AttentionBackendName.torch:
            warnings.warn(
                f"Backend is set to {backend}, but GPUs are not available. Defaulting to torch."
            )
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

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: Optional[int],
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cu_doc_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.rope is not None
        rope_kwargs = {}
        if cu_doc_lens is not None:
            if not isinstance(self.rope, RotaryEmbedding):
                raise NotImplementedError(
                    "Intra-document RoPE (cu_doc_lens) is only supported by RotaryEmbedding; "
                    f"got {type(self.rope).__name__}"
                )
            rope_kwargs["cu_doc_lens"] = cu_doc_lens
        return self.rope(
            q,
            k,
            head_first=False,
            start_pos=start_pos,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
            **rope_kwargs,
        )

    def _prepare_qkv(
        self,
        x: torch.Tensor,
        *,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the query, key, and value tensors from the input, applying QKV clipping,
        QK-norm, and RoPE. This is the shared pre-attention logic used by :meth:`forward`
        (and subclasses such as :class:`LandmarkAttention`).

        :returns: ``(q, k, v)`` with shapes ``(batch_size, seq_len, n_heads (local), head_dim)``,
            ``(batch_size, seq_len, n_kv_heads (local), head_dim)``, and
            ``(batch_size, seq_len, n_kv_heads (local), head_dim)`` respectively.
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
            q, k = self._apply_rope(q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens)

        return q, k, v

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

        # shape: (batch_size, seq_len, n_heads (local), head_dim),
        #        (batch_size, seq_len, n_kv_heads (local), head_dim),
        #        (batch_size, seq_len, n_kv_heads (local), head_dim)
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=cu_doc_lens
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

        # shape: (batch_size, seq_len, n_heads * head_dim)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, n_heads * head_dim)
        att = self._apply_gate(att, x)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def _apply_gate(self, att: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the optional output gate (``att * sigmoid(w_g(x))``) to the attention output, just
        before the output projection. This is the shared gating logic used by :meth:`forward` and by
        the landmark attention variants, all of which compute their attention output and then call
        :attr:`w_out`. When no gate is configured this is a no-op (returns ``att`` unchanged), so
        non-gated models are byte-for-byte unaffected.

        :param att: The attention output of shape ``(batch_size, seq_len, n_heads * head_dim)``
            (n_heads being the *local* head count under tensor parallelism).
        :param x: The block input of shape ``(batch_size, seq_len, d_model)``, used to compute the
            gate.

        :returns: The gated attention output, same shape as ``att``.
        """
        if self.gate is None:
            return att
        assert self.w_g is not None
        B, T, _ = x.shape
        g = self.w_g(x)
        if self.gate.full_precision:
            g = g.float()
        gate_values = torch.sigmoid(g).to(att.dtype)
        if self.gate.granularity == GateGranularity.headwise:
            # head-wise gating: one value per (local) head, broadcast across head_dim. Use -1 for the
            # head dim so this stays correct when heads are sharded under tensor parallelism.
            att = (att.view(B, T, -1, self.head_dim) * gate_values.unsqueeze(-1)).view(B, T, -1)
        else:  # elementwise: one value per output element.
            att = att * gate_values
        return att

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

        if self.w_g is not None:
            plan["w_g"] = colwise_parallel()

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
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        Prepare the module for context-parallelism (ring attention).

        .. important::
            This requires a backend that supports CP, such as "flash_2" or "te".

        :param cp_mesh: The context parallel device sub-mesh.
        :param ring: The ring context parallel style.
        :param uly: The ulysses context parallel style.
        """
        self.backend.apply_cp(cp_mesh, ring=ring, uly=uly)

    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        # Compute std for Q/K/V initialization
        if init_method == InitMethod.fan_in:
            # For fan_in, use 1/√d_in based on actual weight shape (ignores base std parameter)
            # Each projection may have different output dims (n_heads * head_dim vs n_kv_heads * head_dim)
            # but they all have the same input dim
            for w in (self.w_q, self.w_k, self.w_v):
                w_std = w.in_features**-0.5
                init_linear(w, std=w_std, generator=generator)
        else:
            if init_method == InitMethod.normalized:
                std = d_model**-0.5
            for w in (self.w_q, self.w_k, self.w_v):
                init_linear(w, std=std, generator=generator)

        # Initialize attention gate projection if present
        if self.w_g is not None:
            if init_method == InitMethod.fan_in:
                g_std = self.w_g.in_features**-0.5
            else:
                g_std = std
            init_linear(self.w_g, std=g_std, generator=generator)

        # Compute std for w_out initialization
        if init_method == InitMethod.fan_in:
            std = self.w_out.in_features**-0.5
        elif init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

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

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        This accounts for:
        - Linear projections (Q, K, V, output, and gating if enabled)
        - Attention computation (QK^T and softmax(QK^T) @ V)
        - Sliding window attention (reduced effective sequence length)
        """
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        param_flops = 6 * sum(p.numel() for p in self.parameters())

        # Attention computation (QK^T and Attn*V)
        # 12x multiplier: 2 matmuls * 2 ops each * 3 for forward+backward
        # For sliding window attention, effective sequence length is limited by window size
        # Note that flash attention technically uses more flops (14x multiplier) due to recomputation,
        # however, we just compute the idealized flops for SDPA.
        effective_seq_len = min(self.window_size, seq_len) if self.window_size else seq_len
        attn_flops = 12 * self.n_heads * self.head_dim * effective_seq_len

        return param_flops + attn_flops


class LandmarkAttention(Attention):
    """
    Landmark attention, a drop-in :class:`Attention` variant.

    Landmark attention inserts a special "landmark" token after every ``mem_freq`` regular tokens,
    dividing the sequence into blocks of ``block_size = mem_freq + 1`` tokens, and computes a grouped
    (two-level) softmax so that a query attends to a block's tokens gated by the attention weight
    assigned to that block's landmark. The projections, QK-norm, RoPE, optional output gating, weight
    init, and tensor parallel plan are all inherited from :class:`Attention`. When a
    :class:`GateConfig` is supplied the gate is applied to the landmark attention output
    (``att * sigmoid(w_g(x))``) exactly as in the base attention, which lets landmark attention drop
    into gated models like Qwen3.5.

    Two forward implementations are available, selected by ``use_kernel``:

    - **Eager** (default): a dense :func:`~olmo_core.nn.attention.landmark.landmark_grouped_softmax`
      that is fully autograd-differentiable, so it provides a working forward *and backward* on both
      CPU and GPU. This is ``O(T^2)`` in memory.
    - **Fused kernel** (``use_kernel=True``): the Triton kernel
      (:func:`~olmo_core.nn.attention.landmark_kernel.fused_landmark_attention`), which is flash-style
      (``O(T)`` memory) and requires landmark tokens at fixed periodic positions
      (``pos % block_size == block_size - 1``) and ``mem_freq >= 15``. Its forward and backward are
      validated against the eager path (gradients match exactly in fp32). This is the memory-efficient
      path for long-context training; it is CUDA + triton only.

    .. note::
        Generation / KV-caching, context parallelism, intra-document masking, and long-context
        landmark retrieval are not yet supported.

    :param mem_freq: The number of regular tokens between landmark tokens. The landmark block size
        is ``mem_freq + 1``.
    :param use_kernel: Use the fused Triton kernel instead of the eager path. Defaults to ``False``.

    See :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        use_kernel: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "LandmarkAttention does not support sliding window attention"
            )
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        if mem_freq is None or mem_freq < 1:
            raise OLMoConfigurationError(
                f"LandmarkAttention requires mem_freq >= 1 (got {mem_freq})"
            )
        self.mem_freq = mem_freq
        self.block_size = mem_freq + 1
        self.use_kernel = use_kernel
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        # Ulysses context-parallel state, populated by ``apply_cp``. Unlike the base ``Attention``,
        # landmark attention has its own forward and does not route through ``self.backend``, so it
        # performs the Ulysses all-to-all itself (see ``forward``).
        self._cp_pg: Optional[torch.distributed.ProcessGroup] = None
        self._cp_world_size: int = 1

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        Prepare landmark attention for Ulysses context parallelism.

        Ring/zigzag CP splits the sequence into per-rank chunks, which breaks the landmark grouped
        softmax (a query must see the landmark tokens of *all* preceding blocks). Ulysses CP instead
        splits the heads: each rank gathers the full sequence (with ``n_heads / cp_degree`` heads)
        via an all-to-all, so the grouped softmax sees the complete sequence. We therefore only
        support Ulysses, and we perform the all-to-all in :meth:`forward` rather than in
        ``self.backend`` (which landmark attention never uses).

        :param cp_mesh: The context parallel device sub-mesh.
        :param ring: Must be ``None``; ring CP is not supported.
        :param uly: The Ulysses context parallel style.

        :raises OLMoConfigurationError: If ring CP is requested, or if the number of (KV) heads is
            not divisible by the CP degree.
        """
        if ring is not None:
            raise OLMoConfigurationError(
                "LandmarkAttention only supports Ulysses context parallelism, not ring/zigzag CP "
                "(which splits the sequence and breaks the landmark grouped softmax). Use "
                "TransformerContextParallelConfig.ulysses(...)."
            )
        if uly is None:
            raise ValueError("One of 'ring' or 'uly' must be specified")

        cp_size = cp_mesh.size()
        if self.n_heads % cp_size != 0:
            raise OLMoConfigurationError(
                f"Ulysses CP degree ({cp_size}) must divide n_heads ({self.n_heads})"
            )
        if self.n_kv_heads % cp_size != 0:
            raise OLMoConfigurationError(
                f"Ulysses CP degree ({cp_size}) must divide n_kv_heads ({self.n_kv_heads}); "
                f"landmark attention does not support replicating KV heads across CP ranks"
            )

        self._cp_pg = cp_mesh.get_group()
        self._cp_world_size = cp_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_pg is not None

    @torch.compiler.disable
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
        Apply landmark attention to the input of shape ``(batch_size, seq_len, d_model)``.
        """
        if any(
            v is not None
            for v in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
            )
        ):
            raise NotImplementedError(
                "Intra-document masking (cu_doc_lens) is not supported with landmark attention"
            )
        if cache_leftpad is not None or self.kv_cache_manager is not None:
            raise NotImplementedError(
                "KV-caching / generation is not yet supported with landmark attention"
            )

        # ``T_local`` is the per-rank sequence length: the full sequence when CP is disabled, or the
        # CP shard (``T / cp_degree``) under Ulysses CP. RoPE is applied below in ``_prepare_qkv``
        # using the (already CP-sharded) positional buffers, i.e. with correct *global* positions,
        # before the Ulysses all-to-all gathers the full sequence.
        B, T_local, _ = x.shape

        # shape: (B, T_local, n_heads, head_dim), (B, T_local, n_kv_heads, head_dim) x2
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
        )

        if self.cp_enabled:
            assert self._cp_pg is not None
            # Ulysses: gather the full sequence and scatter the heads across CP ranks.
            # (B, T/CP, H, D) -> (B, T, H/CP, D); likewise for the KV heads.
            q = all_to_all_single_cp2hp(q, self._cp_pg)
            k, v = all_to_all_cp2hp([k, v], self._cp_pg)

        # Each rank now holds the full sequence ``T`` (= ``T_local`` without CP).
        T = q.shape[1]
        if T % self.block_size != 0:
            raise OLMoConfigurationError(
                f"Sequence length ({T}) must be a multiple of the landmark block size "
                f"(mem_freq + 1 = {self.block_size})."
            )

        # -> (B, n_heads, T, head_dim), expanding KV heads for GQA. Under CP the head counts here
        # are the per-rank (n_heads / cp_degree) counts; their ratio (n_rep) is unchanged.
        n_rep = q.shape[2] // k.shape[2]
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), n_rep)
        v = repeat_kv(v.transpose(1, 2), n_rep)

        if self.use_kernel:
            if not has_landmark_kernel():
                raise RuntimeError(
                    "LandmarkAttention was built with 'use_kernel=True' but the fused Triton kernel "
                    "is unavailable (install 'triton' and run on a CUDA device)."
                )
            if self.block_size < 16:
                # The kernel tiles by ``block_size`` and uses ``tl.dot``, which requires tile
                # dimensions >= 16.
                raise OLMoConfigurationError(
                    f"The fused landmark kernel requires mem_freq >= 15 (block size "
                    f"{self.block_size} < 16); tl.dot needs tile dimensions of at least 16."
                )
            is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
            # shape: (B, n_heads, T, head_dim)
            att = fused_landmark_attention(
                q, k, v, is_mem, sm_scale=self.softmax_scale, block_size=self.block_size
            )
        else:
            # Eager grouped-softmax path: works on CPU/GPU and is fully autograd-differentiable,
            # so it provides a working training backward without the fused kernel.
            # shape: (B, n_heads, T, head_dim)
            att = self._eager_forward(q, k, v)

        # shape: (B, T, n_heads (local), head_dim)
        att = att.transpose(1, 2)

        if self.cp_enabled:
            assert self._cp_pg is not None
            # Ulysses: scatter the sequence back and gather the heads.
            # (B, T, H/CP, D) -> (B, T/CP, H, D)
            att = all_to_all_single_hp2cp(att.contiguous(), self._cp_pg)

        # shape: (B, T_local, n_heads * head_dim)
        att = att.contiguous().view(B, T_local, -1)

        # shape: (B, T_local, n_heads * head_dim)
        att = self._apply_gate(att, x)

        # shape: (B, T_local, d_model)
        return self.w_out(att)

    def _eager_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, n_heads, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = self._landmark_masks(T, q.device, q.dtype)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        attn = attn + attn_mask
        attn = torch.maximum(
            attn, torch.tensor(torch.finfo(attn.dtype).min, device=attn.device, dtype=attn.dtype)
        )

        probs = landmark_grouped_softmax(
            attn,
            dim=-1,
            is_mem=is_mem.expand(B, n_heads, T, T),
            last_section_mask=last_section_mask.expand(B, 1, T, T),
        ).to(q.dtype)

        # shape: (B, n_heads, T, head_dim)
        return torch.matmul(probs, v)

    def _landmark_masks(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the additive causal attention mask, the landmark mask (``is_mem``), and the
        ``last_section_mask`` for the eager grouped-softmax path. All are batch-independent
        (built with a batch dim of 1) and depend only on ``T`` and the landmark block size.
        """
        return build_landmark_masks(T, self.block_size, device, dtype)


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
            q, k = self._apply_rope(q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens)

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


class FusedAttention(SequenceMixer):
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
        if cu_doc_lens is not None and self.rope is not None:
            raise NotImplementedError(
                "Intra-document RoPE (cu_doc_lens) is not yet supported by FusedAttention"
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
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.backend.apply_cp(cp_mesh, ring=ring, uly=uly)

    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        # Compute std for fused QKV initialization
        if init_method == InitMethod.fan_in:
            std = self.w_qkv.in_features**-0.5
        elif init_method == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(self.w_qkv, std=std, generator=generator)

        # Compute std for w_out initialization
        if init_method == InitMethod.fan_in:
            std = self.w_out.in_features**-0.5
        elif init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        param_flops = 6 * sum(p.numel() for p in self.parameters())

        # Attention computation (QK^T and Attn*V)
        # 12x multiplier: 2 matmuls * 2 ops each * 3 for forward+backward
        attn_flops = 12 * self.n_heads * self.head_dim * seq_len

        return param_flops + attn_flops


# New landmark-attention sequence mixers. These live in their own modules and subclass ``Attention``
# (defined above), so they are imported at the end of this package to avoid a circular import; the
# ``AttentionConfig.build`` branches above reference them by name at call time.
from .landmark_compressive import FastCompressiveLandmarkAttention  # noqa: E402
from .landmark_fast import FastLandmarkAttention  # noqa: E402
from .landmark_shared_vector import SharedVectorLandmarkAttention  # noqa: E402
from .landmark_sparse import SparseLandmarkAttention  # noqa: E402
