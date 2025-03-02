import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import ensure_multiple_of

from .buffer_cache import BufferCache
from .functional import (
    dispatch_flash_attn,
    dispatch_flash_attn_qkvpacked,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
    l2_normalize,
)
from .layer_norm import LayerNorm, LayerNormConfig
from .rope import (
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    RoPEConfig,
    RotaryEmbedding,
)
from .utils import get_tp_wrappers

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
    dtype: DType = DType.float32

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
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "AttentionBase":
        """
        Build the corresponding attention module.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
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
                return FusedAttention(**kwargs)
            elif self.name == "normalized":
                return NormalizedAttention(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class RingAttentionLoadBalancerType(StrEnum):
    """
    An enumeration of the different :class:`RingAttentionLoadBalancer` implementations.
    """

    zig_zag = "zig_zag"
    """
    ➡️ :class:`RingAttentionZigZagLoadBalancer`
    """

    llama3 = "llama3"
    """
    ➡️ :class:`RingAttentionLlama3LoadBalancer`
    """

    def build(self, cp_mesh: DeviceMesh) -> "RingAttentionLoadBalancer":
        """
        Build the load balancer.
        """
        pg = cp_mesh.get_group()
        cp_rank = get_rank(pg)
        cp_world_size = get_world_size(pg)
        if self == self.zig_zag:
            return RingAttentionZigZagLoadBalancer(cp_rank=cp_rank, cp_world_size=cp_world_size)
        elif self == self.llama3:
            return RingAttentionLlama3LoadBalancer(cp_rank=cp_rank, cp_world_size=cp_world_size)
        else:
            raise NotImplementedError(self)


class RingAttentionLoadBalancer(metaclass=ABCMeta):
    """
    A class that handles the logic of sharding inputs on the sequence dimension
    for ring attention (context parallelism).
    """

    def __init__(self, *, cp_rank: int, cp_world_size: int):
        self.cp_rank = cp_rank
        self.cp_world_size = cp_world_size

    @abstractmethod
    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Shard inputs on their sequence dimension, optionally adding padding if needed.

        .. important::
            If using intra-document masking, use :meth:`batch_shard_by_document` instead.

        :returns: The local shards of the inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Same as :meth:`batch_shard` but for strategies that support intra-document masking.

        :returns: The local shards of the inputs and any other additional inputs required for the
            corresponding ring attention implementation.
        """
        raise NotImplementedError


class RingAttentionZigZagLoadBalancer(RingAttentionLoadBalancer):
    """
    Implements the zig-zag load-balancing strategy.
    """

    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        assert len(inputs) == len(seq_dims)
        assert len(set(x.shape[seq_dim] for x, seq_dim in zip(inputs, seq_dims))) == 1
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if length_multiple is None:
            length_multiple = 2 * self.cp_world_size
        elif length_multiple % (2 * self.cp_world_size) != 0:
            raise RuntimeError(
                f"length multiple ({length_multiple}) must be divisible by "
                f"2 x CP degree ({2 * self.cp_world_size})"
            )

        out = []
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            if x.shape[seq_dim] % length_multiple != 0:
                if pad_value is None:
                    raise RuntimeError(
                        f"sequence dimension size ({x.shape[seq_dim]}) must be divisible by "
                        f"{length_multiple}, otherwise provide a padding value"
                    )
                else:
                    x, _ = self.pad(x, seq_dim, pad_value, length_multiple=length_multiple)

            x_chunks = x.chunk(2 * self.cp_world_size, dim=seq_dim)
            local_value = torch.cat(
                [x_chunks[self.cp_rank], x_chunks[2 * self.cp_world_size - self.cp_rank - 1]],
                dim=seq_dim,
            )
            out.append(local_value.contiguous())

        return out

    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        assert len(inputs) == len(seq_dims)
        assert len(set(x.shape[seq_dim] for x, seq_dim in zip(inputs, seq_dims))) == 1
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if cu_doc_lens.device.type != "cpu":
            raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
        if cu_doc_lens.ndim != 1:
            raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
        if cu_doc_lens[0] != 0:
            raise RuntimeError("expected 'cu_doc_lens' to start with a 0")

        out = []
        padding_added = [0 for _ in range(len(cu_doc_lens) - 1)]
        final_padding: Optional[int] = None if length_multiple is None else 0
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            local_values = []
            for i in range(len(cu_doc_lens) - 1):
                start, end = cu_doc_lens[i], cu_doc_lens[i + 1]
                # NOTE: Since 'torch.slice' is not available from the Python API we just call
                # the JIT op directly.
                x_doc_slice = torch.ops.aten.slice(x, dim=seq_dim, start=start, end=end)  # type: ignore
                if x_doc_slice.shape[seq_dim] % (2 * self.cp_world_size) != 0:
                    if pad_value is None:
                        raise RuntimeError(
                            f"document length ({x_doc_slice.shape[seq_dim]}) must be divisible by "
                            f"2 x CP degree ({2 * self.cp_world_size}), otherwise provide a padding value"
                        )
                    else:
                        x_doc_slice, padding = self.pad(x_doc_slice, seq_dim, pad_value)
                        padding_added[i] = padding

                x_chunks = x_doc_slice.chunk(2 * self.cp_world_size, dim=seq_dim)
                local_values.extend(
                    [
                        x_chunks[self.cp_rank],
                        x_chunks[2 * self.cp_world_size - 1 - self.cp_rank],
                    ]
                )
            local_value = torch.cat(local_values, dim=seq_dim).contiguous()
            if length_multiple is not None and local_value.shape[seq_dim] % length_multiple != 0:
                if pad_value is None:
                    raise RuntimeError(
                        "You must provide a 'pad_value' when 'length_multiple' is specified!"
                    )
                else:
                    local_value, final_padding = self.pad(
                        local_value, seq_dim, pad_value, length_multiple=length_multiple
                    )
            out.append(local_value)

        if pad_values is not None:
            cumulative_padding = torch.cat(
                [
                    torch.tensor([0], dtype=cu_doc_lens.dtype, device=cu_doc_lens.device),
                    torch.tensor(padding_added, device=cu_doc_lens.device).cumsum(
                        0, dtype=cu_doc_lens.dtype
                    ),
                ]
            )
            cu_doc_lens = cu_doc_lens + cumulative_padding

        local_cu_doc_lens = cu_doc_lens // self.cp_world_size
        if final_padding is not None:
            local_cu_doc_lens = torch.cat(
                [local_cu_doc_lens, (local_cu_doc_lens[-1] + final_padding).unsqueeze(0)]
            )

        local_max_doc_len = (local_cu_doc_lens[1:] - local_cu_doc_lens[:-1]).max().item()

        return out, dict(cu_doc_lens=local_cu_doc_lens, max_doc_len=local_max_doc_len)

    def pad(
        self,
        x: torch.Tensor,
        seq_dim: int,
        value: Union[int, float],
        length_multiple: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        if length_multiple is None:
            length_multiple = 2 * self.cp_world_size
        pad_to = ensure_multiple_of(x.shape[seq_dim], length_multiple)
        padding_to_add = pad_to - x.shape[seq_dim]
        padding = (0, 0) * (x.ndim - seq_dim - 1) + (0, padding_to_add)
        return F.pad(x, padding, value=value), padding_to_add


class RingAttentionLlama3LoadBalancer(RingAttentionLoadBalancer):
    """
    Implements Llama3's load-balancing strategy.
    """

    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        del inputs, seq_dims, pad_values, length_multiple
        raise NotImplementedError(
            f"{self.__class__.__name__} should only be used with intra-document masking. "
            "Please use the 'batch_shard_by_document()' instead."
        )

    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        try:
            from ring_flash_attn import llama3_flash_attn_prepare_cu_seqlens
        except ImportError as e:
            raise RuntimeError(f"ring-flash-attn is required for {self.__class__.__name__}") from e

        assert len(inputs) == len(seq_dims)
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if cu_doc_lens.device.type != "cpu":
            raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
        if cu_doc_lens.ndim != 1:
            raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
        if cu_doc_lens[0] != 0:
            raise RuntimeError("expected 'cu_doc_lens' to start with a 0")

        if length_multiple is None:
            length_multiple = self.cp_world_size
        else:
            length_multiple = length_multiple * self.cp_world_size

        total_length = int(cu_doc_lens[-1])
        padding_to_add = total_length - ensure_multiple_of(total_length, length_multiple)
        local_length = (total_length + padding_to_add) // self.cp_world_size

        if padding_to_add > 0:
            if pad_values is None:
                raise RuntimeError("'pad_values' is required since padding is needed")

            cu_doc_lens = torch.cat(
                [
                    cu_doc_lens,
                    torch.tensor(
                        [total_length + padding_to_add],
                        dtype=cu_doc_lens.dtype,
                        device=cu_doc_lens.device,
                    ),
                ]
            )

        out = []
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            if x.shape[seq_dim] != total_length:
                raise RuntimeError(
                    f"expected input to be have size {total_length} on the sequence dimension "
                    f"but got {x.shape[seq_dim]}"
                )

            if padding_to_add > 0:
                assert pad_value is not None
                x = self.pad(x, seq_dim, padding_to_add, pad_value)

            # NOTE: Since 'torch.slice' is not available from the Python API we just call
            # the JIT op directly.
            local_value = torch.ops.aten.slice(  # type: ignore
                x,
                dim=seq_dim,
                start=self.cp_rank * local_length,
                end=(self.cp_rank + 1) * local_length,
            ).contiguous()
            out.append(local_value)

        (
            cu_doc_lens_q,
            cu_doc_lens_k,
            max_doc_len_q,
            max_doc_len_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_doc_lens,
            causal=True,
            rank=self.cp_rank,
            world_size=self.cp_world_size,
        )

        return out, dict(
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
        )

    def pad(
        self,
        x: torch.Tensor,
        seq_dim: int,
        padding_to_add: int,
        value: Union[int, float],
    ) -> Tuple[torch.Tensor, int]:
        padding = (0, 0) * (x.ndim - seq_dim - 1) + (0, padding_to_add)
        return F.pad(x, padding, value=value), padding_to_add


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
    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
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
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
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

        self.use_flash = use_flash
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
    ) -> torch.Tensor:
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
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(-1))
        if self.k_norm is not None:
            plan["k_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(-1))
        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        """
        Prepare the module for context-parallelism (ring attention).

        .. important::
            This requires flash-attn and ring-flash-attn.
        """
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True


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
            self.sq_init_scaling
            * torch.ones(self.head_dim * self.n_heads, dtype=dtype, device=init_device)
        )

        self.sk_init_value = 1.0
        self.sk_init_scaling = 1.0 / math.sqrt(d_model)
        self.sk = nn.Parameter(
            self.sk_init_scaling
            * torch.ones(self.head_dim * self.n_kv_heads, dtype=dtype, device=init_device)
        )

        self.sqrt_head_dim = math.sqrt(self.head_dim)

    def reset_parameters(self):
        nn.init.ones_(self.sq)
        self.sq.mul_(self.sq_init_scaling)
        nn.init.ones_(self.sk)
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

    def apply_cp(self, cp_mesh: DeviceMesh, load_balancer: RingAttentionLoadBalancerType):
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True


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
