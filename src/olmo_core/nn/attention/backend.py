from abc import abstractmethod
from typing import Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh

from olmo_core.config import StrEnum
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.buffer_cache import BufferCache

from .flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_3,
    dispatch_flash_attn_3_qkvpacked,
    dispatch_flash_attn_3_with_kvcache,
    dispatch_flash_attn_qkvpacked,
    dispatch_flash_attn_with_kvcache,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
    has_flash_attn_2,
    has_flash_attn_3,
    has_ring_flash_attn,
)
from .ring import RingAttentionLoadBalancerType
from .te_attn_api import TEDotProductAttention, has_te_attn


class AttentionBackendName(StrEnum):
    """
    An enumeration of the different attention backends.
    """

    torch = "torch"
    """
    PyTorch's built-in SDPA ➡️ :class:`TorchAttentionBackend`
    """
    flash = "flash"
    flash_2 = "flash"
    """
    Flash attention 2 from the `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ library.
    To use this with context-parallelism, `ring-flash-attn <https://github.com/zhuzilin/ring-flash-attention>`_
    is also required. ➡️ :class:`FlashAttentionBackend`
    """
    flash_3 = "flash_3"
    """
    Flash attention 3 (beta) from the `flash-attn <https://github.com/Dao-AILab/flash-attention>`_
    library `hopper/` subdirectory. Only supports H100/H800 GPUs. ➡️ :class:`FlashAttention3Backend`
    """
    te = "te"
    """
    Transformer Engine attention ➡️ :class:`TEAttentionBackend`.
    """

    def get_class(self) -> Type["AttentionBackend"]:
        if self == self.torch:
            return TorchAttentionBackend
        elif self in (self.flash, self.flash_2):
            return FlashAttentionBackend
        elif self == self.flash_3:
            return FlashAttention3Backend
        elif self == self.te:
            return TEAttentionBackend
        else:
            raise NotImplementedError(self)

    def build(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ) -> "AttentionBackend":
        return self.get_class()(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=(cache.with_namespace(f"attn_backend={self}") if cache else None),
        )

    def assert_supported(self):
        self.get_class().assert_supported()

    def assert_supports_swa(self):
        self.get_class().assert_supports_swa()

    def assert_supports_cp(self):
        self.get_class().assert_supports_cp()

    def assert_supports_packed_qkv(self):
        self.get_class().assert_supports_packed_qkv()

    def assert_supports_kv_cache(self):
        self.get_class().assert_supports_kv_cache()


class AttentionBackend(nn.Module):
    """
    Encapsulates a backend for the scaled dot-product attention (SDPA) operation.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        self.assert_supported()
        if window_size != (-1, -1):
            self.assert_supports_swa()
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.scale = scale
        self.dropout_p = dropout_p
        self.window_size = window_size
        self.cache = cache
        self.cp_pg: Optional[dist.ProcessGroup] = None
        self.cp_enabled = False
        self.cp_load_balancer: Optional[RingAttentionLoadBalancerType] = None
        self.head_stride: int = 1

    @classmethod
    @abstractmethod
    def assert_supported(cls):
        """
        Validates that this backend is supported on the current system.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_swa(cls):
        """
        Validates that this backend supports sliding window attention (SWA).
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_cp(cls):
        """
        Validates that this backend supports context parallelism.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_packed_qkv(cls):
        """
        Validates that this backend supports taking QKV as a single packed tensor.
        Raises an error if not supported.
        """
        pass

    @classmethod
    @abstractmethod
    def assert_supports_kv_cache(cls):
        """
        Validates that this backend supports KV caching.
        Raises an error if not supported.
        """
        pass

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        """
        Warmup the buffer cache.
        """
        del max_seq_len, device

    @abstractmethod
    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        """
        Run the attention operation.
        """
        raise NotImplementedError

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        """
        Apply context parallelism if supported by the backend.
        """
        self.assert_supports_cp()
        self.cp_pg = cp_mesh.get_group()
        self.cp_load_balancer = load_balancer
        self.cp_enabled = True
        self.cp_head_stride = head_stride


class TorchAttentionBackend(AttentionBackend):
    """
    PyTorch's built-in scaled dot-product attention (SDPA) backend.
    """

    @classmethod
    def assert_supported(cls):
        pass

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support context parallelism")

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_sliding_window_mask(
            seq_len_q=max_seq_len,
            seq_len_kv=max_seq_len,
            device=device,
            window_size=self.window_size,
        )

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        del local_k_slice

        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")

        q, k, v = qkv

        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")

        attn_mask: Optional[torch.Tensor] = None
        if self.window_size != (-1, -1):
            attn_mask = self._get_sliding_window_mask(
                seq_len_q=q.shape[1],
                seq_len_kv=k.shape[1],
                device=q.device,
                window_size=self.window_size,
            )

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
                f"'{self.__class__.__name__}' doesn't support intra-document masking"
            )

        # NOTE: PyTorch's SDPA doesn't support GQA, so we have to do this.
        n_rep = self.n_heads // self.n_kv_heads
        # shape: (batch_size, seq_len, n_heads, head_dim)
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

        # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
        # shape: (batch_size, n_heads, seq_len, head_dim),
        #        (batch_size, n_kv_heads, seq_len, head_dim),
        #        (batch_size, n_kv_heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # shape: (batch_size, n_heads, seq_len, head_dim)
        att = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            is_causal=attn_mask is None,
            scale=self.scale,
        )

        # shape: (batch_size, seq_len, n_heads, head_dim)
        return att.transpose(1, 2).contiguous()

    def _get_sliding_window_mask(
        self,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        key = f"seq_len_q={seq_len_q},seq_len_kv={seq_len_kv},window_size={window_size}"
        if self.cache is not None:
            if (mask := self.cache.get_for_device(key, device)) is not None:
                return mask

            attn_mask = self._build_sliding_window_mask(
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                device=device,
                window_size=window_size,
            )
            self.cache[key] = attn_mask

            return attn_mask

        return self._build_sliding_window_mask(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            device=device,
            window_size=window_size,
        )

    @classmethod
    def _build_sliding_window_mask(
        cls,
        seq_len_q: int,
        seq_len_kv: int,
        device: torch.device,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_kv, device=device, dtype=torch.bool))

        if window_size != (-1, -1):
            sliding_window_left_mask = torch.ones_like(
                causal_mask, dtype=torch.bool, device=device
            ).triu(diagonal=-window_size[0])
            sliding_window_right_mask = torch.ones_like(
                causal_mask, dtype=torch.bool, device=device
            ).tril(diagonal=window_size[1])
            sliding_window_mask = torch.logical_and(
                sliding_window_left_mask,
                sliding_window_right_mask,
            )

            attn_mask = torch.logical_and(
                causal_mask,
                sliding_window_mask,
            )
        else:
            attn_mask = causal_mask

        return attn_mask


class FlashAttentionBackend(AttentionBackend):
    """
    SDPA from the flash-attn package. Additionally, ring-flash-attn is required for context parallelism.
    """

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_2():
            raise RuntimeError(f"'{cls.__name__}' requires the flash-attn package.")

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_cp(cls):
        if not has_ring_flash_attn():
            raise RuntimeError(
                f"'{cls.__name__}' requires the ring-flash-attn package for context parallelism."
            )

    @classmethod
    def assert_supports_packed_qkv(cls):
        pass

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        if isinstance(qkv, torch.Tensor):
            if kv_cache_manager is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with KV caching"
                )

            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with sliding window attention"
                )

            if self.cp_enabled:
                assert self.cp_pg is not None and self.cp_load_balancer is not None
                return dispatch_ring_flash_attn_qkvpacked(
                    qkv,
                    group=self.cp_pg,
                    strategy=self.cp_load_balancer,
                    cu_seqlens=cu_doc_lens,
                    max_seqlen=max_doc_len,
                    dropout_p=self.dropout_p,
                    softmax_scale=self.scale,
                    causal=True,
                )
            else:
                return dispatch_flash_attn_qkvpacked(
                    qkv,
                    cu_seqlens=cu_doc_lens,
                    max_seqlen=max_doc_len,
                    dropout_p=self.dropout_p,
                    softmax_scale=self.scale,
                    causal=True,
                )

        q, k, v = qkv

        if kv_cache_manager:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support KV caching with context parallelism"
                )

            return dispatch_flash_attn_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
                k_cache=kv_cache_manager.k_cache,  # updated in-place
                v_cache=kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=kv_cache_manager.cache_leftpad,
                cache_seqlens=kv_cache_manager.cache_seqlens.expand(
                    kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )

        if self.cp_enabled:
            assert self.cp_pg is not None and self.cp_load_balancer is not None
            return dispatch_ring_flash_attn(
                q,
                k,
                v,
                group=self.cp_pg,
                strategy=self.cp_load_balancer,
                cu_seqlens=cu_doc_lens,
                cu_seqlens_q=cu_doc_lens_q,
                cu_seqlens_k=cu_doc_lens_k,
                max_seqlen=max_doc_len,
                max_seqlen_q=max_doc_len_q,
                max_seqlen_k=max_doc_len_k,
                heads_k_stride=self.cp_head_stride,
                local_k_slice=local_k_slice,
                dropout_p=self.dropout_p,
                causal=True,
                softmax_scale=self.scale,
                window_size=self.window_size,
            )

        return dispatch_flash_attn(
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
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class FlashAttention3Backend(AttentionBackend):
    """
    SDPA from the flash-attn 3 package. Does not currently support context parallelism.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
    ):
        if dropout_p > 0.0:
            raise RuntimeError("dropout_p > 0.0 is not supported for flash-attn 3")
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
        )

    @classmethod
    def assert_supported(cls):
        if not has_flash_attn_3():
            raise RuntimeError(f"'{cls.__name__}' requires the flash-attn 3 package.")

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_cp(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support context parallelism")

    @classmethod
    def assert_supports_packed_qkv(cls):
        pass

    @classmethod
    def assert_supports_kv_cache(cls):
        pass

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        if isinstance(qkv, torch.Tensor):
            if kv_cache_manager is not None:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with KV caching"
                )

            if self.window_size != (-1, -1):
                raise RuntimeError(
                    f"'{self.__class__.__name__}' doesn't support packed QKV with sliding window attention"
                )

            return dispatch_flash_attn_3_qkvpacked(
                qkv,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                softmax_scale=self.scale,
                causal=True,
            )

        q, k, v = qkv

        if kv_cache_manager:
            return dispatch_flash_attn_3_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.window_size,
                k_cache=kv_cache_manager.k_cache,  # updated in-place
                v_cache=kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=kv_cache_manager.cache_leftpad,
                cache_seqlens=kv_cache_manager.cache_seqlens.expand(
                    kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )

        return dispatch_flash_attn_3(
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )


class TEAttentionBackend(AttentionBackend):
    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )
        if not has_te_attn():
            raise RuntimeError("TransformerEngine attention is not available")
        assert TEDotProductAttention is not None
        self.te_attn = TEDotProductAttention(
            self.n_heads,
            self.head_dim,
            num_gqa_groups=self.n_kv_heads,
            attention_dropout=self.dropout_p,
            attn_mask_type="causal",
            window_size=(self.window_size[0], 0),  # be explicit about causal mask
            qkv_format="bshd",
            softmax_scale=self.scale,
        )

    @classmethod
    def assert_supported(cls):
        if not has_te_attn():
            raise RuntimeError(f"'{cls.__name__}' requires NVIDIA's TransformerEngine package.")

    @classmethod
    def assert_supports_swa(cls):
        pass

    @classmethod
    def assert_supports_cp(cls):
        pass

    @classmethod
    def assert_supports_packed_qkv(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support packed QKV")

    @classmethod
    def assert_supports_kv_cache(cls):
        raise RuntimeError(f"'{cls.__name__}' doesn't support KV caching")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        if load_balancer != RingAttentionLoadBalancerType.zig_zag:
            raise RuntimeError(f"'{self.__class__.__name__}' only supports zig-zag load balancing")
        super().apply_cp(cp_mesh, load_balancer, head_stride=head_stride)
        self.te_attn.set_context_parallel_group(
            cp_group=cp_mesh.get_group(),
            cp_global_ranks=dist.get_process_group_ranks(cp_mesh.get_group()),
            cp_stream=torch.cuda.default_stream(),
            #  cp_stream=get_or_init_stream("cp"),  # this doesn't seem to help
            cp_comm_type="p2p",
        )

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        del local_k_slice

        if kv_cache_manager is not None:
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support KV caching")

        if isinstance(qkv, torch.Tensor):
            raise RuntimeError(f"'{self.__class__.__name__}' doesn't support packed QKV")

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
                f"'{self.__class__.__name__}' doesn't currently support intra-document masking"
            )

        q, k, v = qkv
        return self.te_attn(
            q,
            k,
            v,
            cu_seqlens_q=cu_doc_lens if cu_doc_lens is not None else cu_doc_lens_q,
            cu_seqlens_kv=cu_doc_lens if cu_doc_lens is not None else cu_doc_lens_k,
            max_seqlen_q=max_doc_len if max_doc_len is not None else max_doc_len_q,
            max_seqlen_kv=max_doc_len if max_doc_len is not None else max_doc_len_k,
        )


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
