import gc
import logging
import math
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class KVCacheManager(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        page_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._page_size = page_size

        if page_size is not None:
            # Paged mode: cache is a pool of pages.
            max_pages_per_seq = math.ceil(max_seq_len / page_size)
            num_pages = batch_size * max_pages_per_seq
            self.kv_cache_shape = (num_pages, page_size, num_kv_heads, head_dim)

            self.k_cache: torch.Tensor
            self.register_buffer(
                "k_cache",
                torch.zeros(self.kv_cache_shape, device=device, dtype=dtype),
                persistent=False,
            )
            self.v_cache: torch.Tensor
            self.register_buffer(
                "v_cache",
                torch.zeros(self.kv_cache_shape, device=device, dtype=dtype),
                persistent=False,
            )

            # Sequential page table: batch i owns pages [i*max_pages_per_seq, (i+1)*max_pages_per_seq)
            page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
                batch_size, max_pages_per_seq
            )
            self.page_table: Optional[torch.Tensor]
            self.register_buffer("page_table", page_table, persistent=False)

            # Per-batch sequence lengths
            self.cache_seqlens: torch.Tensor
            self.register_buffer(
                "cache_seqlens",
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                persistent=False,
            )
            self.cache_leftpad: torch.Tensor
            self.register_buffer(
                "cache_leftpad",
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                persistent=False,
            )
        else:
            # Dense mode (original behavior).
            self.kv_cache_shape = (batch_size, max_seq_len, num_kv_heads, head_dim)

            self.k_cache: torch.Tensor
            self.register_buffer(
                "k_cache",
                torch.zeros(self.kv_cache_shape, device=device, dtype=dtype),
                persistent=False,
            )
            self.v_cache: torch.Tensor
            self.register_buffer(
                "v_cache",
                torch.zeros(self.kv_cache_shape, device=device, dtype=dtype),
                persistent=False,
            )
            self.cache_leftpad: torch.Tensor
            self.register_buffer(
                "cache_leftpad",
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                persistent=False,
            )
            self.cache_seqlens: torch.Tensor
            self.register_buffer(
                "cache_seqlens",
                torch.zeros((), dtype=torch.int32, device=device),
                persistent=False,
            )
            self.page_table = None

    @property
    def is_paged(self) -> bool:
        return self._page_size is not None

    def record_leftpad(self, leftpad: Optional[torch.Tensor]):
        if leftpad is not None:
            self.cache_leftpad.copy_(leftpad)

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Scatter-write new k/v tokens into the paged cache pool, then increment
        ``cache_seqlens``. Only valid in paged mode.

        :param k: Key tensor of shape ``(batch_size, seq_len, num_kv_heads, head_dim)``.
        :param v: Value tensor of shape ``(batch_size, seq_len, num_kv_heads, head_dim)``.
        """
        assert self.is_paged, "update() is only valid in paged mode"
        assert self.page_table is not None

        B, T, _H, _D = k.shape
        page_size = self._page_size
        assert page_size is not None

        for b in range(B):
            cur_len = int(self.cache_seqlens[b].item())

            for i in range(T):
                abs_pos = cur_len + i
                page_idx = int(self.page_table[b, abs_pos // page_size].item())
                slot = abs_pos % page_size
                self.k_cache[page_idx, slot] = k[b, i]
                self.v_cache[page_idx, slot] = v[b, i]

            self.cache_seqlens[b] += T

    def update_seqlen(self, seqlen: int):
        # IMPORTANT: The flash-attn kernel interprets cache_seqlens as absolute
        # indices within the KV cache sequence dimension. Not as actual logical
        # sequence lengths that exclude padding.
        if self.is_paged:
            # In paged mode, update() already increments cache_seqlens.
            return
        self.cache_seqlens.add_(seqlen)

    def current_position(self) -> torch.Tensor:
        if self.is_paged:
            # In paged mode, cache_seqlens is per-batch (batch_size,).
            # All batch items have the same seqlen, so return a scalar.
            return self.cache_seqlens[0]
        return self.cache_seqlens

    def zero_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_leftpad.zero_()
        self.cache_seqlens.zero_()

    def reallocate(
        self,
        batch_size: int,
        max_seq_len: int,
        _explicitly_free_memory: bool = True,
    ):
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len

        if _explicitly_free_memory and hasattr(self, "cache"):
            # The cache can be large so we explicitly free it before reallocating
            del self.cache
            gc.collect()
            torch.cuda.empty_cache()

        if self.is_paged:
            page_size = self._page_size
            assert page_size is not None
            max_pages_per_seq = math.ceil(max_seq_len / page_size)
            num_pages = batch_size * max_pages_per_seq
            self.kv_cache_shape = (num_pages, page_size, self.num_kv_heads, self.head_dim)

            k = self.k_cache.new_zeros(self.kv_cache_shape)
            v = self.v_cache.new_zeros(self.kv_cache_shape)
            page_table = torch.arange(num_pages, dtype=torch.int32, device=k.device).view(
                batch_size, max_pages_per_seq
            )
            seqlens = self.cache_seqlens.new_zeros((batch_size,))
            leftpad = self.cache_leftpad.new_zeros((batch_size,), dtype=torch.int32)

            self.register_buffer("k_cache", k, persistent=False)
            self.register_buffer("v_cache", v, persistent=False)
            self.register_buffer("page_table", page_table, persistent=False)
            self.register_buffer("cache_seqlens", seqlens, persistent=False)
            self.register_buffer("cache_leftpad", leftpad, persistent=False)
        else:
            self.kv_cache_shape = (batch_size, max_seq_len, self.num_kv_heads, self.head_dim)

            k = self.k_cache.new_zeros(self.kv_cache_shape)
            v = self.v_cache.new_zeros(self.kv_cache_shape)
            leftpad = self.cache_leftpad.new_zeros((batch_size,), dtype=torch.int32)
            seqlens = self.cache_seqlens.new_zeros(())

            # Re-register to preserve persistent=False
            self.register_buffer("k_cache", k, persistent=False)
            self.register_buffer("v_cache", v, persistent=False)
            self.register_buffer("cache_leftpad", leftpad, persistent=False)
            self.register_buffer("cache_seqlens", seqlens, persistent=False)

    def is_reusable(self, batch_size: int, max_seq_len: int) -> bool:
        return self._batch_size == batch_size and self._max_seq_len >= max_seq_len

    def reset(self, batch_size: int, max_seq_len: int):
        """
        Reset the KV cache for new generation parameters.

        If the cache is reusable with the given parameters, it will be zeroed out.
        Otherwise, it will be reallocated with the new dimensions and dtype.

        :param batch_size: The batch size for the cache.
        :param max_seq_len: The maximum sequence length for the cache.
        """
        if self.is_reusable(batch_size, max_seq_len):
            self.zero_cache()
        else:
            log.debug("Unreusable KV cache, reallocating")
            self.reallocate(batch_size, max_seq_len)
