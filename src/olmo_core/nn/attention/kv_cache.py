import gc
import logging
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
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_cache_shape = (batch_size, max_seq_len, self.num_kv_heads, self.head_dim)

        self.register_buffer(
            "k_cache",
            torch.zeros(self.kv_cache_shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(self.kv_cache_shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "cache_leftpad",
            torch.zeros(batch_size, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "cache_seqlens", torch.zeros((), dtype=torch.int32, device=device), persistent=False
        )

    def record_leftpad(self, leftpad: Optional[torch.Tensor]):
        if leftpad is not None:
            self.cache_leftpad.copy_(leftpad)

    def update_seqlen(self, seqlen: int):
        # IMPORTANT: The flash-attn kernel interprets cache_seqlens as absolute
        # indices within the KV cache sequence dimension. Not as actual logical
        # sequence lengths that exclude padding.
        self.cache_seqlens.add_(seqlen)

    def current_position(self) -> torch.Tensor:
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
        self.kv_cache_shape = (batch_size, max_seq_len, self.num_kv_heads, self.head_dim)
        if _explicitly_free_memory and hasattr(self, "cache"):
            # The cache can be large so we explicitly free it before reallocating
            del self.cache
            gc.collect()
            torch.cuda.empty_cache()

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
        return self.kv_cache_shape[0] == batch_size and self.kv_cache_shape[1] >= max_seq_len

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
