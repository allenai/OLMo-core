import gc
import logging
from typing import Optional

import torch

from olmo_core.nn.attention import InferencePhase

log = logging.getLogger(__name__)


class KVCacheManager:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.reallocate(batch_size, max_seq_len, dtype)

    def record_leftpad(self, leftpad: Optional[torch.Tensor]):
        if leftpad is not None:
            self.cache["cache_leftpad"].data.copy_(leftpad)

    def update_seqlen(self, seqlen: int):
        # IMPORTANT: The flash-attn kernel interprets cache_seqlens as absolute
        # indices within the KV cache sequence dimension. Not as actual logical
        # sequence lengths that exclude padding.
        self.cache["cache_seqlens"] += seqlen

    @property
    def phase(self):
        if self.cache["cache_seqlens"] == 0:
            return InferencePhase.prefill
        return InferencePhase.decode

    def current_position(self) -> int:
        if self.phase == InferencePhase.decode:
            return self.cache["cache_seqlens"]
        return 0

    def zero_cache(self):
        self.cache["k_cache"].data.zero_()
        self.cache["v_cache"].data.zero_()
        self.cache["cache_seqlens"].data.zero_()
        self.cache["cache_leftpad"].data.zero_()

    def reallocate(self, batch_size: int, max_seq_len: int, dtype: torch.dtype):
        self.kv_cache_shape = (batch_size, max_seq_len, self.num_kv_heads, self.head_dim)
        self.dtype = dtype

        if hasattr(self, "cache"):
            # The cache can be large so we explicitly free it before reallocating
            del self.cache
            gc.collect()
            torch.cuda.empty_cache()

        self.cache = torch.nn.ParameterDict(
            {
                "k_cache": torch.nn.Parameter(
                    torch.zeros(self.kv_cache_shape, device=self.device, dtype=dtype)
                ),
                "v_cache": torch.nn.Parameter(
                    torch.zeros(self.kv_cache_shape, device=self.device, dtype=dtype)
                ),
                "cache_leftpad": torch.nn.Parameter(
                    torch.zeros(batch_size, dtype=torch.int32, device=self.device)
                ),
                "cache_seqlens": 0,
            }
        )

    def is_reusable(self, batch_size: int, max_seq_len: int, dtype: torch.dtype) -> bool:
        return (
            self.kv_cache_shape[0] == batch_size
            and self.kv_cache_shape[1] >= max_seq_len
            and self.dtype == dtype
        )

    def reset(self, batch_size: int, max_seq_len: int, dtype: torch.dtype):
        if self.is_reusable(batch_size, max_seq_len, dtype):
            self.zero_cache()
        else:
            log.debug("Unreusable KV cache, reallocating")
            self.reallocate(batch_size, max_seq_len, dtype)
