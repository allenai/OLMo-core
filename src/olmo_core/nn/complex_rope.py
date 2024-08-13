from typing import Optional, Tuple

import torch
import torch.nn as nn

from .buffer_cache import BufferCache


class ComplexRotaryEmbedding(nn.Module):
    """
    An implementation of RoPE as a rotation in complex space.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        theta: int = 500_000,
        full_precision: bool = True,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.dim = d_model // n_heads
        self.theta = theta
        self.full_precision = full_precision
        self._cache = cache or BufferCache()

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (freqs_cis := self._cache.get("rope_freqs_cis")) is not None and freqs_cis.shape[
            -2
        ] >= seq_len:
            if freqs_cis.device != device:
                freqs_cis = freqs_cis.to(device)
                self._cache["rope_freqs_cis"] = freqs_cis
            return freqs_cis[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(0, self.dim, 2, device=device, dtype=torch.float)[
                        : (self.dim // 2)
                    ]
                    / self.dim
                )
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._cache["rope_freqs_cis"] = freqs_cis
        return freqs_cis

    def apply_rotary_pos_emb(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(x * freqs_cis).flatten(3)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (B, nh, T, hs), (B, n_kv_h, T, hs)
        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        # shape: (B, nh, T, hs // 2), (B, n_kv_h, T, hs // 2), complex64
        q_ = torch.view_as_complex(q_.reshape(*q_.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k_.reshape(*k_.shape[:-1], -1, 2))

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = (
                q_.shape[-2],
                k_.shape[-2],
            )  # could be different if layer_past not None
            # shape: (1, 1, T, hs // 2)
            freqs_cis = self.get_rotary_embedding(key_len, q_.device).view(1, 1, *q_.shape[-2:])
            q_ = self.apply_rotary_pos_emb(
                freqs_cis[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(freqs_cis, k_)
        return q_.type_as(q), k_.type_as(k)
