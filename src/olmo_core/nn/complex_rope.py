from typing import Optional, Tuple

import torch
import torch.nn as nn

from .buffer_cache import BufferCache


class ComplexRotaryEmbedding(nn.Module):
    """
    An implementation of `RoPE <https://arxiv.org/abs/2104.09864>`_ as a rotation in complex space.

    .. seealso::
        :class:`~olmo_core.nn.rope.RotaryEmbedding`
    """

    def __init__(
        self,
        *,
        head_shape: int,
        theta: int = 500_000,
        full_precision: bool = True,
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()
        self.dim = head_shape
        self.theta = theta
        self.full_precision = full_precision
        self._cache = cache or BufferCache()

    def _get_rotary_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
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

    def _apply_rotary_pos_emb(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(x * freqs_cis).flatten(3)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, head_first: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query (``q``) and key (``k``) matrices.

        :param q: The query matrix of shape ``(batch_size, num_heads, seq_len, head_shape)``
            if ``head_first`` (the default) otherwise ``(batch_size, seq_len, num_heads, head_shape)``.
        :param k: The key matrix of shape ``(batch_size, num_kv_heads, seq_len, head_shape)``
            if ``head_first`` (the default) otherwise
            ``(batch_size, seq_len, num_kv_heads, head_shape)``.

        :returns: The query and key matrices after RoPE has been applied.
        """
        if head_first:
            q_len = q.size(2)
            k_len = k.size(2)
        else:
            q_len = q.size(1)
            k_len = k.size(1)

        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        # shape (complex64):
        #  (B, nh, T, hs // 2), (B, n_kv_h, T, hs // 2) if `head_first`, else
        #  (B, T, nh, hs // 2), (B, T, n_kv_h, hs // 2)
        q_ = torch.view_as_complex(q_.reshape(*q_.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k_.reshape(*k_.shape[:-1], -1, 2))

        with torch.autocast(q.device.type, enabled=False):
            # shape: (T, hs // 2)
            freqs_cis = self._get_rotary_embedding(k_len, q_.device)
            if head_first:
                # shape: (1, 1, T, hs // 2)
                q_ = self._apply_rotary_pos_emb(
                    freqs_cis[None, None, k_len - q_len : k_len, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(freqs_cis[None, None, :, :], k_)
            else:
                # shape: (1, T, 1, hs // 2)
                q_ = self._apply_rotary_pos_emb(
                    freqs_cis[None, k_len - q_len : k_len, None, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(freqs_cis[None, :, None, :], k_)

        return q_.type_as(q), k_.type_as(k)
