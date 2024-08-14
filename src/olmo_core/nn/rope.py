from typing import Optional, Tuple

import torch
import torch.nn as nn

from .buffer_cache import BufferCache


class RotaryEmbedding(nn.Module):
    """
    `Rotary positional embeddings (RoPE) <https://arxiv.org/abs/2104.09864>`_.

    .. seealso::
        :class:`~olmo_core.nn.complex_rope.ComplexRotaryEmbedding`
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

    def _get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self._cache.get("rope_pos_sin")) is not None
            and (pos_cos := self._cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self._cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self._cache["rope_pos_cos"] = pos_cos
            return pos_sin[:seq_len, :], pos_cos[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float) / self.dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin(), positions.cos()
        self._cache["rope_pos_sin"] = pos_sin
        self._cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)

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
        :param head_first: If the head dim comes before the sequence dim.

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

        with torch.autocast(q.device.type, enabled=False):
            # shape: (T, head_shape), (T, head_shape)
            pos_sin, pos_cos = self._get_rotary_embedding(k_len, q_.device)
            pos_sin, pos_cos = pos_sin.type_as(q_), pos_cos.type_as(q_)
            if head_first:
                q_ = self._apply_rotary_pos_emb(
                    pos_sin[None, None, k_len - q_len : k_len, :],
                    pos_cos[None, None, k_len - q_len : k_len, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(
                    pos_sin[None, None, :, :], pos_cos[None, None, :, :], k_
                )
            else:
                q_ = self._apply_rotary_pos_emb(
                    pos_sin[None, k_len - q_len : k_len, None, :],
                    pos_cos[None, k_len - q_len : k_len, None, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(
                    pos_sin[None, :, None, :], pos_cos[None, :, None, :], k_
                )
        return q_.type_as(q), k_.type_as(k)
