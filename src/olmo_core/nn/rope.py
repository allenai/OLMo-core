from typing import Optional, Tuple

import torch
import torch.nn as nn

from .buffer_cache import BufferCache


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
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

    def get_rotary_embedding(
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
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float) / self.dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self._cache["rope_pos_sin"] = pos_sin
        self._cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = (
                q_.shape[-2],
                k_.shape[-2],
            )  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
