from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config, StrEnum
from .buffer_cache import BufferCache

__all__ = [
    "RoPEType",
    "RoPEConfig",
    "RotaryEmbeddingBase",
    "RotaryEmbedding",
    "FusedRotaryEmbedding",
    "ComplexRotaryEmbedding",
]


class RoPEType(StrEnum):
    """
    An enumeration of the different RoPE implementations.

    - "default" ➡️ :class:`RotaryEmbedding`
    - "fused" ➡️ :class:`FusedRotaryEmbedding`
    - "complex" ➡️ :class:`ComplexRotaryEmbedding`
    """

    default = "default"
    fused = "fused"
    complex = "complex"


@dataclass
class RoPEConfig(Config):
    """
    A config for conveniently building any one of the different RoPE classes.

    See :class:`RotaryEmbedding` for a description of the parameters.
    """

    name: RoPEType = RoPEType.default
    """
    - "default" ➡️ :class:`RotaryEmbedding`
    - "fused" ➡️ :class:`FusedRotaryEmbedding`
    - "complex" ➡️ :class:`ComplexRotaryEmbedding`
    """
    theta: int = 500_000
    full_precision: bool = True

    def build(
        self,
        head_shape: int,
        cache: Optional[BufferCache] = None,
    ) -> "RotaryEmbeddingBase":
        """
        Construct the corresponding RoPE class.

        See :class:`RotaryEmbedding` for a description of the parameters.
        """
        kwargs: Dict[str, Any] = dict(
            head_shape=head_shape,
            theta=self.theta,
            full_precision=self.full_precision,
            cache=cache,
        )

        if self.name == "default":
            return RotaryEmbedding(**kwargs)
        elif self.name == "fused":
            return FusedRotaryEmbedding(**kwargs)
        elif self.name == "complex":
            return ComplexRotaryEmbedding(**kwargs)
        else:
            raise NotImplementedError(self.name)


class RotaryEmbeddingBase(nn.Module):
    """
    Base class for RoPE implementations.
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

    @abstractmethod
    def warmup_cache(self, max_seq_len: int, device: torch.device):
        """
        Warmup the buffer cache.
        """
        raise NotImplementedError


class RotaryEmbedding(RotaryEmbeddingBase):
    """
    `Rotary positional embeddings (RoPE) <https://arxiv.org/abs/2104.09864>`_.

    .. seealso::
        - :class:`ComplexRotaryEmbedding`
        - :class:`FusedRotaryEmbedding`

    :param head_shape: The dimensionality of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    """

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

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


class FusedRotaryEmbedding(RotaryEmbeddingBase):
    """
    A "fused" triton-based implementation of :class:`RotaryEmbedding`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    :param head_shape: The dimensionality of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    """

    def __init__(
        self,
        *,
        head_shape: int,
        theta: int = 500_000,
        full_precision: bool = True,
        cache: Optional[BufferCache] = None,
    ):
        from flash_attn.layers.rotary import apply_rotary_emb_qkv_  # type: ignore

        super().__init__(
            head_shape=head_shape, theta=theta, full_precision=full_precision, cache=cache
        )
        self._apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

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
            pos_sin, pos_cos = freqs.sin(), freqs.cos()
        self._cache["rope_pos_sin"] = pos_sin
        self._cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to ``qkv``.

        .. warning::
            This operates on ``qkv`` *in place* unless ``full_precision=True`` and ``qkv``
            is not in full precision.

        :param qkv: The query, key, and value matrix of shape
            ``(batch_size, seq_len, 3, n_heads, head_shape)``.
        """
        if self.full_precision:
            qkv_ = qkv.float()
        else:
            qkv_ = qkv

        pos_sin, pos_cos = self._get_rotary_embedding(qkv_.size(1), qkv_.device)
        pos_sin, pos_cos = pos_sin.type_as(qkv_), pos_cos.type_as(qkv_)
        qkv_ = self._apply_rotary_emb_qkv_(
            qkv_, pos_cos, pos_sin, interleaved=False, seqlen_offsets=0
        )
        return qkv_.type_as(qkv)


class ComplexRotaryEmbedding(RotaryEmbeddingBase):
    """
    An implementation of `RoPE <https://arxiv.org/abs/2104.09864>`_ as a rotation in complex space.

    :param head_shape: The dimensionality of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    """

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

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
