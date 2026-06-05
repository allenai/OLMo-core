"""
``SparseLandmarkAttention`` -- a sparse "landmark-only-across-chunks" attention sequence mixer.

Independent of (and not modifying) the original landmark attention. A query attends *fully* within
its own chunk (causal), but sees **past chunks only through their landmark tokens** (the last
``num_landmarks`` tokens of each chunk) -- the non-landmark content of past chunks is not attended.
Each past chunk is thus compressed to a few keys, so a query attends to about

    block_size (own chunk)  +  num_landmarks * num_past_chunks   keys

instead of the whole prefix. Score cost drops from ``O(T^2)`` to ``O(T*(L + num_landmarks*T/L))``
(``L = block_size``, ``C = T/L`` chunks) -- e.g. ~60x fewer score FLOPs at L=64, num_landmarks=1,
T=64k -- which is the source of the speedup over standard / dense-landmark attention.

Layout: the **last ``num_landmarks`` tokens of every chunk are landmarks**; ``T`` is a multiple of
``L`` and ``1 <= num_landmarks < L``.

  * :func:`sparse_landmark_attention_ref` -- dense O(T^2) masked reference (correctness).
  * :func:`sparse_landmark_attention`     -- efficient chunked form (block-local + landmark gather +
    one combined softmax); pure autograd-differentiable torch ops (backward for free).
  * :class:`SparseLandmarkAttention`       -- the :class:`Attention` sequence mixer wrapping it
    (``AttentionType.sparse_landmark``). A fused Triton kernel is the natural next step for peak speed.
"""
import os
from typing import Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from . import Attention
from .landmark import repeat_kv
from .landmark_sparse_kernel import (
    has_sparse_kernel,
    sparse_landmark_attention_triton_train,
)


def sparse_landmark_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    num_landmarks: int = 1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Dense O(T^2) reference. ``q,k,v``: ``(B, H, T, D)``; ``T`` a multiple of ``block_size``.
    A query at position i (chunk ``c = i // L``) attends to key j iff:
      - same chunk and causal: ``j // L == c`` and ``j <= i``, OR
      - j is a landmark of a strictly-past chunk: ``(j % L) >= L - G`` and ``j // L < c``.
    """
    B, H, T, D = q.shape
    L, G = block_size, num_landmarks
    assert T % L == 0 and 1 <= G < L
    scale = scale if scale is not None else D**-0.5
    dev = q.device

    pos = torch.arange(T, device=dev)
    chunk = pos // L
    is_lm = (pos % L) >= (L - G)

    same_chunk_causal = (chunk[:, None] == chunk[None, :]) & (pos[None, :] <= pos[:, None])
    past_landmark = is_lm[None, :] & (chunk[None, :] < chunk[:, None])
    allowed = same_chunk_causal | past_landmark

    att = torch.matmul(q, k.transpose(-1, -2)) * scale
    att = att.masked_fill(~allowed[None, None], float("-inf"))
    return torch.matmul(torch.softmax(att, dim=-1), v)


def sparse_landmark_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    num_landmarks: int = 1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Efficient chunked form, numerically equal to :func:`sparse_landmark_attention_ref`.
    ``q,k,v``: ``(B, H, T, D)``; ``T`` a multiple of ``block_size``; ``1 <= num_landmarks < block_size``.

    Own-chunk scores are ``(B,H,C,L,L)`` and landmark scores are ``(B,H,C,L,C,G)`` (``C=T/L``), so
    compute/peak-memory is ``O(B*H*T*(L + G*T/L))`` instead of ``O(B*H*T^2)``.
    """
    B, H, T, D = q.shape
    L, G = block_size, num_landmarks
    assert T % L == 0 and 1 <= G < L
    C = T // L
    scale = scale if scale is not None else D**-0.5
    dev = q.device
    neg_inf = float("-inf")

    qc = q.view(B, H, C, L, D)
    kc = k.view(B, H, C, L, D)
    vc = v.view(B, H, C, L, D)

    s_own = torch.matmul(qc, kc.transpose(-1, -2)) * scale
    causal = torch.tril(torch.ones(L, L, device=dev, dtype=torch.bool))
    s_own = s_own.masked_fill(~causal, neg_inf)

    k_lm = kc[:, :, :, L - G :, :]
    v_lm = vc[:, :, :, L - G :, :]

    s_lm = torch.einsum("bhcld,bhkgd->bhclkg", qc, k_lm) * scale
    cidx = torch.arange(C, device=dev)
    past = cidx[None, :] < cidx[:, None]
    s_lm = s_lm.masked_fill(~past[None, None, :, None, :, None], neg_inf)
    s_lm = s_lm.reshape(B, H, C, L, C * G)

    s = torch.cat([s_own, s_lm], dim=-1)
    p = torch.softmax(s, dim=-1)
    p_own = p[..., :L]
    p_lm = p[..., L:].reshape(B, H, C, L, C, G)

    out = torch.matmul(p_own, vc)
    out = out + torch.einsum("bhclkg,bhkgd->bhcld", p_lm, v_lm)
    return out.reshape(B, H, T, D)


def attended_key_count(T: int, block_size: int, num_landmarks: int = 1) -> float:
    """Average #keys a query attends (vs T for full causal) -- a FLOP proxy."""
    L, C, G = block_size, T // block_size, num_landmarks
    return (L + 1) / 2 + G * (C - 1) / 2


class SparseLandmarkAttention(Attention):
    """
    Sparse landmark-only-across-chunks attention as a drop-in :class:`Attention` variant
    (``AttentionType.sparse_landmark``). Each chunk is ``block_size = mem_freq + num_landmarks``
    tokens, the last ``num_landmarks`` of which are landmarks. Pure-torch (autograd) -- works on
    CPU/GPU; context parallelism is not yet supported.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        num_landmarks: int = 1,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("gate") is not None:
            raise OLMoConfigurationError("SparseLandmarkAttention does not support attention gating")
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "SparseLandmarkAttention does not support sliding window attention"
            )
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        if mem_freq is None or mem_freq < 1:
            raise OLMoConfigurationError(f"SparseLandmarkAttention requires mem_freq >= 1 (got {mem_freq})")
        if num_landmarks < 1:
            raise OLMoConfigurationError(f"num_landmarks must be >= 1 (got {num_landmarks})")
        self.mem_freq = mem_freq
        self.num_landmarks = num_landmarks
        self.block_size = mem_freq + num_landmarks
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5

    def apply_cp(self, *args, **kwargs):
        raise NotImplementedError("Context parallelism is not yet supported for SparseLandmarkAttention")

    @torch.compiler.disable
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
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if any(v is not None for v in (cu_doc_lens, cu_doc_lens_q, cu_doc_lens_k, max_doc_len,
                                       max_doc_len_q, max_doc_len_k, local_k_slice)):
            raise NotImplementedError(
                "Intra-document masking (cu_doc_lens) is not supported with sparse landmark attention"
            )
        if cache_leftpad is not None or self.kv_cache_manager is not None:
            raise NotImplementedError(
                "KV-caching / generation is not supported with sparse landmark attention"
            )

        B, T, _ = x.shape
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
        )
        if T % self.block_size != 0:
            raise OLMoConfigurationError(
                f"Sequence length ({T}) must be a multiple of block_size "
                f"(mem_freq + num_landmarks = {self.block_size})."
            )
        n_rep = q.shape[2] // k.shape[2]
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), n_rep)
        v = repeat_kv(v.transpose(1, 2), n_rep)

        # Fused Triton fwd+bwd kernel when available (much faster); eager torch fallback otherwise.
        # Disable with LM_SPARSE_KERNEL=0.
        if q.is_cuda and has_sparse_kernel() and os.environ.get("LM_SPARSE_KERNEL", "1") != "0":
            att = sparse_landmark_attention_triton_train(
                q, k, v, self.block_size, num_landmarks=self.num_landmarks, scale=self.softmax_scale
            )
        else:
            att = sparse_landmark_attention(
                q, k, v, self.block_size, num_landmarks=self.num_landmarks, scale=self.softmax_scale
            )
        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_out(att)
