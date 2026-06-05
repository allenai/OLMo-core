"""
EXPERIMENTAL fused Triton forward kernel for sparse landmark-only-across-chunks attention
(see :mod:`landmark_sparse` for the reference + the pattern). The reference efficient path is
sub-quadratic but materializes the score tensors ``(B,H,C,L,C*G)`` and runs a Python-level softmax;
this kernel fuses everything flash-style:

  one program per (query-block, head): online-softmax over its own chunk (causal) and the
  pre-gathered landmark keys of all strictly-past chunks (``num_landmarks`` per chunk). No score
  tensor is materialized, so it is much faster + lower-memory at long context.

Forward only for now (the natural next step is a flash-style backward); a correctness check vs the
dense reference and a benchmark live in ``landmark_investigation/diagnostic_sparse_triton.py``.
"""
import math
from typing import Optional

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore


def has_sparse_kernel() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _sparse_fwd_kernel(
        Q, K, V, KLM, VLM, Out, sm_scale,
        sqz, sqh, sqm, sqd,           # Q strides (B,H,T,D)  (K,V share these)
        slz, slh, sln, sld,           # KLM/VLM strides (B,H,C*G,D)
        H, N_CTX, N_LM,
        L: tl.constexpr, BLOCK_DMODEL: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        qb = tl.program_id(0)          # query-block (chunk) index
        off_hz = tl.program_id(1)
        offs_l = tl.arange(0, L)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_base = off_hz * sqh + (qb * L + offs_l)[:, None] * sqm + offs_d[None, :] * sqd
        q = tl.load(Q + q_base)        # (L, D)

        m_i = tl.zeros([L], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([L], dtype=tl.float32)
        acc = tl.zeros([L, BLOCK_DMODEL], dtype=tl.float32)

        # ---- own chunk (causal within the chunk) ----
        ko_base = off_hz * sqh + (qb * L + offs_l)[:, None] * sqm + offs_d[None, :] * sqd
        k_own = tl.load(K + ko_base)   # (L, D)
        v_own = tl.load(V + ko_base)
        qk = tl.dot(q, tl.trans(k_own), allow_tf32=False) * sm_scale
        qk = tl.where(offs_l[:, None] >= offs_l[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_own.dtype), v_own, allow_tf32=False)
        m_i = m_new

        # ---- past-chunk landmark keys: KLM[0 : qb*G] (runtime upper bound -> allowed) ----
        n_lm = qb * G
        for start in range(0, n_lm, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask = offs_n < n_lm
            lm_base = off_hz * slh + offs_n[:, None] * sln + offs_d[None, :] * sld
            k_lm = tl.load(KLM + lm_base, mask=mask[:, None], other=0.0)   # (BLOCK_N, D)
            v_lm = tl.load(VLM + lm_base, mask=mask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k_lm), allow_tf32=False) * sm_scale
            qk = tl.where(mask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_lm.dtype), v_lm, allow_tf32=False)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_base = off_hz * sqh + (qb * L + offs_l)[:, None] * sqm + offs_d[None, :] * sqd
        tl.store(Out + o_base, acc.to(Out.dtype.element_ty))


def sparse_landmark_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    num_landmarks: int = 1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused Triton forward for sparse landmark attention. ``q,k,v``: ``(B, H, T, D)`` (contiguous),
    ``T`` a multiple of ``block_size``. Returns ``(B, H, T, D)``. Forward only (no autograd).
    """
    assert has_sparse_kernel(), "sparse Triton kernel requires triton + CUDA"
    B, H, T, D = q.shape
    L, G = block_size, num_landmarks
    assert T % L == 0 and 1 <= G < L
    C = T // L
    scale = scale if scale is not None else D**-0.5
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # pre-gather landmark K/V (last G of each chunk) -> (B, H, C*G, D), contiguous
    k_lm = k.view(B, H, C, L, D)[:, :, :, L - G:, :].reshape(B, H, C * G, D).contiguous()
    v_lm = v.view(B, H, C, L, D)[:, :, :, L - G:, :].reshape(B, H, C * G, D).contiguous()
    o = torch.empty_like(q)

    grid = (C, B * H)
    _sparse_fwd_kernel[grid](
        q, k, v, k_lm, v_lm, o, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_lm.stride(0), k_lm.stride(1), k_lm.stride(2), k_lm.stride(3),
        H, T, C * G,
        L=L, BLOCK_DMODEL=D, G=G, BLOCK_N=L,
        num_warps=4, num_stages=2,
    )
    return o
