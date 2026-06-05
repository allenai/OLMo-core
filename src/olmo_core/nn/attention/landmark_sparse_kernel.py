"""
Fused Triton kernels (forward + backward) for sparse landmark-only-across-chunks attention
(see :mod:`landmark_sparse` for the reference + the pattern). Flash-style, no score materialization:

  * forward: one program per (query-chunk, head) -- online softmax over the chunk's own keys
    (causal) + a pre-gathered buffer of all strictly-past chunks' landmark keys.
  * backward: atomic-free, two kernels --
      - ``_bwd_dq_kernel``   : one program per (query-chunk, head) -> dq, and dk/dv for the chunk's
                               own keys (attended only by that chunk's causal queries).
      - ``_bwd_dklm_kernel`` : one program per (landmark-tile, head) -> dk/dv for the landmark keys,
                               accumulated over all future query-chunks that attend them.
    The landmark dk/dv are written to compact buffers and scatter-added back into dK/dV at the
    landmark positions (the last ``num_landmarks`` of each chunk) in torch.

The autograd ``Function`` :class:`_SparseLandmarkAttnFn` wires fwd+bwd together;
:func:`sparse_landmark_attention_triton_train` is the differentiable entry point used by
``SparseLandmarkAttention``. :func:`sparse_landmark_attention_triton` is a forward-only (inference)
helper. Verified against the dense reference + autograd in
``landmark_investigation/diagnostic_sparse_triton.py``.
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
        Q, K, V, KLM, VLM, Out, Lse, sm_scale,
        sqz, sqh, sqm, sqd,
        slz, slh, sln, sld,
        H, N_CTX, N_LM,
        L: tl.constexpr, BLOCK_DMODEL: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        qb = tl.program_id(0)
        off_hz = tl.program_id(1)
        offs_l = tl.arange(0, L)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_base = off_hz * sqh + (qb * L + offs_l)[:, None] * sqm + offs_d[None, :] * sqd
        q = tl.load(Q + q_base)

        m_i = tl.zeros([L], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([L], dtype=tl.float32)
        acc = tl.zeros([L, BLOCK_DMODEL], dtype=tl.float32)

        k_own = tl.load(K + q_base)
        v_own = tl.load(V + q_base)
        qk = tl.dot(q, tl.trans(k_own), allow_tf32=False) * sm_scale
        qk = tl.where(offs_l[:, None] >= offs_l[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_own.dtype), v_own, allow_tf32=False)
        m_i = m_new

        n_lm = qb * G
        for start in range(0, n_lm, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask = offs_n < n_lm
            lm_base = off_hz * slh + offs_n[:, None] * sln + offs_d[None, :] * sld
            k_lm = tl.load(KLM + lm_base, mask=mask[:, None], other=0.0)
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
        tl.store(Out + q_base, acc.to(Out.dtype.element_ty))
        # log-sum-exp for the backward
        lse = m_i + tl.log(l_i)
        tl.store(Lse + off_hz * N_CTX + qb * L + offs_l, lse)

    @triton.jit
    def _bwd_dq_kernel(
        Q, K, V, KLM, VLM, DO, DQ, DK, DV, Lse, Delta, sm_scale,
        sqz, sqh, sqm, sqd,
        slz, slh, sln, sld,
        H, N_CTX, N_LM,
        L: tl.constexpr, BLOCK_DMODEL: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # dq (own chunk + past landmarks) and dk/dv for the chunk's OWN keys. One program per
        # (query-chunk, head). Atomic-free: dq/dk_own/dv_own are unique to this chunk.
        qb = tl.program_id(0)
        off_hz = tl.program_id(1)
        offs_l = tl.arange(0, L)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        q_base = off_hz * sqh + (qb * L + offs_l)[:, None] * sqm + offs_d[None, :] * sqd

        q = tl.load(Q + q_base)
        do = tl.load(DO + q_base)
        lse = tl.load(Lse + off_hz * N_CTX + qb * L + offs_l)
        delta = tl.load(Delta + off_hz * N_CTX + qb * L + offs_l)

        dq = tl.zeros([L, BLOCK_DMODEL], dtype=tl.float32)

        # own chunk (causal): also produces dk/dv for these keys
        k_own = tl.load(K + q_base)
        v_own = tl.load(V + q_base)
        qk = tl.dot(q, tl.trans(k_own), allow_tf32=False) * sm_scale
        causal = offs_l[:, None] >= offs_l[None, :]
        p = tl.where(causal, tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v_own), allow_tf32=False)
        ds = p * (dp - delta[:, None]) * sm_scale
        dq += tl.dot(ds.to(q.dtype), k_own, allow_tf32=False)
        dk_own = tl.dot(tl.trans(ds.to(q.dtype)), q, allow_tf32=False)
        dv_own = tl.dot(tl.trans(p.to(do.dtype)), do, allow_tf32=False)
        tl.store(DK + q_base, dk_own.to(DK.dtype.element_ty))
        tl.store(DV + q_base, dv_own.to(DV.dtype.element_ty))

        # past landmarks (dq only; their dk/dv handled by the dklm kernel)
        n_lm = qb * G
        for start in range(0, n_lm, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask = offs_n < n_lm
            lm_base = off_hz * slh + offs_n[:, None] * sln + offs_d[None, :] * sld
            k_lm = tl.load(KLM + lm_base, mask=mask[:, None], other=0.0)
            v_lm = tl.load(VLM + lm_base, mask=mask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k_lm), allow_tf32=False) * sm_scale
            p = tl.where(mask[None, :], tl.exp(qk - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v_lm), allow_tf32=False)
            ds = p * (dp - delta[:, None]) * sm_scale
            dq += tl.dot(ds.to(q.dtype), k_lm, allow_tf32=False)

        tl.store(DQ + q_base, dq.to(DQ.dtype.element_ty))

    @triton.jit
    def _bwd_dklm_kernel(
        Q, KLM, VLM, DO, DKLM, DVLM, Lse, Delta, sm_scale,
        sqz, sqh, sqm, sqd,
        slz, slh, sln, sld,
        H, N_CTX, N_LM, N_CHUNK,
        L: tl.constexpr, BLOCK_DMODEL: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # dk/dv for a tile of landmark keys, accumulated over all future query-chunks. One program
        # per (landmark-tile, head). Atomic-free: each landmark key is written by exactly one program.
        tile = tl.program_id(0)
        off_hz = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = offs_n < N_LM
        lm_chunk = offs_n // G  # chunk each landmark belongs to

        lm_base = off_hz * slh + offs_n[:, None] * sln + offs_d[None, :] * sld
        k_lm = tl.load(KLM + lm_base, mask=nmask[:, None], other=0.0)
        v_lm = tl.load(VLM + lm_base, mask=nmask[:, None], other=0.0)
        dk_lm = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv_lm = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        offs_l = tl.arange(0, L)
        # query-chunk qb attends landmark j iff qb > lm_chunk[j]. Static lower bound 0 (runtime
        # lower bounds trip a Triton bug); per-iteration mask drops the qb <= lm_chunk entries.
        for qb in range(0, N_CHUNK):
            q_base = off_hz * sqh + (qb * L + offs_l)[:, None] * sqm + offs_d[None, :] * sqd
            q = tl.load(Q + q_base)
            do = tl.load(DO + q_base)
            lse = tl.load(Lse + off_hz * N_CTX + qb * L + offs_l)
            delta = tl.load(Delta + off_hz * N_CTX + qb * L + offs_l)

            qk = tl.dot(q, tl.trans(k_lm), allow_tf32=False) * sm_scale  # (L, BLOCK_N)
            attend = (qb > lm_chunk[None, :]) & nmask[None, :]
            p = tl.where(attend, tl.exp(qk - lse[:, None]), 0.0)
            dv_lm += tl.dot(tl.trans(p.to(do.dtype)), do, allow_tf32=False)
            dp = tl.dot(do, tl.trans(v_lm), allow_tf32=False)  # (L, BLOCK_N)
            ds = p * (dp - delta[:, None]) * sm_scale
            dk_lm += tl.dot(tl.trans(ds.to(q.dtype)), q, allow_tf32=False)

        tl.store(DKLM + lm_base, dk_lm.to(DKLM.dtype.element_ty), mask=nmask[:, None])
        tl.store(DVLM + lm_base, dv_lm.to(DVLM.dtype.element_ty), mask=nmask[:, None])


def _gather_landmarks(k, v, B, H, C, L, G, D):
    k_lm = k.view(B, H, C, L, D)[:, :, :, L - G:, :].reshape(B, H, C * G, D).contiguous()
    v_lm = v.view(B, H, C, L, D)[:, :, :, L - G:, :].reshape(B, H, C * G, D).contiguous()
    return k_lm, v_lm


def _fwd(q, k, v, L, G, scale):
    B, H, T, D = q.shape
    C = T // L
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    k_lm, v_lm = _gather_landmarks(k, v, B, H, C, L, G, D)
    o = torch.empty_like(q)
    lse = torch.empty((B * H, T), device=q.device, dtype=torch.float32)
    grid = (C, B * H)
    _sparse_fwd_kernel[grid](
        q, k, v, k_lm, v_lm, o, lse, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_lm.stride(0), k_lm.stride(1), k_lm.stride(2), k_lm.stride(3),
        H, T, C * G,
        L=L, BLOCK_DMODEL=D, G=G, BLOCK_N=L, num_warps=4, num_stages=2,
    )
    return o, lse, k_lm, v_lm


def sparse_landmark_attention_triton(q, k, v, block_size, num_landmarks=1, scale=None):
    """Forward-only (inference) fused sparse-landmark attention. ``q,k,v``: (B,H,T,D)."""
    assert has_sparse_kernel()
    L, G = block_size, num_landmarks
    assert q.shape[2] % L == 0 and 1 <= G < L
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    o, _, _, _ = _fwd(q, k, v, L, G, scale)
    return o


if triton is not None:

    class _SparseLandmarkAttnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, block_size, num_landmarks, scale):
            L, G = block_size, num_landmarks
            scale = scale if scale is not None else q.shape[-1] ** -0.5
            o, lse, k_lm, v_lm = _fwd(q, k, v, L, G, scale)
            ctx.save_for_backward(q.contiguous(), k.contiguous(), v.contiguous(), k_lm, v_lm, o, lse)
            ctx.L, ctx.G, ctx.scale = L, G, scale
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, k_lm, v_lm, o, lse = ctx.saved_tensors
            L, G, scale = ctx.L, ctx.G, ctx.scale
            B, H, T, D = q.shape
            C = T // L
            do = do.contiguous()
            delta = (do.float() * o.float()).sum(-1).reshape(B * H, T).contiguous()

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dk_lm = torch.zeros_like(k_lm)
            dv_lm = torch.zeros_like(v_lm)
            grid = (C, B * H)
            _bwd_dq_kernel[grid](
                q, k, v, k_lm, v_lm, do, dq, dk, dv, lse, delta, scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k_lm.stride(0), k_lm.stride(1), k_lm.stride(2), k_lm.stride(3),
                H, T, C * G,
                L=L, BLOCK_DMODEL=D, G=G, BLOCK_N=L, num_warps=4, num_stages=2,
            )
            n_tiles = triton.cdiv(C * G, L)
            _bwd_dklm_kernel[(n_tiles, B * H)](
                q, k_lm, v_lm, do, dk_lm, dv_lm, lse, delta, scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k_lm.stride(0), k_lm.stride(1), k_lm.stride(2), k_lm.stride(3),
                H, T, C * G, C,
                L=L, BLOCK_DMODEL=D, G=G, BLOCK_N=L, num_warps=4, num_stages=2,
            )
            # scatter-add landmark grads back into the last-G positions of each chunk
            dk.view(B, H, C, L, D)[:, :, :, L - G:, :] += dk_lm.view(B, H, C, G, D)
            dv.view(B, H, C, L, D)[:, :, :, L - G:, :] += dv_lm.view(B, H, C, G, D)
            return dq, dk, dv, None, None, None


def sparse_landmark_attention_triton_train(q, k, v, block_size, num_landmarks=1, scale=None):
    """Autograd-enabled fused sparse-landmark attention (Triton fwd + bwd). ``q,k,v``: (B,H,T,D)."""
    assert has_sparse_kernel()
    L, G = block_size, num_landmarks
    assert q.shape[2] % L == 0 and 1 <= G < L
    return _SparseLandmarkAttnFn.apply(q, k, v, L, G, scale)
