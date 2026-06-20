"""
``FastCompressiveLandmarkAttention`` -- a *compressive* variant of :class:`FastLandmarkAttention`
that folds each block's landmark ("memory") token back into the attention output.

In ordinary landmark attention the landmark token of a past block only serves as a *gate*: its
score sets how much weight the whole block receives, and that weight is then spread over the block's
**content** tokens via a within-block softmax. The landmark token's own value contributes nothing to
the output (it is multiplied by zero, see :func:`~olmo_core.nn.attention.landmark.landmark_grouped_softmax`
and the ``normal_p`` paths of :mod:`landmark_kernel`).

The compressive variant keeps the gate (the block's weight still comes from the landmark score,
exactly as before), but the within-block softmax that distributes that weight now **includes the
landmark token** alongside the block's content tokens. The landmark token therefore contributes its
value to the output -- acting as a learned, compressed summary of its block.

Concretely, for a query attending to a fully-past block with (scaled) scores ``s_n`` over the block's
``BLOCK_N`` tokens (``n = BLOCK_N - 1`` is the landmark):

* gate weight of the block ``G = softmax_over_gate(s_landmark)`` -- unchanged from normal landmark;
* within-block distribution ``f_n = softmax_n(s_n)`` over **all** ``BLOCK_N`` tokens (compressive)
  instead of over the ``BLOCK_N - 1`` content tokens only;
* output contribution ``G * sum_n f_n v_n``.

The local ("last") section and the cross-block (gate) softmax are identical to normal landmark, so
``L``/``M`` (the saved softmax stats) are bit-identical to :class:`FastLandmarkAttention`; only the
value accumulation and its gradient change. The fused Triton forward/backward below mirror
:mod:`landmark_fast` one-to-one, swapping the content-only within-block softmax for the full-block
one and adding the landmark token's value gradient.

**Inference.** Training and prefill use the plain compressive softmax above (no extra hyperparameter).
At decode with hard top-k landmark retrieval (see :meth:`FastLandmarkAttention.set_landmark_eval_decode`),
the non-selected blocks are *not* dropped entirely as in the base class: their landmark tokens
collectively retain a fixed fraction ``nonselected_landmark_mass`` (``alpha``) of the attention mass
(split among them by a softmax over their landmark scores), while the remaining ``1 - alpha`` is
distributed over the local section and the selected blocks (content + landmarks) by the compressive
grouped softmax. This lets every past block keep contributing its compressed (landmark) representation
even when it is not in the top-k.
"""

import math
from typing import Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from .landmark_fast import FastLandmarkAttention, _env_int
from .landmark_kernel import _bwd_preprocess, has_landmark_kernel

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore


if triton is not None:

    @triton.jit
    def _fwd_kernel_compressive(
        Q,
        K,
        V,
        sm_scale,
        Out,
        sqz,
        sqh,
        sqm,
        sqd,
        skz,
        skh,
        skn,
        skd,
        svz,
        svh,
        svn,
        svd,
        soz,
        soh,
        som,
        sod,
        L,
        M,
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
    ):
        # Compressive landmark forward. Identical to landmark_kernel._fwd_kernel except that the
        # value contribution of a fully-past block uses the *full-block* within softmax (over all
        # BLOCK_N tokens including the landmark) instead of the content-only ``normal_p``. The gate
        # (cross-block) softmax that produces L/M is untouched, so L/M stay bit-identical to the
        # normal landmark kernel.
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd
        offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd

        m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)

        for start_n in range(0, (N_PREFIX_Q + start_m)):
            k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
            qk += tl.dot(q_vals, k_vals, allow_tf32=False)
            qk *= sm_scale
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float("-inf"))

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            # Compressive within-block softmax over ALL block tokens (content + landmark).
            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_denom = tl.sum(full_p, 1)

            m_curr = tl.maximum(landmark_qk, m_prev)
            m_curr_ = m_curr
            l_prev *= tl.exp(m_prev - m_curr_)
            landmark_p = tl.exp(landmark_qk - m_curr_)
            l_curr = landmark_p + l_prev
            l_rcp = 1.0 / l_curr
            landmark_p *= l_rcp

            acc *= (l_prev * l_rcp)[:, None]
            v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
            acc += tl.dot(
                (landmark_p[:, None] * full_p / full_denom[:, None]).to(Q.dtype.element_ty),
                v_vals,
                allow_tf32=False,
            )

            l_prev = l_curr
            m_prev = m_curr

            offs_n += BLOCK_N
            offs_k += BLOCK_N * skn
            offs_v += BLOCK_N * svn

        # Diagonal (local) block: standard causal softmax, identical to normal landmark.
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        m_curr_ = m_curr
        l_prev *= tl.exp(m_prev - m_curr_)
        p = tl.exp(qk - m_curr_[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        p = p.to(Q.dtype.element_ty)
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
        acc += tl.dot(p, v_vals, allow_tf32=False)

        l_prev = l_curr
        m_prev = m_curr

        offs_L = off_hz * N_CTX_Q + offs_m
        offs_M = off_hz * N_CTX_Q + offs_m
        tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
        tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)
        offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
        tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)

    @triton.jit
    def _bwd_kv_kernel_compressive(
        Q,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DK,
        DV,
        L,
        M,
        D,
        sqz,
        sqh,
        sqm,
        sqd,
        skz,
        skh,
        skn,
        skd,
        svz,
        svh,
        svn,
        svd,
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
    ):
        # dk/dv, one program per (key-block, head); atomic-free. Mirrors landmark_fast._bwd_kv_kernel
        # but with the compressive within-block softmax: every block token (incl. the landmark) gets
        # a within-block value weight, and the landmark score additionally carries the gate gradient.
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        Q += off_z * sqz + off_h * sqh
        K += off_z * skz + off_h * skh
        V += off_z * svz + off_h * svh
        DO += off_z * sqz + off_h * sqh
        DK += off_z * skz + off_h * skh
        DV += off_z * svz + off_h * svh

        offs_d = tl.arange(0, BLOCK_DMODEL)
        D_ptrs = D + off_hz * N_CTX_Q
        m_ptrs = M + off_hz * N_CTX_Q

        start_n = tl.program_id(1) * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)

        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            # Diagonal (local) block: standard causal softmax, identical to normal landmark.
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = first_start_m + tl.arange(0, BLOCK_M)
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M
            offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)

            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk = tl.where(offs_m_real[:, None] >= (offs_n[None, :]), qk, float("-inf"))

            m = tl.load(m_ptrs + offs_m)
            last_p = tl.exp(qk * sm_scale - m[:, None])

            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(last_p.to(Q.dtype.element_ty)), do, allow_tf32=False)

            Di = tl.load(D_ptrs + offs_m)
            last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = last_p * last_dp * sm_scale

            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)
            start_q_index = first_start_m + BLOCK_M

        for i in range(0, N_CTX_Q - start_q_index, BLOCK_M):
            start_m = start_q_index + i
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)

            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            # Compressive within-block softmax over all block tokens (content + landmark).
            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_dist = full_p / tl.sum(full_p, 1)[:, None]

            m = tl.load(m_ptrs + offs_m)
            p = tl.exp(landmark_qk - m)  # gate weight (numerator; /L folded into do_scaled)

            do = tl.load(do_ptrs)

            # dv: every token (incl. landmark) gets within-block weight p * full_dist.
            dv += tl.dot(
                tl.trans((p[:, None] * full_dist).to(Q.dtype.element_ty)),
                do,
                allow_tf32=False,
            )

            Di = tl.load(D_ptrs + offs_m)
            dpv = tl.dot(do, tl.trans(v), allow_tf32=False)  # do_scaled . v_n per col
            # full_D = do_scaled . fv = sum_n full_dist_n * (do_scaled . v_n). Computing it from dpv
            # avoids a separate ``dot(full_dist, v)`` and its (BLOCK_M, head_dim) accumulator, which
            # keeps the head_dim=256 / block=64 backward within A100 shared memory.
            full_D = tl.sum(full_dist * dpv, 1)
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])
            gate_ds = p * (full_D - Di)  # lands on the landmark column only
            ds = within_ds + tl.where(
                tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, gate_ds[:, None], 0.0
            )
            ds *= sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)

        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

    @triton.jit
    def _bwd_q_kernel_compressive(
        Q,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DK,
        DV,
        L,
        M,
        D,
        sqz,
        sqh,
        sqm,
        sqd,
        skz,
        skh,
        skn,
        skd,
        svz,
        svh,
        svn,
        svd,
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
    ):
        # dq, one program per (query-block, head). Causal-only key-block loop, atomic-free. Only
        # implemented for N_PREFIX_Q == 0 (the caller guards this).
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        Q += off_z * sqz + off_h * sqh
        K += off_z * skz + off_h * skh
        V += off_z * svz + off_h * svh
        DO += off_z * sqz + off_h * sqh
        DQ += off_z * sqz + off_h * sqh

        offs_d = tl.arange(0, BLOCK_DMODEL)
        D_ptrs = D + off_hz * N_CTX_Q
        m_ptrs = M + off_hz * N_CTX_Q

        start_m = tl.program_id(1) * BLOCK_M
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        do = tl.load(DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        m = tl.load(m_ptrs + offs_m)
        Di = tl.load(D_ptrs + offs_m)

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, start_m, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + (offs_n[:, None] * skn + offs_d[None, :] * skd))
            v = tl.load(V + (offs_n[:, None] * svn + offs_d[None, :] * svd))

            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_dist = full_p / tl.sum(full_p, 1)[:, None]
            p = tl.exp(landmark_qk - m)
            dpv = tl.dot(do, tl.trans(v), allow_tf32=False)
            # full_D = do_scaled . fv = sum_n full_dist_n * (do_scaled . v_n); see _bwd_kv_kernel.
            full_D = tl.sum(full_dist * dpv, 1)
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])
            gate_ds = p * (full_D - Di)
            ds = within_ds + tl.where(
                tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, gate_ds[:, None], 0.0
            )
            ds *= sm_scale
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)

        # diagonal key block: within-block causal attention (identical to normal landmark).
        offs_n = start_m + tl.arange(0, BLOCK_N)
        k = tl.load(K + (offs_n[:, None] * skn + offs_d[None, :] * skd))
        v = tl.load(V + (offs_n[:, None] * svn + offs_d[None, :] * svd))
        offs_m_real = offs_m + tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk = tl.where(offs_m_real[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        last_p = tl.exp(qk * sm_scale - m[:, None])
        last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = last_p * last_dp * sm_scale
        dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)

        tl.store(DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd), dq)


class _FusedCompressiveLandmarkAttention(torch.autograd.Function):
    """Fused compressive landmark attention (forward + FA2-style backward). The landmark token of
    each past block is included in that block's within-block softmax, so it contributes its value to
    the output. ``L``/``M`` (and hence the gate softmax) are identical to the normal landmark
    kernel; only the value accumulation and its gradient differ."""

    @staticmethod
    def forward(ctx, q, k, v, n_prefix_q, sm_scale, block_size):
        if triton is None:
            raise RuntimeError("Landmark attention requires 'triton' (and a CUDA device).")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch, nheads, seqlen_q, d = q.shape
        assert d <= 256 and q.dtype == k.dtype == v.dtype and q.is_cuda

        BLOCK = block_size
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if d > 128:
            num_warps = _env_int("LM_FAST_FWD_WARPS", 8)
            num_stages = _env_int("LM_FAST_FWD_STAGES", 2)
        else:
            num_warps = _env_int("LM_FAST_FWD_WARPS", 4)
            num_stages = _env_int("LM_FAST_FWD_STAGES", 3)
        _fwd_kernel_compressive[grid](
            q,
            k,
            v,
            sm_scale,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            L,
            m,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            BLOCK=BLOCK,
            BLOCK_DMODEL=d,
            N_PREFIX_Q=n_prefix_q,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = d
        ctx.N_PREFIX_Q = n_prefix_q
        ctx.BLOCK = BLOCK
        return o

    @staticmethod
    def backward(ctx, do):
        if ctx.N_PREFIX_Q != 0:
            raise NotImplementedError(
                "FastCompressiveLandmarkAttention backward only supports no history KV "
                "(N_PREFIX_Q == 0); generation runs without gradients."
            )

        BLOCK = ctx.BLOCK
        q, k, v, o, lse, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(lse)
        _bwd_preprocess[(ctx.grid[0], ctx.grid[1])](
            o,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do,
            lse,
            lse.stride(0),
            lse.stride(1),
            do_scaled,
            delta,
            q.shape[2],
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        args = (
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            lse,
            m,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
        )
        const = dict(BLOCK=BLOCK, BLOCK_DMODEL=ctx.BLOCK_DMODEL, N_PREFIX_Q=ctx.N_PREFIX_Q)
        if ctx.BLOCK_DMODEL > 128:
            warps = _env_int("LM_FAST_WARPS", 8)
            stages = _env_int("LM_FAST_STAGES", 1)
        else:
            warps = _env_int("LM_FAST_WARPS", 4)
            stages = _env_int("LM_FAST_STAGES", 2)
        n_kv_blocks = triton.cdiv(k.shape[2], BLOCK)
        _bwd_kv_kernel_compressive[(ctx.grid[1], n_kv_blocks)](
            *args, **const, num_warps=warps, num_stages=stages
        )
        _bwd_q_kernel_compressive[(ctx.grid[1], ctx.grid[0])](
            *args, **const, num_warps=warps, num_stages=stages
        )
        return dq, dk, dv, None, None, None


def fused_compressive_landmark_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_mem: torch.Tensor,
    sm_scale: float = None,  # type: ignore[assignment]
    block_size: int = 64,
) -> torch.Tensor:
    """Compressive-landmark counterpart of :func:`landmark_fast.fused_landmark_attention_fast`."""
    expected_is_mem = torch.arange(0, is_mem.shape[-1], device=is_mem.device) % block_size == (
        block_size - 1
    )
    assert (is_mem == expected_is_mem).all()
    n_history_kv = k.shape[-2] - q.shape[-2]
    assert n_history_kv % block_size == 0
    n_history_blocks = n_history_kv // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return _FusedCompressiveLandmarkAttention.apply(q, k, v, n_history_blocks, sm_scale, block_size)


class FastCompressiveLandmarkAttention(FastLandmarkAttention):
    """
    Compressive landmark attention (``AttentionType.fast_compressive_landmark``).

    Identical to :class:`FastLandmarkAttention` except that each past block's landmark token is
    folded into that block's within-block softmax, so the landmark contributes its value to the
    output (a learned compressed summary of the block). See the module docstring for the math.

    :param nonselected_landmark_mass: The fraction ``alpha in [0, 1)`` of attention mass reserved,
        at top-k decode time, for the landmark tokens of the *non-selected* blocks (split among them
        by a softmax over their landmark scores). The remaining ``1 - alpha`` is distributed over the
        local section and the selected blocks. Has no effect during training/prefill or when top-k
        retrieval is disabled.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        nonselected_landmark_mass: float = 0.1,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(mem_freq=mem_freq, softmax_scale=softmax_scale, **kwargs)
        if not (0.0 <= nonselected_landmark_mass < 1.0):
            raise OLMoConfigurationError(
                f"nonselected_landmark_mass must be in [0, 1) (got {nonselected_landmark_mass})"
            )
        self.nonselected_landmark_mass = nonselected_landmark_mass

    def set_landmark_eval_decode(
        self,
        prompt_len: int,
        mode: str = "extend_last_block",
        top_k: Optional[int] = None,
        nonselected_landmark_mass: Optional[float] = None,
    ) -> None:
        """Enable "one long local block" decoding (see :class:`FastLandmarkAttention`).

        :param nonselected_landmark_mass: Optionally override the module's default
            :attr:`nonselected_landmark_mass` for this eval run. Only used when ``top_k`` is set.
        """
        super().set_landmark_eval_decode(prompt_len, mode, top_k=top_k)
        if nonselected_landmark_mass is not None:
            if not (0.0 <= nonselected_landmark_mass < 1.0):
                raise OLMoConfigurationError(
                    f"nonselected_landmark_mass must be in [0, 1) (got {nonselected_landmark_mass})"
                )
            self.nonselected_landmark_mass = nonselected_landmark_mass

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not has_landmark_kernel():
            raise RuntimeError(
                "FastCompressiveLandmarkAttention requires the fused Triton kernel "
                "(install 'triton', run on CUDA)."
            )
        if doc_id is not None:
            raise NotImplementedError(
                "Intra-document packing (cu_doc_lens) is not yet supported by "
                "FastCompressiveLandmarkAttention's fused kernel."
            )
        T = q.shape[2]
        is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
        return fused_compressive_landmark_attention(
            q, k, v, is_mem, sm_scale=self.softmax_scale, block_size=self.block_size
        )

    def _compressive_decode_probs(
        self,
        scores: torch.Tensor,
        is_mem: torch.Tensor,
        last_section: torch.Tensor,
        section_start: int,
    ) -> torch.Tensor:
        """Compressive grouped softmax for a single decode query.

        :param scores: Attention logits of shape ``(B, H, 1, total)`` (already scaled).
        :param is_mem: Boolean mask ``(total,)`` marking *past* landmark key positions (the local
            section's landmark is never present, see :meth:`FastLandmarkAttention._decode_one`).
        :param last_section: Boolean mask ``(total,)`` marking the local-section keys.
        :param section_start: Start position of the local section; a multiple of ``block_size`` so
            the past region ``[0, section_start)`` partitions into whole landmark blocks.

        :returns: Attention probabilities of shape ``(B, H, 1, total)``.
        """
        B, H, _, total = scores.shape
        Lb = self.block_size
        S = section_start
        device = scores.device
        neg_inf = torch.finfo(scores.dtype).min

        is_mem_b = is_mem.view(1, 1, 1, total)
        last_section_b = last_section.view(1, 1, 1, total)
        lm_idx = is_mem.nonzero(as_tuple=True)[0]  # past landmark positions
        n_lm = int(lm_idx.numel())

        top_k = self._eval_top_k
        if top_k is not None and n_lm > top_k:
            lm_scores = scores[..., lm_idx]  # (B, H, 1, n_lm)
            keep = torch.zeros_like(lm_scores, dtype=torch.bool)
            keep.scatter_(-1, lm_scores.topk(top_k, dim=-1).indices, True)
            selected = torch.zeros(B, H, 1, total, dtype=torch.bool, device=device)
            selected[..., lm_idx] = keep
            has_nonselected = True
            alpha = float(self.nonselected_landmark_mass)
        else:
            selected = is_mem_b.expand(B, H, 1, total)
            has_nonselected = False
            alpha = 0.0

        # Gate (cross-block) softmax over the selected landmarks + the local section.
        gate_set = selected | last_section_b
        gate_w = torch.softmax(scores.masked_fill(~gate_set, neg_inf), dim=-1)

        final = torch.zeros(B, H, 1, total, dtype=gate_w.dtype, device=device)
        # Local section keys keep their gate weight directly.
        final = torch.where(last_section_b, gate_w, final)
        # Past blocks: full within-block softmax distributes the block's gate weight over its
        # content tokens AND its landmark. Non-selected blocks have gate weight 0 here (their
        # landmark was masked out of ``gate_set``), so they contribute 0 in this term.
        if S > 0:
            within = torch.softmax(scores[..., :S].reshape(B, H, 1, S // Lb, Lb), dim=-1)
            within = within.reshape(B, H, 1, S)
            block_landmark_pos = torch.arange(Lb - 1, S, Lb, device=device)
            block_gate = gate_w[..., block_landmark_pos]  # (B, H, 1, n_past_blocks)
            block_gate_full = block_gate.repeat_interleave(Lb, dim=-1)  # (B, H, 1, S)
            final = final.clone()
            final[..., :S] = block_gate_full * within

        if has_nonselected:
            final = final * (1.0 - alpha)
            nonsel = is_mem_b & (~selected)
            ns_w = torch.softmax(scores.masked_fill(~nonsel, neg_inf), dim=-1)
            final = final + alpha * ns_w

        return final

    def _decode_one(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: int
    ) -> torch.Tensor:
        Lb = self.block_size
        if self._eval_prompt_len is not None and qpos >= self._eval_prompt_len:
            return self._decode_one_eval(q, k, v, qpos)
        if qpos % Lb == Lb - 1:
            k = k[:, :, :qpos]
            v = v[:, :, :qpos]
        total = k.shape[2]
        j = torch.arange(total, device=q.device)
        is_mem = (j % Lb) == (Lb - 1)
        last_section = (j // Lb) == (qpos // Lb)
        section_start = (qpos // Lb) * Lb

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        probs = self._compressive_decode_probs(scores, is_mem, last_section, section_start)
        return torch.matmul(probs.to(v.dtype), v)

    def _decode_one_eval(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: int
    ) -> torch.Tensor:
        Lb = self.block_size
        P = self._eval_prompt_len
        assert P is not None
        section_start = (P // Lb) * Lb if self._eval_decode_mode == "extend_last_block" else P

        total = k.shape[2]
        j = torch.arange(total, device=q.device)
        is_mem = ((j % Lb) == (Lb - 1)) & (j < section_start)
        last_section = j >= section_start

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        probs = self._compressive_decode_probs(scores, is_mem, last_section, section_start)
        return torch.matmul(probs.to(v.dtype), v)
