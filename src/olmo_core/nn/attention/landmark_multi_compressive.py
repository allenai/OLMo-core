"""
``MultiCompressiveLandmarkAttention`` -- a *multi-landmark* variant of
:class:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention` that lets each
block carry **several** landmark ("memory") tokens instead of exactly one.

Layout: a block is ``block_size = mem_freq + num_landmarks`` tokens; the **last ``num_landmarks``
tokens of every block are landmarks** (``num_landmarks == 1`` reproduces the single-landmark
compressive layout). Because the fused Triton kernel tiles by ``block_size`` (``tl.arange`` /
``tl.dot`` need power-of-two tile dims), ``block_size`` is held to a **power of two**: adding
landmarks trades content for landmarks at a fixed block size (e.g. block 32 -> 31 content + 1
landmark, 30 + 2, 28 + 4, ...).

The compressive math is unchanged except for how a past block's single cross-block **gate** weight is
formed from its several landmark scores. With per-landmark (scaled) scores ``s_{b,l}`` the block gate
logit is a **pool of the logits**, selected by ``landmark_gate_pool``:

  * ``"mean"``: ``g_b = mean_l s_{b,l}`` -- average affinity across the block's landmarks
    (count-neutral; every landmark receives gate gradient ``1/num_landmarks``).
  * ``"max"``: ``g_b = max_l s_{b,l}`` -- the block's single best-matching landmark (gate gradient
    flows only to the argmax, matching ``torch.amax``).

The cross-block softmax then runs over ``{g_b for visible past blocks}`` plus the query's own (local)
section, exactly as in the single-landmark kernel; the within-block softmax still spans **all**
``block_size`` tokens (content *and* every landmark), so every landmark folds its value into the
block's compressed summary. ``num_landmarks == 1`` (either pool) is numerically identical to
:func:`~olmo_core.nn.attention.landmark_compressive.fused_compressive_landmark_attention`.

Landmark **queries** (the last ``num_landmarks`` positions of a block) attend their block's content
tokens only -- never any landmark (their own or a sibling's) -- generalizing the single-landmark
"last row attends as position ``block_size - 2``" trick.

This module is opt-in and self-contained: it adds new fused kernels and does not touch the
single-landmark kernels of :mod:`landmark_compressive`, so existing compressive runs are unaffected.
"""

import math
from typing import Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from .landmark_compressive import FastCompressiveLandmarkAttention
from .landmark_fast import _env_int
from .landmark_kernel import _bwd_preprocess, has_landmark_kernel

__all__ = [
    "LANDMARK_GATE_POOLS",
    "multi_compressive_landmark_reference",
    "fused_multi_compressive_landmark_attention",
    "MultiCompressiveLandmarkAttention",
]

# Gate-logit pooling modes. ``"mean"`` -> AGG code 0, ``"max"`` -> AGG code 1 (the constexpr the
# kernels branch on).
LANDMARK_GATE_POOLS = ("mean", "max")
_AGG_MEAN = 0
_AGG_MAX = 1
_POOL_TO_AGG = {"mean": _AGG_MEAN, "max": _AGG_MAX}


try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore


if triton is not None:

    @triton.jit
    def _fwd_kernel_multi_compressive(
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
        DocId,  # int32 (Z, N_BLOCKS) per-block document id, or dummy when DOC_MASK is False
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        N_BLOCKS,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
        DOC_MASK: tl.constexpr,
        NUM_LANDMARKS: tl.constexpr,
        AGG: tl.constexpr,  # 0 = mean, 1 = max
    ):
        # Multi-landmark compressive forward. Generalizes
        # landmark_compressive._fwd_kernel_compressive: the cross-block gate logit of a past block is
        # a mean/max pool over the block's last NUM_LANDMARKS columns (instead of the single last
        # column), and landmark query rows attend their block's content only.
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        if DOC_MASK:
            batch_idx = off_hz // H
            q_doc = tl.load(DocId + batch_idx * N_BLOCKS + (start_m + N_PREFIX_Q))

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)
        # Landmark query rows (last NUM_LANDMARKS of the block) attend their block's content only:
        # force their causal position to the last content token (BLOCK_M - NUM_LANDMARKS - 1). For
        # NUM_LANDMARKS == 1 this is exactly the single-landmark ``-1`` shift on the last row.
        lm_row = tl.arange(0, BLOCK_M) >= (BLOCK_M - NUM_LANDMARKS)
        offs_m_real += tl.where(lm_row, (BLOCK_M - NUM_LANDMARKS - 1) - tl.arange(0, BLOCK_M), 0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        is_lm_col = tl.arange(0, BLOCK_N)[None, :] >= (BLOCK_N - NUM_LANDMARKS)

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
            if DOC_MASK:
                k_doc = tl.load(DocId + batch_idx * N_BLOCKS + start_n)
                qk = tl.where(q_doc == k_doc, qk, -1e30)

            # Pooled cross-block gate logit over the block's last NUM_LANDMARKS columns.
            if AGG == 0:  # mean
                landmark_qk = tl.sum(tl.where(is_lm_col, qk, 0.0), 1) / NUM_LANDMARKS
            else:  # max
                landmark_qk = tl.max(tl.where(is_lm_col, qk, float("-inf")), 1)

            # Compressive within-block softmax over ALL block tokens (content + every landmark).
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

        # Diagonal (local) block: standard causal softmax over content only (landmark columns are
        # masked out via ``offs_m_real``), identical structure to normal landmark.
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
    def _bwd_kv_kernel_multi_compressive(
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
        DocId,
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        N_BLOCKS,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
        DOC_MASK: tl.constexpr,
        NUM_LANDMARKS: tl.constexpr,
        AGG: tl.constexpr,
    ):
        # dk/dv, one program per (key-block, head). Generalizes
        # landmark_compressive._bwd_kv_kernel_compressive: the gate gradient is distributed over the
        # block's landmark columns (mean: 1/NUM_LANDMARKS each; max: the argmax column(s)).
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
        is_lm_col = tl.arange(0, BLOCK_N)[None, :] >= (BLOCK_N - NUM_LANDMARKS)

        start_n = tl.program_id(1) * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)

        if DOC_MASK:
            k_doc = tl.load(DocId + off_z * N_BLOCKS + (start_n // BLOCK_N))

        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            # Diagonal (local) block: standard causal softmax over content only.
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = first_start_m + tl.arange(0, BLOCK_M)
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M
            lm_row = tl.arange(0, BLOCK_M) >= (BLOCK_M - NUM_LANDMARKS)
            offs_m_real += tl.where(
                lm_row, (BLOCK_M - NUM_LANDMARKS - 1) - tl.arange(0, BLOCK_M), 0
            )

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

            doc_keep = 1.0
            if DOC_MASK:
                q_doc = tl.load(DocId + off_z * N_BLOCKS + (start_m // BLOCK_M))
                doc_keep = (q_doc == k_doc).to(tl.float32)

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)

            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale

            if AGG == 0:  # mean
                landmark_qk = tl.sum(tl.where(is_lm_col, qk, 0.0), 1) / NUM_LANDMARKS
            else:  # max
                landmark_qk = tl.max(tl.where(is_lm_col, qk, float("-inf")), 1)

            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_dist = full_p / tl.sum(full_p, 1)[:, None]

            m = tl.load(m_ptrs + offs_m)
            p = tl.exp(landmark_qk - m)  # gate weight (numerator; /L folded into do_scaled)

            do = tl.load(do_ptrs)

            dv += tl.dot(
                tl.trans((doc_keep * p[:, None] * full_dist).to(Q.dtype.element_ty)),
                do,
                allow_tf32=False,
            )

            Di = tl.load(D_ptrs + offs_m)
            dpv = tl.dot(do, tl.trans(v), allow_tf32=False)
            full_D = tl.sum(full_dist * dpv, 1)
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])
            gate_ds = p * (full_D - Di)  # gradient w.r.t. the pooled gate logit
            if AGG == 0:  # mean: split equally over the block's landmark columns
                gate_ds_full = tl.where(is_lm_col, gate_ds[:, None] / NUM_LANDMARKS, 0.0)
            else:  # max: route to the argmax landmark column(s) (torch.amax tie semantics)
                is_max = is_lm_col & (qk == landmark_qk[:, None])
                n_max = tl.sum(is_max.to(tl.float32), 1)
                gate_ds_full = tl.where(is_max, gate_ds[:, None] / n_max[:, None], 0.0)
            ds = within_ds + gate_ds_full
            ds *= sm_scale * doc_keep
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)

        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

    @triton.jit
    def _bwd_q_kernel_multi_compressive(
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
        DocId,
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        N_BLOCKS,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
        DOC_MASK: tl.constexpr,
        NUM_LANDMARKS: tl.constexpr,
        AGG: tl.constexpr,
    ):
        # dq, one program per (query-block, head). Causal-only key-block loop. N_PREFIX_Q == 0 only.
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
        is_lm_col = tl.arange(0, BLOCK_N)[None, :] >= (BLOCK_N - NUM_LANDMARKS)

        start_m = tl.program_id(1) * BLOCK_M
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        do = tl.load(DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        m = tl.load(m_ptrs + offs_m)
        Di = tl.load(D_ptrs + offs_m)

        if DOC_MASK:
            q_doc = tl.load(DocId + off_z * N_BLOCKS + (start_m // BLOCK_M))

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, start_m, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + (offs_n[:, None] * skn + offs_d[None, :] * skd))
            v = tl.load(V + (offs_n[:, None] * svn + offs_d[None, :] * svd))

            doc_keep = 1.0
            if DOC_MASK:
                k_doc = tl.load(DocId + off_z * N_BLOCKS + (start_n // BLOCK_N))
                doc_keep = (q_doc == k_doc).to(tl.float32)

            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            if AGG == 0:  # mean
                landmark_qk = tl.sum(tl.where(is_lm_col, qk, 0.0), 1) / NUM_LANDMARKS
            else:  # max
                landmark_qk = tl.max(tl.where(is_lm_col, qk, float("-inf")), 1)
            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_dist = full_p / tl.sum(full_p, 1)[:, None]
            p = tl.exp(landmark_qk - m)
            dpv = tl.dot(do, tl.trans(v), allow_tf32=False)
            full_D = tl.sum(full_dist * dpv, 1)
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])
            gate_ds = p * (full_D - Di)
            if AGG == 0:  # mean
                gate_ds_full = tl.where(is_lm_col, gate_ds[:, None] / NUM_LANDMARKS, 0.0)
            else:  # max
                is_max = is_lm_col & (qk == landmark_qk[:, None])
                n_max = tl.sum(is_max.to(tl.float32), 1)
                gate_ds_full = tl.where(is_max, gate_ds[:, None] / n_max[:, None], 0.0)
            ds = within_ds + gate_ds_full
            ds *= sm_scale * doc_keep
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)

        # diagonal key block: within-block causal attention over content only.
        offs_n = start_m + tl.arange(0, BLOCK_N)
        k = tl.load(K + (offs_n[:, None] * skn + offs_d[None, :] * skd))
        v = tl.load(V + (offs_n[:, None] * svn + offs_d[None, :] * svd))
        offs_m_real = offs_m + tl.where(
            tl.arange(0, BLOCK_M) >= (BLOCK_M - NUM_LANDMARKS),
            (BLOCK_M - NUM_LANDMARKS - 1) - tl.arange(0, BLOCK_M),
            0,
        )
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk = tl.where(offs_m_real[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        last_p = tl.exp(qk * sm_scale - m[:, None])
        last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = last_p * last_dp * sm_scale
        dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)

        tl.store(DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd), dq)


class _FusedMultiCompressiveLandmarkAttention(torch.autograd.Function):
    """Fused multi-landmark compressive attention (forward + FA2-style backward)."""

    @staticmethod
    def forward(ctx, q, k, v, n_prefix_q, sm_scale, block_size, num_landmarks, agg, doc_id=None):
        if triton is None:
            raise RuntimeError("Landmark attention requires 'triton' (and a CUDA device).")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch, nheads, seqlen_q, d = q.shape
        assert d <= 256 and q.dtype == k.dtype == v.dtype and q.is_cuda

        BLOCK = block_size
        n_blocks = k.shape[2] // BLOCK
        doc_mask = doc_id is not None
        if doc_mask:
            assert doc_id.shape == (batch, n_blocks), (doc_id.shape, (batch, n_blocks))
            doc_id = doc_id.to(device=q.device, dtype=torch.int32).contiguous()
        doc_id_arg = doc_id if doc_mask else torch.empty(1, dtype=torch.int32, device=q.device)
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
        _fwd_kernel_multi_compressive[grid](
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
            doc_id_arg,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            n_blocks,
            BLOCK=BLOCK,
            BLOCK_DMODEL=d,
            N_PREFIX_Q=n_prefix_q,
            DOC_MASK=doc_mask,
            NUM_LANDMARKS=num_landmarks,
            AGG=agg,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.doc_id = doc_id  # None when not packing
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = d
        ctx.N_PREFIX_Q = n_prefix_q
        ctx.BLOCK = BLOCK
        ctx.num_landmarks = num_landmarks
        ctx.agg = agg
        return o

    @staticmethod
    def backward(ctx, do):
        if ctx.N_PREFIX_Q != 0:
            raise NotImplementedError(
                "MultiCompressiveLandmarkAttention backward only supports no history KV "
                "(N_PREFIX_Q == 0); generation runs without gradients."
            )

        BLOCK = ctx.BLOCK
        q, k, v, o, lse, m = ctx.saved_tensors
        doc_id = ctx.doc_id
        doc_mask = doc_id is not None
        n_blocks = k.shape[2] // BLOCK
        doc_id_arg = doc_id if doc_mask else torch.empty(1, dtype=torch.int32, device=q.device)
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
        dims = (q.shape[0], q.shape[1], q.shape[2], k.shape[2], n_blocks)
        const = dict(
            BLOCK=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            N_PREFIX_Q=ctx.N_PREFIX_Q,
            DOC_MASK=doc_mask,
            NUM_LANDMARKS=ctx.num_landmarks,
            AGG=ctx.agg,
        )
        if ctx.BLOCK_DMODEL > 128:
            warps = _env_int("LM_FAST_WARPS", 8)
            stages = _env_int("LM_FAST_STAGES", 1)
        else:
            warps = _env_int("LM_FAST_WARPS", 4)
            stages = _env_int("LM_FAST_STAGES", 2)
        n_kv_blocks = triton.cdiv(k.shape[2], BLOCK)
        stride_args = (
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
        )
        _bwd_kv_kernel_multi_compressive[(ctx.grid[1], n_kv_blocks)](
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
            *stride_args,
            doc_id_arg,
            *dims,
            **const,
            num_warps=warps,
            num_stages=stages,
        )
        _bwd_q_kernel_multi_compressive[(ctx.grid[1], ctx.grid[0])](
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
            *stride_args,
            doc_id_arg,
            *dims,
            **const,
            num_warps=warps,
            num_stages=stages,
        )
        return dq, dk, dv, None, None, None, None, None, None


def fused_multi_compressive_landmark_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_mem: torch.Tensor,
    sm_scale: float = None,  # type: ignore[assignment]
    block_size: int = 32,
    num_landmarks: int = 1,
    agg: str = "mean",
    doc_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused multi-landmark compressive landmark attention.

    Generalizes
    :func:`~olmo_core.nn.attention.landmark_compressive.fused_compressive_landmark_attention` to
    ``num_landmarks`` landmark tokens per block (the last ``num_landmarks`` positions), with the
    block gate formed by ``agg`` (``"mean"`` or ``"max"``) over those landmarks' scores.

    :param is_mem: Boolean ``(T,)`` mask; must mark the last ``num_landmarks`` positions of every
        ``block_size``-token block.
    :param agg: Gate-logit pool, one of :data:`LANDMARK_GATE_POOLS`.
    """
    if agg not in _POOL_TO_AGG:
        raise ValueError(f"Unknown gate pool {agg!r}; expected one of {LANDMARK_GATE_POOLS}")
    expected_is_mem = (torch.arange(0, is_mem.shape[-1], device=is_mem.device) % block_size) >= (
        block_size - num_landmarks
    )
    assert (is_mem == expected_is_mem).all()
    n_history_kv = k.shape[-2] - q.shape[-2]
    assert n_history_kv % block_size == 0
    n_history_blocks = n_history_kv // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return _FusedMultiCompressiveLandmarkAttention.apply(
        q, k, v, n_history_blocks, sm_scale, block_size, num_landmarks, _POOL_TO_AGG[agg], doc_id
    )


def multi_compressive_landmark_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    num_landmarks: int,
    agg: str,
    doc_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense eager reference for multi-landmark compressive attention over ``(B, H, T, d)`` (causal).

    Each past block's gate weight comes from a mean/max pool over the block's ``num_landmarks``
    landmark scores; that weight is spread over the block's content tokens AND every landmark via a
    within-block softmax over all ``block_size`` tokens. The local section is plain causal attention
    over content and never attends its own block's landmarks. ``num_landmarks == 1`` reproduces the
    single-landmark compressive reference exactly.

    :param agg: ``"mean"`` or ``"max"`` (see :data:`LANDMARK_GATE_POOLS`).
    :param doc_id: Optional int32 ``(B, n_blocks)`` per-block document id for packing; cross-document
        blocks receive zero gate weight.
    """
    if agg not in _POOL_TO_AGG:
        raise ValueError(f"Unknown gate pool {agg!r}; expected one of {LANDMARK_GATE_POOLS}")
    B, H, T, d = q.shape
    device = q.device
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-1, -2)).float() * scale  # (B, H, T, T)
    neg_inf = torch.finfo(scores.dtype).min
    n_blocks = T // block_size

    pos = torch.arange(T, device=device)
    sec = pos // block_size
    is_mem = (pos % block_size) >= (block_size - num_landmarks)  # (T,) all landmark cols
    rep_col = (pos % block_size) == (block_size - 1)  # (T,) block representative (last col)
    causal = pos[None, :] <= pos[:, None]
    same_block = sec[None, :] == sec[:, None]
    past_block = sec[None, :] < sec[:, None]
    kmem = is_mem[None, :]

    if doc_id is not None:
        tok_doc = doc_id[:, sec.to(torch.long)]  # (B, T)
        same_doc = (tok_doc[:, :, None] == tok_doc[:, None, :]).view(B, 1, T, T)
    else:
        same_doc = torch.ones(1, 1, T, T, dtype=torch.bool, device=device)

    local_content = (same_block & (~kmem) & causal).view(1, 1, T, T)
    past_rep = (past_block & rep_col[None, :]).view(1, 1, T, T)

    # Pooled per-block gate logit, placed at each block's representative column.
    lm_scores = scores.reshape(B, H, T, n_blocks, block_size)[..., block_size - num_landmarks :]
    if agg == "mean":
        g_b = lm_scores.mean(dim=-1)  # (B, H, T, n_blocks)
    else:
        g_b = lm_scores.amax(dim=-1)
    g_full = g_b.repeat_interleave(block_size, dim=-1)  # (B, H, T, T)
    gate_logits = torch.where(rep_col.view(1, 1, 1, T), g_full, scores)

    gate_set = (local_content | past_rep) & same_doc  # (B, 1, T, T)
    gate_w = torch.softmax(gate_logits.masked_fill(~gate_set, neg_inf), dim=-1)  # (B, H, T, T)

    # Full within-block softmax over every block of keys (only past-block entries are used).
    within = torch.softmax(scores.reshape(B, H, T, n_blocks, block_size), dim=-1)
    within = within.reshape(B, H, T, T)

    block_gate = gate_w[..., rep_col]  # (B, H, T, n_blocks): gate weight at each block's rep col
    block_gate_full = block_gate.repeat_interleave(block_size, dim=-1)  # (B, H, T, T)

    past_mask = past_block.view(1, 1, T, T)
    local_mask = local_content
    final = torch.where(past_mask, block_gate_full * within, torch.zeros_like(within))
    final = torch.where(local_mask, gate_w, final)
    return final.to(v.dtype) @ v


class MultiCompressiveLandmarkAttention(FastCompressiveLandmarkAttention):
    """
    Multi-landmark compressive attention (``AttentionType.multi_compressive_landmark``).

    Like :class:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention` but
    each block carries ``num_landmarks`` landmark tokens (its last ``num_landmarks`` positions), and
    the block's single cross-block gate weight is a ``landmark_gate_pool`` (``"mean"`` or ``"max"``)
    pool over its landmark scores. See the module docstring for the math and layout.

    :param num_landmarks: Number of landmark tokens per block. ``block_size = mem_freq +
        num_landmarks`` must be a power of two >= 16 (the fused kernel tiles by ``block_size``).
    :param landmark_gate_pool: How to pool a block's landmark scores into its gate logit: ``"mean"``
        (average affinity, all landmarks trained) or ``"max"`` (best-matching landmark). See
        :data:`LANDMARK_GATE_POOLS`.
    """

    _supports_ragged_decode: bool = False

    def __init__(
        self,
        *,
        mem_freq: int,
        num_landmarks: int = 1,
        landmark_gate_pool: str = "mean",
        nonselected_landmark_mass: float = 0.1,
        softmax_scale: Optional[float] = None,
        group_landmark_selection: Optional[str] = None,
        **kwargs,
    ):
        if kwargs.pop("gate_temperature", False):
            raise OLMoConfigurationError(
                "gate_temperature is not yet supported with multi_compressive_landmark attention."
            )
        super().__init__(
            mem_freq=mem_freq,
            nonselected_landmark_mass=nonselected_landmark_mass,
            softmax_scale=softmax_scale,
            group_landmark_selection=group_landmark_selection,
            gate_temperature=False,
            **kwargs,
        )
        if num_landmarks < 1:
            raise OLMoConfigurationError(f"num_landmarks must be >= 1 (got {num_landmarks}).")
        if landmark_gate_pool not in LANDMARK_GATE_POOLS:
            raise OLMoConfigurationError(
                f"landmark_gate_pool must be one of {LANDMARK_GATE_POOLS} (got {landmark_gate_pool!r})."
            )
        self.num_landmarks = num_landmarks
        self.landmark_gate_pool = landmark_gate_pool
        self._agg = _POOL_TO_AGG[landmark_gate_pool]
        # Override the parent's block_size (mem_freq + 1). The fused kernel tiles by block_size, so it
        # must be a power of two >= 16 (tl.arange / tl.dot need power-of-two tile dims).
        self.block_size = mem_freq + num_landmarks
        if self.block_size < 16 or (self.block_size & (self.block_size - 1)) != 0:
            raise OLMoConfigurationError(
                "block_size = mem_freq + num_landmarks must be a power of two >= 16 for the fused "
                f"kernel (got mem_freq={mem_freq}, num_landmarks={num_landmarks}, "
                f"block_size={self.block_size})."
            )

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not has_landmark_kernel():
            raise RuntimeError(
                "MultiCompressiveLandmarkAttention requires the fused Triton kernel "
                "(install 'triton', run on CUDA)."
            )
        T = q.shape[2]
        is_mem = (torch.arange(T, device=q.device) % self.block_size) >= (
            self.block_size - self.num_landmarks
        )
        return fused_multi_compressive_landmark_attention(
            q,
            k,
            v,
            is_mem,
            sm_scale=self.softmax_scale,
            block_size=self.block_size,
            num_landmarks=self.num_landmarks,
            agg=self.landmark_gate_pool,
            doc_id=doc_id,
        )

    def _compressive_decode_probs(
        self,
        scores: torch.Tensor,
        is_mem: torch.Tensor,
        last_section: torch.Tensor,
        section_start: int,
        gate_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-landmark compressive grouped softmax for a single decode query.

        Generalizes
        :meth:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention._compressive_decode_probs`:
        each past block's gate weight is a mean/max pool over its ``num_landmarks`` landmark scores.
        ``num_landmarks == 1`` reproduces the single-landmark decode. The passed ``is_mem`` (built
        single-landmark by the caller) is ignored -- the landmark columns are recomputed here from
        ``block_size``/``num_landmarks``. ``gate_scores`` is unused (this variant has no gate
        temperature) and must be ``None``.
        """
        if gate_scores is not None:
            raise NotImplementedError(
                "multi_compressive_landmark decode does not support a separate gate query "
                "(gate_temperature is not supported)."
            )
        if getattr(self, "_eval_flat_softmax", False):
            raise NotImplementedError(
                "the flat-softmax decode ablation is not implemented for multi_compressive_landmark."
            )
        B, H, _, total = scores.shape
        Lb = self.block_size
        Lm = self.num_landmarks
        S = section_start
        device = scores.device
        neg_inf = torch.finfo(scores.dtype).min

        last_section_b = last_section.view(1, 1, 1, total)
        n_blocks = S // Lb

        final = torch.zeros(B, H, 1, total, dtype=scores.dtype, device=device)
        if n_blocks == 0:
            # No past blocks: plain softmax over the local section.
            return torch.softmax(scores.masked_fill(~last_section_b, neg_inf), dim=-1)

        rep_pos = torch.arange(Lb - 1, S, Lb, device=device)  # (n_blocks,) block representative col
        past = scores[..., :S].reshape(B, H, 1, n_blocks, Lb)
        lm_scores = past[..., Lb - Lm :]  # (B, H, 1, n_blocks, Lm)
        if self.landmark_gate_pool == "mean":
            g_b = lm_scores.mean(dim=-1)  # (B, H, 1, n_blocks)
        else:
            g_b = lm_scores.amax(dim=-1)

        top_k = self._eval_top_k
        if top_k is not None and n_blocks > top_k:
            keep = torch.zeros_like(g_b, dtype=torch.bool)
            keep.scatter_(-1, g_b.topk(top_k, dim=-1).indices, True)
            has_nonselected = True
            alpha = float(self.nonselected_landmark_mass)
        else:
            keep = torch.ones(B, H, 1, n_blocks, dtype=torch.bool, device=device)
            has_nonselected = False
            alpha = 0.0

        # Cross-block gate softmax over selected block representatives + the local section. The rep
        # column of each past block carries that block's pooled logit ``g_b``.
        gate_logits = scores.clone()
        gate_logits[..., rep_pos] = g_b
        sel_full = last_section_b.expand(B, H, 1, total).clone()
        sel_rep = torch.zeros(B, H, 1, total, dtype=torch.bool, device=device)
        sel_rep[..., rep_pos] = keep
        gate_participant = sel_full | sel_rep
        gate_w = torch.softmax(gate_logits.masked_fill(~gate_participant, neg_inf), dim=-1)

        # Local section keeps its gate weight directly.
        final = torch.where(last_section_b, gate_w, final)
        # Past blocks: full within-block softmax over content + every landmark, scaled by the block
        # gate (non-selected blocks have gate weight 0 here).
        within = torch.softmax(past, dim=-1).reshape(B, H, 1, S)
        block_gate = gate_w[..., rep_pos]  # (B, H, 1, n_blocks)
        block_gate_full = block_gate.repeat_interleave(Lb, dim=-1)  # (B, H, 1, S)
        final = final.clone()
        final[..., :S] = block_gate_full * within

        if has_nonselected:
            # Reserve ``alpha`` of the mass for the non-selected blocks' landmark tokens, split by a
            # joint softmax over their landmark scores (reduces to the single-landmark behavior when
            # num_landmarks == 1).
            final = final * (1.0 - alpha)
            block_keep_per_pos = keep[..., (torch.arange(S, device=device) // Lb)]  # (B,H,1,S)
            lm_col = (torch.arange(S, device=device) % Lb) >= (Lb - Lm)  # (S,)
            nonsel_lm = lm_col.view(1, 1, 1, S) & (~block_keep_per_pos)  # (B,H,1,S)
            nonsel_full = torch.zeros(B, H, 1, total, dtype=torch.bool, device=device)
            nonsel_full[..., :S] = nonsel_lm
            ns_w = torch.softmax(scores.masked_fill(~nonsel_full, neg_inf), dim=-1)
            final = final + alpha * ns_w

        return final
