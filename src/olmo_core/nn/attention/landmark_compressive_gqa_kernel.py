"""Fused Triton kernel for :class:`CompressiveGQAGroupedAttention` -- the *gate-only* GQA-grouped
compressive landmark attention.

Adapted one-to-one from the compressive kernels in :mod:`landmark_compressive`. The ONLY change is the
source of the cross-block **gate** logit: instead of each head's own ``landmark_qk = q_h . k_landmark``,
the gate uses a second, group-averaged query ``q_gate`` (``= mean over the KV group's heads of q``), so
``landmark_qk = q_gate . k_landmark``. The within-block softmax (content weighting) and the local
(diagonal) section still use the per-head query ``q``.

Because the gate logit now comes from ``q_gate`` while the content/local logits come from ``q``, the
score gradient splits by source:

* ``within_ds`` (all block columns) -> flows to ``q`` (``dq``) and to ``dk`` via ``q``;
* ``gate_ds`` (the landmark column only) -> flows to ``q_gate`` (``dq_gate``) and to ``dk`` via
  ``q_gate``;
* the diagonal/local block -> flows to ``q`` only (no gate there).

``q_gate`` is built OUTSIDE this Function with autograd-tracked ops (``q_gate = mean(q).expand()``), so
returning ``dq_gate`` separately lets autograd distribute the gate gradient back across the group (each
of the ``n_rep`` heads gets ``1/n_rep`` of its block's gate gradient). See
:mod:`landmark_compressive_gqa` for the module + math.
"""

import math
from typing import Optional

import torch

from .landmark_compressive import _env_int
from .landmark_kernel import _bwd_preprocess

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore


if triton is not None:

    @triton.jit
    def _fwd_kernel_gqa_grouped(
        Q,
        QG,  # group-mean query, used ONLY for the cross-block gate landmark logit
        K,
        V,
        sm_scale,
        Out,
        sqz,
        sqh,
        sqm,
        sqd,
        sgz,
        sgh,
        sgm,
        sgd,
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
    ):
        # Mirrors _fwd_kernel_compressive; the only change is that ``landmark_qk`` (the block gate
        # logit) is computed from QG (group-mean query) instead of Q.
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        if DOC_MASK:
            batch_idx = off_hz // H
            q_doc = tl.load(DocId + batch_idx * N_BLOCKS + (start_m + N_PREFIX_Q))

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        offs_qg = off_hz * sgh + offs_m[:, None] * sgm + offs_d[None, :] * sgd
        offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd
        offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd

        m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)
        qg_vals = tl.load(QG + offs_qg, mask=offs_m[:, None] < N_CTX_Q, other=0)

        for start_n in range(0, (N_PREFIX_Q + start_m)):
            k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)

            # Per-head content logits (for the within-block softmax).
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
            qk += tl.dot(q_vals, k_vals, allow_tf32=False)
            qk *= sm_scale
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float("-inf"))
            if DOC_MASK:
                k_doc = tl.load(DocId + batch_idx * N_BLOCKS + start_n)
                qk = tl.where(q_doc == k_doc, qk, -1e30)

            # Group-mean gate logits (SAME masking) -> the block gate comes from the landmark column.
            qk_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=qg_vals.dtype)
            qk_gate += tl.dot(qg_vals, k_vals, allow_tf32=False)
            qk_gate *= sm_scale
            qk_gate = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk_gate, float("-inf"))
            if DOC_MASK:
                qk_gate = tl.where(q_doc == k_doc, qk_gate, -1e30)

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk_gate, float("-inf")), 1
            )
            # Compressive within-block softmax over ALL block tokens (content + landmark), per-head.
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

        # Diagonal (local) block: standard causal softmax on the PER-HEAD query (no gate here).
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
    def _bwd_kv_kernel_gqa_grouped(
        Q,
        QG,
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
        sgz,
        sgh,
        sgm,
        sgd,
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
    ):
        # dk/dv, one program per (key-block, head). Mirrors _bwd_kv_kernel_compressive; the gate
        # gradient (landmark column) flows to dk via QG (group-mean query), the within-block gradient
        # via Q, and the diagonal via Q.
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        Q += off_z * sqz + off_h * sqh
        QG += off_z * sgz + off_h * sgh
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
            # Diagonal (local) block: standard causal softmax on the per-head query (grad -> dk via Q).
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

            doc_keep = 1.0
            if DOC_MASK:
                q_doc = tl.load(DocId + off_z * N_BLOCKS + (start_m // BLOCK_M))
                doc_keep = (q_doc == k_doc).to(tl.float32)

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            qg_ptrs = QG + (offs_m[:, None] * sgm + offs_d[None, :] * sgd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)

            q = tl.load(q_ptrs)
            qg = tl.load(qg_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            qk_gate = tl.dot(qg, tl.trans(k), allow_tf32=False)
            qk_gate *= sm_scale

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk_gate, float("-inf")), 1
            )
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
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])  # all cols -> via Q
            gate_ds = p * (full_D - Di)  # landmark column -> via QG
            within_ds_s = (within_ds * (sm_scale * doc_keep)).to(Q.dtype.element_ty)
            gate_ds_s = (
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, gate_ds[:, None], 0.0)
                * (sm_scale * doc_keep)
            ).to(Q.dtype.element_ty)
            # dk gets the within-block gradient contracted with the per-head query, and the gate
            # gradient (landmark column) contracted with the group-mean query.
            dk += tl.dot(tl.trans(within_ds_s), q, allow_tf32=False)
            dk += tl.dot(tl.trans(gate_ds_s), qg, allow_tf32=False)

        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

    @triton.jit
    def _bwd_q_kernel_gqa_grouped(
        Q,
        QG,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DQG,  # gradient w.r.t. the group-mean query (gate path)
        DK,
        DV,
        L,
        M,
        D,
        sqz,
        sqh,
        sqm,
        sqd,
        sgz,
        sgh,
        sgm,
        sgd,
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
    ):
        # dq (per-head, content + local) and dqg (group-mean query, gate). One program per
        # (query-block, head). N_PREFIX_Q == 0 only (caller guards).
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        Q += off_z * sqz + off_h * sqh
        QG += off_z * sgz + off_h * sgh
        K += off_z * skz + off_h * skh
        V += off_z * svz + off_h * svh
        DO += off_z * sqz + off_h * sqh
        DQ += off_z * sqz + off_h * sqh
        DQG += off_z * sgz + off_h * sgh

        offs_d = tl.arange(0, BLOCK_DMODEL)
        D_ptrs = D + off_hz * N_CTX_Q
        m_ptrs = M + off_hz * N_CTX_Q

        start_m = tl.program_id(1) * BLOCK_M
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        qg = tl.load(QG + (offs_m[:, None] * sgm + offs_d[None, :] * sgd))
        do = tl.load(DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        m = tl.load(m_ptrs + offs_m)
        Di = tl.load(D_ptrs + offs_m)

        if DOC_MASK:
            q_doc = tl.load(DocId + off_z * N_BLOCKS + (start_m // BLOCK_M))

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dqg = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

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
            qk_gate = tl.dot(qg, tl.trans(k), allow_tf32=False)
            qk_gate *= sm_scale
            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk_gate, float("-inf")), 1
            )
            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_dist = full_p / tl.sum(full_p, 1)[:, None]
            p = tl.exp(landmark_qk - m)
            dpv = tl.dot(do, tl.trans(v), allow_tf32=False)
            full_D = tl.sum(full_dist * dpv, 1)
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])  # -> dq (per-head)
            gate_ds = p * (full_D - Di)  # landmark column -> dqg (group-mean query)
            within_ds_s = (within_ds * (sm_scale * doc_keep)).to(Q.dtype.element_ty)
            gate_ds_s = (
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, gate_ds[:, None], 0.0)
                * (sm_scale * doc_keep)
            ).to(Q.dtype.element_ty)
            dq += tl.dot(within_ds_s, k, allow_tf32=False)
            dqg += tl.dot(gate_ds_s, k, allow_tf32=False)

        # diagonal key block: within-block causal attention on the per-head query (grad -> dq only).
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
        tl.store(DQG + (offs_m[:, None] * sgm + offs_d[None, :] * sgd), dqg)


class _FusedCompressiveGQAGrouped(torch.autograd.Function):
    """Fused gate-only GQA-grouped compressive landmark attention. ``q_gate`` is a separate input (the
    group-mean query); the gate landmark logit uses it while content/local use ``q``. Backward returns
    ``dq`` (content/local) and ``dq_gate`` (gate) separately so autograd distributes the gate gradient
    back across the group's heads."""

    @staticmethod
    def forward(ctx, q, q_gate, k, v, n_prefix_q, sm_scale, block_size, doc_id=None):
        if triton is None:
            raise RuntimeError("Landmark attention requires 'triton' (and a CUDA device).")
        q = q.contiguous()
        q_gate = q_gate.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch, nheads, seqlen_q, d = q.shape
        assert d <= 256 and q.dtype == k.dtype == v.dtype == q_gate.dtype and q.is_cuda
        assert q_gate.shape == q.shape

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
        _fwd_kernel_gqa_grouped[grid](
            q,
            q_gate,
            k,
            v,
            sm_scale,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q_gate.stride(0),
            q_gate.stride(1),
            q_gate.stride(2),
            q_gate.stride(3),
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
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, q_gate, k, v, o, L, m)
        ctx.doc_id = doc_id
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
                "CompressiveGQAGroupedAttention backward only supports no history KV "
                "(N_PREFIX_Q == 0); generation runs without gradients."
            )

        BLOCK = ctx.BLOCK
        q, q_gate, k, v, o, lse, m = ctx.saved_tensors
        doc_id = ctx.doc_id
        doc_mask = doc_id is not None
        n_blocks = k.shape[2] // BLOCK
        doc_id_arg = doc_id if doc_mask else torch.empty(1, dtype=torch.int32, device=q.device)
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dqg = torch.zeros_like(q_gate, dtype=torch.float32)
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
        stride_args = (
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q_gate.stride(0),
            q_gate.stride(1),
            q_gate.stride(2),
            q_gate.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )
        dims = (q.shape[0], q.shape[1], q.shape[2], k.shape[2], n_blocks)
        const = dict(
            BLOCK=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            N_PREFIX_Q=ctx.N_PREFIX_Q,
            DOC_MASK=doc_mask,
        )
        if ctx.BLOCK_DMODEL > 128:
            warps = _env_int("LM_FAST_WARPS", 8)
            stages = _env_int("LM_FAST_STAGES", 1)
        else:
            warps = _env_int("LM_FAST_WARPS", 4)
            stages = _env_int("LM_FAST_STAGES", 2)
        n_kv_blocks = triton.cdiv(k.shape[2], BLOCK)
        _bwd_kv_kernel_gqa_grouped[(ctx.grid[1], n_kv_blocks)](
            q,
            q_gate,
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
        _bwd_q_kernel_gqa_grouped[(ctx.grid[1], ctx.grid[0])](
            q,
            q_gate,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dqg,
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
        return dq, dqg, dk, dv, None, None, None, None


def fused_compressive_gqa_grouped_attention(
    q: torch.Tensor,
    q_gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_mem: torch.Tensor,
    sm_scale: float = None,  # type: ignore[assignment]
    block_size: int = 64,
    doc_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gate-only GQA-grouped compressive-landmark attention. ``q`` is the per-head query (content +
    local); ``q_gate`` is the group-mean query used for the cross-block gate landmark logit. Same
    ``doc_id`` packing semantics as :func:`fused_compressive_landmark_attention`."""
    expected_is_mem = torch.arange(0, is_mem.shape[-1], device=is_mem.device) % block_size == (
        block_size - 1
    )
    assert (is_mem == expected_is_mem).all()
    n_history_kv = k.shape[-2] - q.shape[-2]
    assert n_history_kv % block_size == 0
    n_history_blocks = n_history_kv // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return _FusedCompressiveGQAGrouped.apply(
        q, q_gate, k, v, n_history_blocks, sm_scale, block_size, doc_id
    )
