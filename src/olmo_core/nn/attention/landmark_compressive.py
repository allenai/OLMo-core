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

**GQA-aware top-k (`group_landmark_selection`).** Under GQA (``n_kv_heads < n_heads``), ``repeat_kv``
duplicates each KV group's K/V across its ``n_rep`` query heads *before* any landmark scoring runs, so
by default the top-k retrieval above is computed independently per (duplicated) query head: two heads
in the same group see identical landmark keys but, having different queries, can retrieve different
blocks. That defeats the point of GQA at decode time -- the whole reason to share KV across a group is
to share the memory traffic of reading it, and independent per-head retrieval means decode still has to
touch the union of every group member's chosen blocks. ``group_landmark_selection`` (``"mean"`` or
``"max"`` over the group's per-head scores, see :meth:`FastCompressiveLandmarkAttention._group_landmark_scores`)
makes every head in a group agree on the same top-k block set, restoring that saving. Only the
*selection* is shared -- the gate softmax and within-block softmax that weight the output still use
each head's own real scores, so the per-head attention output itself is unchanged for whichever blocks
end up selected. Like ``top_k`` itself, this only affects hard top-k decode; training/prefill are
unaffected (see ``analysis/group_landmark_selection/DESIGN.md`` for the full rationale and the
alternative aggregation methods considered).
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError

from . import landmark_gate_analysis as gate_log
from .landmark_fast import FastLandmarkAttention, _env_int
from .landmark_kernel import _bwd_preprocess, has_landmark_kernel

# Valid values for ``group_landmark_selection`` (``None`` means "off", i.e. today's per-head
# independent selection). A private sentinel (distinct from ``None``) is needed for the
# ``set_landmark_eval_decode`` override parameter, since ``None`` is itself a meaningful value
# there ("force selection back off for this eval run"), not just "leave unset".
#
# ``"inverse_mean"`` is a deliberate *anti-selection* sanity check, not a real method: it shares one
# top-k across the group like ``"mean"``, but keeps the group's LEAST-attended blocks (bottom-k of the
# mean score) instead of the most-attended. It exists to bound how much retrieval quality matters on a
# task -- if metrics barely move when we force attention onto the worst blocks, the task is robust to
# which blocks are retrieved (so a small mean-vs-none gap is expected); if metrics collapse, selection
# genuinely matters. Never use it in production.
_GROUP_LANDMARK_SELECTIONS = (None, "mean", "max", "inverse_mean")
_NO_OVERRIDE = object()

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
        QG,  # gate-only query (defaults to Q itself for the no-temperature/backward-compat case)
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
        DocId,  # int32 (Z, N_BLOCKS) per-block document id, or dummy when DOC_MASK is False
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        N_BLOCKS,  # number of landmark blocks in the key sequence (= N_CTX_KV / BLOCK)
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
        DOC_MASK: tl.constexpr,  # whether to apply intra-document (packing) masking
    ):
        # Compressive landmark forward. Identical to landmark_kernel._fwd_kernel except that the
        # value contribution of a fully-past block uses the *full-block* within softmax (over all
        # BLOCK_N tokens including the landmark) instead of the content-only ``normal_p``. The gate
        # (cross-block) softmax that produces L/M is untouched, so L/M stay bit-identical to the
        # normal landmark kernel. The gate (cross-block) landmark logit is computed from ``QG``
        # rather than ``Q`` -- when ``QG`` is ``Q`` itself (the no-temperature default) this is
        # bit-identical to reading it off ``qk`` directly.
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        # Intra-document masking (sequence packing): each landmark block belongs to exactly one
        # document (boundaries are block-aligned), so cross-document key blocks are floored to a
        # finite large-negative value (see landmark_kernel._fwd_kernel for why the floor is finite,
        # not -inf). Only the grouping loop over strictly-past key blocks needs the gate; the
        # diagonal (own) block below is always same-document.
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

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
            qk += tl.dot(q_vals, k_vals, allow_tf32=False)
            qk *= sm_scale
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float("-inf"))
            if DOC_MASK:
                k_doc = tl.load(DocId + batch_idx * N_BLOCKS + start_n)
                qk = tl.where(q_doc == k_doc, qk, -1e30)

            # Gate-only logits (SAME masking) -> the block gate comes from the landmark column of
            # this, not of ``qk``.
            qk_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=qg_vals.dtype)
            qk_gate += tl.dot(qg_vals, k_vals, allow_tf32=False)
            qk_gate *= sm_scale
            qk_gate = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk_gate, float("-inf"))
            if DOC_MASK:
                qk_gate = tl.where(q_doc == k_doc, qk_gate, -1e30)

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk_gate, float("-inf")), 1
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
    ):
        # dk/dv, one program per (key-block, head); atomic-free. Mirrors landmark_fast._bwd_kv_kernel
        # but with the compressive within-block softmax: every block token (incl. the landmark) gets
        # a within-block value weight, and the landmark score additionally carries the gate gradient
        # (via ``QG``, which is ``Q`` itself in the no-temperature default).
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

        # Document id of this key block (for intra-document / packing masking). Only the landmark-
        # grouping loop over strictly-future query blocks needs the cross-document gate; the
        # diagonal (own-block) contribution below is always same-document.
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

            # Cross-document query blocks received zero weight on this key block in the forward, so
            # they get zero gradient here. ``doc_keep`` is 1.0 for same-document, 0.0 otherwise.
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
            # Compressive within-block softmax over all block tokens (content + landmark).
            full_m = tl.max(qk, 1)
            full_p = tl.exp(qk - full_m[:, None])
            full_dist = full_p / tl.sum(full_p, 1)[:, None]

            m = tl.load(m_ptrs + offs_m)
            p = tl.exp(landmark_qk - m)  # gate weight (numerator; /L folded into do_scaled)

            do = tl.load(do_ptrs)

            # dv: every token (incl. landmark) gets within-block weight p * full_dist. Cross-document
            # query blocks are zeroed via ``doc_keep`` (they contributed nothing in the forward).
            dv += tl.dot(
                tl.trans((doc_keep * p[:, None] * full_dist).to(Q.dtype.element_ty)),
                do,
                allow_tf32=False,
            )

            Di = tl.load(D_ptrs + offs_m)
            dpv = tl.dot(do, tl.trans(v), allow_tf32=False)  # do_scaled . v_n per col
            # full_D = do_scaled . fv = sum_n full_dist_n * (do_scaled . v_n). Computing it from dpv
            # avoids a separate ``dot(full_dist, v)`` and its (BLOCK_M, head_dim) accumulator, which
            # keeps the head_dim=256 / block=64 backward within A100 shared memory.
            full_D = tl.sum(full_dist * dpv, 1)
            # within_ds (all columns) -> dk via Q; gate_ds (landmark column only) -> dk via QG.
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])
            gate_ds = p * (full_D - Di)
            within_ds_s = (within_ds * (sm_scale * doc_keep)).to(Q.dtype.element_ty)
            gate_ds_s = (
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, gate_ds[:, None], 0.0)
                * (sm_scale * doc_keep)
            ).to(Q.dtype.element_ty)
            dk += tl.dot(tl.trans(within_ds_s), q, allow_tf32=False)
            dk += tl.dot(tl.trans(gate_ds_s), qg, allow_tf32=False)

        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

    @triton.jit
    def _bwd_q_kernel_compressive(
        Q,
        QG,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DQG,  # gradient w.r.t. the gate-only query (== DQ's tensor when QG is Q itself)
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
    ):
        # dq (per-head, content + local) and dqg (gate-only query). One program per (query-block,
        # head). Causal-only key-block loop, atomic-free. Only implemented for N_PREFIX_Q == 0 (the
        # caller guards this).
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

            # Cross-document prior key blocks contributed nothing in the forward -> zero gradient.
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
            # full_D = do_scaled . fv = sum_n full_dist_n * (do_scaled . v_n); see _bwd_kv_kernel.
            full_D = tl.sum(full_dist * dpv, 1)
            # within_ds (all columns) -> dq (per-head query); gate_ds (landmark column) -> dqg.
            within_ds = p[:, None] * full_dist * (dpv - full_D[:, None])
            gate_ds = p * (full_D - Di)
            within_ds_s = (within_ds * (sm_scale * doc_keep)).to(Q.dtype.element_ty)
            gate_ds_s = (
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, gate_ds[:, None], 0.0)
                * (sm_scale * doc_keep)
            ).to(Q.dtype.element_ty)
            dq += tl.dot(within_ds_s, k, allow_tf32=False)
            dqg += tl.dot(gate_ds_s, k, allow_tf32=False)

        # diagonal key block: within-block causal attention (identical to normal landmark; no gate
        # here, so it only contributes to dq).
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


class _FusedCompressiveLandmarkAttention(torch.autograd.Function):
    """Fused compressive landmark attention (forward + FA2-style backward). The landmark token of
    each past block is included in that block's within-block softmax, so it contributes its value to
    the output. ``L``/``M`` (and hence the gate softmax) are identical to the normal landmark
    kernel; only the value accumulation and its gradient differ."""

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
        # ``doc_id`` is an int32 (batch, n_blocks) tensor of per-block document ids for sequence
        # packing, or None for the single-document path. Triton still needs a real pointer argument
        # when masking is off, so we pass a 1-element dummy.
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
        _fwd_kernel_compressive[grid](
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
        ctx.doc_id = doc_id  # None when not packing
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
        _bwd_kv_kernel_compressive[(ctx.grid[1], n_kv_blocks)](
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
        _bwd_q_kernel_compressive[(ctx.grid[1], ctx.grid[0])](
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


def fused_compressive_landmark_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_mem: torch.Tensor,
    sm_scale: float = None,  # type: ignore[assignment]
    block_size: int = 64,
    doc_id: Optional[torch.Tensor] = None,
    q_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compressive-landmark counterpart of :func:`landmark_fast.fused_landmark_attention_fast`.

    ``doc_id`` is an optional int32 ``(batch, seq_len_k // block_size)`` per-block document id for
    sequence packing (see :func:`~olmo_core.nn.attention.landmark.build_block_doc_id`); when given,
    cross-document key blocks are masked out so a query never attends across a document boundary.

    :param q_gate: Optional separate query ``(same shape as q)`` used **only** for the cross-block
        gate landmark logit -- the within-block and local (diagonal) softmaxes always use ``q``.
        Defaults to ``q`` itself, which is bit-identical to not having this parameter at all (the
        gate logit is then read directly off ``q @ k^T``, exactly as before this parameter existed).
    """
    expected_is_mem = torch.arange(0, is_mem.shape[-1], device=is_mem.device) % block_size == (
        block_size - 1
    )
    assert (is_mem == expected_is_mem).all()
    n_history_kv = k.shape[-2] - q.shape[-2]
    assert n_history_kv % block_size == 0
    n_history_blocks = n_history_kv // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if q_gate is None:
        q_gate = q
    return _FusedCompressiveLandmarkAttention.apply(
        q, q_gate, k, v, n_history_blocks, sm_scale, block_size, doc_id
    )


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
    :param group_landmark_selection: Under GQA (``n_kv_heads < n_heads``), how to choose the top-k
        landmark blocks *shared* by every query head in a KV group, instead of each head picking its
        own independently:

        * ``None`` (default): unchanged behavior -- each of the ``n_rep`` query heads in a group
          retrieves its own top-k blocks from its own scores, exactly as before ``repeat_kv``
          duplicated the group's K/V. Different heads in the same group can therefore select
          different blocks.
        * ``"mean"``: average the ``n_rep`` heads' landmark scores within each group and take one
          top-k over the average, shared by the whole group.
        * ``"max"``: take the per-landmark max over the ``n_rep`` heads' scores within each group
          (a block is kept if *any* head in the group ranks it highly), then top-k that.

        Only the *selection* (which blocks are eligible) is shared; the gate softmax and the
        within-block softmax that actually weight the output still use each head's own (real, not
        aggregated) scores -- see :meth:`_group_landmark_scores`. This is an inference-only knob
        (mirroring how ``top_k`` itself is only ever applied at eval/decode, never during training
        or prefill): the fused kernel's dense training/prefill forward always does full per-head soft
        gating over every block regardless of this setting, so it has zero effect until
        :meth:`set_landmark_eval_decode` (or the constructor) turns on hard top-k retrieval. See the
        module docstring and ``analysis/group_landmark_selection/DESIGN.md`` for the rationale (GQA's
        whole point is that a KV group's cache reads/bandwidth are shared across its query heads; if
        each head in the group retrieves a different block set, decode still has to touch the union of
        all of them, quietly giving up that saving. Sharing the retrieval decision is what actually
        realizes it).
    """

    # Compressive decode has different grouped-softmax semantics than the base, so it opts out of the
    # base ragged batched-decode path (which would silently use the non-compressive math).
    _supports_ragged_decode: bool = False

    def __init__(
        self,
        *,
        mem_freq: int,
        nonselected_landmark_mass: float = 0.1,
        softmax_scale: Optional[float] = None,
        group_landmark_selection: Optional[str] = None,
        gate_temperature: bool = False,
        **kwargs,
    ):
        super().__init__(mem_freq=mem_freq, softmax_scale=softmax_scale, **kwargs)
        if not (0.0 <= nonselected_landmark_mass < 1.0):
            raise OLMoConfigurationError(
                f"nonselected_landmark_mass must be in [0, 1) (got {nonselected_landmark_mass})"
            )
        self.nonselected_landmark_mass = nonselected_landmark_mass
        if group_landmark_selection not in _GROUP_LANDMARK_SELECTIONS:
            raise OLMoConfigurationError(
                "group_landmark_selection must be one of None/'mean'/'max' "
                f"(got {group_landmark_selection!r})"
            )
        self.group_landmark_selection = group_landmark_selection
        self.gate_temperature = gate_temperature
        self.log_gate_temp = (
            nn.Parameter(
                torch.zeros((), dtype=self.w_q.weight.dtype, device=self.w_q.weight.device)
            )
            if gate_temperature
            else None
        )

    def reset_parameters(self) -> None:
        if self.log_gate_temp is not None:
            nn.init.zeros_(self.log_gate_temp)

    def set_landmark_eval_decode(
        self,
        prompt_len: int,
        mode: str = "extend_last_block",
        top_k: Optional[int] = None,
        nonselected_landmark_mass: Optional[float] = None,
        group_landmark_selection: Optional[str] = _NO_OVERRIDE,  # type: ignore[assignment]
    ) -> None:
        """Enable "one long local block" decoding (see :class:`FastLandmarkAttention`).

        :param nonselected_landmark_mass: Optionally override the module's default
            :attr:`nonselected_landmark_mass` for this eval run. Only used when ``top_k`` is set.
        :param group_landmark_selection: Optionally override the module's default
            :attr:`group_landmark_selection` (``None``/``"mean"``/``"max"``) for this eval run.
            Defaults to a private "no override" sentinel (distinct from ``None``, which is itself a
            legitimate override value here -- "force grouping off for this run" -- unlike
            ``nonselected_landmark_mass`` where ``None`` unambiguously means "don't override").
        """
        # NOT ``super().set_landmark_eval_decode(...)``: this method is shared onto
        # ``DocumentCompressiveLandmarkAttention`` via class-attribute assignment (not inheritance,
        # see landmark_document_compressive.py), so ``self`` is sometimes an instance of an unrelated
        # class hierarchy. Zero-arg ``super()`` closes over ``__class__ == FastCompressiveLandmarkAttention``
        # at compile time and raises ("obj must be an instance or subtype of type") whenever ``self``
        # isn't in that hierarchy. Calling the known leaf implementation explicitly works for both.
        FastLandmarkAttention.set_landmark_eval_decode(self, prompt_len, mode, top_k=top_k)
        if nonselected_landmark_mass is not None:
            if not (0.0 <= nonselected_landmark_mass < 1.0):
                raise OLMoConfigurationError(
                    f"nonselected_landmark_mass must be in [0, 1) (got {nonselected_landmark_mass})"
                )
            self.nonselected_landmark_mass = nonselected_landmark_mass
        if group_landmark_selection is not _NO_OVERRIDE:
            if group_landmark_selection not in _GROUP_LANDMARK_SELECTIONS:
                raise OLMoConfigurationError(
                    "group_landmark_selection must be one of None/'mean'/'max' "
                    f"(got {group_landmark_selection!r})"
                )
            self.group_landmark_selection = group_landmark_selection

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
        T = q.shape[2]
        is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
        q_gate = q * torch.exp(-self.log_gate_temp) if self.log_gate_temp is not None else None
        return fused_compressive_landmark_attention(
            q,
            k,
            v,
            is_mem,
            sm_scale=self.softmax_scale,
            block_size=self.block_size,
            doc_id=doc_id,
            q_gate=q_gate,
        )

    def _group_landmark_scores(self, lm_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate per-head landmark scores across each GQA group, for top-k *ranking only*.

        ``lm_scores`` has shape ``(B, H, 1, n_lm)`` with ``H == n_heads`` (already expanded from
        ``n_kv_heads`` by ``repeat_kv``, so heads ``[g*n_rep, (g+1)*n_rep)`` share KV group ``g``).
        When :attr:`group_landmark_selection` is set, every head's value in the returned tensor is
        replaced by its group's aggregate, so a subsequent ``topk(..., dim=-1)`` picks the *same*
        indices for every head in the group -- one shared retrieval decision per KV group instead of
        ``n_rep`` independent ones. Returns ``lm_scores`` unchanged (same object) when grouping is
        off or there is nothing to group (``n_rep == 1``, i.e. MHA), so that path is bit-identical to
        the pre-existing per-head behavior.

        ``"inverse_mean"`` is the anti-selection sanity check (see :data:`_GROUP_LANDMARK_SELECTIONS`):
        it returns the *negated* group mean so the caller's ``topk`` keeps the group's lowest-scoring
        (least-attended) blocks. Because the returned tensor is used for ranking only -- the gate and
        within-block softmaxes downstream re-read the true ``scores`` -- the selected bad blocks are
        still weighted by their real (low) scores, so this genuinely forces attention onto the blocks
        the group cares least about.

        :param lm_scores: Per-head landmark-key logits, ``(B, H, 1, n_lm)``.
        """
        if self.group_landmark_selection is None:
            return lm_scores
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep == 1:
            return lm_scores
        B, H, one, n_lm = lm_scores.shape
        grouped = lm_scores.view(B, self.n_kv_heads, n_rep, one, n_lm)
        if self.group_landmark_selection == "mean":
            agg = grouped.mean(dim=2, keepdim=True)
        elif self.group_landmark_selection == "inverse_mean":
            # Negate so the caller's top-k picks the LOWEST-mean (least-attended) blocks. Sanity
            # check only; see the class/constant docstrings.
            agg = -grouped.mean(dim=2, keepdim=True)
        else:
            assert self.group_landmark_selection == "max"
            agg = grouped.amax(dim=2, keepdim=True)
        return agg.expand_as(grouped).reshape(B, H, one, n_lm)

    def _compressive_decode_probs(
        self,
        scores: torch.Tensor,
        is_mem: torch.Tensor,
        last_section: torch.Tensor,
        section_start: int,
        gate_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compressive grouped softmax for a single decode query.

        :param scores: Attention logits of shape ``(B, H, 1, total)`` (already scaled).
        :param is_mem: Boolean mask ``(total,)`` marking *past* landmark key positions (the local
            section's landmark is never present, see :meth:`FastLandmarkAttention._decode_one`).
        :param last_section: Boolean mask ``(total,)`` marking the local-section keys.
        :param section_start: Start position of the local section; a multiple of ``block_size`` so
            the past region ``[0, section_start)`` partitions into whole landmark blocks.
        :param gate_scores: Optional separate logits ``(B, H, 1, total)`` used **only** for the
            cross-block GATE (the block rescaling ``G_b`` and the top-k retrieval that ranks blocks) at
            the *past-landmark* columns; the within-block softmax and the local section keep ``scores``.
            Used by :class:`~olmo_core.nn.attention.landmark_compressive_gqa.CompressiveGQAGroupedAttention`
            to gate decode by group-mean landmark scores, matching the group-mean gating it was trained
            with (Version A). ``None`` (default) uses ``scores`` for the gate too -- the per-head decode,
            which is the unchanged behavior (and, with ``group_landmark_selection`` set, the
            selection-only Version B). When provided, its landmark columns are already group-shared, so
            the top-k ranking is naturally group-shared and ``_group_landmark_scores`` is skipped.

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

        # The GATE (block rescaling + top-k ranking) reads ``gate_src``: ``scores`` per-head by default,
        # or group-mean landmark logits (at the landmark columns) when ``gate_scores`` is given. The
        # within-block softmax below always reads the per-head ``scores``.
        gate_src = scores if gate_scores is None else torch.where(is_mem_b, gate_scores, scores)

        top_k = self._eval_top_k
        recording = gate_log.is_enabled()
        if top_k is not None and n_lm > top_k:
            lm_scores = gate_src[..., lm_idx]  # (B, H, 1, n_lm)
            # ``gate_scores`` is already group-shared at landmark columns, so rank on it directly;
            # otherwise apply the selection-only group aggregation (group_landmark_selection).
            rank_scores = lm_scores if gate_scores is not None else self._group_landmark_scores(
                lm_scores
            )
            keep = torch.zeros_like(lm_scores, dtype=torch.bool)
            keep.scatter_(-1, rank_scores.topk(top_k, dim=-1).indices, True)
            selected = torch.zeros(B, H, 1, total, dtype=torch.bool, device=device)
            selected[..., lm_idx] = keep
            has_nonselected = True
            alpha = float(self.nonselected_landmark_mass)
            if recording:
                gate_log.record_layer(
                    getattr(self, "_gate_log_layer_idx", None), keep, lm_idx // Lb, lm_scores
                )
        else:
            selected = is_mem_b.expand(B, H, 1, total)
            has_nonselected = False
            alpha = 0.0
            if recording and top_k is not None and n_lm > 0:
                # n_lm <= top_k: every past block's gate is open this step.
                keep_all = torch.ones(B, H, 1, n_lm, dtype=torch.bool, device=device)
                gate_log.record_layer(
                    getattr(self, "_gate_log_layer_idx", None),
                    keep_all,
                    lm_idx // Lb,
                    gate_src[..., lm_idx],
                )

        if getattr(self, "_eval_flat_softmax", False):
            # Inference-only ablation: keep the hard top-k selection but replace the gated softmax
            # with a plain softmax over exactly the value-carrying support. For the compressive model
            # that is {selected blocks' content + landmark tokens, local section}; non-selected blocks
            # are excluded entirely (no ``nonselected_landmark_mass`` alpha when the flag is on).
            # ``selected`` marks only landmark positions, so index the per-block landmark to recover
            # each block's keep flag and broadcast it over the block's Lb positions.
            # See analysis/flat_softmax_variant_eval.md.
            flat_visible = last_section_b.expand(B, H, 1, total).clone()
            if S > 0:
                block_landmark_pos = torch.arange(Lb - 1, S, Lb, device=device)
                block_sel = selected[..., block_landmark_pos]  # (B, H, 1, n_past_blocks) bool
                flat_visible[..., :S] |= block_sel.repeat_interleave(Lb, dim=-1)
            return torch.softmax(scores.masked_fill(~flat_visible, neg_inf), dim=-1)

        # Gate (cross-block) softmax over the selected landmarks + the local section.
        gate_set = selected | last_section_b
        gate_w = torch.softmax(gate_src.masked_fill(~gate_set, neg_inf), dim=-1)

        final = torch.zeros(B, H, 1, total, dtype=gate_w.dtype, device=device)
        # Local section keys keep their gate weight directly.
        final = torch.where(last_section_b, gate_w, final)
        # Past blocks: full within-block softmax (over the PER-HEAD ``scores``) distributes the block's
        # gate weight over its content tokens AND its landmark. Non-selected blocks have gate weight 0
        # here (their landmark was masked out of ``gate_set``), so they contribute 0 in this term.
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
            ns_w = torch.softmax(gate_src.masked_fill(~nonsel, neg_inf), dim=-1)
            final = final + alpha * ns_w

        return final

    def _decode_gate_scores(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Optional separate GATE logits for decode (see ``gate_scores`` in
        :meth:`_compressive_decode_probs`). ``None`` (the per-head gate, read directly off
        ``scores``) unless :attr:`log_gate_temp` is set, in which case this returns the same
        temperature-scaled gate logits used at train/prefill (see :meth:`_attn_core`), so decode
        matches training. Overridden by
        :class:`~olmo_core.nn.attention.landmark_compressive_gqa.CompressiveGQAGroupedAttention` to
        return group-mean landmark logits so decode matches its group-mean training/prefill gate.

        This method is also shared onto :class:`~olmo_core.nn.attention.landmark_document_compressive.DocumentCompressiveLandmarkAttention`
        via class-attribute assignment (not inheritance -- see that module), which never sets
        :attr:`log_gate_temp` (it doesn't support ``gate_temperature``), so the lookup below uses
        ``getattr`` rather than assuming the attribute exists.
        """
        log_gate_temp = getattr(self, "log_gate_temp", None)
        if log_gate_temp is None:
            return None
        return torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale * torch.exp(
            -log_gate_temp
        )

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
        gate_scores = self._decode_gate_scores(q, k)
        probs = self._compressive_decode_probs(
            scores, is_mem, last_section, section_start, gate_scores=gate_scores
        )
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
        gate_scores = self._decode_gate_scores(q, k)
        probs = self._compressive_decode_probs(
            scores, is_mem, last_section, section_start, gate_scores=gate_scores
        )
        return torch.matmul(probs.to(v.dtype), v)
