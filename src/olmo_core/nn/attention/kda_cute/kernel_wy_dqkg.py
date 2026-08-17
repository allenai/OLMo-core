"""wy_dqkg_fused restructured: the WY branch hoisted out of the K/V loop nest.

fla's chunk_kda_bwd_kernel_wy_dqkg_fused is 6.15ms at prod8192 — the largest backward
stage after iteration 1 — against a ~2.2ms traffic floor. Its inner V-loop carries an
`if i_k == 0` WY branch (dA += dv@v^T, dvb = A@dv, the dv2 store) and a
`tl.debug_barrier()  # DO NOT REMOVE` right before it: a full CTA barrier between every
pair of [BT,BV] MMAs, which defeats the software pipeline num_stages is supposed to buy.

First attempt here (BV=V, one V iteration, barrier dropped) VALIDATED but ran 0.72x at
prod8192: [64,256] operand tiles put a CTA's shared-memory footprint near the 227KB limit,
occupancy collapsed to 1 CTA/SM, and the exposed latency cost more than the barrier did.
The tiling was never the problem; the loop body was.

This version keeps fla's occupancy-friendly tiling (BK/BV autotuned over fla's own range)
and instead HOISTS the WY work into its own V-pass before the K-loop:

    pass 1 (V-tiles): dA += dv @ v^T;  dvb = A @ dv;  dv2 = dvb·beta;  db += Σ dvb·v
    pass 2 (K-tiles): fla's dq/dk/dw/dgk V-reduction — now branch- and barrier-free —
                      then fla's gate/WY tail verbatim
    tail: the triangular A sandwich, unchanged

The inner loop of pass 2 is three straight-line MMAs, so the pipeliner can actually
overlap the h/dh/do/v_new streams. Cost of the hoist: dv is swept once more (L2-resident
[BT,V] tiles). Numerics: every expression, cast point and mask is fla's; only the
accumulation ORDER of the fp32 dA/db across passes changes (fp32 adds, reassociation
noise only — dbg_wy holds the line).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp2  # the same exp2 fla's kernel lowers to


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for BV in [64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "K", "V", "HV"],
)
@triton.jit(do_not_specialize=["T"])
def kda_bwd_wy_dqkg_kernel(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = o_t == min(T, i_t * BT + BT) - 1

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K * V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K * V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT

    p_beta = beta + o_t * HV
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_A = A + o_A[:, None] + o_t[None, :] * (HV * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    # ---- pass 1: the WY du-backward, once, out of the K-loop ----
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]

        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

        b_dA += tl.dot(b_dv, tl.trans(b_v))

        b_dvb = tl.dot(b_A, b_dv)
        b_dv2 = b_dvb * b_beta[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)

        tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), mask=m_tv)

    # ---- pass 2: dq/dk/dw/dgk per K-tile, branch- and barrier-free inner loop ----
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]

        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_tv = m_t[:, None] & (o_v[None, :] < V)
            m_h = (o_v[:, None] < V) & m_k[None, :]
            p_v_new = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
            p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
            p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]

            b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            b_h = tl.load(p_h, mask=m_h, other=0.0)
            b_dh = tl.load(p_dh, mask=m_h, other=0.0)
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = dq + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (HV * K) + o_k[None, :]
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
    p_db = db + o_t * HV
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


def chunk_kda_bwd_wy_dqkg_wide(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    """Drop-in for fla's chunk_kda_bwd_wy_dqkg_fused, falling back on shapes not covered.
    Chunk-parallel, no minimum-grid gate needed."""
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    supported = (
        cu_seqlens is None
        and chunk_indices is None
        and not state_v_first
        and chunk_size == 64
        and K in (64, 128)
        and V % 64 == 0
    )
    if not supported:
        from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused

        return chunk_kda_bwd_wy_dqkg_fused(
            q=q,
            k=k,
            v=v,
            v_new=v_new,
            g=g,
            beta=beta,
            A=A,
            h=h,
            do=do,
            dh=dh,
            dv=dv,
            scale=scale,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

    BT = chunk_size
    NT = triton.cdiv(T, BT)

    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * HV)
    kda_bwd_wy_dqkg_kernel[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
    )
    return dq, dk, dv2, db, dg, dA
