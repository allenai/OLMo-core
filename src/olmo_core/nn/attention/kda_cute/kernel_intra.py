# mypy: ignore-errors
# The CuTe DSL kernels use metaclass-generated attributes and DSL-typed unpacking that mypy
# cannot follow.
"""Phase 1: bwd_intra restructured — one CTA per (chunk, K-slab), sub-chunks internal.

fla's `chunk_kda_bwd_intra` runs grid (NK*NC, NT, B*HV) = 524k CTAs of [BC=16, BK=32] work
at prod8192 and costs 12.6ms, ~5x its floors (ALGORITHM.md). This kernel is the SAME math,
expression for expression — including the one-sided exp2(g_i - g_j) scalar diagonal loops,
which are the numerics law of this op and must never be refactored through a reference row
(see ALGORITHM.md "the numerics law"; the gx16 dbg arm enforces it) — with two structural
changes only:

1. The NC sub-chunk axis moves inside the CTA (tl.static_range unroll). Each CTA touches
   the same dAqk/dAkk [BT,BT] tiles and partner q/k/g tiles up to NC times; in fla those
   hits are spread across 4 CTAs on different SMs (DRAM/L2 re-reads), here they are
   same-CTA L1/L2 hits, and the launch count drops 4x. (Measured alone: 12.62 -> 10.44ms
   at prod8192; widening BK past that measured flat, so it is pinned, not autotuned.)
2. BK is pinned per call (default 64, KDA002_INTRA_BK to A/B) instead of autotuned: with
   one NK for every config, every autotune config writes every db slab, closing the
   staleness hazard measured as db abs-err up to 3.9 (fla's CachedAutotuner does not run
   reset_to_zero's pre_hook on cached-config launches). BK=128 (NK=1) measured 13.34ms —
   register pressure — so small-BK slabs stay.
3. The diagonal [BC,BC] blocks are VECTORIZED: fla's two 16-iteration serial scalar
   j-loops (attributed at ~7.2ms of the 10.4ms kernel via KDA002_INTRA_SKIP=diag) become
   j-axis tensor reductions over [BC,BC,BK] elementwise products — still exactly one
   one-sided exp2(g_r - g_s) per (r,s,d) pair, nothing factorized; the dA diag tile loads
   once for both passes. Reduction-tree reassociation puts outputs at ~1e-6 abs of fla
   instead of bit-exact; dbg_intra budgets 1e-5.

Interface mirrors fla's wrapper (fresh dq2/dk2/dg2 outputs, incoming dq/dk/dg added
in-kernel). Fixed-length only, BT=64, K <= 128; everything else (varlen, safe_gate)
falls back to fla's kernel unchanged.

Falsified here, kept out: merging the two diagonal scalar loops into one (shared k_j/g_j
loads, half the sequential iterations) bought only ~0.1ms of 10.4 and cost the bit-exact
gate (~1e-6 FMA-contraction drift on dk/dg) — the diagonals are throughput-bound, not
iteration-count-bound.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC", "BK", "K", "HV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["B", "T"])
def kda_cute_bwd_intra_kernel(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    T,
    B,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    SKIP_DIAG: tl.constexpr,
    SKIP_OFFDIAG: tl.constexpr,
):
    # SKIP_* are timing-attribution knobs (KDA002_INTRA_SKIP=diag|offdiag): they delete one
    # half of the work at compile time to see what the other half costs. Results are WRONG
    # with either set; dbg_intra.py --time is the only intended caller.
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    all = B * T
    bos = i_b * T
    if i_t * BT >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += (i_k * all + bos) * HV + i_hv

    o_i = tl.arange(0, BC)
    # partial-chunk sub-chunk count, fla's clamp for pass B's upper loop
    NCc = min(NC, tl.cdiv(T - i_t * BT, BC))

    for i_i in tl.static_range(NC):
        i_ti = i_t * BT + i_i * BC
        if i_ti < T:
            o_c = i_ti + o_i
            m_c = o_c < T
            m_ck = m_c[:, None] & m_k[None, :]
            m_dAf = m_c[:, None] & (o_i[None, :] < BT)

            p_g = g + o_c[:, None] * (HV * K) + o_k[None, :]
            b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)
            p_b = beta + o_c * HV
            b_b = tl.load(p_b, mask=m_c, other=0.0)

            # ---- pass A: row side. dq_intra and the dAkk->dwk products for rows in
            # this sub-chunk, from columns in earlier sub-chunks + the diagonal.
            b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
            b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
            if (i_i > 0) & (SKIP_OFFDIAG == 0):
                p_gn = g + i_ti * HV * K + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
                for i_j in range(0, i_i):
                    o_j = i_t * BT + i_j * BC + o_i
                    m_jk = (o_j < T)[:, None] & m_k[None, :]
                    p_kj = k + o_j[:, None] * (H * K) + o_k[None, :]
                    p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
                    p_dAqk = dAqk + o_c[:, None] * (HV * BT) + (i_j * BC + o_i)[None, :]
                    p_dAkk = dAkk + o_c[:, None] * (HV * BT) + (i_j * BC + o_i)[None, :]
                    b_kj = tl.load(p_kj, mask=m_jk, other=0.0)
                    b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
                    b_kg = b_kj * exp2(b_gn - b_gk)
                    b_dAqk = tl.load(p_dAqk, mask=m_dAf, other=0.0)
                    b_dAkk = tl.load(p_dAkk, mask=m_dAf, other=0.0)
                    b_dq2 += tl.dot(b_dAqk, b_kg)
                    b_dk2 += tl.dot(b_dAkk, b_kg)
                b_gqn = exp2(b_g - b_gn)
                b_dq2 *= b_gqn
                b_dk2 *= b_gqn

            # ---- the diagonal [BC,BC] block, BOTH passes, vectorized: still one exp2 per
            # (r, s, d) pair with a one-sided exponent (the numerics law — nothing is
            # factorized through a reference row); only the serial j-loop becomes a j-axis
            # tensor reduction, and the [BC,BC] dA diag tiles are loaded once for the two
            # passes instead of 2*BC scalar-column loads.
            p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
            p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_ck, other=0.0)
            b_k = tl.load(p_k, mask=m_ck, other=0.0)

            if SKIP_DIAG == 0:
                m_cc = m_c[:, None] & m_c[None, :]
                p_dAd_qk = dAqk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
                p_dAd_kk = dAkk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
                b_dAd_qk = tl.load(p_dAd_qk, mask=m_cc, other=0.0)  # [r, s]
                b_dAd_kk = tl.load(p_dAd_kk, mask=m_cc, other=0.0)
                b_kf = b_k.to(tl.float32)

                # pass A rows: dq/dwk[r,d] += sum_j dA[r,j] * k[j,d] * exp2(g_r - g_j), r >= j
                m_jr = (o_i[:, None] <= o_i[None, :])[:, :, None]  # [j, r, 1]
                e_jr = exp2(b_g[None, :, :] - b_g[:, None, :])  # [j, r, d]
                b_kx = b_kf[:, None, :]  # k_j -> [j, 1, d]
                b_dq2 += tl.sum(
                    tl.where(m_jr, tl.trans(b_dAd_qk)[:, :, None] * b_kx * e_jr, 0.0), 0
                )
                b_dk2 += tl.sum(
                    tl.where(m_jr, tl.trans(b_dAd_kk)[:, :, None] * b_kx * e_jr, 0.0), 0
                )

                # pass B columns: dkt[s,d] += sum_r (dAqk[r,s]*q[r,d] + dAkk[r,s]*β_r*k[r,d])
                #                              * exp2(g_r - g_s), r >= s
                m_rs = (o_i[:, None] >= o_i[None, :])[:, :, None]  # [r, s, 1]
                e_rs = exp2(b_g[:, None, :] - b_g[None, :, :])  # [r, s, d]
                b_qx = b_q.to(tl.float32)[:, None, :]
                b_kbx = (b_kf * b_b[:, None])[:, None, :]
                b_dktd = tl.sum(
                    tl.where(
                        m_rs,
                        (b_dAd_qk[:, :, None] * b_qx + b_dAd_kk[:, :, None] * b_kbx) * e_rs,
                        0.0,
                    ),
                    0,
                )
            else:
                b_dktd = tl.zeros([BC, BK], dtype=tl.float32)

            b_db = tl.sum(b_dk2 * b_k, 1)
            b_dk2 *= b_b[:, None]

            p_dq = dq + o_c[:, None] * (HV * K) + o_k[None, :]
            p_dq2 = dq2 + o_c[:, None] * (HV * K) + o_k[None, :]
            p_db = db + o_c * HV

            b_dg2 = b_q * b_dq2
            b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
            tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
            tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

            # ---- pass B: column side. dkt for columns in this sub-chunk, from rows in
            # later sub-chunks + the diagonal.
            b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
            if (i_i < NCc - 1) & (SKIP_OFFDIAG == 0):
                p_gn = g + (min(i_ti + BC, T) - 1) * HV * K + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
                for i_j in range(i_i + 1, NC):
                    if i_j < NCc:
                        o_j = i_t * BT + i_j * BC + o_i
                        m_j = o_j < T
                        m_jk = m_j[:, None] & m_k[None, :]
                        m_dAj = (o_i[:, None] < BT) & m_j[None, :]
                        p_qj = q + o_j[:, None] * (H * K) + o_k[None, :]
                        p_kj2 = k + o_j[:, None] * (H * K) + o_k[None, :]
                        p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
                        p_bj = beta + o_j * HV
                        p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
                        p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
                        b_bj = tl.load(p_bj, mask=m_j, other=0.0)
                        b_qj = tl.load(p_qj, mask=m_jk, other=0.0)
                        b_kbj = tl.load(p_kj2, mask=m_jk, other=0.0) * b_bj[:, None]
                        b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
                        b_dAqk = tl.load(p_dAqk, mask=m_dAj, other=0.0)
                        b_dAkk = tl.load(p_dAkk, mask=m_dAj, other=0.0)
                        b_gkn = exp2(b_gk - b_gn)
                        b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
                        b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0)
                        # (SY 09/17, kept) important to not use bf16 here for precision
                        b_dkt += tl.dot(b_dAqk, b_qg)
                        b_dkt += tl.dot(b_dAkk, b_kbg)
                b_dkt *= exp2(b_gn - b_g)

            # diagonal contribution, computed vectorized above (added after the off-diag
            # scaling, matching fla's accumulation order)
            b_dkt += b_dktd

            p_dk = dk + o_c[:, None] * (HV * K) + o_k[None, :]
            p_dk2 = dk2 + o_c[:, None] * (HV * K) + o_k[None, :]
            p_dg = dg + o_c[:, None] * (HV * K) + o_k[None, :]
            p_dg2 = dg2 + o_c[:, None] * (HV * K) + o_k[None, :]

            b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
            b_dk2 += tl.load(p_dk, mask=m_ck, other=0.0)
            b_dk2 += b_dkt

            tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_ck)
            tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_ck)


def chunk_kda_bwd_intra_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
):
    """Drop-in for fla's chunk_kda_bwd_intra; falls back to it off the supported box."""
    if (
        cu_seqlens is not None
        or safe_gate
        or chunk_size != 64
        or k.shape[-1] > 128
        or os.environ.get("KDA002_INTRA") == "fla"
    ):
        from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra

        return chunk_kda_bwd_intra(
            q=q,
            k=k,
            g=g,
            beta=beta,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dk=dk,
            db=db,
            dg=dg,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
        )

    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BC = min(16, BT)
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    dg2 = torch.empty_like(dg, dtype=torch.float)

    # BK is PINNED per call (not autotuned): every autotune config then has the same NK
    # and writes every db slab, so no benchmarked config can leave stale slab garbage —
    # the hazard only exists when configs differ in NK (fla's CachedAutotuner skips the
    # reset_to_zero pre_hook on cached-config launches). BK=128 measured 13.34ms vs 10.4
    # at tuner-chosen BK (register pressure), so the default is 64; KDA002_INTRA_BK
    # selects for A/B timing.
    # BK sweep with the vectorized diagonal (prod8192): 64 -> 9.26ms, 32 -> 10.19,
    # 16 -> 11.56, 128 -> register pressure. 64 is the standing default.
    BK = min(int(os.environ.get("KDA002_INTRA_BK", "64")), triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)

    skip = os.environ.get("KDA002_INTRA_SKIP", "")
    kda_cute_bwd_intra_kernel[(NK, NT, B * HV)](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SKIP_DIAG=1 if skip == "diag" else 0,
        SKIP_OFFDIAG=1 if skip == "offdiag" else 0,
    )
    db_out = db2.sum(0).add_(db)
    return dq2, dk2, db_out, dg2
