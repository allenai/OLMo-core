"""bwd_intra restructured: fla's chunk_kda_bwd_kernel_intra at BK=K with dot-form diagonals.

fla launches (NK·NC, NT, B·HV) CTAs with BK=32 — at prod8192 that is 524k CTAs of 16x32
work, each k-tile re-reading the SAME fp32 dAqk/dAkk [16,16] tiles (NK=4 multiplicity on
0.54GB tensors), re-reading g/q/k per tile, and walking the diagonal blocks with a
16-iteration scalar loop (no tl.dot, one [BC,BK] exp2 per iteration). dbg_perf attributes
12.6ms = 49.8% of the whole backward to it, ~5x over its ~2.5ms traffic floor — it is
latency/SIMT-bound, not bandwidth-bound.

This kernel computes the identical math with two structural changes, nothing else:
  1. BK = K (one k-tile): grid (NC, NT, B·HV), 4x fewer/8x fatter CTAs, every dA/g/q/k/dq/
     dk/dg byte read once per consumer instead of NK times.
  2. db's NK-slab reduction disappears (NK=1): the kernel adds the incoming db in-place
     to its contribution and writes one output row.
The diagonal blocks keep fla's default scalar loops (safe_gate=False): the direct factor
exp2(g_i - g_j) with i ≥ j is one-sided (decay ⇒ exponent ≤ 0), safe at any gate
magnitude. An earlier revision used fla's SAFE_GATE dot form instead — the gate factor
split around the sub-chunk midpoint, exp2(g_i - g_mid) · exp2(g_mid - g_j) — but that
factorization is only bounded for lower-bounded gate activations (which is why fla gates
it behind safe_gate). KDA initializes exp(A_log) in [1, 16], so per-step decays reach
~16 log2 units per channel and ±8 rows around the midpoint overspill fp32 exp2's ±127:
e_pos/e_neg hit inf, inf · 0 = nan, and the whole backward NaNs on the first real step.
The off-diagonal loops keep fla's first/last-row references, one-sided and safe.

Everything numeric mirrors fla expression for expression (same load dtypes, same cast
points, same accumulation order within each loop) so dbg_intra can hold a tight line
against the fla kernel, and dbg_bwd/spec tolerances judge the tf32-dot diagonals.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp2  # the same exp2 fla's kernel lowers to


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["BK", "NC", "BT", "HV"],
)
@triton.jit(do_not_specialize=["T"])
def kda_bwd_intra_kernel(
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
    db2,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
):
    i_i, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = tl.arange(0, BK)

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
    db += bos * HV + i_hv
    db2 += bos * HV + i_hv

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T
    m_k = o_k < K
    m_ck = m_c[:, None] & m_k[None, :]
    m_dAf = m_c[:, None] & (o_i[None, :] < BT)

    p_g = g + o_c[:, None] * (HV * K) + o_k[None, :]
    b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)
    p_b = beta + o_c * HV
    b_b = tl.load(p_b, mask=m_c, other=0.0)
    p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_ck, other=0.0)
    b_k = tl.load(p_k, mask=m_ck, other=0.0)

    # ---- rows i, columns j <= i: dq2/dk2 = sum_j dA[i,j] · k_j · exp2(g_i - g_j) ----
    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * (HV * K) + o_k
        b_gn = tl.load(p_gn, mask=o_k < K, other=0.0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_jk = (o_j < T)[:, None] & (o_k[None, :] < K)
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

    # diagonal block, scalar-loop form (fla's default safe_gate=False path): the direct
    # exp2(g_i - g_j), i >= j, is one-sided so it cannot overflow — see module docstring.
    o_dA = o_c * (HV * BT) + i_i * BC
    p_kd = k + i_ti * (H * K) + o_k
    p_gd = g + i_ti * (HV * K) + o_k
    for j in range(0, min(BC, T - i_ti)):
        b_dAqk_d = tl.load(dAqk + o_dA + j, mask=m_c, other=0.0)
        b_dAkk_d = tl.load(dAkk + o_dA + j, mask=m_c, other=0.0)
        b_kd = tl.load(p_kd, mask=m_k, other=0.0).to(tl.float32)
        b_gd = tl.load(p_gd, mask=m_k, other=0.0).to(tl.float32)
        m_ij = o_i[:, None] >= j
        b_gqk = exp2(b_g - b_gd[None, :])
        b_dq2 += tl.where(m_ij, b_dAqk_d[:, None] * b_kd[None, :] * b_gqk, 0.0)
        b_dk2 += tl.where(m_ij, b_dAkk_d[:, None] * b_kd[None, :] * b_gqk, 0.0)
        p_kd += H * K
        p_gd += HV * K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = dq + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dq2 = dq2 + o_c[:, None] * (HV * K) + o_k[None, :]
    p_db_in = db + o_c * HV
    p_db2 = db2 + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    b_db += tl.load(p_db_in, mask=m_c, other=0.0)
    tl.store(p_db2, b_db.to(p_db2.dtype.element_ty), mask=m_c)

    # ---- rows i, columns j >= i of dA^T: dkt = sum_j dA[j,i] · x_j · exp2(g_j - g_i) ----
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    NC_e = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC_e - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * (HV * K) + o_k
        b_gn = tl.load(p_gn, mask=o_k < K, other=0.0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC_e):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & (o_k[None, :] < K)
            m_dAj = (o_i[:, None] < BT) & m_j[None, :]
            p_qj = q + o_j[:, None] * (H * K) + o_k[None, :]
            p_kj = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
            p_bj = beta + o_j * HV
            p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
            p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
            b_bj = tl.load(p_bj, mask=m_j, other=0.0)
            b_qj = tl.load(p_qj, mask=m_jk, other=0.0)
            b_kb = tl.load(p_kj, mask=m_jk, other=0.0) * b_bj[:, None]
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, mask=m_dAj, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAj, other=0.0)
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0.0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0.0)
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn - b_g)

    # transposed diagonal block, scalar-loop form (same one-sided-exponent argument)
    o_dA_t = i_ti * (HV * BT) + i_i * BC + o_i
    p_qt = q + i_ti * (H * K) + o_k
    p_kt = k + i_ti * (H * K) + o_k
    p_gt = g + i_ti * (HV * K) + o_k
    p_bt = beta + i_ti * HV
    for j in range(0, min(BC, T - i_ti)):
        b_dAqk_t = tl.load(dAqk + o_dA_t + j * (HV * BT))
        b_dAkk_t = tl.load(dAkk + o_dA_t + j * (HV * BT))
        b_qt = tl.load(p_qt, mask=m_k, other=0.0).to(tl.float32)
        b_kbt = tl.load(p_kt, mask=m_k, other=0.0).to(tl.float32) * tl.load(p_bt)
        b_gt = tl.load(p_gt, mask=m_k, other=0.0).to(tl.float32)
        m_ij = o_i[:, None] <= j
        b_gkq = exp2(b_gt[None, :] - b_g)
        b_dkt += tl.where(m_ij, b_dAqk_t[:, None] * b_qt[None, :] * b_gkq, 0.0)
        b_dkt += tl.where(m_ij, b_dAkk_t[:, None] * b_kbt[None, :] * b_gkq, 0.0)
        p_qt += H * K
        p_kt += H * K
        p_gt += HV * K
        p_bt += HV

    p_dk = dk + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dk2 = dk2 + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dg = dg + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dg2 = dg2 + o_c[:, None] * (HV * K) + o_k[None, :]

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
    b_dk2 += tl.load(p_dk, mask=m_ck, other=0.0)
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_ck)
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_ck)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2, 3]
    ],
    key=["K", "BK", "BT", "HV"],
)
@triton.jit(do_not_specialize=["T"])
def kda_bwd_intra_v2_kernel(
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
    db2,
    T,
    B,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
):
    """v2: one CTA per (k-tile, chunk, b·hv); the sub-chunk pair loops collapse into NC
    column-batched [BT,BC]x[BC,BK] dots because stage 3/5 store dAqk/dAkk with their
    upper triangles explicitly zeroed — the data enforces the triangle, no masks.
    BK=64 keeps the four [BT,BK] fp32 accumulators at ~16 regs/thread each (a BK=K=128
    first cut validated but spilled at prod: 7.30ms vs v1's 5.04).

    WARNING — NOT SAFE under KDA's real initialization, DO NOT WIRE IN as-is: the
    in-block row factors below (exp2(g_r - gLAST_j) / exp2(gLAST_j - g_r)) span up to
    16 gate steps; with exp(A_log) in [1, 16] a step reaches ~16 log2 units, the factor
    overflows fp32 exp2's ±127 and the inf · underflowed-0 products emit NaN. The
    default v1 kernel handles the diagonal blocks with fla's one-sided scalar loops
    instead. Kept for reference only (falsified on perf at prod anyway).

    Gate factorization per source sub-chunk j — the operand factor is <= 1, the row
    factor is <= 1 outside the 16-row block and bounded by 16 gate steps inside it
    (fla's own safety class):
      dq2/dk2:  exp2(g_r - g_s) = exp2(g_r - gLAST_j) * exp2(gLAST_j - g_s),  s in j
      dkt:      exp2(g_s - g_r) = exp2(g_s - gFIRST_j) * exp2(gFIRST_j - g_r)
    The row factor is shared by the dAqk and dAkk dots of each j. dg is finalized
    in-register (per-dim independent), so fla's stage 7 reverse cumsum fuses for free.
    db contributions land in per-k-tile slabs, summed host-side like fla's original."""
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    all_t = B * T
    bos = i_b * T
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
    db += bos * HV + i_hv
    db2 += (i_k * all_t + bos) * HV + i_hv

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k = i_k * BK + tl.arange(0, BK)
    m_tk = m_t[:, None] & (o_k[None, :] < K)
    o_c = tl.arange(0, BC)

    p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
    p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    p_b = beta + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    # ---- rows r, sources s <= r: dq2/dk2, column-batched over source sub-chunks ----
    b_dq2 = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BT, BK], dtype=tl.float32)
    for i_j in range(0, NC):
        j0 = i_t * BT + i_j * BC
        if j0 < T:
            o_s = j0 + o_c
            m_s = o_s < T
            m_sk = m_s[:, None] & (o_k[None, :] < K)
            # reference: LAST valid row of source sub-chunk j
            p_gn = g + (min(j0 + BC, T) - 1) * (HV * K) + o_k
            b_gn = tl.load(p_gn, mask=o_k < K, other=0.0).to(tl.float32)[None, :]
            p_ks = k + o_s[:, None] * (H * K) + o_k[None, :]
            p_gs = g + o_s[:, None] * (HV * K) + o_k[None, :]
            b_ks = tl.load(p_ks, mask=m_sk, other=0.0)
            b_gs = tl.load(p_gs, mask=m_sk, other=0.0)
            # [BT, BC] column tiles; upper triangles are zero IN THE DATA
            p_dAqk = dAqk + o_t[:, None] * (HV * BT) + (i_j * BC + o_c)[None, :]
            p_dAkk = dAkk + o_t[:, None] * (HV * BT) + (i_j * BC + o_c)[None, :]
            m_dA = m_t[:, None] & (o_c[None, :] < BT)
            b_dAqk = tl.load(p_dAqk, mask=m_dA, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dA, other=0.0)
            # contribution = exp2(g_r - gn) * dot(dA, k_s * exp2(gn - g_s)):
            # operand factor <= 1 (gn is the block's last row); row factor <= 1 outside
            # the block, bounded by 16 gate steps inside it — fla's own safety class.
            b_kgn = b_ks * tl.where(m_s[:, None], exp2(b_gn - b_gs), 0.0)
            b_f = tl.where((o_t >= j0)[:, None] & m_t[:, None], exp2(b_g - b_gn), 0.0)
            b_dq2 += tl.dot(b_dAqk, b_kgn) * b_f
            b_dk2 += tl.dot(b_dAkk, b_kgn) * b_f

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq_in = dq + o_t[:, None] * (HV * K) + o_k[None, :]
    p_dq_out = dq2 + o_t[:, None] * (HV * K) + o_k[None, :]
    b_dg = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq_in, mask=m_tk, other=0.0)
    tl.store(p_dq_out, b_dq2.to(p_dq_out.dtype.element_ty), mask=m_tk)
    # contribution only — the wrapper sums the NK slabs and adds the incoming db,
    # exactly fla's original scheme (a slab per k-tile keeps autotune reruns clean)
    p_db_out = db2 + o_t * HV
    tl.store(p_db_out, b_db.to(p_db_out.dtype.element_ty), mask=m_t)

    # ---- rows r, sources s >= r: dkt, row-batched over source sub-chunks ----
    b_dkt = tl.zeros([BT, BK], dtype=tl.float32)
    for i_j in range(0, NC):
        j0 = i_t * BT + i_j * BC
        if j0 < T:
            o_s = j0 + o_c
            m_s = o_s < T
            m_sk = m_s[:, None] & (o_k[None, :] < K)
            # reference: FIRST row of source sub-chunk j
            p_gn = g + j0 * (HV * K) + o_k
            b_gn = tl.load(p_gn, mask=o_k < K, other=0.0).to(tl.float32)[None, :]
            p_qs = q + o_s[:, None] * (H * K) + o_k[None, :]
            p_ks = k + o_s[:, None] * (H * K) + o_k[None, :]
            p_gs = g + o_s[:, None] * (HV * K) + o_k[None, :]
            p_bs = beta + o_s * HV
            b_bs = tl.load(p_bs, mask=m_s, other=0.0)
            b_qs = tl.load(p_qs, mask=m_sk, other=0.0)
            b_kbs = tl.load(p_ks, mask=m_sk, other=0.0) * b_bs[:, None]
            b_gs = tl.load(p_gs, mask=m_sk, other=0.0).to(tl.float32)
            b_e = tl.where(m_s[:, None], exp2(b_gs - b_gn), 0.0)  # <= 1
            b_qg = b_qs * b_e
            b_kbg = b_kbs * b_e
            # [BC, BT] row tiles of dA, transposed in-register for the dots
            p_dAqk = dAqk + o_s[:, None] * (HV * BT) + tl.arange(0, BT)[None, :]
            p_dAkk = dAkk + o_s[:, None] * (HV * BT) + tl.arange(0, BT)[None, :]
            m_dA = m_s[:, None] & (tl.arange(0, BT)[None, :] < BT)
            b_dAqk = tl.load(p_dAqk, mask=m_dA, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dA, other=0.0)
            b_acc = tl.dot(tl.trans(b_dAqk), b_qg)
            b_acc += tl.dot(tl.trans(b_dAkk), b_kbg)
            b_f = tl.where((o_t < j0 + BC)[:, None] & m_t[:, None], exp2(b_gn - b_g), 0.0)
            b_dkt += b_acc * b_f

    p_dk_in = dk + o_t[:, None] * (HV * K) + o_k[None, :]
    p_dk_out = dk2 + o_t[:, None] * (HV * K) + o_k[None, :]
    p_dg_in = dg + o_t[:, None] * (HV * K) + o_k[None, :]
    p_dg_out = dg2 + o_t[:, None] * (HV * K) + o_k[None, :]

    b_dg += (b_dk2 - b_dkt) * b_k + tl.load(p_dg_in, mask=m_tk, other=0.0)
    b_dk2 += tl.load(p_dk_in, mask=m_tk, other=0.0)
    b_dk2 += b_dkt
    tl.store(p_dk_out, b_dk2.to(p_dk_out.dtype.element_ty), mask=m_tk)

    # fla's stage 7, in-register: reverse chunk-local cumsum of dg
    b_dg = tl.cumsum(b_dg, axis=0, reverse=True)
    tl.store(p_dg_out, b_dg.to(p_dg_out.dtype.element_ty), mask=m_tk)


def chunk_kda_bwd_intra_v2(
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
    """bwd_intra WITH fla's stage-7 reverse cumsum fused: returns dg already cumsum'd.
    REQUIRES dAqk/dAkk with zeroed upper triangles (what fla's dAv and both wy impls
    store). Falls back to (v1 intra, separate cumsum) on shapes not covered."""
    B, T, H, K, HV = *k.shape, g.shape[2]
    supported = (
        cu_seqlens is None
        and chunk_indices is None
        and chunk_size == 64
        and K in (64, 128)
        and not safe_gate
    )
    if not supported:
        from fla.ops.utils import chunk_local_cumsum

        dq, dk, db, dg = chunk_kda_bwd_intra_wide(
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
        dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True)
        return dq, dk, db, dg

    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, K)  # BK=K=128 spilled at prod (7.30ms vs v1's 5.04); 64 fits registers
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK, NT, B * HV)
    kda_bwd_intra_v2_kernel[grid](
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
        db=db,
        db2=db2,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        BK=BK,
    )
    db = db2.sum(0).add_(db)
    return dq2, dk2, db, dg2


def chunk_kda_bwd_intra_wide(
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
    """Drop-in for fla's chunk_kda_bwd_intra, falling back on shapes not covered.
    Chunk-parallel, so unlike the scans it needs no minimum-grid gate."""
    B, T, H, K, HV = *k.shape, g.shape[2]
    supported = (
        cu_seqlens is None
        and chunk_indices is None
        and chunk_size == 64
        and K in (64, 128)
        and not safe_gate  # our diagonals already use the safe structure; the flag also
        # changes fla's gate activation contract, which this harness never exercises
    )
    if not supported:
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

    BT = chunk_size
    BC = min(16, BT)
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)

    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = torch.empty_like(db, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NC, NT, B * HV)
    kda_bwd_intra_kernel[grid](
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
        db=db,
        db2=db2,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=K,
        NC=NC,
    )
    return dq2, dk2, db2, dg2
