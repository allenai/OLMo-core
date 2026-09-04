"""Phase 2 B2b, transposed — the wy_dqkg stage as two Triton kernels (ladder idea 005).

Same math as bwd_wy.py (002's kernel_wy2: fla's fused wy_dqkg at full K), re-arranged
around the two occupancy walls the 2026-09-01 probes found in wy2 (the ledger is in
kernels/kda/ideas/005-wy-transposed/NOTES.md; the ncu profile there reads 24.6%/36.2% warps
active against wy2's 12.4%, stage 4.98 -> 4.54ms at prod8192 B=16):

  * wy2's loop accumulates dq/dk/dw as [BT=64, K=128] tiles: M=64 tcgen05 MMAs, operands
    through registers, 255 regs -> 1 CTA/SM. Here the accumulators are TRANSPOSED —
    dq^T = h^T @ do^T, dw^T = h^T @ dv^T, dk^T = dh^T @ v_new^T, all M=K=128 — and the
    loop alone streams at the DRAM roofline (1.7ms for its 13GB at prod8192, 78 regs).
  * The epilogue's live set ([64,128] fp32 x ~8) is what pins wy2 at 255 regs. Here the
    raw accumulators are parked TRANSPOSED in the fp32 outputs at loop exit and the
    elementwise epilogue re-reads them from L2 in K-strips of KS rows, so the kernel
    compiles to 128 regs / 0 spills and two CTAs share an SM. Two is also the ceiling:
    Triton allocates 256 tmem columns for these accumulators, and tmem is 512/SM.
  * The v-side of the loop (dA = dv @ v^T, dvb = A @ dv -> dv2, db), wy2's `dA += dw @ kg^T`,
    and the dA chain (mask/beta, two [64,64] dots) are a separate kernel: in the fused
    kernel they cost more (1.3ms + 0.95ms, both latency chains at 2 CTAs/SM) than the
    ~0.9ms the side kernel takes at 4 CTAs/SM. The main kernel parks -dw^T (bf16, exactly
    wy2's b_dw) and kg = k*exp2(g) (bf16, from its strips) in two [B,T,HV,K] buffers for it,
    so the side kernel never touches g or exp2 (a first cut that recomputed kg there held
    the fp32 g tile at 248 regs and cost 0.5ms); the side kernel finalizes db IN PLACE on
    the main kernel's partial and writes dA, so it must run after it and is never autotuned.

Numerics vs wy2: dq/dk/dv2 bit-identical, dg/db ~1e-6 (reassociated rowsums). dA is
accumulated the way wy2 does it — dv @ v^T then dw @ kg^T into ONE tmem accumulator —
because the first cut (dw @ kg^T in the main kernel, IEEE-added to the side's partial)
flipped bf16 roundings in the chain and put prod-NT dk 30x over dbg_bwd's budget.
Varlen is not plumbed; off the supported shape (K=128, V%64, BT=64) or below the CTA floor
the dispatcher falls back to bwd_wy's kernel, which is why the chain can always call this.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp2

from ..._common.support import MIN_CTAS
from .bwd_wy import chunk_kda_bwd_wy_dqkg_wide

# Pinned configs (probed on b300 at prod8192, the ladder's NOTES.md): BV=32/3 stages/8
# warps/maxnreg=128 for the main kernel (84-92KB smem, 128 regs, 0 spills -> 2 CTAs/SM);
# KS=32 strips (64 spills at 128 regs, 16 no faster). The side kernel is occupancy-bound
# (tmem 128 -> at most 4 CTAs/SM): 8 warps at maxnreg=80 (10 spills, 3 CTAs/SM) 1.12ms beat
# 4 warps uncapped (255 regs, 2 CTAs) 1.27 and 8 warps at 64 (18 spills, 4 CTAs) 1.17; BV=32
# and KD=32 lost. The ladder reads these from KDA005_* env knobs; here they are constants.
MAIN_BV = 32
MAIN_KS = 32
MAIN_STAGES = 3
SIDE_BV = 64
SIDE_WARPS = 8
SIDE_MAXNREG = 80
SIDE_KD = 64  # K-slice of the side kernel's dA dot


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_t_main(
    q, k, v_new, g, beta, A, h, do, dh, dq, dk, dv, dg, db, dw_buf, kg_buf, dgk_buf, scale, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr, KS: tl.constexpr, STATE_V_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    i_last = (min(T, i_t * BT + BT) - 1).to(tl.int64)
    m_last = o_t == i_last

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K*V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K*V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dw_buf += (bos * HV + i_hv) * K
    kg_buf += (bos * HV + i_hv) * K
    dgk_buf += (i_tg * HV + i_hv) * K

    p_beta = beta + o_t * HV
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)
    o_A = tl.arange(0, BT)
    # b_At[t, a] = A_mem[t, a]: the B operand of dkgb^T = dw^T @ A^T (wy2's dot(A, dw)
    # transposed; wy2's b_A is A_mem^T).
    p_At = A + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_At = tl.load(p_At, mask=m_t[:, None] & (o_A[None, :] < BT), other=0.0)

    o_k = tl.arange(0, K)
    m_kt = (o_k < K)[:, None] & m_t[None, :]

    # [do; dv] stacked along rows so one M=K dot yields [dq^T | dw^T]
    o_n = tl.arange(0, 2 * BT)
    o_tn = i_t * BT + (o_n % BT)
    m_tn = o_tn < T
    acc_qw = tl.zeros([K, 2 * BT], dtype=tl.float32)
    acc_k = tl.zeros([K, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)   # strips' dkgb.kg colsum; the side kernel adds dvb.v
    dgk = tl.zeros([K], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        if STATE_V_FIRST:
            p_hT = h + o_k[:, None] + o_v[None, :] * K
            p_dhT = dh + o_k[:, None] + o_v[None, :] * K
        else:
            p_hT = h + o_k[:, None] * V + o_v[None, :]
            p_dhT = dh + o_k[:, None] * V + o_v[None, :]
        b_hT = tl.load(p_hT, mask=m_v[None, :], other=0.0)
        b_dhT = tl.load(p_dhT, mask=m_v[None, :], other=0.0)
        p_dodv = tl.where((o_n < BT)[:, None], do + o_tn[:, None] * (HV*V) + o_v[None, :],
                          dv + o_tn[:, None] * (HV*V) + o_v[None, :])
        b_dodv = tl.load(p_dodv, mask=m_tn[:, None] & m_v[None, :], other=0.0)
        p_vn = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
        b_vn = tl.load(p_vn, mask=m_tv, other=0.0)

        dgk += tl.sum(b_hT * b_dhT, axis=1)
        acc_qw += tl.dot(b_hT, tl.trans(b_dodv))
        acc_k += tl.dot(b_dhT, tl.trans(b_vn))

    # Park dq^T/dk^T raw (fp32) in dq/dk, -dw^T (bf16, = wy2's b_dw) in dw_buf for the side
    # kernel's dA dot, and dkgb^T = dw^T @ A^T (the one dot here, M=128) raw in dg.
    acc_qw3 = tl.permute(tl.reshape(acc_qw, [K, 2, BT]), (0, 2, 1))
    dqT_raw, dwT_raw = tl.split(acc_qw3)
    tl.store(dq + o_t[None, :] * (HV*K) + o_k[:, None], dqT_raw, mask=m_kt)
    tl.store(dk + o_t[None, :] * (HV*K) + o_k[:, None], acc_k, mask=m_kt)
    tl.store(dgk_buf + o_k, dgk)
    b_dwT = -dwT_raw.to(b_At.dtype)
    tl.store(dw_buf + o_t[None, :] * (HV*K) + o_k[:, None], b_dwT, mask=m_kt)
    b_dkgbT = tl.dot(b_dwT, b_At)
    tl.store(dg + o_t[None, :] * (HV*K) + o_k[:, None], b_dkgbT, mask=m_kt)
    tl.debug_barrier()

    # K-strip epilogue over the parked raw tiles ([KS, BT], k contiguous). Dot-free: every
    # strip is loads -> elementwise -> stores, so the live set stays under 128 regs.
    for s in tl.range(0, K // KS, num_stages=1):
        o_ks = s * KS + tl.arange(0, KS)
        m_kst = (o_ks < K)[:, None] & m_t[None, :]
        off_tk = o_t[None, :] * (HV*K) + o_ks[:, None]
        b_gn = tl.load(g + i_last * HV*K + o_ks).to(tl.float32)
        b_gT = tl.load(g + off_tk, mask=m_kst, other=0.0).to(tl.float32)
        b_kT = tl.load(k + o_t[None, :] * (H*K) + o_ks[:, None], mask=m_kst, other=0.0)
        b_qT = tl.load(q + o_t[None, :] * (H*K) + o_ks[:, None], mask=m_kst, other=0.0)
        b_dqT = tl.load(dq + off_tk, mask=m_kst, other=0.0)
        b_dkT = tl.load(dk + off_tk, mask=m_kst, other=0.0)
        b_dkgb_s = tl.load(dg + off_tk, mask=m_kst, other=0.0)
        b_dgk = tl.load(dgk_buf + o_ks)

        # wy2's epilogue, transposed ([k, t] tiles), same one-sided exp2 forms
        b_e = exp2(b_gT)
        b_dqT = b_dqT * b_e * scale
        b_dkT = b_dkT * tl.where(m_t[None, :], exp2(b_gn[:, None] - b_gT), 0)
        b_kg_s = b_kT * b_e
        tl.store(kg_buf + off_tk, b_kg_s.to(kg_buf.dtype.element_ty), mask=m_kst)  # side's dA operand
        b_db += tl.sum(b_dkgb_s * b_kg_s, 0)
        b_kdkT = b_kT * b_dkT
        b_dgk = b_dgk * exp2(b_gn) + tl.sum(b_kdkT, axis=1)
        b_dgT = (b_qT * b_dqT - b_kdkT + m_last[None, :] * b_dgk[:, None]
                 + b_kg_s * b_dkgb_s * b_beta[None, :])
        b_dkT = b_dkT + b_dkgb_s * (b_e * b_beta[None, :])
        tl.store(dq + off_tk, b_dqT.to(dq.dtype.element_ty), mask=m_kst)
        tl.store(dk + off_tk, b_dkT.to(dk.dtype.element_ty), mask=m_kst)
        tl.store(dg + off_tk, b_dgT.to(dg.dtype.element_ty), mask=m_kst)

    tl.store(db + o_t * HV, b_db.to(db.dtype.element_ty), mask=m_t)   # partial; side adds


@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_t_side(
    v, beta, A, dv, dw_buf, kg_buf, dv2, db, dA, T,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    KD: tl.constexpr,
):
    """The v-side of wy2's loop plus its dA path: dv2 = (A @ dv)*beta,
    db += rowsum((A @ dv) * v), dA = chain(dv @ v^T + dw @ kg^T) with dw and kg read from
    the main kernel's buffers and the two dots accumulated in wy2's order into one
    accumulator. db is finalized IN PLACE on the main kernel's partial: launch after it,
    never autotune it."""
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    bos = (i_b * T).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    dw_buf += (bos * HV + i_hv) * K
    kg_buf += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    dA += (bos * HV + i_hv) * BT
    db += bos * HV + i_hv
    b_beta = tl.load(beta + o_t * HV, mask=m_t, other=0.0)
    p_A = A + o_A[:, None] + o_t[None, :] * (HV * BT)      # wy2's b_A = A_mem^T
    b_A = tl.load(p_A, mask=(o_A[:, None] < BT) & m_t[None, :], other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        b_v = tl.load(v + o_t[:, None] * (HV*V) + o_v[None, :], mask=m_tv, other=0.0)
        b_dv = tl.load(dv + o_t[:, None] * (HV*V) + o_v[None, :], mask=m_tv, other=0.0)
        b_dA += tl.dot(b_dv, tl.trans(b_v))
        b_dvb = tl.dot(b_A, b_dv)
        b_db += tl.sum(b_dvb * b_v, 1)
        p_dv2 = dv2 + o_t[:, None] * (HV*V) + o_v[None, :]
        tl.store(p_dv2, (b_dvb * b_beta[:, None]).to(p_dv2.dtype.element_ty), mask=m_tv)
    # wy2's epilogue dot, into the same accumulator: b_dw / b_kg are wy2's -dw.bf16 and
    # kg.bf16 bit-for-bit (the transposed loop's dw matched wy2's), so dA's chain sees
    # wy2's fp32 sum. In K-slices of KD so the operands never sit in registers whole
    # (hoisting the full [64,128] pair above the loop cost 2 CTAs/SM).
    for s in tl.static_range(K // KD):
        o_k = s * KD + tl.arange(0, KD)
        m_tk = m_t[:, None] & (o_k < K)[None, :]
        b_dw = tl.load(dw_buf + o_t[:, None] * (HV*K) + o_k[None, :], mask=m_tk, other=0.0)
        b_kg = tl.load(kg_buf + o_t[:, None] * (HV*K) + o_k[None, :], mask=m_tk, other=0.0)
        b_dA += tl.dot(b_dw, tl.trans(b_kg))
    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
    p_db = db + o_t * HV
    b_db += tl.load(p_db, mask=m_t, other=0.0)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


def _supported(K: int, V: int, BT: int, ctas: int) -> bool:
    """M=K=128 tcgen05 tiles need K=128; V must tile by both BVs; below the CTA floor a
    grid-starved launch loses to wy2's autotuned kernel, so those shapes stay on it."""
    if K != 128 or V % max(MAIN_BV, SIDE_BV) != 0 or BT != 64:
        return False
    return ctas >= MIN_CTAS


def chunk_kda_bwd_wy_dqkg_t(
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
    chunk_size: int = 64,
):
    """Drop-in for bwd_wy.chunk_kda_bwd_wy_dqkg_wide: returns (dq, dk, dv2, db, dg, dA).

    Off the supported shape or below the CTA floor it IS that function."""
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    if not _supported(K, V, BT, NT * B * HV):
        return chunk_kda_bwd_wy_dqkg_wide(
            q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h, do=do, dh=dh, dv=dv,
            scale=scale, state_v_first=state_v_first, chunk_size=chunk_size,
        )
    if scale is None:
        scale = K ** -0.5

    # dq, dk are allocated at HV; the caller reduces to H if GVA. dq/dk/dg double as the
    # parking space for the raw transposed accumulators (fp32, overwritten in place).
    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dg = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    db = torch.empty(B, T, HV, device=g.device, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    dgk_buf = torch.empty(B, NT, HV, K, device=g.device, dtype=torch.float)
    # main -> side hand-off: -dw.bf16 and kg.bf16, the dA dot's operands
    dw_buf = torch.empty(B, T, HV, K, device=g.device, dtype=A.dtype)
    kg_buf = torch.empty(B, T, HV, K, device=g.device, dtype=A.dtype)

    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_wy_t_main[grid](
        q, k, v_new, g, beta, A, h, do, dh, dq, dk, dv, dg, db, dw_buf, kg_buf, dgk_buf, scale, T,
        H=H, HV=HV, K=K, V=V, BT=BT, BV=MAIN_BV, KS=MAIN_KS, STATE_V_FIRST=state_v_first,
        num_warps=8, num_stages=MAIN_STAGES, maxnreg=128,
    )
    chunk_kda_bwd_kernel_wy_t_side[grid](
        v, beta, A, dv, dw_buf, kg_buf, dv2, db, dA, T,
        HV=HV, K=K, V=V, BT=BT, BV=SIDE_BV, KD=SIDE_KD,
        num_warps=SIDE_WARPS, num_stages=3, maxnreg=SIDE_MAXNREG,
    )
    return dq, dk, dv2, db, dg, dA
