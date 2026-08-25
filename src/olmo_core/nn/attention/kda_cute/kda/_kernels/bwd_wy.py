"""Phase 2 B2b — fla 0.5.2's `chunk_kda_bwd_wy_dqkg_fused`, restructured to run at full K.

fla's grid is already one CTA per (chunk, b·hv); the waste is entirely inside the CTA,
in `for i_k in range(cdiv(K, BK))`. Slabbing K costs three separate things:

  1. `v_new`, `do` and `dv` are loaded in the i_v loop, which sits INSIDE the i_k loop,
     so each is re-read NK times per CTA (h/dh/q/k/g/A are read once). At prod8192 that
     is +3.2GB at BK=64 (+9.7GB at BK=32) on top of a ~15.5GB useful footprint.
  2. Every inner dot is [BT,BV] @ [BV,BK] with BK ∈ {32,64} — a tiny N, three of them
     chained on the same two operands. B1's probe already showed this stage is
     latency-bound, not dot-bound (deleting one of the three dots bought 0.12ms of 6.15).
  3. `tl.debug_barrier()` — fla marks it DO NOT REMOVE — is a full CTA barrier executed
     NK·NV times, which is there only because the `if i_k == 0` block reuses the loop's
     smem. It defeats whatever `num_stages` was going to pipeline.

So: BK = K. The i_k loop is gone, `h`/`dh` load as [BV, K], the V-shaped tiles are read
ONCE, the three dots widen to [BT,BV] @ [BV,K], the `i_k == 0` guard disappears (and with
it the reason for the barrier), and the whole exp2/dg/dA epilogue runs once instead of NK
times. Accumulators: dq/dk/dw [BT,K] fp32 + dA [BT,BT] = 448 of 512 tmem cols on Blackwell.

Numerics vs fla: the accumulation order over V is unchanged (i_v ascending, fp32 acc).
What moves is the reduction blocking inside each dot and `b_dA += dot(b_dw, kg^T)`, which
was NK partial sums over BK-wide slices and is now one sum over K — fp32 reassociations of
the same terms, no new exp2 factorization and no gate rewrite. Every store is idempotent
(the autotuner re-runs the kernel once per config trial, and any read-modify-write tensor
would corrupt on the first call per key — the NOTES-002 CachedAutotuner trap).

Varlen is not plumbed (the staged chain never passes cu_seqlens); off the supported shape
or below the CTA floor the dispatcher falls back to fla's fused kernel.
"""

from __future__ import annotations


import torch
import triton
import triton.language as tl

from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs, check_shared_mem

# Measured on b300 (dbg_wysweep.py, prod8192): BV=128 blows smem at num_stages>=2 and
# never wins at 1; num_warps 16/32 spill ~2KB/thread and run 40-80x slower, so 8 warps is
# the ceiling. Everything viable lands within ~1% of 4.95ms, which is the register wall,
# not a tiling choice — keep the space small so autotune trials stay cheap.
BV_LIST = [32, 64] if check_shared_mem('ampere') else [16, 32]

# Serial-ish per-CTA kernels want a full GPU before they beat fla's autotuned one; below
# this many CTAs the launch is grid-starved and the comparison is noise (the same floor
# the other 002 stages use). KDA002_WY=triton forces past it so dbg-sized shapes still
# exercise this kernel.
_MIN_CTAS = 256


@fla_cache_autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in BV_LIST
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'HV', 'K', 'V', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dqkg_wide(
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
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    PROBE_SKIP: tl.constexpr = 0,
):
    # PROBE_SKIP is an attribution knob for dbg_wysweep.py only (the KDA002C_SKIP pattern
    # from the CuTe intra kernel): 1 drops the dgk elementwise reduction, 2 drops the WY
    # block, 3 drops the three main dots. Any nonzero value produces WRONG output — it
    # exists to price a piece, not to run. The shipping path always passes 0.
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
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

    # K is the full head dim here — one slab, no i_k loop.
    o_k = tl.arange(0, K)
    m_tk = m_t[:, None] & (o_k < K)[None, :]

    p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
    p_g = g + o_t[:, None] * (HV*K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

    p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV*K + o_k
    b_gn = tl.load(p_gn, mask=o_k < K, other=0).to(tl.float32)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_dq = tl.zeros([BT, K], dtype=tl.float32)
    b_dk = tl.zeros([BT, K], dtype=tl.float32)
    b_dw = tl.zeros([BT, K], dtype=tl.float32)
    b_dgk = tl.zeros([K], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        m_h = (o_v[:, None] < V) & (o_k < K)[None, :]
        p_v_new = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            p_dh = dh + o_v[:, None] * K + o_k[None, :]
        else:
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
        p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * (HV*V) + o_v[None, :]
        # [BT, BV] — each read ONCE now, not once per K slab
        b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_dv = tl.load(p_dv, mask=m_tv, other=0.0)
        # [BV, K]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_h, other=0.0)

        if PROBE_SKIP != 1:
            b_dgk += tl.sum(b_h * b_dh, axis=0)
        if PROBE_SKIP != 3:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))

        # fla ran this block under `if i_k == 0` to keep it from repeating per K slab;
        # with the slab loop gone it is just the v-loop body, and the debug_barrier that
        # guarded the smem reuse across that branch goes with it.
        if PROBE_SKIP != 2:
            b_dA += tl.dot(b_dv, tl.trans(b_v))
            b_dvb = tl.dot(b_A, b_dv)
            b_dv2 = b_dvb * b_beta[:, None]
            b_db += tl.sum(b_dvb * b_v, 1)
            tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), mask=m_tv)

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

    p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_kdk = b_k * b_dk
    b_dgk += tl.sum(b_kdk, axis=0)
    b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
    b_dk = b_dk + b_dkgb * b_gb

    p_dq = dq + o_t[:, None] * (HV*K) + o_k[None, :]
    p_dk = dk + o_t[:, None] * (HV*K) + o_k[None, :]
    p_dg = dg + o_t[:, None] * (HV*K) + o_k[None, :]
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


def _supported(K: int, V: int, BT: int, ctas: int) -> bool:
    """Full-K tiles need K to be a power of two `tl.arange` can produce and small enough
    that dq/dk/dw fit tmem; V must tile by the smallest configured BV. The CTA floor keeps
    grid-starved shapes on fla, where they are genuinely faster."""
    return (
        K in (32, 64, 128)
        and V % min(BV_LIST) == 0
        and BT == 64
        and ctas >= _MIN_CTAS
    )


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
    chunk_size: int = 64,
):
    B, T, H, K, HV, V = *k.shape, v.shape[2], v.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    # (The tmem CuTe port of this stage lives in the research ladder and is NOT shipped:
    # it is correct but measured 6.93ms against this kernel's 4.98 at prod8192.)
    if state_v_first or not _supported(K, V, BT, NT * B * HV):
        from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused

        return chunk_kda_bwd_wy_dqkg_fused(
            q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h,
            do=do, dh=dh, dv=dv, scale=scale, state_v_first=state_v_first,
            chunk_size=chunk_size,
        )

    # dq, dk are allocated at HV; the caller reduces to H if GVA. All outputs are fresh
    # (never read-modify-write — autotune trials re-run the kernel).
    dq = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_wy_dqkg_wide[grid](
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
        STATE_V_FIRST=state_v_first,
        PROBE_SKIP=0,  # explicit: the attribution knob is never on in the shipping path
    )
    return dq, dk, dv2, db, dg, dA
