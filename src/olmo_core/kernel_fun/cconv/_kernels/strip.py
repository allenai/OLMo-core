"""The strip kernels (ladder idea cconv/001) — channel-strip, time-sequential, one pass
each direction.

Layout of the work (both kernels):

    program = (channel strip of BD, time segment of TS rows, batch row)
    thread  = VEC contiguous channels (VEC = BD / (32*num_warps): 4 -> 8B, 8 -> 16B vectors),
              looping over the segment's rows in groups of G

Because each thread walks TIME sequentially for a fixed set of channels, the W-1 previous
rows of x (and, in the backward, of g = dy * silu'(z)) live in registers as a ring buffer
that rotates by assignment. That is what makes a width-4 conv cost ONE load of x, ONE
sigmoid and ONE store per element instead of fla's W shifted tile loads and W sigmoids
(kernels.py, `for i_w in tl.static_range(0, W)`), and it is what removes the backward's
forward re-run: z is recomputed in-register from the x ring the dw accumulation already
needs.

What actually bounds a kernel like this on a B300 (dbg_bw.py, dbg_asm.py, 2026-09-01):

- the strip access pattern itself streams at 6.2-6.35 TB/s — the same as a flat copy — so
  the tiling is not the problem;
- bytes in flight per SM are. A stream needs ~50-60 KB in flight per SM; in-flight rows live
  in registers, and so does the per-thread state (taps, rings, dw accumulators), so
  registers/thread cap warps/SM and warps x rows-in-flight x bytes/row is the bandwidth.
  The first cut (VEC=8, G=4) compiled to 102 regs fwd / 218 bwd -> 20 / 9 warps per SM ->
  ~40 KB in flight -> 3.9 / 3.4 TB/s. Hence VEC=4 (halves the state) and G=8 (doubles the
  rows in flight) in the autotune space.

The ring is written for WP=4 taps (every ladder config). Narrower W is handled by
zero-weighting the missing taps — checked by the `w2` case — and W>4 is refused.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

WP = tl.constexpr(4)  # padded tap count the register ring is written for; W <= WP


def _configs():
    # BD = VEC * 32 * num_warps; every thread holds exactly one VEC-wide vector per row.
    # maxnreg=128 on the VEC=4 backward: ptxas trades a handful of spills for double the
    # warps per SM, and bytes in flight is what this kernel is short of (NOTES.md). On VEC=8
    # it spills by the hundreds, so only the VEC=4 configs carry it.
    cfgs = [
        triton.Config({"BD": vec * 32 * nw, "G": g}, num_warps=nw)
        for vec in (4, 8)
        for nw in (1, 2, 4)
        for g in (4, 8)
    ]
    cfgs += [
        triton.Config({"BD": 4 * 32 * nw, "G": g}, num_warps=nw, maxnreg=128)
        for nw in (1, 2, 4)
        for g in (4, 8)
    ]
    return cfgs


# --------------------------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------------------------


@triton.jit
def _tap(w, o_d, m_d, W: tl.constexpr, k: tl.constexpr):
    """w[:, W-1-k] as fp32 [BD] — the weight multiplying x[t-k]. Zero for a padded tap."""
    if k < W:
        return tl.load(w + o_d * W + (W - 1 - k), mask=m_d, other=0.0).to(tl.float32)
    else:
        return tl.zeros_like(o_d).to(tl.float32)


@triton.jit
def _sigmoid(z):
    # sigmoid(z) = 0.5 * tanh(0.5 z) + 0.5: two FMAs and ONE MUFU op. `tl.sigmoid` lowers to a
    # libdevice expf plus an IEEE divide — 20-30 instructions against a per-element budget of
    # ~30 (fwd) / ~20 (bwd) at B300 bandwidth. tanh.approx.f32 has ~2^-11 relative error, far
    # inside the 5e-3 / 8e-3 budgets on bf16 outputs; the err_ratio column is the check.
    th = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=r,r", [0.5 * z], dtype=tl.float32, is_pure=True, pack=1
    )
    return 0.5 * th + 0.5


@triton.jit
def _silu(z):
    return z * _sigmoid(z)


@triton.jit
def _dsilu(z, dy):
    s = _sigmoid(z)
    return dy * s * (1.0 + z * (1.0 - s))


@triton.jit
def _ld(p, tt, st, m_d, T, MASK_D: tl.constexpr, MASK_T: tl.constexpr):
    """One row, still in its storage dtype: converting at the use site keeps the in-flight
    rows at half the registers. Masks are constexpr-selected so the steady state has none."""
    if MASK_D and MASK_T:
        return tl.load(p + tt.to(tl.int64) * st, mask=m_d & (tt < T), other=0.0)
    elif MASK_D:
        return tl.load(p + tt.to(tl.int64) * st, mask=m_d, other=0.0)
    elif MASK_T:
        return tl.load(p + tt.to(tl.int64) * st, mask=m_d & (tt < T), other=0.0)
    else:
        return tl.load(p + tt.to(tl.int64) * st)


@triton.jit
def _ld_ring(p, tt, st, m_d, T):
    """A row left of the segment start: zero left of the sequence."""
    return tl.load(p + tt.to(tl.int64) * st, mask=m_d & (tt >= 0) & (tt < T), other=0.0).to(tl.float32)


@triton.jit
def _f(v):
    return v.to(tl.float32)


@triton.jit
def _st(p, tt, st, v, m, EVEN: tl.constexpr):
    o = tl.cast(v, p.dtype.element_ty, fp_downcast_rounding="rtne")
    if EVEN:
        tl.store(p + tt.to(tl.int64) * st, o)
    else:
        tl.store(p + tt.to(tl.int64) * st, o, mask=m)


@triton.jit
def _fwd4(py, t, syt, xa, xb, xc, xd, xm1, xm2, xm3, w0, w1, w2, w3, m_d, T, EVEN: tl.constexpr):
    """Four consecutive rows t..t+3 given their (already loaded) x and the ring behind them.
    Returns the ring for row t+4."""
    xa = _f(xa)
    xb = _f(xb)
    xc = _f(xc)
    xd = _f(xd)
    ya = _silu(w0 * xa + w1 * xm1 + w2 * xm2 + w3 * xm3)
    yb = _silu(w0 * xb + w1 * xa + w2 * xm1 + w3 * xm2)
    yc = _silu(w0 * xc + w1 * xb + w2 * xa + w3 * xm1)
    yd = _silu(w0 * xd + w1 * xc + w2 * xb + w3 * xa)
    _st(py, t, syt, ya, m_d & (t < T), EVEN)
    _st(py, t + 1, syt, yb, m_d & (t + 1 < T), EVEN)
    _st(py, t + 2, syt, yc, m_d & (t + 2 < T), EVEN)
    _st(py, t + 3, syt, yd, m_d & (t + 3 < T), EVEN)
    return xd, xc, xb


@triton.jit
def _bwd4(
    pdx, t, sdt, t0,
    xa, xb, xc, xd, da, db, dc, dd,
    xm1, xm2, xm3, gm1, gm2, gm3,
    dw0, dw1, dw2, dw3,
    w0, w1, w2, w3, m_d, T,
    HEAD: tl.constexpr, EVEN: tl.constexpr,
):
    """Rows t..t+3 of the backward: g for each, dw from this group's g, and dx for rows
    t-3..t (dx[u] = sum_k w'[k] g[u+k], so a row's dx is final three rows later). HEAD marks
    a group that may be the segment's first, whose dx rows t0-3..t0-1 belong to the previous
    segment and are masked off."""
    xa = _f(xa)
    xb = _f(xb)
    xc = _f(xc)
    xd = _f(xd)
    ga = _dsilu(w0 * xa + w1 * xm1 + w2 * xm2 + w3 * xm3, _f(da))
    gb = _dsilu(w0 * xb + w1 * xa + w2 * xm1 + w3 * xm2, _f(db))
    gc = _dsilu(w0 * xc + w1 * xb + w2 * xa + w3 * xm1, _f(dc))
    gd = _dsilu(w0 * xd + w1 * xc + w2 * xb + w3 * xa, _f(dd))
    dw0 += ga * xa + gb * xb + gc * xc + gd * xd
    dw1 += ga * xm1 + gb * xa + gc * xb + gd * xc
    dw2 += ga * xm2 + gb * xm1 + gc * xa + gd * xb
    dw3 += ga * xm3 + gb * xm2 + gc * xm1 + gd * xa
    r = t - 3
    if HEAD:
        _st(pdx, r, sdt, w3 * ga + w2 * gm1 + w1 * gm2 + w0 * gm3, m_d & (r >= t0) & (r < T), False)
        _st(pdx, r + 1, sdt, w3 * gb + w2 * ga + w1 * gm1 + w0 * gm2, m_d & (r + 1 >= t0) & (r + 1 < T), False)
        _st(pdx, r + 2, sdt, w3 * gc + w2 * gb + w1 * ga + w0 * gm1, m_d & (r + 2 >= t0) & (r + 2 < T), False)
    else:
        _st(pdx, r, sdt, w3 * ga + w2 * gm1 + w1 * gm2 + w0 * gm3, m_d & (r < T), EVEN)
        _st(pdx, r + 1, sdt, w3 * gb + w2 * ga + w1 * gm1 + w0 * gm2, m_d & (r + 1 < T), EVEN)
        _st(pdx, r + 2, sdt, w3 * gc + w2 * gb + w1 * ga + w0 * gm1, m_d & (r + 2 < T), EVEN)
    _st(pdx, r + 3, sdt, w3 * gd + w2 * gc + w1 * gb + w0 * ga, m_d & (r + 3 < T), EVEN)
    return xd, xc, xb, gd, gc, gb, dw0, dw1, dw2, dw3


# --------------------------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------------------------
#
# Row groups: see the comment on the loop. The masked path is taken only by a group that can
# run past T (the last group of a ragged final segment).


@triton.autotune(configs=_configs(), key=["D", "W", "B", "T"])
@triton.jit
def cconv_fwd_strip(
    x, y, w,
    B, T, TS,
    sxn, sxt, sxd,
    syn, syt,
    D: tl.constexpr,
    W: tl.constexpr,
    EVEN_T: tl.constexpr,
    BD: tl.constexpr,
    G: tl.constexpr,
):
    i_d, i_s, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    t0 = i_s * TS
    t1 = t0 + TS
    MD: tl.constexpr = D % BD != 0  # channel mask needed at all
    EVEN: tl.constexpr = EVEN_T and not MD  # no store of this program can run past T or D

    px = x + i_b.to(tl.int64) * sxn + o_d * sxd
    py = y + i_b.to(tl.int64) * syn + o_d

    w0 = _tap(w, o_d, m_d, W, 0)
    w1 = _tap(w, o_d, m_d, W, 1)
    w2 = _tap(w, o_d, m_d, W, 2)
    w3 = _tap(w, o_d, m_d, W, 3)

    # the ring: x[t-1], x[t-2], x[t-3] (zeros left of the sequence)
    xm1 = _ld_ring(px, t0 - 1, sxt, m_d, T)
    xm2 = _ld_ring(px, t0 - 2, sxt, m_d, T)
    xm3 = _ld_ring(px, t0 - 3, sxt, m_d, T)

    # Rows in groups of G: every load of a group is issued before any of its stores (Triton
    # cannot prove x and y do not alias, so a load after a store in program order can never be
    # hoisted above it). A software-prefetch variant — next group's loads before this group's
    # math — was measured slower at every config (NOTES.md, 2026-09-01): it doubles the
    # registers the in-flight rows cost, and registers are what bound this kernel.
    # MT: rows can run past T only in a ragged final segment (T % TS != 0), and then every
    # program masks — a runtime "does this group run past T" branch was measured to cost
    # registers (168 vs 122 on the bwd) and speed.
    MT: tl.constexpr = not EVEN_T
    for t in range(t0, t1, G):
        xa = _ld(px, t, sxt, m_d, T, MD, MT)
        xb = _ld(px, t + 1, sxt, m_d, T, MD, MT)
        xc = _ld(px, t + 2, sxt, m_d, T, MD, MT)
        xd = _ld(px, t + 3, sxt, m_d, T, MD, MT)
        if G == 8:
            xe = _ld(px, t + 4, sxt, m_d, T, MD, MT)
            xf = _ld(px, t + 5, sxt, m_d, T, MD, MT)
            xg = _ld(px, t + 6, sxt, m_d, T, MD, MT)
            xh = _ld(px, t + 7, sxt, m_d, T, MD, MT)
        xm1, xm2, xm3 = _fwd4(py, t, syt, xa, xb, xc, xd, xm1, xm2, xm3, w0, w1, w2, w3, m_d, T, EVEN)
        if G == 8:
            xm1, xm2, xm3 = _fwd4(py, t + 4, syt, xe, xf, xg, xh, xm1, xm2, xm3, w0, w1, w2, w3, m_d, T, EVEN)


# --------------------------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------------------------


@triton.autotune(configs=_configs(), key=["D", "W", "B", "T"])
@triton.jit
def cconv_bwd_strip(
    x, dy, dx, dwp, w,
    B, T, TS, NS,
    sxn, sxt, sxd,
    syn, syt, syd,
    sdn, sdt,
    D: tl.constexpr,
    W: tl.constexpr,
    EVEN_T: tl.constexpr,
    BD: tl.constexpr,
    G: tl.constexpr,
):
    i_d, i_s, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    t0 = i_s * TS
    t1 = t0 + TS
    MD: tl.constexpr = D % BD != 0
    EVEN: tl.constexpr = EVEN_T and not MD

    px = x + i_b.to(tl.int64) * sxn + o_d * sxd
    pdy = dy + i_b.to(tl.int64) * syn + o_d * syd
    pdx = dx + i_b.to(tl.int64) * sdn + o_d

    w0 = _tap(w, o_d, m_d, W, 0)
    w1 = _tap(w, o_d, m_d, W, 1)
    w2 = _tap(w, o_d, m_d, W, 2)
    w3 = _tap(w, o_d, m_d, W, 3)

    xm1 = _ld_ring(px, t0 - 1, sxt, m_d, T)
    xm2 = _ld_ring(px, t0 - 2, sxt, m_d, T)
    xm3 = _ld_ring(px, t0 - 3, sxt, m_d, T)
    # g ring: g[t-1], g[t-2], g[t-3]. Rows before t0 are the previous segment's business;
    # their dx is never stored from here, so zeros are fine.
    gm1 = tl.zeros([BD], dtype=tl.float32)
    gm2 = tl.zeros([BD], dtype=tl.float32)
    gm3 = tl.zeros([BD], dtype=tl.float32)

    dw0 = tl.zeros([BD], dtype=tl.float32)
    dw1 = tl.zeros([BD], dtype=tl.float32)
    dw2 = tl.zeros([BD], dtype=tl.float32)
    dw3 = tl.zeros([BD], dtype=tl.float32)

    # The loads sit in a runtime if/else on "can this group run past T". That branch is almost
    # never taken (a ragged final segment only), and it is NOT there for the masks: with the
    # loads in their own basic block ptxas issues all 16 before any of the group's math,
    # where the straight-line form interleaves them with the FMAs to save registers. Measured
    # at maxnreg=128: 0.327 ms with the branch vs 0.380 ms without (NOTES.md). Rows past T
    # load as zero; dy = 0 makes their g zero, which every masked-off dx row and the dw
    # accumulation rely on.
    for t in range(t0, t1, G):
        if t + G <= T:
            xa = _ld(px, t, sxt, m_d, T, MD, False)
            xb = _ld(px, t + 1, sxt, m_d, T, MD, False)
            xc = _ld(px, t + 2, sxt, m_d, T, MD, False)
            xd = _ld(px, t + 3, sxt, m_d, T, MD, False)
            da = _ld(pdy, t, syt, m_d, T, MD, False)
            db = _ld(pdy, t + 1, syt, m_d, T, MD, False)
            dc = _ld(pdy, t + 2, syt, m_d, T, MD, False)
            dd = _ld(pdy, t + 3, syt, m_d, T, MD, False)
            if G == 8:
                xe = _ld(px, t + 4, sxt, m_d, T, MD, False)
                xf = _ld(px, t + 5, sxt, m_d, T, MD, False)
                xg = _ld(px, t + 6, sxt, m_d, T, MD, False)
                xh = _ld(px, t + 7, sxt, m_d, T, MD, False)
                de = _ld(pdy, t + 4, syt, m_d, T, MD, False)
                df = _ld(pdy, t + 5, syt, m_d, T, MD, False)
                dg = _ld(pdy, t + 6, syt, m_d, T, MD, False)
                dh = _ld(pdy, t + 7, syt, m_d, T, MD, False)
        else:
            xa = _ld(px, t, sxt, m_d, T, MD, True)
            xb = _ld(px, t + 1, sxt, m_d, T, MD, True)
            xc = _ld(px, t + 2, sxt, m_d, T, MD, True)
            xd = _ld(px, t + 3, sxt, m_d, T, MD, True)
            da = _ld(pdy, t, syt, m_d, T, MD, True)
            db = _ld(pdy, t + 1, syt, m_d, T, MD, True)
            dc = _ld(pdy, t + 2, syt, m_d, T, MD, True)
            dd = _ld(pdy, t + 3, syt, m_d, T, MD, True)
            if G == 8:
                xe = _ld(px, t + 4, sxt, m_d, T, MD, True)
                xf = _ld(px, t + 5, sxt, m_d, T, MD, True)
                xg = _ld(px, t + 6, sxt, m_d, T, MD, True)
                xh = _ld(px, t + 7, sxt, m_d, T, MD, True)
                de = _ld(pdy, t + 4, syt, m_d, T, MD, True)
                df = _ld(pdy, t + 5, syt, m_d, T, MD, True)
                dg = _ld(pdy, t + 6, syt, m_d, T, MD, True)
                dh = _ld(pdy, t + 7, syt, m_d, T, MD, True)
        xm1, xm2, xm3, gm1, gm2, gm3, dw0, dw1, dw2, dw3 = _bwd4(
            pdx, t, sdt, t0, xa, xb, xc, xd, da, db, dc, dd, xm1, xm2, xm3, gm1, gm2, gm3,
            dw0, dw1, dw2, dw3, w0, w1, w2, w3, m_d, T, True, EVEN)
        if G == 8:
            xm1, xm2, xm3, gm1, gm2, gm3, dw0, dw1, dw2, dw3 = _bwd4(
                pdx, t + 4, sdt, t0, xe, xf, xg, xh, de, df, dg, dh, xm1, xm2, xm3, gm1, gm2, gm3,
                dw0, dw1, dw2, dw3, w0, w1, w2, w3, m_d, T, False, EVEN)

    # Halo: the next segment's first 3 rows of g finish this segment's last 3 dx rows. No dw
    # (those rows belong to the next segment's accumulator); stores only below t1.
    t = t1
    xa = _f(_ld(px, t, sxt, m_d, T, MD, True))
    xb = _f(_ld(px, t + 1, sxt, m_d, T, MD, True))
    xc = _f(_ld(px, t + 2, sxt, m_d, T, MD, True))
    da = _f(_ld(pdy, t, syt, m_d, T, MD, True))
    db = _f(_ld(pdy, t + 1, syt, m_d, T, MD, True))
    dc = _f(_ld(pdy, t + 2, syt, m_d, T, MD, True))
    ga = _dsilu(w0 * xa + w1 * xm1 + w2 * xm2 + w3 * xm3, da)
    gb = _dsilu(w0 * xb + w1 * xa + w2 * xm1 + w3 * xm2, db)
    gc = _dsilu(w0 * xc + w1 * xb + w2 * xa + w3 * xm1, dc)
    r = t - 3
    _st(pdx, r, sdt, w3 * ga + w2 * gm1 + w1 * gm2 + w0 * gm3, m_d & (r < T), False)
    _st(pdx, r + 1, sdt, w3 * gb + w2 * ga + w1 * gm1 + w0 * gm2, m_d & (r + 1 < T), False)
    _st(pdx, r + 2, sdt, w3 * gc + w2 * gb + w1 * ga + w0 * gm1, m_d & (r + 2 < T), False)

    # One fp32 partial per program: dwp[(i_b*NS + i_s), d, W-1-k] = dw_k.
    pp = dwp + (i_b.to(tl.int64) * NS + i_s) * (D * W) + o_d * W
    tl.store(pp + (W - 1), dw0, mask=m_d)
    if W >= 2:
        tl.store(pp + (W - 2), dw1, mask=m_d)
    if W >= 3:
        tl.store(pp + (W - 3), dw2, mask=m_d)
    if W >= 4:
        tl.store(pp + (W - 4), dw3, mask=m_d)


# --------------------------------------------------------------------------------------------
# host
# --------------------------------------------------------------------------------------------

# ~14 programs per B300 SM at one BD=1024 strip: enough rows in flight, short tail. The
# ladder sweeps this through CCONV_TARGET_PROGRAMS; here it is the measured constant.
TARGET_PROGRAMS = 2048
SEG_ALIGN = 8  # the largest G, so TS is a multiple of every group size
MIN_SEG = 64  # below this the backward's 3-row halo re-read is >5% of the segment


def _segment(B: int, T: int, D: int) -> int:
    """Rows per program. Smaller segments mean more programs (and more halo re-reads: the
    backward re-reads 3 rows per segment, ~2% at TS=128); larger ones mean fewer, longer
    loops. Sized so the grid lands near TARGET_PROGRAMS at one 1024-wide strip."""
    strips = max(1, D // 1024)
    ts = (B * T * strips) // TARGET_PROGRAMS
    ts = max(MIN_SEG, (ts // SEG_ALIGN) * SEG_ALIGN)
    return min(ts, triton.cdiv(T, SEG_ALIGN) * SEG_ALIGN)


def cconv_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    W = w.shape[1]
    assert w.shape[0] == D and W <= WP.value, f"weight {tuple(w.shape)}: this kernel handles W <= {WP.value}"
    assert w.is_contiguous()
    y = torch.empty((B, T, D), dtype=x.dtype, device=x.device)
    TS = _segment(B, T, D)
    NS = triton.cdiv(T, TS)

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NS, B)

    cconv_fwd_strip[grid](
        x, y, w,
        B, T, TS,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1),
        D=D, W=W, EVEN_T=(T % TS == 0),
    )
    return y


def cconv_bwd(x: torch.Tensor, w: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    W = w.shape[1]
    assert w.is_contiguous()
    dx = torch.empty((B, T, D), dtype=x.dtype, device=x.device)
    TS = _segment(B, T, D)
    NS = triton.cdiv(T, TS)
    dwp = torch.empty((B * NS, D, W), dtype=torch.float32, device=x.device)

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NS, B)

    cconv_bwd_strip[grid](
        x, dy, dx, dwp, w,
        B, T, TS, NS,
        x.stride(0), x.stride(1), x.stride(2),
        dy.stride(0), dy.stride(1), dy.stride(2),
        dx.stride(0), dx.stride(1),
        D=D, W=W, EVEN_T=(T % TS == 0),
    )
    # Deterministic: a fixed-shape reduction over B*NS fp32 partials, never atomics.
    dw = dwp.sum(0).to(w.dtype)
    return dx, dw
