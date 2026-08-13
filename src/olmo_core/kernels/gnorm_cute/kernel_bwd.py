"""The CuTe DSL backward for the gated RMS norm. See ALGORITHM.md.

One compiled callable, three kernel launches on the same stream (one host dispatch):

  1. `gnorm_cute_bwd`   — warp-per-row streaming pass over the (R, D) view: loads x, g, dy
                          (+ the forward's fp32 rstd), stores dx, dg, and accumulates dw in
                          registers; CTAs reduce their 8 warps' accumulators through smem
                          and write one [D] fp32 partial each.
  2. `gnorm_cute_dw_red1` — 32 CTAs fold the [num_ctas, D] partials to [32, D].
  3. `gnorm_cute_dw_red2` — 1 CTA folds [32, D] to the final dw [D] fp32.

The math is fla's hand-derived backward (fused_norm_gate.py at 0.5.2), per row in fp32:

    x̂  = x·rstd;  y = x̂·w;  s = sigmoid(g)
    swish:   dg = dy·y·(s + g·s·(1−s));  dy' = dy·g·s
    sigmoid: dg = dy·y·s·(1−s);          dy' = dy·s
    dw += dy'·x̂
    wdy = w·dy';  c1 = Σ_d(x̂·wdy)/D;  dx = (wdy − x̂·c1)·rstd

Everything 001 learned carries over: fastmath transcendentals only
(the sigmoid via 0.5+0.5·tanh(x/2, fastmath) — the precise forms made 001 compute-bound),
persistent block-stride grid, and the layout-keyed call cache with outputs (dx, dg, dw)
allocated fresh per call. The dw split is DETERMINISTIC: fixed [num_ctas, D] partials and
fixed-order tree folds, never atomics — same structure as fla's [NS, D] + sum(0), minus
the separate torch dispatch.
"""

from __future__ import annotations

import ctypes
from typing import Type

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator

# Rows per CTA = warps per CTA, matching kernel_fwd so the two kernels share the rstd view.
RPB = 8

# Grid cap: SMs * this. The bwd moves 5 streams against the fwd's 3 with ~2x the flops and
# a dw accumulator adding register pressure, so 001's 64/SM plateau does not transfer —
# and indeed it doesn't: dbg_bwd.py --sweep at prod8192 (B300, 2026-08-13) plateaus at
# 8-12/SM (0.823ms, 95.5% of stream) and degrades slowly above; 2/SM is 5232 GB/s. The
# smaller cap also keeps the dw partial buffer at ~1.2MB.
_CTAS_PER_SM = 8

# First-level fold width of the dw partial reduction (kernel 2's grid).
_RED_CTAS = 32


class GnormBwdKernel:
    """Backward of the gated RMS norm: one warp per row, plus the two dw fold kernels."""

    def __init__(self, io_dtype: Type[cutlass.Numeric], D: int, activation: str):
        assert D % 32 == 0 and D <= 512, f"one-warp-per-row mapping needs 32 | D, got {D}"
        assert activation in ("swish", "silu", "sigmoid")
        self.io_dtype = io_dtype
        self.D = D
        self.V = D // 32
        self.swish = activation in ("swish", "silu")

    @cute.jit
    def __call__(
        self,
        mDy: cute.Tensor,  # (R, D) io dtype
        mX: cute.Tensor,  # (R, D) io dtype
        mG: cute.Tensor,  # (R, D) io dtype
        mW: cute.Tensor,  # (D,) fp32
        mR: cute.Tensor,  # (R/RPB, RPB) fp32 — the forward's rstd
        mDx: cute.Tensor,  # (R, D) io dtype, out
        mDg: cute.Tensor,  # (R, D) io dtype, out
        mPart: cute.Tensor,  # (num_ctas, D) fp32 scratch
        mPart2: cute.Tensor,  # (_RED_CTAS, D) fp32 scratch
        mDw: cute.Tensor,  # (D,) fp32, out
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.gnorm_cute_bwd(mDy, mX, mG, mW, mR, mDx, mDg, mPart, num_ctas).launch(
            grid=(num_ctas, 1, 1), block=[32 * RPB, 1, 1], stream=stream
        )
        self.gnorm_cute_dw_red1(mPart, mPart2, num_ctas).launch(
            grid=(_RED_CTAS, 1, 1), block=[256, 1, 1], stream=stream
        )
        self.gnorm_cute_dw_red2(mPart2, mDw).launch(
            grid=(1, 1, 1), block=[256, 1, 1], stream=stream
        )

    @cute.kernel
    def gnorm_cute_bwd(
        self,
        mDy: cute.Tensor,
        mX: cute.Tensor,
        mG: cute.Tensor,
        mW: cute.Tensor,
        mR: cute.Tensor,
        mDx: cute.Tensor,
        mDg: cute.Tensor,
        mPart: cute.Tensor,
        num_ctas: cutlass.Int32,
    ):
        V, D = self.V, self.D
        f32 = cutlass.Float32

        smem = SmemAllocator()
        # 8 warps x D fp32 accumulator staging for the CTA's dw partial (8KB at D=256)
        sDw = smem.allocate_tensor(f32, cute.make_layout((RPB, D)), byte_alignment=16)

        bidx, _, _ = cute.arch.block_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        tidx, _, _ = cute.arch.thread_idx()

        gW = cute.local_tile(mW, (V,), (lane,))
        rW = cute.make_fragment_like(gW)
        cute.autovec_copy(gW, rW)
        wf = rW.load()

        # dw accumulates in registers across every row this warp touches
        rDwAcc = cute.make_fragment_like(gW)
        rDwAcc.fill(0.0)

        n_blocks = cute.size(mR, mode=[0])
        trips = (n_blocks - bidx + num_ctas - 1) // num_ctas
        for it in cutlass.range(trips):
            blk = bidx + it * num_ctas
            row = blk * RPB + wid

            gDy = cute.local_tile(mDy[row, None], (V,), (lane,))
            gX = cute.local_tile(mX[row, None], (V,), (lane,))
            gG = cute.local_tile(mG[row, None], (V,), (lane,))
            gDx = cute.local_tile(mDx[row, None], (V,), (lane,))
            gDg = cute.local_tile(mDg[row, None], (V,), (lane,))

            rDy = cute.make_fragment_like(gDy)
            rX = cute.make_fragment_like(gX)
            rG = cute.make_fragment_like(gG)
            cute.autovec_copy(gDy, rDy)
            cute.autovec_copy(gX, rX)
            cute.autovec_copy(gG, rG)

            rstd = mR[blk, wid]  # one fp32 per row; 32-lane broadcast of one address

            xh = rX.load().to(f32) * rstd
            dyf = rDy.load().to(f32)
            gf = rG.load().to(f32)
            y = xh * wf

            # sigmoid via fastmath tanh — see 001's lesson on precise-by-default math
            s = 0.5 + 0.5 * cute.math.tanh(gf * f32(0.5), fastmath=True)
            if cutlass.const_expr(self.swish):
                dgv = dyf * y * (s * (1.0 + gf * (1.0 - s)))
                dyp = dyf * gf * s
            else:
                dgv = dyf * y * (s * (1.0 - s))
                dyp = dyf * s

            rDwAcc.store(rDwAcc.load() + dyp * xh)

            wdy = wf * dyp
            partial = (xh * wdy).reduce(cute.ReductionOp.ADD, 0.0, 0)
            c1 = cute.arch.warp_reduction_sum(partial) * f32(1.0 / D)
            dxv = (wdy - xh * c1) * rstd

            rDx = cute.make_fragment_like(gDx)
            rDg = cute.make_fragment_like(gDg)
            rDx.store(dxv.to(self.io_dtype))
            rDg.store(dgv.to(self.io_dtype))
            cute.autovec_copy(rDx, gDx)
            cute.autovec_copy(rDg, gDg)

        # CTA dw partial: warps stage their lane accumulators, one barrier, then the CTA's
        # threads fold the 8 rows column-wise and write [D] fp32, coalesced.
        sDwLane = cute.local_tile(sDw[wid, None], (V,), (lane,))
        cute.autovec_copy(rDwAcc, sDwLane)
        cute.arch.sync_threads()

        for k in cutlass.range_constexpr((D + 255) // 256):
            col = tidx + k * 256
            if col < D:
                acc = f32(0.0)
                for w in cutlass.range_constexpr(RPB):
                    acc = acc + sDw[w, col]
                mPart[bidx, col] = acc

    @cute.kernel
    def gnorm_cute_dw_red1(
        self, mPart: cute.Tensor, mPart2: cute.Tensor, num_ctas: cutlass.Int32
    ):
        D = self.D
        f32 = cutlass.Float32
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        for k in cutlass.range_constexpr((D + 255) // 256):
            col = tidx + k * 256
            if col < D:
                acc = f32(0.0)
                trips = (num_ctas - bidx + _RED_CTAS - 1) // _RED_CTAS
                for i in cutlass.range(trips):
                    acc = acc + mPart[bidx + i * _RED_CTAS, col]
                mPart2[bidx, col] = acc

    @cute.kernel
    def gnorm_cute_dw_red2(self, mPart2: cute.Tensor, mDw: cute.Tensor):
        D = self.D
        f32 = cutlass.Float32
        tidx, _, _ = cute.arch.thread_idx()

        for k in cutlass.range_constexpr((D + 255) // 256):
            col = tidx + k * 256
            if col < D:
                acc = f32(0.0)
                for r in cutlass.range_constexpr(_RED_CTAS):
                    acc = acc + mPart2[r, col]
                mDw[col] = acc


# --------------------------------------------------------------------------------------------
# host wrapper — same cache discipline as kernel_fwd.py
# --------------------------------------------------------------------------------------------

_COMPILE_CACHE: dict = {}
_CALL_CACHE: dict = {}


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def _view2d(t: torch.Tensor):
    ct = from_dlpack(t.detach(), assumed_align=16)
    return ct.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))


def _out_specs(*tensors: torch.Tensor) -> tuple:
    return tuple((tuple(t.shape), t.dtype) for t in tensors)


def _alloc_outs(specs: tuple, device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


_IO_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def gnorm_cute_bwd(
    dy: torch.Tensor,  # [..., D], same shape/dtype as x
    x: torch.Tensor,  # [..., D] bf16/fp16, contiguous
    g: torch.Tensor,
    w: torch.Tensor,  # [D] fp32
    rstd: torch.Tensor,  # [R] fp32, from the forward
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, dg [x's shape 2D-flattened], dw [D] fp32)."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    g2 = g.reshape(-1, D)
    dy2 = dy.reshape(-1, D)
    R = x2.shape[0]
    assert x2.is_contiguous() and g2.is_contiguous() and dy2.is_contiguous()
    assert w.is_contiguous() and w.dtype == torch.float32 and w.shape == (D,)
    assert rstd.is_contiguous() and rstd.dtype == torch.float32 and rstd.shape == (R,)
    assert R % RPB == 0, f"row count {R} not a multiple of {RPB}"
    assert R // RPB >= _RED_CTAS, "dw fold assumes at least _RED_CTAS row-blocks"

    key = (
        (R, D), x.dtype, activation,
        torch.cuda.current_stream().cuda_stream,
    )
    ent = _CALL_CACHE.get(key)
    outs = None
    if ent is None:
        io_dtype = _IO_DTYPES[x.dtype]
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        num_ctas = min(R // RPB, sms * _CTAS_PER_SM)

        dx2 = torch.empty_like(x2)
        dg2 = torch.empty_like(g2)
        dw = torch.empty(D, device=x.device, dtype=torch.float32)
        # internal scratch, never escapes: cache-owned is fine
        part = torch.empty(num_ctas, D, device=x.device, dtype=torch.float32)
        part2 = torch.empty(_RED_CTAS, D, device=x.device, dtype=torch.float32)

        cdy = _view2d(dy2)
        cx = _view2d(x2)
        cg = _view2d(g2)
        cw = from_dlpack(w.detach(), assumed_align=16)
        cr = _view2d(rstd.view(R // RPB, RPB))
        cdx = _view2d(dx2)
        cdg = _view2d(dg2)
        cpart = _view2d(part)
        cpart2 = from_dlpack(part2, assumed_align=16)
        cdw = from_dlpack(dw, assumed_align=16)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        compile_key = (io_dtype, D, activation)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = GnormBwdKernel(io_dtype, D, activation)
            compiled = cute.compile(
                kernel_obj, cdy, cx, cg, cw, cr, cdx, cdg, cpart, cpart2, cdw,
                cutlass.Int32(num_ctas), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled

        args = (cdy, cx, cg, cw, cr, cdx, cdg, cpart, cpart2, cdw,
                cutlass.Int32(num_ctas), stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (dx2, dg2, dw)
        ent = (compiled, args, _out_specs(dx2, dg2, dw), (part, part2))
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, _scratch = ent
    cdy, cx, cg, cw, cr, cdx, cdg, _, _, cdw, _, _ = args
    if outs is None:
        outs = _alloc_outs(out_specs, x.device)
    dx2, dg2, dw = outs
    _retarget(cdy, dy2)
    _retarget(cx, x2)
    _retarget(cg, g2)
    _retarget(cw, w)
    _retarget(cr, rstd)
    _retarget(cdx, dx2)
    _retarget(cdg, dg2)
    _retarget(cdw, dw)
    compiled(*args)
    return dx2, dg2, dw
