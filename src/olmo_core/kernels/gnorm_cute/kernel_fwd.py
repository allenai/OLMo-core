"""The CuTe DSL forward for the gated RMS norm. See ALGORITHM.md / NOTES.md.

One SIMT kernel, `gnorm_cute_fwd`, on the flattened `(R, D)` view:

    per row:  rstd = rsqrt(sum(x^2)/D + eps)          fp32 accum from bf16 loads
              y    = x * rstd * w * act(g)            act = swish | sigmoid
              store y (x's dtype), rstd (fp32, one per row — fla's backward consumes it)

Mapping: one warp per row. At D=256 that is 32 lanes x 8 elements = one 16B vector load per
lane for each of `x` and `g` (8B/4B at D=128/64), so the row's sum of squares is one
`redux/shfl` butterfly in fp32 with no smem round trip and no CTA barrier, and `x` stays in
the lane's registers between the reduction and the scale. 8 warps per CTA, grid = R/8; the
weight is a 32B fp32 load per lane that hits L1 for every row after a CTA's first.

Deliberately simpler than ALGORITHM.md's sketch: no TMA, no smem. This op streams 1.5KB per
row through three contiguous tensors, which is exactly the access pattern a plain vectorized
load/store kernel does at peak (`torch.add` is that kernel, and it sets the ceiling
`dbg_peak.py` measures). TMA pipelining earns its complexity when addresses are strided or
reuse exists; here there is neither, so the complexity budget goes to the host path instead —
the layout-keyed call cache below, without which marshaling alone (~0.28ms, measured in gdn)
would bury every case below T8192.

The grid IS persistent, though (v2): v1 launched one 8-row CTA per row-block and lost prod8192
at 0.844x — 262K CTAs each paying launch + a cold w load + one serial
load->reduce->gate->store chain per warp lifetime left it at 72% of streaming bandwidth. v2
caps the grid at SMs * _CTAS_PER_SM and strides each CTA over row-blocks, so the weight is
loaded into registers once per warp and the loop gives each warp back-to-back rows to overlap.
Below the cap the grid is unchanged, so the T512-T8192 wins are untouched.

Host-side rules learned in gdn, kept here:
  * cache the marshaled views keyed on layout, retarget pointers per call with one ctypes
    word-write each (`cutedsl-call-cache-pointer-poke`);
  * OUTPUTS are allocated fresh per call and retargeted into the cached argument pack —
    a cache-owned output buffer aliases across calls, which no single-call check can see.
"""

from __future__ import annotations

import ctypes
from typing import Type

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

LOG2E = 1.4426950408889634

# Rows per CTA = warps per CTA. 8 warps / 256 threads is the streaming sweet spot: enough
# warps to hide gmem latency at 4 CTAs/SM, small enough that the tail CTA wastes little.
RPB = 8

# Grid cap: SMs * this. Above the cap CTAs stride over row-blocks (the persistent loop);
# below it every block gets its own CTA and the loop runs once. dbg_grid.py's sweep at
# prod8192 (B300, 2026-08-13): 2/SM is 60% of stream, 8/SM 87%, 32/SM 92%, then a plateau
# at 93.8% from 64 to 128 — so 64, and small cases never reach the cap anyway.
_CTAS_PER_SM = 64


class GnormFwdKernel:
    """Forward of the gated RMS norm: one warp per row of the flattened (R, D) view.

    Static config: D (the reduction length — sets the per-lane vector width V = D/32),
    the io dtype, and the activation. R is a dynamic mode of the tensor arguments, so one
    compilation serves every row count.
    """

    def __init__(self, io_dtype: Type[cutlass.Numeric], D: int, activation: str):
        assert D % 32 == 0 and D <= 512, f"one-warp-per-row mapping needs 32 | D, got {D}"
        assert activation in ("swish", "silu", "sigmoid")
        self.io_dtype = io_dtype
        self.D = D
        self.V = D // 32  # elements per lane: 8 at D=256 -> one 16B load
        self.swish = activation in ("swish", "silu")

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (R, D) io dtype
        mG: cute.Tensor,  # (R, D) io dtype
        mW: cute.Tensor,  # (D,) fp32
        mY: cute.Tensor,  # (R, D) io dtype
        mR: cute.Tensor,  # (R/RPB, RPB) fp32 — rstd, 2D so the dynamic mode is not stride-1
        eps: cutlass.Float32,
        num_ctas: cutlass.Int32,  # min(R/RPB, SMs * _CTAS_PER_SM), computed host-side
        stream: cuda.CUstream,
    ):
        # R % RPB == 0 is asserted host-side, so the row-blocks cover R exactly and the
        # kernel's only predicate is the block-stride loop bound.
        self.gnorm_cute_fwd(mX, mG, mW, mY, mR, eps, num_ctas).launch(
            grid=(num_ctas, 1, 1), block=[32 * RPB, 1, 1], stream=stream
        )

    @cute.kernel
    def gnorm_cute_fwd(
        self,
        mX: cute.Tensor,
        mG: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mR: cute.Tensor,
        eps: cutlass.Float32,
        num_ctas: cutlass.Int32,
    ):
        V = self.V
        f32 = cutlass.Float32

        bidx, _, _ = cute.arch.block_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        # The weight lives in this warp's registers across every row it ever touches.
        gW = cute.local_tile(mW, (V,), (lane,))
        rW = cute.make_fragment_like(gW)
        cute.autovec_copy(gW, rW)
        wf = rW.load()

        # CTAs stride over 8-row blocks: at iteration `it` this CTA owns block
        # bidx + it*num_ctas, one row per warp — the CTA's 8 warps read 8 adjacent rows,
        # 4KB contiguous.
        n_blocks = cute.size(mR, mode=[0])
        trips = (n_blocks - bidx + num_ctas - 1) // num_ctas
        for it in cutlass.range(trips):
            blk = bidx + it * num_ctas
            row = blk * RPB + wid

            # This lane's V contiguous elements of the row, as (V,) gmem slices.
            gX = cute.local_tile(mX[row, None], (V,), (lane,))
            gG = cute.local_tile(mG[row, None], (V,), (lane,))
            gY = cute.local_tile(mY[row, None], (V,), (lane,))

            rX = cute.make_fragment_like(gX)
            rG = cute.make_fragment_like(gG)
            # Both loads issued before the reduction so they overlap; x is reused from
            # registers after the butterfly, never re-read.
            cute.autovec_copy(gX, rX)
            cute.autovec_copy(gG, rG)

            xf = rX.load().to(f32)
            partial = (xf * xf).reduce(cute.ReductionOp.ADD, 0.0, 0)
            total = cute.arch.warp_reduction_sum(partial)
            # eps inside the sqrt, matching the ref exactly.
            rstd = cute.math.rsqrt(total * f32(1.0 / self.D) + eps)

            gf = rG.load().to(f32)
            # sigmoid via tanh: sig(x) = 0.5 + 0.5*tanh(x/2), with fastmath so the lowering
            # is cheap. The naive 1/(1+exp(-x)) with DEFAULT flags lowers to a precise exp2
            # polynomial plus a precise div — dozens of issue slots per element, which made
            # v2 compute-bound at 72% of streaming bandwidth on a memory-bound op. (A
            # per-element tanh.approx.f32 variant measured slightly WORSE than this vector
            # fastmath form — dbg_grid.py, 2026-08-13 — so this is not leaving MUFU on the
            # table.)
            sig = 0.5 + 0.5 * cute.math.tanh(gf * f32(0.5), fastmath=True)
            if cutlass.const_expr(self.swish):
                gate = gf * sig
            else:
                gate = sig

            yf = xf * rstd * wf * gate
            rY = cute.make_fragment_like(gY)
            rY.store(yf.to(self.io_dtype))
            cute.autovec_copy(rY, gY)

            if lane == 0:
                mR[blk, wid] = rstd


# --------------------------------------------------------------------------------------------
# host wrapper
# --------------------------------------------------------------------------------------------

_COMPILE_CACHE: dict = {}

# Marshaling (from_dlpack + mark_compact_shape_dynamic per tensor) costs ~0.28ms of host time
# in gdn's measurement — several times the entire T512 kernel here. The marshaled views depend
# only on the input LAYOUTS, so cache the fully-built argument pack keyed on
# (shape, stride, dtype, activation, eps, stream) and per call retarget each view's device
# pointer with one ctypes word-write. Outputs are NOT cached: y/rstd are allocated per call
# and retargeted alongside the inputs, because a cache-owned output buffer would alias across
# calls — invisible to the bench (one call per iteration, compared immediately), fatal in a
# layer stack.
_CALL_CACHE: dict = {}


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def _view2d(t: torch.Tensor):
    """(R, D) row-major view with R dynamic, D static. Never the stride-1 mode."""
    ct = from_dlpack(t.detach(), assumed_align=16)
    return ct.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))


def _out_specs(*tensors: torch.Tensor) -> tuple:
    return tuple((tuple(t.shape), t.dtype) for t in tensors)


def _alloc_outs(specs: tuple, device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


_IO_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def gnorm_cute_fwd(
    x: torch.Tensor,  # [..., D] bf16/fp16, contiguous
    g: torch.Tensor,  # same shape/dtype as x
    w: torch.Tensor,  # [D] fp32
    activation: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (y [same shape as x], rstd [R] fp32) — rstd exactly as fla's backward wants it."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    g2 = g.reshape(-1, D)
    R = x2.shape[0]
    assert x2.is_contiguous() and g2.is_contiguous() and w.is_contiguous()
    assert R % RPB == 0, f"row count {R} not a multiple of {RPB}"
    assert w.dtype == torch.float32 and w.shape == (D,)

    key = (
        (R, D), x.dtype, activation, float(eps),
        torch.cuda.current_stream().cuda_stream,
    )
    ent = _CALL_CACHE.get(key)
    outs = None  # set on the miss path, where outputs are allocated to build the views
    if ent is None:
        io_dtype = _IO_DTYPES[x.dtype]
        y2 = torch.empty_like(x2)
        rstd = torch.empty(R, device=x.device, dtype=torch.float32)

        cx = _view2d(x2)
        cg = _view2d(g2)
        cw = from_dlpack(w.detach(), assumed_align=16)
        cy = _view2d(y2)
        cr = _view2d(rstd.view(R // RPB, RPB))
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        num_ctas = min(R // RPB, sms * _CTAS_PER_SM)

        compile_key = (io_dtype, D, activation)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = GnormFwdKernel(io_dtype, D, activation)
            compiled = cute.compile(
                kernel_obj, cx, cg, cw, cy, cr, cutlass.Float32(eps),
                cutlass.Int32(num_ctas), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled

        args = (cx, cg, cw, cy, cr, cutlass.Float32(eps), cutlass.Int32(num_ctas), stream)
        if len(_CALL_CACHE) >= 64:  # distinct layouts are few; this is a leak backstop
            _CALL_CACHE.clear()
        outs = (y2, rstd)
        ent = (compiled, args, _out_specs(y2, rstd))
        _CALL_CACHE[key] = ent

    compiled, args, out_specs = ent
    cx, cg, cw, cy, cr, _, _, _ = args
    if outs is None:
        outs = _alloc_outs(out_specs, x.device)
    y2, rstd = outs
    _retarget(cx, x2)
    _retarget(cg, g2)
    _retarget(cw, w)
    _retarget(cy, y2)
    _retarget(cr, rstd)
    compiled(*args)
    return y2.view(x.shape), rstd
