# mypy: ignore-errors
# The CuTe DSL kernels use metaclass-generated attributes and DSL-typed unpacking that mypy
# cannot follow.
"""bwd_intra in CuTe DSL — a pure-SIMT chunk-resident kernel (no MMA, no tmem, no pipes).

Why this shape: the Triton restructure plateaued at 9.26ms (prod8192) with the diagonal
work attributed at ~6ms and the compute floors far lower (exp2 ~1ms, FMA ~0.7ms, traffic
~1.3ms — ALGORITHM.md). Triton cannot overlap the halves or control layouts; this kernel
holds one whole chunk's tiles in smem and walks the triangle in two balanced SIMT sweeps.

Structure — one CTA per (chunk, b*hv), 512 threads as (32 d-lanes) x (16 row-lanes):
one warp spans a full row; each thread does one row per 16-block with mirror pairing
(even blocks r0+rlane, odd blocks r0+15-rlane) so diagonal work is uniform per thread,
and owns 4 STRIDED columns d = lane + 32c — consecutive-column ownership was 4-way
bank-conflicted on every scalar load (v2: 27ms); strided is conflict-free and keeps the
4x amortization of the warp-uniform dA loads:

  load    : q,k (bf16), g2 (fp32), dAqk,dAkk (fp32), beta -> smem, cooperative coalesced
  sweep 1 : per (r,d): dqA and dwk (the dAqk/dAkk row-side sums)
              cross-block pairs factor through the S-BLOCK END boundary e(s)=16*(j+1):
              exp2(g_r - g_s) = exp2(g_r - g_e) * exp2(g_e - g_s), BOTH factors <= 0 in the
              exponent at any gate magnitude (r >= e > s on a decreasing g) — the same
              safety class as fla's own kg operand, per-pair exp2 eliminated;
              prescale kb[s,d] = k * exp2(g_e(s) - g_s) built once (fp32, matching fla's
              tf32-dot precision, not bf16 — SY's "not bf16" note on the fla kernel).
              diagonal pairs (same 16-block) keep EXACTLY one one-sided exp2 per (r,s,d):
              the numerics law — nothing inside a diagonal block is ever factorized.
            writes dq2 (+incoming) and db (full-warp butterfly, direct store), keeps
            P1 = beta*dwk and P2 = q*dqA in registers for the column side. All three
            prescale arrays (kb, and the column side's qb[r] = q*exp2(g_r - g_b0(r)),
            kbb[r] = beta_r*k*exp2(g_r - g_b0(r)), through the R-BLOCK START boundary)
            are built in ONE phase up front — no buffer reuse, so no barrier between
            the sweeps, and both run fused per block: the column side (dkt for s = r)
            depends on no other thread's row side. The two-phase variant kept P1/P2
            live across the whole kernel and spilled 1008B/thread (2x slower than v1).
            Epilogue: dk2 = dk_in + P1 + dkt; dg2 = dg_in + P2 + (P1 - dkt)*k.

Every exp2 argument in this file is <= 0. The gx16 arm of dbg_intra_cute.py is the guard.

Outputs match fla's chunk_kda_bwd_intra at the ~1e-6 abs level (different-but-valid
boundary references and reduction orders; fla itself computes cross-block gate products in
two exp2s of the same class). db is produced complete in-kernel (no NK slab); the host
adds the incoming db, mirroring fla's wrapper.

K=128, BT=64, BC=16, fixed-length only; the wrapper falls back to the Triton kernel
elsewhere. [[cutedsl-math-precise-by-default]]: all exp2 through the fastmath path.
"""

from __future__ import annotations

import ctypes
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

BT = 64
BC = 16
K = 128
NBLK = BT // BC  # 4
THREADS = 512  # 32 d-groups (4 cols each) x 16 row-lanes; 1024 spilled registers (0.45x)
NDG = 32  # d-groups per row: one warp spans a full row
VEC = K // NDG  # 4 strided columns per thread (d = lane + 32c)


def _exp2(x):
    return cute.math.exp2(x, fastmath=True)


# compile-time attribution knobs (wrong numerics when set — timing only; ncu is blocked
# on this box, so stage attribution goes through skip-variant compiles like the Triton
# kernel's KDA002_INTRA_SKIP)
_SKIP = os.environ.get("KDA002C_SKIP", "")
SKIP_DIAG = _SKIP in ("diag", "both")
SKIP_CROSS = _SKIP in ("cross", "both")
SKIP_IO = _SKIP == "io"  # skip the incoming-grad gmem reads (epilogue latency probe)


class KdaIntraBwdKernel:
    """See module docstring. io_dtype is q/k/beta's dtype (bf16/fp16); everything else fp32."""

    def __init__(self, io_dtype):
        self.io_dtype = io_dtype
        self.f32 = cutlass.Float32
        self.sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _setup_attributes(self):
        # smem layouts, all row-major (row, d): a sweep step reads one full row across the
        # 128 d-lanes -> coalesced, bank-parallel; dA reads are warp-uniform broadcasts.
        # (Built here, not __init__ — cute.make_layout needs the jit MLIR context.)
        self.q_layout = cute.make_layout((BT, K), stride=(K, 1))
        self.k_layout = cute.make_layout((BT, K), stride=(K, 1))
        self.g_layout = cute.make_layout((BT, K), stride=(K, 1))
        self.da_layout = cute.make_layout((BT, BT), stride=(BT, 1))
        self.beta_layout = cute.make_layout(BT)
        # prescale, all three arrays resident at once (no buffer reuse -> no phase
        # barrier between the sweeps -> per-block sweep fusion keeps p1/p2 liveness
        # short; the reuse variant spilled 1008B/thread):
        # rows [0,48): kb[s] for s<48; rows [48,96): qb[r], rows [96,144): kbb[r], r>=16.
        self.pre_layout = cute.make_layout((3 * (BT - BC), K), stride=(K, 1))

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (T, K, HV_q, B) io_dtype — indexed at head hv // (HV//H)
        k: cute.Tensor,  # (T, K, HV_q, B)
        g: cute.Tensor,  # (T, K, HV, B) fp32
        beta: cute.Tensor,  # (T, HV, B) io_dtype
        dAqk: cute.Tensor,  # (T, BT, HV, B) fp32
        dAkk: cute.Tensor,  # (T, BT, HV, B) fp32
        dq_in: cute.Tensor,  # (T, K, HV, B) fp32
        dk_in: cute.Tensor,  # (T, K, HV, B) fp32
        dg_in: cute.Tensor,  # (T, K, HV, B) fp32
        dq_out: cute.Tensor,  # (T, K, HV, B) fp32
        dk_out: cute.Tensor,  # (T, K, HV, B) fp32
        dg_out: cute.Tensor,  # (T, K, HV, B) fp32
        db_out: cute.Tensor,  # (T, HV, B) fp32
        gsize: cutlass.Int32,  # HV // H, to map hv -> q/k head
        stream: cuda.CUstream,
    ):
        self._setup_attributes()

        @cute.struct
        class SharedStorage:
            smem_g: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.g_layout)], 128  # type: ignore
            ]
            smem_daqk: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.da_layout)], 128  # type: ignore
            ]
            smem_dakk: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.da_layout)], 128  # type: ignore
            ]
            smem_pre: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.pre_layout)], 128  # type: ignore
            ]
            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.q_layout)], 128  # type: ignore
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.k_layout)], 128  # type: ignore
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.beta_layout)], 128  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}")

        T = cute.size(g, mode=[0])
        HV = cute.size(g, mode=[2])
        B = cute.size(g, mode=[3])
        NT = T // BT
        grid = (NT * HV * B, 1, 1)

        # min_blocks_per_mp=1 -> nvvm.minctasm: smem is dynamic so ptxas can't see that
        # this footprint caps at 1 CTA/SM; without the hint it targets 2 CTAs (64 regs)
        # and spills 2KB/thread.
        self.kernel(
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq_in,
            dk_in,
            dg_in,
            dq_out,
            dk_out,
            dg_out,
            db_out,
            gsize,
        ).launch(grid=grid, block=[THREADS, 1, 1], min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAq: cute.Tensor,
        mAk: cute.Tensor,
        mDqIn: cute.Tensor,
        mDkIn: cute.Tensor,
        mDgIn: cute.Tensor,
        mDqOut: cute.Tensor,
        mDkOut: cute.Tensor,
        mDgOut: cute.Tensor,
        mDbOut: cute.Tensor,
        gsize: cutlass.Int32,
    ):
        f32 = self.f32
        # Region isolation: layouts built during the host trace cannot be referenced inside
        # the kernel region (001's pattern) — rebuild here.
        self._setup_attributes()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        HV = cute.size(mG, mode=[2])
        T = cute.size(mG, mode=[0])
        NT = T // BT
        t_idx = bidx % NT
        hv_idx = (bidx // NT) % HV
        b_idx = bidx // (NT * HV)
        h_idx = hv_idx // gsize
        row0 = t_idx * BT  # first T-row of this chunk

        dg = tidx % NDG  # this thread's 4 strided d columns: d = dg + NDG*c (bank-conflict-free)
        rlane = tidx // NDG  # 0..15, warp-uniform: one warp spans a full row

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sG = storage.smem_g.get_tensor(self.g_layout)
        sAq = storage.smem_daqk.get_tensor(self.da_layout)
        sAk = storage.smem_dakk.get_tensor(self.da_layout)
        sPre = storage.smem_pre.get_tensor(self.pre_layout)
        sQ = storage.smem_q.get_tensor(self.q_layout)
        sK = storage.smem_k.get_tensor(self.k_layout)
        sBeta = storage.smem_beta.get_tensor(self.beta_layout)

        # ---- cooperative loads: adjacent threads -> adjacent d (gmem is K-innermost) ----
        for i in cutlass.range_constexpr(BT * K // THREADS):  # 16 rounds
            idx = i * THREADS + tidx
            r = idx // K
            dd = idx % K
            sG[r, dd] = mG[row0 + r, dd, hv_idx, b_idx]
            sQ[r, dd] = mQ[row0 + r, dd, h_idx, b_idx]
            sK[r, dd] = mK[row0 + r, dd, h_idx, b_idx]
        for i in cutlass.range_constexpr(BT * BT // THREADS):  # 8 rounds
            idx = i * THREADS + tidx
            r = idx // BT
            ss = idx % BT
            sAq[r, ss] = mAq[row0 + r, ss, hv_idx, b_idx]
            sAk[r, ss] = mAk[row0 + r, ss, hv_idx, b_idx]
        if tidx < BT:
            sBeta[tidx] = f32(mBeta[row0 + tidx, hv_idx, b_idx])
        # prescale (single phase, all three arrays):
        #   kb[s]  = k[s] * exp2(g_e(s) - g_s)      s in [0,48),  e(s) = s's block END
        #   qb[r]  = q[r] * exp2(g_r - g_b0(r))     r in [16,64), b0(r) = r's block START
        #   kbb[r] = beta_r * k[r] * exp2(g_r - g_b0(r))
        # both factorization exponents <= 0 at any gate magnitude.
        self.sync_barrier.arrive_and_wait()
        for i in cutlass.range_constexpr((BT - BC) * K // THREADS):  # 12 rounds
            idx = i * THREADS + tidx
            s = idx // K
            dd = idx % K
            e_row = (s // BC + 1) * BC
            sPre[s, dd] = f32(sK[s, dd]) * _exp2(sG[e_row, dd] - sG[s, dd])
        for i in cutlass.range_constexpr((BT - BC) * K // THREADS):  # 12 rounds
            idx = i * THREADS + tidx
            rr = idx // K + BC  # r in [16, 64)
            dd = idx % K
            b0 = (rr // BC) * BC
            f = _exp2(sG[rr, dd] - sG[b0, dd])
            sPre[rr + (BT - 2 * BC), dd] = f32(sQ[rr, dd]) * f
            sPre[rr + (2 * BT - 3 * BC), dd] = sBeta[rr] * f32(sK[rr, dd]) * f
        self.sync_barrier.arrive_and_wait()

        # ==================== fused sweeps, one block per iteration ====================
        # Each thread handles one row per 16-block (blk = i, constexpr), 4 d columns.
        # Mirror pairing balances the triangle: even blocks take r0+rlane, odd blocks
        # r0+15-rlane, so every thread's diagonal totals 34 pairs. All predicates are
        # warp-uniform (rlane = tidx//NDG is shared by a warp's 32 lanes). The column
        # side (dkt for s = r) needs nothing from other threads' row side, so both run
        # inside one block iteration — p1/p2 live a few lines, not across the kernel
        # (the two-phase version spilled 1008B/thread and ran 2x slower than v1).
        for i in cutlass.range_constexpr(NBLK):
            r0 = i * BC
            r = r0 + (rlane if i % 2 == 0 else BC - 1 - rlane)
            gr = [sG[r, dg + NDG * c] for c in range(VEC)]
            # prefetch this block's incoming grads well before use: issued here, the
            # loads overlap the cross+diag compute below (read-at-use cost 3.25ms of
            # the 9.52 — pure gmem latency, the traffic itself is ~0.4ms)
            dqin = [f32(0.0) for _ in range(VEC)]
            if cutlass.const_expr(not SKIP_IO):
                dqin = [mDqIn[row0 + r, dg + NDG * c, hv_idx, b_idx] for c in range(VEC)]
            accq = [f32(0.0) for _ in range(VEC)]
            acck = [f32(0.0) for _ in range(VEC)]
            # cross-block: per earlier block j, an exp2-free inner sum over its 16 columns,
            # then one post-factor exp2(g_r - g_e(j)) — both factorization exponents <= 0.
            # (inner s-loop dynamic, not unrolled: full unrolling let ptxas hoist
            # whole 16-iteration load batches and spill ~1KB/thread)
            for j in cutlass.range_constexpr(0 if SKIP_CROSS else i):
                e_row = (j + 1) * BC
                fj = [_exp2(gr[c] - sG[e_row, dg + NDG * c]) for c in range(VEC)]
                sq0 = f32(0.0)
                sq1 = f32(0.0)
                sq2 = f32(0.0)
                sq3 = f32(0.0)
                sk0 = f32(0.0)
                sk1 = f32(0.0)
                sk2 = f32(0.0)
                sk3 = f32(0.0)
                for s in cutlass.range(j * BC, (j + 1) * BC, unroll=8):
                    aq = sAq[r, s]
                    ak = sAk[r, s]
                    t0 = sPre[s, dg]
                    t1 = sPre[s, dg + NDG]
                    t2 = sPre[s, dg + 2 * NDG]
                    t3 = sPre[s, dg + 3 * NDG]
                    sq0 += aq * t0
                    sq1 += aq * t1
                    sq2 += aq * t2
                    sq3 += aq * t3
                    sk0 += ak * t0
                    sk1 += ak * t1
                    sk2 += ak * t2
                    sk3 += ak * t3
                accq[0] += fj[0] * sq0
                accq[1] += fj[1] * sq1
                accq[2] += fj[2] * sq2
                accq[3] += fj[3] * sq3
                acck[0] += fj[0] * sk0
                acck[1] += fj[1] * sk1
                acck[2] += fj[2] * sk2
                acck[3] += fj[3] * sk3
            # diagonal: one one-sided exp2 per pair — the numerics law. Dynamic loop to
            # the true (warp-uniform) bound r+1: no predicate, no wasted iterations, and
            # no unroll-driven spills. Loop-carried values are named scalars (the DSL
            # tracks rebinding by name, not list slots).
            a0 = accq[0]
            a1 = accq[1]
            a2 = accq[2]
            a3 = accq[3]
            b0 = acck[0]
            b1 = acck[1]
            b2 = acck[2]
            b3 = acck[3]
            for s in cutlass.range(r0, r0 if SKIP_DIAG else r + 1, unroll=4):
                aq = sAq[r, s]
                ak = sAk[r, s]
                t0 = f32(sK[s, dg]) * _exp2(gr[0] - sG[s, dg])
                t1 = f32(sK[s, dg + NDG]) * _exp2(gr[1] - sG[s, dg + NDG])
                t2 = f32(sK[s, dg + 2 * NDG]) * _exp2(gr[2] - sG[s, dg + 2 * NDG])
                t3 = f32(sK[s, dg + 3 * NDG]) * _exp2(gr[3] - sG[s, dg + 3 * NDG])
                a0 += aq * t0
                a1 += aq * t1
                a2 += aq * t2
                a3 += aq * t3
                b0 += ak * t0
                b1 += ak * t1
                b2 += ak * t2
                b3 += ak * t3
            accq[0] = a0
            accq[1] = a1
            accq[2] = a2
            accq[3] = a3
            acck[0] = b0
            acck[1] = b1
            acck[2] = b2
            acck[3] = b3
            # row-side outputs and short-lived staging for the column side
            v = f32(0.0)
            beta_r = sBeta[r]
            p1r = [f32(0.0) for _ in range(VEC)]
            p2r = [f32(0.0) for _ in range(VEC)]
            for c in cutlass.range_constexpr(VEC):
                mDqOut[row0 + r, dg + NDG * c, hv_idx, b_idx] = accq[c] + dqin[c]
                p1r[c] = beta_r * acck[c]
                p2r[c] = f32(sQ[r, dg + NDG * c]) * accq[c]
                v += acck[c] * f32(sK[r, dg + NDG * c])
            # db[r] = sum_d dwk * k: this warp owns the whole row — butterfly, lane 0 stores
            v += cute.arch.shuffle_sync_bfly(v, offset=16)
            v += cute.arch.shuffle_sync_bfly(v, offset=8)
            v += cute.arch.shuffle_sync_bfly(v, offset=4)
            v += cute.arch.shuffle_sync_bfly(v, offset=2)
            v += cute.arch.shuffle_sync_bfly(v, offset=1)
            if tidx % NDG == 0:
                mDbOut[row0 + r, hv_idx, b_idx] = v

            # column side for s = r (same row, so p1/p2 are still in registers)
            s = r
            dkin = [f32(0.0) for _ in range(VEC)]
            dgin = [f32(0.0) for _ in range(VEC)]
            if cutlass.const_expr(not SKIP_IO):
                dkin = [mDkIn[row0 + s, dg + NDG * c, hv_idx, b_idx] for c in range(VEC)]
                dgin = [mDgIn[row0 + s, dg + NDG * c, hv_idx, b_idx] for c in range(VEC)]
            dkt = [f32(0.0) for _ in range(VEC)]
            # cross-block: later blocks j, inner sums over prescaled rows, post-factor
            # exp2(g_b0 - g_s) <= 1
            for j in cutlass.range_constexpr(NBLK if SKIP_CROSS else i + 1, NBLK):
                b0r = j * BC
                fj = [_exp2(sG[b0r, dg + NDG * c] - gr[c]) for c in range(VEC)]
                sb0 = f32(0.0)
                sb1 = f32(0.0)
                sb2 = f32(0.0)
                sb3 = f32(0.0)
                for rr in cutlass.range(b0r, b0r + BC, unroll=8):
                    aq = sAq[rr, s]
                    ak = sAk[rr, s]
                    sb0 += aq * sPre[rr + (BT - 2 * BC), dg]
                    sb1 += aq * sPre[rr + (BT - 2 * BC), dg + NDG]
                    sb2 += aq * sPre[rr + (BT - 2 * BC), dg + 2 * NDG]
                    sb3 += aq * sPre[rr + (BT - 2 * BC), dg + 3 * NDG]
                    sb0 += ak * sPre[rr + (2 * BT - 3 * BC), dg]
                    sb1 += ak * sPre[rr + (2 * BT - 3 * BC), dg + NDG]
                    sb2 += ak * sPre[rr + (2 * BT - 3 * BC), dg + 2 * NDG]
                    sb3 += ak * sPre[rr + (2 * BT - 3 * BC), dg + 3 * NDG]
                dkt[0] += fj[0] * sb0
                dkt[1] += fj[1] * sb1
                dkt[2] += fj[2] * sb2
                dkt[3] += fj[3] * sb3
            # diagonal: r in [s, r0+BC), one one-sided exp2 per pair; dynamic lower
            # bound s (warp-uniform), same rationale as the row-side diagonal
            k0 = dkt[0]
            k1 = dkt[1]
            k2 = dkt[2]
            k3 = dkt[3]
            for rr in cutlass.range((r0 + BC) if SKIP_DIAG else s, r0 + BC, unroll=4):
                aq = sAq[rr, s]
                ak = sAk[rr, s]
                beta_rr = sBeta[rr]
                k0 += (aq * f32(sQ[rr, dg]) + ak * beta_rr * f32(sK[rr, dg])) * _exp2(
                    sG[rr, dg] - gr[0]
                )
                k1 += (aq * f32(sQ[rr, dg + NDG]) + ak * beta_rr * f32(sK[rr, dg + NDG])) * _exp2(
                    sG[rr, dg + NDG] - gr[1]
                )
                k2 += (
                    aq * f32(sQ[rr, dg + 2 * NDG]) + ak * beta_rr * f32(sK[rr, dg + 2 * NDG])
                ) * _exp2(sG[rr, dg + 2 * NDG] - gr[2])
                k3 += (
                    aq * f32(sQ[rr, dg + 3 * NDG]) + ak * beta_rr * f32(sK[rr, dg + 3 * NDG])
                ) * _exp2(sG[rr, dg + 3 * NDG] - gr[3])
            dkt[0] = k0
            dkt[1] = k1
            dkt[2] = k2
            dkt[3] = k3
            # epilogue: combine with the row-side partials and the incoming grads
            for c in cutlass.range_constexpr(VEC):
                kf = f32(sK[s, dg + NDG * c])
                mDkOut[row0 + s, dg + NDG * c, hv_idx, b_idx] = dkin[c] + p1r[c] + dkt[c]
                mDgOut[row0 + s, dg + NDG * c, hv_idx, b_idx] = (
                    dgin[c] + p2r[c] + (p1r[c] - dkt[c]) * kf
                )


# ------------------------------- host side ------------------------------------------


def _cute_view(t: torch.Tensor, perm: tuple[int, ...], dyn_modes: tuple[int, ...]):
    # 001's convention verbatim (see that file): stride order from the unpermuted tensor.
    t = t.detach()
    base_order = sorted(range(t.dim()), key=lambda i: -t.stride(i))
    new_of_old = {old: new for new, old in enumerate(perm)}
    stride_order = tuple(new_of_old[dd] for dd in base_order)
    tt = t.permute(*perm)
    ct = from_dlpack(tt, assumed_align=16)
    for m in dyn_modes:
        ct = ct.mark_compact_shape_dynamic(mode=m, stride_order=stride_order)
    return ct


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


_COMPILE_CACHE: dict = {}
_CALL_CACHE: dict = {}


def kda_cute_intra_call(
    q: torch.Tensor,  # [B,T,H,K] bf16/fp16
    k: torch.Tensor,  # [B,T,H,K]
    g: torch.Tensor,  # [B,T,HV,K] fp32 (chunk-local cumsum, log2)
    beta: torch.Tensor,  # [B,T,HV]
    dAqk: torch.Tensor,  # [B,T,HV,BT] fp32
    dAkk: torch.Tensor,  # [B,T,HV,BT] fp32
    dq: torch.Tensor,  # [B,T,HV,K] fp32 incoming
    dk: torch.Tensor,
    dg: torch.Tensor,
):
    B, T, HV, Kdim = g.shape
    H = q.shape[2]
    assert Kdim == K and dAqk.shape[3] == BT and T % BT == 0

    key = tuple((t.shape, t.stride(), t.dtype) for t in (q, k, g, beta, dAqk, dAkk, dq, dk, dg)) + (
        torch.cuda.current_stream().cuda_stream,
    )
    ent = _CALL_CACHE.get(key)
    outs = None
    if ent is None:
        dq2 = torch.empty_like(dq)
        dk2 = torch.empty_like(dk)
        dg2 = torch.empty_like(dg, dtype=torch.float)
        db2 = torch.empty(B, T, HV, device=beta.device, dtype=torch.float)

        io_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype,)

        cq = _cute_view(q, (1, 3, 2, 0), (0, 2, 3))
        ck = _cute_view(k, (1, 3, 2, 0), (0, 2, 3))
        cg = _cute_view(g, (1, 3, 2, 0), (0, 2, 3))
        cbeta = _cute_view(beta, (1, 2, 0), (0, 1, 2))
        cdaqk = _cute_view(dAqk, (1, 3, 2, 0), (0, 2, 3))
        cdakk = _cute_view(dAkk, (1, 3, 2, 0), (0, 2, 3))
        cdq = _cute_view(dq, (1, 3, 2, 0), (0, 2, 3))
        cdk = _cute_view(dk, (1, 3, 2, 0), (0, 2, 3))
        cdg = _cute_view(dg, (1, 3, 2, 0), (0, 2, 3))
        cdq2 = _cute_view(dq2, (1, 3, 2, 0), (0, 2, 3))
        cdk2 = _cute_view(dk2, (1, 3, 2, 0), (0, 2, 3))
        cdg2 = _cute_view(dg2, (1, 3, 2, 0), (0, 2, 3))
        cdb2 = _cute_view(db2, (1, 2, 0), (0, 1, 2))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = KdaIntraBwdKernel(io_dtype)
            # ptxas left alone targets 64 regs (an occupancy this smem footprint can
            # never reach) and spills 1-2KB/thread; cap at 128 (1 CTA x 512 thr fits).
            maxreg = int(os.environ.get("KDA002C_MAXREG", "128"))
            compiled = cute.compile(
                kernel_obj,
                cq,
                ck,
                cg,
                cbeta,
                cdaqk,
                cdakk,
                cdq,
                cdk,
                cdg,
                cdq2,
                cdk2,
                cdg2,
                cdb2,
                cutlass.Int32(HV // H),
                stream,
                options=f"--ptxas-options --maxrregcount={maxreg}",
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (
            cq,
            ck,
            cg,
            cbeta,
            cdaqk,
            cdakk,
            cdq,
            cdk,
            cdg,
            cdq2,
            cdk2,
            cdg2,
            cdb2,
            cutlass.Int32(HV // H),
            stream,
        )
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (dq2, dk2, dg2, db2)
        out_specs = tuple((tuple(t.shape), t.dtype) for t in outs)
        ent = (compiled, args, out_specs)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs = ent
    if outs is None:
        outs = tuple(torch.empty(shape, device=q.device, dtype=dtype) for shape, dtype in out_specs)
    dq2, dk2, dg2, db2 = outs
    cq, ck, cg, cbeta, cdaqk, cdakk, cdq, cdk, cdg, cdq2, cdk2, cdg2, cdb2, _, _ = args
    _retarget(cq, q)
    _retarget(ck, k)
    _retarget(cg, g)
    _retarget(cbeta, beta)
    _retarget(cdaqk, dAqk)
    _retarget(cdakk, dAkk)
    _retarget(cdq, dq)
    _retarget(cdk, dk)
    _retarget(cdg, dg)
    _retarget(cdq2, dq2)
    _retarget(cdk2, dk2)
    _retarget(cdg2, dg2)
    _retarget(cdb2, db2)
    compiled(*args)
    return dq2, dk2, db2, dg2


def chunk_kda_bwd_intra_cutedsl(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    db,
    dg,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
    safe_gate=False,
):
    """fla-wrapper-shaped entry; falls back to the Triton kernel off the supported box."""
    if (
        cu_seqlens is not None
        or safe_gate
        or chunk_size != BT
        or k.shape[-1] != K
        or k.shape[1] % BT != 0
        # small grids underfill the SMs (1 CTA per (chunk, b*hv)) and the per-call
        # marshaling isn't amortized — T512 rows regressed 0.90x; the Triton kernel
        # wins below a few waves of the 148-SM box. KDA002_INTRA=cutedsl forces the
        # CuTe path anyway (dbg_intra_cute's small correctness arms need it).
        or (
            g.shape[0] * (k.shape[1] // BT) * g.shape[2] < 1024
            and os.environ.get("KDA002_INTRA", "") != "cutedsl"
        )
    ):
        from .kernel_intra import chunk_kda_bwd_intra_cute

        return chunk_kda_bwd_intra_cute(
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
    dq2, dk2, db2, dg2 = kda_cute_intra_call(q, k, g, beta, dAqk, dAkk, dq, dk, dg)
    return dq2, dk2, db2.add_(db), dg2
