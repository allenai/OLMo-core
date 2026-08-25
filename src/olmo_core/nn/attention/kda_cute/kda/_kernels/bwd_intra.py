"""003's bwd_intra — phase 1: tcgen05 MMA off-diagonals inside the 16-warp SIMT kernel.

The cross-block sweeps (2.59ms of the 5.87ms SIMT floor — the KDA003C_SKIP attribution)
move to 12 tiny tf32 GEMMs issued by warp 0 into tmem; the 16x16 diagonals, prescales,
grad io and epilogue stay exactly as the phase-0 SIMT kernel left them. No warp
specialization: one PipelineUmmaAsync commit/wait separates issue from a 4-warp-group
staging pass that applies the per-block exp2 post-factors during t2r and lands the
combined cross terms in smem, where the block loop reads them as one LDS.128 each.

The factorization is the SAME per-block-boundary form the SIMT cross sweeps used
(kb/qb/kbb through each block's own end/start boundary — every exp2 argument <= 0,
the gate-magnitude law), so the MMA computes bit-comparable inner sums, just in tf32:
  row side, target block i, j < i:
    Qq_j[d, r] = sum_{s in blk j} kb[s,d]·dAqk[r,s]   (Qk_j with dAkk)
    accq_cross[r,d] = sum_j exp2(g_r - g_e(j))[d] · Qq_j[d,r]
  col side, source block i, j > i:
    P_j[d, s] = sum_{r in blk j} qb[r,d]·dAqk[r,s] + kbb[r,d]·dAkk[r,s]  (chained accs)
    dkt_cross[s,d] = sum_j exp2(g_b0(j) - g_s)[d] · P_j[d,s]

Probe-validated pieces (probe_mma.py, 2026-08-24): tf32 trivial tiled MMAs at
(M=128, N in {16,32,48}, k=16) with A mn-major + B K-MAJOR (the mn-major B descriptor
produced deterministically wrong results at byte-identical smem placement — falsified
twice, so the col-side dA blocks are stored transposed instead); canonical operand smem
written through make_smem_layout_epi views; ONE wide A buffer k-sliced per GEMM by
fragment k-tile index; chained accumulation into one tmem acc; warp-0 issue +
PipelineUmmaAsync -> 512-thread consumer_wait; [128,16] strip t2r at tmem col offsets.

smem: the three canonical A buffers (kb/qb/kbb, 24KB each) are dead once the MMAs
commit, so the staging arrays (accq/acck/dkt, 24KB each) overlay them exactly.
Peak footprint ~192KB, 1 CTA/SM as before.

--- phase-0 header (still true for everything the MMAs didn't take) ---

Lane ownership is consecutive (d = 4*lane+c) with genuine vector accesses — every
per-lane read of its 4 columns is one LDS.128 (fp32) / LDS.64 (bf16), ditto the gmem
grad reads/writes. v2's 4-way bank conflicts on consecutive ownership were a property
of stride-4 SCALAR loads. Structure — one CTA per (chunk, b*hv), 512 threads as
(32 d-groups) x (16 row-lanes); mirror pairing balances the diagonal triangle work
(even blocks r0+rlane, odd blocks r0+15-rlane). The diagonal pairs keep EXACTLY one
one-sided exp2 per (r,s,d): nothing inside a diagonal block is ever factorized.
db is produced complete in-kernel (full-warp butterfly); the host adds the incoming db.

Every exp2 argument in this file is <= 0. The gx16 arm of dbg_intra_cute.py is the guard.
K=128, BT=64, BC=16, fixed-length only; the wrapper falls back to the Triton kernel
elsewhere. [[cutedsl-math-precise-by-default]]: all exp2 through the fastmath path.
"""

from __future__ import annotations


import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# The call cache lives in _common: one copy of the marshal/poke/release scheme for
# every kernel in this package. In the research ladder each kernel carried its own,
# which is how the keepalive leak was fixed in one of them and missed in three.
from ..._common.cache import (  # noqa: F401
    alloc_outs as _alloc_outs,
    cute_view as _cute_view,
    out_specs as _out_specs,
    release_keepalives as _release_keepalives,
    retarget as _retarget,
)

BT = 64
BC = 16
K = 128
NBLK = BT // BC  # 4
THREADS = 512  # 32 d-groups (4 cols each) x 16 row-lanes
NDG = 32  # d-groups per row: one warp spans a full row
VEC = K // NDG  # 4 consecutive columns per thread (d = 4*lane + c), one vector load

# tmem column offsets of the nine accumulators (f32 cols; 288 of the 512 allocated).
# Bases are 16-col aligned — probe_mma validated an acc at col 48.
TM_QQ = (0, 48, 80)     # Qq_j [128, 48/32/16], j = 0,1,2
TM_QK = (96, 144, 176)  # Qk_j
TM_P = (192, 208, 240)  # P_j [128, 16/32/48], j = 1,2,3

# staging tasks: (group, kind, dst_block). Weights (t2r count) ~balanced per group.
STAGE_TASKS = ((0, "row", 3), (1, "row", 2), (1, "col", 2),
               (2, "row", 1), (2, "col", 0), (3, "col", 1))


def _exp2(x):
    return cute.math.exp2(x, fastmath=True)


# The research tree drives these from KDA003C_SKIP to time half a kernel at a time; they
# produce WRONG results by construction, so in a library they are constants. Left in place
# rather than deleted so this file still diffs cleanly against the ladder it came from —
# every branch they guard is eliminated when the kernel is traced.
SKIP_DIAG = False
SKIP_CROSS = False
SKIP_IO = False
SKIP_STAGE = False
SKIP_MMA = False

# ptxas left alone targets 64 registers — an occupancy this smem footprint can never reach —
# and spills 1-2KB/thread. 128 fits one 512-thread CTA per SM, which is the shape this
# kernel is built for.
_MAXREG = 128

# One CTA per (chunk, b*hv): below a few waves of a 148-SM box the Triton fallback wins and
# the per-call marshaling is not amortized (T512 rows regressed to 0.90x).
_MIN_CTAS = 1024


class KdaIntraBwdKernel:
    """See module docstring. io_dtype is q/k/beta's dtype (bf16/fp16); everything else fp32."""

    def __init__(self, io_dtype, fold_dg=False, emit_bf16=False):
        # fold_dg: emit dg already chunk-reverse-cumsum'd (fla's chunk_local_cumsum
        # reverse=True contract), deleting the separate dg_cumsum launch from the chain.
        # The CTA owns its whole 64-row chunk, so the fold is local: the block loop
        # banks per-row dg into smem instead of gmem, then a segmented suffix scan
        # (warp w owns rows 4w..4w+3) writes the final rows out coalesced.
        # emit_bf16: write dq/dk in io_dtype instead of fp32, deleting the backward's
        # two .to(q.dtype) cast launches. Bit-identical to cast-after (same fp32 value,
        # same round-to-nearest); caller must keep it OFF when HV > H — the gva group
        # reduction sums dq/dk AFTER intra and must stay fp32.
        self.io_dtype = io_dtype
        self.fold_dg = fold_dg
        self.emit_bf16 = emit_bf16
        self.f32 = cutlass.Float32
        self.tf32 = cutlass.TFloat32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=THREADS
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    def _make_mmas(self):
        # one trivial tiled MMA per N in {16, 32, 48}; A mn-major (d contiguous),
        # B k-major everywhere (mn-major B falsified — module docstring), tf32 in,
        # fp32 acc. Shared by both GEMM sides.
        tf32, acc, grp = self.tf32, self.f32, self.cta_group
        return {
            n: sm100_utils.make_trivial_tiled_mma(
                tf32, tf32,
                tcgen05.OperandMajorMode("mn"), tcgen05.OperandMajorMode("k"),
                acc, grp, (K, n), tcgen05.OperandSource.SMEM,
            )
            for n in (16, 32, 48)
        }

    def _setup_attributes(self):
        # smem layouts, all row-major (row, d) unless stated. (Built here, not
        # __init__ — cute.make_layout needs the jit MLIR context.)
        self.q_layout = cute.make_layout((BT, K), stride=(K, 1))
        self.g_layout = cute.make_layout((BT, K), stride=(K, 1))
        self.da_layout = cute.make_layout((BT, BT), stride=(BT, 1))
        self.beta_layout = cute.make_layout(BT)
        # vector views over the SAME storage: inner mode = a lane's VEC consecutive
        # columns, so its per-row read is one slice .load() -> LDS.128 f32 / LDS.64
        # bf16.  dvec_layout regroups a gmem K-row the same way for the grad io.
        self.gv_layout = cute.make_layout((BT, NDG, VEC), stride=(K, VEC, 1))
        self.dvec_layout = cute.make_layout((NDG, VEC), stride=(VEC, 1))

        mmas = self._make_mmas()
        tf32 = self.tf32
        # canonical A operand: one [K, 48] mn-major buffer each for kb / qb / kbb;
        # per-GEMM 16-wide k-slices via fragment k-tile indices. The epi view is the
        # (m, k)-indexable layout over the same bytes (probe-validated byte match).
        self.aop_layout = sm100_utils.make_smem_layout_a(
            mmas[48], (K, 48, 3 * BC), tf32, 1
        )
        self.aop_epi = sm100_utils.make_smem_layout_epi(
            tf32, utils.LayoutEnum.COL_MAJOR, (K, 3 * BC), 1
        )
        # canonical B operands, all k-major, one buffer per (side, j):
        #   row j: dA[r in (e(j), BT), s in blk j] as B[n=r-e(j), k=s-16j]
        #   col j: dA[r in blk j, s < 16j]        as B[n=s, k=r-16j]  (transposed store)
        self.brow_layouts = tuple(
            sm100_utils.make_smem_layout_b(
                mmas[BT - BC * (j + 1)], (K, BT - BC * (j + 1), BC), tf32, 1
            )
            for j in range(3)
        )
        self.brow_epis = tuple(
            sm100_utils.make_smem_layout_epi(
                tf32, utils.LayoutEnum.ROW_MAJOR, (BT - BC * (j + 1), BC), 1
            )
            for j in range(3)
        )
        self.bcol_layouts = tuple(
            sm100_utils.make_smem_layout_b(mmas[BC * j], (K, BC * j, BC), tf32, 1)
            for j in (1, 2, 3)
        )
        self.bcol_epis = tuple(
            sm100_utils.make_smem_layout_epi(
                tf32, utils.LayoutEnum.ROW_MAJOR, (BC * j, BC), 1
            )
            for j in (1, 2, 3)
        )
        # staging overlays: plain f32 (row, d) views recast over the A buffers' bytes
        # (dead once the MMAs commit). 48 rows x 128 d = exactly one A buffer each.
        self.stage_layout = cute.make_layout((BT - BC, K), stride=(K, 1))
        self.stagev_layout = cute.make_layout(
            (BT - BC, NDG, VEC), stride=(K, VEC, 1)
        )
        # dg-fold scan: 16 per-segment totals (segment = 4 rows), vector-view shaped
        # like one gv_layout row block. Overlays the dead sAccqS bytes post-block-loop.
        self.segv_layout = cute.make_layout((BC, NDG, VEC), stride=(K, VEC, 1))

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,     # (T, K, HV_q, B) io_dtype — indexed at head hv // (HV//H)
        k: cute.Tensor,     # (T, K, HV_q, B)
        g: cute.Tensor,     # (T, K, HV, B) fp32
        beta: cute.Tensor,  # (T, HV, B) io_dtype
        dAqk: cute.Tensor,  # (T, BT, HV, B) fp32
        dAkk: cute.Tensor,  # (T, BT, HV, B) fp32
        dq_in: cute.Tensor,   # (T, K, HV, B) fp32
        dk_in: cute.Tensor,   # (T, K, HV, B) fp32
        dg_in: cute.Tensor,   # (T, K, HV, B) fp32
        dq_out: cute.Tensor,  # (T, K, HV, B) fp32
        dk_out: cute.Tensor,  # (T, K, HV, B) fp32
        dg_out: cute.Tensor,  # (T, K, HV, B) fp32
        db_out: cute.Tensor,  # (T, HV, B) fp32
        gsize: cutlass.Int32,  # HV // H, to map hv -> q/k head
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        tf32 = self.tf32
        if cutlass.const_expr(cute.cosize(self.aop_layout) != 48 * K):
            raise ValueError(
                f"A operand layout padded ({cute.cosize(self.aop_layout)} != {48 * K}):"
                " the staging overlay assumption is broken"
            )

        @cute.struct
        class SharedStorage:
            umma_full: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            smem_g: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.g_layout)], 128  # type: ignore
            ]
            smem_daqk: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.da_layout)], 128  # type: ignore
            ]
            smem_dakk: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.da_layout)], 128  # type: ignore
            ]
            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.q_layout)], 128  # type: ignore
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.q_layout)], 128  # type: ignore
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[self.f32, cute.cosize(self.beta_layout)], 128  # type: ignore
            ]
            # canonical MMA operands (tf32). kb/qb/kbb are overlaid by the staging
            # arrays after the MMAs complete.
            smem_kb: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.aop_layout)], 1024  # type: ignore
            ]
            smem_qb: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.aop_layout)], 1024  # type: ignore
            ]
            smem_kbb: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.aop_layout)], 1024  # type: ignore
            ]
            smem_brq0: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.brow_layouts[0])], 1024  # type: ignore
            ]
            smem_brq1: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.brow_layouts[1])], 1024  # type: ignore
            ]
            smem_brq2: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.brow_layouts[2])], 1024  # type: ignore
            ]
            smem_brk0: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.brow_layouts[0])], 1024  # type: ignore
            ]
            smem_brk1: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.brow_layouts[1])], 1024  # type: ignore
            ]
            smem_brk2: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.brow_layouts[2])], 1024  # type: ignore
            ]
            smem_bcq1: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.bcol_layouts[0])], 1024  # type: ignore
            ]
            smem_bcq2: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.bcol_layouts[1])], 1024  # type: ignore
            ]
            smem_bcq3: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.bcol_layouts[2])], 1024  # type: ignore
            ]
            smem_bck1: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.bcol_layouts[0])], 1024  # type: ignore
            ]
            smem_bck2: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.bcol_layouts[1])], 1024  # type: ignore
            ]
            smem_bck3: cute.struct.Align[
                cute.struct.MemRange[tf32, cute.cosize(self.bcol_layouts[2])], 1024  # type: ignore
            ]
            # dg-fold bank: the block loop's per-row dg lands here instead of gmem,
            # scanned + written out after the loop. Allocated unconditionally — the
            # footprint is 1 CTA/SM with or without it (230.4KB of 232448B cap).
            smem_dgacc: cute.struct.Align[
                cute.struct.MemRange[self.f32, BT * K], 128  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
            )

        T = cute.size(g, mode=[0])
        HV = cute.size(g, mode=[2])
        B = cute.size(g, mode=[3])
        NT = T // BT
        grid = (NT * HV * B, 1, 1)

        # min_blocks_per_mp=1 -> nvvm.minctasm: smem is dynamic so ptxas can't see that
        # this footprint caps at 1 CTA/SM; without the hint it targets 2 CTAs (64 regs)
        # and spills 2KB/thread.
        self.kda_cute_intra(
            q, k, g, beta, dAqk, dAkk,
            dq_in, dk_in, dg_in, dq_out, dk_out, dg_out, db_out,
            gsize,
        ).launch(
            grid=grid, block=[THREADS, 1, 1], min_blocks_per_mp=1, stream=stream
        )

    # Named, not `kernel`: the method name is what shows up in a CUDA launch trace, and the
    # rest of this chain launches as kda_cute_fwd / kda_cute_b1 / kda_cute_dhu. A uniform
    # prefix is what lets a witness check assert that ALL FOUR stages ran ours — the
    # existing expect_kernels=("kda_cute",) passes if any single one did, so three stages
    # could silently fall back to fla and the bench would still read green.
    @cute.kernel
    def kda_cute_intra(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mG: cute.Tensor, mBeta: cute.Tensor,
        mAq: cute.Tensor, mAk: cute.Tensor,
        mDqIn: cute.Tensor, mDkIn: cute.Tensor, mDgIn: cute.Tensor,
        mDqOut: cute.Tensor, mDkOut: cute.Tensor, mDgOut: cute.Tensor,
        mDbOut: cute.Tensor,
        gsize: cutlass.Int32,
    ):
        f32 = self.f32
        tf32 = self.tf32
        # Region isolation: layouts/TiledMma built during the host trace cannot be
        # referenced inside the kernel region — rebuild here.
        self._setup_attributes()
        mmas = self._make_mmas()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        HV = cute.size(mG, mode=[2])
        T = cute.size(mG, mode=[0])
        NT = T // BT
        t_idx = bidx % NT
        hv_idx = (bidx // NT) % HV
        b_idx = bidx // (NT * HV)
        h_idx = hv_idx // gsize
        row0 = t_idx * BT  # first T-row of this chunk

        dg = tidx % NDG  # this thread's 4 consecutive d columns: d = VEC*dg + c
        rlane = tidx // NDG  # 0..15, warp-uniform: one warp spans a full row

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sG = storage.smem_g.get_tensor(self.g_layout)
        sAq = storage.smem_daqk.get_tensor(self.da_layout)
        sAk = storage.smem_dakk.get_tensor(self.da_layout)
        sQ = storage.smem_q.get_tensor(self.q_layout)
        sK = storage.smem_k.get_tensor(self.q_layout)
        sBeta = storage.smem_beta.get_tensor(self.beta_layout)
        # vector views of the same buffers
        sGv = storage.smem_g.get_tensor(self.gv_layout)
        sQv = storage.smem_q.get_tensor(self.gv_layout)
        sKv = storage.smem_k.get_tensor(self.gv_layout)

        # MMA operand tensors + their (m, k)-indexable epi write views
        sKb = storage.smem_kb.get_tensor(
            self.aop_layout.outer, swizzle=self.aop_layout.inner
        )
        sKbE = storage.smem_kb.get_tensor(
            self.aop_epi.outer, swizzle=self.aop_epi.inner
        )
        sQb = storage.smem_qb.get_tensor(
            self.aop_layout.outer, swizzle=self.aop_layout.inner
        )
        sQbE = storage.smem_qb.get_tensor(
            self.aop_epi.outer, swizzle=self.aop_epi.inner
        )
        sKbb = storage.smem_kbb.get_tensor(
            self.aop_layout.outer, swizzle=self.aop_layout.inner
        )
        sKbbE = storage.smem_kbb.get_tensor(
            self.aop_epi.outer, swizzle=self.aop_epi.inner
        )
        _brq_mr = (storage.smem_brq0, storage.smem_brq1, storage.smem_brq2)
        _brk_mr = (storage.smem_brk0, storage.smem_brk1, storage.smem_brk2)
        _bcq_mr = (storage.smem_bcq1, storage.smem_bcq2, storage.smem_bcq3)
        _bck_mr = (storage.smem_bck1, storage.smem_bck2, storage.smem_bck3)
        sBrq = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_brq_mr, self.brow_layouts)
        )
        sBrqE = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_brq_mr, self.brow_epis)
        )
        sBrk = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_brk_mr, self.brow_layouts)
        )
        sBrkE = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_brk_mr, self.brow_epis)
        )
        sBcq = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_bcq_mr, self.bcol_layouts)
        )
        sBcqE = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_bcq_mr, self.bcol_epis)
        )
        sBck = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_bck_mr, self.bcol_layouts)
        )
        sBckE = tuple(
            mr.get_tensor(ly.outer, swizzle=ly.inner)
            for mr, ly in zip(_bck_mr, self.bcol_epis)
        )
        # staging overlays over the A buffers (valid only after the MMAs commit)
        _kb_flat = storage.smem_kb.get_tensor(cute.make_layout(48 * K))
        _qb_flat = storage.smem_qb.get_tensor(cute.make_layout(48 * K))
        _kbb_flat = storage.smem_kbb.get_tensor(cute.make_layout(48 * K))
        sAccqS = cute.make_tensor(
            cute.recast_ptr(_kb_flat.iterator, dtype=f32), self.stage_layout
        )
        sAccqSV = cute.make_tensor(
            cute.recast_ptr(_kb_flat.iterator, dtype=f32), self.stagev_layout
        )
        sAcckS = cute.make_tensor(
            cute.recast_ptr(_qb_flat.iterator, dtype=f32), self.stage_layout
        )
        sAcckSV = cute.make_tensor(
            cute.recast_ptr(_qb_flat.iterator, dtype=f32), self.stagev_layout
        )
        sDktS = cute.make_tensor(
            cute.recast_ptr(_kbb_flat.iterator, dtype=f32), self.stage_layout
        )
        sDktSV = cute.make_tensor(
            cute.recast_ptr(_kbb_flat.iterator, dtype=f32), self.stagev_layout
        )
        # dg-fold views: the (row, dgroup, VEC) bank, and the 16 segment totals
        # overlaying the dead sAccqS bytes (only touched after the post-loop barrier,
        # which orders them against the block loop's staged reads).
        sDgAcc = storage.smem_dgacc.get_tensor(self.gv_layout)
        sSegV = cute.make_tensor(
            cute.recast_ptr(_kb_flat.iterator, dtype=f32), self.segv_layout
        )

        # umma -> simt pipe: producer = warp 0's MMA commit, consumers = all threads
        umma_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.umma_full.data_ptr(),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1, 1))

        # ---- tmem ----
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=THREADS)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=1,
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr_base = tmem.retrieve_ptr(f32)

        # NOTE: no closures over tmem_ptr_base — a closure that captures a variable
        # cannot be CALLED inside dynamic control flow (the warp-0 / group branches),
        # so acc/strip tensors are built inline where needed.
        def acc_tensor(mma, tile_n, offset, base):
            shape = mma.partition_shape_C((K, tile_n))
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return cute.make_tensor(base + offset, fake.layout)

        # ---- cooperative loads: adjacent threads -> adjacent d (gmem K-innermost) ----
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
        self.sync_barrier.arrive_and_wait()

        # ---- prescale into the canonical A operands + dA copies into B operands ----
        # kb[s]  = k[s] * exp2(g_e(s) - g_s)      s in [0,48),  e(s) = s's block END
        # qb[r]  = q[r] * exp2(g_r - g_b0(r))     r in [16,64), b0(r) = r's block START
        # kbb[r] = beta_r * k[r] * exp2(g_r - g_b0(r))
        # both factorization exponents <= 0 at any gate magnitude.
        if cutlass.const_expr(not SKIP_CROSS):
            for i in cutlass.range_constexpr((BT - BC) * K // THREADS):  # 12 rounds
                idx = i * THREADS + tidx
                s = idx // K
                dd = idx % K
                e_row = (s // BC + 1) * BC
                v = f32(sK[s, dd]) * _exp2(sG[e_row, dd] - sG[s, dd])
                sKbE[dd, s, 0] = v.to(tf32)
            for i in cutlass.range_constexpr((BT - BC) * K // THREADS):  # 12 rounds
                idx = i * THREADS + tidx
                rr = idx // K + BC  # r in [16, 64)
                dd = idx % K
                b0 = (rr // BC) * BC
                f = _exp2(sG[rr, dd] - sG[b0, dd])
                sQbE[dd, rr - BC, 0] = (f32(sQ[rr, dd]) * f).to(tf32)
                sKbbE[dd, rr - BC, 0] = (sBeta[rr] * f32(sK[rr, dd]) * f).to(tf32)
            # dA -> B operands (smem->smem). Row side keeps dA's orientation; the col
            # side stores the block transposed (k-major B[n=s, k=r]).
            for j in cutlass.range_constexpr(3):
                nj = BT - BC * (j + 1)
                for i in cutlass.range_constexpr((nj * BC + THREADS - 1) // THREADS):
                    idx = i * THREADS + tidx
                    if idx < nj * BC:
                        n = idx // BC
                        kx = idx % BC
                        sBrqE[j][n, kx, 0] = sAq[BC * (j + 1) + n, BC * j + kx].to(tf32)
                        sBrkE[j][n, kx, 0] = sAk[BC * (j + 1) + n, BC * j + kx].to(tf32)
            for j in cutlass.range_constexpr(1, 4):
                nj = BC * j
                for i in cutlass.range_constexpr((nj * BC + THREADS - 1) // THREADS):
                    idx = i * THREADS + tidx
                    if idx < nj * BC:
                        n = idx // BC   # s in [0, 16j)
                        kx = idx % BC   # r - 16j
                        sBcqE[j - 1][n, kx, 0] = sAq[BC * j + kx, n].to(tf32)
                        sBckE[j - 1][n, kx, 0] = sAk[BC * j + kx, n].to(tf32)
        self.sync_barrier.arrive_and_wait()

        # ---- warp 0: issue the 12 GEMMs, commit ----
        if cutlass.const_expr(not SKIP_CROSS):
            # strip t2r infra FIRST: mma.set() inside the warp-0 region redefines the
            # mma SSA values in a child region, so anything built from `mmas` must
            # bind before that branch (IR dominance).
            mma16 = mmas[16]
            shape16 = mma16.partition_shape_C((K, BC))
            fake16 = mma16.make_fragment_C(cute.append(shape16, 1))
            t2r_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(2), tcgen05.Pack.NONE), f32
            )
            local_tidx = tidx % 128
            group = warp_idx // 4  # warp group 4g..4g+3 spans all tmem subpartitions
            strip0 = cute.make_tensor(tmem_ptr_base + 0, fake16.layout)
            strip0_2d = strip0[((None, None), 0, 0, None)]
            tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, strip0_2d[None, None, 0])
            thr_t2r = tiled_t2r.get_slice(local_tidx)
            coords = thr_t2r.partition_D(cute.make_identity_tensor((K, BC)))
            fragq = cute.make_rmem_tensor(coords.shape, f32)
            fragk = cute.make_rmem_tensor(coords.shape, f32)
            outq = cute.make_rmem_tensor(coords.shape, f32)
            outk = cute.make_rmem_tensor(coords.shape, f32)

            # (-1 under SKIP_MMA: issue code compiles identically but never runs)
            if warp_idx == (0 if not SKIP_MMA else -1):
                producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                umma_pipe.producer_acquire(producer)
                for j in cutlass.range_constexpr(3):  # row side
                    nj = BT - BC * (j + 1)
                    mma = mmas[nj]
                    trA = mma.make_fragment_A(sKb)
                    trBq = mma.make_fragment_B(sBrq[j])
                    trBk = mma.make_fragment_B(sBrk[j])
                    tQq = acc_tensor(mma, nj, TM_QQ[j], tmem_ptr_base)
                    tQk = acc_tensor(mma, nj, TM_QK[j], tmem_ptr_base)
                    nkt = cute.size(trBq, mode=[2])  # k-tiles per 16-wide GEMM
                    for kk in cutlass.range_constexpr(nkt):
                        mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                        cute.gemm(
                            mma, tQq[None, None, None, 0],
                            trA[None, None, j * nkt + kk, 0],
                            trBq[None, None, kk, 0],
                            tQq[None, None, None, 0],
                        )
                    for kk in cutlass.range_constexpr(nkt):
                        mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                        cute.gemm(
                            mma, tQk[None, None, None, 0],
                            trA[None, None, j * nkt + kk, 0],
                            trBk[None, None, kk, 0],
                            tQk[None, None, None, 0],
                        )
                for j in cutlass.range_constexpr(1, 4):  # col side, chained pair
                    nj = BC * j
                    mma = mmas[nj]
                    trAq = mma.make_fragment_A(sQb)
                    trAk = mma.make_fragment_A(sKbb)
                    trBq = mma.make_fragment_B(sBcq[j - 1])
                    trBk = mma.make_fragment_B(sBck[j - 1])
                    tP = acc_tensor(mma, nj, TM_P[j - 1], tmem_ptr_base)
                    nkt = cute.size(trBq, mode=[2])
                    for kk in cutlass.range_constexpr(nkt):
                        mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                        cute.gemm(
                            mma, tP[None, None, None, 0],
                            trAq[None, None, (j - 1) * nkt + kk, 0],
                            trBq[None, None, kk, 0],
                            tP[None, None, None, 0],
                        )
                    for kk in cutlass.range_constexpr(nkt):
                        mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                        cute.gemm(
                            mma, tP[None, None, None, 0],
                            trAk[None, None, (j - 1) * nkt + kk, 0],
                            trBk[None, None, kk, 0],
                            tP[None, None, None, 0],
                        )
                umma_pipe.producer_commit(producer)

            if cutlass.const_expr(not SKIP_MMA):
                consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                umma_pipe.consumer_wait(consumer)

            # ---- staging: 4 warp groups over the 6 dst blocks. Each task t2rs its
            # [128,16] strips, applies the per-j exp2 post-factor elementwise (coords
            # from the identity partition), and lands the combined cross terms in the
            # (row, d) staging arrays the block loop reads back as LDS.128. ----
            for tg, tkind, tdst in (() if SKIP_STAGE else STAGE_TASKS):
                if group == tg:
                    if cutlass.const_expr(tkind == "row"):
                        for e in cutlass.range_constexpr(cute.size(outq)):
                            outq[e] = f32(0.0)
                            outk[e] = f32(0.0)
                        for j in cutlass.range_constexpr(tdst):
                            off = BC * (tdst - j - 1)
                            stq = cute.make_tensor(
                                tmem_ptr_base + TM_QQ[j] + off, fake16.layout
                            )
                            stk = cute.make_tensor(
                                tmem_ptr_base + TM_QK[j] + off, fake16.layout
                            )
                            srcq = thr_t2r.partition_S(stq[((None, None), 0, 0, None)])
                            srck = thr_t2r.partition_S(stk[((None, None), 0, 0, None)])
                            cute.copy(tiled_t2r, srcq[None, None, None, 0], fragq)
                            cute.copy(tiled_t2r, srck[None, None, None, 0], fragk)
                            cute.arch.fence_view_async_tmem_load()
                            for e in cutlass.range_constexpr(cute.size(outq)):
                                ddc, ccc = coords[e]
                                r = BC * tdst + ccc
                                fj = _exp2(sG[r, ddc] - sG[BC * (j + 1), ddc])
                                outq[e] += fj * fragq[e]
                                outk[e] += fj * fragk[e]
                        for e in cutlass.range_constexpr(cute.size(outq)):
                            ddc, ccc = coords[e]
                            sAccqS[BC * (tdst - 1) + ccc, ddc] = outq[e]
                            sAcckS[BC * (tdst - 1) + ccc, ddc] = outk[e]
                    if cutlass.const_expr(tkind == "col"):
                        for e in cutlass.range_constexpr(cute.size(outq)):
                            outq[e] = f32(0.0)
                        for j in cutlass.range_constexpr(tdst + 1, 4):
                            stp = cute.make_tensor(
                                tmem_ptr_base + TM_P[j - 1] + BC * tdst, fake16.layout
                            )
                            src = thr_t2r.partition_S(stp[((None, None), 0, 0, None)])
                            cute.copy(tiled_t2r, src[None, None, None, 0], fragq)
                            cute.arch.fence_view_async_tmem_load()
                            for e in cutlass.range_constexpr(cute.size(outq)):
                                ddc, ccc = coords[e]
                                s = BC * tdst + ccc
                                fj = _exp2(sG[BC * j, ddc] - sG[s, ddc])
                                outq[e] += fj * fragq[e]
                        for e in cutlass.range_constexpr(cute.size(outq)):
                            ddc, ccc = coords[e]
                            sDktS[BC * tdst + ccc, ddc] = outq[e]
            self.sync_barrier.arrive_and_wait()

        # dynamic-loop vars must exist before staged control flow; the prescale block
        # that used to define them is compiled out under KDA003C_SKIP=cross
        s = cutlass.Int32(0)
        rr = cutlass.Int32(0)

        # ==================== fused block loop: diagonals + epilogue ====================
        # Each thread handles one row per 16-block (blk = i, constexpr), 4 d columns.
        # Mirror pairing balances the triangle. The cross terms are one staged LDS.128
        # per side now; the diagonal pairs keep EXACTLY one one-sided exp2 per (r,s,d).
        for i in cutlass.range_constexpr(NBLK):
            r0 = i * BC
            r = r0 + (rlane if i % 2 == 0 else BC - 1 - rlane)
            grv = sGv[r, dg, None].load()
            gr = [grv[c] for c in range(VEC)]
            # prefetch this block's incoming grads well before use
            dqin = [f32(0.0) for _ in range(VEC)]
            if cutlass.const_expr(not SKIP_IO):
                rdqin = cute.make_tensor(
                    mDqIn[row0 + r, None, hv_idx, b_idx].iterator, self.dvec_layout
                )
                dqv = rdqin[dg, None].load()
                dqin = [dqv[c] for c in range(VEC)]
            accq = [f32(0.0) for _ in range(VEC)]
            acck = [f32(0.0) for _ in range(VEC)]
            if cutlass.const_expr(i > 0 and not SKIP_CROSS):
                av = sAccqSV[r - BC, dg, None].load()
                bv = sAcckSV[r - BC, dg, None].load()
                accq = [av[c] for c in range(VEC)]
                acck = [bv[c] for c in range(VEC)]
            # diagonal: one one-sided exp2 per pair — the numerics law. Dynamic loop to
            # the true (warp-uniform) bound r+1: no predicate, no wasted iterations, and
            # no unroll-driven spills. Loop-carried values are named scalars.
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
                kv = sKv[s, dg, None].load()  # LDS.64 (bf16 x4)
                gv = sGv[s, dg, None].load()  # LDS.128
                t0 = f32(kv[0]) * _exp2(gr[0] - gv[0])
                t1 = f32(kv[1]) * _exp2(gr[1] - gv[1])
                t2 = f32(kv[2]) * _exp2(gr[2] - gv[2])
                t3 = f32(kv[3]) * _exp2(gr[3] - gv[3])
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
            qv = sQv[r, dg, None].load()
            kvr = sKv[r, dg, None].load()  # row r == col-side s: reused by the epilogue
            p1r = [f32(0.0) for _ in range(VEC)]
            p2r = [f32(0.0) for _ in range(VEC)]
            oq = cute.make_fragment_like(cute.make_layout(VEC), f32)
            for c in cutlass.range_constexpr(VEC):
                oq[c] = accq[c] + dqin[c]
                p1r[c] = beta_r * acck[c]
                p2r[c] = f32(qv[c]) * accq[c]
                v += acck[c] * f32(kvr[c])
            rdqo = cute.make_tensor(
                mDqOut[row0 + r, None, hv_idx, b_idx].iterator, self.dvec_layout
            )
            if cutlass.const_expr(self.emit_bf16):
                oqh = cute.make_fragment_like(cute.make_layout(VEC), self.io_dtype)
                for c in cutlass.range_constexpr(VEC):
                    oqh[c] = oq[c].to(self.io_dtype)
                cute.autovec_copy(oqh, rdqo[dg, None])
            else:
                cute.autovec_copy(oq, rdqo[dg, None])
            # db[r] = sum_d dwk * k: this warp owns the whole row — butterfly, lane 0
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
                rdkin = cute.make_tensor(
                    mDkIn[row0 + s, None, hv_idx, b_idx].iterator, self.dvec_layout
                )
                rdgin = cute.make_tensor(
                    mDgIn[row0 + s, None, hv_idx, b_idx].iterator, self.dvec_layout
                )
                dkv = rdkin[dg, None].load()
                dgv = rdgin[dg, None].load()
                dkin = [dkv[c] for c in range(VEC)]
                dgin = [dgv[c] for c in range(VEC)]
            dkt = [f32(0.0) for _ in range(VEC)]
            if cutlass.const_expr(i < NBLK - 1 and not SKIP_CROSS):
                dv = sDktSV[s, dg, None].load()
                dkt = [dv[c] for c in range(VEC)]
            # diagonal: r in [s, r0+BC), one one-sided exp2 per pair
            k0 = dkt[0]
            k1 = dkt[1]
            k2 = dkt[2]
            k3 = dkt[3]
            for rr in cutlass.range(
                (r0 + BC) if SKIP_DIAG else s, r0 + BC, unroll=4
            ):
                aq = sAq[rr, s]
                ak = sAk[rr, s]
                beta_rr = sBeta[rr]
                qv2 = sQv[rr, dg, None].load()
                kv2 = sKv[rr, dg, None].load()
                gv2 = sGv[rr, dg, None].load()
                k0 += (
                    aq * f32(qv2[0]) + ak * beta_rr * f32(kv2[0])
                ) * _exp2(gv2[0] - gr[0])
                k1 += (
                    aq * f32(qv2[1]) + ak * beta_rr * f32(kv2[1])
                ) * _exp2(gv2[1] - gr[1])
                k2 += (
                    aq * f32(qv2[2]) + ak * beta_rr * f32(kv2[2])
                ) * _exp2(gv2[2] - gr[2])
                k3 += (
                    aq * f32(qv2[3]) + ak * beta_rr * f32(kv2[3])
                ) * _exp2(gv2[3] - gr[3])
            dkt[0] = k0
            dkt[1] = k1
            dkt[2] = k2
            dkt[3] = k3
            # epilogue: combine with the row-side partials and the incoming grads
            # (kvr is the row-side load of sK[r] and s == r)
            ok_ = cute.make_fragment_like(cute.make_layout(VEC), f32)
            og_ = cute.make_fragment_like(cute.make_layout(VEC), f32)
            for c in cutlass.range_constexpr(VEC):
                kf = f32(kvr[c])
                ok_[c] = dkin[c] + p1r[c] + dkt[c]
                og_[c] = dgin[c] + p2r[c] + (p1r[c] - dkt[c]) * kf
            rdko = cute.make_tensor(
                mDkOut[row0 + s, None, hv_idx, b_idx].iterator, self.dvec_layout
            )
            rdgo = cute.make_tensor(
                mDgOut[row0 + s, None, hv_idx, b_idx].iterator, self.dvec_layout
            )
            if cutlass.const_expr(self.emit_bf16):
                okh = cute.make_fragment_like(cute.make_layout(VEC), self.io_dtype)
                for c in cutlass.range_constexpr(VEC):
                    okh[c] = ok_[c].to(self.io_dtype)
                cute.autovec_copy(okh, rdko[dg, None])
            else:
                cute.autovec_copy(ok_, rdko[dg, None])
            if cutlass.const_expr(self.fold_dg):
                cute.autovec_copy(og_, sDgAcc[s, dg, None])
            else:
                cute.autovec_copy(og_, rdgo[dg, None])

        if cutlass.const_expr(self.fold_dg):
            # chunk-local reverse cumsum of dg (inclusive suffix sum — fla's
            # chunk_local_cumsum(reverse=True) contract; the separate launch is gone
            # from the chain). Warp w owns rows 4w..4w+3 over its lane's VEC columns,
            # so every output row is one coalesced 512B store; segment totals cross
            # warps through sSegV.
            self.sync_barrier.arrive_and_wait()
            t0 = f32(0.0)
            t1 = f32(0.0)
            t2 = f32(0.0)
            t3 = f32(0.0)
            for e in cutlass.range_constexpr(4):
                v4 = sDgAcc[4 * rlane + e, dg, None].load()
                t0 += v4[0]
                t1 += v4[1]
                t2 += v4[2]
                t3 += v4[3]
            segf = cute.make_fragment_like(cute.make_layout(VEC), f32)
            segf[0] = t0
            segf[1] = t1
            segf[2] = t2
            segf[3] = t3
            cute.autovec_copy(segf, sSegV[rlane, dg, None])
            self.sync_barrier.arrive_and_wait()
            o0 = f32(0.0)
            o1 = f32(0.0)
            o2 = f32(0.0)
            o3 = f32(0.0)
            for rl2 in cutlass.range(rlane + 1, BC):
                v4 = sSegV[rl2, dg, None].load()
                o0 += v4[0]
                o1 += v4[1]
                o2 += v4[2]
                o3 += v4[3]
            og2 = cute.make_fragment_like(cute.make_layout(VEC), f32)
            for e in cutlass.range_constexpr(4):
                row = 4 * rlane + 3 - e
                v4 = sDgAcc[row, dg, None].load()
                o0 += v4[0]
                o1 += v4[1]
                o2 += v4[2]
                o3 += v4[3]
                og2[0] = o0
                og2[1] = o1
                og2[2] = o2
                og2[3] = o3
                rdgo2 = cute.make_tensor(
                    mDgOut[row0 + row, None, hv_idx, b_idx].iterator, self.dvec_layout
                )
                cute.autovec_copy(og2, rdgo2[dg, None])

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)


# ------------------------------- host side ------------------------------------------


_COMPILE_CACHE: dict = {}
_CALL_CACHE: dict = {}


def kda_cute_intra_call(
    q: torch.Tensor,     # [B,T,H,K] bf16/fp16
    k: torch.Tensor,     # [B,T,H,K]
    g: torch.Tensor,     # [B,T,HV,K] fp32 (chunk-local cumsum, log2)
    beta: torch.Tensor,  # [B,T,HV]
    dAqk: torch.Tensor,  # [B,T,HV,BT] fp32
    dAkk: torch.Tensor,  # [B,T,HV,BT] fp32
    dq: torch.Tensor,    # [B,T,HV,K] fp32 incoming
    dk: torch.Tensor,
    dg: torch.Tensor,
    fold_dg: bool = False,
    emit_bf16: bool = False,
):
    B, T, HV, Kdim = g.shape
    H = q.shape[2]
    assert Kdim == K and dAqk.shape[3] == BT and T % BT == 0

    key = tuple(
        (t.shape, t.stride(), t.dtype) for t in (q, k, g, beta, dAqk, dAkk, dq, dk, dg)
    ) + (torch.cuda.current_stream().cuda_stream, fold_dg, emit_bf16)
    ent = _CALL_CACHE.get(key)
    outs = None
    if ent is None:
        out_dtype = q.dtype if emit_bf16 else dq.dtype
        dq2 = torch.empty_like(dq, dtype=out_dtype)
        dk2 = torch.empty_like(dk, dtype=out_dtype)
        dg2 = torch.empty_like(dg, dtype=torch.float)
        db2 = torch.empty(B, T, HV, device=beta.device, dtype=torch.float)

        io_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, fold_dg, emit_bf16)

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
            kernel_obj = KdaIntraBwdKernel(io_dtype, fold_dg, emit_bf16)
            # ptxas left alone targets 64 regs (an occupancy this smem footprint can
            # never reach) and spills 1-2KB/thread; cap at 128 (1 CTA x 512 thr fits).
            maxreg = _MAXREG
            compiled = cute.compile(
                kernel_obj, cq, ck, cg, cbeta, cdaqk, cdakk,
                cdq, cdk, cdg, cdq2, cdk2, cdg2, cdb2,
                cutlass.Int32(HV // H), stream,
                options=f"--ptxas-options --maxrregcount={maxreg}",
            )
            _COMPILE_CACHE[compile_key] = compiled
        # See 002's kernel_fwd._release_keepalives: this entry would otherwise pin the
        # first backward's q/k/g2/beta/dAqk/dAkk/dq/dk/dg and its own four outputs.
        _release_keepalives(
            cq, ck, cg, cbeta, cdaqk, cdakk, cdq, cdk, cdg, cdq2, cdk2, cdg2, cdb2
        )
        args = (
            cq, ck, cg, cbeta, cdaqk, cdakk, cdq, cdk, cdg,
            cdq2, cdk2, cdg2, cdb2, cutlass.Int32(HV // H), stream,
        )
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (dq2, dk2, dg2, db2)
        out_specs = tuple((tuple(t.shape), t.dtype) for t in outs)
        ent = (compiled, args, out_specs)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs = ent
    if outs is None:
        outs = tuple(
            torch.empty(shape, device=q.device, dtype=dtype) for shape, dtype in out_specs
        )
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
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
    cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=False,
    fold_dg=False, emit_bf16=False,
):
    """fla-wrapper-shaped entry; falls back to the Triton kernel off the supported box.

    fold_dg=True changes the dg output contract: dg comes back already chunk-reverse-
    cumsum'd (fla's chunk_local_cumsum(reverse=True)), honored on BOTH paths so callers
    can drop the chain's dg_cumsum stage unconditionally.

    emit_bf16=True writes dq/dk in q's dtype (bit-identical to casting the fp32 values
    after — the backward's .to(q.dtype) then no-ops). Best-effort: forced off for gva
    (the HV>H group reduction sums dq/dk after intra in fp32) and ignored by the
    fallback path, whose fp32 outputs the backward casts as before."""
    emit_bf16 = emit_bf16 and g.shape[2] == q.shape[2]
    if (
        cu_seqlens is not None
        or safe_gate
        or chunk_size != BT
        or k.shape[-1] != K
        or k.shape[1] % BT != 0
        # small grids underfill the SMs (1 CTA per (chunk, b*hv)) and the per-call
        # marshaling isn't amortized — T512 rows regressed 0.90x; the Triton kernel
        # wins below a few waves of the 148-SM box. KDA003_INTRA=cutedsl forces the
        # CuTe path anyway (dbg_intra_cute's small correctness arms need it).
        or g.shape[0] * (k.shape[1] // BT) * g.shape[2] < _MIN_CTAS
    ):
        # 003 has no Triton kernel of its own — the small-grid/off-shape fallback is
        # 002's, which in turn falls back to fla off the box.
        from .bwd_intra_triton import chunk_kda_bwd_intra_cute

        out = chunk_kda_bwd_intra_cute(
            q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
            dq=dq, dk=dk, db=db, dg=dg,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            chunk_size=chunk_size, safe_gate=safe_gate,
        )
        if fold_dg:
            from fla.ops.utils import chunk_local_cumsum

            dq2, dk2, db2, dg2 = out
            dg2 = chunk_local_cumsum(dg2, chunk_size=chunk_size, reverse=True)
            return dq2, dk2, db2, dg2
        return out
    dq2, dk2, db2, dg2 = kda_cute_intra_call(
        q, k, g, beta, dAqk, dAkk, dq, dk, dg, fold_dg=fold_dg, emit_bf16=emit_bf16
    )
    return dq2, dk2, db2.add_(db), dg2
