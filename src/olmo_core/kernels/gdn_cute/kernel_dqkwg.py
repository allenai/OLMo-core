"""CuTe DSL port of fla's `chunk_bwd_dqkwg` (backward stage 5). See NOTES.md / ALGORITHM.md.

The math, per (chunk c, b, hv), verbatim from fla's `chunk_bwd_kernel_dqkwg` at BK=K:

    v-loop:  DS  += do @ v_new^T                  [BT,BT]   (S-like, fp32 acc)
             DQ1 += do @ h_c                      [BT,K]    (h read [K,V] V-contig = k-major B)
             DK1 += v_new @ dh_c                  [BT,K]
             DW  += dv @ h_c                      [BT,K]    (stored negated)
             hdh += sum(h_c . dh_c)               scalar
    then:    dq_part = DQ1 * exp2(g2_i) * scale
             dk_part = DK1 * exp2(G - g2_i)
             dg_last = hdh * exp2(G) + sum(dk_part . k)
             ds      = tril(DS * exp2(g2_i - g2_j)) * scale   -> bf16 (cast HERE, like fla)
             dq = dq_part + ds @ k ;  dk = dk_part + ds^T @ q
             dw = -DW
             dg_i = rowsum(dq . q) - rowsum(dk . k), plus dg_last on row BT-1

Structure notes (all patterns lifted from kernel_fwd.py, which is 002's kernel):

- Chunk-parallel: no recurrence. Grid = B*HV*nseg CTAs, each owning a contiguous range of
  chunks so the TMA/MMA/SIMT pipelines stream across chunk boundaries instead of paying a
  fill per chunk. nseg is a runtime arg chosen host-side for occupancy.
- tmem holds four fp32 accumulators: DS [64,64] + DQ1/DK1/DW [64,128] = 448 of 512 cols.
  There is no room to give ds@k / ds^T@q their own accumulators, so they are issued with
  ACCUMULATE=False into columns whose v-loop values were *read out earliest*: DST into
  DQ1's columns (A reads DQ1 before committing ds_pipe, which orders the overwrite) and
  DSK into DW's columns (B reads DW early and signals dw_rd). That keeps every acc-free
  wait of chunk c+1's v-loop on a prompt reader, so MMA never stalls behind the long
  SIMT epilogue tail — the mistake the first cut made.
- Two SIMT warpgroups split the epilogue: A (warps 4..7) owns the dq path (ds build, dq
  scale+add+store, rowsum dq.q); B (warps 8..11) owns the dk/dw/dg path (sum h.dh, dk
  scale+add+store, dw negate+store, rowsum dk.k, dg combine+store). dg needs both groups'
  rowsums, handed over through smem under a cross-group named barrier.
- sum(h.dh) never touches the swizzle: h and dh have byte-identical smem layouts, so B
  reads both buffers through the same *flat* view — a bijective permutation applied to both
  operands of a full-tile dot leaves the sum unchanged.
- dg rowsums re-read the just-written dq/dk staging buffers (plain row-major, same as the
  fwd's o buffer) rather than reducing MMA fragments across lanes, which would need a
  lane->row mapping. dq is bf16-rounded at that point; the ~2^-9 relative noise this adds
  to dg is far inside the 0.02 tolerance and it keeps the reduction trivially correct.
- The rounding contract follows fla exactly where it matters: ds is masked+decayed+scaled
  in fp32 then cast to bf16 *before* the second-pass MMAs; h/dh/v_new/do/dv are consumed
  as the bf16 they arrive as; every accumulation is fp32.

Logical gmem mode order (M/N, K_contract, rest...):
    do, v_new, dv:  (T, V, HV, B)
    h, dh:          (K, V, NT, HV, B)   -- [B,NT,HV,K,V] storage, V contiguous
    q, k:           (T, K, H, B)  plus transposed views qt, kt: (K, T, H, B)
    g2:             (BT, NT, HV, B) fp32, staged host-side like the fwd
    dq, dk, dw:     (T, K, HV, B) outputs
    dg:             (T, HV, B) fp32 output, plain scatter
"""

from __future__ import annotations

import ctypes
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class GdnBwdDqkwgKernel:
    """dq/dk/dw/dg backward stage. One CTA per (b, hv, chunk-segment); serial over its chunks.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT group A (ds, dq), 8..11 = SIMT group B
    (dk, dw, dg). Warps 2,3 idle through the role branch and only take part in the
    alloc/dealloc barriers.
    """

    def __init__(self, io_dtype: Type[cutlass.Numeric], K: int, V: int):
        self.io_dtype = io_dtype
        self.acc_dtype = cutlass.Float32
        self.BT = 64
        self.K = K
        self.BV = 64

        assert K == 128, "this port assumes BK == K == 128 (fla's hopper+ config)"
        assert V % self.BV == 0

        # MMA tile shapes (M, N, K_contract)
        self.tile_ds = (self.BT, self.BT, self.BV)  # do @ v^T, per v-tile
        self.tile_dqh = (self.BT, self.K, self.BV)  # do @ h (also v @ dh, dv @ h)
        self.tile_dsk = (self.BT, self.K, self.BT)  # ds @ k^T (also ds^T @ q^T)

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0  # in-stream producer (do/v/dv/h/dh) — never blocks on qk
        self.mma_warp_id = 1
        self.qk_warp_id = 2  # qk-stream producer, so its 1-stage acquire stalls only itself
        self.red_warp_id = 3  # sum(h.dh) reducer, so B never consumes the in-stream
        self.a_warp_id = (4, 5, 6, 7)  # ds + dq path
        self.b_warp_id = (8, 9, 10, 11)  # dk + dw + dg path
        self.threads_per_cta = 32 * 12

        self.in_stages = 2  # (do, v, dv, h, dh) per v-tile
        self.qk_stages = 2  # (qt, kt, g2) per chunk

        self.a_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
        self.b_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=128)
        self.ab_sync_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=256)
        # rendezvous between the reducer warp and B, once per chunk (32 + 128 threads)
        self.red_sync_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=160)
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    # ---------------------------------------------------------------------------------

    def _make_tiled_mmas(self):
        io, acc, grp = self.io_dtype, self.acc_dtype, self.cta_group
        mma_ds = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_ds[:2], tcgen05.OperandSource.SMEM,
        )
        mma_dqh = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_dqh[:2], tcgen05.OperandSource.SMEM,
        )
        # ds @ k^T: A = ds [BT,BT] k-major from smem, B = k [K,BT] mn-major (the kt view,
        # loaded through its own TMA atom exactly like the fwd's DH mma).
        mma_dsk = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dsk[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_ds, mma_dqh, mma_dsk

    def _setup_attributes(self):
        mma_ds, mma_dqh, mma_dsk = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        # do / v / dv all live in [BT, BV] k-major A-operand buffers; v doubles as the B
        # operand of DS (same square k-major layout family, the fwd's q-as-A-of-two-mmas
        # trick).
        self.do_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dqh, self.tile_dqh, self.io_dtype, self.in_stages
        )
        self.v_smem_layout = self.do_smem_layout
        self.dv_smem_layout = self.do_smem_layout
        # h / dh as B operands of DQ1/DK1/DW ([K, BV] k-major, V contiguous in gmem)
        self.h_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dqh, self.tile_dqh, self.io_dtype, self.in_stages
        )
        # qt / kt swizzled mn-major B operands of the second-pass mmas, plus g2, double
        # buffered so the second-pass round trip never waits on this load. There are no
        # plain q/k copies: the SIMT rowsums and the dk.k sum read q/k straight from
        # gmem — those bytes were just TMA'd for qt/kt, so the reads hit L2.
        self.qtkt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dsk, self.tile_dsk, self.io_dtype, self.qk_stages
        )
        self.g2_smem_layout = cute.make_layout((BT, self.qk_stages))

        # ds / ds^T as A operands of the second-pass mmas, written by SIMT A through epi
        # views of the same bytes (ROW_MAJOR direct, COL_MAJOR + stmatrix-transpose for
        # the transposed copy — the fwd's h and vp patterns respectively). Packed as two
        # "stages" of one buffer (stage 0 = ds, stage 1 = ds^T) so the pair spans one
        # contiguous 16KB range that A's dq staging buffer can alias after the second
        # pass has consumed it.
        self.ds_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dsk, self.tile_dsk, self.io_dtype, 2
        )
        self.ds_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BT, BT), 2
        )
        self.dst_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BT), 2
        )

        # output staging: plain (BT, K) row-major like the fwd's o buffer, so the dg
        # rowsums can re-read rows trivially. A owns one (dq); B shares one for dw then dk.
        self.out_smem_layout = cute.make_layout((BT, K, 1), stride=(K, 1, BT * K))

        self.num_in_load_bytes = (
            3 * cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.do_smem_layout, (None, None, None, 0))
            )
            + 2 * cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.h_smem_layout, (None, None, None, 0))
            )
        )
        self.num_qk_load_bytes = (
            2 * cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.qtkt_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(cutlass.Float32, cute.slice_(self.g2_smem_layout, (None, 0)))
        )

        (
            self.tmem_ds_offset,
            self.tmem_dq_offset,
            self.tmem_dk_offset,
            self.tmem_dw_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_ds, mma_dqh)

    def _plan_tmem(self, mma_ds, mma_dqh):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        ds = acc_cols(mma_ds, self.tile_ds)
        dqh = acc_cols(mma_dqh, self.tile_dqh)

        off_ds = 0
        off_dq = off_ds + ds
        off_dk = off_dq + dqh
        off_dw = off_dk + dqh
        total_ = off_dw + dqh
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_ds, off_dq, off_dk, off_dw, total

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        do: cute.Tensor,  # (T, V, HV, B)
        v: cute.Tensor,  # (T, V, HV, B) — v_new
        dv: cute.Tensor,  # (T, V, HV, B)
        h: cute.Tensor,  # (K, V, NT, HV, B)
        dh: cute.Tensor,  # (K, V, NT, HV, B)
        q: cute.Tensor,  # (T, K, H, B)
        k: cute.Tensor,  # (T, K, H, B)
        qt: cute.Tensor,  # (K, T, H, B) — same storage as q
        kt: cute.Tensor,  # (K, T, H, B) — same storage as k
        g2: cute.Tensor,  # (BT, NT, HV, B) fp32
        dq: cute.Tensor,  # (T, K, HV, B) out
        dk: cute.Tensor,  # (T, K, HV, B) out
        dw: cute.Tensor,  # (T, K, HV, B) out
        dg: cute.Tensor,  # (T, HV, B) fp32 out
        scale: cutlass.Float32,
        nseg: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_ds, mma_dqh, mma_dsk = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), do,
            cute.slice_(self.do_smem_layout, (None, None, None, 0)),
            self.tile_dqh, mma_dqh, cluster_vmnk,
        )
        tma_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), v,
            cute.slice_(self.v_smem_layout, (None, None, None, 0)),
            self.tile_dqh, mma_dqh, cluster_vmnk,
        )
        tma_dv, tma_tensor_dv = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), dv,
            cute.slice_(self.dv_smem_layout, (None, None, None, 0)),
            self.tile_dqh, mma_dqh, cluster_vmnk,
        )
        tma_h, tma_tensor_h = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), h,
            cute.slice_(self.h_smem_layout, (None, None, None, 0)),
            self.tile_dqh, mma_dqh, cluster_vmnk,
        )
        tma_dh, tma_tensor_dh = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), dh,
            cute.slice_(self.h_smem_layout, (None, None, None, 0)),
            self.tile_dqh, mma_dqh, cluster_vmnk,
        )
        tma_qt, tma_tensor_qt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), qt,
            cute.slice_(self.qtkt_smem_layout, (None, None, None, 0)),
            self.tile_dsk, mma_dsk, cluster_vmnk,
        )
        tma_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), kt,
            cute.slice_(self.qtkt_smem_layout, (None, None, None, 0)),
            self.tile_dsk, mma_dsk, cluster_vmnk,
        )
        g2_cta_v_layout = cute.slice_(cute.make_identity_layout(g2.shape), (None, 0, 0, 0))
        tma_g2, tma_tensor_g2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), g2,
            cute.slice_(self.g2_smem_layout, (None, 0)),
            g2_cta_v_layout,
        )
        tma_dq, tma_tensor_dq = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dq,
            cute.slice_(self.out_smem_layout, (None, None, 0)),
            (BT, K),
        )
        tma_dk, tma_tensor_dk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dk,
            cute.slice_(self.out_smem_layout, (None, None, 0)),
            (BT, K),
        )
        tma_dw, tma_tensor_dw = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dw,
            cute.slice_(self.out_smem_layout, (None, None, 0)),
            (BT, K),
        )

        B = cute.size(do, mode=[3])
        HV = cute.size(do, mode=[2])
        grid = (B * HV * nseg, 1, 1)

        swz_align, lin_align = 1024, 128

        # Every `*_full` range backs both halves of a pipeline's mbarrier array — each
        # needs 2 * num_stages Int64s (see the fwd's SharedStorage comment). The small
        # members sit at the END of the struct: a prefix of scratch before the first
        # 1024-aligned buffer would round up to a full alignment block, and the budget
        # has no kilobyte to spare.
        @cute.struct
        class SharedStorage:
            smem_do: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.do_smem_layout)], swz_align  # type: ignore
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.v_smem_layout)], swz_align  # type: ignore
            ]
            smem_dv: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dv_smem_layout)], swz_align  # type: ignore
            ]
            smem_h: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.h_smem_layout)], swz_align  # type: ignore
            ]
            smem_dh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.h_smem_layout)], swz_align  # type: ignore
            ]
            smem_qt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.qtkt_smem_layout)], swz_align  # type: ignore
            ]
            smem_kt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.qtkt_smem_layout)], swz_align  # type: ignore
            ]
            smem_g2: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.g2_smem_layout)], lin_align  # type: ignore
            ]
            smem_ds: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.ds_smem_layout)], swz_align  # type: ignore
            ]
            smem_out_b: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.out_smem_layout)], lin_align  # type: ignore
            ]
            in_full: cute.struct.MemRange[cutlass.Int64, self.in_stages * 2]  # type: ignore
            qk_full: cute.struct.MemRange[cutlass.Int64, self.qk_stages * 2]  # type: ignore
            ds_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dk1rd_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dsf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dq1f_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dskf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dk1f_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dwf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dstf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            # cross-group dg scratch: 128 A-pair partials, 128 B-pair partials, 128 dkk
            # partials, 1 reduced dg_last, 2x32 parity-buffered h.dh partials (reducer warp)
            dg_scratch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 128 + 128 + 128 + 1 + 64], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
            )

        self.gdn_cute_bwd_dqkwg(
            tma_do, tma_tensor_do,
            tma_v, tma_tensor_v,
            tma_dv, tma_tensor_dv,
            tma_h, tma_tensor_h,
            tma_dh, tma_tensor_dh,
            q,
            k,
            tma_qt, tma_tensor_qt,
            tma_kt, tma_tensor_kt,
            tma_g2, tma_tensor_g2,
            tma_dq, tma_tensor_dq,
            tma_dk, tma_tensor_dk,
            tma_dw, tma_tensor_dw,
            dg,
            scale,
            nseg,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def gdn_cute_bwd_dqkwg(
        self,
        tma_do: cute.CopyAtom, mDO: cute.Tensor,
        tma_v: cute.CopyAtom, mV: cute.Tensor,
        tma_dv: cute.CopyAtom, mDV: cute.Tensor,
        tma_h: cute.CopyAtom, mH: cute.Tensor,
        tma_dh: cute.CopyAtom, mDH: cute.Tensor,
        mQ: cute.Tensor,  # plain view for direct gmem reads (rowsums)
        mK: cute.Tensor,
        tma_qt: cute.CopyAtom, mQT: cute.Tensor,
        tma_kt: cute.CopyAtom, mKT: cute.Tensor,
        tma_g2: cute.CopyAtom, mG2: cute.Tensor,
        tma_dq: cute.CopyAtom, mDQ: cute.Tensor,
        tma_dk: cute.CopyAtom, mDK: cute.Tensor,
        tma_dw: cute.CopyAtom, mDW: cute.Tensor,
        mDG: cute.Tensor,
        scale: cutlass.Float32,
        nseg: cutlass.Int32,
    ):
        BT, K, BV = self.BT, self.K, self.BV
        io = self.io_dtype
        f32 = self.acc_dtype
        # Region isolation: rebuild static config inside the kernel region (see fwd).
        self._setup_attributes()
        mma_ds, mma_dqh, mma_dsk = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_do, tma_v, tma_dv, tma_h, tma_dh,
                         tma_qt, tma_kt, tma_g2, tma_dq, tma_dk, tma_dw]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mDO, mode=[2])
        H = cute.size(mQ, mode=[2])
        T = cute.size(mQ, mode=[0])
        V = cute.size(mDO, mode=[1])
        NT = T // BT
        NVI = V // BV  # static: V is a static mode of the view
        seg = bidx % nseg
        hv_idx = (bidx // nseg) % HV
        b_idx = bidx // (nseg * HV)
        h_idx = hv_idx // (HV // H)
        cpc = (NT + nseg - 1) // nseg  # chunks per CTA
        c0 = seg * cpc
        cnt = cutlass.min(cpc, NT - c0)  # host guarantees cnt >= 1

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sDO = storage.smem_do.get_tensor(self.do_smem_layout.outer, swizzle=self.do_smem_layout.inner)
        sV = storage.smem_v.get_tensor(self.v_smem_layout.outer, swizzle=self.v_smem_layout.inner)
        sDV = storage.smem_dv.get_tensor(self.dv_smem_layout.outer, swizzle=self.dv_smem_layout.inner)
        sH = storage.smem_h.get_tensor(self.h_smem_layout.outer, swizzle=self.h_smem_layout.inner)
        sDH = storage.smem_dh.get_tensor(self.h_smem_layout.outer, swizzle=self.h_smem_layout.inner)
        sQT = storage.smem_qt.get_tensor(self.qtkt_smem_layout.outer, swizzle=self.qtkt_smem_layout.inner)
        sKT = storage.smem_kt.get_tensor(self.qtkt_smem_layout.outer, swizzle=self.qtkt_smem_layout.inner)
        # flat (K, BT, stage) coordinate views of the same swizzled bytes, for the SIMT
        # scalar reads (rowsums, dk.k) — the operand layout's nested modes flatten
        # colexicographically to exactly this profile.
        qtkt_flat = cute.make_layout((self.K, self.BT, self.qk_stages))
        sQTv = cute.make_tensor(sQT.iterator, cute.composition(sQT.layout, qtkt_flat))
        sKTv = cute.make_tensor(sKT.iterator, cute.composition(sKT.layout, qtkt_flat))
        sG2 = storage.smem_g2.get_tensor(self.g2_smem_layout)
        # ds at "stage" 0, ds^T at "stage" 1, all in one 16KB member; sOutA reuses the
        # same bytes for the dq staging buffer. Safe because the second-pass gemms (DST
        # then DSK) have consumed both operands before A ever writes dq_out (A waits
        # dskf), and A's next-chunk ds write is fenced behind bar4, by which point a0's
        # store-pipe acquire has retired the dq TMA store.
        sDS = storage.smem_ds.get_tensor(self.ds_smem_layout.outer, swizzle=self.ds_smem_layout.inner)
        sDS_epi = storage.smem_ds.get_tensor(self.ds_epi_layout.outer, swizzle=self.ds_epi_layout.inner)
        sDST_epi = storage.smem_ds.get_tensor(self.dst_epi_layout.outer, swizzle=self.dst_epi_layout.inner)
        sOutA = storage.smem_ds.get_tensor(self.out_smem_layout)
        sOutB = storage.smem_out_b.get_tensor(self.out_smem_layout)

        # flat aliases of the h/dh bytes for the sum(h.dh) reduction: 16B chunk j*128+t
        # sits at elements (j*128+t)*8, i.e. thread t strides 16B -> conflict-free. The
        # swizzle permutes both buffers identically, so pairing flat elements is exact.
        stage_elems = cute.cosize(cute.slice_(self.h_smem_layout, (None, None, None, 0)))
        n16 = stage_elems // (8 * 128)  # 16B chunks per thread per stage
        hdh_flat_layout = cute.make_layout(
            (8, n16, 128, self.in_stages), stride=(1, 8 * 128, 8, stage_elems)
        )
        sHflat = storage.smem_h.get_tensor(hdh_flat_layout)
        sDHflat = storage.smem_dh.get_tensor(hdh_flat_layout)

        # dg scratch, one flat tensor: [0,128) A pair-partials, [128,256) B pair-partials,
        # [256,384) dkk partials, [384] the reduced dg_last, [385,449) parity-buffered
        # h.dh partials from the reducer warp
        sScr = storage.dg_scratch.get_tensor(cute.make_layout(128 + 128 + 128 + 1 + 64))

        # ---- pipelines ----
        a_threads = 32 * len(self.a_warp_id)
        b_threads = 32 * len(self.b_warp_id)
        in_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.in_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            tx_count=self.num_in_load_bytes,
            barrier_storage=storage.in_full.data_ptr(),
            defer_sync=True,
        )
        qk_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.qk_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, a_threads + b_threads
            ),
            tx_count=self.num_qk_load_bytes,
            barrier_storage=storage.qk_full.data_ptr(),
            defer_sync=True,
        )

        def make_simt_to_mma_pipe(ptr, producer_threads):
            return pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, producer_threads
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                barrier_storage=ptr,
                defer_sync=True,
            )

        def make_mma_to_simt_pipe(ptr, consumer_threads):
            return pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, consumer_threads
                ),
                barrier_storage=ptr,
                defer_sync=True,
            )

        # ds smem operand: A -> MMA. Committed only after A has also read the DQ1 acc, so
        # its wait doubles as the "safe to overwrite DQ1's columns with DST" signal.
        ds_pipe = make_simt_to_mma_pipe(storage.ds_full.data_ptr(), a_threads)
        # B -> MMA: DW acc read out, safe to overwrite its columns with DSK.
        dw_rd = make_simt_to_mma_pipe(storage.dk1rd_full.data_ptr(), b_threads)
        # MMA -> SIMT acc-full events. Their producer_acquire is the acc-free wait.
        dsf_pipe = make_mma_to_simt_pipe(storage.dsf_full.data_ptr(), a_threads)
        dq1f_pipe = make_mma_to_simt_pipe(storage.dq1f_full.data_ptr(), a_threads)
        dskf_pipe = make_mma_to_simt_pipe(storage.dskf_full.data_ptr(), a_threads)
        dk1f_pipe = make_mma_to_simt_pipe(storage.dk1f_full.data_ptr(), b_threads)
        dwf_pipe = make_mma_to_simt_pipe(storage.dwf_full.data_ptr(), b_threads)
        dstf_pipe = make_mma_to_simt_pipe(storage.dstf_full.data_ptr(), b_threads)

        pipeline_init_arrive(cluster_shape_mn=(1, 1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1, 1))

        # ---- tmem ----
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=0, num_threads=self.threads_per_cta
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.a_warp_id[0],
        )
        tmem.allocate(self.num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

        def acc_tensor(mma, tile, offset):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return cute.make_tensor(tmem_ptr_base + offset, fake.layout)

        tDS = acc_tensor(mma_ds, self.tile_ds, self.tmem_ds_offset)
        tDQ = acc_tensor(mma_dqh, self.tile_dqh, self.tmem_dq_offset)
        tDK = acc_tensor(mma_dqh, self.tile_dqh, self.tmem_dk_offset)
        tDW = acc_tensor(mma_dqh, self.tile_dqh, self.tmem_dw_offset)
        # The second-pass accs alias the two accs whose readouts happen earliest, so no
        # gemm of chunk c+1 ever waits on a late reader of chunk c: DST reuses DQ1's
        # columns (A reads DQ1 before committing ds_pipe, which orders the overwrite) and
        # DSK reuses DW's columns (B reads DW early and signals dw_rd).
        tDSK = acc_tensor(mma_dsk, self.tile_dsk, self.tmem_dw_offset)
        tDST = acc_tensor(mma_dsk, self.tile_dsk, self.tmem_dq_offset)

        # ---- global tiles (per (b, hv), open over chunks / v-tiles) ----
        gDO = cute.local_tile(mDO, (BT, BV), (None, None, hv_idx, b_idx))  # (BT,BV,NT,NV)
        gV = cute.local_tile(mV, (BT, BV), (None, None, hv_idx, b_idx))
        gDVin = cute.local_tile(mDV, (BT, BV), (None, None, hv_idx, b_idx))
        gH = cute.local_tile(mH, (K, BV), (0, None, None, hv_idx, b_idx))  # (K,BV,NV,NT)
        gDH = cute.local_tile(mDH, (K, BV), (0, None, None, hv_idx, b_idx))
        gQ = cute.local_tile(mQ, (BT, K), (None, 0, h_idx, b_idx))  # (BT,K,NT)
        gK = cute.local_tile(mK, (BT, K), (None, 0, h_idx, b_idx))
        gQT = cute.local_tile(mQT, (K, BT), (0, None, h_idx, b_idx))  # (K,BT,NT)
        gKT = cute.local_tile(mKT, (K, BT), (0, None, h_idx, b_idx))
        gG2 = mG2[(None, None, hv_idx, b_idx)]  # (BT, NT)
        gDQ = cute.local_tile(mDQ, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)
        gDK = cute.local_tile(mDK, (BT, K), (None, 0, hv_idx, b_idx))
        gDW = cute.local_tile(mDW, (BT, K), (None, 0, hv_idx, b_idx))

        # ==========================================================================
        # TMA warp 0: the in-stream (do/v/dv/h/dh). Its only wait is on the in_pipe
        # itself, so the byte stream that dominates the roofline never queues behind
        # the qk pipe's end-of-chunk consumers.
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_dqh = mma_dqh.get_slice(0)

            tDO_mma = thr_mma_dqh.partition_A(gDO)
            tV_mma = thr_mma_dqh.partition_A(gV)
            tDV_mma = thr_mma_dqh.partition_A(gDVin)
            tH_mma = thr_mma_dqh.partition_B(gH)
            tDH_mma = thr_mma_dqh.partition_B(gDH)

            cta1 = cute.make_layout(1)
            tDOs, tDOg = cpasync.tma_partition(
                tma_do, 0, cta1, cute.group_modes(sDO, 0, 3), cute.group_modes(tDO_mma, 0, 3)
            )
            tVs, tVg = cpasync.tma_partition(
                tma_v, 0, cta1, cute.group_modes(sV, 0, 3), cute.group_modes(tV_mma, 0, 3)
            )
            tDVs, tDVg = cpasync.tma_partition(
                tma_dv, 0, cta1, cute.group_modes(sDV, 0, 3), cute.group_modes(tDV_mma, 0, 3)
            )
            tHs, tHg = cpasync.tma_partition(
                tma_h, 0, cta1, cute.group_modes(sH, 0, 3), cute.group_modes(tH_mma, 0, 3)
            )
            tDHs, tDHg = cpasync.tma_partition(
                tma_dh, 0, cta1, cute.group_modes(sDH, 0, 3), cute.group_modes(tDH_mma, 0, 3)
            )

            in_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.in_stages
            )

            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                for iv in cutlass.range(NVI, unroll_full=True):
                    in_pipe.producer_acquire(in_producer)
                    bar = in_pipe.producer_get_barrier(in_producer)
                    st = in_producer.index
                    cute.copy(tma_do, tDOg[None, c, iv], tDOs[None, st], tma_bar_ptr=bar)
                    cute.copy(tma_v, tVg[None, c, iv], tVs[None, st], tma_bar_ptr=bar)
                    cute.copy(tma_dv, tDVg[None, c, iv], tDVs[None, st], tma_bar_ptr=bar)
                    cute.copy(tma_h, tHg[None, iv, c], tHs[None, st], tma_bar_ptr=bar)
                    cute.copy(tma_dh, tDHg[None, iv, c], tDHs[None, st], tma_bar_ptr=bar)
                    in_producer.advance()

            in_pipe.producer_tail(in_producer)

        # ==========================================================================
        # TMA warp 2: the qk-stream (q/k/qt/kt/g2). Blocks on the previous chunk's
        # epilogue consumers, but only stalls itself.
        # ==========================================================================
        elif warp_idx == self.qk_warp_id:
            thr_mma_dsk = mma_dsk.get_slice(0)
            tQT_mma = thr_mma_dsk.partition_B(gQT)
            tKT_mma = thr_mma_dsk.partition_B(gKT)

            cta1 = cute.make_layout(1)
            tQTs, tQTg = cpasync.tma_partition(
                tma_qt, 0, cta1, cute.group_modes(sQT, 0, 3), cute.group_modes(tQT_mma, 0, 3)
            )
            tKTs, tKTg = cpasync.tma_partition(
                tma_kt, 0, cta1, cute.group_modes(sKT, 0, 3), cute.group_modes(tKT_mma, 0, 3)
            )
            tG2s, tG2g = cpasync.tma_partition(
                tma_g2, 0, cta1, cute.group_modes(sG2, 0, 1), cute.group_modes(gG2, 0, 1)
            )

            qk_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.qk_stages
            )

            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                qk_pipe.producer_acquire(qk_producer)
                qbar = qk_pipe.producer_get_barrier(qk_producer)
                qst = qk_producer.index
                cute.copy(tma_qt, tQTg[None, c], tQTs[None, qst], tma_bar_ptr=qbar)
                cute.copy(tma_kt, tKTg[None, c], tKTs[None, qst], tma_bar_ptr=qbar)
                cute.copy(tma_g2, tG2g[None, c], tG2s[None, qst], tma_bar_ptr=qbar)
                qk_producer.advance()

            qk_pipe.producer_tail(qk_producer)

        # ==========================================================================
        # Reducer warp 3: sum(h.dh) through the flat paired views, so B never touches
        # the in-stream and TMA is throttled only by MMA's prompt releases. Partials
        # go to parity-buffered smem; B's thread 0 folds them into dg_last.
        # ==========================================================================
        elif warp_idx == self.red_warp_id:
            io_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), io)
            lane = tidx % 32
            rHf = cute.make_rmem_tensor((8, 8), io)
            rDHf = cute.make_rmem_tensor((8, 8), io)
            acc8 = cute.make_rmem_tensor((8,), f32)

            in_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.in_stages
            )

            for i in cutlass.range(cnt, unroll=1):
                for e in cutlass.range(8, unroll_full=True):
                    acc8[e] = cutlass.Float32(0.0)
                for iv in cutlass.range(NVI, unroll_full=True):
                    in_pipe.consumer_wait(in_consumer)
                    st = in_consumer.index
                    for part in cutlass.range(4, unroll_full=True):
                        col = lane + 32 * part
                        cute.copy(io_cp_atom, sHflat[(None, None, col, st)], rHf)
                        cute.copy(io_cp_atom, sDHflat[(None, None, col, st)], rDHf)
                        for v in cutlass.range(8, unroll_full=True):
                            for e in cutlass.range(8, unroll_full=True):
                                acc8[e] += rHf[(e, v)].to(f32) * rDHf[(e, v)].to(f32)
                    in_pipe.consumer_release(in_consumer, pipeline.PipelineOp.AsyncThread)
                    in_consumer.advance()
                hdh = cutlass.Float32(0.0)
                for e in cutlass.range(8, unroll_full=True):
                    hdh += acc8[e]
                sScr[385 + (i % 2) * 32 + lane] = hdh
                cute.arch.fence_proxy("async.shared", space="cta")
                self.red_sync_barrier.arrive_and_wait()

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrDO_ds = mma_ds.make_fragment_A(sDO)
            tCrV_ds = mma_ds.make_fragment_B(sV)
            tCrDO = mma_dqh.make_fragment_A(sDO)
            tCrV = mma_dqh.make_fragment_A(sV)
            tCrDVf = mma_dqh.make_fragment_A(sDV)
            tCrH = mma_dqh.make_fragment_B(sH)
            tCrDH = mma_dqh.make_fragment_B(sDH)
            tCrDS = mma_dsk.make_fragment_A(sDS)  # stage 0 = ds, stage 1 = ds^T
            tCrQT = mma_dsk.make_fragment_B(sQT)
            tCrKT = mma_dsk.make_fragment_B(sKT)

            in_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.in_stages
            )
            qk_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qk_stages
            )
            ds_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dwrd_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dsf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dq1f_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dskf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dk1f_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dwf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dstf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for i in cutlass.range(cnt, unroll=1):
                for iv in cutlass.range(NVI, unroll_full=True):
                    in_pipe.consumer_wait(in_consumer)
                    st = in_consumer.index
                    # acc-free waits sit directly before the first gemm that writes each
                    # acc, so a gemm group only ever waits on its own acc's last reader —
                    # all of which are prompt in the SIMT groups' orderings.
                    if iv == 0:
                        dsf_pipe.producer_acquire(dsf_producer)
                    # DS += do @ v^T
                    for kk in cutlass.range(cute.size(tCrV_ds, mode=[2]), unroll_full=True):
                        mma_ds.set(
                            tcgen05.Field.ACCUMULATE, cutlass.Boolean(iv != 0 or kk != 0)
                        )
                        cute.gemm(
                            mma_ds, tDS[None, None, None, 0],
                            tCrDO_ds[None, None, kk, st],
                            tCrV_ds[None, None, kk, st],
                            tDS[None, None, None, 0],
                        )
                    # DQ1 += do @ h (columns also hold DST: wait both of their readers)
                    if iv == 0:
                        dq1f_pipe.producer_acquire(dq1f_producer)
                        dstf_pipe.producer_acquire(dstf_producer)
                    for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                        mma_dqh.set(
                            tcgen05.Field.ACCUMULATE, cutlass.Boolean(iv != 0 or kk != 0)
                        )
                        cute.gemm(
                            mma_dqh, tDQ[None, None, None, 0],
                            tCrDO[None, None, kk, st],
                            tCrH[None, None, kk, st],
                            tDQ[None, None, None, 0],
                        )
                    # DK1 += v @ dh
                    if iv == 0:
                        dk1f_pipe.producer_acquire(dk1f_producer)
                    for kk in cutlass.range(cute.size(tCrDH, mode=[2]), unroll_full=True):
                        mma_dqh.set(
                            tcgen05.Field.ACCUMULATE, cutlass.Boolean(iv != 0 or kk != 0)
                        )
                        cute.gemm(
                            mma_dqh, tDK[None, None, None, 0],
                            tCrV[None, None, kk, st],
                            tCrDH[None, None, kk, st],
                            tDK[None, None, None, 0],
                        )
                    # DW += dv @ h (columns also hold DSK: wait both of their readers)
                    if iv == 0:
                        dwf_pipe.producer_acquire(dwf_producer)
                        dskf_pipe.producer_acquire(dskf_producer)
                    for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                        mma_dqh.set(
                            tcgen05.Field.ACCUMULATE, cutlass.Boolean(iv != 0 or kk != 0)
                        )
                        cute.gemm(
                            mma_dqh, tDW[None, None, None, 0],
                            tCrDVf[None, None, kk, st],
                            tCrH[None, None, kk, st],
                            tDW[None, None, None, 0],
                        )
                    in_pipe.consumer_release(in_consumer, pipeline.PipelineOp.TCGen05Mma)
                    in_consumer.advance()

                dsf_pipe.producer_commit(dsf_producer)
                dsf_producer.advance()
                dq1f_pipe.producer_commit(dq1f_producer)
                dq1f_producer.advance()
                dk1f_pipe.producer_commit(dk1f_producer)
                dk1f_producer.advance()
                dwf_pipe.producer_commit(dwf_producer)
                dwf_producer.advance()

                # second pass: DST (ds^T @ q^T into DQ1's columns) FIRST — its readout
                # by B gates chunk c+1's DQ1 gemms, and DSK's completion is what allows
                # A's dq_out write to alias the ds/dst bytes — then DSK (ds @ k^T into
                # DW's columns).
                qk_pipe.consumer_wait(qk_consumer)
                qst = qk_consumer.index
                ds_pipe.consumer_wait(ds_consumer)  # also: A finished reading DQ1
                for kk in cutlass.range(cute.size(tCrQT, mode=[2]), unroll_full=True):
                    mma_dsk.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dsk, tDST[None, None, None, 0],
                        tCrDS[None, None, kk, 1],
                        tCrQT[None, None, kk, qst],
                        tDST[None, None, None, 0],
                    )
                dstf_pipe.producer_commit(dstf_producer)
                dstf_producer.advance()

                dw_rd.consumer_wait(dwrd_consumer)  # B finished reading DW
                for kk in cutlass.range(cute.size(tCrKT, mode=[2]), unroll_full=True):
                    mma_dsk.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dsk, tDSK[None, None, None, 0],
                        tCrDS[None, None, kk, 0],
                        tCrKT[None, None, kk, qst],
                        tDSK[None, None, None, 0],
                    )
                dskf_pipe.producer_commit(dskf_producer)
                dskf_producer.advance()

                dw_rd.consumer_release(dwrd_consumer)
                dwrd_consumer.advance()
                ds_pipe.consumer_release(ds_consumer)
                ds_consumer.advance()
                qk_pipe.consumer_release(qk_consumer, pipeline.PipelineOp.TCGen05Mma)
                qk_consumer.advance()

            dsf_pipe.producer_tail(dsf_producer)
            dq1f_pipe.producer_tail(dq1f_producer)
            dskf_pipe.producer_tail(dskf_producer)
            dk1f_pipe.producer_tail(dk1f_producer)
            dwf_pipe.producer_tail(dwf_producer)
            dstf_pipe.producer_tail(dstf_producer)

        # ==========================================================================
        # SIMT group A (warps 4..7): ds build, dq scale+add+store, rowsum(dq.q)
        # ==========================================================================
        elif (
            warp_idx == self.a_warp_id[0]
            or warp_idx == self.a_warp_id[1]
            or warp_idx == self.a_warp_id[2]
            or warp_idx == self.a_warp_id[3]
        ):
            t2r_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), f32
            )
            f32_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), f32)

            # --- DS -> ds/ds^T ---
            tDS_2d = tDS[((None, None), 0, 0, None)]
            tiled_t2r_ds = tcgen05.make_tmem_copy(t2r_atom, tDS_2d[None, None, 0])
            thr_t2r_ds = tiled_t2r_ds.get_slice(local_tidx)
            tTR_tDS = thr_t2r_ds.partition_S(tDS_2d)
            coordDS = thr_t2r_ds.partition_D(cute.make_identity_tensor((BT, BT)))
            tTR_rDS = cute.make_rmem_tensor(coordDS.shape, f32)
            tDecDS = cute.make_rmem_tensor(tTR_rDS.shape, f32)
            sG2_row64 = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BT, self.qk_stages), stride=(0, 1, BT)),
            )
            sG2_col64 = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BT, self.qk_stages), stride=(1, 0, BT)),
            )
            tDSsG2r = thr_t2r_ds.partition_D(sG2_row64)
            tDSrG2r = cute.make_rmem_tensor(
                cute.slice_(tDSsG2r.shape, (None, None, None, 0)), f32
            )
            tDSsG2c = thr_t2r_ds.partition_D(sG2_col64)
            tDSrG2c = cute.make_rmem_tensor(
                cute.slice_(tDSsG2c.shape, (None, None, None, 0)), f32
            )
            # ds -> ROW_MAJOR epi view (fwd's h pattern); ds^T -> COL_MAJOR epi view with
            # stmatrix transpose (fwd's vp pattern). Same rmem values feed both.
            r2s_ds_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            r2s_dst_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ds = cute.make_tiled_copy_D(r2s_ds_atom, tiled_t2r_ds)
            tiled_r2s_dst = cute.make_tiled_copy_D(r2s_dst_atom, tiled_t2r_ds)
            thr_r2s_ds = tiled_r2s_ds.get_slice(local_tidx)
            thr_r2s_dst = tiled_r2s_dst.get_slice(local_tidx)
            tRS_sDS = thr_r2s_ds.partition_D(sDS_epi)
            tRS_sDST = thr_r2s_dst.partition_D(sDST_epi)
            tRS_rDS = cute.make_rmem_tensor(
                cute.slice_(tRS_sDS.shape, (None, None, None, 0)), io
            )
            tRS_rDST = cute.make_rmem_tensor(
                cute.slice_(tRS_sDST.shape, (None, None, None, 0)), io
            )

            # --- DQ (DQ1 then DSK) -> dq ---
            tDQ_2d = tDQ[((None, None), 0, 0, None)]
            tiled_t2r_dq = tcgen05.make_tmem_copy(t2r_atom, tDQ_2d[None, None, 0])
            thr_t2r_dq = tiled_t2r_dq.get_slice(local_tidx)
            tTR_tDQ = thr_t2r_dq.partition_S(tDQ_2d)
            tDSK_2d = tDSK[((None, None), 0, 0, None)]
            tTR_tDSK = thr_t2r_dq.partition_S(tDSK_2d)
            coordDQ = thr_t2r_dq.partition_D(cute.make_identity_tensor((BT, K)))
            tTR_rDQ = cute.make_rmem_tensor(coordDQ.shape, f32)
            tDQpart = cute.make_rmem_tensor(tTR_rDQ.shape, f32)
            sG2_rowk = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, K, self.qk_stages), stride=(1, 0, BT)),
            )
            tDQsG2 = thr_t2r_dq.partition_D(sG2_rowk)
            tDQrG2 = cute.make_rmem_tensor(
                cute.slice_(tDQsG2.shape, (None, None, None, 0)), f32
            )
            r2s_out_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_out = cute.make_tiled_copy_D(r2s_out_atom, tiled_t2r_dq)
            thr_r2s_out = tiled_r2s_out.get_slice(local_tidx)
            tRS_sOutA = thr_r2s_out.partition_D(sOutA)
            tRS_rOutA = cute.make_rmem_tensor(
                cute.slice_(tRS_sOutA.shape, (None, None, None, 0)), io
            )

            bSG_sOutA, bSG_gDQ = cpasync.tma_partition(
                tma_dq, 0, cute.make_layout(1),
                cute.group_modes(sOutA, 0, 2), cute.group_modes(gDQ, 0, 2),
            )
            tma_store_a = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, a_threads),
            )

            qk_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qk_stages
            )
            dsf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dq1f_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dskf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            ds_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            rr = local_tidx >> 1  # rowsum: 2 threads per row
            cc0 = (local_tidx & 1) * (K // 2)
            accA4 = cute.make_rmem_tensor((4,), f32)

            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                qk_pipe.consumer_wait(qk_consumer)
                qst = qk_consumer.index
                qcrd = (None, None, None, qst)

                # Readouts of the v-loop accs come FIRST and back to back: their releases
                # are what chunk c+1's v-loop gemms wait on, so nothing may sit before
                # them. The ds build and everything after is off the inter-chunk path.
                dsf_pipe.consumer_wait(dsf_consumer)
                cute.copy(tiled_t2r_ds, tTR_tDS[None, None, None, 0], tTR_rDS)
                cute.arch.fence_view_async_tmem_load()
                dsf_pipe.consumer_release(dsf_consumer)
                dsf_consumer.advance()
                dq1f_pipe.consumer_wait(dq1f_consumer)
                cute.copy(tiled_t2r_dq, tTR_tDQ[None, None, None, 0], tTR_rDQ)
                cute.arch.fence_view_async_tmem_load()
                dq1f_pipe.consumer_release(dq1f_consumer)
                dq1f_consumer.advance()

                # ds = tril(DS * exp2(g_i - g_j)) * scale, cast to io dtype. Committing
                # ds_pipe only now (after the DQ1 read above) is what makes the DST
                # overwrite of DQ1's columns safe.
                cute.copy(f32_cp_atom, tDSsG2r[qcrd], tDSrG2r)
                cute.copy(f32_cp_atom, tDSsG2c[qcrd], tDSrG2c)
                for j in cutlass.range(cute.size(tTR_rDS), unroll_full=True, vectorize=True):
                    tDecDS[j] = tDSrG2c[j] - tDSrG2r[j]
                for j in cutlass.range(cute.size(tTR_rDS), unroll_full=True):
                    mi, nj = coordDS[j]
                    if mi < nj:
                        tDecDS[j] = cutlass.Float32(-float("inf"))
                for j in cutlass.range(cute.size(tTR_rDS), unroll_full=True, vectorize=True):
                    d = cute.math.exp2(tDecDS[j], fastmath=True)
                    ds_ij = (tTR_rDS[j] * d * scale).to(io)
                    tRS_rDS[j] = ds_ij
                    tRS_rDST[j] = ds_ij
                ds_pipe.producer_acquire(ds_producer)
                cute.copy(tiled_r2s_ds, tRS_rDS, tRS_sDS[None, None, None, 0])
                cute.copy(tiled_r2s_dst, tRS_rDST, tRS_sDST[None, None, None, 1])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.a_sync_barrier.arrive_and_wait()
                ds_pipe.producer_commit(ds_producer)
                ds_producer.advance()

                # dq_part = DQ1 * exp2(g_i) * scale — off the second-pass critical path,
                # so it runs while the MMA warp works through DST/DSK.
                cute.copy(f32_cp_atom, tDQsG2[qcrd], tDQrG2)
                for j in cutlass.range(cute.size(tTR_rDQ), unroll_full=True, vectorize=True):
                    gd = cute.math.exp2(tDQrG2[j], fastmath=True)
                    tDQpart[j] = tTR_rDQ[j] * gd * scale

                # dq = dq_part + ds @ k (DSK lands in DW's columns)
                dskf_pipe.consumer_wait(dskf_consumer)
                cute.copy(tiled_t2r_dq, tTR_tDSK[None, None, None, 0], tTR_rDQ)
                cute.arch.fence_view_async_tmem_load()
                dskf_pipe.consumer_release(dskf_consumer)
                dskf_consumer.advance()
                for j in cutlass.range(cute.size(tTR_rDQ), unroll_full=True, vectorize=True):
                    tRS_rOutA[j] = (tDQpart[j] + tTR_rDQ[j]).to(io)
                cute.copy(tiled_r2s_out, tRS_rOutA, tRS_sOutA[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.a_sync_barrier.arrive_and_wait()
                if warp_idx == self.a_warp_id[0]:
                    cute.copy(tma_dq, bSG_sOutA[None, 0], bSG_gDQ[None, c])
                    tma_store_a.producer_commit()
                    tma_store_a.producer_acquire()

                # rowsum(dq . q) from the staging buffer (bf16-rounded dq, see docstring)
                # q values come from the swizzled qt buffer, transposed coords — the
                # bytes are already resident for the DST mma, no separate plain copy.
                for e in cutlass.range(4, unroll_full=True):
                    accA4[e] = cutlass.Float32(0.0)
                for jo in cutlass.range(K // 8, unroll=2):
                    for ji in cutlass.range(4, unroll_full=True):
                        cj = cc0 + jo * 4 + ji
                        accA4[ji] += (
                            sOutA[(rr, cj, 0)].to(f32) * sQTv[(cj, rr, qst)].to(f32)
                        )
                sScr[local_tidx] = (accA4[0] + accA4[1]) + (accA4[2] + accA4[3])
                cute.arch.fence_proxy("async.shared", space="cta")
                # release before bar4: A's last qk use is the rowsum above. The sScr
                # cross-chunk safety argument only needs B's (late) qk release.
                qk_pipe.consumer_release(qk_consumer, pipeline.PipelineOp.AsyncThread)
                qk_consumer.advance()
                self.ab_sync_barrier.arrive_and_wait()

            tma_store_a.producer_tail()

        # ==========================================================================
        # SIMT group B (warps 8..11): sum(h.dh), dk, dw, dg
        # ==========================================================================
        elif (
            warp_idx == self.b_warp_id[0]
            or warp_idx == self.b_warp_id[1]
            or warp_idx == self.b_warp_id[2]
            or warp_idx == self.b_warp_id[3]
        ):
            t2r_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), f32
            )
            f32_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), f32)
            io_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), io)

            tDK_2d = tDK[((None, None), 0, 0, None)]
            tiled_t2r_dk = tcgen05.make_tmem_copy(t2r_atom, tDK_2d[None, None, 0])
            thr_t2r_dk = tiled_t2r_dk.get_slice(local_tidx)
            tTR_tDK = thr_t2r_dk.partition_S(tDK_2d)
            tDW_2d = tDW[((None, None), 0, 0, None)]
            tTR_tDW = thr_t2r_dk.partition_S(tDW_2d)
            tDST_2d = tDST[((None, None), 0, 0, None)]
            tTR_tDST = thr_t2r_dk.partition_S(tDST_2d)
            coordDK = thr_t2r_dk.partition_D(cute.make_identity_tensor((BT, K)))
            tTR_rDK = cute.make_rmem_tensor(coordDK.shape, f32)
            tDKpart = cute.make_rmem_tensor(tTR_rDK.shape, f32)
            sG2_rowk = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, K, self.qk_stages), stride=(1, 0, BT)),
            )
            tDKsG2 = thr_t2r_dk.partition_D(sG2_rowk)
            tDKrG2 = cute.make_rmem_tensor(
                cute.slice_(tDKsG2.shape, (None, None, None, 0)), f32
            )
            r2s_out_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_out = cute.make_tiled_copy_D(r2s_out_atom, tiled_t2r_dk)
            thr_r2s_out = tiled_r2s_out.get_slice(local_tidx)
            tRS_sOutB = thr_r2s_out.partition_D(sOutB)
            tRS_rOutB = cute.make_rmem_tensor(
                cute.slice_(tRS_sOutB.shape, (None, None, None, 0)), io
            )

            bSG_sOutB_dk, bSG_gDK = cpasync.tma_partition(
                tma_dk, 0, cute.make_layout(1),
                cute.group_modes(sOutB, 0, 2), cute.group_modes(gDK, 0, 2),
            )
            bSG_sOutB_dw, bSG_gDW = cpasync.tma_partition(
                tma_dw, 0, cute.make_layout(1),
                cute.group_modes(sOutB, 0, 2), cute.group_modes(gDW, 0, 2),
            )
            tma_store_b = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, b_threads),
            )

            qk_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.qk_stages
            )
            dk1f_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dwf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dstf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dwrd_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            rr = local_tidx >> 1
            cc0 = (local_tidx & 1) * (K // 2)
            dkk4 = cute.make_rmem_tensor((4,), f32)
            tot4 = cute.make_rmem_tensor((4,), f32)
            acc4 = cute.make_rmem_tensor((4,), f32)

            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i

                qk_pipe.consumer_wait(qk_consumer)
                qst = qk_consumer.index
                qcrd = (None, None, None, qst)
                g_last = sG2[BT - 1, qst]
                exp_g_last = cute.math.exp2(g_last, fastmath=True)

                # dk_part = DK1 * exp2(G - g_i); dg_last partial = sum(dk_part . k).
                # DK1's release is on chunk c+1's critical path — t2r, release, then work.
                dk1f_pipe.consumer_wait(dk1f_consumer)
                cute.copy(tiled_t2r_dk, tTR_tDK[None, None, None, 0], tTR_rDK)
                cute.arch.fence_view_async_tmem_load()
                dk1f_pipe.consumer_release(dk1f_consumer)
                dk1f_consumer.advance()
                cute.copy(f32_cp_atom, tDKsG2[qcrd], tDKrG2)
                for e in cutlass.range(4, unroll_full=True):
                    dkk4[e] = cutlass.Float32(0.0)
                for j in cutlass.range(cute.size(tTR_rDK), unroll_full=True):
                    gd = cute.math.exp2(g_last - tDKrG2[j], fastmath=True)
                    tDKpart[j] = tTR_rDK[j] * gd
                    tt, kk2 = coordDK[j]
                    dkk4[j % 4] += tDKpart[j] * sKTv[(kk2, tt, qst)].to(f32)
                dkk = (dkk4[0] + dkk4[1]) + (dkk4[2] + dkk4[3])

                # dw = -DW; reading DW frees its columns for the DSK overwrite (dw_rd)
                dwf_pipe.consumer_wait(dwf_consumer)
                cute.copy(tiled_t2r_dk, tTR_tDW[None, None, None, 0], tTR_rDK)
                cute.arch.fence_view_async_tmem_load()
                dwf_pipe.consumer_release(dwf_consumer)
                dwf_consumer.advance()
                dw_rd.producer_acquire(dwrd_producer)
                dw_rd.producer_commit(dwrd_producer)
                dwrd_producer.advance()
                for j in cutlass.range(cute.size(tTR_rDK), unroll_full=True, vectorize=True):
                    tRS_rOutB[j] = (-tTR_rDK[j]).to(io)
                cute.copy(tiled_r2s_out, tRS_rOutB, tRS_sOutB[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.b_sync_barrier.arrive_and_wait()
                if warp_idx == self.b_warp_id[0]:
                    cute.copy(tma_dw, bSG_sOutB_dw[None, 0], bSG_gDW[None, c])
                    tma_store_b.producer_commit()
                    tma_store_b.producer_acquire()

                # one scalar reduction covers dg_last entirely; thread 0's serial sum
                # overlaps the DST wait below. The b-barrier also orders b0's dw-store
                # acquire before the dk overwrite of sOutB; the red-barrier synchronizes
                # with the reducer warp's h.dh partials for this chunk.
                sScr[256 + local_tidx] = dkk
                cute.arch.fence_proxy("async.shared", space="cta")
                self.b_sync_barrier.arrive_and_wait()
                self.red_sync_barrier.arrive_and_wait()
                if local_tidx == 0:
                    for e in cutlass.range(4, unroll_full=True):
                        tot4[e] = cutlass.Float32(0.0)
                    for jo in cutlass.range(32, unroll=4):
                        for ji in cutlass.range(4, unroll_full=True):
                            tot4[ji] += sScr[256 + jo * 4 + ji]
                    for jo in cutlass.range(8, unroll=2):
                        for ji in cutlass.range(4, unroll_full=True):
                            tot4[ji] += (
                                exp_g_last * sScr[385 + (i % 2) * 32 + jo * 4 + ji]
                            )
                    sScr[384] = (tot4[0] + tot4[1]) + (tot4[2] + tot4[3])
                cute.arch.fence_proxy("async.shared", space="cta")

                # dk = dk_part + ds^T @ q (DST lands in DQ1's columns)
                dstf_pipe.consumer_wait(dstf_consumer)
                cute.copy(tiled_t2r_dk, tTR_tDST[None, None, None, 0], tTR_rDK)
                cute.arch.fence_view_async_tmem_load()
                dstf_pipe.consumer_release(dstf_consumer)
                dstf_consumer.advance()
                for j in cutlass.range(cute.size(tTR_rDK), unroll_full=True, vectorize=True):
                    tRS_rOutB[j] = (tDKpart[j] + tTR_rDK[j]).to(io)
                cute.copy(tiled_r2s_out, tRS_rOutB, tRS_sOutB[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.b_sync_barrier.arrive_and_wait()

                # rowsum(dk . k), then dk store
                for e in cutlass.range(4, unroll_full=True):
                    acc4[e] = cutlass.Float32(0.0)
                for jo in cutlass.range(K // 8, unroll=2):
                    for ji in cutlass.range(4, unroll_full=True):
                        cj = cc0 + jo * 4 + ji
                        acc4[ji] += (
                            sOutB[(rr, cj, 0)].to(f32) * sKTv[(cj, rr, qst)].to(f32)
                        )
                sScr[128 + local_tidx] = (acc4[0] + acc4[1]) + (acc4[2] + acc4[3])
                cute.arch.fence_proxy("async.shared", space="cta")
                if warp_idx == self.b_warp_id[0]:
                    cute.copy(tma_dk, bSG_sOutB_dk[None, 0], bSG_gDK[None, c])
                    tma_store_b.producer_commit()
                    tma_store_b.producer_acquire()

                # dg = rowsum(dq.q) - rowsum(dk.k) (+ dg_last on the last row)
                self.ab_sync_barrier.arrive_and_wait()
                if local_tidx < BT:
                    r = local_tidx
                    val = (sScr[2 * r] + sScr[2 * r + 1]) - (
                        sScr[128 + 2 * r] + sScr[128 + 2 * r + 1]
                    )
                    if r == BT - 1:
                        val += sScr[384]
                    mDG[(c * BT + r, hv_idx, b_idx)] = val

                qk_pipe.consumer_release(qk_consumer, pipeline.PipelineOp.AsyncThread)
                qk_consumer.advance()

            tma_store_b.producer_tail()

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)
        return


# --------------------------------------------------------------------------------------------
# host wrapper — same layout-keyed call cache + ctypes pointer retargeting as the fwd
# --------------------------------------------------------------------------------------------

_COMPILE_CACHE: dict = {}
_CALL_CACHE: dict = {}


def _cute_view(t: torch.Tensor, perm: tuple[int, ...], dyn_modes: tuple[int, ...]):
    # See kernel_fwd._cute_view: stride order from the unpermuted tensor mapped through perm.
    t = t.detach()
    base_order = sorted(range(t.dim()), key=lambda i: -t.stride(i))
    new_of_old = {old: new for new, old in enumerate(perm)}
    stride_order = tuple(new_of_old[d] for d in base_order)
    tt = t.permute(*perm)
    ct = from_dlpack(tt, assumed_align=16)
    for m in dyn_modes:
        ct = ct.mark_compact_shape_dynamic(mode=m, stride_order=stride_order)
    return ct


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def _spec(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.dtype)


def _alloc(specs: tuple, device: torch.device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


def _call_key(*tensors, extra=()):
    def sig(t):
        return (t.shape, t.stride(), t.dtype)

    return tuple(sig(t) for t in tensors) + tuple(extra) + (
        torch.cuda.current_stream().cuda_stream,
    )


def _pick_nseg(B: int, HV: int, NT: int) -> int:
    # enough CTAs for ~7+ waves on ~150 SMs, then re-derive so every segment is non-empty
    import os

    target = int(os.environ.get("GDN_DQKWG_CTAS", "1024"))
    nseg = min(NT, max(1, -(-target // (B * HV))))
    cpc = -(-NT // nseg)
    return -(-NT // cpc)


def gdn_cute_bwd_dqkwg_call(
    q: torch.Tensor,  # [B,T,H,K]
    k: torch.Tensor,
    v: torch.Tensor,  # [B,T,HV,V] — v_new
    do: torch.Tensor,  # [B,T,HV,V]
    dv: torch.Tensor,  # [B,T,HV,V]
    h: torch.Tensor,  # [B,NT,HV,K,V]
    dh: torch.Tensor,  # [B,NT,HV,K,V]
    g2: torch.Tensor,  # [B,T,HV] fp32
    scale: float,
):
    key = _call_key(q, k, v, do, dv, h, dh, extra=(scale,))
    ent = _CALL_CACHE.get(key)
    if ent is None:
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        NT = T // 64
        assert T % 64 == 0

        dq = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
        dk = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
        dw = torch.empty(B, T, HV, K, device=q.device, dtype=q.dtype)
        dg = torch.empty(B, T, HV, device=q.device, dtype=torch.float32)
        g2c = torch.empty(B, HV, NT, 64, device=g2.device, dtype=g2.dtype)

        io_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V, HV, H)

        cdo = _cute_view(do, (1, 3, 2, 0), (0, 2, 3))
        cv = _cute_view(v, (1, 3, 2, 0), (0, 2, 3))
        cdv = _cute_view(dv, (1, 3, 2, 0), (0, 2, 3))
        ch = _cute_view(h, (3, 4, 1, 2, 0), (2, 3, 4))
        cdh = _cute_view(dh, (3, 4, 1, 2, 0), (2, 3, 4))
        cq = _cute_view(q, (1, 3, 2, 0), (0, 2, 3))
        ck = _cute_view(k, (1, 3, 2, 0), (0, 2, 3))
        cqt = _cute_view(q, (3, 1, 2, 0), (1, 2, 3))
        ckt = _cute_view(k, (3, 1, 2, 0), (1, 2, 3))
        cg2 = _cute_view(g2c, (3, 2, 1, 0), (1, 2, 3))
        cdq = _cute_view(dq, (1, 3, 2, 0), (0, 2, 3))
        cdk = _cute_view(dk, (1, 3, 2, 0), (0, 2, 3))
        cdw = _cute_view(dw, (1, 3, 2, 0), (0, 2, 3))
        cdg = _cute_view(dg, (1, 2, 0), (0, 2))

        nseg = _pick_nseg(B, HV, NT)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = GdnBwdDqkwgKernel(io_dtype, K, V)
            compiled = cute.compile(
                kernel_obj, cdo, cv, cdv, ch, cdh, cq, ck, cqt, ckt, cg2,
                cdq, cdk, cdw, cdg,
                cutlass.Float32(scale), cutlass.Int32(nseg), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (cdo, cv, cdv, ch, cdh, cq, ck, cqt, ckt, cg2, cdq, cdk, cdw, cdg,
                cutlass.Float32(scale), cutlass.Int32(nseg), stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        ent = (compiled, args, (_spec(dq), _spec(dk), _spec(dw), _spec(dg)), g2c)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, g2c = ent
    cdo, cv, cdv, ch, cdh, cq, ck, cqt, ckt, _, cdq, cdk, cdw, cdg, _, _, _ = args
    # Fresh outputs per call - see the package docstring on why they are not cache-owned.
    dq, dk, dw, dg = _alloc(out_specs, q.device)
    _retarget(cdq, dq)
    _retarget(cdk, dk)
    _retarget(cdw, dw)
    _retarget(cdg, dg)
    _retarget(cdo, do)
    _retarget(cv, v)
    _retarget(cdv, dv)
    _retarget(ch, h)
    _retarget(cdh, dh)
    _retarget(cq, q)
    _retarget(ck, k)
    _retarget(cqt, q)
    _retarget(ckt, k)
    B, HV, NT, BT = g2c.shape
    g2c.view(B, HV, NT * BT).copy_(g2.transpose(1, 2))
    compiled(*args)
    return dq, dk, dw, dg


def chunk_bwd_dqkwg_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
    **kw,
):
    """Drop-in for fla's chunk_bwd_dqkwg on the shapes this idea targets; falls back
    to fla otherwise so the stage table never breaks."""
    B, T, H, K = q.shape
    HV = v.shape[2]
    supported = (
        chunk_size == 64 and T % 64 == 0 and K == 128 and v.shape[3] % 64 == 0
        and g is not None and w is not None and dv is not None and not kw.get("cu_seqlens")
        and q.dtype in (torch.bfloat16, torch.float16) and g.dtype == torch.float32
    )
    if not supported:
        from fla.ops.common.chunk_o import chunk_bwd_dqkwg

        return chunk_bwd_dqkwg(
            q=q, k=k, v=v, do=do, h=h, dh=dh, w=w, g=g, dv=dv,
            scale=scale, chunk_size=chunk_size, **kw,
        )

    dq, dk, dw, dg = gdn_cute_bwd_dqkwg_call(
        q, k, v, do, dv, h, dh, g, float(scale if scale is not None else K**-0.5)
    )
    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dq, dk, dw, dg
