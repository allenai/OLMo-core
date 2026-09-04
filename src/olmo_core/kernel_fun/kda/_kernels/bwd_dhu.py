"""Phase 2 B2a — CuTe port of fla's `chunk_gated_delta_rule_bwd_dhu` (stage 4), the
reverse state scan. No fusion: B1's dq experiment measured the cross-CTA reduce tax at
~2x DRAM per fused K-shaped output (ALGORITHM.md "B1 verdict"), so this port keeps
fla's stage boundary and takes the scan win alone (gdn 003's dhu port was its 1.87x
stage; kernel_scan.py mirrored is the template).

Per (b, hv, v_tile) CTA, dh^T [BV, K] fp32 resident in SIMT registers (init dht),
chunks newest first; per chunk c = NT-1 .. 0:

    dhck[c] = dh                     (bf16 -> HBM pre-update, COL_MAJOR trans staging)
    DV  = kg @ dh_c^T                (MMA; kg pre-scaled by recompute — no gate here)
    dv2 = DV + dv_in                 (SIMT) -> dv2 HBM + WD's A operand (bf16)
    QD  = do^T @ qg                  (MMA, both operands straight TMA loads)
    WD  = dv2^T @ w                  (MMA)
    dh  = exp2(gd) * dh + scale*QD - WD   (SIMT regs, per-dim K-vector decay)
    after chunk 0: dh0 = dh          (fp32 scatter)

Two UPD accumulators instead of gdn's folded dog = do*scale operand: fla multiplies the
fp32 q-side dot by scale AFTER accumulation, and keeping that order reproduces fla's
arithmetic exactly (only MMA-vs-tl.dot reassociation remains, ~1e-6) — and it makes
do^T a pure mn-major TMA operand (V is contiguous in gmem, so do^T [BV, BT] is
M-contiguous as-is; no SIMT scaling pass at all). Costs one extra [BV,K] accumulator
(tmem 320/512 cols) and one extra t2r per chunk, overlapped: QD commits while SIMT is
still assembling dv2.

Logical gmem mode order (M/N, K_contract, rest) per operand:
    kg:      (T, K, HV, B)      A of DV, k-major
    do^T:    (V, T, HV, B)      A of QD, mn-major (same storage as do)
    qg^T:    (K, T, HV, B)      B of QD, mn-major (same storage as qg)
    w^T:     (K, T, HV, B)      B of WD, mn-major (same storage as w)
    dv_in:   (T, V, HV, B)      SIMT-only, linear smem
    gd:      (K, NT, HV, B)     fp32 per-chunk decay vectors (g2 last rows)
    dht:     (K, V, HV, B)      fp32 scalar gather, once per CTA
    dv2:     (T, V, HV, B)      TMA store
    dhck:    (V, K, NT, HV, B)  TMA store (bf16), COL_MAJOR staging (TMA can't transpose)
    dh0:     (K, V, HV, B)      fp32 scalar scatter, once per CTA
"""

from __future__ import annotations

from typing import Type

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
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


class KdaB2aDhuKernel:
    """Reverse dh scan. One CTA per (b, hv, v_tile); serial over chunks, newest first.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT recurrence (dv2, dh update, dh0,
    checkpoint/dv2 stores). Warps 2,3 idle through the role branch and only join
    alloc/dealloc barriers.
    """

    def __init__(self, io_dtype: Type[cutlass.Numeric], K: int, V_TILE: int):
        self.io_dtype = io_dtype
        self.acc_dtype = cutlass.Float32
        self.BT = 64
        self.K = K
        self.BV = V_TILE

        assert K in (64, 128), "K must be 64 or 128"
        assert self.BV == 64, "V tile is 64"

        # MMA tile shapes (M, N, K_contract)
        self.tile_dv = (self.BT, self.BV, self.K)  # kg @ dh^T
        self.tile_upd = (self.BV, self.K, self.BT)  # do^T @ qg  /  dv2^T @ w

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0
        self.mma_warp_id = 1
        self.simt_warp_id = (4, 5, 6, 7)
        self.threads_per_cta = 32 * 8

        self.input_stages = 2
        self.dh_stages = 2

        self.simt_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * len(self.simt_warp_id)
        )
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    # ---------------------------------------------------------------------------------

    def _make_tiled_mmas(self):
        io, acc, grp = self.io_dtype, self.acc_dtype, self.cta_group
        mma_dv = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_dv[:2], tcgen05.OperandSource.SMEM,
        )
        # QD: A = do^T [BV, BT] mn-major (V contiguous in gmem == M contiguous here)
        mma_qd = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("mn"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_upd[:2], tcgen05.OperandSource.SMEM,
        )
        # WD: A = dv2^T [BV, BT] k-major (SIMT trans-stmatrix built)
        mma_wd = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_upd[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_dv, mma_qd, mma_wd

    def _setup_attributes(self):
        mma_dv, mma_qd, mma_wd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.kg_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dv, self.tile_dv, self.io_dtype, self.input_stages
        )
        self.dot_smem_layout = sm100_utils.make_smem_layout_a(
            mma_qd, self.tile_upd, self.io_dtype, self.input_stages
        )
        self.qgt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_qd, self.tile_upd, self.io_dtype, self.input_stages
        )
        self.wt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_wd, self.tile_upd, self.io_dtype, self.input_stages
        )
        # dv_in is SIMT-only: plain row-major (BT, BV), BV contiguous
        self.dvi_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        self.gd_smem_layout = cute.make_layout((K, self.input_stages))

        # dh^T as B operand of DV ([BV, K] k-major, staged); written by SIMT through
        # the ROW_MAJOR epi view of the same bytes.
        self.dh_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dv, self.tile_dv, self.io_dtype, self.dh_stages
        )
        self.dh_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BV, K), self.dh_stages
        )
        # checkpoint staging: COL_MAJOR (BV, K) in gmem order (TMA cannot transpose)
        self.cks_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BV, K), 1
        )
        # dv2 as A operand of WD ([BV, BT] k-major); written via transpose stmatrix
        # through the COL_MAJOR (BT, BV) epi view of the same bytes.
        self.dv2n_smem_layout = sm100_utils.make_smem_layout_a(
            mma_wd, self.tile_upd, self.io_dtype, 1
        )
        self.ops_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BV), 1
        )
        # dv2 store staging: (BT, BV) BV-contiguous to match gmem
        self.dv2s_smem_layout = cute.make_layout((BT, BV, 1), stride=(BV, 1, BT * BV))

        self.num_mma_load_bytes = sum(
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(lay, (None, None, None, 0))
            )
            for lay in (
                self.kg_smem_layout, self.dot_smem_layout,
                self.qgt_smem_layout, self.wt_smem_layout,
            )
        )
        self.num_simt_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.dvi_smem_layout, (None, None, 0))
            )
            + cute.size_in_bytes(
                cutlass.Float32, cute.slice_(self.gd_smem_layout, (None, 0))
            )
        )

        (
            self.tmem_dv_offset,
            self.tmem_qd_offset,
            self.tmem_wd_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_dv, mma_qd, mma_wd)

    def _plan_tmem(self, mma_dv, mma_qd, mma_wd):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        dv = acc_cols(mma_dv, self.tile_dv)
        qd = acc_cols(mma_qd, self.tile_upd)
        wd = acc_cols(mma_wd, self.tile_upd)
        off_dv = 0
        off_qd = off_dv + dv
        off_wd = off_qd + qd
        total_ = off_wd + wd
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_dv, off_qd, off_wd, total

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        kg: cute.Tensor,  # (T, K, HV, B)
        dot: cute.Tensor,  # (V, T, HV, B) — same storage as do
        qgt: cute.Tensor,  # (K, T, HV, B) — same storage as qg
        wt: cute.Tensor,  # (K, T, HV, B) — same storage as w
        dvi: cute.Tensor,  # (T, V, HV, B)
        gd: cute.Tensor,  # (K, NT, HV, B) fp32
        dht: cute.Tensor,  # (K, V, HV, B) fp32
        dv2: cute.Tensor,  # (T, V, HV, B) out
        dhck: cute.Tensor,  # (V, K, NT, HV, B) out, io dtype
        dh0: cute.Tensor,  # (K, V, HV, B) fp32 out
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_dv, mma_qd, mma_wd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_kg, tma_tensor_kg = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), kg,
            cute.slice_(self.kg_smem_layout, (None, None, None, 0)),
            self.tile_dv, mma_dv, cluster_vmnk,
        )
        tma_dot, tma_tensor_dot = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), dot,
            cute.slice_(self.dot_smem_layout, (None, None, None, 0)),
            self.tile_upd, mma_qd, cluster_vmnk,
        )
        tma_qgt, tma_tensor_qgt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), qgt,
            cute.slice_(self.qgt_smem_layout, (None, None, None, 0)),
            self.tile_upd, mma_qd, cluster_vmnk,
        )
        tma_wt, tma_tensor_wt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), wt,
            cute.slice_(self.wt_smem_layout, (None, None, None, 0)),
            self.tile_upd, mma_wd, cluster_vmnk,
        )
        tma_dvi, tma_tensor_dvi = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), dvi,
            cute.slice_(self.dvi_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        gd_cta_v_layout = cute.slice_(
            cute.make_identity_layout(gd.shape), (None, 0, 0, 0)
        )
        tma_gd, tma_tensor_gd = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gd,
            cute.slice_(self.gd_smem_layout, (None, 0)),
            gd_cta_v_layout,
        )
        tma_dv2, tma_tensor_dv2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dv2,
            cute.slice_(self.dv2s_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        tma_ck, tma_tensor_ck = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dhck,
            cute.slice_(self.cks_smem_layout, (None, None, 0)),
            (BV, K),
        )

        B = cute.size(kg, mode=[3])
        HV = cute.size(kg, mode=[2])
        NV = cute.size(dvi, mode=[1]) // BV
        grid = (B * HV * NV, 1, 1)

        swz_align, lin_align = 1024, 128

        # Every `*_full` range backs both halves of a pipeline's mbarrier array —
        # 2 * num_stages Int64s (under-sizing aliases the next pipeline).
        @cute.struct
        class SharedStorage:
            mmain_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            simtin_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            dh_full: cute.struct.MemRange[cutlass.Int64, self.dh_stages * 2]  # type: ignore
            dv2n_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dvf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            qdf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            wdf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_kg: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.kg_smem_layout)], swz_align  # type: ignore
            ]
            smem_dot: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dot_smem_layout)], swz_align  # type: ignore
            ]
            smem_qgt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.qgt_smem_layout)], swz_align  # type: ignore
            ]
            smem_wt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.wt_smem_layout)], swz_align  # type: ignore
            ]
            smem_dvi: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dvi_smem_layout)], lin_align  # type: ignore
            ]
            smem_gd: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.gd_smem_layout)], lin_align  # type: ignore
            ]
            smem_dh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dh_smem_layout)], swz_align  # type: ignore
            ]
            smem_cks: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.cks_smem_layout)], swz_align  # type: ignore
            ]
            smem_dv2n: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dv2n_smem_layout)], swz_align  # type: ignore
            ]
            smem_dv2s: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dv2s_smem_layout)], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
            )

        self.kda_cute_dhu(
            tma_kg, tma_tensor_kg,
            tma_dot, tma_tensor_dot,
            tma_qgt, tma_tensor_qgt,
            tma_wt, tma_tensor_wt,
            tma_dvi, tma_tensor_dvi,
            tma_gd, tma_tensor_gd,
            tma_dv2, tma_tensor_dv2,
            tma_ck, tma_tensor_ck,
            dht,
            dh0,
            scale,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def kda_cute_dhu(
        self,
        tma_kg: cute.CopyAtom, mKG: cute.Tensor,
        tma_dot: cute.CopyAtom, mDOT: cute.Tensor,
        tma_qgt: cute.CopyAtom, mQGT: cute.Tensor,
        tma_wt: cute.CopyAtom, mWT: cute.Tensor,
        tma_dvi: cute.CopyAtom, mDVI: cute.Tensor,
        tma_gd: cute.CopyAtom, mGd: cute.Tensor,
        tma_dv2: cute.CopyAtom, mDV2: cute.Tensor,
        tma_ck: cute.CopyAtom, mCK: cute.Tensor,
        mDHT: cute.Tensor,
        mDH0: cute.Tensor,
        scale: cutlass.Float32,
    ):
        BT, K, BV = self.BT, self.K, self.BV
        io = self.io_dtype
        f32 = self.acc_dtype
        # Layouts/TiledMma from the host trace cannot cross the region boundary — rebuild.
        self._setup_attributes()
        mma_dv, mma_qd, mma_wd = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_kg, tma_dot, tma_qgt, tma_wt, tma_dvi, tma_gd, tma_dv2, tma_ck]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mKG, mode=[2])
        NV = cute.size(mDVI, mode=[1]) // BV
        T = cute.size(mKG, mode=[0])
        NT = T // BT
        v_idx = bidx % NV
        hv_idx = (bidx // NV) % HV
        b_idx = bidx // (NV * HV)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sKG = storage.smem_kg.get_tensor(self.kg_smem_layout.outer, swizzle=self.kg_smem_layout.inner)
        sDOT = storage.smem_dot.get_tensor(self.dot_smem_layout.outer, swizzle=self.dot_smem_layout.inner)
        sQGT = storage.smem_qgt.get_tensor(self.qgt_smem_layout.outer, swizzle=self.qgt_smem_layout.inner)
        sWT = storage.smem_wt.get_tensor(self.wt_smem_layout.outer, swizzle=self.wt_smem_layout.inner)
        sDVI = storage.smem_dvi.get_tensor(self.dvi_smem_layout)
        sGd = storage.smem_gd.get_tensor(self.gd_smem_layout)
        sDH = storage.smem_dh.get_tensor(self.dh_smem_layout.outer, swizzle=self.dh_smem_layout.inner)
        sDH_epi = storage.smem_dh.get_tensor(self.dh_epi_layout.outer, swizzle=self.dh_epi_layout.inner)
        sCKS = storage.smem_cks.get_tensor(self.cks_smem_layout.outer, swizzle=self.cks_smem_layout.inner)
        sDV2N = storage.smem_dv2n.get_tensor(self.dv2n_smem_layout.outer, swizzle=self.dv2n_smem_layout.inner)
        sDV2N_epi = storage.smem_dv2n.get_tensor(self.ops_epi_layout.outer, swizzle=self.ops_epi_layout.inner)
        sDV2S = storage.smem_dv2s.get_tensor(self.dv2s_smem_layout)

        # ---- pipelines ----
        simt_threads = 32 * len(self.simt_warp_id)
        mmain_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_mma_load_bytes,
            barrier_storage=storage.mmain_full.data_ptr(),
            defer_sync=True,
        )
        simtin_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, simt_threads
            ),
            tx_count=self.num_simt_load_bytes,
            barrier_storage=storage.simtin_full.data_ptr(),
            defer_sync=True,
        )

        def make_simt_to_mma_pipe(ptr, stages):
            return pipeline.PipelineAsyncUmma.create(
                num_stages=stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, simt_threads
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                barrier_storage=ptr,
                defer_sync=True,
            )

        def make_mma_to_simt_pipe(ptr):
            return pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, simt_threads
                ),
                barrier_storage=ptr,
                defer_sync=True,
            )

        dh_pipe = make_simt_to_mma_pipe(storage.dh_full.data_ptr(), self.dh_stages)
        dv2n_pipe = make_simt_to_mma_pipe(storage.dv2n_full.data_ptr(), 1)
        dvf_pipe = make_mma_to_simt_pipe(storage.dvf_full.data_ptr())
        qdf_pipe = make_mma_to_simt_pipe(storage.qdf_full.data_ptr())
        wdf_pipe = make_mma_to_simt_pipe(storage.wdf_full.data_ptr())

        pipeline_init_arrive(cluster_shape_mn=(1, 1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1, 1))

        # ---- tmem ----
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=0, num_threads=self.threads_per_cta
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.simt_warp_id[0],
        )
        tmem.allocate(self.num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

        def acc_tensor(mma, tile, offset):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return cute.make_tensor(tmem_ptr_base + offset, fake.layout)

        tDV = acc_tensor(mma_dv, self.tile_dv, self.tmem_dv_offset)
        tQD = acc_tensor(mma_qd, self.tile_upd, self.tmem_qd_offset)
        tWD = acc_tensor(mma_wd, self.tile_upd, self.tmem_wd_offset)

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gKG = cute.local_tile(mKG, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)
        gDOT = cute.local_tile(mDOT, (BV, BT), (v_idx, None, hv_idx, b_idx))  # (BV,BT,NT)
        gQGT = cute.local_tile(mQGT, (K, BT), (0, None, hv_idx, b_idx))  # (K,BT,NT)
        gWT = cute.local_tile(mWT, (K, BT), (0, None, hv_idx, b_idx))
        gDVI = cute.local_tile(mDVI, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gGd = mGd[(None, None, hv_idx, b_idx)]  # (K, NT)
        gDV2 = cute.local_tile(mDV2, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gCK = cute.local_tile(mCK, (BV, K), (v_idx, 0, None, hv_idx, b_idx))  # (BV,K,NT)

        # ==========================================================================
        # TMA warp — chunks walked newest-first
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_dv = mma_dv.get_slice(0)
            thr_mma_qd = mma_qd.get_slice(0)
            thr_mma_wd = mma_wd.get_slice(0)

            tKG_mma = thr_mma_dv.partition_A(gKG)
            tDOT_mma = thr_mma_qd.partition_A(gDOT)
            tQGT_mma = thr_mma_qd.partition_B(gQGT)
            tWT_mma = thr_mma_wd.partition_B(gWT)

            cta1 = cute.make_layout(1)
            tKGs, tKGg = cpasync.tma_partition(
                tma_kg, 0, cta1, cute.group_modes(sKG, 0, 3), cute.group_modes(tKG_mma, 0, 3)
            )
            tDOTs, tDOTg = cpasync.tma_partition(
                tma_dot, 0, cta1, cute.group_modes(sDOT, 0, 3), cute.group_modes(tDOT_mma, 0, 3)
            )
            tQGTs, tQGTg = cpasync.tma_partition(
                tma_qgt, 0, cta1, cute.group_modes(sQGT, 0, 3), cute.group_modes(tQGT_mma, 0, 3)
            )
            tWTs, tWTg = cpasync.tma_partition(
                tma_wt, 0, cta1, cute.group_modes(sWT, 0, 3), cute.group_modes(tWT_mma, 0, 3)
            )
            tDVIs, tDVIg = cpasync.tma_partition(
                tma_dvi, 0, cta1, cute.group_modes(sDVI, 0, 2), cute.group_modes(gDVI, 0, 2)
            )
            tGds, tGdg = cpasync.tma_partition(
                tma_gd, 0, cta1, cute.group_modes(sGd, 0, 1), cute.group_modes(gGd, 0, 1)
            )

            mmain_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            simtin_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )

            for rc in cutlass.range(NT, unroll=1):
                c = NT - 1 - rc
                mmain_pipe.producer_acquire(mmain_producer)
                bar = mmain_pipe.producer_get_barrier(mmain_producer)
                cute.copy(tma_kg, tKGg[None, c], tKGs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_dot, tDOTg[None, c], tDOTs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_qgt, tQGTg[None, c], tQGTs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_wt, tWTg[None, c], tWTs[None, mmain_producer.index], tma_bar_ptr=bar)
                mmain_producer.advance()

                simtin_pipe.producer_acquire(simtin_producer)
                sbar = simtin_pipe.producer_get_barrier(simtin_producer)
                cute.copy(tma_dvi, tDVIg[None, c], tDVIs[None, simtin_producer.index], tma_bar_ptr=sbar)
                cute.copy(tma_gd, tGdg[None, c], tGds[None, simtin_producer.index], tma_bar_ptr=sbar)
                simtin_producer.advance()

            mmain_pipe.producer_tail(mmain_producer)
            simtin_pipe.producer_tail(simtin_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrKG = mma_dv.make_fragment_A(sKG)
            tCrDH = mma_dv.make_fragment_B(sDH)
            tCrDOT = mma_qd.make_fragment_A(sDOT)
            tCrQGT = mma_qd.make_fragment_B(sQGT)
            tCrDV2N = mma_wd.make_fragment_A(sDV2N)
            tCrWT = mma_wd.make_fragment_B(sWT)

            mmain_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            simtin_mma_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            dh_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.dh_stages
            )
            dv2n_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dvf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            qdf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            wdf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for rc in cutlass.range(NT, unroll=1):
                # DV = kg dh^T — heads the critical path (SIMT needs it for dv2)
                mmain_pipe.consumer_wait(mmain_consumer)
                dh_pipe.consumer_wait(dh_consumer)
                dvf_pipe.producer_acquire(dvf_producer)
                for kk in cutlass.range(cute.size(tCrDH, mode=[2]), unroll_full=True):
                    mma_dv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dv, tDV[None, None, None, 0],
                        tCrKG[None, None, kk, mmain_consumer.index],
                        tCrDH[None, None, kk, dh_consumer.index],
                        tDV[None, None, None, 0],
                    )
                dvf_pipe.producer_commit(dvf_producer)
                dvf_producer.advance()
                dh_pipe.consumer_release(dh_consumer)
                dh_consumer.advance()

                # QD = do^T qg — pure TMA operands, lands while SIMT builds dv2
                qdf_pipe.producer_acquire(qdf_producer)
                for kk in cutlass.range(cute.size(tCrDOT, mode=[2]), unroll_full=True):
                    mma_qd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_qd, tQD[None, None, None, 0],
                        tCrDOT[None, None, kk, mmain_consumer.index],
                        tCrQGT[None, None, kk, mmain_consumer.index],
                        tQD[None, None, None, 0],
                    )
                qdf_pipe.producer_commit(qdf_producer)
                qdf_producer.advance()

                # WD = dv2^T w
                dv2n_pipe.consumer_wait(dv2n_consumer)
                wdf_pipe.producer_acquire(wdf_producer)
                for kk in cutlass.range(cute.size(tCrDV2N, mode=[2]), unroll_full=True):
                    mma_wd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_wd, tWD[None, None, None, 0],
                        tCrDV2N[None, None, kk, 0],
                        tCrWT[None, None, kk, mmain_consumer.index],
                        tWD[None, None, None, 0],
                    )
                wdf_pipe.producer_commit(wdf_producer)
                wdf_producer.advance()
                dv2n_pipe.consumer_release(dv2n_consumer)
                dv2n_consumer.advance()
                mmain_pipe.consumer_release(mmain_consumer)
                mmain_consumer.advance()
                # umma half of simtin's empty arrive (data untouched by this warp)
                simtin_pipe.consumer_wait(simtin_mma_consumer)
                simtin_pipe.consumer_release(
                    simtin_mma_consumer, pipeline.PipelineOp.TCGen05Mma
                )
                simtin_mma_consumer.advance()

            dvf_pipe.producer_tail(dvf_producer)
            qdf_pipe.producer_tail(qdf_producer)
            wdf_pipe.producer_tail(wdf_producer)

        # ==========================================================================
        # SIMT warps 4..7: dv2, the dh update, checkpoint/dv2 stores, dh0.
        # ==========================================================================
        elif (
            warp_idx == self.simt_warp_id[0]
            or warp_idx == self.simt_warp_id[1]
            or warp_idx == self.simt_warp_id[2]
            or warp_idx == self.simt_warp_id[3]
        ):
            t2r_64_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), f32
            )
            f32_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), f32)
            io_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), io)

            # --- DV -> dv2 ---
            tDV_2d = tDV[((None, None), 0, 0, None)]
            tiled_t2r_dv = tcgen05.make_tmem_copy(t2r_64_atom, tDV_2d[None, None, 0])
            thr_t2r_dv = tiled_t2r_dv.get_slice(local_tidx)
            tTR_tDV = thr_t2r_dv.partition_S(tDV_2d)
            tTR_rDV = cute.make_rmem_tensor(
                thr_t2r_dv.partition_D(cute.make_identity_tensor((BT, BV))).shape, f32
            )
            tDVsDVI = thr_t2r_dv.partition_D(sDVI)
            tDVrDVI = cute.make_rmem_tensor(
                cute.slice_(tDVsDVI.shape, (None, None, None, 0)), io
            )
            r2s_x16t_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_dv2n = cute.make_tiled_copy_D(r2s_x16t_atom, tiled_t2r_dv)
            thr_r2s_dv2n = tiled_r2s_dv2n.get_slice(local_tidx)
            tRS_sDV2N = thr_r2s_dv2n.partition_D(sDV2N_epi)
            tRS_rDV2N = cute.make_rmem_tensor(
                cute.slice_(tRS_sDV2N.shape, (None, None, None, 0)), io
            )
            r2s_x16_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_dv2s = cute.make_tiled_copy_D(r2s_x16_atom, tiled_t2r_dv)
            thr_r2s_dv2s = tiled_r2s_dv2s.get_slice(local_tidx)
            tRS_sDV2S = thr_r2s_dv2s.partition_D(sDV2S)
            tRS_rDV2S = cute.make_rmem_tensor(
                cute.slice_(tRS_sDV2S.shape, (None, None, None, 0)), io
            )

            # --- QD/WD -> dh ---
            tQD_2d = tQD[((None, None), 0, 0, None)]
            tiled_t2r_upd = tcgen05.make_tmem_copy(t2r_64_atom, tQD_2d[None, None, 0])
            thr_t2r_upd = tiled_t2r_upd.get_slice(local_tidx)
            tTR_tQD = thr_t2r_upd.partition_S(tQD_2d)
            coordUPD = thr_t2r_upd.partition_D(cute.make_identity_tensor((BV, K)))
            tTR_rQD = cute.make_rmem_tensor(coordUPD.shape, f32)
            tWD_2d = tWD[((None, None), 0, 0, None)]
            tiled_t2r_wd = tcgen05.make_tmem_copy(t2r_64_atom, tWD_2d[None, None, 0])
            thr_t2r_wd = tiled_t2r_wd.get_slice(local_tidx)
            tTR_tWD = thr_t2r_wd.partition_S(tWD_2d)
            tTR_rWD = cute.make_rmem_tensor(coordUPD.shape, f32)
            tDHreg = cute.make_rmem_tensor(coordUPD.shape, f32)
            sGd_bcast = cute.make_tensor(
                sGd.iterator,
                cute.make_layout((BV, K, self.input_stages), stride=(0, 1, K)),
            )
            tUPDsGd = thr_t2r_upd.partition_D(sGd_bcast)
            tUPDrGd = cute.make_rmem_tensor(
                cute.slice_(tUPDsGd.shape, (None, None, None, 0)), f32
            )
            r2s_dh_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_dh = cute.make_tiled_copy_D(r2s_dh_atom, tiled_t2r_upd)
            thr_r2s_dh = tiled_r2s_dh.get_slice(local_tidx)
            tRS_sDH = thr_r2s_dh.partition_D(sDH_epi)
            tRS_rDH = cute.make_rmem_tensor(
                cute.slice_(tRS_sDH.shape, (None, None, None, 0)), io
            )
            r2s_ck_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ck = cute.make_tiled_copy_D(r2s_ck_atom, tiled_t2r_upd)
            thr_r2s_ck = tiled_r2s_ck.get_slice(local_tidx)
            tRS_sCKS = thr_r2s_ck.partition_D(sCKS)
            tRS_rCKS = cute.make_rmem_tensor(
                cute.slice_(tRS_sCKS.shape, (None, None, None, 0)), io
            )

            # TMA store plumbing (dv2 per chunk; dh checkpoint per chunk)
            bSG_sDV2S, bSG_gDV2 = cpasync.tma_partition(
                tma_dv2, 0, cute.make_layout(1),
                cute.group_modes(sDV2S, 0, 2), cute.group_modes(gDV2, 0, 2),
            )
            bSG_sCK, bSG_gCK = cpasync.tma_partition(
                tma_ck, 0, cute.make_layout(1),
                cute.group_modes(sCKS, 0, 2), cute.group_modes(gCK, 0, 2),
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, simt_threads
                ),
            )

            simtin_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            dvf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            qdf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            wdf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dh_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dh_stages
            )
            dv2n_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # ---- dh := dht; checkpoint NT-1 is exactly that value ----
            for i in cutlass.range(cute.size(tDHreg), unroll_full=True):
                vv, kk = coordUPD[i]
                tDHreg[i] = mDHT[(kk, v_idx * BV + vv, hv_idx, b_idx)]
            dh_pipe.producer_acquire(dh_producer)
            for i in cutlass.range(cute.size(tDHreg), unroll_full=True, vectorize=True):
                vio = tDHreg[i].to(io)
                tRS_rDH[i] = vio
                tRS_rCKS[i] = vio
            cute.copy(tiled_r2s_dh, tRS_rDH, tRS_sDH[None, None, None, dh_producer.index])
            cute.copy(tiled_r2s_ck, tRS_rCKS, tRS_sCKS[None, None, None, 0])
            cute.arch.fence_proxy("async.shared", space="cta")
            self.simt_sync_barrier.arrive_and_wait()
            # commit dh BEFORE the checkpoint store (record-006 deferral): the store's
            # completion wait lands on the next dv2-store acquire, off the dh chain.
            dh_pipe.producer_commit(dh_producer)
            dh_producer.advance()
            if warp_idx == self.simt_warp_id[0]:
                cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, NT - 1])
                tma_store_pipeline.producer_commit()

            for rc in cutlass.range(NT, unroll=1):
                c = NT - 1 - rc
                simtin_pipe.consumer_wait(simtin_consumer)
                scrd = (None, None, None, simtin_consumer.index)
                cute.copy(io_cp_atom, tDVsDVI[scrd], tDVrDVI)

                # dv2 = DV + dv_in
                dvf_pipe.consumer_wait(dvf_consumer)
                cute.copy(tiled_t2r_dv, tTR_tDV[None, None, None, 0], tTR_rDV)
                cute.arch.fence_view_async_tmem_load()
                dvf_pipe.consumer_release(dvf_consumer)
                dvf_consumer.advance()
                dv2n_pipe.producer_acquire(dv2n_producer)
                for i in cutlass.range(
                    cute.size(tTR_rDV), unroll_full=True, vectorize=True
                ):
                    v2 = (tTR_rDV[i] + tDVrDVI[i].to(f32)).to(io)
                    tRS_rDV2N[i] = v2
                    tRS_rDV2S[i] = v2
                # the MMA operand first: it unblocks WD
                cute.copy(tiled_r2s_dv2n, tRS_rDV2N, tRS_sDV2N[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                dv2n_pipe.producer_commit(dv2n_producer)
                dv2n_producer.advance()
                # acquire FIRST: waits on last chunk's stores, not the fresh ones
                if warp_idx == self.simt_warp_id[0]:
                    tma_store_pipeline.producer_acquire()
                self.simt_sync_barrier.arrive_and_wait()
                cute.copy(tiled_r2s_dv2s, tRS_rDV2S, tRS_sDV2S[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                if warp_idx == self.simt_warp_id[0]:
                    cute.copy(tma_dv2, bSG_sDV2S[None, 0], bSG_gDV2[None, c])
                    tma_store_pipeline.producer_commit()

                # dh update: dh = exp2(gd) * dh + scale*QD - WD (per-dim decay)
                cute.copy(f32_cp_atom, tUPDsGd[scrd], tUPDrGd)
                qdf_pipe.consumer_wait(qdf_consumer)
                cute.copy(tiled_t2r_upd, tTR_tQD[None, None, None, 0], tTR_rQD)
                cute.arch.fence_view_async_tmem_load()
                qdf_pipe.consumer_release(qdf_consumer)
                qdf_consumer.advance()
                wdf_pipe.consumer_wait(wdf_consumer)
                cute.copy(tiled_t2r_wd, tTR_tWD[None, None, None, 0], tTR_rWD)
                cute.arch.fence_view_async_tmem_load()
                wdf_pipe.consumer_release(wdf_consumer)
                wdf_consumer.advance()
                for i in cutlass.range(
                    cute.size(tDHreg), unroll_full=True, vectorize=True
                ):
                    dec = cute.math.exp2(tUPDrGd[i], fastmath=True)
                    tDHreg[i] = dec * tDHreg[i] + (scale * tTR_rQD[i] - tTR_rWD[i])
                if c > 0:
                    dh_pipe.producer_acquire(dh_producer)
                    for i in cutlass.range(
                        cute.size(tDHreg), unroll_full=True, vectorize=True
                    ):
                        vio = tDHreg[i].to(io)
                        tRS_rDH[i] = vio
                        tRS_rCKS[i] = vio
                    cute.copy(
                        tiled_r2s_dh, tRS_rDH, tRS_sDH[None, None, None, dh_producer.index]
                    )
                    cute.copy(tiled_r2s_ck, tRS_rCKS, tRS_sCKS[None, None, None, 0])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.simt_sync_barrier.arrive_and_wait()
                    dh_pipe.producer_commit(dh_producer)
                    dh_producer.advance()
                    if warp_idx == self.simt_warp_id[0]:
                        cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, c - 1])
                        tma_store_pipeline.producer_commit()

                simtin_pipe.consumer_release(
                    simtin_consumer, pipeline.PipelineOp.AsyncThread
                )
                simtin_consumer.advance()

            # ---- dh0 (fp32): once-per-kernel plain global scatter ----
            for i in cutlass.range(cute.size(tDHreg), unroll_full=True):
                vv, kk = coordUPD[i]
                mDH0[(kk, v_idx * BV + vv, hv_idx, b_idx)] = tDHreg[i]

            tma_store_pipeline.producer_tail()

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)
        return


# --------------------------------------------------------------------------------------------
# host wrapper
# --------------------------------------------------------------------------------------------

_COMPILE_CACHE: dict = {}


# Layout-keyed call cache with ctypes pointer retargeting; outputs are NEVER cached —
# allocated per call and retargeted (see kernel_fwd.py's note above _CALL_CACHE).
_CALL_CACHE: dict = {}


def _call_key(qg, kg, w, g2, dht, do, dv, scale):
    def sig(t):
        return (t.shape, t.stride(), t.dtype)

    return (sig(qg), sig(kg), sig(w), sig(g2), sig(dht), sig(do), sig(dv), scale,
            torch.cuda.current_stream().cuda_stream)


def kda_cute_dhu_call(
    qg: torch.Tensor,  # [B,T,HV,K] bf16/fp16 — q * exp2(g2), from recompute
    kg: torch.Tensor,  # [B,T,HV,K] — k * exp2(G - g2)
    w: torch.Tensor,  # [B,T,HV,K]
    g2: torch.Tensor,  # [B,T,HV,K] fp32 (only last rows used)
    dht: torch.Tensor,  # [B,HV,K,V] fp32
    do: torch.Tensor,  # [B,T,HV,V]
    dv: torch.Tensor,  # [B,T,HV,V] — stage 3's dv
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dh checkpoints [B,NT,HV,K,V] bf16, dh0 [B,HV,K,V] fp32, dv2 [B,T,HV,V])."""
    key = _call_key(qg, kg, w, g2, dht, do, dv, scale)
    ent = _CALL_CACHE.get(key)
    outs = None
    if ent is None:
        B, T, HV, K = kg.shape
        V = do.shape[3]
        NT = T // 64
        assert T % 64 == 0
        assert V % 64 == 0

        dh = torch.empty(B, NT, HV, K, V, device=kg.device, dtype=kg.dtype)
        dh0 = torch.empty(B, HV, K, V, device=kg.device, dtype=torch.float32)
        dv2 = torch.empty(B, T, HV, V, device=kg.device, dtype=dv.dtype)
        gdc = torch.empty(B, HV, NT, K, device=g2.device, dtype=g2.dtype)

        io_dtype = cutlass.BFloat16 if kg.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V)

        ckg = _cute_view(kg, (1, 3, 2, 0), (0, 2, 3))
        cdot = _cute_view(do, (3, 1, 2, 0), (1, 2, 3))
        cqgt = _cute_view(qg, (3, 1, 2, 0), (1, 2, 3))
        cwt = _cute_view(w, (3, 1, 2, 0), (1, 2, 3))
        cdvi = _cute_view(dv, (1, 3, 2, 0), (0, 2, 3))
        cgd = _cute_view(gdc, (3, 2, 1, 0), (1, 2, 3))
        cdht = _cute_view(dht, (2, 3, 1, 0), (2, 3))
        cdv2 = _cute_view(dv2, (1, 3, 2, 0), (0, 2, 3))
        cdhck = _cute_view(dh, (4, 3, 1, 2, 0), (2, 3, 4))
        cdh0 = _cute_view(dh0, (2, 3, 1, 0), (2, 3))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = KdaB2aDhuKernel(io_dtype, K, V_TILE=64)
            compiled = cute.compile(
                kernel_obj, ckg, cdot, cqgt, cwt, cdvi, cgd, cdht, cdv2, cdhck, cdh0,
                cutlass.Float32(scale), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        # See kernel_fwd._release_keepalives: dh alone is 2 GiB at prod8192 and this entry
        # would pin it, plus kg/do/qg/w/dv/dht, for the life of the process.
        _release_keepalives(ckg, cdot, cqgt, cwt, cdvi, cgd, cdht, cdv2, cdhck, cdh0)
        args = (ckg, cdot, cqgt, cwt, cdvi, cgd, cdht, cdv2, cdhck, cdh0,
                cutlass.Float32(scale), stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (dh, dh0, dv2)
        out_specs = tuple((tuple(t.shape), t.dtype) for t in outs)
        ent = (compiled, args, out_specs, gdc)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, gdc = ent
    ckg, cdot, cqgt, cwt, cdvi, _, cdht, cdv2, cdhck, cdh0, _, _ = args
    if outs is None:
        outs = tuple(
            torch.empty(shape, device=kg.device, dtype=dtype)
            for shape, dtype in out_specs
        )
    dh, dh0, dv2 = outs
    _retarget(cdhck, dh)
    _retarget(cdh0, dh0)
    _retarget(cdv2, dv2)
    _retarget(ckg, kg)
    _retarget(cdot, do)
    _retarget(cqgt, qg)
    _retarget(cwt, w)
    _retarget(cdvi, dv)
    _retarget(cdht, dht)
    gdc.copy_(g2[:, 63::64].transpose(1, 2))
    compiled(*args)
    return dh, dh0, dv2


# Serial scans need a full GPU (gdn 003 record 005's lesson); below the floor fall back
# to fla. KDA002_B2A=cutedsl forces past it for dbg-sized shapes; =fla pins fla.
_MIN_CTAS = 256


def kda_dhu_b2a(qg, kg, w, g2, h0, dht, do, dv, scale, chunk_size):
    """Stage-4 dispatcher, argument-for-argument fla's chunk_gated_delta_rule_bwd_dhu."""
    B, T, HV, K = kg.shape
    V = do.shape[-1]
    supported = (
        chunk_size == 64
        and T % 64 == 0
        and K in (64, 128)
        and V % 64 == 0
        and kg.dtype in (torch.bfloat16, torch.float16)
        # dht is None when the caller did not ask for a final state; this kernel requires
        # it, so that call takes fla's dhu and gives up ~0.7ms.
        and dht is not None
        and B * HV * (V // 64) >= _MIN_CTAS
    )
    if not supported:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

        return chunk_gated_delta_rule_bwd_dhu(
            q=qg, k=kg, w=w, gk=g2, h0=h0, dht=dht, do=do, dv=dv,
            scale=scale, chunk_size=chunk_size,
        )
    dh, dh0, dv2 = kda_cute_dhu_call(qg, kg, w, g2, dht, do, dv, float(scale))
    # fla's contract: dh0 exists only when h0 was provided (it is h0's gradient; the
    # kernel computes it unconditionally since its value never depends on h0).
    return dh, (dh0 if h0 is not None else None), dv2
