"""CuTe port of fla's `chunk_gated_delta_rule_bwd_dhu` (USE_GK path) — the reverse state
scan of the kda backward (stage 4).

Ported from gdn 003's kernel_dhu.py (the USE_G scalar-gate scan) by the ALGORITHM.md diff
table. Per (b, hv) and per chunk c = NT-1 .. 0, with dh [K, V] the running state gradient
(fp32, starts at dht):

    dh_ck[c] = dh                                 (bf16 checkpoint -> HBM, pre-update)
    dv2      = kg @ dh_bf16 + dv                  (completes stage-3's intra dv; NO gate
                                                   factor — kg carries exp2(G - g2))
    dh[d,:]  = exp2(gd[d]) * dh[d,:] + scale * (qg^T @ do)[d,:] - (w^T @ dv2)[d,:]
    after chunk 0: dh0 = dh                       (fp32)

vs gdn: q and k arrive PRE-SCALED (qg = q*exp2(g2), kg = k*exp2(G-g2), per VALUE head, from
fla's recompute_w_u), so gdn's dog = do*exp2(g2_i)*scale SIMT build loses its gate — what
remains is dos = do*scale — and the dv2 rescale is deleted. The scalar chunk decay widens
to a K-VECTOR gd = g2[last row of chunk], applied per fragment element via a broadcast
view (001's state-decay pattern). Same CTA/warp/pipeline structure as gdn's kernel; the
two tiled MMAs keep their shapes with operands renamed:

    DV  = kg @ dh^T          tile (BT, BV, K), A = kg k-major TMA, B = sDH k-major (SIMT)
    UPD = dos^T @ qg - dv2^T @ w   tile (BV, K, BT), A = SIMT trans-stmatrix,
                                                     B = qg^T/w^T mn-major TMA

The subtraction is an accumulating MMA with a negated A operand: SIMT writes -dv2 into
the operand buffer and +dv2 into the store staging (bf16 negation is exact).

Precision contract vs fla: dh stays fp32 in registers across chunks; it is cast to bf16
exactly where fla casts (the kg@dh operand and the checkpoint store). The one deliberate
divergence: fla computes dot(qg_bf16, do_bf16) in fp32 and multiplies scale after; ours
rounds do*scale to bf16 for the MMA operand. Same order of error as gdn 003's dog note
(measured there at dh 3.8e-5 / dh0 2.5e-3, well inside stage tolerances).

Logical gmem mode order (M/N, K_contract, rest) per operand:
    kg:        (T, K, HV, B)      A of DV (per-value-head — no h_idx anywhere here)
    qg^T, w^T: (K, T, HV, B)      mn-major B of UPD
    do, dv:    (T, V, HV, B)      SIMT-only, linear smem
    gd:        (K, NT, HV, B)     fp32 per-chunk decay vectors, K contiguous
    dht, dh0:  (K, V, HV, B)      fp32 scalar gather/scatter, once per CTA
    dv2:       (T, V, HV, B)      TMA store
    dh_ck:     (V, K, NT, HV, B)  TMA store of the [BV, K] checkpoint slice straight from
                                  the swizzled operand bytes (ROW_MAJOR epi view of sDH)
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


class KdaBwdDhuKernel:
    """Reverse dh scan. One CTA per (b, hv, v_tile); serial over chunks, newest first.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT recurrence (dos, dv2, the dh update,
    dh0). Warps 2,3 idle through the role branch and only join alloc/dealloc barriers.
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
        self.tile_upd = (self.BV, self.K, self.BT)  # dos^T @ qg  /  (-dv2)^T @ w

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
            io,
            tcgen05.OperandMajorMode("k"),
            tcgen05.OperandMajorMode("k"),
            acc,
            grp,
            self.tile_dv[:2],
            tcgen05.OperandSource.SMEM,
        )
        mma_upd = sm100_utils.make_trivial_tiled_mma(
            io,
            tcgen05.OperandMajorMode("k"),
            tcgen05.OperandMajorMode("mn"),
            acc,
            grp,
            self.tile_upd[:2],
            tcgen05.OperandSource.SMEM,
        )
        return mma_dv, mma_upd

    def _setup_attributes(self):
        mma_dv, mma_upd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.kg_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dv, self.tile_dv, self.io_dtype, self.input_stages
        )
        self.qgt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_upd, self.tile_upd, self.io_dtype, self.input_stages
        )
        self.wt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_upd, self.tile_upd, self.io_dtype, self.input_stages
        )
        # do / dv are SIMT-only: plain row-major (BT, BV), BV contiguous
        self.do_smem_layout = cute.make_layout((BT, BV, self.input_stages), stride=(BV, 1, BT * BV))
        self.dvi_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        # gd: the per-chunk decay K-vector (g2's last row), fp32. gdn staged [BT] g2
        # rows here; the kda scan needs no per-row gate (qg/kg carry it).
        self.gd_smem_layout = cute.make_layout((K, self.input_stages))

        # dh^T as B operand of DV ([BV, K] k-major, staged); written by SIMT through the
        # ROW_MAJOR epi view of the same bytes (the forward's sH pattern).
        self.dh_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dv, self.tile_dv, self.io_dtype, self.dh_stages
        )
        self.dh_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BV, K), self.dh_stages
        )
        # Checkpoint store staging. TMA cannot transpose: the operand bytes above are
        # K-contiguous but the gmem checkpoint slice is V-contiguous, so the store needs
        # its own buffer in gmem order — COL_MAJOR (BV, K), filled by a transpose
        # stmatrix from the same fragments.
        self.cks_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BV, K), 1
        )
        # dos / -dv2 as A operands of UPD ([BV, BT] k-major); written via transpose
        # stmatrix through COL_MAJOR (BT, BV) epi views (the forward's v~' pattern).
        self.dos_smem_layout = sm100_utils.make_smem_layout_a(
            mma_upd, self.tile_upd, self.io_dtype, 1
        )
        self.dv2n_smem_layout = sm100_utils.make_smem_layout_a(
            mma_upd, self.tile_upd, self.io_dtype, 1
        )
        self.ops_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BV), 1
        )
        # dv2 store staging: (BT, BV) BV-contiguous to match gmem
        self.dv2s_smem_layout = cute.make_layout((BT, BV, 1), stride=(BV, 1, BT * BV))

        self.num_kqw_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.kg_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.qgt_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.wt_smem_layout, (None, None, None, 0))
            )
        )
        self.num_dod_load_bytes = (
            cute.size_in_bytes(self.io_dtype, cute.slice_(self.do_smem_layout, (None, None, 0)))
            + cute.size_in_bytes(self.io_dtype, cute.slice_(self.dvi_smem_layout, (None, None, 0)))
            + cute.size_in_bytes(cutlass.Float32, cute.slice_(self.gd_smem_layout, (None, 0)))
        )

        (
            self.tmem_dv_offset,
            self.tmem_upd_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_dv, mma_upd)

    def _plan_tmem(self, mma_dv, mma_upd):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        dv = acc_cols(mma_dv, self.tile_dv)
        upd = acc_cols(mma_upd, self.tile_upd)
        off_dv = 0
        off_upd = off_dv + dv
        total_ = off_upd + upd
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_dv, off_upd, total

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        kg: cute.Tensor,  # (T, K, HV, B)
        qgt: cute.Tensor,  # (K, T, HV, B) — same storage as qg
        wt: cute.Tensor,  # (K, T, HV, B) — same storage as w
        do: cute.Tensor,  # (T, V, HV, B)
        dvi: cute.Tensor,  # (T, V, HV, B) — stage 3's intra dv
        gd: cute.Tensor,  # (K, NT, HV, B) fp32 — per-chunk decay vector (g2 last row)
        dht: cute.Tensor,  # (K, V, HV, B) fp32
        dv2: cute.Tensor,  # (T, V, HV, B) out
        dhck: cute.Tensor,  # (V, K, NT, HV, B) out, io dtype
        dh0: cute.Tensor,  # (K, V, HV, B) fp32 out
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_dv, mma_upd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_kg, tma_tensor_kg = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            kg,
            cute.slice_(self.kg_smem_layout, (None, None, None, 0)),
            self.tile_dv,
            mma_dv,
            cluster_vmnk,
        )
        tma_qgt, tma_tensor_qgt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            qgt,
            cute.slice_(self.qgt_smem_layout, (None, None, None, 0)),
            self.tile_upd,
            mma_upd,
            cluster_vmnk,
        )
        tma_wt, tma_tensor_wt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            wt,
            cute.slice_(self.wt_smem_layout, (None, None, None, 0)),
            self.tile_upd,
            mma_upd,
            cluster_vmnk,
        )
        tma_do, tma_tensor_do = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            do,
            cute.slice_(self.do_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        tma_dvi, tma_tensor_dvi = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            dvi,
            cute.slice_(self.dvi_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        gd_cta_v_layout = cute.slice_(cute.make_identity_layout(gd.shape), (None, 0, 0, 0))
        tma_gd, tma_tensor_gd = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gd,
            cute.slice_(self.gd_smem_layout, (None, 0)),
            gd_cta_v_layout,
        )
        tma_dv2, tma_tensor_dv2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dv2,
            cute.slice_(self.dv2s_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        tma_ck, tma_tensor_ck = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            dhck,
            cute.slice_(self.cks_smem_layout, (None, None, 0)),
            (BV, K),
        )

        B = cute.size(kg, mode=[3])
        HV = cute.size(wt, mode=[2])
        NV = cute.size(do, mode=[1]) // BV
        grid = (B * HV * NV, 1, 1)

        swz_align, lin_align = 1024, 128

        # Each `*_full` range backs both halves of a pipeline's mbarrier array —
        # 2 * num_stages Int64s. Under-sizing aliases the next pipeline and only
        # deadlocks once the pipe wraps.
        @cute.struct
        class SharedStorage:
            kqw_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            dod_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            dh_full: cute.struct.MemRange[cutlass.Int64, self.dh_stages * 2]  # type: ignore
            dosa_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dv2n_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dvf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            updf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_kg: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.kg_smem_layout)], swz_align  # type: ignore
            ]
            smem_qgt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.qgt_smem_layout)], swz_align  # type: ignore
            ]
            smem_wt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.wt_smem_layout)], swz_align  # type: ignore
            ]
            smem_do: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.do_smem_layout)], lin_align  # type: ignore
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
            smem_dos: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dos_smem_layout)], swz_align  # type: ignore
            ]
            smem_dv2n: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dv2n_smem_layout)], swz_align  # type: ignore
            ]
            smem_dv2s: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dv2s_smem_layout)], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}")

        self.kda_cute_bwd_dhu(
            tma_kg,
            tma_tensor_kg,
            tma_qgt,
            tma_tensor_qgt,
            tma_wt,
            tma_tensor_wt,
            tma_do,
            tma_tensor_do,
            tma_dvi,
            tma_tensor_dvi,
            tma_gd,
            tma_tensor_gd,
            tma_dv2,
            tma_tensor_dv2,
            tma_ck,
            tma_tensor_ck,
            dht,
            dh0,
            scale,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def kda_cute_bwd_dhu(
        self,
        tma_kg: cute.CopyAtom,
        mKG: cute.Tensor,
        tma_qgt: cute.CopyAtom,
        mQGT: cute.Tensor,
        tma_wt: cute.CopyAtom,
        mWT: cute.Tensor,
        tma_do: cute.CopyAtom,
        mDO: cute.Tensor,
        tma_dvi: cute.CopyAtom,
        mDVI: cute.Tensor,
        tma_gd: cute.CopyAtom,
        mGd: cute.Tensor,
        tma_dv2: cute.CopyAtom,
        mDV2: cute.Tensor,
        tma_ck: cute.CopyAtom,
        mCK: cute.Tensor,
        mDHT: cute.Tensor,
        mDH0: cute.Tensor,
        scale: cutlass.Float32,
    ):
        BT, K, BV = self.BT, self.K, self.BV
        io = self.io_dtype
        f32 = self.acc_dtype
        # Layouts/TiledMma from the host trace cannot cross the region boundary — rebuild.
        self._setup_attributes()
        mma_dv, mma_upd = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_kg, tma_qgt, tma_wt, tma_do, tma_dvi, tma_gd, tma_dv2, tma_ck]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mWT, mode=[2])
        NV = cute.size(mDO, mode=[1]) // BV
        T = cute.size(mKG, mode=[0])
        NT = T // BT
        v_idx = bidx % NV
        hv_idx = (bidx // NV) % HV
        b_idx = bidx // (NV * HV)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sKG = storage.smem_kg.get_tensor(
            self.kg_smem_layout.outer, swizzle=self.kg_smem_layout.inner
        )
        sQGT = storage.smem_qgt.get_tensor(
            self.qgt_smem_layout.outer, swizzle=self.qgt_smem_layout.inner
        )
        sWT = storage.smem_wt.get_tensor(
            self.wt_smem_layout.outer, swizzle=self.wt_smem_layout.inner
        )
        sDO = storage.smem_do.get_tensor(self.do_smem_layout)
        sDVI = storage.smem_dvi.get_tensor(self.dvi_smem_layout)
        sGd = storage.smem_gd.get_tensor(self.gd_smem_layout)
        sDH = storage.smem_dh.get_tensor(
            self.dh_smem_layout.outer, swizzle=self.dh_smem_layout.inner
        )
        sDH_epi = storage.smem_dh.get_tensor(
            self.dh_epi_layout.outer, swizzle=self.dh_epi_layout.inner
        )
        sCKS = storage.smem_cks.get_tensor(
            self.cks_smem_layout.outer, swizzle=self.cks_smem_layout.inner
        )
        sDOS = storage.smem_dos.get_tensor(
            self.dos_smem_layout.outer, swizzle=self.dos_smem_layout.inner
        )
        sDOS_epi = storage.smem_dos.get_tensor(
            self.ops_epi_layout.outer, swizzle=self.ops_epi_layout.inner
        )
        sDV2N = storage.smem_dv2n.get_tensor(
            self.dv2n_smem_layout.outer, swizzle=self.dv2n_smem_layout.inner
        )
        sDV2N_epi = storage.smem_dv2n.get_tensor(
            self.ops_epi_layout.outer, swizzle=self.ops_epi_layout.inner
        )
        sDV2S = storage.smem_dv2s.get_tensor(self.dv2s_smem_layout)

        # ---- pipelines ----
        simt_threads = 32 * len(self.simt_warp_id)
        kqw_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_kqw_load_bytes,
            barrier_storage=storage.kqw_full.data_ptr(),
            defer_sync=True,
        )
        # do/dv/gd are SIMT-only data, but the pipe keeps the proven multi-consumer
        # plumbing: the MMA warp waits and releases each stage without reading it.
        dod_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(pipeline.Agent.Thread, simt_threads),
            tx_count=self.num_dod_load_bytes,
            barrier_storage=storage.dod_full.data_ptr(),
            defer_sync=True,
        )

        def make_simt_to_mma_pipe(ptr, stages):
            return pipeline.PipelineAsyncUmma.create(
                num_stages=stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, simt_threads),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                barrier_storage=ptr,
                defer_sync=True,
            )

        def make_mma_to_simt_pipe(ptr):
            return pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, simt_threads),
                barrier_storage=ptr,
                defer_sync=True,
            )

        dh_pipe = make_simt_to_mma_pipe(storage.dh_full.data_ptr(), self.dh_stages)
        dosa_pipe = make_simt_to_mma_pipe(storage.dosa_full.data_ptr(), 1)
        dv2n_pipe = make_simt_to_mma_pipe(storage.dv2n_full.data_ptr(), 1)
        dvf_pipe = make_mma_to_simt_pipe(storage.dvf_full.data_ptr())
        updf_pipe = make_mma_to_simt_pipe(storage.updf_full.data_ptr())

        pipeline_init_arrive(cluster_shape_mn=(1, 1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1, 1))

        # ---- tmem ----
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
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
        tUPD = acc_tensor(mma_upd, self.tile_upd, self.tmem_upd_offset)

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gKG = cute.local_tile(mKG, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)
        gQGT = cute.local_tile(mQGT, (K, BT), (0, None, hv_idx, b_idx))  # (K,BT,NT)
        gWT = cute.local_tile(mWT, (K, BT), (0, None, hv_idx, b_idx))
        gDO = cute.local_tile(mDO, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gDVI = cute.local_tile(mDVI, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gGd = mGd[(None, None, hv_idx, b_idx)]  # (K, NT)
        gDV2 = cute.local_tile(mDV2, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gCK = cute.local_tile(mCK, (BV, K), (v_idx, 0, None, hv_idx, b_idx))  # (BV,K,NT)

        # ==========================================================================
        # TMA warp — chunks walked newest-first
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_dv = mma_dv.get_slice(0)
            thr_mma_upd = mma_upd.get_slice(0)

            tKG_mma = thr_mma_dv.partition_A(gKG)
            tQGT_mma = thr_mma_upd.partition_B(gQGT)
            tWT_mma = thr_mma_upd.partition_B(gWT)

            cta1 = cute.make_layout(1)
            tKGs, tKGg = cpasync.tma_partition(
                tma_kg, 0, cta1, cute.group_modes(sKG, 0, 3), cute.group_modes(tKG_mma, 0, 3)
            )
            tQGTs, tQGTg = cpasync.tma_partition(
                tma_qgt, 0, cta1, cute.group_modes(sQGT, 0, 3), cute.group_modes(tQGT_mma, 0, 3)
            )
            tWTs, tWTg = cpasync.tma_partition(
                tma_wt, 0, cta1, cute.group_modes(sWT, 0, 3), cute.group_modes(tWT_mma, 0, 3)
            )
            tDOs, tDOg = cpasync.tma_partition(
                tma_do, 0, cta1, cute.group_modes(sDO, 0, 2), cute.group_modes(gDO, 0, 2)
            )
            tDVIs, tDVIg = cpasync.tma_partition(
                tma_dvi, 0, cta1, cute.group_modes(sDVI, 0, 2), cute.group_modes(gDVI, 0, 2)
            )
            tGds, tGdg = cpasync.tma_partition(
                tma_gd, 0, cta1, cute.group_modes(sGd, 0, 1), cute.group_modes(gGd, 0, 1)
            )

            kqw_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            dod_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )

            for rc in cutlass.range(NT, unroll=1):
                c = NT - 1 - rc
                kqw_pipe.producer_acquire(kqw_producer)
                bar = kqw_pipe.producer_get_barrier(kqw_producer)
                cute.copy(tma_kg, tKGg[None, c], tKGs[None, kqw_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_qgt, tQGTg[None, c], tQGTs[None, kqw_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_wt, tWTg[None, c], tWTs[None, kqw_producer.index], tma_bar_ptr=bar)
                kqw_producer.advance()

                dod_pipe.producer_acquire(dod_producer)
                dbar = dod_pipe.producer_get_barrier(dod_producer)
                cute.copy(tma_do, tDOg[None, c], tDOs[None, dod_producer.index], tma_bar_ptr=dbar)
                cute.copy(
                    tma_dvi, tDVIg[None, c], tDVIs[None, dod_producer.index], tma_bar_ptr=dbar
                )
                cute.copy(tma_gd, tGdg[None, c], tGds[None, dod_producer.index], tma_bar_ptr=dbar)
                dod_producer.advance()

            kqw_pipe.producer_tail(kqw_producer)
            dod_pipe.producer_tail(dod_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrKG = mma_dv.make_fragment_A(sKG)
            tCrDH = mma_dv.make_fragment_B(sDH)
            tCrDOS = mma_upd.make_fragment_A(sDOS)
            tCrDV2N = mma_upd.make_fragment_A(sDV2N)
            tCrQGT = mma_upd.make_fragment_B(sQGT)
            tCrWT = mma_upd.make_fragment_B(sWT)

            kqw_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            dod_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            dh_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.dh_stages
            )
            dosa_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dv2n_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dvf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            updf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for rc in cutlass.range(NT, unroll=1):
                # this warp never reads do/dv/gd — wait+release keeps the pipe's
                # umma-side accounting satisfied
                dod_pipe.consumer_wait(dod_consumer)
                dod_pipe.consumer_release(dod_consumer, pipeline.PipelineOp.TCGen05Mma)
                dod_consumer.advance()

                # DV = kg @ dh^T — heads the critical path
                dh_pipe.consumer_wait(dh_consumer)
                kqw_pipe.consumer_wait(kqw_consumer)
                dvf_pipe.producer_acquire(dvf_producer)
                for kk in cutlass.range(cute.size(tCrDH, mode=[2]), unroll_full=True):
                    mma_dv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dv,
                        tDV[None, None, None, 0],
                        tCrKG[None, None, kk, kqw_consumer.index],
                        tCrDH[None, None, kk, dh_consumer.index],
                        tDV[None, None, None, 0],
                    )
                dvf_pipe.producer_commit(dvf_producer)
                dvf_producer.advance()
                dh_pipe.consumer_release(dh_consumer)
                dh_consumer.advance()

                # UPD = dos^T @ qg — dos only needs do, so it lands while SIMT
                # is still building dv2
                dosa_pipe.consumer_wait(dosa_consumer)
                updf_pipe.producer_acquire(updf_producer)
                for kk in cutlass.range(cute.size(tCrDOS, mode=[2]), unroll_full=True):
                    mma_upd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_upd,
                        tUPD[None, None, None, 0],
                        tCrDOS[None, None, kk, 0],
                        tCrQGT[None, None, kk, kqw_consumer.index],
                        tUPD[None, None, None, 0],
                    )
                dosa_pipe.consumer_release(dosa_consumer)
                dosa_consumer.advance()

                # UPD += (-dv2)^T @ w
                dv2n_pipe.consumer_wait(dv2n_consumer)
                for kk in cutlass.range(cute.size(tCrDV2N, mode=[2]), unroll_full=True):
                    mma_upd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                    cute.gemm(
                        mma_upd,
                        tUPD[None, None, None, 0],
                        tCrDV2N[None, None, kk, 0],
                        tCrWT[None, None, kk, kqw_consumer.index],
                        tUPD[None, None, None, 0],
                    )
                updf_pipe.producer_commit(updf_producer)
                updf_producer.advance()
                dv2n_pipe.consumer_release(dv2n_consumer)
                dv2n_consumer.advance()
                kqw_pipe.consumer_release(kqw_consumer)
                kqw_consumer.advance()

            dvf_pipe.producer_tail(dvf_producer)
            updf_pipe.producer_tail(updf_producer)

        # ==========================================================================
        # SIMT warps 4..7: dos, dv2, the dh update, checkpoint/dv2 stores, dh0.
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

            # --- DV -> dos / dv2 ---
            tDV_2d = tDV[((None, None), 0, 0, None)]
            tiled_t2r_dv = tcgen05.make_tmem_copy(t2r_64_atom, tDV_2d[None, None, 0])
            thr_t2r_dv = tiled_t2r_dv.get_slice(local_tidx)
            tTR_tDV = thr_t2r_dv.partition_S(tDV_2d)
            # rmem operands of a tmem copy must be sized from the D partition of a
            # non-tmem tensor — partition_S folds the lane mode and oversizes.
            tTR_rDV = cute.make_rmem_tensor(
                thr_t2r_dv.partition_D(cute.make_identity_tensor((BT, BV))).shape, f32
            )
            tDVsDO = thr_t2r_dv.partition_D(sDO)
            tDVrDO = cute.make_rmem_tensor(cute.slice_(tDVsDO.shape, (None, None, None, 0)), io)
            tDVsDVI = thr_t2r_dv.partition_D(sDVI)
            tDVrDVI = cute.make_rmem_tensor(cute.slice_(tDVsDVI.shape, (None, None, None, 0)), io)
            r2s_x16t_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ops = cute.make_tiled_copy_D(r2s_x16t_atom, tiled_t2r_dv)
            thr_r2s_ops = tiled_r2s_ops.get_slice(local_tidx)
            tRS_sDOS = thr_r2s_ops.partition_D(sDOS_epi)
            tRS_sDV2N = thr_r2s_ops.partition_D(sDV2N_epi)
            tRS_rDOS = cute.make_rmem_tensor(cute.slice_(tRS_sDOS.shape, (None, None, None, 0)), io)
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

            # --- UPD -> dh ---
            tUPD_2d = tUPD[((None, None), 0, 0, None)]
            tiled_t2r_upd = tcgen05.make_tmem_copy(t2r_64_atom, tUPD_2d[None, None, 0])
            thr_t2r_upd = tiled_t2r_upd.get_slice(local_tidx)
            tTR_tUPD = thr_t2r_upd.partition_S(tUPD_2d)
            coordUPD = thr_t2r_upd.partition_D(cute.make_identity_tensor((BV, K)))
            tTR_rUPD = cute.make_rmem_tensor(coordUPD.shape, f32)
            tDHreg = cute.make_rmem_tensor(tTR_rUPD.shape, f32)
            # The per-chunk decay is a K-VECTOR (gdn: one scalar). Broadcast it across
            # the BV mode of the UPD fragment so each element picks up exp2's argument
            # for its own k — 001's state-decay pattern, identical fragment geometry.
            sGd_bcast = cute.make_tensor(
                sGd.iterator,
                cute.make_layout((BV, K, self.input_stages), stride=(0, 1, K)),
            )
            tUPDsGd = thr_t2r_upd.partition_D(sGd_bcast)
            tUPDrGd = cute.make_rmem_tensor(cute.slice_(tUPDsGd.shape, (None, None, None, 0)), f32)
            r2s_dh_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_dh = cute.make_tiled_copy_D(r2s_dh_atom, tiled_t2r_upd)
            thr_r2s_dh = tiled_r2s_dh.get_slice(local_tidx)
            tRS_sDH = thr_r2s_dh.partition_D(sDH_epi)
            tRS_rDH = cute.make_rmem_tensor(cute.slice_(tRS_sDH.shape, (None, None, None, 0)), io)
            # checkpoint staging: same values, transpose stmatrix into the BV-contiguous
            # buffer the TMA store can express. make_tiled_copy_D keeps the base tiling's
            # value->thread assignment, so element i matches tRS_rDH's element i.
            r2s_ck_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ck = cute.make_tiled_copy_D(r2s_ck_atom, tiled_t2r_upd)
            thr_r2s_ck = tiled_r2s_ck.get_slice(local_tidx)
            tRS_sCKS = thr_r2s_ck.partition_D(sCKS)
            tRS_rCKS = cute.make_rmem_tensor(cute.slice_(tRS_sCKS.shape, (None, None, None, 0)), io)

            # TMA store plumbing (dv2 per chunk; dh checkpoint per chunk)
            bSG_sDV2S, bSG_gDV2 = cpasync.tma_partition(
                tma_dv2,
                0,
                cute.make_layout(1),
                cute.group_modes(sDV2S, 0, 2),
                cute.group_modes(gDV2, 0, 2),
            )
            bSG_sCK, bSG_gCK = cpasync.tma_partition(
                tma_ck,
                0,
                cute.make_layout(1),
                cute.group_modes(sCKS, 0, 2),
                cute.group_modes(gCK, 0, 2),
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, simt_threads),
            )

            dod_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            dvf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            updf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dh_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dh_stages
            )
            dosa_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
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
            # commit dh BEFORE the checkpoint store: the store's completion is off the
            # recurrence critical path — its wait is deferred to the dv2-store acquire
            # (num_stages=1 waits on ALL outstanding groups), which also protects the
            # sCKS overwrite a chunk later. (Pattern proven in kernel_fwdh.py: took the
            # fwd_h stage from 0.86x to 1.03x.)
            dh_pipe.producer_commit(dh_producer)
            dh_producer.advance()
            if warp_idx == self.simt_warp_id[0]:
                cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, NT - 1])
                tma_store_pipeline.producer_commit()

            for rc in cutlass.range(NT, unroll=1):
                c = NT - 1 - rc
                dod_pipe.consumer_wait(dod_consumer)
                dcrd = (None, None, None, dod_consumer.index)

                # dos = do * scale — no DV dependency, ships first (the per-dim gate
                # that gdn folded in here lives in the pre-scaled qg operand now)
                cute.copy(io_cp_atom, tDVsDO[dcrd], tDVrDO)
                dosa_pipe.producer_acquire(dosa_producer)
                for i in cutlass.range(cute.size(tTR_rDV), unroll_full=True, vectorize=True):
                    tRS_rDOS[i] = (tDVrDO[i].to(f32) * scale).to(io)
                cute.copy(tiled_r2s_ops, tRS_rDOS, tRS_sDOS[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                dosa_pipe.producer_commit(dosa_producer)
                dosa_producer.advance()

                # dv2 = DV + dv_local (kg carried gdn's exp2(G - g2_i) factor)
                cute.copy(io_cp_atom, tDVsDVI[dcrd], tDVrDVI)
                dvf_pipe.consumer_wait(dvf_consumer)
                cute.copy(tiled_t2r_dv, tTR_tDV[None, None, None, 0], tTR_rDV)
                cute.arch.fence_view_async_tmem_load()
                dvf_pipe.consumer_release(dvf_consumer)
                dvf_consumer.advance()
                dv2n_pipe.producer_acquire(dv2n_producer)
                for i in cutlass.range(cute.size(tTR_rDV), unroll_full=True, vectorize=True):
                    dv2f = tTR_rDV[i] + tDVrDVI[i].to(f32)
                    tRS_rDV2N[i] = (-dv2f).to(io)
                    tRS_rDV2S[i] = dv2f.to(io)
                # the negated operand first: it unblocks the MMA's second UPD term
                cute.copy(tiled_r2s_ops, tRS_rDV2N, tRS_sDV2N[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                dv2n_pipe.producer_commit(dv2n_producer)
                dv2n_producer.advance()
                # acquire FIRST: it waits on last chunk's stores (long complete), not
                # the one about to be issued — sDV2S/sCKS reuse stays safe via the
                # barrier, and no fresh-store completion wait lands on the dh chain.
                if warp_idx == self.simt_warp_id[0]:
                    tma_store_pipeline.producer_acquire()
                self.simt_sync_barrier.arrive_and_wait()
                cute.copy(tiled_r2s_dv2s, tRS_rDV2S, tRS_sDV2S[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                if warp_idx == self.simt_warp_id[0]:
                    cute.copy(tma_dv2, bSG_sDV2S[None, 0], bSG_gDV2[None, c])
                    tma_store_pipeline.producer_commit()

                # dh update: per-dim decay — each fragment element's k picks its own factor.
                cute.copy(f32_cp_atom, tUPDsGd[dcrd], tUPDrGd)
                updf_pipe.consumer_wait(updf_consumer)
                cute.copy(tiled_t2r_upd, tTR_tUPD[None, None, None, 0], tTR_rUPD)
                cute.arch.fence_view_async_tmem_load()
                updf_pipe.consumer_release(updf_consumer)
                updf_consumer.advance()
                for i in cutlass.range(cute.size(tDHreg), unroll_full=True, vectorize=True):
                    dec = cute.math.exp2(tUPDrGd[i], fastmath=True)
                    tDHreg[i] = dec * tDHreg[i] + tTR_rUPD[i]
                if rc + 1 < NT:
                    dh_pipe.producer_acquire(dh_producer)
                    for i in cutlass.range(cute.size(tDHreg), unroll_full=True, vectorize=True):
                        vio = tDHreg[i].to(io)
                        tRS_rDH[i] = vio
                        tRS_rCKS[i] = vio
                    cute.copy(tiled_r2s_dh, tRS_rDH, tRS_sDH[None, None, None, dh_producer.index])
                    cute.copy(tiled_r2s_ck, tRS_rCKS, tRS_sCKS[None, None, None, 0])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.simt_sync_barrier.arrive_and_wait()
                    # dh first — the checkpoint store is off the recurrence critical
                    # path (completion deferred to the next dv2-store acquire).
                    dh_pipe.producer_commit(dh_producer)
                    dh_producer.advance()
                    if warp_idx == self.simt_warp_id[0]:
                        cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, c - 1])
                        tma_store_pipeline.producer_commit()

                dod_pipe.consumer_release(dod_consumer, pipeline.PipelineOp.AsyncThread)
                dod_consumer.advance()

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


def _cute_view(t: torch.Tensor, perm: tuple[int, ...], dyn_modes: tuple[int, ...]):
    # Mode order must come from the unpermuted tensor's strides — see kernel_fwd.py.
    t = t.detach()
    base_order = sorted(range(t.dim()), key=lambda i: -t.stride(i))
    new_of_old = {old: new for new, old in enumerate(perm)}
    stride_order = tuple(new_of_old[d] for d in base_order)
    tt = t.permute(*perm)
    ct = from_dlpack(tt, assumed_align=16)
    for m in dyn_modes:
        ct = ct.mark_compact_shape_dynamic(mode=m, stride_order=stride_order)
    return ct


# Layout-keyed call cache with ctypes pointer retargeting — same contract as the other
# stage kernels: outputs (dh, dh0, dv2) are allocated PER CALL and retargeted (never
# cache-owned, see gdn's dbg_alias writeup); internal scratch (gdc) stays cached and is
# refilled before each launch.
_CALL_CACHE: dict = {}


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def _out_specs(*tensors: torch.Tensor) -> tuple:
    """(shape, dtype) per output — what the cache keeps instead of the tensors themselves."""
    return tuple((tuple(t.shape), t.dtype) for t in tensors)


def _alloc_outs(specs: tuple, device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


def _call_key(qg, kg, w, do, dv, g2, dht, scale):
    def sig(t):
        return (t.shape, t.stride(), t.dtype) if t is not None else None

    return (
        sig(qg),
        sig(kg),
        sig(w),
        sig(do),
        sig(dv),
        sig(g2),
        sig(dht),
        scale,
        torch.cuda.current_stream().cuda_stream,
    )


def kda_cute_dhu_call(
    qg: torch.Tensor,  # [B,T,HV,K] bf16/fp16 — q * exp2(g2), from recompute_w_u
    kg: torch.Tensor,  # [B,T,HV,K] — k * exp2(G - g2)
    w: torch.Tensor,  # [B,T,HV,K]
    do: torch.Tensor,  # [B,T,HV,V]
    dv: torch.Tensor,  # [B,T,HV,V] — stage 3's intra dv
    g2: torch.Tensor,  # [B,T,HV,K] fp32, chunk-local cumsum / ln2 (only last rows used)
    dht: torch.Tensor | None,  # [B,HV,K,V] fp32 or None
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = _call_key(qg, kg, w, do, dv, g2, dht, scale)
    ent = _CALL_CACHE.get(key)
    outs = None  # set on the miss path, where the outputs were allocated to build the views
    if ent is None:
        B, T, HV, K = qg.shape
        V = do.shape[3]
        NT = T // 64
        assert T % 64 == 0
        assert V % 64 == 0

        dh = torch.empty(B, NT, HV, K, V, device=qg.device, dtype=qg.dtype)
        dh0 = torch.empty(B, HV, K, V, device=qg.device, dtype=torch.float32)
        dv2 = torch.empty(B, T, HV, V, device=qg.device, dtype=do.dtype)
        gdc = torch.empty(B, HV, NT, K, device=g2.device, dtype=g2.dtype)
        dht_t = (
            dht
            if dht is not None
            else torch.zeros(B, HV, K, V, device=qg.device, dtype=torch.float32)
        )

        io_dtype = cutlass.BFloat16 if qg.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V)

        ckg = _cute_view(kg, (1, 3, 2, 0), (0, 2, 3))
        cqgt = _cute_view(qg, (3, 1, 2, 0), (1, 2, 3))
        cwt = _cute_view(w, (3, 1, 2, 0), (1, 2, 3))
        cdo = _cute_view(do, (1, 3, 2, 0), (0, 2, 3))
        cdvi = _cute_view(dv, (1, 3, 2, 0), (0, 2, 3))
        cgd = _cute_view(gdc, (3, 2, 1, 0), (1, 2, 3))
        cdht = _cute_view(dht_t, (2, 3, 1, 0), (2, 3))
        cdv2 = _cute_view(dv2, (1, 3, 2, 0), (0, 2, 3))
        cck = _cute_view(dh, (4, 3, 1, 2, 0), (2, 3, 4))
        cdh0 = _cute_view(dh0, (2, 3, 1, 0), (2, 3))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = KdaBwdDhuKernel(io_dtype, K, V_TILE=64)
            compiled = cute.compile(
                kernel_obj,
                ckg,
                cqgt,
                cwt,
                cdo,
                cdvi,
                cgd,
                cdht,
                cdv2,
                cck,
                cdh0,
                cutlass.Float32(scale),
                stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (
            ckg,
            cqgt,
            cwt,
            cdo,
            cdvi,
            cgd,
            cdht,
            cdv2,
            cck,
            cdh0,
            cutlass.Float32(scale),
            stream,
        )
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (dh, dh0, dv2)
        ent = (compiled, args, _out_specs(dh, dh0, dv2), gdc, dht_t if dht is None else None)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, gdc, zeros_dht = ent
    ckg, cqgt, cwt, cdo, cdvi, _, cdht, cdv2, cck, cdh0, _, _ = args
    if outs is None:
        outs = _alloc_outs(out_specs, qg.device)
    dh, dh0, dv2 = outs
    _retarget(cck, dh)
    _retarget(cdh0, dh0)
    _retarget(cdv2, dv2)
    _retarget(ckg, kg)
    _retarget(cqgt, qg)
    _retarget(cwt, w)
    _retarget(cdo, do)
    _retarget(cdvi, dv)
    if dht is not None:
        _retarget(cdht, dht)
    # refill the decay staging in place: g2's last row per chunk, [B,T,HV,K] -> [B,HV,NT,K]
    gdc.copy_(g2[:, 63::64].transpose(1, 2))
    compiled(*args)
    return dh, dh0, dv2


# Minimum grid size (CTAs) for the cute path; dbg_dhu sets this to 0 to force the cute
# kernel on small correctness shapes.
_MIN_CTAS = 256


def chunk_gated_delta_rule_bwd_dhu_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    """Drop-in for fla's chunk_gated_delta_rule_bwd_dhu on the kda (gk, per-dim) path,
    falling back on shapes the cute kernel does not cover. `q`/`k` must be the pre-scaled
    qg/kg (per VALUE head) — exactly what fla's chunk_kda_bwd passes."""
    B, T, HQ, K = q.shape
    HV, V = do.shape[2], do.shape[3]
    supported = (
        gk is not None
        and g is None
        and HQ == HV  # qg/kg ride at value heads; H-headed operands mean a non-kda caller
        and not state_v_first
        and cu_seqlens is None
        and chunk_indices is None
        and chunk_size == 64
        and T % 64 == 0
        and K in (64, 128)
        and V % 64 == 0
        and q.dtype in (torch.bfloat16, torch.float16)
        # the serial scan only wins with a grid that fills the GPU: at gva's 64 CTAs the
        # gdn kernel lost 1.4ms to fla, at prod8192's 1024 it won 1.4x. _MIN_CTAS is a
        # module global so dbg_dhu can zero it — otherwise every dbg-sized case silently
        # falls back and the "comparison" is fla vs fla.
        and B * HV * (V // 64) >= _MIN_CTAS
    )
    if not supported:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

        return chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )

    if scale is None:
        scale = K**-0.5
    dh, dh0, dv2 = kda_cute_dhu_call(q, k, w, do, dv, gk, dht, float(scale))
    return dh, (dh0 if h0 is not None else None), dv2
