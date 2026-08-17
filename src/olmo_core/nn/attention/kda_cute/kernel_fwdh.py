"""CuTe port of fla's `chunk_gated_delta_rule_fwd_h` (USE_GK path) — the forward state
scan, re-run by the kda backward for the h checkpoints and v_new (stage 2).

Ported from gdn 003's kernel_fwdh.py (the USE_G scalar-gate scan) by the ALGORITHM.md diff
table. Per (b, hv) and per chunk c = 0 .. NT-1, with h [K, V] the running state (fp32):

    h_ck[c] = h                                   (bf16 checkpoint -> HBM, pre-update)
    v'      = u - w @ h_bf16                      (v_new -> HBM, no gating — kg carries it)
    h[d,:]  = exp2(gd[d]) * h[d,:] + (kg^T @ v')[d,:]

vs gdn: the k operand arrives PRE-SCALED (kg = k*exp2(G-g2), per VALUE head, from fla's
recompute_w_u), so gdn's SIMT v~ = v'*exp2(G-g2_i) build is deleted — the UPD operand is v'
itself, the same value as the v_new store. The scalar chunk decay exp2(G) widens to a
K-VECTOR gd = g2[last row of chunk] (fp32, sliced host-side, 001's gdc pattern), applied
per fragment element via a broadcast view — the state-decay geometry 001's fused forward
already landed.

Same one-CTA-per-(b, hv, v_tile) serial scan, state TRANSPOSED — h^T [BV, K] — in the SIMT
warpgroup's registers. The two tiled MMAs are unchanged in shape:

    WH  = w @ h^T       tile (BT, BV, K), A = w k-major TMA, B = sH k-major (SIMT)
    UPD = v'^T @ kg     tile (BV, K, BT), A = SIMT trans-stmatrix, B = kg^T mn-major TMA

Precision contract vs fla (USE_GK): identical cast points — h stays fp32 in registers, cast
to bf16 at the w@h operand and the checkpoint store; v' is rounded to bf16 once and that one
value is both v_new and the UPD operand (fla: `b_v.to(k.dtype)` then dot, store from the
same fp32). No deliberate divergence.

Logical gmem mode order (M/N, K_contract, rest) per operand:
    w:      (T, K, HV, B)         A of WH
    kg^T:   (K, T, HV, B)         mn-major B of UPD (per-value-head — no h_idx here)
    u:      (T, V, HV, B)         SIMT-only, linear smem
    gd:     (K, NT, HV, B)        fp32 per-chunk decay vectors, K contiguous
    h0:     (K, V, HV, B)         fp32 scalar gather, once per CTA
    v_new:  (T, V, HV, B)         TMA store
    h_ck:   (V, K, NT, HV, B)     TMA store; the operand bytes are K-contiguous and TMA
                                  cannot transpose, so the store goes through its own
                                  COL_MAJOR (BV, K) staging buffer (see kernel_dhu.py)
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


class KdaFwdHKernel:
    """Forward h scan. One CTA per (b, hv, v_tile); serial over chunks, oldest first.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT recurrence (v', the h update).
    Warps 2,3 idle through the role branch and only join alloc/dealloc barriers.
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
        self.tile_wh = (self.BT, self.BV, self.K)  # w @ h^T
        self.tile_upd = (self.BV, self.K, self.BT)  # v'^T @ kg

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0
        self.mma_warp_id = 1
        self.simt_warp_id = (4, 5, 6, 7)
        self.threads_per_cta = 32 * 8

        self.input_stages = 2
        self.h_stages = 2

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
        mma_wh = sm100_utils.make_trivial_tiled_mma(
            io,
            tcgen05.OperandMajorMode("k"),
            tcgen05.OperandMajorMode("k"),
            acc,
            grp,
            self.tile_wh[:2],
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
        return mma_wh, mma_upd

    def _setup_attributes(self):
        mma_wh, mma_upd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.w_smem_layout = sm100_utils.make_smem_layout_a(
            mma_wh, self.tile_wh, self.io_dtype, self.input_stages
        )
        self.kgt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_upd, self.tile_upd, self.io_dtype, self.input_stages
        )
        # u is SIMT-only: plain row-major (BT, BV), BV contiguous
        self.u_smem_layout = cute.make_layout((BT, BV, self.input_stages), stride=(BV, 1, BT * BV))
        # gd: the per-chunk decay K-vector (g2's last row), fp32. gdn staged [BT] g2
        # rows here; the kda scan needs no per-row gate (kg carries it), only the decay.
        self.gd_smem_layout = cute.make_layout((K, self.input_stages))

        # h^T as B operand of WH ([BV, K] k-major, staged); written by SIMT through the
        # ROW_MAJOR epi view of the same bytes.
        self.h_smem_layout = sm100_utils.make_smem_layout_b(
            mma_wh, self.tile_wh, self.io_dtype, self.h_stages
        )
        self.h_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BV, K), self.h_stages
        )
        # Checkpoint store staging: TMA cannot transpose, so the V-contiguous gmem slice
        # gets its own COL_MAJOR (BV, K) buffer filled by a transpose stmatrix.
        self.cks_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BV, K), 1
        )
        # v' as A operand of UPD ([BV, BT] k-major); written via transpose stmatrix
        # through a COL_MAJOR (BT, BV) epi view.
        self.vta_smem_layout = sm100_utils.make_smem_layout_a(
            mma_upd, self.tile_upd, self.io_dtype, 1
        )
        self.ops_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BV), 1
        )
        # v_new store staging: (BT, BV) BV-contiguous to match gmem
        self.vns_smem_layout = cute.make_layout((BT, BV, 1), stride=(BV, 1, BT * BV))

        self.num_kw_load_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.w_smem_layout, (None, None, None, 0))
        ) + cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.kgt_smem_layout, (None, None, None, 0))
        )
        self.num_ud_load_bytes = cute.size_in_bytes(
            self.io_dtype, cute.slice_(self.u_smem_layout, (None, None, 0))
        ) + cute.size_in_bytes(cutlass.Float32, cute.slice_(self.gd_smem_layout, (None, 0)))

        (
            self.tmem_wh_offset,
            self.tmem_upd_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_wh, mma_upd)

    def _plan_tmem(self, mma_wh, mma_upd):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        wh = acc_cols(mma_wh, self.tile_wh)
        upd = acc_cols(mma_upd, self.tile_upd)
        off_wh = 0
        off_upd = off_wh + wh
        total_ = off_upd + upd
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_wh, off_upd, total

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        w: cute.Tensor,  # (T, K, HV, B)
        kgt: cute.Tensor,  # (K, T, HV, B) — same storage as kg
        u: cute.Tensor,  # (T, V, HV, B)
        gd: cute.Tensor,  # (K, NT, HV, B) fp32 — per-chunk decay vector (g2 last row)
        h0: cute.Tensor,  # (K, V, HV, B) fp32
        vnew: cute.Tensor,  # (T, V, HV, B) out
        hck: cute.Tensor,  # (V, K, NT, HV, B) out, io dtype
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_wh, mma_upd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            w,
            cute.slice_(self.w_smem_layout, (None, None, None, 0)),
            self.tile_wh,
            mma_wh,
            cluster_vmnk,
        )
        tma_kgt, tma_tensor_kgt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(),
            kgt,
            cute.slice_(self.kgt_smem_layout, (None, None, None, 0)),
            self.tile_upd,
            mma_upd,
            cluster_vmnk,
        )
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            u,
            cute.slice_(self.u_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        gd_cta_v_layout = cute.slice_(cute.make_identity_layout(gd.shape), (None, 0, 0, 0))
        tma_gd, tma_tensor_gd = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gd,
            cute.slice_(self.gd_smem_layout, (None, 0)),
            gd_cta_v_layout,
        )
        tma_vn, tma_tensor_vn = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            vnew,
            cute.slice_(self.vns_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        tma_ck, tma_tensor_ck = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            hck,
            cute.slice_(self.cks_smem_layout, (None, None, 0)),
            (BV, K),
        )

        B = cute.size(w, mode=[3])
        HV = cute.size(w, mode=[2])
        NV = cute.size(u, mode=[1]) // BV
        grid = (B * HV * NV, 1, 1)

        swz_align, lin_align = 1024, 128

        # Each `*_full` range backs both halves of a pipeline's mbarrier array —
        # 2 * num_stages Int64s. Under-sizing aliases the next pipeline and only
        # deadlocks once the pipe wraps.
        @cute.struct
        class SharedStorage:
            kw_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            ud_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            h_full: cute.struct.MemRange[cutlass.Int64, self.h_stages * 2]  # type: ignore
            vta_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            whf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            updf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_w: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.w_smem_layout)], swz_align  # type: ignore
            ]
            smem_kgt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.kgt_smem_layout)], swz_align  # type: ignore
            ]
            smem_u: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.u_smem_layout)], lin_align  # type: ignore
            ]
            smem_gd: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.gd_smem_layout)], lin_align  # type: ignore
            ]
            smem_h: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.h_smem_layout)], swz_align  # type: ignore
            ]
            smem_cks: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.cks_smem_layout)], swz_align  # type: ignore
            ]
            smem_vta: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.vta_smem_layout)], swz_align  # type: ignore
            ]
            smem_vns: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.vns_smem_layout)], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}")

        self.kda_cute_fwdh(
            tma_w,
            tma_tensor_w,
            tma_kgt,
            tma_tensor_kgt,
            tma_u,
            tma_tensor_u,
            tma_gd,
            tma_tensor_gd,
            tma_vn,
            tma_tensor_vn,
            tma_ck,
            tma_tensor_ck,
            h0,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def kda_cute_fwdh(
        self,
        tma_w: cute.CopyAtom,
        mW: cute.Tensor,
        tma_kgt: cute.CopyAtom,
        mKGT: cute.Tensor,
        tma_u: cute.CopyAtom,
        mU: cute.Tensor,
        tma_gd: cute.CopyAtom,
        mGd: cute.Tensor,
        tma_vn: cute.CopyAtom,
        mVN: cute.Tensor,
        tma_ck: cute.CopyAtom,
        mCK: cute.Tensor,
        mH0: cute.Tensor,
    ):
        BT, K, BV = self.BT, self.K, self.BV
        io = self.io_dtype
        f32 = self.acc_dtype
        # Layouts/TiledMma from the host trace cannot cross the region boundary — rebuild.
        self._setup_attributes()
        mma_wh, mma_upd = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_w, tma_kgt, tma_u, tma_gd, tma_vn, tma_ck]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mW, mode=[2])
        NV = cute.size(mU, mode=[1]) // BV
        T = cute.size(mW, mode=[0])
        NT = T // BT
        v_idx = bidx % NV
        hv_idx = (bidx // NV) % HV
        b_idx = bidx // (NV * HV)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sW = storage.smem_w.get_tensor(self.w_smem_layout.outer, swizzle=self.w_smem_layout.inner)
        sKGT = storage.smem_kgt.get_tensor(
            self.kgt_smem_layout.outer, swizzle=self.kgt_smem_layout.inner
        )
        sU = storage.smem_u.get_tensor(self.u_smem_layout)
        sGd = storage.smem_gd.get_tensor(self.gd_smem_layout)
        sH = storage.smem_h.get_tensor(self.h_smem_layout.outer, swizzle=self.h_smem_layout.inner)
        sH_epi = storage.smem_h.get_tensor(self.h_epi_layout.outer, swizzle=self.h_epi_layout.inner)
        sCKS = storage.smem_cks.get_tensor(
            self.cks_smem_layout.outer, swizzle=self.cks_smem_layout.inner
        )
        sVTA = storage.smem_vta.get_tensor(
            self.vta_smem_layout.outer, swizzle=self.vta_smem_layout.inner
        )
        sVTA_epi = storage.smem_vta.get_tensor(
            self.ops_epi_layout.outer, swizzle=self.ops_epi_layout.inner
        )
        sVNS = storage.smem_vns.get_tensor(self.vns_smem_layout)

        # ---- pipelines ----
        simt_threads = 32 * len(self.simt_warp_id)
        kw_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_kw_load_bytes,
            barrier_storage=storage.kw_full.data_ptr(),
            defer_sync=True,
        )
        # u/gd are SIMT-only data, but the pipe keeps the proven multi-consumer
        # plumbing: the MMA warp waits and releases each stage without reading it.
        ud_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(pipeline.Agent.Thread, simt_threads),
            tx_count=self.num_ud_load_bytes,
            barrier_storage=storage.ud_full.data_ptr(),
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

        h_pipe = make_simt_to_mma_pipe(storage.h_full.data_ptr(), self.h_stages)
        vta_pipe = make_simt_to_mma_pipe(storage.vta_full.data_ptr(), 1)
        whf_pipe = make_mma_to_simt_pipe(storage.whf_full.data_ptr())
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

        tWH = acc_tensor(mma_wh, self.tile_wh, self.tmem_wh_offset)
        tUPD = acc_tensor(mma_upd, self.tile_upd, self.tmem_upd_offset)

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gW = cute.local_tile(mW, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)
        gKGT = cute.local_tile(mKGT, (K, BT), (0, None, hv_idx, b_idx))  # (K,BT,NT)
        gU = cute.local_tile(mU, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gGd = mGd[(None, None, hv_idx, b_idx)]  # (K, NT)
        gVN = cute.local_tile(mVN, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gCK = cute.local_tile(mCK, (BV, K), (v_idx, 0, None, hv_idx, b_idx))  # (BV,K,NT)

        # ==========================================================================
        # TMA warp — chunks walked oldest-first
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_wh = mma_wh.get_slice(0)
            thr_mma_upd = mma_upd.get_slice(0)

            tW_mma = thr_mma_wh.partition_A(gW)
            tKGT_mma = thr_mma_upd.partition_B(gKGT)

            cta1 = cute.make_layout(1)
            tWs, tWg = cpasync.tma_partition(
                tma_w, 0, cta1, cute.group_modes(sW, 0, 3), cute.group_modes(tW_mma, 0, 3)
            )
            tKGTs, tKGTg = cpasync.tma_partition(
                tma_kgt, 0, cta1, cute.group_modes(sKGT, 0, 3), cute.group_modes(tKGT_mma, 0, 3)
            )
            tUs, tUg = cpasync.tma_partition(
                tma_u, 0, cta1, cute.group_modes(sU, 0, 2), cute.group_modes(gU, 0, 2)
            )
            tGds, tGdg = cpasync.tma_partition(
                tma_gd, 0, cta1, cute.group_modes(sGd, 0, 1), cute.group_modes(gGd, 0, 1)
            )

            kw_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            ud_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )

            for c in cutlass.range(NT, unroll=1):
                kw_pipe.producer_acquire(kw_producer)
                bar = kw_pipe.producer_get_barrier(kw_producer)
                cute.copy(tma_w, tWg[None, c], tWs[None, kw_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_kgt, tKGTg[None, c], tKGTs[None, kw_producer.index], tma_bar_ptr=bar)
                kw_producer.advance()

                ud_pipe.producer_acquire(ud_producer)
                dbar = ud_pipe.producer_get_barrier(ud_producer)
                cute.copy(tma_u, tUg[None, c], tUs[None, ud_producer.index], tma_bar_ptr=dbar)
                cute.copy(tma_gd, tGdg[None, c], tGds[None, ud_producer.index], tma_bar_ptr=dbar)
                ud_producer.advance()

            kw_pipe.producer_tail(kw_producer)
            ud_pipe.producer_tail(ud_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrW = mma_wh.make_fragment_A(sW)
            tCrH = mma_wh.make_fragment_B(sH)
            tCrVTA = mma_upd.make_fragment_A(sVTA)
            tCrKGT = mma_upd.make_fragment_B(sKGT)

            kw_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            ud_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            h_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.h_stages
            )
            vta_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            whf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            updf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for c in cutlass.range(NT, unroll=1):
                # this warp never reads u/gd — wait+release keeps the pipe's
                # umma-side accounting satisfied
                ud_pipe.consumer_wait(ud_consumer)
                ud_pipe.consumer_release(ud_consumer, pipeline.PipelineOp.TCGen05Mma)
                ud_consumer.advance()

                # WH = w @ h^T — heads the critical path
                h_pipe.consumer_wait(h_consumer)
                kw_pipe.consumer_wait(kw_consumer)
                whf_pipe.producer_acquire(whf_producer)
                for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                    mma_wh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_wh,
                        tWH[None, None, None, 0],
                        tCrW[None, None, kk, kw_consumer.index],
                        tCrH[None, None, kk, h_consumer.index],
                        tWH[None, None, None, 0],
                    )
                whf_pipe.producer_commit(whf_producer)
                whf_producer.advance()
                h_pipe.consumer_release(h_consumer)
                h_consumer.advance()

                # UPD = v'^T @ kg
                vta_pipe.consumer_wait(vta_consumer)
                updf_pipe.producer_acquire(updf_producer)
                for kk in cutlass.range(cute.size(tCrVTA, mode=[2]), unroll_full=True):
                    mma_upd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_upd,
                        tUPD[None, None, None, 0],
                        tCrVTA[None, None, kk, 0],
                        tCrKGT[None, None, kk, kw_consumer.index],
                        tUPD[None, None, None, 0],
                    )
                updf_pipe.producer_commit(updf_producer)
                updf_producer.advance()
                vta_pipe.consumer_release(vta_consumer)
                vta_consumer.advance()
                kw_pipe.consumer_release(kw_consumer)
                kw_consumer.advance()

            whf_pipe.producer_tail(whf_producer)
            updf_pipe.producer_tail(updf_producer)

        # ==========================================================================
        # SIMT warps 4..7: v', the h update, checkpoint/v_new stores.
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

            # --- WH -> v' ---
            tWH_2d = tWH[((None, None), 0, 0, None)]
            tiled_t2r_wh = tcgen05.make_tmem_copy(t2r_64_atom, tWH_2d[None, None, 0])
            thr_t2r_wh = tiled_t2r_wh.get_slice(local_tidx)
            tTR_tWH = thr_t2r_wh.partition_S(tWH_2d)
            # rmem operands of a tmem copy must be sized from the D partition of a
            # non-tmem tensor — partition_S folds the lane mode and oversizes.
            tTR_rWH = cute.make_rmem_tensor(
                thr_t2r_wh.partition_D(cute.make_identity_tensor((BT, BV))).shape, f32
            )
            tWHsU = thr_t2r_wh.partition_D(sU)
            tWHrU = cute.make_rmem_tensor(cute.slice_(tWHsU.shape, (None, None, None, 0)), io)
            r2s_x16t_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_vta = cute.make_tiled_copy_D(r2s_x16t_atom, tiled_t2r_wh)
            thr_r2s_vta = tiled_r2s_vta.get_slice(local_tidx)
            tRS_sVTA = thr_r2s_vta.partition_D(sVTA_epi)
            tRS_rVTA = cute.make_rmem_tensor(cute.slice_(tRS_sVTA.shape, (None, None, None, 0)), io)
            r2s_x16_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_vns = cute.make_tiled_copy_D(r2s_x16_atom, tiled_t2r_wh)
            thr_r2s_vns = tiled_r2s_vns.get_slice(local_tidx)
            tRS_sVNS = thr_r2s_vns.partition_D(sVNS)
            tRS_rVNS = cute.make_rmem_tensor(cute.slice_(tRS_sVNS.shape, (None, None, None, 0)), io)

            # --- UPD -> h ---
            tUPD_2d = tUPD[((None, None), 0, 0, None)]
            tiled_t2r_upd = tcgen05.make_tmem_copy(t2r_64_atom, tUPD_2d[None, None, 0])
            thr_t2r_upd = tiled_t2r_upd.get_slice(local_tidx)
            tTR_tUPD = thr_t2r_upd.partition_S(tUPD_2d)
            coordUPD = thr_t2r_upd.partition_D(cute.make_identity_tensor((BV, K)))
            tTR_rUPD = cute.make_rmem_tensor(coordUPD.shape, f32)
            tHreg = cute.make_rmem_tensor(tTR_rUPD.shape, f32)
            # The per-chunk decay is a K-VECTOR (gdn: one scalar). Broadcast it across
            # the BV mode of the UPD fragment so each element picks up exp2's argument
            # for its own k — 001's state-decay pattern, identical fragment geometry.
            sGd_bcast = cute.make_tensor(
                sGd.iterator,
                cute.make_layout((BV, K, self.input_stages), stride=(0, 1, K)),
            )
            tUPDsGd = thr_t2r_upd.partition_D(sGd_bcast)
            tUPDrGd = cute.make_rmem_tensor(cute.slice_(tUPDsGd.shape, (None, None, None, 0)), f32)
            r2s_h_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_h = cute.make_tiled_copy_D(r2s_h_atom, tiled_t2r_upd)
            thr_r2s_h = tiled_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_r2s_h.partition_D(sH_epi)
            tRS_rH = cute.make_rmem_tensor(cute.slice_(tRS_sH.shape, (None, None, None, 0)), io)
            # checkpoint staging: same values, transpose stmatrix into the BV-contiguous
            # buffer the TMA store can express. make_tiled_copy_D keeps the base tiling's
            # value->thread assignment, so element i matches tRS_rH's element i.
            r2s_ck_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ck = cute.make_tiled_copy_D(r2s_ck_atom, tiled_t2r_upd)
            thr_r2s_ck = tiled_r2s_ck.get_slice(local_tidx)
            tRS_sCKS = thr_r2s_ck.partition_D(sCKS)
            tRS_rCKS = cute.make_rmem_tensor(cute.slice_(tRS_sCKS.shape, (None, None, None, 0)), io)

            # TMA store plumbing (v_new per chunk; h checkpoint per chunk)
            bSG_sVNS, bSG_gVN = cpasync.tma_partition(
                tma_vn,
                0,
                cute.make_layout(1),
                cute.group_modes(sVNS, 0, 2),
                cute.group_modes(gVN, 0, 2),
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

            ud_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            whf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            updf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            h_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.h_stages
            )
            vta_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # ---- h := h0; checkpoint 0 is exactly that value ----
            for i in cutlass.range(cute.size(tHreg), unroll_full=True):
                vv, kk = coordUPD[i]
                tHreg[i] = mH0[(kk, v_idx * BV + vv, hv_idx, b_idx)]
            h_pipe.producer_acquire(h_producer)
            for i in cutlass.range(cute.size(tHreg), unroll_full=True, vectorize=True):
                vio = tHreg[i].to(io)
                tRS_rH[i] = vio
                tRS_rCKS[i] = vio
            cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[None, None, None, h_producer.index])
            cute.copy(tiled_r2s_ck, tRS_rCKS, tRS_sCKS[None, None, None, 0])
            cute.arch.fence_proxy("async.shared", space="cta")
            self.simt_sync_barrier.arrive_and_wait()
            # commit h BEFORE the checkpoint store: the store's completion is off the
            # recurrence critical path. Its wait is deferred to the next v_new-store
            # acquire (num_stages=1 waits on ALL outstanding groups), which also
            # protects the sCKS overwrite a chunk later.
            h_pipe.producer_commit(h_producer)
            h_producer.advance()
            if warp_idx == self.simt_warp_id[0]:
                cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, 0])
                tma_store_pipeline.producer_commit()

            for c in cutlass.range(NT, unroll=1):
                ud_pipe.consumer_wait(ud_consumer)
                dcrd = (None, None, None, ud_consumer.index)
                cute.copy(io_cp_atom, tWHsU[dcrd], tWHrU)

                # v' = u - WH; the same bf16 value is the UPD operand AND v_new (kg
                # carries what gdn's v~ rescale did here).
                whf_pipe.consumer_wait(whf_consumer)
                cute.copy(tiled_t2r_wh, tTR_tWH[None, None, None, 0], tTR_rWH)
                cute.arch.fence_view_async_tmem_load()
                whf_pipe.consumer_release(whf_consumer)
                whf_consumer.advance()
                vta_pipe.producer_acquire(vta_producer)
                for i in cutlass.range(cute.size(tTR_rWH), unroll_full=True, vectorize=True):
                    vp = (tWHrU[i].to(f32) - tTR_rWH[i]).to(io)
                    tRS_rVTA[i] = vp
                    tRS_rVNS[i] = vp
                # the MMA operand first: it unblocks UPD
                cute.copy(tiled_r2s_vta, tRS_rVTA, tRS_sVTA[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                vta_pipe.producer_commit(vta_producer)
                vta_producer.advance()
                # acquire FIRST: it waits on last chunk's stores (long complete), not
                # the one about to be issued — sVNS/sCKS reuse stays safe via the
                # barrier, and no fresh-store completion wait lands on the h chain.
                if warp_idx == self.simt_warp_id[0]:
                    tma_store_pipeline.producer_acquire()
                self.simt_sync_barrier.arrive_and_wait()
                cute.copy(tiled_r2s_vns, tRS_rVNS, tRS_sVNS[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                if warp_idx == self.simt_warp_id[0]:
                    cute.copy(tma_vn, bSG_sVNS[None, 0], bSG_gVN[None, c])
                    tma_store_pipeline.producer_commit()

                # h update: per-dim decay — each fragment element's k picks its own factor.
                cute.copy(f32_cp_atom, tUPDsGd[dcrd], tUPDrGd)
                updf_pipe.consumer_wait(updf_consumer)
                cute.copy(tiled_t2r_upd, tTR_tUPD[None, None, None, 0], tTR_rUPD)
                cute.arch.fence_view_async_tmem_load()
                updf_pipe.consumer_release(updf_consumer)
                updf_consumer.advance()
                for i in cutlass.range(cute.size(tHreg), unroll_full=True, vectorize=True):
                    dec = cute.math.exp2(tUPDrGd[i], fastmath=True)
                    tHreg[i] = dec * tHreg[i] + tTR_rUPD[i]
                if c + 1 < NT:
                    h_pipe.producer_acquire(h_producer)
                    for i in cutlass.range(cute.size(tHreg), unroll_full=True, vectorize=True):
                        vio = tHreg[i].to(io)
                        tRS_rH[i] = vio
                        tRS_rCKS[i] = vio
                    cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[None, None, None, h_producer.index])
                    cute.copy(tiled_r2s_ck, tRS_rCKS, tRS_sCKS[None, None, None, 0])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.simt_sync_barrier.arrive_and_wait()
                    # h first — the checkpoint store is off the recurrence critical
                    # path (completion deferred to the next v_new-store acquire).
                    h_pipe.producer_commit(h_producer)
                    h_producer.advance()
                    if warp_idx == self.simt_warp_id[0]:
                        cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, c + 1])
                        tma_store_pipeline.producer_commit()

                ud_pipe.consumer_release(ud_consumer, pipeline.PipelineOp.AsyncThread)
                ud_consumer.advance()

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
# stage kernels: outputs are allocated PER CALL and retargeted (never cache-owned, see
# gdn's dbg_alias writeup); internal scratch (gdc) stays cached and is refilled before
# each launch.
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


def _call_key(kg, w, u, g2, h0):
    def sig(t):
        return (t.shape, t.stride(), t.dtype) if t is not None else None

    return (sig(kg), sig(w), sig(u), sig(g2), sig(h0), torch.cuda.current_stream().cuda_stream)


def kda_cute_fwdh_call(
    kg: torch.Tensor,  # [B,T,HV,K] bf16/fp16 — k * exp2(G - g2), from recompute_w_u
    w: torch.Tensor,  # [B,T,HV,K]
    u: torch.Tensor,  # [B,T,HV,V]
    g2: torch.Tensor,  # [B,T,HV,K] fp32, chunk-local cumsum / ln2 (only last rows used)
    h0: torch.Tensor | None,  # [B,HV,K,V] fp32 or None
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _call_key(kg, w, u, g2, h0)
    ent = _CALL_CACHE.get(key)
    outs = None  # set on the miss path, where the outputs were allocated to build the views
    if ent is None:
        B, T, HV, K = kg.shape
        V = u.shape[3]
        NT = T // 64
        assert T % 64 == 0
        assert V % 64 == 0

        h = torch.empty(B, NT, HV, K, V, device=kg.device, dtype=kg.dtype)
        v_new = torch.empty(B, T, HV, V, device=kg.device, dtype=u.dtype)
        # Per-chunk decay vectors, [B,HV,NT,K] with K contiguous so the cute view
        # (K,NT,HV,B) has a static stride-1 mode 0. Persistent scratch, refilled below.
        gdc = torch.empty(B, HV, NT, K, device=g2.device, dtype=g2.dtype)
        h0_t = (
            h0
            if h0 is not None
            else torch.zeros(B, HV, K, V, device=kg.device, dtype=torch.float32)
        )

        io_dtype = cutlass.BFloat16 if kg.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V)

        cw = _cute_view(w, (1, 3, 2, 0), (0, 2, 3))
        ckgt = _cute_view(kg, (3, 1, 2, 0), (1, 2, 3))
        cu = _cute_view(u, (1, 3, 2, 0), (0, 2, 3))
        cgd = _cute_view(gdc, (3, 2, 1, 0), (1, 2, 3))
        ch0 = _cute_view(h0_t, (2, 3, 1, 0), (2, 3))
        cvn = _cute_view(v_new, (1, 3, 2, 0), (0, 2, 3))
        chck = _cute_view(h, (4, 3, 1, 2, 0), (2, 3, 4))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = KdaFwdHKernel(io_dtype, K, V_TILE=64)
            compiled = cute.compile(
                kernel_obj,
                cw,
                ckgt,
                cu,
                cgd,
                ch0,
                cvn,
                chck,
                stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (cw, ckgt, cu, cgd, ch0, cvn, chck, stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (h, v_new)
        ent = (compiled, args, _out_specs(h, v_new), gdc, h0_t if h0 is None else None)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, gdc, zeros_h0 = ent
    cw, ckgt, cu, _, ch0, cvn, chck, _ = args
    if outs is None:
        outs = _alloc_outs(out_specs, kg.device)
    h, v_new = outs
    _retarget(chck, h)
    _retarget(cvn, v_new)
    _retarget(cw, w)
    _retarget(ckgt, kg)
    _retarget(cu, u)
    if h0 is not None:
        _retarget(ch0, h0)
    # refill the decay staging in place: g2's last row per chunk, [B,T,HV,K] -> [B,HV,NT,K]
    gdc.copy_(g2[:, 63::64].transpose(1, 2))
    compiled(*args)
    return h, v_new


# Minimum grid size (CTAs) for the cute path; dbg_fwdh sets this to 0 to force the cute
# kernel on small correctness shapes.
_MIN_CTAS = 256


def chunk_gated_delta_rule_fwd_h_cute(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    """Drop-in for fla's chunk_gated_delta_rule_fwd_h on the kda (gk, per-dim) path,
    falling back on shapes the cute kernel does not cover. `k` must be the pre-scaled kg
    (per VALUE head) — exactly what fla's chunk_kda_bwd passes. The backward's recompute
    path never wants the final state, so output_final_state is delegated rather than
    ported."""
    B, T, HK, K = k.shape
    HV, V = u.shape[2], u.shape[3]
    supported = (
        gk is not None
        and g is None
        and HK == HV  # kg rides at value heads; an H-headed k means a non-kda caller
        and not state_v_first
        and not output_final_state
        and save_new_value
        and cu_seqlens is None
        and chunk_indices is None
        and chunk_size == 64
        and T % 64 == 0
        and K in (64, 128)
        and V % 64 == 0
        and k.dtype in (torch.bfloat16, torch.float16)
        # the serial scan only wins with a grid that fills the GPU — same gate as
        # bwd_dhu. _MIN_CTAS is a module global so dbg_fwdh can zero it — otherwise
        # every dbg-sized case silently falls back and the "comparison" is fla vs fla.
        and B * HV * (V // 64) >= _MIN_CTAS
    )
    if not supported:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

        return chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )

    h, v_new = kda_cute_fwdh_call(k, w, u, gk, initial_state)
    return h, v_new, None
