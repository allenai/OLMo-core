"""CuTe port of fla's `chunk_gated_delta_rule_bwd_dhu` — the reverse state scan (stage 4).

Per (b, hv) and per chunk c = NT-1 .. 0, with dh [K, V] the running state gradient
(fp32, starts at dht):

    dh_ck[c] = dh                                    (bf16 checkpoint -> HBM, pre-update)
    dv2      = (k @ dh_bf16) * exp2(G - g2_i) + dv   (completes stage-3's dv_local)
    dh       = exp2(G) * dh + scale * (q * exp2(g2))^T @ do - w^T @ dv2
    after chunk 0: dh0 = dh                          (fp32)

This kernel is the forward scan's mirror, and it reuses kernel_fwd.py's structure
verbatim: one CTA per (b, hv, v_tile) with the state held TRANSPOSED — dh^T [BV, K] —
in the SIMT warpgroup's registers, so the state's M-dim is 64 and one warpgroup owns the
whole recurrence. The transposed update is

    dv2^T-free form:  dh^T += dog^T @ q - dv2^T @ w,   dog = do * exp2(g2_i) * scale

i.e. the gate/scale factor is folded into *do*, not q (it sits on the contraction index,
so either side works). That choice keeps every SIMT-built tile [BT, BV]-shaped and makes
q and w pure TMA operands ([K, BT] mn-major views, the forward's kt pattern). The two
tiled MMAs are the forward's mma_wh and mma_dh with roles renamed:

    DV  = k @ dh^T          tile (BT, BV, K), A = k k-major TMA, B = sDH k-major (SIMT)
    UPD = dog^T q - dv2^T w tile (BV, K, BT), A = SIMT trans-stmatrix, B = q^T/w^T mn TMA

The subtraction is an accumulating MMA with a negated A operand: SIMT writes -dv2 into
the operand buffer and +dv2 into the store staging (bf16 negation is exact).

Precision contract vs fla: dh stays fp32 in registers across chunks; it is cast to bf16
exactly where fla casts (the k@dh operand and the checkpoint store). The one deliberate
divergence: fla's q-side term is a tf32 dot of fp32 (q*exp2(g2)); ours rounds dog to
bf16. The error lands on a term that random-walks over NT chunks — measured well inside
the stage tolerances (see dbg_dhu.py).

Logical gmem mode order (M/N, K_contract, rest) per operand:
    k:        (T, K, H, B)        A of DV
    q^T, w^T: (K, T, H|HV, B)     mn-major B of UPD
    do, dv:   (T, V, HV, B)       SIMT-only, linear smem
    g2:       (BT, NT, HV, B)     fp32, staged host-side so mode 0 stays static
    dht, dh0: (K, V, HV, B)       fp32 scalar gather/scatter, once per CTA
    dv2:      (T, V, HV, B)       TMA store
    dh_ck:    (V, K, NT, HV, B)   TMA store of the [BV, K] checkpoint slice straight from
                                  the swizzled operand bytes (ROW_MAJOR epi view of sDH)
"""

from __future__ import annotations

import ctypes
from typing import Type

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class GdnBwdDhuKernel:
    """Reverse dh scan. One CTA per (b, hv, v_tile); serial over chunks, newest first.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT recurrence (dog, dv2, the dh update,
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
        self.tile_dv = (self.BT, self.BV, self.K)  # k @ dh^T
        self.tile_upd = (self.BV, self.K, self.BT)  # dog^T @ q  /  (-dv2)^T @ w

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
        mma_upd = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_upd[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_dv, mma_upd

    def _setup_attributes(self):
        mma_dv, mma_upd = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.k_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dv, self.tile_dv, self.io_dtype, self.input_stages
        )
        self.qt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_upd, self.tile_upd, self.io_dtype, self.input_stages
        )
        self.wt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_upd, self.tile_upd, self.io_dtype, self.input_stages
        )
        # do / dv are SIMT-only: plain row-major (BT, BV), BV contiguous
        self.do_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        self.dvi_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        self.g2_smem_layout = cute.make_layout((BT, self.input_stages))

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
        # dog / -dv2 as A operands of UPD ([BV, BT] k-major); written via transpose
        # stmatrix through COL_MAJOR (BT, BV) epi views (the forward's v~' pattern).
        self.dog_smem_layout = sm100_utils.make_smem_layout_a(
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
                self.io_dtype, cute.slice_(self.k_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.qt_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.wt_smem_layout, (None, None, None, 0))
            )
        )
        self.num_dod_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.do_smem_layout, (None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.dvi_smem_layout, (None, None, 0))
            )
            + cute.size_in_bytes(
                cutlass.Float32, cute.slice_(self.g2_smem_layout, (None, 0))
            )
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
        k: cute.Tensor,  # (T, K, H, B)
        qt: cute.Tensor,  # (K, T, H, B) — same storage as q
        wt: cute.Tensor,  # (K, T, HV, B) — same storage as w
        do: cute.Tensor,  # (T, V, HV, B)
        dvi: cute.Tensor,  # (T, V, HV, B) — stage 3's dv_local
        g2: cute.Tensor,  # (BT, NT, HV, B) fp32
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

        tma_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), k,
            cute.slice_(self.k_smem_layout, (None, None, None, 0)),
            self.tile_dv, mma_dv, cluster_vmnk,
        )
        tma_qt, tma_tensor_qt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), qt,
            cute.slice_(self.qt_smem_layout, (None, None, None, 0)),
            self.tile_upd, mma_upd, cluster_vmnk,
        )
        tma_wt, tma_tensor_wt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), wt,
            cute.slice_(self.wt_smem_layout, (None, None, None, 0)),
            self.tile_upd, mma_upd, cluster_vmnk,
        )
        tma_do, tma_tensor_do = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), do,
            cute.slice_(self.do_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        tma_dvi, tma_tensor_dvi = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), dvi,
            cute.slice_(self.dvi_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        g2_cta_v_layout = cute.slice_(
            cute.make_identity_layout(g2.shape), (None, 0, 0, 0)
        )
        tma_g2, tma_tensor_g2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), g2,
            cute.slice_(self.g2_smem_layout, (None, 0)),
            g2_cta_v_layout,
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

        B = cute.size(k, mode=[3])
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
            doga_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dv2n_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dvf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            updf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.k_smem_layout)], swz_align  # type: ignore
            ]
            smem_qt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.qt_smem_layout)], swz_align  # type: ignore
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
            smem_g2: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.g2_smem_layout)], lin_align  # type: ignore
            ]
            smem_dh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dh_smem_layout)], swz_align  # type: ignore
            ]
            smem_cks: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.cks_smem_layout)], swz_align  # type: ignore
            ]
            smem_dog: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dog_smem_layout)], swz_align  # type: ignore
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

        self.gdn_cute_bwd_dhu(
            tma_k, tma_tensor_k,
            tma_qt, tma_tensor_qt,
            tma_wt, tma_tensor_wt,
            tma_do, tma_tensor_do,
            tma_dvi, tma_tensor_dvi,
            tma_g2, tma_tensor_g2,
            tma_dv2, tma_tensor_dv2,
            tma_ck, tma_tensor_ck,
            dht,
            dh0,
            scale,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def gdn_cute_bwd_dhu(
        self,
        tma_k: cute.CopyAtom, mK: cute.Tensor,
        tma_qt: cute.CopyAtom, mQT: cute.Tensor,
        tma_wt: cute.CopyAtom, mWT: cute.Tensor,
        tma_do: cute.CopyAtom, mDO: cute.Tensor,
        tma_dvi: cute.CopyAtom, mDVI: cute.Tensor,
        tma_g2: cute.CopyAtom, mG2: cute.Tensor,
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
        mma_dv, mma_upd = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_k, tma_qt, tma_wt, tma_do, tma_dvi, tma_g2, tma_dv2, tma_ck]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mWT, mode=[2])
        H = cute.size(mK, mode=[2])
        NV = cute.size(mDO, mode=[1]) // BV
        T = cute.size(mK, mode=[0])
        NT = T // BT
        v_idx = bidx % NV
        hv_idx = (bidx // NV) % HV
        b_idx = bidx // (NV * HV)
        h_idx = hv_idx // (HV // H)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sK = storage.smem_k.get_tensor(self.k_smem_layout.outer, swizzle=self.k_smem_layout.inner)
        sQT = storage.smem_qt.get_tensor(self.qt_smem_layout.outer, swizzle=self.qt_smem_layout.inner)
        sWT = storage.smem_wt.get_tensor(self.wt_smem_layout.outer, swizzle=self.wt_smem_layout.inner)
        sDO = storage.smem_do.get_tensor(self.do_smem_layout)
        sDVI = storage.smem_dvi.get_tensor(self.dvi_smem_layout)
        sG2 = storage.smem_g2.get_tensor(self.g2_smem_layout)
        sDH = storage.smem_dh.get_tensor(self.dh_smem_layout.outer, swizzle=self.dh_smem_layout.inner)
        sDH_epi = storage.smem_dh.get_tensor(self.dh_epi_layout.outer, swizzle=self.dh_epi_layout.inner)
        sCKS = storage.smem_cks.get_tensor(self.cks_smem_layout.outer, swizzle=self.cks_smem_layout.inner)
        sDOG = storage.smem_dog.get_tensor(self.dog_smem_layout.outer, swizzle=self.dog_smem_layout.inner)
        sDOG_epi = storage.smem_dog.get_tensor(self.ops_epi_layout.outer, swizzle=self.ops_epi_layout.inner)
        sDV2N = storage.smem_dv2n.get_tensor(self.dv2n_smem_layout.outer, swizzle=self.dv2n_smem_layout.inner)
        sDV2N_epi = storage.smem_dv2n.get_tensor(self.ops_epi_layout.outer, swizzle=self.ops_epi_layout.inner)
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
        # do/dv/g2 are SIMT-only data, but the pipe keeps the proven multi-consumer
        # plumbing: the MMA warp waits and releases each stage without reading it.
        dod_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, simt_threads
            ),
            tx_count=self.num_dod_load_bytes,
            barrier_storage=storage.dod_full.data_ptr(),
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
        doga_pipe = make_simt_to_mma_pipe(storage.doga_full.data_ptr(), 1)
        dv2n_pipe = make_simt_to_mma_pipe(storage.dv2n_full.data_ptr(), 1)
        dvf_pipe = make_mma_to_simt_pipe(storage.dvf_full.data_ptr())
        updf_pipe = make_mma_to_simt_pipe(storage.updf_full.data_ptr())

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
        tUPD = acc_tensor(mma_upd, self.tile_upd, self.tmem_upd_offset)

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gK = cute.local_tile(mK, (BT, K), (None, 0, h_idx, b_idx))  # (BT,K,NT)
        gQT = cute.local_tile(mQT, (K, BT), (0, None, h_idx, b_idx))  # (K,BT,NT)
        gWT = cute.local_tile(mWT, (K, BT), (0, None, hv_idx, b_idx))
        gDO = cute.local_tile(mDO, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gDVI = cute.local_tile(mDVI, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gG2 = mG2[(None, None, hv_idx, b_idx)]  # (BT, NT)
        gDV2 = cute.local_tile(mDV2, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gCK = cute.local_tile(mCK, (BV, K), (v_idx, 0, None, hv_idx, b_idx))  # (BV,K,NT)

        # ==========================================================================
        # TMA warp — chunks walked newest-first
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_dv = mma_dv.get_slice(0)
            thr_mma_upd = mma_upd.get_slice(0)

            tK_mma = thr_mma_dv.partition_A(gK)
            tQT_mma = thr_mma_upd.partition_B(gQT)
            tWT_mma = thr_mma_upd.partition_B(gWT)

            cta1 = cute.make_layout(1)
            tKs, tKg = cpasync.tma_partition(
                tma_k, 0, cta1, cute.group_modes(sK, 0, 3), cute.group_modes(tK_mma, 0, 3)
            )
            tQTs, tQTg = cpasync.tma_partition(
                tma_qt, 0, cta1, cute.group_modes(sQT, 0, 3), cute.group_modes(tQT_mma, 0, 3)
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
            tG2s, tG2g = cpasync.tma_partition(
                tma_g2, 0, cta1, cute.group_modes(sG2, 0, 1), cute.group_modes(gG2, 0, 1)
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
                cute.copy(tma_k, tKg[None, c], tKs[None, kqw_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_qt, tQTg[None, c], tQTs[None, kqw_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_wt, tWTg[None, c], tWTs[None, kqw_producer.index], tma_bar_ptr=bar)
                kqw_producer.advance()

                dod_pipe.producer_acquire(dod_producer)
                dbar = dod_pipe.producer_get_barrier(dod_producer)
                cute.copy(tma_do, tDOg[None, c], tDOs[None, dod_producer.index], tma_bar_ptr=dbar)
                cute.copy(tma_dvi, tDVIg[None, c], tDVIs[None, dod_producer.index], tma_bar_ptr=dbar)
                cute.copy(tma_g2, tG2g[None, c], tG2s[None, dod_producer.index], tma_bar_ptr=dbar)
                dod_producer.advance()

            kqw_pipe.producer_tail(kqw_producer)
            dod_pipe.producer_tail(dod_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrK = mma_dv.make_fragment_A(sK)
            tCrDH = mma_dv.make_fragment_B(sDH)
            tCrDOG = mma_upd.make_fragment_A(sDOG)
            tCrDV2N = mma_upd.make_fragment_A(sDV2N)
            tCrQT = mma_upd.make_fragment_B(sQT)
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
            doga_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dv2n_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dvf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            updf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for rc in cutlass.range(NT, unroll=1):
                # this warp never reads do/dv/g2 — wait+release keeps the pipe's
                # umma-side accounting satisfied
                dod_pipe.consumer_wait(dod_consumer)
                dod_pipe.consumer_release(dod_consumer, pipeline.PipelineOp.TCGen05Mma)
                dod_consumer.advance()

                # DV = k @ dh^T — heads the critical path
                dh_pipe.consumer_wait(dh_consumer)
                kqw_pipe.consumer_wait(kqw_consumer)
                dvf_pipe.producer_acquire(dvf_producer)
                for kk in cutlass.range(cute.size(tCrDH, mode=[2]), unroll_full=True):
                    mma_dv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dv, tDV[None, None, None, 0],
                        tCrK[None, None, kk, kqw_consumer.index],
                        tCrDH[None, None, kk, dh_consumer.index],
                        tDV[None, None, None, 0],
                    )
                dvf_pipe.producer_commit(dvf_producer)
                dvf_producer.advance()
                dh_pipe.consumer_release(dh_consumer)
                dh_consumer.advance()

                # UPD = dog^T @ q — dog only needs do/g2, so it lands while SIMT
                # is still building dv2
                doga_pipe.consumer_wait(doga_consumer)
                updf_pipe.producer_acquire(updf_producer)
                for kk in cutlass.range(cute.size(tCrDOG, mode=[2]), unroll_full=True):
                    mma_upd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_upd, tUPD[None, None, None, 0],
                        tCrDOG[None, None, kk, 0],
                        tCrQT[None, None, kk, kqw_consumer.index],
                        tUPD[None, None, None, 0],
                    )
                doga_pipe.consumer_release(doga_consumer)
                doga_consumer.advance()

                # UPD += (-dv2)^T @ w
                dv2n_pipe.consumer_wait(dv2n_consumer)
                for kk in cutlass.range(cute.size(tCrDV2N, mode=[2]), unroll_full=True):
                    mma_upd.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                    cute.gemm(
                        mma_upd, tUPD[None, None, None, 0],
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
        # SIMT warps 4..7: dog, dv2, the dh update, checkpoint/dv2 stores, dh0.
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

            # --- DV -> dog / dv2 ---
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
            tDVrDO = cute.make_rmem_tensor(
                cute.slice_(tDVsDO.shape, (None, None, None, 0)), io
            )
            tDVsDVI = thr_t2r_dv.partition_D(sDVI)
            tDVrDVI = cute.make_rmem_tensor(
                cute.slice_(tDVsDVI.shape, (None, None, None, 0)), io
            )
            sG2_rowv = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BV, self.input_stages), stride=(1, 0, BT)),
            )
            tDVsG2 = thr_t2r_dv.partition_D(sG2_rowv)
            tDVrG2 = cute.make_rmem_tensor(
                cute.slice_(tDVsG2.shape, (None, None, None, 0)), f32
            )
            r2s_x16t_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ops = cute.make_tiled_copy_D(r2s_x16t_atom, tiled_t2r_dv)
            thr_r2s_ops = tiled_r2s_ops.get_slice(local_tidx)
            tRS_sDOG = thr_r2s_ops.partition_D(sDOG_epi)
            tRS_sDV2N = thr_r2s_ops.partition_D(sDV2N_epi)
            tRS_rDOG = cute.make_rmem_tensor(
                cute.slice_(tRS_sDOG.shape, (None, None, None, 0)), io
            )
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
            r2s_dh_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_dh = cute.make_tiled_copy_D(r2s_dh_atom, tiled_t2r_upd)
            thr_r2s_dh = tiled_r2s_dh.get_slice(local_tidx)
            tRS_sDH = thr_r2s_dh.partition_D(sDH_epi)
            tRS_rDH = cute.make_rmem_tensor(
                cute.slice_(tRS_sDH.shape, (None, None, None, 0)), io
            )
            # checkpoint staging: same values, transpose stmatrix into the BV-contiguous
            # buffer the TMA store can express. make_tiled_copy_D keeps the base tiling's
            # value->thread assignment, so element i matches tRS_rDH's element i.
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

            dod_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            dvf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            updf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dh_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dh_stages
            )
            doga_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
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
                g_last = sG2[BT - 1, dod_consumer.index]
                exp_g_last = cute.math.exp2(g_last, fastmath=True)
                cute.copy(f32_cp_atom, tDVsG2[dcrd], tDVrG2)

                # dog = do * exp2(g2_i) * scale — no DV dependency, ships first
                cute.copy(io_cp_atom, tDVsDO[dcrd], tDVrDO)
                doga_pipe.producer_acquire(doga_producer)
                for i in cutlass.range(
                    cute.size(tTR_rDV), unroll_full=True, vectorize=True
                ):
                    gd = cute.math.exp2(tDVrG2[i], fastmath=True)
                    tRS_rDOG[i] = (tDVrDO[i].to(f32) * gd * scale).to(io)
                cute.copy(tiled_r2s_ops, tRS_rDOG, tRS_sDOG[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                doga_pipe.producer_commit(doga_producer)
                doga_producer.advance()

                # dv2 = DV * exp2(G - g2_i) + dv_local
                cute.copy(io_cp_atom, tDVsDVI[dcrd], tDVrDVI)
                dvf_pipe.consumer_wait(dvf_consumer)
                cute.copy(tiled_t2r_dv, tTR_tDV[None, None, None, 0], tTR_rDV)
                cute.arch.fence_view_async_tmem_load()
                dvf_pipe.consumer_release(dvf_consumer)
                dvf_consumer.advance()
                dv2n_pipe.producer_acquire(dv2n_producer)
                for i in cutlass.range(
                    cute.size(tTR_rDV), unroll_full=True, vectorize=True
                ):
                    dec = cute.math.exp2(g_last - tDVrG2[i], fastmath=True)
                    dv2f = tTR_rDV[i] * dec + tDVrDVI[i].to(f32)
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

                # dh update
                updf_pipe.consumer_wait(updf_consumer)
                cute.copy(tiled_t2r_upd, tTR_tUPD[None, None, None, 0], tTR_rUPD)
                cute.arch.fence_view_async_tmem_load()
                updf_pipe.consumer_release(updf_consumer)
                updf_consumer.advance()
                for i in cutlass.range(
                    cute.size(tDHreg), unroll_full=True, vectorize=True
                ):
                    tDHreg[i] = exp_g_last * tDHreg[i] + tTR_rUPD[i]
                if rc + 1 < NT:
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
                    # dh first — the checkpoint store is off the recurrence critical
                    # path (completion deferred to the next dv2-store acquire).
                    dh_pipe.producer_commit(dh_producer)
                    dh_producer.advance()
                    if warp_idx == self.simt_warp_id[0]:
                        cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, c - 1])
                        tma_store_pipeline.producer_commit()

                dod_pipe.consumer_release(
                    dod_consumer, pipeline.PipelineOp.AsyncThread
                )
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
# stage kernels: outputs (dh, dh0, dv2) are OWNED by the cache entry and overwritten by
# the next same-layout call. Note dh is the [B,NT,HV,K,V] checkpoint tensor (2.15GB at
# prod8192) and stays resident for the cache's lifetime.
_CALL_CACHE: dict = {}


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def _spec(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.dtype)


def _alloc(specs: tuple, device: torch.device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


def _call_key(q, k, w, do, dv, g2, dht, scale):
    def sig(t):
        return (t.shape, t.stride(), t.dtype) if t is not None else None

    return (sig(q), sig(k), sig(w), sig(do), sig(dv), sig(g2), sig(dht), scale,
            torch.cuda.current_stream().cuda_stream)


def gdn_cute_dhu_call(
    q: torch.Tensor,  # [B,T,H,K] bf16/fp16
    k: torch.Tensor,
    w: torch.Tensor,  # [B,T,HV,K]
    do: torch.Tensor,  # [B,T,HV,V]
    dv: torch.Tensor,  # [B,T,HV,V] — stage 3's dv_local
    g2: torch.Tensor,  # [B,T,HV] fp32, chunk-local cumsum / ln2
    dht: torch.Tensor | None,  # [B,HV,K,V] fp32 or None
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = _call_key(q, k, w, do, dv, g2, dht, scale)
    ent = _CALL_CACHE.get(key)
    if ent is None:
        B, T, H, K = q.shape
        HV, V = do.shape[2], do.shape[3]
        NT = T // 64
        assert T % 64 == 0
        assert V % 64 == 0

        dh = torch.empty(B, NT, HV, K, V, device=q.device, dtype=q.dtype)
        dh0 = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32)
        dv2 = torch.empty(B, T, HV, V, device=q.device, dtype=do.dtype)
        g2c = torch.empty(B, HV, NT, 64, device=g2.device, dtype=g2.dtype)
        dht_t = dht if dht is not None else torch.zeros(
            B, HV, K, V, device=q.device, dtype=torch.float32
        )

        io_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V)

        ck = _cute_view(k, (1, 3, 2, 0), (0, 2, 3))
        cqt = _cute_view(q, (3, 1, 2, 0), (1, 2, 3))
        cwt = _cute_view(w, (3, 1, 2, 0), (1, 2, 3))
        cdo = _cute_view(do, (1, 3, 2, 0), (0, 2, 3))
        cdvi = _cute_view(dv, (1, 3, 2, 0), (0, 2, 3))
        cg2 = _cute_view(g2c, (3, 2, 1, 0), (1, 2, 3))
        cdht = _cute_view(dht_t, (2, 3, 1, 0), (2, 3))
        cdv2 = _cute_view(dv2, (1, 3, 2, 0), (0, 2, 3))
        cck = _cute_view(dh, (4, 3, 1, 2, 0), (2, 3, 4))
        cdh0 = _cute_view(dh0, (2, 3, 1, 0), (2, 3))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = GdnBwdDhuKernel(io_dtype, K, V_TILE=64)
            compiled = cute.compile(
                kernel_obj, ck, cqt, cwt, cdo, cdvi, cg2, cdht, cdv2, cck, cdh0,
                cutlass.Float32(scale), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (ck, cqt, cwt, cdo, cdvi, cg2, cdht, cdv2, cck, cdh0,
                cutlass.Float32(scale), stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        ent = (compiled, args, (_spec(dh), _spec(dh0), _spec(dv2)), g2c,
               dht_t if dht is None else None)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, g2c, zeros_dht = ent
    ck, cqt, cwt, cdo, cdvi, _, cdht, cdv2, cck, cdh0, _, _ = args
    # Fresh outputs per call - see the package docstring on why they are not cache-owned.
    dh, dh0, dv2 = _alloc(out_specs, q.device)
    _retarget(cck, dh)
    _retarget(cdh0, dh0)
    _retarget(cdv2, dv2)
    _retarget(ck, k)
    _retarget(cqt, q)
    _retarget(cwt, w)
    _retarget(cdo, do)
    _retarget(cdvi, dv)
    if dht is not None:
        _retarget(cdht, dht)
    B, HV, NT, BT = g2c.shape
    g2c.view(B, HV, NT * BT).copy_(g2.transpose(1, 2))
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
    """Drop-in for fla's chunk_gated_delta_rule_bwd_dhu, falling back on shapes the
    cute kernel does not cover."""
    B, T, H, K = q.shape
    HV, V = do.shape[2], do.shape[3]
    supported = (
        g is not None
        and gk is None
        and not state_v_first
        and cu_seqlens is None
        and chunk_indices is None
        and chunk_size == 64
        and T % 64 == 0
        and K in (64, 128)
        and V % 64 == 0
        and q.dtype in (torch.bfloat16, torch.float16)
        # the serial scan only wins with a grid that fills the GPU: at gva's 64 CTAs the
        # cute kernel lost 1.4ms to fla (bench 004 vs 003), at prod8192's 1024 it wins 1.4x.
        # _MIN_CTAS is a module global so dbg_dhu can zero it — otherwise every dbg-sized
        # case silently falls back and the "comparison" is fla vs fla.
        and B * HV * (V // 64) >= _MIN_CTAS
    )
    if not supported:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

        return chunk_gated_delta_rule_bwd_dhu(
            q=q, k=k, w=w, do=do, dv=dv, g=g, gk=gk, h0=h0, dht=dht, scale=scale,
            state_v_first=state_v_first, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size, chunk_indices=chunk_indices,
        )

    if scale is None:
        scale = K**-0.5
    dh, dh0, dv2 = gdn_cute_dhu_call(q, k, w, do, dv, g, dht, float(scale))
    return dh, (dh0 if h0 is not None else None), dv2
