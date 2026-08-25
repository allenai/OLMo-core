"""The CuTe DSL forward for KDA. See NOTES.md / ALGORITHM.md.

Ported from gdn 002's pipelined fused fwd_h + fwd_o kernel (kernels/gdn/ideas/002-cute-pipeline/
kernel.py, copied and modified — see the ALGORITHM.md diff table). The per-dim gate mostly
disappears from the kernel because fla's intra stage pre-scales the operands:

    qg = q * exp2(g2)                 (intra stores it, disable_recompute=True path)
    kg = k * exp2(G - g2)             (intra always stores it)
    Aqk[i,j] = scale * q_i k_j exp2(g2_i - g2_j), i >= j    (intra computes it — the per-dim
              gate sits INSIDE this dot product, which kills gdn's S1-then-scale trick, so
              Aqk arrives as a loaded operand and gdn's S1 MMA + SIMT Aqk build are deleted)

Per (b, hv, v-tile) CTA, with h [K, BV] fp32 resident in SIMT-warp registers, per chunk c:

    WH = w @ h_c                       (MMA, h from smem bf16)
    v'  = u - WH                       (SIMT)         -> smem bf16 (OI and DH operands)
    OH = qg @ h_c                      (MMA)
    OI = Aqk @ v'                      (MMA, Aqk from smem — TMA loaded)
    DH = kg^T @ v'                     (MMA, kg as mn-major B)
    h_{c+1}[d,:] = exp2(G_d) * h_c[d,:] + DH[d,:]   (SIMT regs, fp32; G a K-VECTOR, gdn's
                                                     one scalar decay widened per dimension)
    o_c = scale * OH + OI              (SIMT epilogue -> TMA store; Aqk already carries scale)

after the last chunk, ht = h fp32 -> TMA store. G = g2[last row of chunk] (log2-space); the
kernel receives it as a per-chunk [K] fp32 vector (gd), sliced host-side from fla's cumsum.
Precision matches fla's structure: qg/kg/w/u/Aqk are the very tensors fla's Triton kernels
produce (bf16), h is cast to bf16 exactly where fla casts (the w@h / qg@h operands), v' is
cast to bf16 before the DH dot, accumulation is fp32 everywhere.

Logical mode order convention (unchanged from gdn): every gmem tensor is viewed so an MMA
operand reads (M-or-N, K_contract, rest...):
    qg, w as A of the BT-M mmas:   (T, K, HV, B)
    kg as mn-major B of DH:        (K, T, HV, B)
    Aqk as A of OI:                (T, BT, HV, B)
    u, o:                          (T, V, HV, B)
    gd:                            (K, NT, HV, B) fp32, K contiguous
    h0, ht:                        (K, V, HV, B) fp32
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


class KdaFwdStateKernel:
    """Fused fwd_h + fwd_o. One CTA per (b, hv, v_tile); serial over chunks.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT-recurrence (v', h update, ht),
    8..11 = SIMT-epilogue (o readout + TMA store). Both SIMT quads are warpgroup-aligned —
    tmem load/store atoms address tmem rows per warp, so a group must be warps 4k..4k+3.
    Warps 2,3 idle through the role branch and only participate in alloc/dealloc barriers.

    vs gdn: the epilogue group lost its Aqk job (Aqk is loaded, not built) and keeps only o;
    it stays a separate group so the o readout never stalls the recurrence.
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
        self.tile_wh = (self.BT, self.BV, self.K)  # w @ h   (also qg @ h)
        self.tile_oi = (self.BT, self.BV, self.BT)  # Aqk @ v'
        self.tile_dh = (self.BV, self.K, self.BT)  # DH^T = v'^T @ kg

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0
        self.mma_warp_id = 1
        self.simt_warp_id = (4, 5, 6, 7)  # recurrence group: v', h update, ht
        self.epi_warp_id = (8, 9, 10, 11)  # epilogue group: o
        self.threads_per_cta = 32 * 12

        self.input_stages = 2
        self.h_stages = 2

        self.simt_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * len(self.simt_warp_id)
        )
        self.epi_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=32 * len(self.epi_warp_id)
        )
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    # ---------------------------------------------------------------------------------

    def _make_tiled_mmas(self):
        io, acc, grp = self.io_dtype, self.acc_dtype, self.cta_group
        mma_wh = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_wh[:2], tcgen05.OperandSource.SMEM,
        )
        # OI: A = Aqk [BT,BT] from SMEM (TMA-loaded; gdn built it in tmem from S1 — deleted)
        mma_oi = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_oi[:2], tcgen05.OperandSource.SMEM,
        )
        # DH^T = v'^T @ kg: A = v' [BV,BT] k-major (as stored), B = kg [K,BT] mn-major,
        # loaded through its own TMA view of the kg storage.
        mma_dh = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dh[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_wh, mma_oi, mma_dh

    def _setup_attributes(self):
        mma_wh, mma_oi, mma_dh = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.qg_smem_layout = sm100_utils.make_smem_layout_a(
            mma_wh, self.tile_wh, self.io_dtype, self.input_stages
        )
        self.kgt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dh, self.tile_dh, self.io_dtype, self.input_stages
        )
        self.w_smem_layout = sm100_utils.make_smem_layout_a(
            mma_wh, self.tile_wh, self.io_dtype, self.input_stages
        )
        self.aqk_smem_layout = sm100_utils.make_smem_layout_a(
            mma_oi, self.tile_oi, self.io_dtype, self.input_stages
        )
        # u is SIMT-only: plain row-major (BT, BV), BV contiguous
        self.u_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        # gd: the per-chunk decay K-vector (g2's last row), fp32. gdn staged BT scalars here.
        self.gd_smem_layout = cute.make_layout((K, self.input_stages))

        # h as B operand of WH/OH ([BV, K] k-major, staged); written by SIMT through the
        # ROW_MAJOR epi view of the same bytes (mamba2's P pattern).
        self.h_smem_layout = sm100_utils.make_smem_layout_b(
            mma_wh, self.tile_wh, self.io_dtype, self.h_stages
        )
        self.h_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BV, K), self.h_stages
        )
        # v' as B operand of OI ([BV, BT] k-major); same value written twice, once per
        # operand layout (vpd is DH's A). gdn's vpd held v~' — that rescale lives in kg now.
        self.vp_smem_layout = sm100_utils.make_smem_layout_b(
            mma_oi, self.tile_oi, self.io_dtype, 1
        )
        self.vp_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BV), 1
        )
        self.vpd_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dh, self.tile_dh, self.io_dtype, 1
        )
        # o staging: (BT, BV) BV-contiguous to match gmem
        self.o_smem_layout = cute.make_layout((BT, BV, 1), stride=(BV, 1, BT * BV))

        # One TMA pipe per consumer side: the MMA warp consumes qg/kgt/w/aqk, the
        # recurrence SIMT group consumes u/gd. (gdn split qk vs wug and had g2 read by both
        # SIMT groups; the epilogue group here consumes no TMA data at all.)
        self.num_mma_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.qg_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.kgt_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.w_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.aqk_smem_layout, (None, None, None, 0))
            )
        )
        self.num_simt_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.u_smem_layout, (None, None, 0))
            )
            + cute.size_in_bytes(
                cutlass.Float32, cute.slice_(self.gd_smem_layout, (None, 0))
            )
        )

        (
            self.tmem_wh_offset,
            self.tmem_oh_offset,
            self.tmem_oi_offset,
            self.tmem_dh_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_wh, mma_oi, mma_dh)

    def _plan_tmem(self, mma_wh, mma_oi, mma_dh):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        wh = acc_cols(mma_wh, self.tile_wh)
        oi = acc_cols(mma_oi, self.tile_oi)
        dh = acc_cols(mma_dh, self.tile_dh)

        off_wh = 0
        off_oh = off_wh + wh
        off_oi = off_oh + wh
        off_dh = off_oi + oi
        total_ = off_dh + dh
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_wh, off_oh, off_oi, off_dh, total

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        qg: cute.Tensor,  # (T, K, HV, B) — q * exp2(g2), fla's intra stores it
        w: cute.Tensor,  # (T, K, HV, B)
        kgt: cute.Tensor,  # (K, T, HV, B) — same storage as kg
        u: cute.Tensor,  # (T, V, HV, B)
        aqk: cute.Tensor,  # (T, BT, HV, B) — intra's Aqk, scale folded in
        gd: cute.Tensor,  # (K, NT, HV, B) fp32 — per-chunk decay vector (g2 last row)
        h0: cute.Tensor,  # (K, V, HV, B) fp32
        o: cute.Tensor,  # (T, V, HV, B)
        ht: cute.Tensor,  # (K, V, HV, B) fp32
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_wh, mma_oi, mma_dh = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_qg, tma_tensor_qg = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), qg,
            cute.slice_(self.qg_smem_layout, (None, None, None, 0)),
            self.tile_wh, mma_wh, cluster_vmnk,
        )
        tma_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), w,
            cute.slice_(self.w_smem_layout, (None, None, None, 0)),
            self.tile_wh, mma_wh, cluster_vmnk,
        )
        tma_kgt, tma_tensor_kgt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), kgt,
            cute.slice_(self.kgt_smem_layout, (None, None, None, 0)),
            self.tile_dh, mma_dh, cluster_vmnk,
        )
        tma_aqk, tma_tensor_aqk = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), aqk,
            cute.slice_(self.aqk_smem_layout, (None, None, None, 0)),
            self.tile_oi, mma_oi, cluster_vmnk,
        )
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), u,
            cute.slice_(self.u_smem_layout, (None, None, 0)),
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
        tma_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), o,
            cute.slice_(self.o_smem_layout, (None, None, 0)),
            (BT, BV),
        )

        B = cute.size(qg, mode=[3])
        HV = cute.size(w, mode=[2])
        NV = cute.size(u, mode=[1]) // BV
        grid = (B * HV * NV, 1, 1)

        swz_align, lin_align = 1024, 128

        # Every `*_full` range backs both halves of a pipeline's mbarrier array: the
        # create() helpers place the full barriers at the base and the empty ones at
        # base + num_stages, so each needs 2 * num_stages Int64s. Under-sizing one
        # silently aliases the next pipeline's barriers — which only deadlocks once the
        # pipe has to wrap (NT >= stages + 1), so short sequences look fine.
        @cute.struct
        class SharedStorage:
            mmain_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            simtin_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            h_full: cute.struct.MemRange[cutlass.Int64, self.h_stages * 2]  # type: ignore
            vp_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            vpd_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            wh_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            oho_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            oio_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dh_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_qg: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.qg_smem_layout)], swz_align  # type: ignore
            ]
            smem_kgt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.kgt_smem_layout)], swz_align  # type: ignore
            ]
            smem_w: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.w_smem_layout)], swz_align  # type: ignore
            ]
            smem_aqk: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.aqk_smem_layout)], swz_align  # type: ignore
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
            smem_vp: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.vp_smem_layout)], swz_align  # type: ignore
            ]
            smem_vpd: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.vp_smem_layout)], swz_align  # type: ignore
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.o_smem_layout)], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
            )

        self.kda_cute_fwd(
            tma_qg, tma_tensor_qg,
            tma_w, tma_tensor_w,
            tma_kgt, tma_tensor_kgt,
            tma_aqk, tma_tensor_aqk,
            tma_u, tma_tensor_u,
            tma_gd, tma_tensor_gd,
            tma_o, tma_tensor_o,
            ht,
            h0,
            scale,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def kda_cute_fwd(
        self,
        tma_qg: cute.CopyAtom, mQG: cute.Tensor,
        tma_w: cute.CopyAtom, mW: cute.Tensor,
        tma_kgt: cute.CopyAtom, mKGT: cute.Tensor,
        tma_aqk: cute.CopyAtom, mAqk: cute.Tensor,
        tma_u: cute.CopyAtom, mU: cute.Tensor,
        tma_gd: cute.CopyAtom, mGd: cute.Tensor,
        tma_o: cute.CopyAtom, mO: cute.Tensor,
        mHT: cute.Tensor,
        mH0: cute.Tensor,
        scale: cutlass.Float32,
    ):
        BT, K, BV = self.BT, self.K, self.BV
        io = self.io_dtype
        f32 = self.acc_dtype
        # Region isolation: layouts/TiledMma built during the host trace cannot be referenced
        # inside the kernel region. They are pure functions of static config — rebuild here.
        self._setup_attributes()
        mma_wh, mma_oi, mma_dh = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_qg, tma_w, tma_kgt, tma_aqk, tma_u, tma_gd, tma_o]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mW, mode=[2])
        NV = cute.size(mU, mode=[1]) // BV
        T = cute.size(mQG, mode=[0])
        NT = T // BT
        v_idx = bidx % NV
        hv_idx = (bidx // NV) % HV
        b_idx = bidx // (NV * HV)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQG = storage.smem_qg.get_tensor(self.qg_smem_layout.outer, swizzle=self.qg_smem_layout.inner)
        sKGT = storage.smem_kgt.get_tensor(self.kgt_smem_layout.outer, swizzle=self.kgt_smem_layout.inner)
        sW = storage.smem_w.get_tensor(self.w_smem_layout.outer, swizzle=self.w_smem_layout.inner)
        sAqk = storage.smem_aqk.get_tensor(self.aqk_smem_layout.outer, swizzle=self.aqk_smem_layout.inner)
        sU = storage.smem_u.get_tensor(self.u_smem_layout)
        sGd = storage.smem_gd.get_tensor(self.gd_smem_layout)
        sH = storage.smem_h.get_tensor(self.h_smem_layout.outer, swizzle=self.h_smem_layout.inner)
        sH_epi = storage.smem_h.get_tensor(self.h_epi_layout.outer, swizzle=self.h_epi_layout.inner)
        sVp = storage.smem_vp.get_tensor(self.vp_smem_layout.outer, swizzle=self.vp_smem_layout.inner)
        sVp_epi = storage.smem_vp.get_tensor(self.vp_epi_layout.outer, swizzle=self.vp_epi_layout.inner)
        sVpd = storage.smem_vpd.get_tensor(self.vpd_smem_layout.outer, swizzle=self.vpd_smem_layout.inner)
        sVpd_epi = storage.smem_vpd.get_tensor(self.vp_epi_layout.outer, swizzle=self.vp_epi_layout.inner)
        sO = storage.smem_o.get_tensor(self.o_smem_layout)

        # ---- pipelines ----
        # The MMA warp consumes every matmul operand (qg, kgt, w, aqk — one TMA pipe), the
        # recurrence group consumes u and gd (the other). The epilogue group consumes no TMA
        # data — its inputs are the OH/OI accumulators.
        simt_threads = 32 * len(self.simt_warp_id)
        epi_threads = 32 * len(self.epi_warp_id)
        mmain_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_mma_load_bytes,
            barrier_storage=storage.mmain_full.data_ptr(),
            defer_sync=True,
        )
        # Not PipelineTmaAsync: its consumer_release only arrives from lane 0 of each
        # warp while the empty barrier expects the full consumer_group count, so
        # producer_tail deadlocks. This is gdn 002's pipe; the MMA warp supplies the
        # umma-side arrive even though it never reads u/gd.
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

        def make_simt_to_mma_pipe(ptr, stages, producer_threads):
            return pipeline.PipelineAsyncUmma.create(
                num_stages=stages,
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

        h_pipe = make_simt_to_mma_pipe(storage.h_full.data_ptr(), self.h_stages, simt_threads)
        vp_pipe = make_simt_to_mma_pipe(storage.vp_full.data_ptr(), 1, simt_threads)
        vpd_pipe = make_simt_to_mma_pipe(storage.vpd_full.data_ptr(), 1, simt_threads)
        wh_pipe = make_mma_to_simt_pipe(storage.wh_full.data_ptr(), simt_threads)
        oho_pipe = make_mma_to_simt_pipe(storage.oho_full.data_ptr(), epi_threads)
        oio_pipe = make_mma_to_simt_pipe(storage.oio_full.data_ptr(), epi_threads)
        dh_pipe = make_mma_to_simt_pipe(storage.dh_full.data_ptr(), simt_threads)

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

        tWH = acc_tensor(mma_wh, self.tile_wh, self.tmem_wh_offset)
        tOH = acc_tensor(mma_wh, self.tile_wh, self.tmem_oh_offset)
        tOI = acc_tensor(mma_oi, self.tile_oi, self.tmem_oi_offset)
        tDH = acc_tensor(mma_dh, self.tile_dh, self.tmem_dh_offset)

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gQG = cute.local_tile(mQG, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)
        gW = cute.local_tile(mW, (BT, K), (None, 0, hv_idx, b_idx))
        gKGT = cute.local_tile(mKGT, (K, BT), (0, None, hv_idx, b_idx))  # (K,BT,NT)
        gAqk = cute.local_tile(mAqk, (BT, BT), (None, 0, hv_idx, b_idx))  # (BT,BT,NT)
        gU = cute.local_tile(mU, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gGd = mGd[(None, None, hv_idx, b_idx)]  # (K, NT)
        gO = cute.local_tile(mO, (BT, BV), (None, v_idx, hv_idx, b_idx))

        # ==========================================================================
        # TMA warp
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_wh = mma_wh.get_slice(0)
            thr_mma_oi = mma_oi.get_slice(0)
            thr_mma_dh = mma_dh.get_slice(0)

            tQG_mma = thr_mma_wh.partition_A(gQG)
            tW_mma = thr_mma_wh.partition_A(gW)
            tKGT_mma = thr_mma_dh.partition_B(gKGT)
            tAqk_mma = thr_mma_oi.partition_A(gAqk)

            cta1 = cute.make_layout(1)
            tQGs, tQGg = cpasync.tma_partition(
                tma_qg, 0, cta1, cute.group_modes(sQG, 0, 3), cute.group_modes(tQG_mma, 0, 3)
            )
            tWs, tWg = cpasync.tma_partition(
                tma_w, 0, cta1, cute.group_modes(sW, 0, 3), cute.group_modes(tW_mma, 0, 3)
            )
            tKGTs, tKGTg = cpasync.tma_partition(
                tma_kgt, 0, cta1, cute.group_modes(sKGT, 0, 3), cute.group_modes(tKGT_mma, 0, 3)
            )
            tAqks, tAqkg = cpasync.tma_partition(
                tma_aqk, 0, cta1, cute.group_modes(sAqk, 0, 3), cute.group_modes(tAqk_mma, 0, 3)
            )
            tUs, tUg = cpasync.tma_partition(
                tma_u, 0, cta1, cute.group_modes(sU, 0, 2), cute.group_modes(gU, 0, 2)
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

            for c in cutlass.range(NT, unroll=1):
                mmain_pipe.producer_acquire(mmain_producer)
                bar = mmain_pipe.producer_get_barrier(mmain_producer)
                cute.copy(tma_qg, tQGg[None, c], tQGs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_w, tWg[None, c], tWs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_kgt, tKGTg[None, c], tKGTs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_aqk, tAqkg[None, c], tAqks[None, mmain_producer.index], tma_bar_ptr=bar)
                mmain_producer.advance()

                simtin_pipe.producer_acquire(simtin_producer)
                sbar = simtin_pipe.producer_get_barrier(simtin_producer)
                cute.copy(tma_u, tUg[None, c], tUs[None, simtin_producer.index], tma_bar_ptr=sbar)
                cute.copy(tma_gd, tGdg[None, c], tGds[None, simtin_producer.index], tma_bar_ptr=sbar)
                simtin_producer.advance()

            mmain_pipe.producer_tail(mmain_producer)
            simtin_pipe.producer_tail(simtin_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrQG = mma_wh.make_fragment_A(sQG)
            tCrW = mma_wh.make_fragment_A(sW)
            tCrH = mma_wh.make_fragment_B(sH)
            tCrKGT = mma_dh.make_fragment_B(sKGT)
            tCrAqk = mma_oi.make_fragment_A(sAqk)
            tCrVp = mma_oi.make_fragment_B(sVp)
            tCrVpd = mma_dh.make_fragment_A(sVpd)

            mmain_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            simtin_mma_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            h_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.h_stages
            )
            vp_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            vpd_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            wh_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            oho_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            oio_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dh_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for c in cutlass.range(NT, unroll=1):
                # WH = w h — first: it heads the critical path.
                mmain_pipe.consumer_wait(mmain_consumer)
                h_pipe.consumer_wait(h_consumer)
                wh_pipe.producer_acquire(wh_producer)
                for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                    mma_wh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_wh, tWH[None, None, None, 0],
                        tCrW[None, None, kk, mmain_consumer.index],
                        tCrH[None, None, kk, h_consumer.index],
                        tWH[None, None, None, 0],
                    )
                wh_pipe.producer_commit(wh_producer)
                wh_producer.advance()

                # OH = qg h
                oho_pipe.producer_acquire(oho_producer)
                for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                    mma_wh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_wh, tOH[None, None, None, 0],
                        tCrQG[None, None, kk, mmain_consumer.index],
                        tCrH[None, None, kk, h_consumer.index],
                        tOH[None, None, None, 0],
                    )
                oho_pipe.producer_commit(oho_producer)
                oho_producer.advance()
                h_pipe.consumer_release(h_consumer)
                h_consumer.advance()

                # DH = kg^T v' — before OI so the h update never queues behind o-work.
                vpd_pipe.consumer_wait(vpd_consumer)
                dh_pipe.producer_acquire(dh_producer)
                for kk in cutlass.range(cute.size(tCrVpd, mode=[2]), unroll_full=True):
                    mma_dh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dh, tDH[None, None, None, 0],
                        tCrVpd[None, None, kk, vpd_consumer.index],
                        tCrKGT[None, None, kk, mmain_consumer.index],
                        tDH[None, None, None, 0],
                    )
                dh_pipe.producer_commit(dh_producer)
                dh_producer.advance()
                vpd_pipe.consumer_release(vpd_consumer)
                vpd_consumer.advance()

                # OI = Aqk v'
                vp_pipe.consumer_wait(vp_consumer)
                oio_pipe.producer_acquire(oio_producer)
                for kk in cutlass.range(cute.size(tCrVp, mode=[2]), unroll_full=True):
                    mma_oi.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_oi, tOI[None, None, None, 0],
                        tCrAqk[None, None, kk, mmain_consumer.index],
                        tCrVp[None, None, kk, vp_consumer.index],
                        tOI[None, None, None, 0],
                    )
                oio_pipe.producer_commit(oio_producer)
                oio_producer.advance()
                vp_pipe.consumer_release(vp_consumer)
                vp_consumer.advance()
                # aqk was this stage's last user
                mmain_pipe.consumer_release(mmain_consumer)
                mmain_consumer.advance()
                # The umma half of simtin's empty arrive — data untouched by this warp,
                # but the barrier count includes one TCGen05Mma consumer. By this point
                # the SIMT group has long consumed u (v' fed OI above), so the wait is
                # never the limiter.
                simtin_pipe.consumer_wait(simtin_mma_consumer)
                simtin_pipe.consumer_release(
                    simtin_mma_consumer, pipeline.PipelineOp.TCGen05Mma
                )
                simtin_mma_consumer.advance()

            wh_pipe.producer_tail(wh_producer)
            oho_pipe.producer_tail(oho_producer)
            oio_pipe.producer_tail(oio_producer)
            dh_pipe.producer_tail(dh_producer)

        # ==========================================================================
        # SIMT recurrence warps 4..7: v' from WH, the h update from DH, ht.
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
            # rmem operands of a tmem copy must be sized from the *D* partition. partition_S
            # on the tmem side folds the lane mode to stride 0, so it reports the whole warp
            # tile as one thread's values. That oversized rmem tensor builds and even
            # verifies at the cute level, then blows up 32x-too-wide tmem_load/store vectors
            # that segfault cute-to-nvvm (which runs with enable_verifier=False). Always
            # partition_D against a non-tmem tensor.
            tTR_rWH = cute.make_rmem_tensor(
                thr_t2r_wh.partition_D(cute.make_identity_tensor((BT, BV))).shape, f32
            )
            tWHsU = thr_t2r_wh.partition_D(sU)
            tWHrU = cute.make_rmem_tensor(
                cute.slice_(tWHsU.shape, (None, None, None, 0)), io
            )
            r2s_x16_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_vp = cute.make_tiled_copy_D(r2s_x16_atom, tiled_t2r_wh)
            thr_r2s_vp = tiled_r2s_vp.get_slice(local_tidx)
            tRS_sVp = thr_r2s_vp.partition_D(sVp_epi)
            tRS_sVpd = thr_r2s_vp.partition_D(sVpd_epi)
            tRS_rVp = cute.make_rmem_tensor(
                cute.slice_(tRS_sVp.shape, (None, None, None, 0)), io
            )

            # --- DH -> h ---
            tDH_2d = tDH[((None, None), 0, 0, None)]
            t2r_128_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), f32
            )
            tiled_t2r_dh = tcgen05.make_tmem_copy(t2r_128_atom, tDH_2d[None, None, 0])
            thr_t2r_dh = tiled_t2r_dh.get_slice(local_tidx)
            tTR_tDH = thr_t2r_dh.partition_S(tDH_2d)
            coordDH = thr_t2r_dh.partition_D(cute.make_identity_tensor((BV, K)))
            tTR_rDH = cute.make_rmem_tensor(coordDH.shape, f32)
            tHreg = cute.make_rmem_tensor(tTR_rDH.shape, f32)
            # The per-chunk decay is a K-VECTOR (gdn: one scalar). Broadcast it across the
            # BV mode of the DH fragment so each fragment element picks up exp2's argument
            # for its own k — same broadcast-view trick gdn used for g2 rows, other axis.
            sGd_bcast = cute.make_tensor(
                sGd.iterator,
                cute.make_layout((BV, K, self.input_stages), stride=(0, 1, K)),
            )
            tDHsGd = thr_t2r_dh.partition_D(sGd_bcast)
            tDHrGd = cute.make_rmem_tensor(
                cute.slice_(tDHsGd.shape, (None, None, None, 0)), f32
            )
            r2s_h_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_h = cute.make_tiled_copy_D(r2s_h_atom, tiled_t2r_dh)
            thr_r2s_h = tiled_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_r2s_h.partition_D(sH_epi)
            tRS_rH = cute.make_rmem_tensor(
                cute.slice_(tRS_sH.shape, (None, None, None, 0)), io
            )

            simtin_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            wh_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dh_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            h_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.h_stages
            )
            vp_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            vpd_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # ---- h := h0 ----
            for i in cutlass.range(cute.size(tHreg), unroll_full=True):
                vv, kk = coordDH[i]
                tHreg[i] = mH0[(kk, v_idx * BV + vv, hv_idx, b_idx)]
            h_pipe.producer_acquire(h_producer)
            for i in cutlass.range(cute.size(tHreg), unroll_full=True, vectorize=True):
                tRS_rH[i] = tHreg[i].to(io)
            cute.copy(tiled_r2s_h, tRS_rH, tRS_sH[None, None, None, h_producer.index])
            cute.arch.fence_proxy("async.shared", space="cta")
            self.simt_sync_barrier.arrive_and_wait()
            h_pipe.producer_commit(h_producer)
            h_producer.advance()

            for c in cutlass.range(NT, unroll=1):
                simtin_pipe.consumer_wait(simtin_consumer)
                scrd = (None, None, None, simtin_consumer.index)

                # v' = u - WH. gdn also built v~' = v'·exp2(G-g2) here; kg carries that
                # factor now, so both stores hold the same value in two operand layouts.
                wh_pipe.consumer_wait(wh_consumer)
                cute.copy(tiled_t2r_wh, tTR_tWH[None, None, None, 0], tTR_rWH)
                cute.arch.fence_view_async_tmem_load()
                wh_pipe.consumer_release(wh_consumer)
                wh_consumer.advance()
                cute.copy(io_cp_atom, tWHsU[scrd], tWHrU)
                vp_pipe.producer_acquire(vp_producer)
                vpd_pipe.producer_acquire(vpd_producer)
                for i in cutlass.range(
                    cute.size(tTR_rWH), unroll_full=True, vectorize=True
                ):
                    tRS_rVp[i] = (tWHrU[i].to(f32) - tTR_rWH[i]).to(io)
                # vpd first: it feeds DH, the next hop of the critical path; vp only feeds
                # the off-path OI, so its store+commit can trail behind a second barrier.
                cute.copy(tiled_r2s_vp, tRS_rVp, tRS_sVpd[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                vpd_pipe.producer_commit(vpd_producer)
                vpd_producer.advance()
                cute.copy(tiled_r2s_vp, tRS_rVp, tRS_sVp[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                vp_pipe.producer_commit(vp_producer)
                vp_producer.advance()

                # h update: per-dim decay — each fragment element's k picks its own factor.
                cute.copy(f32_cp_atom, tDHsGd[scrd], tDHrGd)
                dh_pipe.consumer_wait(dh_consumer)
                cute.copy(tiled_t2r_dh, tTR_tDH[None, None, None, 0], tTR_rDH)
                cute.arch.fence_view_async_tmem_load()
                dh_pipe.consumer_release(dh_consumer)
                dh_consumer.advance()
                for i in cutlass.range(
                    cute.size(tHreg), unroll_full=True, vectorize=True
                ):
                    dec = cute.math.exp2(tDHrGd[i], fastmath=True)
                    tHreg[i] = dec * tHreg[i] + tTR_rDH[i]
                if c + 1 < NT:
                    h_pipe.producer_acquire(h_producer)
                    for i in cutlass.range(
                        cute.size(tHreg), unroll_full=True, vectorize=True
                    ):
                        tRS_rH[i] = tHreg[i].to(io)
                    cute.copy(
                        tiled_r2s_h, tRS_rH, tRS_sH[None, None, None, h_producer.index]
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.simt_sync_barrier.arrive_and_wait()
                    h_pipe.producer_commit(h_producer)
                    h_producer.advance()

                simtin_pipe.consumer_release(
                    simtin_consumer, pipeline.PipelineOp.AsyncThread
                )
                simtin_consumer.advance()

            # ---- ht (fp32): once-per-kernel plain global scatter ----
            for i in cutlass.range(cute.size(tHreg), unroll_full=True):
                vv, kk = coordDH[i]
                mHT[(kk, v_idx * BV + vv, hv_idx, b_idx)] = tHreg[i]

        # ==========================================================================
        # SIMT epilogue warps 8..11: o from OH/OI. Off the critical path — this group
        # trails the recurrence without ever stalling it. (gdn's Aqk job is gone: Aqk is
        # a loaded operand now, and o needs no gate — qg carried it into OH.)
        # ==========================================================================
        elif (
            warp_idx == self.epi_warp_id[0]
            or warp_idx == self.epi_warp_id[1]
            or warp_idx == self.epi_warp_id[2]
            or warp_idx == self.epi_warp_id[3]
        ):
            t2r_64_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), f32
            )

            tOH_2d = tOH[((None, None), 0, 0, None)]
            tiled_t2r_oh = tcgen05.make_tmem_copy(t2r_64_atom, tOH_2d[None, None, 0])
            thr_t2r_oh = tiled_t2r_oh.get_slice(local_tidx)
            tTR_tOH = thr_t2r_oh.partition_S(tOH_2d)
            tTR_rOH = cute.make_rmem_tensor(
                thr_t2r_oh.partition_D(cute.make_identity_tensor((BT, BV))).shape, f32
            )
            tOI_2d = tOI[((None, None), 0, 0, None)]
            tiled_t2r_oio = tcgen05.make_tmem_copy(t2r_64_atom, tOI_2d[None, None, 0])
            thr_t2r_oio = tiled_t2r_oio.get_slice(local_tidx)
            tTR_tOI = thr_t2r_oio.partition_S(tOI_2d)
            tTR_rOI = cute.make_rmem_tensor(
                thr_t2r_oio.partition_D(cute.make_identity_tensor((BT, BV))).shape, f32
            )
            r2s_o_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_o = cute.make_tiled_copy_D(r2s_o_atom, tiled_t2r_oh)
            thr_r2s_o = tiled_r2s_o.get_slice(local_tidx)
            tRS_sO = thr_r2s_o.partition_D(sO)
            tRS_rO = cute.make_rmem_tensor(
                cute.slice_(tRS_sO.shape, (None, None, None, 0)), io
            )

            bSG_sO, bSG_gO = cpasync.tma_partition(
                tma_o, 0, cute.make_layout(1),
                cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2),
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epi_warp_id)
                ),
            )

            oho_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            oio_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            for c in cutlass.range(NT, unroll=1):
                oho_pipe.consumer_wait(oho_consumer)
                cute.copy(tiled_t2r_oh, tTR_tOH[None, None, None, 0], tTR_rOH)
                oio_pipe.consumer_wait(oio_consumer)
                cute.copy(tiled_t2r_oio, tTR_tOI[None, None, None, 0], tTR_rOI)
                cute.arch.fence_view_async_tmem_load()
                oho_pipe.consumer_release(oho_consumer)
                oho_consumer.advance()
                oio_pipe.consumer_release(oio_consumer)
                oio_consumer.advance()
                # scale only on the OH term: fla folds scale into Aqk at the intra stage,
                # so OI already carries it (gla's o kernel does b_o *= scale BEFORE += A@v).
                for i in cutlass.range(
                    cute.size(tTR_rOH), unroll_full=True, vectorize=True
                ):
                    tRS_rO[i] = (scale * tTR_rOH[i] + tTR_rOI[i]).to(io)
                cute.copy(tiled_r2s_o, tRS_rO, tRS_sO[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epi_sync_barrier.arrive_and_wait()
                if warp_idx == self.epi_warp_id[0]:
                    cute.copy(tma_o, bSG_sO[None, 0], bSG_gO[None, c])
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
                self.epi_sync_barrier.arrive_and_wait()

            tma_store_pipeline.producer_tail()

        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr_base)
        return


# --------------------------------------------------------------------------------------------
# host wrapper
# --------------------------------------------------------------------------------------------

_COMPILE_CACHE: dict = {}


# Same call-cache/pointer-poke scheme as gdn (see that file's long comment): marshaled views
# are cached by (shape, stride, dtype) signature and their descriptors retargeted per call
# with one ctypes word-write each. Outputs are NEVER cached — allocated per call and
# retargeted, so same-layout calls (every layer of a model) don't overwrite each other.
# Internal scratch (gdc) may be cached: it is refilled before each launch and never escapes.
_CALL_CACHE: dict = {}


def _call_key(qg, kg, w, u, aqk, g2, h0, scale):
    def sig(t):
        return (t.shape, t.stride(), t.dtype)

    return (sig(qg), sig(kg), sig(w), sig(u), sig(aqk), sig(g2), sig(h0), scale,
            torch.cuda.current_stream().cuda_stream)


_AQK_UPPER_MASK: dict = {}


def _zero_aqk_upper(aqk: torch.Tensor) -> None:
    """Zero the upper triangle of every 64x64 Aqk chunk tile, in place.

    fla allocates Aqk with torch.empty and its intra kernels only ever store the
    diagonal and lower 16x16 blocks; fla's own o kernel masks i>=j at load time, so
    the garbage above the diagonal is invisible to it — but our MMA contracts the
    full tile. masked_fill_, not multiply: pool garbage can be nan, and nan*0 = nan.
    """
    B, T, HV, BT = aqk.shape
    key = (aqk.device, BT)
    m = _AQK_UPPER_MASK.get(key)
    if m is None:
        m = torch.ones(BT, BT, dtype=torch.bool, device=aqk.device).triu_(1)[:, None, :]
        _AQK_UPPER_MASK[key] = m
    aqk.view(B, T // BT, BT, HV, BT).masked_fill_(m, 0)


def kda_cute_fwd_call(
    qg: torch.Tensor,  # [B,T,HV,K] bf16/fp16 — q * exp2(g2), from intra
    kg: torch.Tensor,  # [B,T,HV,K] — k * exp2(G - g2), from intra
    w: torch.Tensor,  # [B,T,HV,K]
    u: torch.Tensor,  # [B,T,HV,V]
    aqk: torch.Tensor,  # [B,T,HV,BT] — intra's Aqk, scale folded in
    g2: torch.Tensor,  # [B,T,HV,K] fp32, chunk-local cumsum / ln2 (only last rows used)
    h0: torch.Tensor,  # [B,HV,K,V] fp32
    scale: float,
    aqk_prezeroed: bool = False,  # 004 hook: its intra kernel emits zero-padded Aqk
) -> tuple[torch.Tensor, torch.Tensor]:
    if not aqk_prezeroed:
        _zero_aqk_upper(aqk)
    key = _call_key(qg, kg, w, u, aqk, g2, h0, scale)
    ent = _CALL_CACHE.get(key)
    outs = None  # set on the miss path, where the outputs were allocated to build the views
    if ent is None:
        B, T, HV, K = qg.shape
        V = u.shape[3]
        BT = aqk.shape[3]
        assert BT == 64, "cute path implements chunk_size=64"
        assert T % 64 == 0, "T must be a multiple of the chunk size"
        assert V % 64 == 0

        o = torch.empty(B, T, HV, V, device=qg.device, dtype=qg.dtype)
        ht = torch.empty(B, HV, K, V, device=qg.device, dtype=torch.float32)
        # Per-chunk decay vectors, [B,HV,NT,K] with K contiguous so the cute view
        # (K,NT,HV,B) has a static stride-1 mode 0 — dynamic modes must not include the
        # innermost one. A persistent buffer: refilled on every call below.
        gdc = torch.empty(B, HV, T // 64, K, device=g2.device, dtype=g2.dtype)

        io_dtype = cutlass.BFloat16 if qg.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V)  # V is a static layout mode

        # logical (M/N, K, rest) views — see module docstring
        cqg = _cute_view(qg, (1, 3, 2, 0), (0, 2, 3))
        ckgt = _cute_view(kg, (3, 1, 2, 0), (1, 2, 3))
        cw = _cute_view(w, (1, 3, 2, 0), (0, 2, 3))
        cu = _cute_view(u, (1, 3, 2, 0), (0, 2, 3))
        caqk = _cute_view(aqk, (1, 3, 2, 0), (0, 2, 3))
        cgd = _cute_view(gdc, (3, 2, 1, 0), (1, 2, 3))
        ch0 = _cute_view(h0, (2, 3, 1, 0), (2, 3))
        co = _cute_view(o, (1, 3, 2, 0), (0, 2, 3))
        cht = _cute_view(ht, (2, 3, 1, 0), (2, 3))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = KdaFwdStateKernel(io_dtype, K, V_TILE=64)
            compiled = cute.compile(
                kernel_obj, cqg, cw, ckgt, cu, caqk, cgd, ch0, co, cht,
                cutlass.Float32(scale), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        _release_keepalives(cqg, cw, ckgt, cu, caqk, cgd, ch0, co, cht)
        args = (cqg, cw, ckgt, cu, caqk, cgd, ch0, co, cht, cutlass.Float32(scale), stream)
        if len(_CALL_CACHE) >= 64:  # distinct layouts are few; this is a leak backstop
            _CALL_CACHE.clear()
        outs = (o, ht)
        ent = (compiled, args, _out_specs(o, ht), gdc)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, gdc = ent
    cqg, cw, ckgt, cu, caqk, _, ch0, co, cht, _, _ = args
    if outs is None:
        outs = _alloc_outs(out_specs, qg.device)
    o, ht = outs
    _retarget(co, o)
    _retarget(cht, ht)
    _retarget(cqg, qg)
    _retarget(cw, w)
    _retarget(ckgt, kg)
    _retarget(cu, u)
    _retarget(caqk, aqk)
    _retarget(ch0, h0)
    B, HV, NT, K = gdc.shape
    # refill the decay staging in place: g2's last row per chunk, [B,T,HV,K] -> [B,HV,NT,K]
    gdc.copy_(g2[:, 63::64].transpose(1, 2))
    compiled(*args)
    return o, ht
