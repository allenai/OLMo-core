"""Phase 2 B1 — the CuTe forward re-scan fused with dq's h consumer. See ALGORITHM.md.

Replaces stage 2 (fla's `chunk_gated_delta_rule_fwd_h` re-run) and the dq half of stage 5.
kernel_fwd.py's scan skeleton (per-dim decay: gd K-vector broadcast, kg carries the v~
scaling so v' stores un-decayed — fla's USE_GK contract) crossed with gdn 003
kernel_fwdh.py's store machinery (h checkpoint + v_new staging, the record-006 store-wait
deferrals). Per (b, hv, v_tile) CTA, h [K, BV] fp32 resident in SIMT registers, per chunk:

    hck[c] = h                      (bf16 -> HBM pre-update, via COL_MAJOR trans staging)
    WH  = w @ h_c                   (MMA, h from smem bf16)
    DQ  = do @ h_c^T                (MMA — dq's ONLY h dependence, stolen from wy_dqkg;
                                     h's smem bytes double as the mn-major B operand)
    v'  = u - WH                    (SIMT) -> v_new HBM + DH's A operand
    DH  = kg^T @ v'                 (MMA)
    h   = exp2(gd) * h + DH         (SIMT regs, per-dim K-vector decay)

DQ is RAW (no gate, no scale): the wy_dqkg variant (kernel_wy.py) applies scale*exp2(g2)
where it already loads g2. The V-tile reduction crosses CTAs, so dq is zero-initialized
host-side and the epilogue warpgroup (8..11, the warps that assembled o in the forward)
t2r's DQ and issues cp.reduce.async.bulk.tensor.add (TMA reduce) from a fp32 staging
buffer — fp32 adds in L2, no atomic loop, off the recurrence critical path.

Logical gmem mode order (M/N, K_contract, rest) per operand:
    w:      (T, K, HV, B)      A of WH
    do:     (T, V, HV, B)      A of DQ (same view as u)
    kg^T:   (K, T, HV, B)      mn-major B of DH (same storage as kg)
    u:      (T, V, HV, B)      SIMT-only, linear smem
    gd:     (K, NT, HV, B)     fp32 per-chunk decay vectors (g2 last rows), K contiguous
    h0:     (K, V, HV, B)      fp32 scalar gather, once per CTA
    v_new:  (T, V, HV, B)      TMA store
    hck:    (V, K, NT, HV, B)  TMA store (bf16); smem operand bytes are K-contiguous and
                               TMA cannot transpose -> COL_MAJOR staging (gdn 003's fix)
    dq:     (T, K, HV, B)      fp32 TMA reduce-add
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


class KdaB1ScanKernel:
    """Fused re-scan + dq. One CTA per (b, hv, v_tile); serial over chunks.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT recurrence (v', h update, h/v_new
    stores), 8..11 = dq epilogue (DQ readout + TMA reduce-add). Warps 2,3 idle through
    the role branch and only participate in alloc/dealloc barriers.
    """

    def __init__(self, io_dtype: Type[cutlass.Numeric], K: int, V_TILE: int,
                 enable_dq: bool = True):
        # enable_dq=False compiles the pure rescan (no do load, no DQ MMA, no reduce)
        # — the attribution knob that splits the dq fusion's cost into traffic vs
        # pipeline stall (KDA002_B1=nodq at the dispatcher).
        self.enable_dq = enable_dq
        self.io_dtype = io_dtype
        self.acc_dtype = cutlass.Float32
        self.BT = 64
        self.K = K
        self.BV = V_TILE

        assert K in (64, 128), "K must be 64 or 128"
        assert self.BV == 64, "V tile is 64"

        # MMA tile shapes (M, N, K_contract)
        self.tile_wh = (self.BT, self.BV, self.K)  # w @ h
        self.tile_dq = (self.BT, self.K, self.BV)  # do @ h^T
        self.tile_dh = (self.BV, self.K, self.BT)  # DH^T = v'^T @ kg

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0
        self.mma_warp_id = 1
        self.simt_warp_id = (4, 5, 6, 7)
        self.epi_warp_id = (8, 9, 10, 11)
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
        # DQ: A = do [BT, BV] k-major; B = h as [K, BV] mn-major — the SAME smem bytes as
        # WH's [BV, K] k-major B operand (both are "K-index fastest"), checked below.
        mma_dq = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dq[:2], tcgen05.OperandSource.SMEM,
        )
        mma_dh = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dh[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_wh, mma_dq, mma_dh

    def _setup_attributes(self):
        mma_wh, mma_dq, mma_dh = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.w_smem_layout = sm100_utils.make_smem_layout_a(
            mma_wh, self.tile_wh, self.io_dtype, self.input_stages
        )
        self.do_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dq, self.tile_dq, self.io_dtype, self.input_stages
        )
        self.kgt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dh, self.tile_dh, self.io_dtype, self.input_stages
        )
        # u is SIMT-only: plain row-major (BT, BV), BV contiguous
        self.u_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        # gd: the per-chunk decay K-vector (g2's last row), fp32
        self.gd_smem_layout = cute.make_layout((K, self.input_stages))

        # h as B operand of WH ([BV, K] k-major, staged); written by SIMT through the
        # ROW_MAJOR epi view; ALSO read by the DQ MMA through the mn-major [K, BV] view
        # of the same bytes.
        self.h_smem_layout = sm100_utils.make_smem_layout_b(
            mma_wh, self.tile_wh, self.io_dtype, self.h_stages
        )
        self.hdq_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dq, self.tile_dq, self.io_dtype, self.h_stages
        )
        self.h_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BV, K), self.h_stages
        )
        # h checkpoint staging: TMA cannot transpose; the V-contiguous gmem slice gets a
        # COL_MAJOR (BV, K) buffer filled by a transpose stmatrix (gdn 003 record 004).
        self.cks_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BV, K), 1
        )
        # v' as A operand of DH ([BV, BT] k-major); written via transpose stmatrix
        # through the COL_MAJOR (BT, BV) epi view of the same bytes.
        self.vpd_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dh, self.tile_dh, self.io_dtype, 1
        )
        self.vp_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BV), 1
        )
        # v_new store staging: (BT, BV) BV-contiguous to match gmem
        self.vns_smem_layout = cute.make_layout((BT, BV, 1), stride=(BV, 1, BT * BV))
        # dq reduce staging: (BT, K) fp32, K contiguous to match gmem
        self.dqs_smem_layout = cute.make_layout((BT, K, 1), stride=(K, 1, BT * K))

        self.num_mma_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.w_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.kgt_smem_layout, (None, None, None, 0))
            )
        )
        if self.enable_dq:
            self.num_mma_load_bytes += cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.do_smem_layout, (None, None, None, 0))
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
            self.tmem_dq_offset,
            self.tmem_dh_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_wh, mma_dq, mma_dh)

    def _plan_tmem(self, mma_wh, mma_dq, mma_dh):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        wh = acc_cols(mma_wh, self.tile_wh)
        dq = acc_cols(mma_dq, self.tile_dq)
        dh = acc_cols(mma_dh, self.tile_dh)
        off_wh = 0
        off_dq = off_wh + wh
        off_dh = off_dq + dq
        total_ = off_dh + dh
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_wh, off_dq, off_dh, total

    def _check_h_alias(self):
        """The DQ MMA reads h through hdq_smem_layout but the bytes are written once,
        through h_smem_layout's epi view. (BV, K) k-major and (K, BV) mn-major are both
        "K-index fastest" 256B-row swizzled layouts, so the physical mapping should
        coincide — this checks the cheap invariants (swizzle atom, footprint); the
        authoritative check is numerical: dbg_bwd's forced-cute lane, where a wrong
        alias scrambles dq by O(1) (the atom-folded coordinate profiles make an
        element-wise crd2idx comparison impossible at trace time)."""
        lh = self.h_smem_layout  # ((BV, K) k-major, stages) composed w/ swizzle
        lq = self.hdq_smem_layout  # ((K, BV) mn-major, stages)
        if str(lh.inner) != str(lq.inner) or cute.cosize(lh) != cute.cosize(lq):
            raise ValueError(
                f"h alias mismatch: {lh} vs {lq} — give DQ its own staging buffer "
                "(see ALGORITHM.md B1 design)"
            )

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        w: cute.Tensor,  # (T, K, HV, B)
        kgt: cute.Tensor,  # (K, T, HV, B) — same storage as kg
        do: cute.Tensor,  # (T, V, HV, B)
        u: cute.Tensor,  # (T, V, HV, B)
        gd: cute.Tensor,  # (K, NT, HV, B) fp32
        h0: cute.Tensor,  # (K, V, HV, B) fp32
        vnew: cute.Tensor,  # (T, V, HV, B) out
        hck: cute.Tensor,  # (V, K, NT, HV, B) out, io dtype
        dq: cute.Tensor,  # (T, K, HV, B) fp32 out — ZEROED by the caller, reduce-add
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        self._check_h_alias()
        mma_wh, mma_dq, mma_dh = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), w,
            cute.slice_(self.w_smem_layout, (None, None, None, 0)),
            self.tile_wh, mma_wh, cluster_vmnk,
        )
        tma_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), do,
            cute.slice_(self.do_smem_layout, (None, None, None, 0)),
            self.tile_dq, mma_dq, cluster_vmnk,
        )
        tma_kgt, tma_tensor_kgt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), kgt,
            cute.slice_(self.kgt_smem_layout, (None, None, None, 0)),
            self.tile_dh, mma_dh, cluster_vmnk,
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
        tma_vn, tma_tensor_vn = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), vnew,
            cute.slice_(self.vns_smem_layout, (None, None, 0)),
            (BT, BV),
        )
        tma_ck, tma_tensor_ck = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), hck,
            cute.slice_(self.cks_smem_layout, (None, None, 0)),
            (BV, K),
        )
        tma_dq, tma_tensor_dq = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(), dq,
            cute.slice_(self.dqs_smem_layout, (None, None, 0)),
            (BT, K),
        )

        B = cute.size(w, mode=[3])
        HV = cute.size(w, mode=[2])
        NV = cute.size(u, mode=[1]) // BV
        grid = (B * HV * NV, 1, 1)

        swz_align, lin_align = 1024, 128

        # Every `*_full` range backs both halves of a pipeline's mbarrier array —
        # 2 * num_stages Int64s. Under-sizing aliases the next pipeline and only
        # deadlocks once the pipe wraps.
        @cute.struct
        class SharedStorage:
            mmain_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            simtin_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            h_full: cute.struct.MemRange[cutlass.Int64, self.h_stages * 2]  # type: ignore
            vpd_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            wh_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dqf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dh_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_w: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.w_smem_layout)], swz_align  # type: ignore
            ]
            smem_do: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.do_smem_layout)], swz_align  # type: ignore
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
            smem_vpd: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.vpd_smem_layout)], swz_align  # type: ignore
            ]
            smem_vns: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.vns_smem_layout)], lin_align  # type: ignore
            ]
            smem_dqs: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.dqs_smem_layout)], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
            )

        self.kda_cute_b1(
            tma_w, tma_tensor_w,
            tma_do, tma_tensor_do,
            tma_kgt, tma_tensor_kgt,
            tma_u, tma_tensor_u,
            tma_gd, tma_tensor_gd,
            tma_vn, tma_tensor_vn,
            tma_ck, tma_tensor_ck,
            tma_dq, tma_tensor_dq,
            h0,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def kda_cute_b1(
        self,
        tma_w: cute.CopyAtom, mW: cute.Tensor,
        tma_do: cute.CopyAtom, mDO: cute.Tensor,
        tma_kgt: cute.CopyAtom, mKGT: cute.Tensor,
        tma_u: cute.CopyAtom, mU: cute.Tensor,
        tma_gd: cute.CopyAtom, mGd: cute.Tensor,
        tma_vn: cute.CopyAtom, mVN: cute.Tensor,
        tma_ck: cute.CopyAtom, mCK: cute.Tensor,
        tma_dq: cute.CopyAtom, mDQ: cute.Tensor,
        mH0: cute.Tensor,
    ):
        BT, K, BV = self.BT, self.K, self.BV
        io = self.io_dtype
        f32 = self.acc_dtype
        # Layouts/TiledMma from the host trace cannot cross the region boundary — rebuild.
        self._setup_attributes()
        mma_wh, mma_dq, mma_dh = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_w, tma_do, tma_kgt, tma_u, tma_gd, tma_vn, tma_ck, tma_dq]:
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
        sDO = storage.smem_do.get_tensor(self.do_smem_layout.outer, swizzle=self.do_smem_layout.inner)
        sKGT = storage.smem_kgt.get_tensor(self.kgt_smem_layout.outer, swizzle=self.kgt_smem_layout.inner)
        sU = storage.smem_u.get_tensor(self.u_smem_layout)
        sGd = storage.smem_gd.get_tensor(self.gd_smem_layout)
        sH = storage.smem_h.get_tensor(self.h_smem_layout.outer, swizzle=self.h_smem_layout.inner)
        sHdq = storage.smem_h.get_tensor(self.hdq_smem_layout.outer, swizzle=self.hdq_smem_layout.inner)
        sH_epi = storage.smem_h.get_tensor(self.h_epi_layout.outer, swizzle=self.h_epi_layout.inner)
        sCKS = storage.smem_cks.get_tensor(self.cks_smem_layout.outer, swizzle=self.cks_smem_layout.inner)
        sVpd = storage.smem_vpd.get_tensor(self.vpd_smem_layout.outer, swizzle=self.vpd_smem_layout.inner)
        sVpd_epi = storage.smem_vpd.get_tensor(self.vp_epi_layout.outer, swizzle=self.vp_epi_layout.inner)
        sVNS = storage.smem_vns.get_tensor(self.vns_smem_layout)
        sDQS = storage.smem_dqs.get_tensor(self.dqs_smem_layout)

        # ---- pipelines ----
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
        # Not PipelineTmaAsync — see kernel_fwd.py (consumer_release arrives from lane 0
        # only vs the full-count empty barrier; producer_tail deadlocks).
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
        vpd_pipe = make_simt_to_mma_pipe(storage.vpd_full.data_ptr(), 1, simt_threads)
        wh_pipe = make_mma_to_simt_pipe(storage.wh_full.data_ptr(), simt_threads)
        dqf_pipe = make_mma_to_simt_pipe(storage.dqf_full.data_ptr(), epi_threads)
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
        tDQ = acc_tensor(mma_dq, self.tile_dq, self.tmem_dq_offset)
        tDH = acc_tensor(mma_dh, self.tile_dh, self.tmem_dh_offset)

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gW = cute.local_tile(mW, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)
        gDO = cute.local_tile(mDO, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gKGT = cute.local_tile(mKGT, (K, BT), (0, None, hv_idx, b_idx))  # (K,BT,NT)
        gU = cute.local_tile(mU, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gGd = mGd[(None, None, hv_idx, b_idx)]  # (K, NT)
        gVN = cute.local_tile(mVN, (BT, BV), (None, v_idx, hv_idx, b_idx))
        gCK = cute.local_tile(mCK, (BV, K), (v_idx, 0, None, hv_idx, b_idx))  # (BV,K,NT)
        gDQ = cute.local_tile(mDQ, (BT, K), (None, 0, hv_idx, b_idx))  # (BT,K,NT)

        # ==========================================================================
        # TMA warp
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_wh = mma_wh.get_slice(0)
            thr_mma_dq = mma_dq.get_slice(0)
            thr_mma_dh = mma_dh.get_slice(0)

            tW_mma = thr_mma_wh.partition_A(gW)
            tDO_mma = thr_mma_dq.partition_A(gDO)
            tKGT_mma = thr_mma_dh.partition_B(gKGT)

            cta1 = cute.make_layout(1)
            tWs, tWg = cpasync.tma_partition(
                tma_w, 0, cta1, cute.group_modes(sW, 0, 3), cute.group_modes(tW_mma, 0, 3)
            )
            tDOs, tDOg = cpasync.tma_partition(
                tma_do, 0, cta1, cute.group_modes(sDO, 0, 3), cute.group_modes(tDO_mma, 0, 3)
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

            mmain_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            simtin_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )

            for c in cutlass.range(NT, unroll=1):
                mmain_pipe.producer_acquire(mmain_producer)
                bar = mmain_pipe.producer_get_barrier(mmain_producer)
                cute.copy(tma_w, tWg[None, c], tWs[None, mmain_producer.index], tma_bar_ptr=bar)
                if cutlass.const_expr(self.enable_dq):
                    cute.copy(tma_do, tDOg[None, c], tDOs[None, mmain_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_kgt, tKGTg[None, c], tKGTs[None, mmain_producer.index], tma_bar_ptr=bar)
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
            tCrW = mma_wh.make_fragment_A(sW)
            tCrH = mma_wh.make_fragment_B(sH)
            tCrDO = mma_dq.make_fragment_A(sDO)
            tCrHdq = mma_dq.make_fragment_B(sHdq)
            tCrKGT = mma_dh.make_fragment_B(sKGT)
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
            vpd_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            wh_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dqf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
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

                # DQ = do h^T — h's second (and last) consumer this chunk.
                if cutlass.const_expr(self.enable_dq):
                    dqf_pipe.producer_acquire(dqf_producer)
                    for kk in cutlass.range(cute.size(tCrDO, mode=[2]), unroll_full=True):
                        mma_dq.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                        cute.gemm(
                            mma_dq, tDQ[None, None, None, 0],
                            tCrDO[None, None, kk, mmain_consumer.index],
                            tCrHdq[None, None, kk, h_consumer.index],
                            tDQ[None, None, None, 0],
                        )
                    dqf_pipe.producer_commit(dqf_producer)
                    dqf_producer.advance()
                h_pipe.consumer_release(h_consumer)
                h_consumer.advance()

                # DH = kg^T v'
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
                # kgt was this stage's last user
                mmain_pipe.consumer_release(mmain_consumer)
                mmain_consumer.advance()
                # umma half of simtin's empty arrive (data untouched by this warp)
                simtin_pipe.consumer_wait(simtin_mma_consumer)
                simtin_pipe.consumer_release(
                    simtin_mma_consumer, pipeline.PipelineOp.TCGen05Mma
                )
                simtin_mma_consumer.advance()

            wh_pipe.producer_tail(wh_producer)
            dqf_pipe.producer_tail(dqf_producer)
            dh_pipe.producer_tail(dh_producer)

        # ==========================================================================
        # SIMT recurrence warps 4..7: v' from WH, the h update from DH, h/v_new stores.
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
            tWHrU = cute.make_rmem_tensor(
                cute.slice_(tWHsU.shape, (None, None, None, 0)), io
            )
            r2s_x16t_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_vpd = cute.make_tiled_copy_D(r2s_x16t_atom, tiled_t2r_wh)
            thr_r2s_vpd = tiled_r2s_vpd.get_slice(local_tidx)
            tRS_sVpd = thr_r2s_vpd.partition_D(sVpd_epi)
            tRS_rVpd = cute.make_rmem_tensor(
                cute.slice_(tRS_sVpd.shape, (None, None, None, 0)), io
            )
            r2s_x16_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_vns = cute.make_tiled_copy_D(r2s_x16_atom, tiled_t2r_wh)
            thr_r2s_vns = tiled_r2s_vns.get_slice(local_tidx)
            tRS_sVNS = thr_r2s_vns.partition_D(sVNS)
            tRS_rVNS = cute.make_rmem_tensor(
                cute.slice_(tRS_sVNS.shape, (None, None, None, 0)), io
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
            # per-dim decay: broadcast the K-vector across the BV mode of the fragment
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
            # checkpoint staging: same values, transpose stmatrix into the BV-contiguous
            # buffer the TMA store can express (gdn 003: TMA stores cannot transpose).
            r2s_ck_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s_ck = cute.make_tiled_copy_D(r2s_ck_atom, tiled_t2r_dh)
            thr_r2s_ck = tiled_r2s_ck.get_slice(local_tidx)
            tRS_sCKS = thr_r2s_ck.partition_D(sCKS)
            tRS_rCKS = cute.make_rmem_tensor(
                cute.slice_(tRS_sCKS.shape, (None, None, None, 0)), io
            )

            # TMA store plumbing (v_new per chunk; h checkpoint per chunk)
            bSG_sVNS, bSG_gVN = cpasync.tma_partition(
                tma_vn, 0, cute.make_layout(1),
                cute.group_modes(sVNS, 0, 2), cute.group_modes(gVN, 0, 2),
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
            wh_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dh_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            h_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.h_stages
            )
            vpd_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # ---- h := h0; checkpoint 0 is exactly that value ----
            for i in cutlass.range(cute.size(tHreg), unroll_full=True):
                vv, kk = coordDH[i]
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
            # commit h BEFORE the checkpoint store: store completion stays off the
            # recurrence critical path (gdn 003 record 006's deferral).
            h_pipe.producer_commit(h_producer)
            h_producer.advance()
            if warp_idx == self.simt_warp_id[0]:
                cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, 0])
                tma_store_pipeline.producer_commit()

            for c in cutlass.range(NT, unroll=1):
                simtin_pipe.consumer_wait(simtin_consumer)
                scrd = (None, None, None, simtin_consumer.index)
                cute.copy(io_cp_atom, tWHsU[scrd], tWHrU)

                # v' = u - WH (un-decayed: kg carries the exp2(G - g2) factor)
                wh_pipe.consumer_wait(wh_consumer)
                cute.copy(tiled_t2r_wh, tTR_tWH[None, None, None, 0], tTR_rWH)
                cute.arch.fence_view_async_tmem_load()
                wh_pipe.consumer_release(wh_consumer)
                wh_consumer.advance()
                vpd_pipe.producer_acquire(vpd_producer)
                for i in cutlass.range(
                    cute.size(tTR_rWH), unroll_full=True, vectorize=True
                ):
                    vp = (tWHrU[i].to(f32) - tTR_rWH[i]).to(io)
                    tRS_rVpd[i] = vp
                    tRS_rVNS[i] = vp
                # the MMA operand first: it unblocks DH
                cute.copy(tiled_r2s_vpd, tRS_rVpd, tRS_sVpd[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                vpd_pipe.producer_commit(vpd_producer)
                vpd_producer.advance()
                # acquire FIRST: waits on last chunk's stores (long complete), not the
                # one about to be issued — sVNS/sCKS reuse stays safe via the barrier.
                if warp_idx == self.simt_warp_id[0]:
                    tma_store_pipeline.producer_acquire()
                self.simt_sync_barrier.arrive_and_wait()
                cute.copy(tiled_r2s_vns, tRS_rVNS, tRS_sVNS[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                if warp_idx == self.simt_warp_id[0]:
                    cute.copy(tma_vn, bSG_sVNS[None, 0], bSG_gVN[None, c])
                    tma_store_pipeline.producer_commit()

                # h update: per-dim decay — each fragment element's k picks its factor
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
                        vio = tHreg[i].to(io)
                        tRS_rH[i] = vio
                        tRS_rCKS[i] = vio
                    cute.copy(
                        tiled_r2s_h, tRS_rH, tRS_sH[None, None, None, h_producer.index]
                    )
                    cute.copy(tiled_r2s_ck, tRS_rCKS, tRS_sCKS[None, None, None, 0])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.simt_sync_barrier.arrive_and_wait()
                    # h first — the checkpoint store is off the critical path
                    h_pipe.producer_commit(h_producer)
                    h_producer.advance()
                    if warp_idx == self.simt_warp_id[0]:
                        cute.copy(tma_ck, bSG_sCK[None, 0], bSG_gCK[None, c + 1])
                        tma_store_pipeline.producer_commit()

                simtin_pipe.consumer_release(
                    simtin_consumer, pipeline.PipelineOp.AsyncThread
                )
                simtin_consumer.advance()

            tma_store_pipeline.producer_tail()

        # ==========================================================================
        # SIMT epilogue warps 8..11: DQ readout -> fp32 staging -> TMA reduce-add.
        # Off the critical path — trails the recurrence without stalling it.
        # ==========================================================================
        elif (
            warp_idx == self.epi_warp_id[0]
            or warp_idx == self.epi_warp_id[1]
            or warp_idx == self.epi_warp_id[2]
            or warp_idx == self.epi_warp_id[3]
        ):
            t2r_dq_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), f32
            )
            f32_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), f32)

            tDQ_2d = tDQ[((None, None), 0, 0, None)]
            tiled_t2r_dq = tcgen05.make_tmem_copy(t2r_dq_atom, tDQ_2d[None, None, 0])
            thr_t2r_dq = tiled_t2r_dq.get_slice(local_tidx)
            tTR_tDQ = thr_t2r_dq.partition_S(tDQ_2d)
            tTR_rDQ = cute.make_rmem_tensor(
                thr_t2r_dq.partition_D(cute.make_identity_tensor((BT, K))).shape, f32
            )
            tDQsS = thr_t2r_dq.partition_D(sDQS)

            bSG_sDQS, bSG_gDQ = cpasync.tma_partition(
                tma_dq, 0, cute.make_layout(1),
                cute.group_modes(sDQS, 0, 2), cute.group_modes(gDQ, 0, 2),
            )
            dq_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, epi_threads
                ),
            )

            dqf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            for c in cutlass.range(NT if cutlass.const_expr(self.enable_dq) else 0, unroll=1):
                dqf_pipe.consumer_wait(dqf_consumer)
                cute.copy(tiled_t2r_dq, tTR_tDQ[None, None, None, 0], tTR_rDQ)
                cute.arch.fence_view_async_tmem_load()
                dqf_pipe.consumer_release(dqf_consumer)
                dqf_consumer.advance()
                # acquire FIRST (deferral): waits on chunk c-1's reduce, protecting the
                # sDQS overwrite below; the fresh reduce's completion never blocks us.
                if warp_idx == self.epi_warp_id[0]:
                    dq_store_pipeline.producer_acquire()
                self.epi_sync_barrier.arrive_and_wait()
                cute.copy(f32_cp_atom, tTR_rDQ, tDQsS[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epi_sync_barrier.arrive_and_wait()
                if warp_idx == self.epi_warp_id[0]:
                    cute.copy(tma_dq, bSG_sDQS[None, 0], bSG_gDQ[None, c])
                    dq_store_pipeline.producer_commit()

            dq_store_pipeline.producer_tail()

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


def _dq_buffer(shape, device, enable_dq: bool) -> torch.Tensor:
    """dq's [B,T,HV,K] fp32 buffer — or a 16-byte stand-in when the fusion is off.

    With enable_dq=False (the default at prod — the dq fusion measured negative) the kernel
    is COMPILED without its dq epilogue: it never reads or writes this tensor, and the
    wrapper returns None instead of it. Only the descriptor is needed, so allocating the
    real thing costs a gigabyte of allocator churn per backward at prod8192 (and pins the
    same gigabyte in the call cache) for a buffer nothing touches. as_strided over a short
    base is deliberately out of bounds — that is safe here and ONLY here, because no code
    path dereferences it; if a torch version ever refuses, fall back to the allocation.
    """
    if enable_dq:
        # reduce-add accumulated: MUST start zeroed on every call.
        return torch.zeros(shape, device=device, dtype=torch.float32)
    strides, acc = [], 1
    for d in reversed(tuple(shape)):
        strides.append(acc)
        acc *= int(d)
    try:
        base = torch.empty(4, device=device, dtype=torch.float32)
        return torch.as_strided(base, tuple(shape), tuple(reversed(strides)))
    except RuntimeError:
        return torch.empty(shape, device=device, dtype=torch.float32)


def _call_key(kg, w, u, g2, h0, do, enable_dq):
    def sig(t):
        return (t.shape, t.stride(), t.dtype)

    return (sig(kg), sig(w), sig(u), sig(g2), sig(h0), sig(do), enable_dq,
            torch.cuda.current_stream().cuda_stream)


def kda_cute_b1_call(
    kg: torch.Tensor,  # [B,T,HV,K] bf16/fp16 — k * exp2(G - g2), from recompute
    w: torch.Tensor,  # [B,T,HV,K]
    u: torch.Tensor,  # [B,T,HV,V]
    g2: torch.Tensor,  # [B,T,HV,K] fp32, chunk-local cumsum / ln2 (only last rows used)
    h0: torch.Tensor,  # [B,HV,K,V] fp32
    do: torch.Tensor,  # [B,T,HV,V]
    enable_dq: bool = True,
):
    """Returns (h checkpoints [B,NT,HV,K,V] bf16, v_new [B,T,HV,V], dq_raw [B,T,HV,K] fp32).

    dq_raw is Σ_v do@h^T with NO gate/scale — kernel_wy.py applies scale*exp2(g2).
    enable_dq=False compiles/runs the pure rescan and returns dq_raw=None."""
    key = _call_key(kg, w, u, g2, h0, do, enable_dq)
    ent = _CALL_CACHE.get(key)
    outs = None
    if ent is None:
        B, T, HV, K = kg.shape
        V = u.shape[3]
        NT = T // 64
        assert T % 64 == 0, "T must be a multiple of the chunk size"
        assert V % 64 == 0

        h = torch.empty(B, NT, HV, K, V, device=kg.device, dtype=kg.dtype)
        v_new = torch.empty(B, T, HV, V, device=kg.device, dtype=u.dtype)
        dq = _dq_buffer((B, T, HV, K), kg.device, enable_dq)
        # Per-chunk decay vectors, K contiguous (persistent scratch, refilled per call)
        gdc = torch.empty(B, HV, NT, K, device=g2.device, dtype=g2.dtype)

        io_dtype = cutlass.BFloat16 if kg.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V, enable_dq)

        cw = _cute_view(w, (1, 3, 2, 0), (0, 2, 3))
        ckgt = _cute_view(kg, (3, 1, 2, 0), (1, 2, 3))
        cdo = _cute_view(do, (1, 3, 2, 0), (0, 2, 3))
        cu = _cute_view(u, (1, 3, 2, 0), (0, 2, 3))
        cgd = _cute_view(gdc, (3, 2, 1, 0), (1, 2, 3))
        ch0 = _cute_view(h0, (2, 3, 1, 0), (2, 3))
        cvn = _cute_view(v_new, (1, 3, 2, 0), (0, 2, 3))
        chck = _cute_view(h, (4, 3, 1, 2, 0), (2, 3, 4))
        cdq = _cute_view(dq, (1, 3, 2, 0), (0, 2, 3))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = KdaB1ScanKernel(io_dtype, K, V_TILE=64, enable_dq=enable_dq)
            compiled = cute.compile(
                kernel_obj, cw, ckgt, cdo, cu, cgd, ch0, cvn, chck, cdq, stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        # See kernel_fwd._release_keepalives: without this the entry pins the first call's
        # w/kg/do/u/h0 and its outputs (h alone is 2 GiB at prod8192) forever.
        _release_keepalives(cw, ckgt, cdo, cu, cgd, ch0, cvn, chck, cdq)
        args = (cw, ckgt, cdo, cu, cgd, ch0, cvn, chck, cdq, stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        outs = (h, v_new, dq)
        out_specs = tuple((tuple(t.shape), t.dtype) for t in outs)
        ent = (compiled, args, out_specs, gdc)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, gdc = ent
    cw, ckgt, cdo, cu, _, ch0, cvn, chck, cdq, _ = args
    if outs is None:
        (hs, hd), (vs, vd), (qs, qd) = out_specs
        h = torch.empty(hs, device=kg.device, dtype=hd)
        v_new = torch.empty(vs, device=kg.device, dtype=vd)
        dq = _dq_buffer(qs, kg.device, enable_dq)
        outs = (h, v_new, dq)
    h, v_new, dq = outs
    _retarget(chck, h)
    _retarget(cvn, v_new)
    _retarget(cdq, dq)
    _retarget(cw, w)
    _retarget(ckgt, kg)
    _retarget(cdo, do)
    _retarget(cu, u)
    _retarget(ch0, h0)
    # refill the decay staging in place: g2's last row per chunk
    gdc.copy_(g2[:, 63::64].transpose(1, 2))
    compiled(*args)
    return h, v_new, (dq if enable_dq else None)


# Minimum grid size (CTAs) for the cute path — serial scans need a full GPU (gdn 003
# record 005: the dhu kernel lost 1.4ms to fla at a 64-CTA grid). KDA002_B1=cutedsl
# forces past it so dbg-sized shapes exercise the cute kernel (gdn record 006's
# process bug: without the override, every dbg case compared fla vs fla).
_MIN_CTAS = 256


def kda_rescan_b1(kg, w, u, g2, h0, do, chunk_size):
    """Stage-2 dispatcher. Returns (h, v_new, dq_raw | None); dq_raw None means the
    fla fallback ran and wy_dqkg must compute dq itself."""
    B, T, HV, K = kg.shape
    V = u.shape[-1]
    supported = (
        chunk_size == 64
        and T % 64 == 0
        and K in (64, 128)
        and V % 64 == 0
        and kg.dtype in (torch.bfloat16, torch.float16)
        and B * HV * (V // 64) >= _MIN_CTAS
    )
    if not supported:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g2, initial_state=h0,
            output_final_state=False, chunk_size=chunk_size,
        )
        return h, v_new, None
    # Default is the pure rescan: the dq fusion measured NEGATIVE at prod8192
    # (2026-08-18 dbg_b1perf: rescan-only 1.37ms vs fla 1.88 = +0.51ms; the dq add-on
    # costs 0.75ms in-kernel at its own traffic floor — 4.3GB fp32 cross-CTA reduce +
    # do load — plus 0.15ms zeros, while the no-dq wy variant only returned 0.12ms:
    # fla's wy had both dq operands loaded anyway and is latency-bound, not dot-bound).
    # KDA002_B1=cutedsl keeps the fusion compilable for B2 experiments, where the
    # reduce-add is the pattern every fused V-reduced output (dk/dw) will need.
    # enable_dq=False: the dq fusion (B1 computing do@h^T for the wy stage) measured
    # negative and wy_dqkg computes dq itself. The kernel compiles without its dq epilogue.
    return kda_cute_b1_call(kg, w, u, g2, h0, do, enable_dq=False)
