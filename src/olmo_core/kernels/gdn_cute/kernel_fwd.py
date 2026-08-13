"""The CuTe DSL forward for the chunked gated delta rule. See NOTES.md / ALGORITHM.md.

003 does not change the forward: this file is a verbatim copy of 002-cute-pipeline's
kernel.py, taken at the commit that recorded 1.61x on prod8192 fwd. It is copied rather than
imported so that this idea's history snapshots contain the code that actually ran — the whole
point of the ledger. If 002's forward improves later, re-copy it and say so in a History line;
do not let the two drift silently.

002 scope: 001's fused fwd_h + fwd_o kernel restructured for overlap. The per-chunk critical
path is only h -> WH -> v'/v~' -> DH -> h update; everything else (S1, Aqk, OH, OI, the o
epilogue) is off-path. So: a second SIMT warpgroup (warps 8..11) owns the off-path work
(Aqk, o) while warps 4..7 keep the recurrence; Aqk is double-buffered in tmem so the two
groups decouple; and the MMA warp issues next chunk's S1 as soon as its inputs land instead
of at the top of the next iteration. The WY stage (gate cumsum, kkt+solve_tril, w/u) is
still fla's Triton kernels; the `A` tensor is kept because fla's recompute-based backward
consumes exactly it (see __init__.py's autograd wrapper).

Per (b, hv, v-tile) CTA, with h [K, BV] fp32 resident in SIMT-warp registers, per chunk c:

    S1 = q @ k^T                       (MMA, [BT,BT], indep of h)
    Aqk[i,j] = exp2(g2_i - g2_j) * S1[i,j] for i>=j else 0     (SIMT -> tmem bf16)
    WH = w @ h_c                       (MMA, h from smem bf16)
    v'  = u - WH                       (SIMT)         -> smem bf16 (readout operand)
    v~' = v' * exp2(G - g2_i)          (SIMT)         -> smem bf16 (state operand)
    OH = q @ h_c                       (MMA)
    OI = Aqk @ v'                      (MMA, A from tmem)
    DH = k^T @ v~'                     (MMA, k as mn-major A)
    h_{c+1} = exp2(G) * h_c + DH       (SIMT regs, fp32) -> smem bf16 for chunk c+1
    o_c = scale * (exp2(g2_i) * OH + OI)   (SIMT epilogue -> TMA store)

after the last chunk, ht = h fp32 -> TMA store. G = g2[last row of chunk]; all decays are
log2-space (g2 = chunk-local inclusive cumsum(g) / ln2). Precision matches fla's structure:
h is cast to bf16 exactly where fla casts (the w@h / q@h operands), v' is cast to bf16 before
the DH dot, accumulation is fp32 everywhere.

Logical mode order convention (adapted from the mamba2_ssd example): every gmem tensor is
viewed so an MMA operand reads (M-or-N, K_contract, rest...):
    q,k,w as A/B of the BT-M mmas: (T, K, H|HV, B)
    k as mn-major A of DH:         (K, T, H, B)
    u, o:                          (T, V, HV, B)
    g2:                            (T, HV, B) fp32, T contiguous
    h0, ht:                        (K, V, HV, B) fp32
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


class GdnFwdStateKernel:
    """Fused fwd_h + fwd_o. One CTA per (b, hv, v_tile); serial over chunks.

    Warps: 0 = TMA producer, 1 = MMA, 4..7 = SIMT-recurrence (v'/v~', h update, ht),
    8..11 = SIMT-epilogue (Aqk, o readout + TMA store). Both SIMT quads are
    warpgroup-aligned — tmem load/store atoms address tmem rows per warp, so a group must
    be warps 4k..4k+3. Warps 2,3 idle through the role branch and only participate in
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
        self.tile_s1 = (self.BT, self.BT, self.K)  # q @ k^T
        self.tile_wh = (self.BT, self.BV, self.K)  # w @ h   (also q @ h)
        self.tile_oi = (self.BT, self.BV, self.BT)  # Aqk @ v'
        self.tile_dh = (self.BV, self.K, self.BT)  # DH^T = v~'^T @ k

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0
        self.mma_warp_id = 1
        self.simt_warp_id = (4, 5, 6, 7)  # recurrence group: v'/v~', h update, ht
        self.epi_warp_id = (8, 9, 10, 11)  # epilogue group: Aqk, o
        self.threads_per_cta = 32 * 12

        self.input_stages = 2
        self.h_stages = 2
        self.aqk_stages = 2

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
        mma_s1 = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_s1[:2], tcgen05.OperandSource.SMEM,
        )
        mma_wh = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_wh[:2], tcgen05.OperandSource.SMEM,
        )
        # OI: A = Aqk [BT,BT] from TMEM
        mma_oi = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_oi[:2], tcgen05.OperandSource.TMEM,
        )
        # DH^T = v~'^T @ k: A = v~' [BV,BT] k-major (as stored), B = k [K,BT] mn-major —
        # k is loaded a second time through its own TMA atom (16KB/chunk) rather than
        # aliasing S1's buffer under two swizzle schemes.
        mma_dh = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dh[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_s1, mma_wh, mma_oi, mma_dh

    def _setup_attributes(self):
        mma_s1, mma_wh, mma_oi, mma_dh = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV

        self.q_smem_layout = sm100_utils.make_smem_layout_a(
            mma_s1, self.tile_s1, self.io_dtype, self.input_stages
        )
        self.k_smem_layout = sm100_utils.make_smem_layout_b(
            mma_s1, self.tile_s1, self.io_dtype, self.input_stages
        )
        self.kt_smem_layout = sm100_utils.make_smem_layout_b(
            mma_dh, self.tile_dh, self.io_dtype, self.input_stages
        )
        self.w_smem_layout = sm100_utils.make_smem_layout_a(
            mma_wh, self.tile_wh, self.io_dtype, self.input_stages
        )
        # u is SIMT-only: plain row-major (BT, BV), BV contiguous
        self.u_smem_layout = cute.make_layout(
            (BT, BV, self.input_stages), stride=(BV, 1, BT * BV)
        )
        self.g2_smem_layout = cute.make_layout((BT, self.input_stages))

        # h as B operand of WH/OH ([BV, K] k-major, staged); written by SIMT through the
        # COL_MAJOR epi view of the same bytes (mamba2's P pattern).
        self.h_smem_layout = sm100_utils.make_smem_layout_b(
            mma_wh, self.tile_wh, self.io_dtype, self.h_stages
        )
        # DH^T acc fragments are [BV,K] row-major-ish -> plain stmatrix store into a
        # ROW_MAJOR (BV,K) view = exactly the K-contiguous B-operand bytes of sH.
        self.h_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, (BV, K), self.h_stages
        )
        # v'/v~' as B operands of OI/DH ([BV, BT] k-major), same write pattern
        self.vp_smem_layout = sm100_utils.make_smem_layout_b(
            mma_oi, self.tile_oi, self.io_dtype, 1
        )
        self.vp_epi_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.COL_MAJOR, (BT, BV), 1
        )
        # v~' is the A operand of DH^T: its own buffer, A-operand layout ([BV,BT] k-major)
        self.vpd_smem_layout = sm100_utils.make_smem_layout_a(
            mma_dh, self.tile_dh, self.io_dtype, 1
        )
        # o staging: (BT, BV) BV-contiguous to match gmem
        self.o_smem_layout = cute.make_layout((BT, BV, 1), stride=(BV, 1, BT * BV))
        # Aqk lives in tmem (io dtype), written by SIMT like mamba2's Q
        self.aqk_tmem_layout = sm100_utils.make_smem_layout_a(
            mma_oi, self.tile_oi, self.io_dtype, self.aqk_stages
        )

        self.num_qk_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.q_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.k_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.kt_smem_layout, (None, None, None, 0))
            )
        )
        self.num_wug_load_bytes = (
            cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.w_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.io_dtype, cute.slice_(self.u_smem_layout, (None, None, 0))
            )
            + cute.size_in_bytes(
                cutlass.Float32, cute.slice_(self.g2_smem_layout, (None, 0))
            )
        )

        (
            self.tmem_s1_offset,
            self.tmem_aqk_offset,
            self.tmem_wh_offset,
            self.tmem_oh_offset,
            self.tmem_oi_offset,
            self.tmem_dh_offset,
            self.num_tmem_cols,
        ) = self._plan_tmem(mma_s1, mma_wh, mma_oi, mma_dh)

    def _plan_tmem(self, mma_s1, mma_wh, mma_oi, mma_dh):
        def acc_cols(mma, tile):
            shape = mma.partition_shape_C(tile[:2])
            fake = mma.make_fragment_C(cute.append(shape, 1))
            return tcgen05.find_tmem_tensor_col_offset(fake)

        s1 = acc_cols(mma_s1, self.tile_s1)
        wh = acc_cols(mma_wh, self.tile_wh)
        oi = acc_cols(mma_oi, self.tile_oi)
        dh = acc_cols(mma_dh, self.tile_dh)
        aqk_fake = mma_oi.make_fragment_A(self.aqk_tmem_layout.outer.shape)
        aqk = tcgen05.find_tmem_tensor_col_offset(aqk_fake)

        off_s1 = 0
        off_aqk = off_s1 + s1
        off_wh = off_aqk + aqk
        off_oh = off_wh + wh
        off_oi = off_oh + wh
        off_dh = off_oi + oi
        total_ = off_dh + dh
        total = 1
        while total < total_:
            total *= 2
        assert total <= 512, f"tmem overflow: {total_} cols"
        return off_s1, off_aqk, off_wh, off_oh, off_oi, off_dh, total

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (T, K, H, B)
        k: cute.Tensor,  # (T, K, H, B)
        kt: cute.Tensor,  # (K, T, H, B) — same storage as k
        w: cute.Tensor,  # (T, K, HV, B)
        u: cute.Tensor,  # (T, V, HV, B)
        g2: cute.Tensor,  # (BT, NT, HV, B) fp32 — chunk dim pre-split so mode 0 stays static
        h0: cute.Tensor,  # (K, V, HV, B) fp32
        o: cute.Tensor,  # (T, V, HV, B)
        ht: cute.Tensor,  # (K, V, HV, B) fp32
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_s1, mma_wh, mma_oi, mma_dh = self._make_tiled_mmas()
        BT, K, BV = self.BT, self.K, self.BV
        cluster_vmnk = (1, 1, 1, 1)

        tma_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), q,
            cute.slice_(self.q_smem_layout, (None, None, None, 0)),
            self.tile_s1, mma_s1, cluster_vmnk,
        )
        tma_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), k,
            cute.slice_(self.k_smem_layout, (None, None, None, 0)),
            self.tile_s1, mma_s1, cluster_vmnk,
        )
        tma_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), kt,
            cute.slice_(self.kt_smem_layout, (None, None, None, 0)),
            self.tile_dh, mma_dh, cluster_vmnk,
        )
        tma_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), w,
            cute.slice_(self.w_smem_layout, (None, None, None, 0)),
            self.tile_wh, mma_wh, cluster_vmnk,
        )
        tma_u, tma_tensor_u = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), u,
            cute.slice_(self.u_smem_layout, (None, None, 0)),
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
        tma_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), o,
            cute.slice_(self.o_smem_layout, (None, None, 0)),
            (BT, BV),
        )

        B = cute.size(q, mode=[3])
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
            qk_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            wug_full: cute.struct.MemRange[cutlass.Int64, self.input_stages * 2]  # type: ignore
            h_full: cute.struct.MemRange[cutlass.Int64, self.h_stages * 2]  # type: ignore
            vp_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            vpd_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            aqk_full: cute.struct.MemRange[cutlass.Int64, self.aqk_stages * 2]  # type: ignore
            s1_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            wh_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            oho_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            oio_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dh_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.q_smem_layout)], swz_align  # type: ignore
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.k_smem_layout)], swz_align  # type: ignore
            ]
            smem_kt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.kt_smem_layout)], swz_align  # type: ignore
            ]
            smem_w: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.w_smem_layout)], swz_align  # type: ignore
            ]
            smem_u: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.u_smem_layout)], lin_align  # type: ignore
            ]
            smem_g2: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.g2_smem_layout)], lin_align  # type: ignore
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

        self.gdn_cute_fwd(
            tma_q, tma_tensor_q,
            tma_k, tma_tensor_k,
            tma_kt, tma_tensor_kt,
            tma_w, tma_tensor_w,
            tma_u, tma_tensor_u,
            tma_g2, tma_tensor_g2,
            tma_o, tma_tensor_o,
            ht,
            h0,
            scale,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def gdn_cute_fwd(
        self,
        tma_q: cute.CopyAtom, mQ: cute.Tensor,
        tma_k: cute.CopyAtom, mK: cute.Tensor,
        tma_kt: cute.CopyAtom, mKT: cute.Tensor,
        tma_w: cute.CopyAtom, mW: cute.Tensor,
        tma_u: cute.CopyAtom, mU: cute.Tensor,
        tma_g2: cute.CopyAtom, mG2: cute.Tensor,
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
        mma_s1, mma_wh, mma_oi, mma_dh = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_q, tma_k, tma_kt, tma_w, tma_u, tma_g2, tma_o]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mW, mode=[2])
        H = cute.size(mQ, mode=[2])
        NV = cute.size(mU, mode=[1]) // BV
        T = cute.size(mQ, mode=[0])
        NT = T // BT
        v_idx = bidx % NV
        hv_idx = (bidx // NV) % HV
        b_idx = bidx // (NV * HV)
        h_idx = hv_idx // (HV // H)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.smem_q.get_tensor(self.q_smem_layout.outer, swizzle=self.q_smem_layout.inner)
        sK = storage.smem_k.get_tensor(self.k_smem_layout.outer, swizzle=self.k_smem_layout.inner)
        sKT = storage.smem_kt.get_tensor(self.kt_smem_layout.outer, swizzle=self.kt_smem_layout.inner)
        sW = storage.smem_w.get_tensor(self.w_smem_layout.outer, swizzle=self.w_smem_layout.inner)
        sU = storage.smem_u.get_tensor(self.u_smem_layout)
        sG2 = storage.smem_g2.get_tensor(self.g2_smem_layout)
        sH = storage.smem_h.get_tensor(self.h_smem_layout.outer, swizzle=self.h_smem_layout.inner)
        sH_epi = storage.smem_h.get_tensor(self.h_epi_layout.outer, swizzle=self.h_epi_layout.inner)
        sVp = storage.smem_vp.get_tensor(self.vp_smem_layout.outer, swizzle=self.vp_smem_layout.inner)
        sVp_epi = storage.smem_vp.get_tensor(self.vp_epi_layout.outer, swizzle=self.vp_epi_layout.inner)
        sVpd = storage.smem_vpd.get_tensor(self.vpd_smem_layout.outer, swizzle=self.vpd_smem_layout.inner)
        sVpd_epi = storage.smem_vpd.get_tensor(self.vp_epi_layout.outer, swizzle=self.vp_epi_layout.inner)
        sO = storage.smem_o.get_tensor(self.o_smem_layout)

        # ---- pipelines ----
        # Two SIMT groups: "simt" (warps 4..7) owns the recurrence-side exchanges
        # (h, vp, vpd producers; wh, dh consumers), "epi" (warps 8..11) owns the
        # off-critical-path ones (aqk producer; s1, oho, oio consumers). wug (w/u/g2
        # loads) is consumed by both groups — g2 is read on both sides — so its async
        # consumer group spans the two quads.
        simt_threads = 32 * len(self.simt_warp_id)
        epi_threads = 32 * len(self.epi_warp_id)
        qk_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=self.num_qk_load_bytes,
            barrier_storage=storage.qk_full.data_ptr(),
            defer_sync=True,
        )
        wug_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.input_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, simt_threads + epi_threads
            ),
            tx_count=self.num_wug_load_bytes,
            barrier_storage=storage.wug_full.data_ptr(),
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
        aqk_pipe = make_simt_to_mma_pipe(
            storage.aqk_full.data_ptr(), self.aqk_stages, epi_threads
        )
        s1_pipe = make_mma_to_simt_pipe(storage.s1_full.data_ptr(), epi_threads)
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

        tS1 = acc_tensor(mma_s1, self.tile_s1, self.tmem_s1_offset)
        tWH = acc_tensor(mma_wh, self.tile_wh, self.tmem_wh_offset)
        tOH = acc_tensor(mma_wh, self.tile_wh, self.tmem_oh_offset)
        tOI = acc_tensor(mma_oi, self.tile_oi, self.tmem_oi_offset)
        tDH = acc_tensor(mma_dh, self.tile_dh, self.tmem_dh_offset)

        tAqk_fake = mma_oi.make_fragment_A(self.aqk_tmem_layout.outer.shape)
        tAqk = cute.make_tensor(
            cute.recast_ptr(tmem_ptr_base + self.tmem_aqk_offset, dtype=io),
            tAqk_fake.layout,
        )

        # ---- global tiles (per (b, hv, v_idx), open over chunks) ----
        gQ = cute.local_tile(mQ, (BT, K), (None, 0, h_idx, b_idx))  # (BT,K,NT)
        gK = cute.local_tile(mK, (BT, K), (None, 0, h_idx, b_idx))
        gKT = cute.local_tile(mKT, (K, BT), (0, None, h_idx, b_idx))  # (K,BT,NT)
        gW = cute.local_tile(mW, (BT, K), (None, 0, hv_idx, b_idx))
        gU = cute.local_tile(mU, (BT, BV), (None, v_idx, hv_idx, b_idx))  # (BT,BV,NT)
        gG2 = mG2[(None, None, hv_idx, b_idx)]  # (BT, NT)
        gO = cute.local_tile(mO, (BT, BV), (None, v_idx, hv_idx, b_idx))

        # ==========================================================================
        # TMA warp
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_s1 = mma_s1.get_slice(0)
            thr_mma_wh = mma_wh.get_slice(0)
            thr_mma_dh = mma_dh.get_slice(0)

            tQ_mma = thr_mma_s1.partition_A(gQ)
            tK_mma = thr_mma_s1.partition_B(gK)
            tKT_mma = thr_mma_dh.partition_B(gKT)
            tW_mma = thr_mma_wh.partition_A(gW)

            cta1 = cute.make_layout(1)
            tQs, tQg = cpasync.tma_partition(
                tma_q, 0, cta1, cute.group_modes(sQ, 0, 3), cute.group_modes(tQ_mma, 0, 3)
            )
            tKs, tKg = cpasync.tma_partition(
                tma_k, 0, cta1, cute.group_modes(sK, 0, 3), cute.group_modes(tK_mma, 0, 3)
            )
            tKTs, tKTg = cpasync.tma_partition(
                tma_kt, 0, cta1, cute.group_modes(sKT, 0, 3), cute.group_modes(tKT_mma, 0, 3)
            )
            tWs, tWg = cpasync.tma_partition(
                tma_w, 0, cta1, cute.group_modes(sW, 0, 3), cute.group_modes(tW_mma, 0, 3)
            )
            tUs, tUg = cpasync.tma_partition(
                tma_u, 0, cta1, cute.group_modes(sU, 0, 2), cute.group_modes(gU, 0, 2)
            )
            tG2s, tG2g = cpasync.tma_partition(
                tma_g2, 0, cta1, cute.group_modes(sG2, 0, 1), cute.group_modes(gG2, 0, 1)
            )

            qk_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )
            wug_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.input_stages
            )

            for c in cutlass.range(NT, unroll=1):
                qk_pipe.producer_acquire(qk_producer)
                bar = qk_pipe.producer_get_barrier(qk_producer)
                cute.copy(tma_q, tQg[None, c], tQs[None, qk_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_k, tKg[None, c], tKs[None, qk_producer.index], tma_bar_ptr=bar)
                cute.copy(tma_kt, tKTg[None, c], tKTs[None, qk_producer.index], tma_bar_ptr=bar)
                qk_producer.advance()

                wug_pipe.producer_acquire(wug_producer)
                wbar = wug_pipe.producer_get_barrier(wug_producer)
                cute.copy(tma_w, tWg[None, c], tWs[None, wug_producer.index], tma_bar_ptr=wbar)
                cute.copy(tma_u, tUg[None, c], tUs[None, wug_producer.index], tma_bar_ptr=wbar)
                cute.copy(tma_g2, tG2g[None, c], tG2s[None, wug_producer.index], tma_bar_ptr=wbar)
                wug_producer.advance()

            qk_pipe.producer_tail(qk_producer)
            wug_pipe.producer_tail(wug_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrQ = mma_s1.make_fragment_A(sQ)
            tCrK = mma_s1.make_fragment_B(sK)
            tCrQh = mma_wh.make_fragment_A(sQ)  # q reused as A of OH (same layout family)
            tCrW = mma_wh.make_fragment_A(sW)
            tCrH = mma_wh.make_fragment_B(sH)
            tCrKT = mma_dh.make_fragment_B(sKT)
            tCrVp = mma_oi.make_fragment_B(sVp)
            tCrVpd = mma_dh.make_fragment_A(sVpd)
            tCrAqk_fake = mma_oi.make_fragment_A(self.aqk_tmem_layout.outer.shape)
            tCrAqk = cute.make_tensor(
                cute.recast_ptr(tmem_ptr_base + self.tmem_aqk_offset, dtype=io),
                tCrAqk_fake.layout,
            )

            qk_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            wug_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            h_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.h_stages
            )
            vp_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            vpd_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            aqk_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.aqk_stages
            )
            s1_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            wh_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            oho_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            oio_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dh_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # Prologue: S1_0 — depends only on qk stage 0, so it goes ahead of the loop
            # and every later S1 is issued for chunk c+1 at the *end* of iteration c.
            # That way the epilogue group can compute Aqk_{c+1} while the recurrence
            # side of chunk c+1 is still in flight.
            qk_pipe.consumer_wait(qk_consumer)
            s1_pipe.producer_acquire(s1_producer)
            for kk in cutlass.range(cute.size(tCrK, mode=[2]), unroll_full=True):
                mma_s1.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                cute.gemm(
                    mma_s1, tS1[None, None, None, 0],
                    tCrQ[None, None, kk, qk_consumer.index],
                    tCrK[None, None, kk, qk_consumer.index],
                    tS1[None, None, None, 0],
                )
            s1_pipe.producer_commit(s1_producer)
            s1_producer.advance()

            for c in cutlass.range(NT, unroll=1):
                # WH = w h — first: it heads the critical path.
                wug_pipe.consumer_wait(wug_consumer)
                h_pipe.consumer_wait(h_consumer)
                wh_pipe.producer_acquire(wh_producer)
                for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                    mma_wh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_wh, tWH[None, None, None, 0],
                        tCrW[None, None, kk, wug_consumer.index],
                        tCrH[None, None, kk, h_consumer.index],
                        tWH[None, None, None, 0],
                    )
                wh_pipe.producer_commit(wh_producer)
                wh_producer.advance()
                wug_pipe.consumer_release(wug_consumer, pipeline.PipelineOp.TCGen05Mma)
                wug_consumer.advance()

                # OH = q h
                oho_pipe.producer_acquire(oho_producer)
                for kk in cutlass.range(cute.size(tCrH, mode=[2]), unroll_full=True):
                    mma_wh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_wh, tOH[None, None, None, 0],
                        tCrQh[None, None, kk, qk_consumer.index],
                        tCrH[None, None, kk, h_consumer.index],
                        tOH[None, None, None, 0],
                    )
                oho_pipe.producer_commit(oho_producer)
                oho_producer.advance()
                h_pipe.consumer_release(h_consumer)
                h_consumer.advance()

                # DH = k^T v~' — before OI so the h update never queues behind o-work.
                vpd_pipe.consumer_wait(vpd_consumer)
                dh_pipe.producer_acquire(dh_producer)
                for kk in cutlass.range(cute.size(tCrVpd, mode=[2]), unroll_full=True):
                    mma_dh.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_dh, tDH[None, None, None, 0],
                        tCrVpd[None, None, kk, vpd_consumer.index],
                        tCrKT[None, None, kk, qk_consumer.index],
                        tDH[None, None, None, 0],
                    )
                dh_pipe.producer_commit(dh_producer)
                dh_producer.advance()
                vpd_pipe.consumer_release(vpd_consumer)
                vpd_consumer.advance()
                qk_pipe.consumer_release(qk_consumer)
                qk_consumer.advance()

                # OI = Aqk v'
                aqk_pipe.consumer_wait(aqk_consumer)
                vp_pipe.consumer_wait(vp_consumer)
                oio_pipe.producer_acquire(oio_producer)
                for kk in cutlass.range(cute.size(tCrVp, mode=[2]), unroll_full=True):
                    mma_oi.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_oi, tOI[None, None, None, 0],
                        tCrAqk[None, None, kk, aqk_consumer.index],
                        tCrVp[None, None, kk, vp_consumer.index],
                        tOI[None, None, None, 0],
                    )
                oio_pipe.producer_commit(oio_producer)
                oio_producer.advance()
                aqk_pipe.consumer_release(aqk_consumer)
                aqk_consumer.advance()
                vp_pipe.consumer_release(vp_consumer)
                vp_consumer.advance()

                # S1 for chunk c+1
                if c + 1 < NT:
                    qk_pipe.consumer_wait(qk_consumer)
                    s1_pipe.producer_acquire(s1_producer)
                    for kk in cutlass.range(cute.size(tCrK, mode=[2]), unroll_full=True):
                        mma_s1.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                        cute.gemm(
                            mma_s1, tS1[None, None, None, 0],
                            tCrQ[None, None, kk, qk_consumer.index],
                            tCrK[None, None, kk, qk_consumer.index],
                            tS1[None, None, None, 0],
                        )
                    s1_pipe.producer_commit(s1_producer)
                    s1_producer.advance()

            s1_pipe.producer_tail(s1_producer)
            wh_pipe.producer_tail(wh_producer)
            oho_pipe.producer_tail(oho_producer)
            oio_pipe.producer_tail(oio_producer)
            dh_pipe.producer_tail(dh_producer)

        # ==========================================================================
        # SIMT recurrence warps 4..7: v'/v~' from WH, the h update from DH, ht.
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

            # --- WH -> v'/v~' ---
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
            sG2_rowv = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BV, self.input_stages), stride=(1, 0, BT)),
            )
            tWHsG2 = thr_t2r_wh.partition_D(sG2_rowv)
            tWHrG2 = cute.make_rmem_tensor(
                cute.slice_(tWHsG2.shape, (None, None, None, 0)), f32
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
            tRS_rVpd = cute.make_rmem_tensor(
                cute.slice_(tRS_sVpd.shape, (None, None, None, 0)), io
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
            r2s_h_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_h = cute.make_tiled_copy_D(r2s_h_atom, tiled_t2r_dh)
            thr_r2s_h = tiled_r2s_h.get_slice(local_tidx)
            tRS_sH = thr_r2s_h.partition_D(sH_epi)
            tRS_rH = cute.make_rmem_tensor(
                cute.slice_(tRS_sH.shape, (None, None, None, 0)), io
            )

            wug_consumer = pipeline.make_pipeline_state(
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
                wug_pipe.consumer_wait(wug_consumer)
                g_last = sG2[BT - 1, wug_consumer.index]
                exp_g_last = cute.math.exp2(g_last, fastmath=True)
                wcrd = (None, None, None, wug_consumer.index)

                # v', v~'
                wh_pipe.consumer_wait(wh_consumer)
                cute.copy(tiled_t2r_wh, tTR_tWH[None, None, None, 0], tTR_rWH)
                cute.arch.fence_view_async_tmem_load()
                wh_pipe.consumer_release(wh_consumer)
                wh_consumer.advance()
                cute.copy(io_cp_atom, tWHsU[wcrd], tWHrU)
                cute.copy(f32_cp_atom, tWHsG2[wcrd], tWHrG2)
                vp_pipe.producer_acquire(vp_producer)
                vpd_pipe.producer_acquire(vpd_producer)
                for i in cutlass.range(
                    cute.size(tTR_rWH), unroll_full=True, vectorize=True
                ):
                    vprime = tWHrU[i].to(f32) - tTR_rWH[i]
                    tRS_rVp[i] = vprime.to(io)
                    dec = cute.math.exp2(g_last - tWHrG2[i], fastmath=True)
                    tRS_rVpd[i] = (vprime * dec).to(io)
                # v~' first: it feeds DH, the next hop of the critical path; v' only feeds
                # the off-path OI, so its store+commit can trail behind a second barrier.
                cute.copy(tiled_r2s_vp, tRS_rVpd, tRS_sVpd[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                vpd_pipe.producer_commit(vpd_producer)
                vpd_producer.advance()
                cute.copy(tiled_r2s_vp, tRS_rVp, tRS_sVp[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.simt_sync_barrier.arrive_and_wait()
                vp_pipe.producer_commit(vp_producer)
                vp_producer.advance()

                # h update
                dh_pipe.consumer_wait(dh_consumer)
                cute.copy(tiled_t2r_dh, tTR_tDH[None, None, None, 0], tTR_rDH)
                cute.arch.fence_view_async_tmem_load()
                dh_pipe.consumer_release(dh_consumer)
                dh_consumer.advance()
                for i in cutlass.range(
                    cute.size(tHreg), unroll_full=True, vectorize=True
                ):
                    tHreg[i] = exp_g_last * tHreg[i] + tTR_rDH[i]
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

                wug_pipe.consumer_release(
                    wug_consumer, pipeline.PipelineOp.AsyncThread
                )
                wug_consumer.advance()

            # ---- ht (fp32): once-per-kernel plain global scatter ----
            for i in cutlass.range(cute.size(tHreg), unroll_full=True):
                vv, kk = coordDH[i]
                mHT[(kk, v_idx * BV + vv, hv_idx, b_idx)] = tHreg[i]

        # ==========================================================================
        # SIMT epilogue warps 8..11: Aqk from S1, o from OH/OI. Off the critical
        # path — this group trails the recurrence without ever stalling it.
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
            f32_cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), f32)

            # --- S1 -> Aqk ---
            tS1_2d = tS1[((None, None), 0, 0, None)]
            tiled_t2r_s1 = tcgen05.make_tmem_copy(t2r_64_atom, tS1_2d[None, None, 0])
            thr_t2r_s1 = tiled_t2r_s1.get_slice(local_tidx)
            tTR_tS1 = thr_t2r_s1.partition_S(tS1_2d)
            coordS1 = thr_t2r_s1.partition_D(cute.make_identity_tensor((BT, BT)))
            tTR_rS1 = cute.make_rmem_tensor(coordS1.shape, f32)
            tDec = cute.make_rmem_tensor(tTR_rS1.shape, f32)
            sG2_row = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BT, self.input_stages), stride=(0, 1, BT)),
            )
            sG2_col = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BT, self.input_stages), stride=(1, 0, BT)),
            )
            tS1sG2r = thr_t2r_s1.partition_D(sG2_row)
            tS1rG2r = cute.make_rmem_tensor(
                cute.slice_(tS1sG2r.shape, (None, None, None, 0)), f32
            )
            tS1sG2c = thr_t2r_s1.partition_D(sG2_col)
            tS1rG2c = cute.make_rmem_tensor(
                cute.slice_(tS1sG2c.shape, (None, None, None, 0)), f32
            )
            r2t_aqk_atom = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE), io
            )
            tiled_r2t_aqk = tcgen05.make_tmem_copy(r2t_aqk_atom, tAqk)
            thr_r2t_aqk = tiled_r2t_aqk.get_slice(local_tidx)
            tRT_tAqk = thr_r2t_aqk.partition_D(tAqk)
            # r2t: tmem is the D side here, so the rmem source comes from partition_S of a
            # plain (BT,BT) tensor — same lane-folding trap as in the recurrence group,
            # mirrored.
            tRT_rAqk = cute.make_rmem_tensor(
                cute.slice_(
                    thr_r2t_aqk.partition_S(
                        cute.make_identity_tensor(tAqk.shape)
                    ).shape,
                    (None, None, None, None, 0),
                ),
                io,
            )

            # --- OH/OI -> o ---
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
            sG2_rowv = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BV, self.input_stages), stride=(1, 0, BT)),
            )
            tOHsG2 = thr_t2r_oh.partition_D(sG2_rowv)
            tOHrG2 = cute.make_rmem_tensor(
                cute.slice_(tOHsG2.shape, (None, None, None, 0)), f32
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

            wug_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.input_stages
            )
            s1_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            oho_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            oio_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            aqk_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.aqk_stages
            )

            for c in cutlass.range(NT, unroll=1):
                wug_pipe.consumer_wait(wug_consumer)
                wcrd = (None, None, None, wug_consumer.index)

                # Aqk
                s1_pipe.consumer_wait(s1_consumer)
                cute.copy(tiled_t2r_s1, tTR_tS1[None, None, None, 0], tTR_rS1)
                cute.arch.fence_view_async_tmem_load()
                s1_pipe.consumer_release(s1_consumer)
                s1_consumer.advance()
                cute.copy(f32_cp_atom, tS1sG2r[wcrd], tS1rG2r)
                cute.copy(f32_cp_atom, tS1sG2c[wcrd], tS1rG2c)
                aqk_pipe.producer_acquire(aqk_producer)
                # mamba2 segsum structure: vectorized arithmetic, minimal scalar loop
                # for the coordinate mask, exp2(-inf)=0 folds the tril mask into decay
                for i in cutlass.range(
                    cute.size(tTR_rS1), unroll_full=True, vectorize=True
                ):
                    tDec[i] = tS1rG2c[i] - tS1rG2r[i]
                for i in cutlass.range(cute.size(tTR_rS1), unroll_full=True):
                    mi, nj = coordS1[i]
                    if mi < nj:
                        tDec[i] = cutlass.Float32(-float("inf"))
                for i in cutlass.range(
                    cute.size(tTR_rS1), unroll_full=True, vectorize=True
                ):
                    d = cute.math.exp2(tDec[i], fastmath=True)
                    tRT_rAqk[i] = (tTR_rS1[i] * d).to(io)
                cute.copy(
                    tiled_r2t_aqk, tRT_rAqk,
                    tRT_tAqk[None, None, None, None, aqk_producer.index],
                )
                cute.arch.fence_view_async_tmem_store()
                self.epi_sync_barrier.arrive_and_wait()
                aqk_pipe.producer_commit(aqk_producer)
                aqk_producer.advance()

                # o
                oho_pipe.consumer_wait(oho_consumer)
                cute.copy(tiled_t2r_oh, tTR_tOH[None, None, None, 0], tTR_rOH)
                oio_pipe.consumer_wait(oio_consumer)
                cute.copy(tiled_t2r_oio, tTR_tOI[None, None, None, 0], tTR_rOI)
                cute.arch.fence_view_async_tmem_load()
                oho_pipe.consumer_release(oho_consumer)
                oho_consumer.advance()
                oio_pipe.consumer_release(oio_consumer)
                oio_consumer.advance()
                cute.copy(f32_cp_atom, tOHsG2[wcrd], tOHrG2)
                for i in cutlass.range(
                    cute.size(tTR_rOH), unroll_full=True, vectorize=True
                ):
                    gd = cute.math.exp2(tOHrG2[i], fastmath=True)
                    tRS_rO[i] = (scale * (gd * tTR_rOH[i] + tTR_rOI[i])).to(io)
                cute.copy(tiled_r2s_o, tRS_rO, tRS_sO[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epi_sync_barrier.arrive_and_wait()
                if warp_idx == self.epi_warp_id[0]:
                    cute.copy(tma_o, bSG_sO[None, 0], bSG_gO[None, c])
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
                self.epi_sync_barrier.arrive_and_wait()

                wug_pipe.consumer_release(
                    wug_consumer, pipeline.PipelineOp.AsyncThread
                )
                wug_consumer.advance()

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
    # cute wants the modes ordered outermost-to-innermost, and checks that against the
    # strides. tt.dim_order() is not usable: it places size-1 modes by contiguity
    # heuristics, so e.g. (512,128,2,1):(256,1,128,131072) comes back as (0,2,3,1) and cute
    # rejects it. Sorting tt's strides directly has the mirror problem — size-1 modes tie
    # with their neighbour and land on the wrong side. Take the order from the unpermuted
    # tensor (where descending stride is unambiguous) and map it through perm.
    # detach: inside an autograd.Function the incoming leaves still carry requires_grad, and
    # dlpack refuses to export those. The gradient path is hand-written in the wrapper, so
    # nothing here needs the autograd edge — only the storage.
    t = t.detach()
    base_order = sorted(range(t.dim()), key=lambda i: -t.stride(i))
    new_of_old = {old: new for new, old in enumerate(perm)}
    stride_order = tuple(new_of_old[d] for d in base_order)
    tt = t.permute(*perm)
    ct = from_dlpack(tt, assumed_align=16)
    for m in dyn_modes:
        ct = ct.mark_compact_shape_dynamic(mode=m, stride_order=stride_order)
    return ct


# Marshaling a call — nine _cute_view()s plus a fresh g2c staging tensor — costs ~0.28ms of
# host time, 4x the T2048 kernel itself. But the marshaled views depend on the input
# *layouts*, not the input *storage*: the compiled callable reads each tensor argument
# through a small C descriptor whose word[0] is the raw device pointer (verified against
# runtime.py's build_memref_desc). So cache the fully-marshaled call keyed on
# (shape, stride, dtype) signatures, and per call retarget the six input descriptors with
# one ctypes word-write each. Fresh fla-allocated w/u/g2 every call — the case that made a
# pointer-keyed cache useless — costs six pokes and one g2 copy.
#
# Outputs are deliberately NOT owned by the cache entry: only their (shape, dtype) is
# remembered, and each call allocates fresh ones and re-points their descriptors by the same
# one-word write. Recycling them would be the aliasing contract of a persistent-workspace
# kernel - harmless for a bench harness that builds fresh inputs per arm, fatal in a model,
# where every GDN layer has identical shapes and so would overwrite the `o` the previous layer
# saved for its backward.
_CALL_CACHE: dict = {}


def _retarget(ct, t: torch.Tensor) -> None:
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def _spec(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.dtype)


def _alloc(specs: tuple, device: torch.device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


def _call_key(q, k, w, u, g2, h0, scale):
    def sig(t):
        return (t.shape, t.stride(), t.dtype)

    return (sig(q), sig(k), sig(w), sig(u), sig(g2), sig(h0), scale,
            torch.cuda.current_stream().cuda_stream)


def gdn_cute_fwd_call(
    q: torch.Tensor,  # [B,T,H,K] bf16/fp16, L2-normalized
    k: torch.Tensor,
    w: torch.Tensor,  # [B,T,HV,K]
    u: torch.Tensor,  # [B,T,HV,V]
    g2: torch.Tensor,  # [B,T,HV] fp32, chunk-local cumsum / ln2
    h0: torch.Tensor,  # [B,HV,K,V] fp32
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _call_key(q, k, w, u, g2, h0, scale)
    ent = _CALL_CACHE.get(key)
    if ent is None:
        B, T, H, K = q.shape
        HV, V = u.shape[2], u.shape[3]
        assert T % 64 == 0, "T must be a multiple of the chunk size"
        assert V % 64 == 0

        o = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)
        ht = torch.empty(B, HV, K, V, device=q.device, dtype=torch.float32)
        # [B,HV,NT,BT] with BT contiguous, so the cute view (BT,NT,HV,B) has a static
        # stride-1 mode 0 — dynamic modes must not include the innermost one. A persistent
        # buffer: the cute view points at its storage, refilled on every call below.
        g2c = torch.empty(B, HV, T // 64, 64, device=g2.device, dtype=g2.dtype)

        io_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V)  # V is a static layout mode

        # logical (M/N, K, rest) views — see module docstring
        cq = _cute_view(q, (1, 3, 2, 0), (0, 2, 3))
        ck = _cute_view(k, (1, 3, 2, 0), (0, 2, 3))
        ckt = _cute_view(k, (3, 1, 2, 0), (1, 2, 3))
        cw = _cute_view(w, (1, 3, 2, 0), (0, 2, 3))
        cu = _cute_view(u, (1, 3, 2, 0), (0, 2, 3))
        cg2 = _cute_view(g2c, (3, 2, 1, 0), (1, 2, 3))
        ch0 = _cute_view(h0, (2, 3, 1, 0), (2, 3))
        co = _cute_view(o, (1, 3, 2, 0), (0, 2, 3))
        cht = _cute_view(ht, (2, 3, 1, 0), (2, 3))

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = GdnFwdStateKernel(io_dtype, K, V_TILE=64)
            compiled = cute.compile(
                kernel_obj, cq, ck, ckt, cw, cu, cg2, ch0, co, cht,
                cutlass.Float32(scale), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (cq, ck, ckt, cw, cu, cg2, ch0, co, cht, cutlass.Float32(scale), stream)
        if len(_CALL_CACHE) >= 64:  # distinct layouts are few; this is a leak backstop
            _CALL_CACHE.clear()
        ent = (compiled, args, (_spec(o), _spec(ht)), g2c)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, g2c = ent
    cq, ck, ckt, cw, cu, _, ch0, co, cht, _, _ = args
    # Fresh outputs per call - see the package docstring on why they are not cache-owned.
    o, ht = _alloc(out_specs, q.device)
    _retarget(co, o)
    _retarget(cht, ht)
    _retarget(cq, q)
    _retarget(ck, k)
    _retarget(ckt, k)
    _retarget(cw, w)
    _retarget(cu, u)
    _retarget(ch0, h0)
    B, HV, NT, BT = g2c.shape
    # refill the staging buffer in place; copy_ handles the transposed source directly
    g2c.view(B, HV, NT * BT).copy_(g2.transpose(1, 2))
    compiled(*args)
    return o, ht


def gdn_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    scale: float | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward only; drops the residuals the autograd wrapper needs."""
    o, ht, _, _ = gdn_cute_fwd_with_residuals(q, k, v, g, beta, h0, scale, chunk_size)
    return o, ht


def gdn_cute_fwd_with_residuals(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    scale: float | None,
    chunk_size: int,
):
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2 as FLA_RCP_LN2

    if chunk_size != 64:
        raise NotImplementedError("cute path only implements chunk_size=64")
    if scale is None:
        scale = q.shape[-1] ** -0.5

    g2 = chunk_local_cumsum(g, chunk_size=chunk_size, scale=FLA_RCP_LN2)
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g2, beta=beta, chunk_size=chunk_size
    )
    o, ht = gdn_cute_fwd_call(q, k, w, u, g2, h0, float(scale))
    return o, ht, g2, A
