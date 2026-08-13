"""CuTe DSL port of fla's `prepare_wy_repr_bwd` (backward stage 6). See NOTES.md / ALGORITHM.md.

The math, per (chunk c, b, hv), with s = beta, e = exp2(g2), A loaded transposed (fla's b_A):

    pass 1:  DA_K  = dw @ k^T                      [BT,BT]  fp32
             DA_V  = du @ v^T                      [BT,BT]
             DKBG  = A^T @ dw                      [BT,K]
             DVB   = A^T @ du                      [BT,V]
    then:    dA    = tril_strict(DA_K * (s*e)[None,:] + DA_V * s[None,:])  -> bf16
             dk1   = DKBG * (e*s)[:,None]                    (held in registers)
             dv    = DVB * s[:,None]                         (stored)
             db   += e * rowsum(DKBG . k) + rowsum(DVB . v)
             dg   += s * e * rowsum(DKBG . k)
    sandwich: S1 = dA @ A^T ; S2 = A^T @ S1.bf16   (fla's two dots, same casts)
             dAf   = tril_strict(-S2 * exp2(g_i - g_j))      -> bf16, plus dAf^T * s[:,None]
    pass 2:  AKK   = k @ k^T
             DKB   = dAf @ k ;  TERM2 = (dAf * s[:,None])^T @ k
             dk    = dk1 + DKB * s[:,None] + TERM2           (stored)
             p     = dAf . AKK   (elementwise)
             db   += rowsum(p)            [identity: rowsum(dot(dAf,k) . k) == rowsum(dAf . AKK)]
             dg   += s * rowsum(p) - colsum(p * s[:,None])

Key deviations from fla's rounding, all strictly more accurate and inside the stage tolerance:
- fla rounds kbg = k*(s*e) and vb = v*s to bf16 *before* the pass-1 dots; those scales sit on
  the *output column* index, so we consume raw bf16 k/v/dw/du and scale the fp32 accs instead.
  This is what makes pass 1 free of SIMT-built operands.
- fla's dk term dot(dA, k) * s and trans(dot(trans(k*s).bf16, dA)) become one bf16 dA operand
  (bit-identical to fla's) plus a beta-scaled transposed copy (noise equivalent to fla's
  bf16(k*s)); fla's rowsum(dot(dA,k).k) for db becomes the AKK identity above, which avoids a
  whole extra 128-col readout.

Structure notes (everything else is kernel_dqkwg.py's skeleton verbatim):
- Grid = B*HV*nseg CTAs, each a contiguous chunk range. Warps: 0 = k/dw/A/beta/g producer
  (2-stage), 2 = v/du producer (1-stage; refill hides under the sandwich), 1 = MMA,
  4..7 = SIMT A (dA build, sandwich casts, dAf, AKK.p, db/dg epilogue), 8..11 = SIMT B
  (dk1/dv readouts + rowsums, dk combine + stores).
- tmem: DA_K@0(64) DA_V@64(64) DKBG@128(128) DVB@256(V). Aliases: S1@0, S2@64, AKK@256,
  DKB@128, TERM2@DVB+64 (fresh columns when V < 192). Point-of-use producer_acquires pair
  every alias with its *reader's* release, per the mma-warp-period rule.
- Per-row reductions (db/dg) exploit the Ld16x256b t2r shape: 16 rows x 2 lanes per warp, so
  every lane's fragment lives in ONE row and the two lanes of a row are told apart by their
  first column (0 vs nonzero). Partials go to parity-buffered smem slots; the parity plus the
  end-of-chunk ab barrier is what makes chunk c's epilogue reads safe against chunk c+1/c+2
  writes without a mid-chunk rendezvous.
- p = dAf.AKK is staged to a plain fp32 [64,64] view over the dA_m/S1 operand bytes (both are
  dead once S2 completes; same-warpgroup program order covers the next-chunk overwrite).
  Column sums read it lane-per-column, which is bank-conflict-free without padding.

Logical gmem mode order (M/N, K_contract, rest...):
    k:            (T, K, H, B)
    dw, dk(out):  (T, K, HV, B)
    v, du, dv(out): (T, V, HV, B)
    A:            (T, BT, HV, B)   -- b_A^T relative to fla's load; the kernel wants it this way
    beta, g2:     (BT, NT, HV, B) fp32, staged host-side like the fwd's g2
    db, dg:       (T, HV, B) fp32 outputs, plain scatter
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


class GdnBwdWyKernel:
    """dk/dv/db/dg backward stage 6. One CTA per (b, hv, chunk-segment); serial over chunks.

    Warps: 0 = k/dw/A/beta/g TMA producer, 1 = MMA, 2 = v/du TMA producer, 4..7 = SIMT
    group A (dA path + db/dg epilogue), 8..11 = SIMT group B (dk/dv path). Warp 3 idles
    through the role branch and only takes part in the alloc/dealloc barriers.
    """

    def __init__(self, io_dtype: Type[cutlass.Numeric], K: int, V: int):
        self.io_dtype = io_dtype
        self.acc_dtype = cutlass.Float32
        self.BT = 64
        self.K = K
        self.V = V

        assert K == 128, "this port assumes BK == K == 128"
        assert V % 64 == 0 and V <= 256, "V must fit one chunk of tmem"
        self.NV64 = V // 64

        # MMA tile shapes (M, N, K_contract)
        self.tile_dak = (self.BT, self.BT, self.K)  # dw @ k^T (also k @ k^T)
        self.tile_dav = (self.BT, self.BT, self.V)  # du @ v^T
        self.tile_s1 = (self.BT, self.BT, self.BT)  # dA_m @ A (k-major x k-major)
        self.tile_dkbg = (self.BT, self.K, self.BT)  # A^T @ dw (mn x mn)
        self.tile_dvb = (self.BT, self.V, self.BT)  # A^T @ du (mn x mn)
        self.tile_s2 = (self.BT, self.BT, self.BT)  # A^T @ S1 (mn x mn)
        self.tile_dkb = (self.BT, self.K, self.BT)  # dAf @ k (k x mn)

        self.cta_group = tcgen05.CtaGroup.ONE

        self.tma_warp_id = 0  # k / dw / A / beta / g producer
        self.mma_warp_id = 1
        self.vdu_warp_id = 2  # v / du producer, 1-stage so its stall is only its own
        self.a_warp_id = (4, 5, 6, 7)  # dA path, db/dg epilogue
        self.b_warp_id = (8, 9, 10, 11)  # dk / dv path
        self.threads_per_cta = 32 * 12

        self.big_stages = 2  # (k, dw, A, beta, g) per chunk
        self.vdu_stages = 1  # (v, du) per chunk; refill hides under the sandwich

        self.a_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
        self.b_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=128)
        self.ab_sync_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=256)
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.threads_per_cta
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    # ---------------------------------------------------------------------------------

    def _make_tiled_mmas(self):
        io, acc, grp = self.io_dtype, self.acc_dtype, self.cta_group
        # (64,64) k-major x k-major: DA_K, DA_V, AKK, S1
        mma_kk = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("k"),
            acc, grp, self.tile_dak[:2], tcgen05.OperandSource.SMEM,
        )
        # (64,K) mn x mn: DKBG = A^T @ dw
        mma_gmn = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("mn"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dkbg[:2], tcgen05.OperandSource.SMEM,
        )
        # (64,V) mn x mn: DVB = A^T @ du
        mma_vmn = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("mn"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dvb[:2], tcgen05.OperandSource.SMEM,
        )
        # (64,64) mn x mn: S2 = A^T @ S1
        mma_smn = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("mn"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_s2[:2], tcgen05.OperandSource.SMEM,
        )
        # (64,K) k x mn: DKB / TERM2 = dAf-family @ k
        mma_kmn = sm100_utils.make_trivial_tiled_mma(
            io, tcgen05.OperandMajorMode("k"), tcgen05.OperandMajorMode("mn"),
            acc, grp, self.tile_dkb[:2], tcgen05.OperandSource.SMEM,
        )
        return mma_kk, mma_gmn, mma_vmn, mma_smn, mma_kmn

    def _setup_attributes(self):
        mma_kk, mma_gmn, mma_vmn, mma_smn, mma_kmn = self._make_tiled_mmas()
        BT, K, V = self.BT, self.K, self.V
        io = self.io_dtype

        # k [BT,K] k-major B of DA_K; also read as k-major A (AKK) and mn-major B
        # (DKB/TERM2) through views over the same bytes. dw mirrors it (k-major A of
        # DA_K, mn-major B of DKBG). All three constructions land on the same swizzle
        # family (128B lines); bring-up verifies numerically.
        self.k_smem_layout = sm100_utils.make_smem_layout_b(
            mma_kk, self.tile_dak, io, self.big_stages
        )
        self.k_a_layout = sm100_utils.make_smem_layout_a(
            mma_kk, self.tile_dak, io, self.big_stages
        )
        self.k_mn_layout = sm100_utils.make_smem_layout_b(
            mma_kmn, self.tile_dkb, io, self.big_stages
        )
        self.dw_smem_layout = self.k_a_layout
        self.dw_mn_layout = sm100_utils.make_smem_layout_b(
            mma_gmn, self.tile_dkbg, io, self.big_stages
        )
        # v [BT,V] k-major B of DA_V; du [BT,V] k-major A of DA_V + mn-major B of DVB
        self.v_smem_layout = sm100_utils.make_smem_layout_b(
            mma_kk, self.tile_dav, io, self.vdu_stages
        )
        self.du_smem_layout = sm100_utils.make_smem_layout_a(
            mma_kk, self.tile_dav, io, self.vdu_stages
        )
        self.du_mn_layout = sm100_utils.make_smem_layout_b(
            mma_vmn, self.tile_dvb, io, self.vdu_stages
        )
        # A [BT(t), BT(a)] k-major B of S1; mn-major A of DKBG/DVB/S2 through a view
        self.a_smem_layout = sm100_utils.make_smem_layout_b(
            mma_kk, self.tile_s1, io, self.big_stages
        )
        self.a_mn_layout = sm100_utils.make_smem_layout_a(
            mma_gmn, self.tile_dkbg, io, self.big_stages
        )
        # dA_m / S1c pair: two "stages" of one member, both written ROW_MAJOR by SIMT A.
        # stage 0 feeds S1 as a k-major A; stage 1 feeds S2 as an mn-major B.
        self.dam_smem_layout = sm100_utils.make_smem_layout_a(mma_kk, self.tile_s1, io, 2)
        self.s1_mn_layout = sm100_utils.make_smem_layout_b(mma_smn, self.tile_s2, io, 2)
        self.dam_epi_layout = sm100_utils.make_smem_layout_epi(
            io, utils.LayoutEnum.ROW_MAJOR, (BT, BT), 2
        )
        # dAf / dAf^T*s pair: A operands of DKB / TERM2 (dqkwg's ds/dst pattern verbatim)
        self.dap_smem_layout = sm100_utils.make_smem_layout_a(mma_kmn, self.tile_dkb, io, 2)
        self.dap_epi_layout = sm100_utils.make_smem_layout_epi(
            io, utils.LayoutEnum.ROW_MAJOR, (BT, BT), 2
        )
        self.dapt_epi_layout = sm100_utils.make_smem_layout_epi(
            io, utils.LayoutEnum.COL_MAJOR, (BT, BT), 2
        )
        # beta / g2 vectors
        self.bg_smem_layout = cute.make_layout((BT, self.big_stages))
        # output staging: dv rows of V then (reusing the bytes) dk rows of K
        self.out_elems = BT * max(V, K)
        self.out_v_layout = cute.make_layout((BT, V, 1), stride=(V, 1, BT * V))
        self.out_v3_layout = cute.make_layout(
            (BT, 64, self.NV64), stride=(V, 1, 64)
        )
        self.out_k_layout = cute.make_layout((BT, K, 1), stride=(K, 1, BT * K))

        self.num_big_load_bytes = (
            cute.size_in_bytes(io, cute.slice_(self.k_smem_layout, (None, None, None, 0)))
            + cute.size_in_bytes(io, cute.slice_(self.dw_smem_layout, (None, None, None, 0)))
            + cute.size_in_bytes(io, cute.slice_(self.a_smem_layout, (None, None, None, 0)))
            + 2 * cute.size_in_bytes(cutlass.Float32, cute.slice_(self.bg_smem_layout, (None, 0)))
        )
        self.num_vdu_load_bytes = (
            cute.size_in_bytes(io, cute.slice_(self.v_smem_layout, (None, None, None, 0)))
            + cute.size_in_bytes(io, cute.slice_(self.du_smem_layout, (None, None, None, 0)))
        )

        # scratch slots: parity-buffered per-row partials. The Ld16x256b partition puts
        # 4 lanes on each row (each lane owning two rows, r and r+8, col pairs
        # interleaved by 8 — see dbg_coords.py), so every per-row reduction gets 4
        # slots keyed by the lane's column quad.
        self.db2_span = 256 * self.NV64  # per parity
        self.scr_db1 = 0  # [0, 512)
        self.scr_db2 = 512  # [512, 512 + 2*db2_span)
        self.scr_p3 = 512 + 2 * self.db2_span
        self.scr_size = self.scr_p3 + 512

        # tmem plan
        self.tmem_dak_offset = 0
        self.tmem_dav_offset = 64
        self.tmem_dkbg_offset = 128
        self.tmem_dvb_offset = 256
        self.tmem_s1_offset = 0
        self.tmem_s2_offset = 64
        self.tmem_akk_offset = 256
        self.tmem_dkb_offset = 128
        self.tmem_term2_offset = 256 + 64 if V >= 192 else 256 + V
        self.num_tmem_cols = 512
        assert max(256 + V, self.tmem_term2_offset + K) <= 512

    # ---------------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,  # (T, K, H, B)
        dw: cute.Tensor,  # (T, K, HV, B)
        a: cute.Tensor,  # (T, BT, HV, B)
        v: cute.Tensor,  # (T, V, HV, B)
        du: cute.Tensor,  # (T, V, HV, B)
        bet: cute.Tensor,  # (BT, NT, HV, B) fp32, staged
        g2: cute.Tensor,  # (BT, NT, HV, B) fp32, staged
        dv: cute.Tensor,  # (T, V, HV, B) out
        dk: cute.Tensor,  # (T, K, HV, B) out
        db: cute.Tensor,  # (T, HV, B) fp32 out
        dg: cute.Tensor,  # (T, HV, B) fp32 out
        nseg: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        mma_kk, mma_gmn, mma_vmn, mma_smn, mma_kmn = self._make_tiled_mmas()
        BT, K, V = self.BT, self.K, self.V
        cluster_vmnk = (1, 1, 1, 1)

        tma_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), k,
            cute.slice_(self.k_smem_layout, (None, None, None, 0)),
            self.tile_dak, mma_kk, cluster_vmnk,
        )
        tma_dw, tma_tensor_dw = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), dw,
            cute.slice_(self.dw_smem_layout, (None, None, None, 0)),
            self.tile_dak, mma_kk, cluster_vmnk,
        )
        tma_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), a,
            cute.slice_(self.a_smem_layout, (None, None, None, 0)),
            self.tile_s1, mma_kk, cluster_vmnk,
        )
        tma_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), v,
            cute.slice_(self.v_smem_layout, (None, None, None, 0)),
            self.tile_dav, mma_kk, cluster_vmnk,
        )
        tma_du, tma_tensor_du = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), du,
            cute.slice_(self.du_smem_layout, (None, None, None, 0)),
            self.tile_dav, mma_kk, cluster_vmnk,
        )
        bg_cta_v_layout = cute.slice_(cute.make_identity_layout(bet.shape), (None, 0, 0, 0))
        tma_bet, tma_tensor_bet = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), bet,
            cute.slice_(self.bg_smem_layout, (None, 0)),
            bg_cta_v_layout,
        )
        tma_g2, tma_tensor_g2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), g2,
            cute.slice_(self.bg_smem_layout, (None, 0)),
            bg_cta_v_layout,
        )
        tma_dv, tma_tensor_dv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dv,
            cute.slice_(self.out_v_layout, (None, None, 0)),
            (BT, V),
        )
        tma_dk, tma_tensor_dk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dk,
            cute.slice_(self.out_k_layout, (None, None, 0)),
            (BT, K),
        )

        B = cute.size(v, mode=[3])
        HV = cute.size(v, mode=[2])
        grid = (B * HV * nseg, 1, 1)

        swz_align, lin_align = 1024, 128

        @cute.struct
        class SharedStorage:
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.k_smem_layout)], swz_align  # type: ignore
            ]
            smem_dw: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dw_smem_layout)], swz_align  # type: ignore
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.v_smem_layout)], swz_align  # type: ignore
            ]
            smem_du: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.du_smem_layout)], swz_align  # type: ignore
            ]
            smem_a: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.a_smem_layout)], swz_align  # type: ignore
            ]
            smem_dam: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dam_smem_layout)], swz_align  # type: ignore
            ]
            smem_dap: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(self.dap_smem_layout)], swz_align  # type: ignore
            ]
            smem_out: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, self.out_elems], lin_align  # type: ignore
            ]
            smem_bet: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.bg_smem_layout)], lin_align  # type: ignore
            ]
            smem_g2: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.bg_smem_layout)], lin_align  # type: ignore
            ]
            big_full: cute.struct.MemRange[cutlass.Int64, self.big_stages * 2]  # type: ignore
            vdu_full: cute.struct.MemRange[cutlass.Int64, self.vdu_stages * 2]  # type: ignore
            dam_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            s1c_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dap_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            brd_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            daf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dkbgf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dvbf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            s1f_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            s2f_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            akkf_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            dk2f_full: cute.struct.MemRange[cutlass.Int64, 2]  # type: ignore
            tmem_holding_buf: cutlass.Int32
            scr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.scr_size], lin_align  # type: ignore
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
            )

        self.gdn_cute_wy_bwd(
            tma_k, tma_tensor_k,
            tma_dw, tma_tensor_dw,
            tma_a, tma_tensor_a,
            tma_v, tma_tensor_v,
            tma_du, tma_tensor_du,
            tma_bet, tma_tensor_bet,
            tma_g2, tma_tensor_g2,
            tma_dv, tma_tensor_dv,
            tma_dk, tma_tensor_dk,
            db,
            dg,
            nseg,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], stream=stream)

    # ---------------------------------------------------------------------------------

    @cute.kernel
    def gdn_cute_wy_bwd(
        self,
        tma_k: cute.CopyAtom, mK: cute.Tensor,
        tma_dw: cute.CopyAtom, mDW: cute.Tensor,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_v: cute.CopyAtom, mV: cute.Tensor,
        tma_du: cute.CopyAtom, mDU: cute.Tensor,
        tma_bet: cute.CopyAtom, mBET: cute.Tensor,
        tma_g2: cute.CopyAtom, mG2: cute.Tensor,
        tma_dv: cute.CopyAtom, mDV: cute.Tensor,
        tma_dk: cute.CopyAtom, mDK: cute.Tensor,
        mDB: cute.Tensor,
        mDG: cute.Tensor,
        nseg: cutlass.Int32,
    ):
        BT, K, V, NV64 = self.BT, self.K, self.V, self.NV64
        io = self.io_dtype
        f32 = self.acc_dtype
        self._setup_attributes()
        mma_kk, mma_gmn, mma_vmn, mma_smn, mma_kmn = self._make_tiled_mmas()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % 128

        if warp_idx == self.tma_warp_id:
            for atom in [tma_k, tma_dw, tma_a, tma_bet, tma_g2, tma_dv, tma_dk]:
                cpasync.prefetch_descriptor(atom)
        if warp_idx == self.vdu_warp_id:
            for atom in [tma_v, tma_du]:
                cpasync.prefetch_descriptor(atom)

        bidx, _, _ = cute.arch.block_idx()
        HV = cute.size(mV, mode=[2])
        H = cute.size(mK, mode=[2])
        T = cute.size(mK, mode=[0])
        NT = T // BT
        seg = bidx % nseg
        hv_idx = (bidx // nseg) % HV
        b_idx = bidx // (nseg * HV)
        h_idx = hv_idx // (HV // H)
        cpc = (NT + nseg - 1) // nseg
        c0 = seg * cpc
        cnt = cutlass.min(cpc, NT - c0)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sK = storage.smem_k.get_tensor(self.k_smem_layout.outer, swizzle=self.k_smem_layout.inner)
        sKa = storage.smem_k.get_tensor(self.k_a_layout.outer, swizzle=self.k_a_layout.inner)
        sKmn = storage.smem_k.get_tensor(self.k_mn_layout.outer, swizzle=self.k_mn_layout.inner)
        sDW = storage.smem_dw.get_tensor(self.dw_smem_layout.outer, swizzle=self.dw_smem_layout.inner)
        sDWmn = storage.smem_dw.get_tensor(self.dw_mn_layout.outer, swizzle=self.dw_mn_layout.inner)
        sV = storage.smem_v.get_tensor(self.v_smem_layout.outer, swizzle=self.v_smem_layout.inner)
        sDU = storage.smem_du.get_tensor(self.du_smem_layout.outer, swizzle=self.du_smem_layout.inner)
        sDUmn = storage.smem_du.get_tensor(self.du_mn_layout.outer, swizzle=self.du_mn_layout.inner)
        sA = storage.smem_a.get_tensor(self.a_smem_layout.outer, swizzle=self.a_smem_layout.inner)
        sAmn = storage.smem_a.get_tensor(self.a_mn_layout.outer, swizzle=self.a_mn_layout.inner)
        sDAM = storage.smem_dam.get_tensor(self.dam_smem_layout.outer, swizzle=self.dam_smem_layout.inner)
        sS1B = storage.smem_dam.get_tensor(self.s1_mn_layout.outer, swizzle=self.s1_mn_layout.inner)
        sDAM_epi = storage.smem_dam.get_tensor(self.dam_epi_layout.outer, swizzle=self.dam_epi_layout.inner)
        sDAP = storage.smem_dap.get_tensor(self.dap_smem_layout.outer, swizzle=self.dap_smem_layout.inner)
        sDAP_epi = storage.smem_dap.get_tensor(self.dap_epi_layout.outer, swizzle=self.dap_epi_layout.inner)
        sDAPT_epi = storage.smem_dap.get_tensor(self.dapt_epi_layout.outer, swizzle=self.dapt_epi_layout.inner)
        sOutV = storage.smem_out.get_tensor(self.out_v_layout)
        sOutV3 = storage.smem_out.get_tensor(self.out_v3_layout)
        sOutK = storage.smem_out.get_tensor(self.out_k_layout)
        sBET = storage.smem_bet.get_tensor(self.bg_smem_layout)
        sG2 = storage.smem_g2.get_tensor(self.bg_smem_layout)

        # flat coordinate views of the swizzled operand bytes, for SIMT scalar reads
        k_flat = cute.make_layout((BT, K, self.big_stages))
        sKv = cute.make_tensor(sK.iterator, cute.composition(sK.layout, k_flat))
        v_flat = cute.make_layout((BT, V, self.vdu_stages))
        sVv = cute.make_tensor(sV.iterator, cute.composition(sV.layout, v_flat))

        # p = dAf . AKK staged fp32 over the dA_m/S1 bytes (dead after S2; see docstring)
        sP = cute.make_tensor(
            cute.recast_ptr(sDAM.iterator, dtype=f32), cute.make_layout((BT, BT))
        )

        sScr = storage.scr.get_tensor(cute.make_layout(self.scr_size))

        # ---- pipelines ----
        a_threads = 32 * len(self.a_warp_id)
        b_threads = 32 * len(self.b_warp_id)
        big_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.big_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, a_threads + b_threads
            ),
            tx_count=self.num_big_load_bytes,
            barrier_storage=storage.big_full.data_ptr(),
            defer_sync=True,
        )
        vdu_pipe = pipeline.PipelineTmaMultiConsumersAsync.create(
            num_stages=self.vdu_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_async=pipeline.CooperativeGroup(pipeline.Agent.Thread, b_threads),
            tx_count=self.num_vdu_load_bytes,
            barrier_storage=storage.vdu_full.data_ptr(),
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

        dam_pipe = make_simt_to_mma_pipe(storage.dam_full.data_ptr(), a_threads)
        s1c_pipe = make_simt_to_mma_pipe(storage.s1c_full.data_ptr(), a_threads)
        dap_pipe = make_simt_to_mma_pipe(storage.dap_full.data_ptr(), a_threads)
        brd_pipe = make_simt_to_mma_pipe(storage.brd_full.data_ptr(), b_threads)
        daf_pipe = make_mma_to_simt_pipe(storage.daf_full.data_ptr(), a_threads)
        s1f_pipe = make_mma_to_simt_pipe(storage.s1f_full.data_ptr(), a_threads)
        s2f_pipe = make_mma_to_simt_pipe(storage.s2f_full.data_ptr(), a_threads)
        akkf_pipe = make_mma_to_simt_pipe(storage.akkf_full.data_ptr(), a_threads)
        dkbgf_pipe = make_mma_to_simt_pipe(storage.dkbgf_full.data_ptr(), b_threads)
        dvbf_pipe = make_mma_to_simt_pipe(storage.dvbf_full.data_ptr(), b_threads)
        dk2f_pipe = make_mma_to_simt_pipe(storage.dk2f_full.data_ptr(), b_threads)

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

        tDAK = acc_tensor(mma_kk, self.tile_dak, self.tmem_dak_offset)
        tDAV = acc_tensor(mma_kk, self.tile_dak, self.tmem_dav_offset)
        tDKBG = acc_tensor(mma_gmn, self.tile_dkbg, self.tmem_dkbg_offset)
        tDVB = acc_tensor(mma_vmn, self.tile_dvb, self.tmem_dvb_offset)
        tS1 = acc_tensor(mma_kk, self.tile_dak, self.tmem_s1_offset)
        tS2 = acc_tensor(mma_smn, self.tile_s2, self.tmem_s2_offset)
        tAKK = acc_tensor(mma_kk, self.tile_dak, self.tmem_akk_offset)
        tDKB = acc_tensor(mma_kmn, self.tile_dkb, self.tmem_dkb_offset)
        tTERM2 = acc_tensor(mma_kmn, self.tile_dkb, self.tmem_term2_offset)
        # 64-wide readout views over DVB's columns
        tDVBrd = tuple(
            acc_tensor(mma_kk, self.tile_dak, self.tmem_dvb_offset + 64 * vi)
            for vi in range(NV64)
        )

        # ---- global tiles ----
        gK = cute.local_tile(mK, (BT, K), (None, 0, h_idx, b_idx))  # (BT,K,NT)
        gDW = cute.local_tile(mDW, (BT, K), (None, 0, hv_idx, b_idx))
        gA = cute.local_tile(mA, (BT, BT), (None, 0, hv_idx, b_idx))
        gV = cute.local_tile(mV, (BT, V), (None, 0, hv_idx, b_idx))  # (BT,V,NT)
        gDU = cute.local_tile(mDU, (BT, V), (None, 0, hv_idx, b_idx))
        gBET = mBET[(None, None, hv_idx, b_idx)]  # (BT, NT)
        gG2 = mG2[(None, None, hv_idx, b_idx)]
        gDV = cute.local_tile(mDV, (BT, V), (None, 0, hv_idx, b_idx))
        gDK = cute.local_tile(mDK, (BT, K), (None, 0, hv_idx, b_idx))

        # ==========================================================================
        # TMA warp 0: k / dw / A / beta / g, 2-stage
        # ==========================================================================
        if warp_idx == self.tma_warp_id:
            thr_mma_kk = mma_kk.get_slice(0)
            tK_mma = thr_mma_kk.partition_B(gK)
            tDW_mma = thr_mma_kk.partition_A(gDW)
            tA_mma = thr_mma_kk.partition_B(gA)

            cta1 = cute.make_layout(1)
            tKs, tKg = cpasync.tma_partition(
                tma_k, 0, cta1, cute.group_modes(sK, 0, 3), cute.group_modes(tK_mma, 0, 3)
            )
            tDWs, tDWg = cpasync.tma_partition(
                tma_dw, 0, cta1, cute.group_modes(sDW, 0, 3), cute.group_modes(tDW_mma, 0, 3)
            )
            tAs, tAg = cpasync.tma_partition(
                tma_a, 0, cta1, cute.group_modes(sA, 0, 3), cute.group_modes(tA_mma, 0, 3)
            )
            tBs, tBg = cpasync.tma_partition(
                tma_bet, 0, cta1, cute.group_modes(sBET, 0, 1), cute.group_modes(gBET, 0, 1)
            )
            tGs, tGg = cpasync.tma_partition(
                tma_g2, 0, cta1, cute.group_modes(sG2, 0, 1), cute.group_modes(gG2, 0, 1)
            )

            big_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.big_stages
            )
            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                big_pipe.producer_acquire(big_producer)
                bar = big_pipe.producer_get_barrier(big_producer)
                st = big_producer.index
                cute.copy(tma_k, tKg[None, c], tKs[None, st], tma_bar_ptr=bar)
                cute.copy(tma_dw, tDWg[None, c], tDWs[None, st], tma_bar_ptr=bar)
                cute.copy(tma_a, tAg[None, c], tAs[None, st], tma_bar_ptr=bar)
                cute.copy(tma_bet, tBg[None, c], tBs[None, st], tma_bar_ptr=bar)
                cute.copy(tma_g2, tGg[None, c], tGs[None, st], tma_bar_ptr=bar)
                big_producer.advance()
            big_pipe.producer_tail(big_producer)

        # ==========================================================================
        # TMA warp 2: v / du, 1-stage (refill hides under the sandwich + pass 2)
        # ==========================================================================
        elif warp_idx == self.vdu_warp_id:
            thr_mma_kk = mma_kk.get_slice(0)
            tV_mma = thr_mma_kk.partition_B(gV)
            tDU_mma = thr_mma_kk.partition_A(gDU)

            cta1 = cute.make_layout(1)
            tVs, tVg = cpasync.tma_partition(
                tma_v, 0, cta1, cute.group_modes(sV, 0, 3), cute.group_modes(tV_mma, 0, 3)
            )
            tDUs, tDUg = cpasync.tma_partition(
                tma_du, 0, cta1, cute.group_modes(sDU, 0, 3), cute.group_modes(tDU_mma, 0, 3)
            )

            vdu_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.vdu_stages
            )
            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                vdu_pipe.producer_acquire(vdu_producer)
                bar = vdu_pipe.producer_get_barrier(vdu_producer)
                st = vdu_producer.index
                cute.copy(tma_v, tVg[None, c], tVs[None, st], tma_bar_ptr=bar)
                cute.copy(tma_du, tDUg[None, c], tDUs[None, st], tma_bar_ptr=bar)
                vdu_producer.advance()
            vdu_pipe.producer_tail(vdu_producer)

        # ==========================================================================
        # MMA warp
        # ==========================================================================
        elif warp_idx == self.mma_warp_id:
            tCrDW_a = mma_kk.make_fragment_A(sDW)
            tCrK_b = mma_kk.make_fragment_B(sK)
            tCrDU_a = mma_kk.make_fragment_A(sDU)
            tCrV_b = mma_kk.make_fragment_B(sV)
            tCrK_a = mma_kk.make_fragment_A(sKa)
            tCrDAM = mma_kk.make_fragment_A(sDAM)  # stage 0 = dA_m, stage 1 = S1c (unused)
            tCrA_b = mma_kk.make_fragment_B(sA)
            tCrA_g = mma_gmn.make_fragment_A(sAmn)
            tCrDW_mn = mma_gmn.make_fragment_B(sDWmn)
            tCrA_v = mma_vmn.make_fragment_A(sAmn)
            tCrDU_mn = mma_vmn.make_fragment_B(sDUmn)
            tCrA_s = mma_smn.make_fragment_A(sAmn)
            tCrS1_mn = mma_smn.make_fragment_B(sS1B)  # stage 1 = S1c
            tCrDAP = mma_kmn.make_fragment_A(sDAP)  # stage 0 = dAf, stage 1 = dAf^T*s
            tCrK_mn = mma_kmn.make_fragment_B(sKmn)

            big_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.big_stages
            )
            vdu_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.vdu_stages
            )
            dam_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            s1c_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dap_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            brd_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            daf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            s1f_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            s2f_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            akkf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dkbgf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dvbf_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dk2f_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            for i in cutlass.range(cnt, unroll=1):
                big_pipe.consumer_wait(big_consumer)
                st = big_consumer.index
                vdu_pipe.consumer_wait(vdu_consumer)

                # Every producer_acquire below is an acc-free wait paired with the
                # aliasing acc's *reader* release, placed at point of use.
                # DA_K = dw @ k^T (columns also hold S1)
                daf_pipe.producer_acquire(daf_producer)
                s1f_pipe.producer_acquire(s1f_producer)
                for kk in cutlass.range(cute.size(tCrDW_a, mode=[2]), unroll_full=True):
                    mma_kk.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_kk, tDAK[None, None, None, 0],
                        tCrDW_a[None, None, kk, st],
                        tCrK_b[None, None, kk, st],
                        tDAK[None, None, None, 0],
                    )
                # DA_V = du @ v^T (columns also hold S2)
                s2f_pipe.producer_acquire(s2f_producer)
                for kk in cutlass.range(cute.size(tCrDU_a, mode=[2]), unroll_full=True):
                    mma_kk.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_kk, tDAV[None, None, None, 0],
                        tCrDU_a[None, None, kk, 0],
                        tCrV_b[None, None, kk, 0],
                        tDAV[None, None, None, 0],
                    )
                daf_pipe.producer_commit(daf_producer)
                daf_producer.advance()
                # DKBG = A^T @ dw (columns also hold DKB)
                dkbgf_pipe.producer_acquire(dkbgf_producer)
                dk2f_pipe.producer_acquire(dk2f_producer)
                for kk in cutlass.range(cute.size(tCrA_g, mode=[2]), unroll_full=True):
                    mma_gmn.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_gmn, tDKBG[None, None, None, 0],
                        tCrA_g[None, None, kk, st],
                        tCrDW_mn[None, None, kk, st],
                        tDKBG[None, None, None, 0],
                    )
                dkbgf_pipe.producer_commit(dkbgf_producer)
                dkbgf_producer.advance()
                # DVB = A^T @ du (columns also hold AKK and TERM2)
                dvbf_pipe.producer_acquire(dvbf_producer)
                akkf_pipe.producer_acquire(akkf_producer)
                for kk in cutlass.range(cute.size(tCrA_v, mode=[2]), unroll_full=True):
                    mma_vmn.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_vmn, tDVB[None, None, None, 0],
                        tCrA_v[None, None, kk, st],
                        tCrDU_mn[None, None, kk, 0],
                        tDVB[None, None, None, 0],
                    )
                dvbf_pipe.producer_commit(dvbf_producer)
                dvbf_producer.advance()
                vdu_pipe.consumer_release(vdu_consumer, pipeline.PipelineOp.TCGen05Mma)
                vdu_consumer.advance()

                # S1 = dA_m @ A^T-as-loaded
                dam_pipe.consumer_wait(dam_consumer)
                for kk in cutlass.range(cute.size(tCrDAM, mode=[2]), unroll_full=True):
                    mma_kk.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_kk, tS1[None, None, None, 0],
                        tCrDAM[None, None, kk, 0],
                        tCrA_b[None, None, kk, st],
                        tS1[None, None, None, 0],
                    )
                s1f_pipe.producer_commit(s1f_producer)
                s1f_producer.advance()
                dam_pipe.consumer_release(dam_consumer)
                dam_consumer.advance()

                # AKK = k @ k^T, gated on B having read DKBG/DVB (whose columns it takes)
                brd_pipe.consumer_wait(brd_consumer)
                for kk in cutlass.range(cute.size(tCrK_a, mode=[2]), unroll_full=True):
                    mma_kk.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_kk, tAKK[None, None, None, 0],
                        tCrK_a[None, None, kk, st],
                        tCrK_b[None, None, kk, st],
                        tAKK[None, None, None, 0],
                    )
                akkf_pipe.producer_commit(akkf_producer)
                akkf_producer.advance()
                brd_pipe.consumer_release(brd_consumer)
                brd_consumer.advance()

                # S2 = A^T @ S1c
                s1c_pipe.consumer_wait(s1c_consumer)
                for kk in cutlass.range(cute.size(tCrA_s, mode=[2]), unroll_full=True):
                    mma_smn.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_smn, tS2[None, None, None, 0],
                        tCrA_s[None, None, kk, st],
                        tCrS1_mn[None, None, kk, 1],
                        tS2[None, None, None, 0],
                    )
                s2f_pipe.producer_commit(s2f_producer)
                s2f_producer.advance()
                s1c_pipe.consumer_release(s1c_consumer)
                s1c_consumer.advance()

                # DKB = dAf @ k ; TERM2 = (dAf^T * s) @ k
                dap_pipe.consumer_wait(dap_consumer)
                for kk in cutlass.range(cute.size(tCrDAP, mode=[2]), unroll_full=True):
                    mma_kmn.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_kmn, tDKB[None, None, None, 0],
                        tCrDAP[None, None, kk, 0],
                        tCrK_mn[None, None, kk, st],
                        tDKB[None, None, None, 0],
                    )
                for kk in cutlass.range(cute.size(tCrDAP, mode=[2]), unroll_full=True):
                    mma_kmn.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kk != 0))
                    cute.gemm(
                        mma_kmn, tTERM2[None, None, None, 0],
                        tCrDAP[None, None, kk, 1],
                        tCrK_mn[None, None, kk, st],
                        tTERM2[None, None, None, 0],
                    )
                dk2f_pipe.producer_commit(dk2f_producer)
                dk2f_producer.advance()
                dap_pipe.consumer_release(dap_consumer)
                dap_consumer.advance()

                big_pipe.consumer_release(big_consumer, pipeline.PipelineOp.TCGen05Mma)
                big_consumer.advance()

            daf_pipe.producer_tail(daf_producer)
            s1f_pipe.producer_tail(s1f_producer)
            s2f_pipe.producer_tail(s2f_producer)
            akkf_pipe.producer_tail(akkf_producer)
            dkbgf_pipe.producer_tail(dkbgf_producer)
            dvbf_pipe.producer_tail(dvbf_producer)
            dk2f_pipe.producer_tail(dk2f_producer)

        # ==========================================================================
        # SIMT group A (warps 4..7): dA build, sandwich casts, dAf pair, p, db/dg
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

            tDAK_2d = tDAK[((None, None), 0, 0, None)]
            tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tDAK_2d[None, None, 0])
            thr_t2r = tiled_t2r.get_slice(local_tidx)
            tTR_tDAK = thr_t2r.partition_S(tDAK_2d)
            tDAV_2d = tDAV[((None, None), 0, 0, None)]
            tTR_tDAV = thr_t2r.partition_S(tDAV_2d)
            tS1_2d = tS1[((None, None), 0, 0, None)]
            tTR_tS1 = thr_t2r.partition_S(tS1_2d)
            tS2_2d = tS2[((None, None), 0, 0, None)]
            tTR_tS2 = thr_t2r.partition_S(tS2_2d)
            tAKK_2d = tAKK[((None, None), 0, 0, None)]
            tTR_tAKK = thr_t2r.partition_S(tAKK_2d)
            coord64 = thr_t2r.partition_D(cute.make_identity_tensor((BT, BT)))
            rX = cute.make_rmem_tensor(coord64.shape, f32)
            rY = cute.make_rmem_tensor(coord64.shape, f32)
            rVAL = cute.make_rmem_tensor(coord64.shape, f32)
            rWC = cute.make_rmem_tensor(coord64.shape, f32)

            # column gathers (beta[nj], g[nj]); row values are per-lane scalars
            sG2_col = cute.make_tensor(
                sG2.iterator,
                cute.make_layout((BT, BT, self.big_stages), stride=(0, 1, BT)),
            )
            sBET_col = cute.make_tensor(
                sBET.iterator,
                cute.make_layout((BT, BT, self.big_stages), stride=(0, 1, BT)),
            )
            tGc_s = thr_t2r.partition_D(sG2_col)
            tGc = cute.make_rmem_tensor(cute.slice_(tGc_s.shape, (None, None, None, 0)), f32)
            tBc_s = thr_t2r.partition_D(sBET_col)
            tBc = cute.make_rmem_tensor(cute.slice_(tBc_s.shape, (None, None, None, 0)), f32)

            r2s_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            r2s_t_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), io
            )
            tiled_r2s = cute.make_tiled_copy_D(r2s_atom, tiled_t2r)
            tiled_r2s_t = cute.make_tiled_copy_D(r2s_t_atom, tiled_t2r)
            thr_r2s = tiled_r2s.get_slice(local_tidx)
            thr_r2s_t = tiled_r2s_t.get_slice(local_tidx)
            tRS_sDAM = thr_r2s.partition_D(sDAM_epi)
            tRS_sDAP = thr_r2s.partition_D(sDAP_epi)
            tRS_sDAPT = thr_r2s_t.partition_D(sDAPT_epi)
            tRS_rIO = cute.make_rmem_tensor(
                cute.slice_(tRS_sDAM.shape, (None, None, None, 0)), io
            )
            tRS_rDAP = cute.make_rmem_tensor(
                cute.slice_(tRS_sDAP.shape, (None, None, None, 0)), io
            )
            tRS_rDAPT = cute.make_rmem_tensor(
                cute.slice_(tRS_sDAPT.shape, (None, None, None, 0)), io
            )

            big_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.big_stages
            )
            daf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            s1f_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            s2f_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            akkf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dam_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            s1c_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            dap_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # lane geometry (dbg_coords.py): rows {row_lo, row_lo+8}, element j is in
            # row_lo when (j & 2) == 0; the lane's column quad is col0 >> 1.
            row_lo, col00 = coord64[0]
            cq = col00 >> 1
            pl2 = cute.make_rmem_tensor((2,), f32)
            ph2 = cute.make_rmem_tensor((2,), f32)

            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                par = i % 2
                big_pipe.consumer_wait(big_consumer)
                st = big_consumer.index
                qcrd = (None, None, None, st)
                g_lo = sG2[(row_lo, st)]
                g_hi = sG2[(row_lo + 8, st)]
                b_lo = sBET[(row_lo, st)]
                b_hi = sBET[(row_lo + 8, st)]
                # column gathers + the exp2s hoisted off the daf-gated chain: wc = b*2^g
                cute.copy(f32_cp_atom, tGc_s[qcrd], tGc)
                cute.copy(f32_cp_atom, tBc_s[qcrd], tBc)
                for j in cutlass.range(cute.size(rWC), unroll_full=True, vectorize=True):
                    rWC[j] = tBc[j] * cute.math.exp2(tGc[j], fastmath=True)

                # DA_K / DA_V -> dA_m (masked, column-scaled, bf16)
                daf_pipe.consumer_wait(daf_consumer)
                cute.copy(tiled_t2r, tTR_tDAK[None, None, None, 0], rX)
                cute.copy(tiled_t2r, tTR_tDAV[None, None, None, 0], rY)
                cute.arch.fence_view_async_tmem_load()
                daf_pipe.consumer_release(daf_consumer)
                daf_consumer.advance()
                for j in cutlass.range(cute.size(rX), unroll_full=True, vectorize=True):
                    rVAL[j] = rX[j] * rWC[j] + rY[j] * tBc[j]
                for j in cutlass.range(cute.size(rX), unroll_full=True):
                    mi, nj = coord64[j]
                    if mi <= nj:
                        rVAL[j] = cutlass.Float32(0.0)
                for j in cutlass.range(cute.size(rX), unroll_full=True, vectorize=True):
                    tRS_rIO[j] = rVAL[j].to(io)
                dam_pipe.producer_acquire(dam_producer)
                cute.copy(tiled_r2s, tRS_rIO, tRS_sDAM[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.a_sync_barrier.arrive_and_wait()
                dam_pipe.producer_commit(dam_producer)
                dam_producer.advance()

                # S1 -> bf16
                s1f_pipe.consumer_wait(s1f_consumer)
                cute.copy(tiled_t2r, tTR_tS1[None, None, None, 0], rX)
                cute.arch.fence_view_async_tmem_load()
                s1f_pipe.consumer_release(s1f_consumer)
                s1f_consumer.advance()
                for j in cutlass.range(cute.size(rX), unroll_full=True, vectorize=True):
                    tRS_rIO[j] = rX[j].to(io)
                s1c_pipe.producer_acquire(s1c_producer)
                cute.copy(tiled_r2s, tRS_rIO, tRS_sDAM[None, None, None, 1])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.a_sync_barrier.arrive_and_wait()
                s1c_pipe.producer_commit(s1c_producer)
                s1c_producer.advance()

                # S2 -> dAf (gated, masked, negated, bf16) + beta-scaled transpose
                s2f_pipe.consumer_wait(s2f_consumer)
                cute.copy(tiled_t2r, tTR_tS2[None, None, None, 0], rX)
                cute.arch.fence_view_async_tmem_load()
                s2f_pipe.consumer_release(s2f_consumer)
                s2f_consumer.advance()
                for j in cutlass.range(cute.size(rX), unroll_full=True):
                    g_mi = g_lo if (j & 2) == 0 else g_hi
                    d = cute.math.exp2(g_mi - tGc[j], fastmath=True)
                    rVAL[j] = -(rX[j] * d)
                for j in cutlass.range(cute.size(rX), unroll_full=True):
                    mi, nj = coord64[j]
                    if mi <= nj:
                        rVAL[j] = cutlass.Float32(0.0)
                for j in cutlass.range(cute.size(rX), unroll_full=True):
                    b_mi = b_lo if (j & 2) == 0 else b_hi
                    bb = rVAL[j].to(io)
                    tRS_rDAP[j] = bb
                    tRS_rDAPT[j] = (bb.to(f32) * b_mi).to(io)
                dap_pipe.producer_acquire(dap_producer)
                cute.copy(tiled_r2s, tRS_rDAP, tRS_sDAP[None, None, None, 0])
                cute.copy(tiled_r2s_t, tRS_rDAPT, tRS_sDAPT[None, None, None, 1])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.a_sync_barrier.arrive_and_wait()
                dap_pipe.producer_commit(dap_producer)
                dap_producer.advance()

                # AKK -> p = dAf . AKK, staged + row partials
                akkf_pipe.consumer_wait(akkf_consumer)
                cute.copy(tiled_t2r, tTR_tAKK[None, None, None, 0], rY)
                cute.arch.fence_view_async_tmem_load()
                akkf_pipe.consumer_release(akkf_consumer)
                akkf_consumer.advance()
                for e in cutlass.range(2, unroll_full=True):
                    pl2[e] = cutlass.Float32(0.0)
                    ph2[e] = cutlass.Float32(0.0)
                for j in cutlass.range(cute.size(rY), unroll_full=True):
                    mi, nj = coord64[j]
                    pv = tRS_rDAP[j].to(f32) * rY[j]
                    sP[(mi, nj)] = pv
                    if (j & 2) == 0:
                        pl2[j & 1] += pv
                    else:
                        ph2[j & 1] += pv
                sScr[self.scr_p3 + par * 256 + 4 * row_lo + cq] = pl2[0] + pl2[1]
                sScr[self.scr_p3 + par * 256 + 4 * (row_lo + 8) + cq] = ph2[0] + ph2[1]
                cute.arch.fence_proxy("async.shared", space="cta")
                self.a_sync_barrier.arrive_and_wait()
                self.ab_sync_barrier.arrive_and_wait()

                # db / dg epilogue, off the inter-chunk critical path (all of A's acc
                # releases and operand commits are already behind us)
                if local_tidx < BT:
                    r = local_tidx
                    db1 = cutlass.Float32(0.0)
                    p3 = cutlass.Float32(0.0)
                    for s in cutlass.range(4, unroll_full=True):
                        db1 += sScr[self.scr_db1 + par * 256 + 4 * r + s]
                        p3 += sScr[self.scr_p3 + par * 256 + 4 * r + s]
                    db2 = cutlass.Float32(0.0)
                    for s in cutlass.range(4 * NV64, unroll_full=True):
                        vi2, s4 = s // 4, s % 4
                        db2 += sScr[
                            self.scr_db2 + par * self.db2_span + 256 * vi2 + 4 * r + s4
                        ]
                    mDB[(c * BT + r, hv_idx, b_idx)] = db1 + db2 + p3
                else:
                    r = local_tidx - BT
                    db1 = cutlass.Float32(0.0)
                    p3 = cutlass.Float32(0.0)
                    for s in cutlass.range(4, unroll_full=True):
                        db1 += sScr[self.scr_db1 + par * 256 + 4 * r + s]
                        p3 += sScr[self.scr_p3 + par * 256 + 4 * r + s]
                    b_r = sBET[(r, st)]
                    colacc = cute.make_rmem_tensor((4,), f32)
                    for e in cutlass.range(4, unroll_full=True):
                        colacc[e] = cutlass.Float32(0.0)
                    for jo in cutlass.range(BT // 4, unroll=4):
                        for ji in cutlass.range(4, unroll_full=True):
                            i2 = jo * 4 + ji
                            colacc[ji] += sP[(i2, r)] * sBET[(i2, st)]
                    col = (colacc[0] + colacc[1]) + (colacc[2] + colacc[3])
                    mDG[(c * BT + r, hv_idx, b_idx)] = b_r * (db1 + p3) - col

                # sP is aliased onto the dA_m/S1 operand bytes, and the dg branch above is the
                # only reader of it - the db branch (threads < BT) never touches sP, so it
                # falls straight through to the next chunk, takes dam_pipe's producer_acquire
                # (which waits on the MMA consumer, not on us) and its r2s copy overwrites sP
                # while the dg half is still column-summing it. The next a_sync_barrier sits
                # *after* that copy, so nothing else stops it. Hence this barrier: no thread of
                # group A may start chunk c+1's dA_m store until every thread has finished
                # reading chunk c's p. Only bites when a CTA owns more than one chunk, which
                # happens at production sizes (chunks-per-CTA is 32 at B=16/T=8192) and never
                # at the small shapes a unit test can afford.
                self.a_sync_barrier.arrive_and_wait()

                big_pipe.consumer_release(big_consumer, pipeline.PipelineOp.AsyncThread)
                big_consumer.advance()

        # ==========================================================================
        # SIMT group B (warps 8..11): dk1/dv readouts + rowsums, dk combine + stores
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

            tDKBG_2d = tDKBG[((None, None), 0, 0, None)]
            tiled_t2r_128 = tcgen05.make_tmem_copy(t2r_atom, tDKBG_2d[None, None, 0])
            thr_t2r_128 = tiled_t2r_128.get_slice(local_tidx)
            tTR_tDKBG = thr_t2r_128.partition_S(tDKBG_2d)
            tDKB_2d = tDKB[((None, None), 0, 0, None)]
            tTR_tDKB = thr_t2r_128.partition_S(tDKB_2d)
            tTERM2_2d = tTERM2[((None, None), 0, 0, None)]
            tTR_tTERM2 = thr_t2r_128.partition_S(tTERM2_2d)
            coord128 = thr_t2r_128.partition_D(cute.make_identity_tensor((BT, K)))
            rF = cute.make_rmem_tensor(coord128.shape, f32)
            tDKpart = cute.make_rmem_tensor(rF.shape, f32)

            tDVB0_2d = tDVBrd[0][((None, None), 0, 0, None)]
            tiled_t2r_64 = tcgen05.make_tmem_copy(t2r_atom, tDVB0_2d[None, None, 0])
            thr_t2r_64 = tiled_t2r_64.get_slice(local_tidx)
            coordB64 = thr_t2r_64.partition_D(cute.make_identity_tensor((BT, 64)))
            rF64 = cute.make_rmem_tensor(coordB64.shape, f32)

            r2s_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), io
            )
            tiled_r2s_dv = cute.make_tiled_copy_D(r2s_atom, tiled_t2r_64)
            thr_r2s_dv = tiled_r2s_dv.get_slice(local_tidx)
            tRS_sOutV = thr_r2s_dv.partition_D(sOutV3)
            tRS_rOutV = cute.make_rmem_tensor(
                cute.slice_(tRS_sOutV.shape, (None, None, None, 0)), io
            )
            tiled_r2s_dk = cute.make_tiled_copy_D(r2s_atom, tiled_t2r_128)
            thr_r2s_dk = tiled_r2s_dk.get_slice(local_tidx)
            tRS_sOutK = thr_r2s_dk.partition_D(sOutK)
            tRS_rOutK = cute.make_rmem_tensor(
                cute.slice_(tRS_sOutK.shape, (None, None, None, 0)), io
            )

            bSG_sOutV, bSG_gDV = cpasync.tma_partition(
                tma_dv, 0, cute.make_layout(1),
                cute.group_modes(sOutV, 0, 2), cute.group_modes(gDV, 0, 2),
            )
            bSG_sOutK, bSG_gDK = cpasync.tma_partition(
                tma_dk, 0, cute.make_layout(1),
                cute.group_modes(sOutK, 0, 2), cute.group_modes(gDK, 0, 2),
            )
            tma_store_b = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, b_threads),
            )

            big_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.big_stages
            )
            vdu_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.vdu_stages
            )
            dkbgf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dvbf_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            dk2f_consumer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            brd_producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            # lane geometry (dbg_coords.py): rows {row_lo, row_lo+8}, element j is in
            # row_lo when (j & 2) == 0; the lane's column quad is col0 >> 1.
            row_lo, col00 = coord128[0]
            cq = col00 >> 1
            row_lo64, col0064 = coordB64[0]
            cq64 = col0064 >> 1
            pl2 = cute.make_rmem_tensor((2,), f32)
            ph2 = cute.make_rmem_tensor((2,), f32)

            for i in cutlass.range(cnt, unroll=1):
                c = c0 + i
                par = i % 2
                big_pipe.consumer_wait(big_consumer)
                st = big_consumer.index
                vdu_pipe.consumer_wait(vdu_consumer)
                g_lo = sG2[(row_lo, st)]
                g_hi = sG2[(row_lo + 8, st)]
                b_lo = sBET[(row_lo, st)]
                b_hi = sBET[(row_lo + 8, st)]
                b_lo64 = sBET[(row_lo64, st)]
                b_hi64 = sBET[(row_lo64 + 8, st)]
                ge_lo = cute.math.exp2(g_lo, fastmath=True)
                ge_hi = cute.math.exp2(g_hi, fastmath=True)

                # DKBG -> dk1 (held) + db/dg row partials
                dkbgf_pipe.consumer_wait(dkbgf_consumer)
                cute.copy(tiled_t2r_128, tTR_tDKBG[None, None, None, 0], rF)
                cute.arch.fence_view_async_tmem_load()
                dkbgf_pipe.consumer_release(dkbgf_consumer)
                dkbgf_consumer.advance()
                for e in cutlass.range(2, unroll_full=True):
                    pl2[e] = cutlass.Float32(0.0)
                    ph2[e] = cutlass.Float32(0.0)
                for j in cutlass.range(cute.size(rF), unroll_full=True):
                    tt, cc = coord128[j]
                    kv = rF[j] * sKv[(tt, cc, st)].to(f32)
                    if (j & 2) == 0:
                        tDKpart[j] = rF[j] * (ge_lo * b_lo)
                        pl2[j & 1] += kv
                    else:
                        tDKpart[j] = rF[j] * (ge_hi * b_hi)
                        ph2[j & 1] += kv
                sScr[self.scr_db1 + par * 256 + 4 * row_lo + cq] = ge_lo * (pl2[0] + pl2[1])
                sScr[self.scr_db1 + par * 256 + 4 * (row_lo + 8) + cq] = ge_hi * (
                    ph2[0] + ph2[1]
                )

                # DVB -> dv (staged) + db row partials, 64 columns at a time
                dvbf_pipe.consumer_wait(dvbf_consumer)
                for vi in cutlass.range_constexpr(NV64):
                    tDVB_2d = tDVBrd[vi][((None, None), 0, 0, None)]
                    tTR_tDVB = thr_t2r_64.partition_S(tDVB_2d)
                    cute.copy(tiled_t2r_64, tTR_tDVB[None, None, None, 0], rF64)
                    cute.arch.fence_view_async_tmem_load()
                    for e in cutlass.range(2, unroll_full=True):
                        pl2[e] = cutlass.Float32(0.0)
                        ph2[e] = cutlass.Float32(0.0)
                    for j in cutlass.range(cute.size(rF64), unroll_full=True):
                        tt, cc = coordB64[j]
                        vv = rF64[j] * sVv[(tt, cc + 64 * vi, 0)].to(f32)
                        if (j & 2) == 0:
                            tRS_rOutV[j] = (rF64[j] * b_lo64).to(io)
                            pl2[j & 1] += vv
                        else:
                            tRS_rOutV[j] = (rF64[j] * b_hi64).to(io)
                            ph2[j & 1] += vv
                    cute.copy(tiled_r2s_dv, tRS_rOutV, tRS_sOutV[None, None, None, vi])
                    sScr[
                        self.scr_db2 + par * self.db2_span + 256 * vi + 4 * row_lo64 + cq64
                    ] = pl2[0] + pl2[1]
                    sScr[
                        self.scr_db2 + par * self.db2_span + 256 * vi
                        + 4 * (row_lo64 + 8) + cq64
                    ] = ph2[0] + ph2[1]
                dvbf_pipe.consumer_release(dvbf_consumer)
                dvbf_consumer.advance()
                brd_pipe.producer_acquire(brd_producer)
                brd_pipe.producer_commit(brd_producer)
                brd_producer.advance()
                vdu_pipe.consumer_release(vdu_consumer, pipeline.PipelineOp.AsyncThread)
                vdu_consumer.advance()

                cute.arch.fence_proxy("async.shared", space="cta")
                self.b_sync_barrier.arrive_and_wait()
                if warp_idx == self.b_warp_id[0]:
                    cute.copy(tma_dv, bSG_sOutV[None, 0], bSG_gDV[None, c])
                    tma_store_b.producer_commit()
                    tma_store_b.producer_acquire()
                # second rendezvous so b0's store-acquire retires before any warp
                # overwrites sOut with dk (dqkwg's dw->dk pattern)
                self.b_sync_barrier.arrive_and_wait()

                # DKB + TERM2 -> dk = dk1 + DKB*s + TERM2
                dk2f_pipe.consumer_wait(dk2f_consumer)
                cute.copy(tiled_t2r_128, tTR_tDKB[None, None, None, 0], rF)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(rF), unroll_full=True):
                    b_mi = b_lo if (j & 2) == 0 else b_hi
                    tDKpart[j] += rF[j] * b_mi
                cute.copy(tiled_t2r_128, tTR_tTERM2[None, None, None, 0], rF)
                cute.arch.fence_view_async_tmem_load()
                dk2f_pipe.consumer_release(dk2f_consumer)
                dk2f_consumer.advance()
                for j in cutlass.range(cute.size(rF), unroll_full=True, vectorize=True):
                    tRS_rOutK[j] = (tDKpart[j] + rF[j]).to(io)
                cute.copy(tiled_r2s_dk, tRS_rOutK, tRS_sOutK[None, None, None, 0])
                cute.arch.fence_proxy("async.shared", space="cta")
                self.b_sync_barrier.arrive_and_wait()
                if warp_idx == self.b_warp_id[0]:
                    cute.copy(tma_dk, bSG_sOutK[None, 0], bSG_gDK[None, c])
                    tma_store_b.producer_commit()
                    tma_store_b.producer_acquire()

                big_pipe.consumer_release(big_consumer, pipeline.PipelineOp.AsyncThread)
                big_consumer.advance()
                self.ab_sync_barrier.arrive_and_wait()

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
    import os

    target = int(os.environ.get("GDN_WYBWD_CTAS", "1024"))
    nseg = min(NT, max(1, -(-target // (B * HV))))
    cpc = -(-NT // nseg)
    return -(-NT // cpc)


def gdn_cute_wy_bwd_call(
    k: torch.Tensor,  # [B,T,H,K]
    v: torch.Tensor,  # [B,T,HV,V]
    beta: torch.Tensor,  # [B,T,HV] fp32
    A: torch.Tensor,  # [B,T,HV,BT]
    dw: torch.Tensor,  # [B,T,HV,K]
    du: torch.Tensor,  # [B,T,HV,V]
    g2: torch.Tensor,  # [B,T,HV] fp32
):
    key = _call_key(k, v, A, dw, du)
    ent = _CALL_CACHE.get(key)
    if ent is None:
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        NT = T // 64
        assert T % 64 == 0

        dk = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
        dv = torch.empty_like(v)
        db = torch.empty(B, T, HV, device=k.device, dtype=torch.float32)
        dg = torch.empty(B, T, HV, device=k.device, dtype=torch.float32)
        bc = torch.empty(B, HV, NT, 64, device=k.device, dtype=torch.float32)
        g2c = torch.empty(B, HV, NT, 64, device=k.device, dtype=torch.float32)

        io_dtype = cutlass.BFloat16 if k.dtype == torch.bfloat16 else cutlass.Float16
        compile_key = (io_dtype, K, V, HV, H)

        ck = _cute_view(k, (1, 3, 2, 0), (0, 2, 3))
        cdw = _cute_view(dw, (1, 3, 2, 0), (0, 2, 3))
        ca = _cute_view(A, (1, 3, 2, 0), (0, 2, 3))
        cv = _cute_view(v, (1, 3, 2, 0), (0, 2, 3))
        cdu = _cute_view(du, (1, 3, 2, 0), (0, 2, 3))
        cbc = _cute_view(bc, (3, 2, 1, 0), (1, 2, 3))
        cg2 = _cute_view(g2c, (3, 2, 1, 0), (1, 2, 3))
        cdv = _cute_view(dv, (1, 3, 2, 0), (0, 2, 3))
        cdk = _cute_view(dk, (1, 3, 2, 0), (0, 2, 3))
        cdb = _cute_view(db, (1, 2, 0), (0, 2))
        cdg = _cute_view(dg, (1, 2, 0), (0, 2))

        nseg = _pick_nseg(B, HV, NT)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            kernel_obj = GdnBwdWyKernel(io_dtype, K, V)
            compiled = cute.compile(
                kernel_obj, ck, cdw, ca, cv, cdu, cbc, cg2, cdv, cdk, cdb, cdg,
                cutlass.Int32(nseg), stream,
            )
            _COMPILE_CACHE[compile_key] = compiled
        args = (ck, cdw, ca, cv, cdu, cbc, cg2, cdv, cdk, cdb, cdg,
                cutlass.Int32(nseg), stream)
        if len(_CALL_CACHE) >= 64:
            _CALL_CACHE.clear()
        ent = (compiled, args, (_spec(dk), _spec(dv), _spec(db), _spec(dg)), bc, g2c)
        _CALL_CACHE[key] = ent

    compiled, args, out_specs, bc, g2c = ent
    ck, cdw, ca, cv, cdu, _, _, cdv, cdk, cdb, cdg, _, _ = args
    # Fresh outputs per call - see the package docstring on why they are not cache-owned.
    dk, dv, db, dg = _alloc(out_specs, k.device)
    _retarget(cdk, dk)
    _retarget(cdv, dv)
    _retarget(cdb, db)
    _retarget(cdg, dg)
    _retarget(ck, k)
    _retarget(cdw, dw)
    _retarget(ca, A)
    _retarget(cv, v)
    _retarget(cdu, du)
    B, HV, NT, BTc = bc.shape
    bc.view(B, HV, NT * BTc).copy_(beta.transpose(1, 2))
    g2c.view(B, HV, NT * BTc).copy_(g2.transpose(1, 2))
    compiled(*args)
    return dk, dv, db, dg


def prepare_wy_repr_bwd_cute(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens=None,
    chunk_indices=None,
    **kw,
):
    """Drop-in for fla's prepare_wy_repr_bwd on the shapes this idea targets; falls back
    to fla otherwise so the stage table never breaks."""
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    supported = (
        A.shape[-1] == 64 and T % 64 == 0 and K == 128 and V % 64 == 0 and V <= 256
        and g is not None and g.dtype == torch.float32 and cu_seqlens is None
        and k.dtype in (torch.bfloat16, torch.float16)
        and dw.dtype == k.dtype and du.dtype == k.dtype
    )
    if not supported:
        from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd

        return prepare_wy_repr_bwd(
            k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=g,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, **kw,
        )

    dk, dv, db, dg = gdn_cute_wy_bwd_call(
        k, v, beta.float(), A, dw, du, g
    )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg
