"""The forward intra+solve stage: fla's two kernels, plus a write-only zero fill.

fla's token_parallel kernel stores only j <= t of each in-chunk tile and its solve stores
only the lower blocks, because fla's own consumer masks at load time. Our scan contracts
the full 64x64 tile, so the upper triangles have to be zero — and reading the whole 268MB
tile to overwrite an eighth of it (a masked_fill) costs more than writing the zeros. This
kernel writes them, nothing else, and lets Akk allocate as torch.empty.

That is the entire local contribution to this stage, and it is deliberate: three rebuilds
of the intra+solve pair (a CuTe SIMT monolith at 6.75ms, a Triton 3D-cube form at 8.11, a
Triton 2D-row form at 4.90) all LOST to fla's own kernels at 3.04ms. The forward pair does
not carry the tiny-CTA disease the backward's did — token_parallel and the solve run within
~1.6-1.9x of their floors. Site-shaped outputs like these A matrices want tensor cores for
the contraction, which is what fla already does; see kernels/kda/ideas/004-fwd-block for
the full postmortem.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit(do_not_specialize=['T'])
def kda_fwd_zero_upper_kernel(
    Aqk,
    Akk,
    T,
    HV: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
):
    # write-only replacement for the chain's Aqk masked_fill (which READS the whole
    # 268MB tile to write ~1/8 of it) and Akk's torch.zeros memset. Aqk needs its
    # full in-chunk upper triangle (token_parallel stores only j <= t); Akk needs
    # only the upper 16x16 BLOCKS (the solve's diagonal stores carry exact zeros
    # above their diagonals, and it writes every lower block).
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    bos = i_b * T
    if i_t * BT >= T:
        return
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    o_i = tl.arange(0, BC)
    o_j = tl.arange(0, BT)
    b_z = tl.zeros([BC, BT], dtype=Aqk.dtype.element_ty)
    for i_i in tl.static_range(NC):
        o_c = i_t * BT + i_i * BC + o_i
        m_c = o_c < T
        p_q = Aqk + o_c[:, None] * (HV * BT) + o_j[None, :]
        p_k = Akk + o_c[:, None] * (HV * BT) + o_j[None, :]
        m_up = m_c[:, None] & (o_j[None, :] > (i_i * BC + o_i)[:, None])
        m_blk = m_c[:, None] & (o_j[None, :] >= (i_i + 1) * BC)
        tl.store(p_q, b_z, mask=m_up)
        tl.store(p_k, b_z, mask=m_blk)


def chunk_kda_fwd_intra_zerofill(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
):
    """fla's token_parallel + solve VERBATIM, with the masked_fill and Akk memset
    replaced by the write-only zero kernel — the phase-1 keeper path (the three
    rebuilt-diag attempts all lost to fla's own kernels, see NOTES history 002)."""
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_kernel_inter_solve_fused
    from fla.ops.kda.chunk_intra_token_parallel import (
        chunk_kda_fwd_intra_token_parallel,
    )

    B, T, H, K = k.shape
    HV = g.shape[2]
    BT = chunk_size
    BC = 16
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    kda_fwd_zero_upper_kernel[(NT, B * HV)](
        Aqk=Aqk, Akk=Akk, T=T, HV=HV, BT=BT, BC=BC, NC=NC, num_warps=2
    )
    Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
        q=q, k=k, gk=g, beta=beta, Aqk=Aqk, Akk=Akkd, scale=scale,
        cu_seqlens=None, chunk_size=BT, sub_chunk_size=BC,
    )
    chunk_kda_fwd_kernel_inter_solve_fused[(NT, B * HV)](
        q=q, k=k, g=g, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale,
        cu_seqlens=None, chunk_indices=None,
        T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, NC=NC, USE_SAFE_GATE=False,
    )
    return Aqk, Akk
