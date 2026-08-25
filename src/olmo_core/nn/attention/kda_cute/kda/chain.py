"""The KDA chain: which kernel runs at each stage, and in what order.

This is fla 0.5.2's own decomposition of `chunk_kda`, launch for launch, with five stages
replaced. Keeping fla's structure is deliberate — it is what makes a stage-by-stage
comparison meaningful, and it is why the parity tests can hold to fla's own tolerances.

Forward:
    gate+cumsum   fla   kda_gate_chunk_cumsum (fused activation, or the plain cumsum)
    intra+solve   fla   token_parallel + inter_solve_fused, preceded by OUR zero-fill of
                        Aqk's upper triangles (fla leaves them uninitialized and masks at
                        load; our scan contracts the full tile). The zero-fill replaces a
                        masked_fill that read the whole 268MB tile to write an eighth of it.
    w/u prep      fla   recompute_w_u_fwd
    scan + o      OURS  fused state scan and readout, state resident in registers

Backward (fla's recompute path):
    1 recompute   fla   recompute_w_u_fwd
    2 rescan      OURS  B1: forward re-scan
    3 dAv         fla   chunk_kda_bwd_dAv
    4 dhu         OURS  B2a: reverse dh/dv scan, dh^T resident in registers
    5 wy_dqkg     OURS  B2b: full-K restructure of fla's fused kernel — same math, one K
                        slab instead of NK, so v_new/do/dv are read once
    6 intra       OURS  tcgen05 off-diagonals + SIMT diagonals, with the dg reverse-cumsum
                        and the bf16 dq/dk cast folded into its epilogue
    7 dg_cumsum   —     identity: stage 6 already emitted it

Every one of ours falls back to fla's kernel off its supported shape or below the CTA
floor, so this table is always safe to call; `is_supported` exists so a caller can know
whether that happened instead of reading a silent 1.00x.
"""

from __future__ import annotations

import torch

from .._common.support import log_once


def gate_cumsum(g, A_log, dt_bias, chunk_size, lower_bound=None):
    """The chunk-local cumsum, with the gate activation fused in when there is one.

    With use_gate_in_kernel the op receives w_g(x) raw and owes
    -exp(A_log)*softplus(g + dt_bias) before the cumsum. Doing that in eager torch is four
    passes over an fp32 [B,T,HV,K] tensor — at prod8192 that is ~2ms and ~2GiB of saved
    activations PER LAYER, against a measured 1.9ms for the entire fused stage. fla fuses
    it into the cumsum this chain launches anyway, so the fused form is free.
    """
    from fla.ops.utils.constant import RCP_LN2

    if A_log is None:
        from fla.ops.utils import chunk_local_cumsum

        return chunk_local_cumsum(g=g, scale=RCP_LN2, chunk_size=chunk_size)

    from fla.ops.kda.gate import kda_gate_chunk_cumsum

    return kda_gate_chunk_cumsum(
        g=g, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size,
        lower_bound=lower_bound,
    )


def forward(q, k, v, g2, beta, h0, scale, chunk_size):
    """Returns (o, ht, Aqk, Akk) — o, the final state, and the backward's two residuals."""
    from fla.ops.kda.wy_fast import recompute_w_u_fwd

    from ._kernels import fwd_intra_triton, fwd_state

    Aqk, Akk = fwd_intra_triton.chunk_kda_fwd_intra_zerofill(
        q, k, g2, beta, float(scale), chunk_size
    )
    w, u, qg, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Akk, q=q, gk=g2)
    # aqk_prezeroed: the zero-fill above already cleared the upper triangles, so the scan
    # skips the masked_fill it would otherwise do.
    o, ht = fwd_state.kda_cute_fwd_call(
        qg, kg, w, u, Aqk, g2, h0, float(scale), aqk_prezeroed=True
    )
    return o, ht, Aqk, Akk


def backward(q, k, v, g2, beta, Aqk, Akk, h0, do, dht, scale, chunk_size):
    """Returns (dq, dk, dv, dbeta, dg, dh0). dg is w.r.t. the pre-cumsum decay."""
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv
    from fla.ops.kda.wy_fast import recompute_w_u_fwd

    from ._kernels import bwd_dhu, bwd_intra, bwd_scan, bwd_wy

    H, HV = q.shape[2], v.shape[2]

    w, u, qg, kg = recompute_w_u_fwd(q=q, k=k, v=v, beta=beta, A=Akk, gk=g2)
    # dq0 is B1's raw do@h^T when the dq fusion is on. It is off: the fusion measured
    # negative, so wy_dqkg computes dq itself and B1 returns None here.
    h, v_new, _dq0 = bwd_scan.kda_rescan_b1(kg, w, u, g2, h0, do, chunk_size)
    dAqk, dv = chunk_kda_bwd_dAv(
        q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale, chunk_size=chunk_size,
    )
    dh, dh0, dv = bwd_dhu.kda_dhu_b2a(qg, kg, w, g2, h0, dht, do, dv, scale, chunk_size)
    dq, dk, dv, db, dg, dAkk = bwd_wy.chunk_kda_bwd_wy_dqkg_wide(
        q=q, k=k, v=v, v_new=v_new, g=g2, beta=beta, A=Akk, h=h,
        do=do, dh=dh, dv=dv, scale=scale, chunk_size=chunk_size,
    )
    # fold_dg: intra emits dg already chunk-reverse-cumsum'd, so fla's dg_cumsum stage
    # disappears. emit_bf16: dq/dk come out in q's dtype, making the wrapper's casts no-ops
    # — but only at HV == H, since the GVA reduction below must sum in fp32.
    dq, dk, db, dg = bwd_intra.chunk_kda_bwd_intra_cutedsl(
        q=q, k=k, g=g2, beta=beta, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dk=dk, db=db, dg=dg, chunk_size=chunk_size,
        fold_dg=True, emit_bf16=HV == H,
    )
    # The GVA reduction sits where fla puts it: after intra, before the (folded) cumsum.
    if HV > H:
        G = HV // H
        dq = dq.view(*dq.shape[:2], H, G, dq.shape[-1]).sum(dim=3)
        dk = dk.view(*dk.shape[:2], H, G, dk.shape[-1]).sum(dim=3)
    return dq, dk, dv, db, dg, dh0


def gate_backward(g_org, A_log, dt_bias, dg, lower_bound=None):
    """Close the gate: dg w.r.t. the raw input, plus dA_log and ddt_bias.

    Applied to the already-reverse-cumsum'd dg, exactly where fla applies it.
    """
    from fla.ops.kda.gate import kda_gate_bwd

    return kda_gate_bwd(
        g=g_org, A_log=A_log, dt_bias=dt_bias, dyg=dg, lower_bound=lower_bound
    )


_ZEROS: dict = {}


def zero_state(B, HV, K, V, device) -> torch.Tensor:
    """A shared read-only zero initial state.

    The scan kernels read the initial state unconditionally — there is no null-h0 branch —
    and production almost always passes initial_state=None. Nothing writes this buffer
    (dh0 is a separate output), so one per (shape, device) is shared across every layer
    rather than allocated per call: 134 MiB at prod8192, once.
    """
    key = (B, HV, K, V, device)
    z = _ZEROS.get(key)
    if z is None:
        z = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)
        _ZEROS[key] = z
        log_once(f"kernel-fun kda: allocated a shared zero state {tuple(z.shape)}")
    return z
