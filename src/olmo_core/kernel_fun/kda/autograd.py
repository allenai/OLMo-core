"""The autograd Function behind `chunk_kda`.

Residuals are fla's own set — q, k, v, g2, beta, Aqk, Akk, h0 (+ the l2norm rstds) — so
peak activation memory matches fla's, which is a requirement for a drop-in and not an
accident. Two deliberate differences, both fla's own behaviour rather than ours:

  - with the fused gate, g2 is NOT saved. The backward recomputes it from the raw (and
    half-size) g, exactly as fla's recompute path does. Saving it would cost 1 GiB per
    layer at prod8192 for a tensor a single kernel launch reproduces.
  - dq/dk come back already in q's dtype when the intra kernel can emit them that way
    (HV == H), so the casts below are no-ops rather than two extra launches.

Everything here runs under `torch.compiler.disable` at the entry point, so nothing in this
file is traced. See ops.py for why.
"""

from __future__ import annotations

import torch

from . import chain


def _make_fn():
    """Built lazily: fla's decorators import triton, and importing this package must stay
    cheap on a machine that will never call it."""
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

    class ChunkKDAFunction(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(ctx, q, k, v, g, beta, h0, scale, chunk_size, use_qk_l2norm,
                    A_log, dt_bias, lower_bound, h0_was_none):
            q_rstd = k_rstd = None
            if use_qk_l2norm:
                from fla.modules.l2norm import l2norm_fwd

                q, q_rstd = l2norm_fwd(q)
                k, k_rstd = l2norm_fwd(k)

            g2 = chain.gate_cumsum(g, A_log, dt_bias, chunk_size, lower_bound)
            o, ht, Aqk, Akk = chain.forward(q, k, v, g2, beta, h0, scale, chunk_size)

            ctx.save_for_backward(
                q, q_rstd, k, k_rstd, v,
                None if A_log is not None else g2,
                beta, Aqk, Akk, h0,
                g if A_log is not None else None, A_log, dt_bias,
            )
            ctx.scale = scale
            ctx.chunk_size = chunk_size
            ctx.lower_bound = lower_bound
            ctx.h0_was_none = h0_was_none
            return o, ht

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do, dht):
            (q, q_rstd, k, k_rstd, v, g2, beta, Aqk, Akk, h0,
             g_org, A_log, dt_bias) = ctx.saved_tensors
            if g2 is None:
                g2 = chain.gate_cumsum(
                    g_org, A_log, dt_bias, ctx.chunk_size, ctx.lower_bound
                )
            dq, dk, dv, db, dg, dh0 = chain.backward(
                q=q, k=k, v=v, g2=g2, beta=beta, Aqk=Aqk, Akk=Akk, h0=h0,
                do=do.contiguous(),
                # dht is None when the caller never asked for a final state and nothing
                # differentiated it; the dhu stage needs a tensor, and a zero one is the
                # honest gradient. Shared and read-only, like the zero initial state.
                dht=dht.contiguous() if dht is not None else chain.zero_state(
                    *h0.shape, h0.device
                ),
                scale=ctx.scale, chunk_size=ctx.chunk_size,
            )
            if q_rstd is not None:
                from fla.modules.l2norm import l2norm_bwd

                dq = l2norm_bwd(q, q_rstd, dq)
                dk = l2norm_bwd(k, k_rstd, dk)

            dA = dbias = None
            if A_log is not None:
                dg, dA, dbias = chain.gate_backward(
                    g_org, A_log, dt_bias, dg, ctx.lower_bound
                )
            g_ref = g_org if g_org is not None else g2
            return (
                dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dg.to(g_ref.dtype), db.to(beta.dtype),
                None if ctx.h0_was_none else dh0,
                None, None, None, dA, dbias, None, None,
            )

    return ChunkKDAFunction


_FN = None


def apply(q, k, v, g, beta, h0, scale, chunk_size, use_qk_l2norm,
          A_log, dt_bias, lower_bound, h0_was_none):
    global _FN
    if _FN is None:
        _FN = _make_fn()
    return _FN.apply(
        q, k, v, g, beta, h0, scale, chunk_size, use_qk_l2norm,
        A_log, dt_bias, lower_bound, h0_was_none,
    )
