"""A ``chunk_kda``-compatible entry point backed by the cute-kda kernels.

The kernels come from the ``kernel-fun-2`` harness (kda idea ``002-gdn-parity``) and are
copied verbatim into this package:

- Forward: fla's chunk-local cumsum + intra stage (Triton, unchanged), then a fused CuTe
  scan+readout (:mod:`.kernel_fwd`) that keeps the recurrent state in registers.
- Backward: fla's own six-stage decomposition of ``chunk_kda_bwd``, with the two serial
  scans ported to CuTe (:mod:`.kernel_fwdh`, :mod:`.kernel_dhu`) and the two dominant
  chunk-parallel stages restructured in Triton (:mod:`.kernel_intra_bwd`,
  :mod:`.kernel_wy_dqkg`). The remaining stages stay fla's.

At the kda harness's production shape (B16 x T8192 x H16, K128/V256, chunk 64 on B300)
this measured 1.26x on the forward and 1.47x on the training step over fla's monolith.

The CuTe kernels target Blackwell (tcgen05); :func:`cute_kda_supported` gates on device
capability and shape so callers can fall back to ``fla.ops.kda.chunk_kda``.

Set ``OLMO_CUTE_KDA_ALLFLA=1`` to keep this staged decomposition but force every stage
back to fla's kernels — the backward is then bit-identical to fla's monolith, which
isolates any numerics question to the CuTe/Triton stage swaps.
``OLMO_CUTE_KDA_STAGES=a,b`` swaps in only the named stages (bisect).
``OLMO_CUTE_KDA_CHECK=1`` asserts each backward stage's outputs are finite so a bad
kernel fails at the guilty stage — set it on smoke runs.
"""

from __future__ import annotations

import os
from functools import cache

import torch
import torch.nn.functional as F

__all__ = ["cute_chunk_kda", "cute_kda_supported"]


@cache
def _has_cute() -> bool:
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except Exception:
        return False
    return True


@torch.compiler.disable
def cute_kda_supported(
    q: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
) -> bool:
    """Whether :func:`cute_chunk_kda` supports this call, else use fla's ``chunk_kda``.

    The CuTe forward/scan kernels implement the fixed-length, chunk-size-64 recompute
    path on Blackwell (sm100+); anything else must go to fla.
    """
    if cu_seqlens is not None:
        return False
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 10:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    T, K, V = q.shape[1], q.shape[-1], v.shape[-1]
    if chunk_size != 64 or T % 64 != 0:
        return False
    if K not in (64, 128) or V % 64 != 0:
        return False
    return _has_cute()


def _fla_stages() -> dict:
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_wy_dqkg_fused
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    from fla.ops.kda.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum

    return {
        # fla's chunk_kda_bwd (recompute path, fixed-length), launch for launch.
        "recompute_w_u": recompute_w_u_fwd,
        "fwd_h": chunk_gated_delta_rule_fwd_h,
        "dAv": chunk_kda_bwd_dAv,
        "bwd_dhu": chunk_gated_delta_rule_bwd_dhu,
        "wy_dqkg": chunk_kda_bwd_wy_dqkg_fused,
        "bwd_intra": chunk_kda_bwd_intra,
        "cumsum_rev": chunk_local_cumsum,
    }


_STAGES: dict | None = None


def _stages() -> dict:
    """The backward stage table, built lazily so importing this module stays cheap.

    Each CuTe wrapper falls back to fla's kernel on shapes it does not support (small
    grids below its ``_MIN_CTAS``), so the table is always safe to call.
    """
    global _STAGES
    if _STAGES is None:
        _STAGES = _fla_stages()
        if os.environ.get("OLMO_CUTE_KDA_ALLFLA", "0") != "1":
            from .kernel_dhu import chunk_gated_delta_rule_bwd_dhu_cute
            from .kernel_fwdh import chunk_gated_delta_rule_fwd_h_cute
            from .kernel_intra_bwd import chunk_kda_bwd_intra_wide
            from .kernel_wy_dqkg import chunk_kda_bwd_wy_dqkg_wide

            swaps = {
                "fwd_h": chunk_gated_delta_rule_fwd_h_cute,
                "bwd_dhu": chunk_gated_delta_rule_bwd_dhu_cute,
                "bwd_intra": chunk_kda_bwd_intra_wide,
                "wy_dqkg": chunk_kda_bwd_wy_dqkg_wide,
            }
            # Debug bisect knob: comma-separated stage names to swap in (default: all).
            only = os.environ.get("OLMO_CUTE_KDA_STAGES")
            if only is not None:
                swaps = {k: v for k, v in swaps.items() if k in only.split(",")}
            _STAGES.update(swaps)
    return _STAGES


def _check_finite(stage: str, **tensors: torch.Tensor | None) -> None:
    """OLMO_CUTE_KDA_CHECK=1: assert stage outputs are finite, so a bad kernel fails at
    the guilty stage instead of surfacing steps later as a nan loss. Syncs per stage —
    debug/smoke runs only."""
    for name, t in tensors.items():
        if t is not None and not torch.isfinite(t).all():
            raise RuntimeError(
                f"cute-kda backward stage '{stage}' produced non-finite '{name}' "
                f"(shape {tuple(t.shape)}); rerun with OLMO_CUTE_KDA_STAGES/ALLFLA to bisect"
            )


@torch.compiler.disable  # keeps compiled autograd off the cute/ctypes host code too
def _kda_bwd(q, k, v, g2, beta, Aqk, Akk, h0, do, dht, scale, chunk_size):
    """fla's ``chunk_kda_bwd`` (recompute path, fixed-length) as explicit stages.

    Argument names and call order follow ``fla/ops/kda/chunk_bwd.py::chunk_kda_bwd``
    exactly; with ``OLMO_CUTE_KDA_ALLFLA=1`` this is bit-identical to that function.
    """
    S = _stages()
    check = os.environ.get("OLMO_CUTE_KDA_CHECK", "0") == "1"

    w, u, qg, kg = S["recompute_w_u"](k=k, v=v, beta=beta, A=Akk, gk=g2, q=q)
    if check:
        _check_finite("recompute_w_u", w=w, u=u, qg=qg, kg=kg)
    h, v_new, _ = S["fwd_h"](
        k=kg,
        w=w,
        u=u,
        gk=g2,
        initial_state=h0,
        output_final_state=False,
        chunk_size=chunk_size,
    )
    if check:
        _check_finite("fwd_h", h=h, v_new=v_new)
    dAqk, dv = S["dAv"](
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        chunk_size=chunk_size,
    )
    if check:
        _check_finite("dAv", dAqk=dAqk, dv=dv)
    dh, dh0, dv = S["bwd_dhu"](
        q=qg,
        k=kg,
        w=w,
        gk=g2,
        h0=h0,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        chunk_size=chunk_size,
    )
    if check:
        _check_finite("bwd_dhu", dh=dh, dh0=dh0, dv=dv)
    dq, dk, dv, db, dg, dAkk = S["wy_dqkg"](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g2,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        chunk_size=chunk_size,
    )
    if check:
        _check_finite("wy_dqkg", dq=dq, dk=dk, dv=dv, db=db, dg=dg, dAkk=dAkk)
    dq, dk, db, dg = S["bwd_intra"](
        q=q,
        k=k,
        g=g2,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        chunk_size=chunk_size,
    )
    if check:
        _check_finite("bwd_intra", dq=dq, dk=dk, db=db, dg=dg)
    dg = S["cumsum_rev"](dg, chunk_size=chunk_size, reverse=True)
    H, HV = q.shape[2], v.shape[2]
    if HV > H:
        G = HV // H
        dq = dq.view(*dq.shape[:2], H, G, dq.shape[-1]).sum(dim=3)
        dk = dk.view(*dk.shape[:2], H, G, dk.shape[-1]).sum(dim=3)
    return dq, dk, dv, db, dg, dh0


_FN: type[torch.autograd.Function] | None = None


def _make_autograd_fn() -> type[torch.autograd.Function]:
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

    class ChunkKDACuteFunction(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(ctx, q, k, v, g, beta, h0, scale, chunk_size, use_qk_l2norm):
            # In-op q/k norm delegates to fla's l2norm kernels, exactly as fla's own
            # chunk_kda wrapper does, so dq/dk are gradients w.r.t. the raw q/k.
            q_rstd = k_rstd = None
            if use_qk_l2norm:
                from fla.modules.l2norm import l2norm_fwd

                q, q_rstd = l2norm_fwd(q)
                k, k_rstd = l2norm_fwd(k)
            if scale is None:
                scale = q.shape[-1] ** -0.5
            # The CuTe forward reads the initial state unconditionally.
            ctx.h0_was_none = h0 is None
            if h0 is None:
                B, HV, K, V = q.shape[0], v.shape[2], q.shape[-1], v.shape[-1]
                h0 = torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
            with torch.no_grad():
                from . import kernel_fwd

                o, ht, g2, Aqk, Akk = kernel_fwd.kda_cute_fwd_with_residuals(
                    q, k, v, g, beta, h0, scale, chunk_size
                )
            ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g2, beta, Aqk, Akk, h0)
            ctx.scale = scale
            ctx.chunk_size = chunk_size
            return o, ht

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do, dht):
            q, q_rstd, k, k_rstd, v, g2, beta, Aqk, Akk, h0 = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0 = _kda_bwd(
                q=q,
                k=k,
                v=v,
                g2=g2,
                beta=beta,
                Aqk=Aqk,
                Akk=Akk,
                h0=h0,
                do=do.contiguous(),
                dht=dht.contiguous() if dht is not None else None,
                scale=ctx.scale,
                chunk_size=ctx.chunk_size,
            )
            if q_rstd is not None:
                from fla.modules.l2norm import l2norm_bwd

                dq = l2norm_bwd(q, q_rstd, dq)
                dk = l2norm_bwd(k, k_rstd, dk)
            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                dg.to(g2.dtype),
                db.to(beta.dtype),
                None if ctx.h0_was_none else dh0,
                None,
                None,
                None,
            )

    return ChunkKDACuteFunction


# Never let dynamo trace the cute host path: it drives the kernels through ctypes
# pointer pokes and per-layout call caches, and under tracing torch.cuda.current_stream()
# proxies to a device-agnostic torch.Stream with no .cuda_stream. In a compiled block this
# takes a graph break instead, same as fla's own Triton wrappers.
@torch.compiler.disable
def cute_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in for ``fla.ops.kda.chunk_kda`` on the shapes :func:`cute_kda_supported`
    accepts (fixed-length, chunk 64, Blackwell).

    ``use_gate_in_kernel`` matches fla's semantics — ``g`` is the raw gate input and the
    decay is ``-exp(A_log) * softplus(g + dt_bias)`` — but here the transform runs as
    (differentiable) torch ops in fp32 rather than fused into the cumsum kernel, so
    ``A_log``/``dt_bias`` gradients come from autograd.
    """
    global _FN
    if _FN is None:
        _FN = _make_autograd_fn()
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True"
        HV, K = v.shape[2], q.shape[-1]
        g = g.float()
        if dt_bias is not None:
            g = g + dt_bias.float().view(1, 1, HV, K)
        g = -A_log.float().exp().view(1, 1, HV, 1) * F.softplus(g)
    o, ht = _FN.apply(q, k, v, g, beta, initial_state, scale, chunk_size, use_qk_l2norm_in_kernel)
    return o, (ht if output_final_state else None)
