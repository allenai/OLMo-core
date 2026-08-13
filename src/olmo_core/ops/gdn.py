"""
The chunked gated delta rule, with a CuTe DSL backend and a
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_ fallback.

:func:`chunk_gated_delta_rule` is the entry point. It takes the same arguments as fla's
``chunk_gated_delta_rule`` (the subset :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet`
uses) and routes to one of:

- ``GDNBackend.cute`` — the CuTe DSL kernels in :mod:`olmo_core.kernels.gdn_cute`, which
  replace fla's forward outright and four of its seven backward stages. Measured on a B300 at
  ``B=16, T=8192, H=HV=16, K=128, V=256`` in bf16: forward 1.62x, forward+backward 1.27x.
- ``GDNBackend.fla`` — fla's Triton kernels, unchanged.

``GDNBackend.auto`` (the default) picks ``cute`` when the CuTe DSL is installed, the GPU is
Blackwell or newer, and the shapes are in the supported envelope (see
:func:`gdn_cute_unsupported_reason`); otherwise it falls back to ``fla``.

Note that even on the ``cute`` path fla is still required: the chunk-local WY representation in
the forward and three of the seven backward stages are fla's Triton kernels. The CuTe port
covers the state-scan and chunk-parallel stages, which is where the time is.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from olmo_core.config import StrEnum
from olmo_core.kernels.gdn_cute import has_gdn_cute

log = logging.getLogger(__name__)

__all__ = [
    "GDNBackend",
    "GDN_CUTE_CHUNK_SIZE",
    "GDN_CUTE_HEAD_K_DIM",
    "GDN_CUTE_MAX_HEAD_V_DIM",
    "chunk_gated_delta_rule",
    "gdn_cute_unsupported_reason",
]


#: The only chunk size the CuTe kernels implement. Baked into the tile shapes, not a tunable.
GDN_CUTE_CHUNK_SIZE = 64

#: The only ``head_k_dim`` the CuTe kernels implement end-to-end. The forward and the two state
#: scans also handle 64, but ``chunk_bwd_dqkwg`` and ``prepare_wy_repr_bwd`` assume ``BK == K ==
#: 128``, so anything else would silently run half the backward on fla. Gate rather than
#: half-fall-back.
GDN_CUTE_HEAD_K_DIM = 128

#: The largest ``head_v_dim`` the CuTe kernels implement: ``prepare_wy_repr_bwd`` needs the
#: whole V extent to fit one chunk of TMEM.
GDN_CUTE_MAX_HEAD_V_DIM = 256


class GDNBackend(StrEnum):
    """
    An enumeration of the available gated delta rule kernel backends.
    """

    auto = "auto"
    """
    Use :data:`cute` where it's available and supports the shapes, otherwise :data:`fla`.
    """

    cute = "cute"
    """
    The CuTe DSL kernels in :mod:`olmo_core.kernels.gdn_cute`. Requires a Blackwell
    (``sm_100+``) GPU and ``nvidia-cutlass-dsl``, and raises if the shapes aren't supported.
    """

    fla = "fla"
    """
    fla's Triton kernels.
    """


def gdn_cute_unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = GDN_CUTE_CHUNK_SIZE,
) -> Optional[str]:
    """
    Check whether the CuTe gated delta rule kernels can handle these inputs.

    :param q: Queries of shape ``(batch_size, seq_len, n_v_heads, head_k_dim)``, i.e. already
        expanded to the value-head count if GVA is in play.
    :param k: Keys, same shape as ``q``.
    :param v: Values of shape ``(batch_size, seq_len, n_v_heads, head_v_dim)``.
    :param cu_seqlens: Cumulative sequence lengths for variable-length inputs.
    :param chunk_size: The chunk size for the chunked scan.

    :returns: ``None`` if the inputs are supported, otherwise a human-readable reason why not.
    """
    if not has_gdn_cute(q.device):
        return "requires nvidia-cutlass-dsl and a Blackwell (sm_100+) GPU"
    if cu_seqlens is not None:
        return "variable-length inputs (cu_seqlens) are not implemented"
    if chunk_size != GDN_CUTE_CHUNK_SIZE:
        return f"only chunk_size={GDN_CUTE_CHUNK_SIZE} is implemented, got {chunk_size}"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"q/k/v must be bf16 or fp16, got {q.dtype}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return f"q/k/v must share a dtype, got {q.dtype}/{k.dtype}/{v.dtype}"

    T, n_heads, head_k_dim = q.shape[1], q.shape[2], q.shape[3]
    n_v_heads, head_v_dim = v.shape[2], v.shape[3]
    if n_heads != n_v_heads:
        return (
            f"q/k must be expanded to the value-head count before the kernel, "
            f"got {n_heads} vs {n_v_heads}"
        )
    if head_k_dim != GDN_CUTE_HEAD_K_DIM:
        return f"only head_k_dim={GDN_CUTE_HEAD_K_DIM} is implemented, got {head_k_dim}"
    if head_v_dim % 64 != 0 or head_v_dim > GDN_CUTE_MAX_HEAD_V_DIM:
        return (
            f"head_v_dim must be a multiple of 64 and at most "
            f"{GDN_CUTE_MAX_HEAD_V_DIM}, got {head_v_dim}"
        )
    if T % GDN_CUTE_CHUNK_SIZE != 0:
        return f"seq_len must be a multiple of {GDN_CUTE_CHUNK_SIZE}, got {T}"
    return None


_STAGES: Optional[Dict[str, Any]] = None


def _stages() -> Dict[str, Any]:
    """
    The backward, as a table of stages keyed by fla's own names.

    Built lazily so this module imports without fla or the CuTe DSL present. Four stages are
    CuTe and three are still fla's; each CuTe wrapper falls back to fla's kernel on shapes it
    doesn't cover (small grids included — the serial scans only beat fla once the grid fills
    the GPU), so the table is always safe to call.
    """
    global _STAGES
    if _STAGES is None:
        from fla.ops.common.chunk_o import chunk_bwd_dv_local
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
        from fla.ops.utils import chunk_local_cumsum

        from olmo_core.kernels.gdn_cute.kernel_dhu import (
            chunk_gated_delta_rule_bwd_dhu_cute,
        )
        from olmo_core.kernels.gdn_cute.kernel_dqkwg import chunk_bwd_dqkwg_cute
        from olmo_core.kernels.gdn_cute.kernel_fwdh import (
            chunk_gated_delta_rule_fwd_h_cute,
        )
        from olmo_core.kernels.gdn_cute.kernel_wy_bwd import prepare_wy_repr_bwd_cute

        _STAGES = {
            "recompute_w_u": recompute_w_u_fwd,  # fla
            "fwd_h": chunk_gated_delta_rule_fwd_h_cute,
            "dv_local": chunk_bwd_dv_local,  # fla
            "bwd_dhu": chunk_gated_delta_rule_bwd_dhu_cute,
            "dqkwg": chunk_bwd_dqkwg_cute,
            "wy_bwd": prepare_wy_repr_bwd_cute,
            "cumsum": chunk_local_cumsum,  # fla
        }
    return _STAGES


def _gdn_bwd(q, k, v, g2, beta, A, h0, do, dht, scale, chunk_size):
    """
    The backward, stage by stage. Mirrors fla's ``chunk_gated_delta_rule_bwd`` exactly.

    Kept flat and explicit rather than factored: the sequence of stages, the tensors that flow
    between them, and the three separate writes to ``dv`` should all be visible at a glance.
    """
    S = _stages()

    # 1. w, u from the saved bf16 A.
    w, u = S["recompute_w_u"](k=k, v=v, beta=beta, A=A, g=g2)

    # 2. Re-run the forward state scan for the h checkpoints and v_new.
    h, v_new, _ = S["fwd_h"](
        k=k,
        w=w,
        u=u,
        g=g2,
        initial_state=h0,
        output_final_state=False,
        chunk_size=chunk_size,
    )

    # 3. Intra-chunk part of dv.
    dv = S["dv_local"](q=q, k=k, g=g2, do=do, scale=scale, chunk_size=chunk_size)

    # 4. Reverse state scan: dh checkpoints, dh0, and dv completed with the state part.
    dh, dh0, dv = S["bwd_dhu"](
        q=q,
        k=k,
        w=w,
        g=g2,
        h0=h0,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        chunk_size=chunk_size,
    )

    # 5. dq, dk, dw, dg — chunk-parallel, reads h and dh back.
    dq, dk, dw, dg = S["dqkwg"](
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g2,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        chunk_size=chunk_size,
    )

    # 6. The WY-representation backward; overwrites dv, contributes a second dk and dg.
    dk2, dv, dbeta, dg2 = S["wy_bwd"](k=k, v=v, beta=beta, g=g2, A=A, dw=dw, du=dv)
    dk = dk + dk2
    dg = dg + dg2

    # 7. dg through a reverse chunk-local cumsum.
    dg = S["cumsum"](dg, chunk_size=chunk_size, reverse=True)
    return dq, dk, dv, dbeta, dg, dh0


_ZERO_STATE: Dict[Any, torch.Tensor] = {}


def _zero_initial_state(
    batch_size: int, n_v_heads: int, head_k_dim: int, head_v_dim: int, device: torch.device
) -> torch.Tensor:
    """
    A zero initial recurrent state, cached per shape.

    The CuTe forward requires a real ``h0`` tensor. It's read-only on every path that sees it
    (the forward, the ``fwd_h`` recompute and ``bwd_dhu``, which writes ``dh0`` separately), so
    one buffer can be shared rather than re-zeroing tens of MB per layer per step.
    """
    key = (batch_size, n_v_heads, head_k_dim, head_v_dim, device)
    state = _ZERO_STATE.get(key)
    if state is None:
        if len(_ZERO_STATE) >= 8:  # distinct shapes are few; this is a leak backstop
            _ZERO_STATE.clear()
        state = torch.zeros(
            batch_size,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            device=device,
            dtype=torch.float32,
        )
        _ZERO_STATE[key] = state
    return state


class ChunkGatedDeltaRuleCute(torch.autograd.Function):
    """
    The autograd wrapper around the CuTe gated delta rule kernels.

    Mirrors the structure of fla's ``ChunkGatedDeltaRuleFunction``, including where the optional
    L2 norm of ``q``/``k`` is applied, so the two backends are numerically comparable. Call
    :func:`chunk_gated_delta_rule` rather than this directly.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        chunk_size: int,
    ):
        from fla.modules.l2norm import l2norm_fwd

        from olmo_core.kernels.gdn_cute.kernel_fwd import gdn_cute_fwd_with_residuals

        # The kernels index gmem through CuTe views built from shape/stride, so a non-contiguous
        # input would be read wrong rather than rejected. fla's `input_guard` does the same.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # g and beta are per-head scalars, so the cast is cheap; fp32 is what the kernels were
        # validated against, and it keeps the forward and backward consistent with each other.
        g_dtype, beta_dtype = g.dtype, beta.dtype
        g = g.contiguous().float()
        beta = beta.contiguous().float()

        q_rstd = k_rstd = None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        h0 = initial_state
        if h0 is None:
            h0 = _zero_initial_state(v.shape[0], v.shape[2], q.shape[3], v.shape[3], q.device)
        else:
            h0 = h0.contiguous().float()

        # `g2` is the log2-space chunk-local cumsum and `A` the bf16 triangular inverse — the
        # residuals the backward recomputes from, exactly as fla's does.
        o, ht, g2, A = gdn_cute_fwd_with_residuals(q, k, v, g, beta, h0, scale, chunk_size)

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g2, beta, A, h0)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.g_dtype = g_dtype
        ctx.beta_dtype = beta_dtype
        # None rather than a dtype doubles as "there was no initial state", so no dh0 is
        # returned for it.
        ctx.initial_state_dtype = None if initial_state is None else initial_state.dtype
        return o, (ht if output_final_state else None)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        from fla.modules.l2norm import l2norm_bwd

        q, q_rstd, k, k_rstd, v, g2, beta, A, h0 = ctx.saved_tensors

        dq, dk, dv, dbeta, dg, dh0 = _gdn_bwd(
            q=q,
            k=k,
            v=v,
            g2=g2,
            beta=beta,
            A=A,
            h0=h0,
            do=do.contiguous(),
            dht=None if dht is None else dht.contiguous().float(),
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(ctx.g_dtype),
            dbeta.to(ctx.beta_dtype),
            None,  # scale
            None if ctx.initial_state_dtype is None else dh0.to(ctx.initial_state_dtype),
            None,  # output_final_state
            None,  # use_qk_l2norm_in_kernel
            None,  # chunk_size
        )


_WARNED_REASONS: set = set()


# Not traceable: the CuTe path re-points memref descriptors with ctypes pointer writes, and the
# fla path disables the compiler for itself anyway. Graph-break here rather than at some
# arbitrary depth inside.
@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    backend: GDNBackend = GDNBackend.auto,
    chunk_size: int = GDN_CUTE_CHUNK_SIZE,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    The chunked gated delta rule, dispatched to the best available backend.

    :param q: Queries of shape ``(batch_size, seq_len, n_v_heads, head_k_dim)``, already
        expanded to the value-head count if GVA is in play.
    :param k: Keys, same shape as ``q``.
    :param v: Values of shape ``(batch_size, seq_len, n_v_heads, head_v_dim)``.
    :param g: The log-space forget gate of shape ``(batch_size, seq_len, n_v_heads)``.
    :param beta: The delta rule step size of shape ``(batch_size, seq_len, n_v_heads)``.
    :param scale: The scale applied to ``q``. Defaults to ``head_k_dim ** -0.5``.
    :param initial_state: An optional initial recurrent state of shape
        ``(batch_size, n_v_heads, head_k_dim, head_v_dim)``.
    :param output_final_state: Whether to return the final recurrent state.
    :param use_qk_l2norm_in_kernel: L2-normalize ``q`` and ``k`` as part of the op.
    :param cu_seqlens: Cumulative sequence lengths for variable-length inputs. Only the
        :data:`~GDNBackend.fla` backend implements this.
    :param backend: Which kernel backend to use.
    :param chunk_size: The chunk size for the chunked scan.

    :returns: The output of shape ``(batch_size, seq_len, n_v_heads, head_v_dim)``, and the
        final recurrent state if ``output_final_state`` else ``None``.

    :raises RuntimeError: If ``backend`` names a backend that can't run these inputs.
    """
    # Imported here, not at module scope: `olmo_core.nn.attention.recurrent` imports this
    # module, so a top-level import of anything under `olmo_core.nn.attention` makes the cycle
    # resolvable only in one import order.
    from olmo_core.nn.attention.flash_linear_attn_api import (
        dispatch_chunk_gated_delta_rule,
        has_fla,
    )

    assert has_fla(), "the gated delta rule requires flash-linear-attention (fla)"

    if backend != GDNBackend.fla:
        reason = gdn_cute_unsupported_reason(q, k, v, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        if reason is None:
            if scale is None:
                scale = q.shape[-1] ** -0.5
            return ChunkGatedDeltaRuleCute.apply(  # type: ignore[return-value]
                q,
                k,
                v,
                g,
                beta,
                scale,
                initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                chunk_size,
            )
        if backend == GDNBackend.cute:
            raise RuntimeError(f"the CuTe gated delta rule kernels cannot be used here: {reason}")
        # auto: fall back, but say so once per distinct reason. Silently running the slow path
        # is how a "1.27x faster" branch turns out to have been fla all along.
        if reason not in _WARNED_REASONS:
            _WARNED_REASONS.add(reason)
            log.warning("Falling back to the fla gated delta rule kernels: %s", reason)

    return dispatch_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
