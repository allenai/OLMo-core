"""
SonicMoE adapter for the non-expert-parallel MoE-v2 path.

The routed-expert computation is exposed as two opaque :func:`torch.library.custom_op` kernels,
``olmo_core::sonic_moe_fwd`` and ``olmo_core::sonic_moe_bwd``, joined by an autograd rule. That
lets ``torch.compile`` trace straight through the MoE block with no graph break: Dynamo sees a
single node in the forward and a single node in the backward graph, and Inductor keeps fusing the
code on either side of them. The kernel bodies replay exactly what SonicMoE's own autograd
functions do (``_UpProjection`` and ``_DownProjection`` in ``sonicmoe.functional``), so numerics
are unchanged relative to calling ``moe_general_routing_inputs`` directly.

Weight layout: this adapter treats the first half of :attr:`RoutedExperts.w_up_gate` as the
SwiGLU *gate* projection and the second half as the *up* projection, which is the reverse of the
grouped-MM backend. Fresh runs are unaffected since both halves are identically initialized, but
a checkpoint trained with one backend must not be resumed with the other without swapping halves.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .routed_experts import RoutedExperts

log = logging.getLogger(__name__)


def _prepare_sonic_inputs(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: RoutedExperts,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build Sonic's general-routing inputs and strided weight views (used by tests)."""
    num_tokens, top_k = expert_indices.shape
    token_indices = torch.arange(
        num_tokens,
        device=x.device,
        dtype=torch.int32,
    ).repeat_interleave(top_k)
    flat_expert_indices = expert_indices.reshape(-1).to(dtype=torch.int32).contiguous()
    router_scores = expert_weights.reshape(-1).to(dtype=torch.float32).contiguous()

    # Sonic's concatenated SwiGLU layout is [gate, up]. For fresh Sonic runs we deliberately
    # assign that meaning to the two identically initialized halves of OLMo's existing
    # ``w_up_gate`` parameter. No data is reordered: these permutes only expose Sonic's required
    # [2I, H, E] and [H, I, E] strided views.
    w1 = experts.w_up_gate.permute(1, 2, 0)
    w2 = experts.w_down.permute(2, 1, 0)
    return token_indices, flat_expert_indices, router_scores, w1, w2


_QUACK_CONFIGS_PATCHED = False

ALLOW_BROKEN_2CTA_ENV_VAR = "OLMO_SONIC_ALLOW_BROKEN_2CTA_CONFIGS"
"""
Set to ``1`` to skip :func:`_patch_quack_sm100_configs` (for reproducing the upstream bug only).
"""


def _patch_quack_sm100_configs() -> None:
    """
    Remove the Blackwell GEMM configs whose fused-SwiGLU epilogue is silently wrong.

    QuACK releases up to and including 0.6.4 corrupt the post-activation output of the gated GEMM
    epilogue when the autotuner selects a ``tile_m=128, cluster_m=2`` config on SM100 (sonic-moe
    issue #63, fixed upstream in QuACK PR #133 on 2026-08-21, unreleased as of QuACK 0.6.4).
    Whether a given shape is affected depends on which config the autotuner happens to pick, so
    the failure is shape-dependent and silent. Dropping those configs from the candidate list is
    the same workaround QuACK PR #131 proposed; it costs only the speed of those configs.

    Sonic's import-time patch already points QuACK's gated autotuners at the module attribute
    ``quack.gemm_config._get_sm100_configs``, so wrapping that attribute after importing
    ``sonicmoe.functional`` filters both the forward (SwiGLU) and backward (dSwiGLU) GEMMs.
    """
    global _QUACK_CONFIGS_PATCHED
    if _QUACK_CONFIGS_PATCHED:
        return
    _QUACK_CONFIGS_PATCHED = True
    if os.environ.get(ALLOW_BROKEN_2CTA_ENV_VAR) == "1":
        log.warning(
            "%s=1: leaving QuACK's broken SM100 tile_m=128/cluster_m=2 gated configs enabled",
            ALLOW_BROKEN_2CTA_ENV_VAR,
        )
        return

    import quack.gemm_config as gemm_config
    import sonicmoe.functional  # noqa: F401  # applies Sonic's own config patches first

    original = gemm_config._get_sm100_configs

    def _filtered_sm100_configs(*args, **kwargs):
        configs = original(*args, **kwargs)
        kept = [c for c in configs if not (c.tile_m == 128 and c.cluster_m == 2)]
        if len(kept) != len(configs):
            log.info(
                "Filtered %d of %d SM100 GEMM configs (tile_m=128, cluster_m=2) with the broken "
                "gated epilogue (sonic-moe #63)",
                len(configs) - len(kept),
                len(configs),
            )
        return kept

    gemm_config._get_sm100_configs = _filtered_sm100_configs


def _require_sonic() -> None:
    try:
        import sonicmoe  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise RuntimeError(
            "The Sonic MoE backend requires the optional 'sonic-moe' dependency "
            "(pip install 'ai2-olmo-core[sonic]')."
        ) from exc
    _patch_quack_sm100_configs()


@torch.library.custom_op("olmo_core::sonic_moe_fwd", mutates_args=())
def _sonic_moe_fwd(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    router_scores: torch.Tensor,
    w_up_gate: torch.Tensor,
    w_down: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    SonicMoE forward for fixed top-k routing.

    :param x: ``[T, H]`` routed inputs.
    :param expert_indices: ``[T * K]`` flattened expert ids, token-major.
    :param router_scores: ``[T * K]`` flattened routing weights.
    :param w_up_gate: ``[E, 2I, H]`` expert projection weights (Sonic order ``[gate; up]``).
    :param w_down: ``[E, I, H]`` expert down-projection weights.

    :returns: ``(out, h, expert_offsets, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
        token_offsets)`` where everything after ``out`` is saved for the backward kernel.
    """
    from quack.gemm_interface import gemm, gemm_act
    from sonicmoe.functional.forward import _router_forward
    from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton

    x = x.contiguous()
    T, H = x.shape
    TK = expert_indices.numel()
    E = w_up_gate.shape[0]
    K = TK // T
    I2 = w_up_gate.shape[1]
    I = I2 // 2
    device = x.device

    # [E, 2I, H] -> [2I, H, E] and [E, I, H] -> [H, I, E]: Sonic's expected views, no copies.
    w1 = w_up_gate.permute(1, 2, 0)
    w2 = w_down.permute(2, 1, 0)

    token_indices = torch.arange(T, device=device, dtype=torch.int32).repeat_interleave(K)
    expert_indices = expert_indices.to(dtype=torch.int32).contiguous()
    scores = router_scores.to(dtype=torch.float32).contiguous()

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    token_offsets = torch.empty(T + 1, dtype=torch.int32, device=device)
    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        T,
        E,
        expert_frequency,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offsets,
    )

    # Up projection with fused SwiGLU; ``h`` is the pre-activation kept for backward.
    a = torch.empty(TK, I, dtype=x.dtype, device=device)
    h = torch.empty(TK, I2, dtype=x.dtype, device=device)
    gemm_act(
        x,
        w1.permute(2, 1, 0),
        activation="swiglu",
        cu_seqlens_m=expert_offsets,
        A_idx=x_gather_idx,
        preact_out=h,
        postact_out=a,
        store_preact=True,
        bias=None,
        concat_layout=("B",),
    )

    # Down projection, then weighted gather-sum over each token's K experts.
    y = torch.empty(TK, H, dtype=x.dtype, device=device)
    gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=expert_offsets, bias=None)
    out = torch.empty(T, H, dtype=x.dtype, device=device)
    _router_forward(
        y=y,
        o=out,
        topk_scores=scores,
        s_reverse_scatter_idx=s_reverse_scatter_idx,
        num_activated_expert_per_token_offset=token_offsets,
        varlen_K_max=E,
        H=H,
        is_varlen_K=True,
    )
    return (
        out,
        h,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offsets,
    )


@_sonic_moe_fwd.register_fake
def _sonic_moe_fwd_fake(x, expert_indices, router_scores, w_up_gate, w_down):
    T, H = x.shape
    TK = expert_indices.numel()
    E, I2, _ = w_up_gate.shape
    i32 = {"dtype": torch.int32, "device": x.device}
    return (
        x.new_empty(T, H),
        x.new_empty(TK, I2),
        torch.empty(E + 1, **i32),
        torch.empty(TK, **i32),
        torch.empty(TK, **i32),
        torch.empty(TK, **i32),
        torch.empty(T + 1, **i32),
    )


@torch.library.custom_op("olmo_core::sonic_moe_bwd", mutates_args=())
def _sonic_moe_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    w_up_gate: torch.Tensor,
    w_down: torch.Tensor,
    h: torch.Tensor,
    scores: torch.Tensor,
    expert_offsets: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    token_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SonicMoE backward, mirroring ``_DownProjection.backward`` then ``_UpProjection.backward``.

    :returns: ``(grad_x, grad_w_up_gate, grad_w_down, grad_scores)`` with the weight gradients in
        the parameters' own ``[E, 2I, H]`` / ``[E, I, H]`` layouts.
    """
    from quack.gemm_interface import gemm
    from sonicmoe.functional.backward import (
        _down_projection_backward_act,
        _token_broadcast_backward,
        _up_projection_backward_act,
    )

    grad_out = grad_out.contiguous()
    x = x.contiguous()
    scores = scores.to(dtype=torch.float32).contiguous()
    T, H = x.shape
    E = w_up_gate.shape[0]
    TK = x_gather_idx.numel()
    w1 = w_up_gate.permute(1, 2, 0)  # [2I, H, E]
    w2 = w_down.permute(2, 1, 0)  # [H, I, E]
    I = w2.shape[1]
    device = x.device

    # Down projection backward: dh (through SwiGLU), dw2, and routing-score grads.
    dw2 = torch.empty_like(w2)
    dh = torch.empty_like(h)
    a_prime = torch.empty(TK, I, dtype=h.dtype, device=device)
    ds = torch.empty_like(scores)
    _down_projection_backward_act(
        dout=grad_out,
        h=h,
        w2=w2,
        dh=dh,
        ds=ds,
        b2=None,
        db2=None,
        a_prime=a_prime,
        topk_scores=scores,
        expert_frequency_offset=expert_offsets,
        x_gather_idx=x_gather_idx,
        s_scatter_idx=s_scatter_idx,
        activation_type="swiglu",
    )
    gemm(
        grad_out.T,
        a_prime,
        out=dw2.permute(2, 0, 1),
        cu_seqlens_k=expert_offsets,
        A_idx=x_gather_idx,
        batch_idx_permute=None,
        dynamic_scheduler=False,
    )

    # Up projection backward: dx (expanded then reduced over K) and dw1.
    dx_expanded = torch.empty(TK, H, dtype=dh.dtype, device=device)
    dw1 = torch.empty_like(w1)
    _up_projection_backward_act(
        w1=w1,
        dx_expanded=dx_expanded,
        dh=dh,
        db1=None,
        expert_frequency_offset=expert_offsets,
        is_glu_activation=True,
        concat_layout=True,
    )
    gemm(
        x.T,
        dh,
        out=dw1.permute(2, 1, 0),
        cu_seqlens_k=expert_offsets,
        A_idx=x_gather_idx,
        batch_idx_permute=None,
        dynamic_scheduler=False,
        concat_layout=("out",),
    )
    dx = torch.empty(T, H, dtype=dh.dtype, device=device)
    _token_broadcast_backward(
        dx_reduced=dx,
        dx_expanded=dx_expanded,
        s_reverse_scatter_idx=s_reverse_scatter_idx,
        num_activated_expert_per_token_offset=token_offsets,
        varlen_K_max=E,
        H=H,
        is_varlen_K=True,
    )

    # ``empty_like`` preserved the parameters' memory order through the views, so these permutes
    # back to [E, 2I, H] / [E, I, H] are contiguous and copy-free.
    return dx, dw1.permute(2, 0, 1).contiguous(), dw2.permute(2, 1, 0).contiguous(), ds


@_sonic_moe_bwd.register_fake
def _sonic_moe_bwd_fake(
    grad_out,
    x,
    w_up_gate,
    w_down,
    h,
    scores,
    expert_offsets,
    x_gather_idx,
    s_scatter_idx,
    s_reverse_scatter_idx,
    token_offsets,
):
    return (
        torch.empty_like(x),
        torch.empty_like(w_up_gate),
        torch.empty_like(w_down),
        torch.empty_like(scores),
    )


def _cpu_routing(expert_indices: torch.Tensor, T: int, E: int):
    """Reference routing metadata: pairs sorted by expert, matching Sonic's tensor roles."""
    K = expert_indices.numel() // T
    token_of_pair = torch.arange(T, dtype=torch.int64).repeat_interleave(K)
    order = torch.argsort(expert_indices.to(torch.int64), stable=True)  # sorted pos -> pair
    counts = torch.bincount(expert_indices.to(torch.int64), minlength=E)
    expert_offsets = torch.zeros(E + 1, dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    x_gather_idx = token_of_pair[order].to(torch.int32)  # sorted pos -> token
    s_reverse_scatter_idx = order.to(torch.int32)  # sorted pos -> pair
    s_scatter_idx = torch.empty_like(s_reverse_scatter_idx)
    s_scatter_idx[order] = torch.arange(order.numel(), dtype=torch.int32)  # pair -> sorted pos
    token_offsets = (torch.arange(T + 1) * K).to(torch.int32)
    return x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, expert_offsets, token_offsets


def _cpu_reference_forward(x, expert_indices, scores, w_up_gate, w_down, x_gather_idx, order):
    """Sonic-order SwiGLU experts on CPU. Returns (out, h) with rows in sorted-pair order."""
    T = x.shape[0]
    I = w_down.shape[1]
    experts_sorted = expert_indices.to(torch.int64)[order]
    xs = x[x_gather_idx.to(torch.int64)].float()
    w1 = w_up_gate[experts_sorted].float()  # [TK, 2I, H]
    h = torch.einsum("th,tih->ti", xs, w1)  # [TK, 2I] = [gate ; up]
    gate, up = h[:, :I], h[:, I:]
    a = up * torch.nn.functional.silu(gate)
    y = torch.einsum("ti,tih->th", a, w_down[experts_sorted].float())
    weighted = y * scores.float()[order].unsqueeze(-1)
    out = torch.zeros(T, x.shape[1], dtype=torch.float32)
    out.index_add_(0, x_gather_idx.to(torch.int64), weighted)
    return out.to(x.dtype), h.to(x.dtype)


@_sonic_moe_fwd.register_kernel("cpu")
def _sonic_moe_fwd_cpu(x, expert_indices, router_scores, w_up_gate, w_down):
    T = x.shape[0]
    E = w_up_gate.shape[0]
    scores = router_scores.to(torch.float32).contiguous()
    x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, expert_offsets, token_offsets = (
        _cpu_routing(expert_indices, T, E)
    )
    order = s_reverse_scatter_idx.to(torch.int64)
    out, h = _cpu_reference_forward(
        x, expert_indices, scores, w_up_gate, w_down, x_gather_idx, order
    )
    return (
        out,
        h,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offsets,
    )


@_sonic_moe_bwd.register_kernel("cpu")
def _sonic_moe_bwd_cpu(
    grad_out,
    x,
    w_up_gate,
    w_down,
    h,
    scores,
    expert_offsets,
    x_gather_idx,
    s_scatter_idx,
    s_reverse_scatter_idx,
    token_offsets,
):
    """Analytic reference backward (autograd is unavailable inside a custom-op kernel)."""
    E = w_up_gate.shape[0]
    I = w_down.shape[1]
    counts = (expert_offsets[1:] - expert_offsets[:-1]).to(torch.int64)
    experts_sorted = torch.repeat_interleave(torch.arange(E), counts)
    order = s_reverse_scatter_idx.to(torch.int64)  # sorted pos -> pair
    tokens = x_gather_idx.to(torch.int64)  # sorted pos -> token
    scores_sorted = scores.to(torch.float32)[order]

    xs = x[tokens].float()
    w1s = w_up_gate[experts_sorted].float()  # [TK, 2I, H]
    w2s = w_down[experts_sorted].float()  # [TK, I, H]
    hf = h.float()
    gate, up = hf[:, :I], hf[:, I:]
    silu_gate = torch.nn.functional.silu(gate)
    a = up * silu_gate
    y = torch.einsum("ti,tih->th", a, w2s)

    g = grad_out.float()[tokens]  # [TK, H]
    ds_sorted = (g * y).sum(-1)
    dy = g * scores_sorted.unsqueeze(-1)
    da = torch.einsum("th,tih->ti", dy, w2s)
    dw2_pairs = torch.einsum("ti,th->tih", a, dy)
    sig = torch.sigmoid(gate)
    dsilu = sig * (1 + gate * (1 - sig))
    dh = torch.cat([da * up * dsilu, da * silu_gate], dim=-1)
    dxs = torch.einsum("ti,tih->th", dh, w1s)
    dw1_pairs = torch.einsum("ti,th->tih", dh, xs)

    dx = torch.zeros(x.shape, dtype=torch.float32).index_add_(0, tokens, dxs)
    dw_up_gate = torch.zeros(w_up_gate.shape, dtype=torch.float32).index_add_(
        0, experts_sorted, dw1_pairs
    )
    dw_down = torch.zeros(w_down.shape, dtype=torch.float32).index_add_(
        0, experts_sorted, dw2_pairs
    )
    ds = torch.empty(scores.numel(), dtype=torch.float32)
    ds[order] = ds_sorted
    return dx.to(x.dtype), dw_up_gate.to(w_up_gate.dtype), dw_down.to(w_down.dtype), ds


def _sonic_setup_context(ctx, inputs, output):
    x, _expert_indices, router_scores, w_up_gate, w_down = inputs
    _out, h, expert_offsets, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, token_offsets = (
        output
    )
    ctx.mark_non_differentiable(
        h, expert_offsets, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, token_offsets
    )
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        x,
        w_up_gate,
        w_down,
        h,
        router_scores,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offsets,
    )
    ctx.router_scores_dtype = router_scores.dtype
    ctx.router_scores_shape = tuple(router_scores.shape)


def _sonic_backward(ctx, grad_out, *_unused_grads):
    (
        x,
        w_up_gate,
        w_down,
        h,
        scores,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offsets,
    ) = ctx.saved_tensors
    if grad_out is None:
        return None, None, None, None, None
    dx, dw_up_gate, dw_down, ds = torch.ops.olmo_core.sonic_moe_bwd(
        grad_out,
        x,
        w_up_gate,
        w_down,
        h,
        scores,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offsets,
    )
    ds = ds.view(ctx.router_scores_shape).to(ctx.router_scores_dtype)
    return dx, None, ds, dw_up_gate, dw_down


torch.library.register_autograd(
    "olmo_core::sonic_moe_fwd", _sonic_backward, setup_context=_sonic_setup_context
)


def sonic_moe_forward(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: RoutedExperts,
) -> torch.Tensor:
    """
    Execute routed experts with SonicMoE, including gather and weighted combine.

    This is a single custom op in both the forward and backward graphs, so it can sit inside a
    ``torch.compile`` region without a graph break. It is limited to fresh, BF16, bias-free, non-EP
    SwiGLU runs; the first half of :attr:`RoutedExperts.w_up_gate` is interpreted as the gate
    projection and the second as the up projection (see the module docstring).

    :param x: ``[T, H]`` routed inputs.
    :param expert_indices: ``[T, K]`` selected expert ids per token.
    :param expert_weights: ``[T, K]`` routing weights per token.
    :param experts: The routed experts whose ``w_up_gate`` / ``w_down`` parameters to use.

    :returns: ``[T, H]`` combined expert outputs.

    .. note::
        CPU tensors use a slow pure-PyTorch reference implementation of both kernels, intended
        for tests; it needs no SonicMoE install.
    """
    if x.is_cuda:
        _require_sonic()
    out, *_ = torch.ops.olmo_core.sonic_moe_fwd(
        x,
        expert_indices.reshape(-1),
        expert_weights.reshape(-1),
        experts.w_up_gate,
        experts.w_down,
    )
    return out
