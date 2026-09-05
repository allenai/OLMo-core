"""
One compute budget over BOTH learned routers -- the nested-width FFN router
(:mod:`olmo_core.nn.nested_ffn_moe`) and the KV-cache router
(:mod:`olmo_core.nn.attention.kv_route`) -- so a fine-tune decides *per task* how to split a
FLOP saving between FFN width and attention/KV keeping instead of being handed two separate
targets.

The joint expected cost is the FLOP-share-weighted sum of the two routers' differentiable means::

    cost = s_ffn * E[ffn_cost] + s_attn * E[keep] + (1 - s_ffn - s_attn)
    loss += lambda * |cost - target|

where ``s_ffn`` is the routed FFN layers' share of dense per-token training FLOPs and ``s_attn``
the routed attention layers' share of *length-dependent* (QK^T, PV) FLOPs, both evaluated at the
trainer's ``seq_len``. The unrouted remainder (embeddings, projections, recurrent layers, ...) is
fixed cost. ``target`` is a fraction of dense FLOPs, annealed from 1.0 like the per-router targets.

The per-router budgets are switched off when this is installed (their ``budget_weight`` is set to
0); their metrics and straight-through machinery are untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

log = logging.getLogger(__name__)

__all__ = ["install_joint_budget", "joint_budget_loss"]


def install_joint_budget(
    model: Any, *, target: float, seq_len: int, weight: float = 1.0, anneal_calls: int = 0
) -> Dict[str, Any]:
    """
    Attach a joint budget to a model with BOTH routers enabled.

    :param model: A :class:`~olmo_core.nn.transformer.Transformer` after ``enable_nested_ffn_moe``
        and ``enable_kv_route``.
    :param target: Total training FLOPs as a fraction of dense (e.g. ``0.5``).
    :param seq_len: Sequence length the FLOP shares are evaluated at.
    :param weight: ``lambda`` on the two-sided term.
    :param anneal_calls: Linear anneal of the target from 1.0 over this many forwards.

    :raises RuntimeError: If either router is missing.
    """
    nffn = getattr(model, "_nested_ffn_moe", None)
    kvr = getattr(model, "_kv_route", None)
    if nffn is None or kvr is None:
        raise RuntimeError(
            "install_joint_budget needs both enable_nested_ffn_moe and enable_kv_route"
        )
    dense = float(model.num_flops_per_token(seq_len))
    ffn_routed = sum(
        model.blocks[str(li)].feed_forward.num_flops_per_token(seq_len)
        for li in range(nffn["start_layer"], len(model.blocks))
    )
    attn_routed = sum(
        model.blocks[str(li)].attention.num_flops_per_token(seq_len)
        - model.blocks[str(li)].attention.num_flops_per_token(0)
        for li in kvr["routed"]
    )
    s_ffn, s_attn = ffn_routed / dense, attn_routed / dense
    nffn["holder"].budget_weight = 0.0
    kvr["holder"].budget_weight = 0.0
    jb = {
        "target": float(target),
        "weight": float(weight),
        "anneal_calls": int(anneal_calls),
        "s_ffn": s_ffn,
        "s_attn": s_attn,
        "s_fixed": max(0.0, 1.0 - s_ffn - s_attn),
    }
    model._joint_budget = jb
    log.info(
        "Joint budget: target %.3f of dense FLOPs; shares ffn %.3f attn-score %.3f fixed %.3f (seq_len %d)",
        target,
        s_ffn,
        s_attn,
        jb["s_fixed"],
        seq_len,
    )
    return jb


def joint_budget_loss(model: Any) -> Optional[torch.Tensor]:
    """The joint term for the current forward (``None`` if nothing was routed)."""
    jb = getattr(model, "_joint_budget", None)
    if jb is None:
        return None
    ffn_h = model._nested_ffn_moe["holder"]
    kv_h = model._kv_route["holder"]
    parts = []
    if ffn_h._exp_costs:
        parts.append(jb["s_ffn"] * torch.stack(ffn_h._exp_costs).mean())
    else:
        parts.append(torch.tensor(jb["s_ffn"]))
    if kv_h._exp_keep:
        parts.append(
            jb["s_attn"] * torch.stack(kv_h._exp_keep).sum() / max(1e-9, sum(kv_h._exp_weights))
        )
    else:
        parts.append(torch.tensor(jb["s_attn"]))
    if not any(p.requires_grad for p in parts):
        return None
    cost = sum(parts) + jb["s_fixed"]
    calls = max(ffn_h.calls, kv_h.calls)
    tgt = jb["target"]
    if jb["anneal_calls"] > 0:
        tgt = 1.0 + (jb["target"] - 1.0) * min(1.0, calls / jb["anneal_calls"])
    jb["last_cost"] = float(cost.detach())
    jb["last_target"] = tgt
    return jb["weight"] * (cost - tgt).abs()
