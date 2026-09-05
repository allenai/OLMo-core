"""
One compute budget over the learned routers -- the nested-width FFN router
(:mod:`olmo_core.nn.nested_ffn_moe`), the KV-cache router (:mod:`olmo_core.nn.attention.kv_route`)
and, when enabled, per-token block skipping (:mod:`olmo_core.nn.block_skip`) -- so a fine-tune
decides *per task* how to split a FLOP saving between FFN width, attention/KV keeping and depth,
instead of being handed separate targets.

Per-token training FLOPs are decomposed per block ``b`` at a reference sequence length::

    cost = s_fixed + sum_b keep_b * ( s_proj_b + s_ffn_b * ffn_cost_b + s_attn_b * kv_keep_b )
    loss += lambda * |cost - target|

where ``s_*`` are the block's shares of dense per-token FLOPs (projections; FFN; length-dependent
QK^T/PV attention scores), ``keep_b`` is the block-skip router's expected run fraction (1 when the
block is not skip-routed), ``ffn_cost_b`` the FFN router's expected relative width (1 when the
block's FFN is not routed) and ``kv_keep_b`` the KV router's expected keep fraction (1 when not
routed). ``s_fixed`` is everything outside the blocks (embeddings, LM head). ``target`` is a
fraction of dense FLOPs, annealed from 1.0 like the per-router targets.

The per-router budgets are switched off when this is installed (their ``budget_weight`` is set to
0); their metrics and straight-through machinery are untouched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger(__name__)

__all__ = ["install_joint_budget", "joint_budget_loss"]


def _block_shares(model: Any, seq_len: int) -> Dict[str, Any]:
    dense = float(model.num_flops_per_token(seq_len))
    per: List[Dict[str, float]] = []
    for key, blk in model.blocks.items():
        ffn = (
            float(blk.feed_forward.num_flops_per_token(seq_len))
            if hasattr(blk, "feed_forward")
            else 0.0
        )
        attn_mod = getattr(blk, "attention", None)
        attn_all = float(attn_mod.num_flops_per_token(seq_len)) if attn_mod is not None else 0.0
        attn_score = 0.0
        if attn_mod is not None and hasattr(attn_mod, "w_q"):
            attn_score = attn_all - float(attn_mod.num_flops_per_token(0))
        per.append(
            {
                "proj": (attn_all - attn_score) / dense,
                "ffn": ffn / dense,
                "attn": attn_score / dense,
            }
        )
    total_blocks = sum(p["proj"] + p["ffn"] + p["attn"] for p in per)
    return {"per_block": per, "s_fixed": max(0.0, 1.0 - total_blocks)}


def install_joint_budget(
    model: Any, *, target: float, seq_len: int, weight: float = 1.0, anneal_calls: int = 0
) -> Dict[str, Any]:
    """
    Attach a joint budget to a model with at least one of the routers enabled.

    :param model: A :class:`~olmo_core.nn.transformer.Transformer` after the ``enable_*`` calls.
    :param target: Total training FLOPs as a fraction of dense (e.g. ``0.5``).
    :param seq_len: Sequence length the FLOP shares are evaluated at (a REAL example length, not
        the padded window: at 65k attention scores are ~50% of dense FLOPs vs ~12% at 8k).
    :param weight: ``lambda`` on the two-sided term.
    :param anneal_calls: Linear anneal of the target from 1.0 over this many forwards.

    :raises RuntimeError: If no router is enabled.
    """
    nffn = getattr(model, "_nested_ffn_moe", None)
    kvr = getattr(model, "_kv_route", None)
    bsk = getattr(model, "_block_skip", None)
    if nffn is None and kvr is None and bsk is None:
        raise RuntimeError("install_joint_budget needs at least one enabled router")
    shares = _block_shares(model, seq_len)
    for h in (nffn, kvr, bsk):
        if h is not None:
            h["holder"].budget_weight = 0.0
    n_blocks = len(shares["per_block"])
    jb = {
        "target": float(target),
        "weight": float(weight),
        "anneal_calls": int(anneal_calls),
        "per_block": shares["per_block"],
        "s_fixed": shares["s_fixed"],
        "ffn_layers": set(range(nffn["start_layer"], n_blocks)) if nffn is not None else set(),
        "kv_layers": set(kvr["routed"]) if kvr is not None else set(),
        "skip_layers": set(bsk["routed"]) if bsk is not None else set(),
    }
    model._joint_budget = jb
    s_ffn = sum(p["ffn"] for i, p in enumerate(shares["per_block"]) if i in jb["ffn_layers"])
    s_attn = sum(p["attn"] for i, p in enumerate(shares["per_block"]) if i in jb["kv_layers"])
    s_skip = sum(
        p["proj"] + p["ffn"] + p["attn"]
        for i, p in enumerate(shares["per_block"])
        if i in jb["skip_layers"]
    )
    jb["s_ffn"], jb["s_attn"], jb["s_skip"] = s_ffn, s_attn, s_skip
    log.info(
        "Joint budget: target %.3f of dense FLOPs (seq_len %d); routable shares ffn %.3f attn-score %.3f "
        "skippable-blocks %.3f; fixed outside blocks %.3f",
        target,
        seq_len,
        s_ffn,
        s_attn,
        s_skip,
        jb["s_fixed"],
    )
    print(
        f"[joint-budget] target {target:.3f}; shares ffn {s_ffn:.3f} attn {s_attn:.3f} "
        f"skippable {s_skip:.3f} fixed {jb['s_fixed']:.3f} (seq_len {seq_len})",
        flush=True,
    )
    return jb


def _per_layer_expectations(
    holder: Any, attr_costs: str, attr_layers: str
) -> Dict[int, torch.Tensor]:
    costs = getattr(holder, attr_costs, None) or []
    layers = getattr(holder, attr_layers, None) or []
    if len(costs) != len(layers):
        return {}
    return {int(li): c for li, c in zip(layers, costs)}


def joint_budget_loss(model: Any) -> Optional[torch.Tensor]:
    """The joint term for the current forward (``None`` if nothing differentiable was routed)."""
    jb = getattr(model, "_joint_budget", None)
    if jb is None:
        return None
    nffn = getattr(model, "_nested_ffn_moe", None)
    kvr = getattr(model, "_kv_route", None)
    bsk = getattr(model, "_block_skip", None)
    ffn_e = _per_layer_expectations(nffn["holder"], "_exp_costs", "_exp_layers") if nffn else {}
    kv_e = _per_layer_expectations(kvr["holder"], "_exp_keep", "_exp_layers") if kvr else {}
    sk_e = _per_layer_expectations(bsk["holder"], "_exp_keep", "_exp_layers") if bsk else {}
    if not (ffn_e or kv_e or sk_e):
        return None
    cost: Any = jb["s_fixed"]
    for i, sh in enumerate(jb["per_block"]):
        inner = sh["proj"] + sh["ffn"] * ffn_e.get(i, 1.0) + sh["attn"] * kv_e.get(i, 1.0)
        cost = cost + sk_e.get(i, 1.0) * inner
    if not torch.is_tensor(cost) or not cost.requires_grad:
        return None
    calls = max(h["holder"].calls for h in (nffn, kvr, bsk) if h is not None)
    tgt = jb["target"]
    if jb["anneal_calls"] > 0:
        tgt = 1.0 + (jb["target"] - 1.0) * min(1.0, calls / jb["anneal_calls"])
    jb["last_cost"] = float(cost.detach())
    jb["last_target"] = tgt
    return jb["weight"] * (cost - tgt).abs()
