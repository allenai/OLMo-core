"""Per-(token, layer) SKIP router on top of the frozen token router (stage 2), shared by the tests and
the trainer. No shared model file is modified: the block stack is driven from here, reusing
``Transformer._prepare_inputs`` / ``blocks`` / ``lm_head`` and the existing per-block ``soft_keep``.

Semantics (exact at binary gates). Skipping block ``l`` for token ``t`` removes ``t`` from that layer
only: its residual passes through (``h_out = h_in``), it is not a key/value at an attention layer,
it neither writes nor decays the GDN state, and the removal-aware short conv skips it. Relaxation
with a per-layer gate ``a[t, l]`` in [0, 1] and the frozen token gate ``g[t]``:

* ``h_out = h_in + a * (block(h_in) - h_in)``  (computed so that a = 1 returns ``block(h_in)`` and
  a = 0 returns ``h_in`` bit-exactly);
* the block's mixer gets ``soft_keep = g * a[:, l]`` -- attention ``+log(g a)`` key bias, GDN beta and
  log-decay scaled by ``g a``, removal-aware conv on ``g a``.

Router (MoD-style): ``logit[t, l] = b_l + w_l . RMSnorm(h_t^(l)) / sqrt(d)`` at every block input
(h detached), 32 x (d + 1) parameters. Only ELIGIBLE tokens (document body tokens the token router
kept) are routed; everything else has ``a = 1``.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
LR_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LR_DIR))
for _p in (LR_DIR, os.path.join(REPO, "debug", "devloss_grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import router_lib as RL  # noqa: E402

GateFn = Callable[[int, torch.Tensor], Optional[torch.Tensor]]


# ------------------------------------------------------------------------------------------------
# router
# ------------------------------------------------------------------------------------------------
VARIANTS = ("perlayer", "shared", "shared_tok")


class LayerRouter(nn.Module):
    """``perlayer``: ``b_l + w_l . x / sqrt(d)`` (L x (d + 1) params).
    ``shared``: one vector over ``[x ; onehot(l) ; is_attn(l)]`` plus a bias -- ``b + w . x / sqrt(d) +
    u_l + v * is_attn(l)`` (d + L + 2 params).
    ``shared_tok``: ``shared`` plus the token router's per-token features (gold flag, the 58 position
    features, the RMS-normed input embedding / sqrt(d)), constant across layers.

    ``x = RMSnorm(h_t^(l))`` of the (detached) block input."""

    def __init__(self, n_layers: int, d: int, variant: str = "perlayer", attn_layers: Sequence[int] = ()):
        super().__init__()
        assert variant in VARIANTS, variant
        self.n_layers, self.d, self.variant = int(n_layers), int(d), variant
        self.attn_layers = sorted(int(v) for v in attn_layers)
        if variant == "perlayer":
            self.b = nn.Parameter(torch.zeros(self.n_layers))
            self.w = nn.Parameter(torch.zeros(self.n_layers, self.d))
        else:
            self.b = nn.Parameter(torch.zeros(1))
            self.w = nn.Parameter(torch.zeros(self.d))
            self.u = nn.Parameter(torch.zeros(self.n_layers))
            self.v = nn.Parameter(torch.zeros(1))
        if variant == "shared_tok":
            self.w_pos = nn.Parameter(torch.zeros(RL.N_POS))
            self.w_gold = nn.Parameter(torch.zeros(1))
            self.w_emb = nn.Parameter(torch.zeros(self.d))

    def param_groups(self, lr: float, w_lr: float):
        big = [p for n, p in self.named_parameters() if n in ("w", "w_emb")]
        small = [p for n, p in self.named_parameters() if n not in ("w", "w_emb")]
        return [{"params": small, "lr": lr}, {"params": big, "lr": w_lr}]

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def logits(self, layer: int, h: torch.Tensor, tokfeat: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """(N, d) block inputs of the routed tokens -> (N,) raw logit (fp32, h detached).
        ``tokfeat`` (``shared_tok``): ``pos`` (N, 58), ``gold`` (N,), ``e`` (N, d) aligned with ``h``."""
        x = h.detach().float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.variant == "perlayer":
            return self.b[layer] + (x @ self.w[layer]) / math.sqrt(self.d)
        z = self.b + (x @ self.w) / math.sqrt(self.d) + self.u[layer] + (self.v if layer in self.attn_layers else 0.0)
        if self.variant == "shared_tok":
            assert tokfeat is not None, "shared_tok needs the per-token features"
            z = z + tokfeat["pos"].float() @ self.w_pos + tokfeat["gold"].float() * self.w_gold \
                + (tokfeat["e"].float() @ self.w_emb) / math.sqrt(self.d)
        return z

    def state(self) -> dict:
        return {"n_layers": self.n_layers, "d": self.d, "variant": self.variant, "attn_layers": self.attn_layers,
                "params": {n: p.detach().cpu() for n, p in self.named_parameters()}}

    @classmethod
    def from_state(cls, st: dict) -> "LayerRouter":
        r = cls(st["n_layers"], st["d"], st.get("variant", "perlayer"), st.get("attn_layers", ()))
        with torch.no_grad():
            for n, p in r.named_parameters():
                p.copy_(st["params"][n])
        return r


class Gater:
    """Per-layer gate provider for :func:`forward_layerskip`. ``elig`` (B, T) bool marks routed
    positions; every other position has ``a = 1``. Modes:

    * ``sample``: hard-concrete gate on ``router logit + c`` (training);
    * ``det``: binary ``router logit + c > 0`` (evaluation);
    * ``fixed``: ``values`` (L, B, T) float/bool given (baselines, tests); non-eligible forced to 1;
    * ``ones``: a = 1 everywhere (soft path with nothing skipped).

    Records per layer the detached raw logits and the gate at eligible positions."""

    def __init__(self, elig: torch.Tensor, mode: str, router: Optional[LayerRouter] = None, c: float = 0.0,
                 beta: float = 0.5, seed: int = 0, st: bool = False, values: Optional[torch.Tensor] = None,
                 tokfeat: Optional[Dict[str, torch.Tensor]] = None):
        """``tokfeat``: per-routed-token features in ``torch.nonzero(elig)`` order (``shared_tok``)."""
        self.elig, self.mode, self.router, self.c, self.tokfeat = elig, mode, router, float(c), tokfeat
        self.beta, self.seed, self.st, self.values = beta, seed, st, values
        self.bi, self.ti = torch.nonzero(elig, as_tuple=True)
        self.raw: Dict[int, torch.Tensor] = {}
        self.gates: Dict[int, torch.Tensor] = {}

    def __call__(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        B, T = self.elig.shape
        fx = self.mode == "fixed" and self.values is not None and self.values.is_floating_point()
        dt = self.values.dtype if fx else torch.float32  # float64 gradcheck keeps its precision
        ones = torch.ones(B, T, dtype=dt, device=h.device)
        if self.mode == "ones" or self.bi.numel() == 0:
            self.gates[layer] = torch.ones(int(self.bi.numel()), device=h.device)
            return ones
        if self.mode == "fixed":
            z = self.values[layer][self.bi, self.ti].to(device=h.device, dtype=dt)
        else:
            r = self.router.logits(layer, h[self.bi, self.ti], self.tokfeat)
            self.raw[layer] = r.detach()
            la = r + self.c
            if self.mode == "det":
                z = (la > 0).float()
            else:
                gen = torch.Generator().manual_seed(int(self.seed) * 1009 + layer)
                z = RL.hard_concrete(la, self.beta, gen, straight_through=self.st, st_clamp=True)
        self.gates[layer] = z.detach()
        return ones.index_put((self.bi, self.ti), z)


# ------------------------------------------------------------------------------------------------
# forward passes
# ------------------------------------------------------------------------------------------------
def residual_mix(h_in: torch.Tensor, out: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``h_in + a (out - h_in)`` in fp32, exact at both binary ends: a = 1 -> ``out``, a = 0 -> ``h_in``."""
    a3 = a.unsqueeze(-1).to(torch.float32)
    hf, of = h_in.float(), out.float()
    d = of - hf
    mixed = torch.where(a3 >= 0.5, of + (a3 - 1.0) * d, hf + a3 * d)
    return mixed.to(out.dtype)


def _check_model(model) -> None:
    for attr in ("_nested_ffn_moe", "_kv_route", "_block_skip", "_joint_budget"):
        assert getattr(model, attr, None) is None, f"forward_layerskip does not support {attr}"
    assert not getattr(model, "budget_attach", False)


def _embed(model, ids: torch.Tensor) -> torch.Tensor:
    h = model.embeddings(ids)
    if model.embed_scale is not None:
        h = h * model.embed_scale
    if model.embedding_norm is not None:
        h = model.embedding_norm(h)
    return h


def forward_layerskip(model, x: torch.Tensor, *, logits_to_keep, position_ids: Optional[torch.Tensor] = None,
                      tok_keep: Optional[torch.Tensor] = None, gate: Optional[GateFn] = None) -> torch.Tensor:
    """Logits under token gate ``tok_keep`` (B, T) and per-layer gates ``gate(layer, h_in) -> (B, T)``.

    ``gate=None`` is exactly ``model(x, soft_keep=tok_keep, position_ids=...)``; a gate returning ones
    is bit-identical to that too (``residual_mix`` returns ``out`` at a = 1 and ``tok_keep * 1``)."""
    _check_model(model)
    model.eval()
    model._pooled_keep_holder = None
    kw = {} if position_ids is None else {"position_ids": position_ids}
    ids, _, abk, pbk, lmk = model._prepare_inputs(x, None, logits_to_keep=logits_to_keep, **kw)
    assert set(abk) <= {"position_ids"}, sorted(abk)
    h = _embed(model, ids)
    for key, block in model.blocks.items():
        layer = int(key)
        a = gate(layer, h) if gate is not None else None
        sk = tok_keep if a is None else (a if tok_keep is None else tok_keep.to(a.dtype) * a)
        bk = {**abk, **pbk.get(layer, {})}
        if sk is not None:
            bk["soft_keep"] = sk
        out = block(h, **bk)
        h = out if a is None else residual_mix(h, out, a)
    return model.lm_head(h, **lmk)


def forward_reference(model, x: torch.Tensor, keep_layers: torch.Tensor, *, logits_to_keep,
                      position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """EXPLICIT per-layer removal (B = 1): at layer l the block runs on the sub-sequence of columns
    with ``keep_layers[l]`` True (original positions); the other columns copy their input through.
    ``keep_layers`` (L, T) bool already folds in the token gate (a token-dropped column is False at
    every layer). No ``soft_keep`` anywhere: the reference for the relaxation."""
    _check_model(model)
    assert x.shape[0] == 1
    model.eval()
    model._pooled_keep_holder = None
    T = x.shape[1]
    pos = position_ids if position_ids is not None else torch.arange(T, device=x.device)[None]
    ids, _, abk, pbk, lmk = model._prepare_inputs(x, None, logits_to_keep=logits_to_keep, position_ids=pos)
    h = _embed(model, ids)
    for key, block in model.blocks.items():
        layer = int(key)
        K = torch.nonzero(keep_layers[layer].to(x.device)).flatten()
        out = block(h[:, K], position_ids=pos[:, K], **pbk.get(layer, {}))
        h = h.clone()
        h[:, K] = out
    return model.lm_head(h, **lmk)


# ------------------------------------------------------------------------------------------------
# FLOP model (collect_grid.forward_flops extended to per-(token, layer) skip counts)
# ------------------------------------------------------------------------------------------------
def _flop_consts():
    import collect_grid as CG

    return CG


def layer_flops(n_active: Sequence[float]) -> float:
    """FLOPs of one forward whose layer l runs ``n_active[l]`` columns (gather semantics: a skipped
    column is neither query nor key there; quadratic attention term per layer on its own count)."""
    CG = _flop_consts()
    f = 0.0
    for layer in range(CG.N_LAYERS):
        t = float(n_active[layer])
        f += (CG.ATTN_LIN * t + CG.ATTN_QUAD * t * t) if layer in CG.ATTN_LAYERS else CG.GDN_LIN * t
    return f


def flops_ratio(T_full: int, T2: int, n_skip: Sequence[float]) -> float:
    """(token-compacted row of T2 columns with ``n_skip[l]`` columns skipped at layer l) / full row."""
    CG = _flop_consts()
    if len(n_skip) != CG.N_LAYERS:  # tiny test models: the FLOP model is Qwen3.5-4B's
        return float("nan")
    return layer_flops([T2 - s for s in n_skip]) / CG.forward_flops(T_full)


def attn_layers() -> List[int]:
    return sorted(_flop_consts().ATTN_LAYERS)
