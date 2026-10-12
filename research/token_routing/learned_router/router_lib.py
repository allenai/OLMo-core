"""Learned linear token router for slot-less (detach-style) compaction: shared by the trainer
(``train_router.py``) and the dev-loss grid driver (``ctc_devloss_grid.py`` ``sel="router"``), so
train-time and eval-time features are computed by the same code.

For every ROUTED token -- a document body token (inside ``<|box_start|>..<|box_end|>``, not a
marker), gold documents included -- the router emits one logit

    z = b + w_pos . f_pos + w_gold * gold + w_emb . e_rms / sqrt(d)

and keeps the token with probability ``sigmoid(z)``. Markers and everything outside documents
(system prompt, query, answer) are never routed and always kept. ``e_rms`` is the frozen model's
input-embedding row, RMS-normalised in fp32. ``f_pos`` is length-agnostic (FEATURE_VERSION 1):

* offset from the document's first body token: one-hot 0..15, then log2 buckets 16-31, 32-63, ...
  (12 buckets, the last open-ended)                                       -> 28
* the same for the offset from the document's last body token              -> 28
* relative position within the document body, j / (n - 1)                  -> 1
* the document's relative index in the context, d / (n_docs - 1)           -> 1

Drop semantics (``keep_markers=True``): kept body tokens stay real at their ORIGINAL RoPE positions,
dropped ones vanish (no slot), and every document's two markers survive even when its whole body is
dropped (an empty ``<|box_start|><|box_end|>`` pair). See README.md for why markers are forced.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

FEATURE_VERSION = 2  # 2 adds the bounded relative-position family ``rel`` (w_rel); version-1 states load with w_rel = 0
N_ONEHOT = 16
N_LOG = 12
N_OFF = N_ONEHOT + N_LOG  # 28
N_POS = 2 * N_OFF + 2  # 58
POS_NAMES = (
    [f"start{o}" for o in range(N_ONEHOT)]
    + [f"start<2^{k + 5}" for k in range(N_LOG)]
    + [f"end{o}" for o in range(N_ONEHOT)]
    + [f"end<2^{k + 5}" for k in range(N_LOG)]
    + ["rel_in_doc", "doc_rel_idx"]
)
VARIANTS = {
    "full": dict(emb=True, gold=True),
    "nogold": dict(emb=True, gold=False),
    "noemb": dict(emb=False, gold=True),
    # without the two CONTINUOUS position features (relative position in doc, doc's relative index):
    # offsets from doc start/end one-hot + log buckets only (2026-09-29, contradiction ranked by them)
    "nocontpos": dict(emb=True, gold=True, contpos=False),
    # nocontpos + the bounded one-hot RELATIVE-position family (``rel``, 2026-09-30): makes
    # gold_fl20p8_noslot's keep mask exactly representable (see ``fl20p8_hand_state``)
    "relpos": dict(emb=True, gold=True, contpos=False, rel=True),
    "relpos_noemb": dict(emb=False, gold=True, contpos=False, rel=True),
    # feature ladder (2026-09-30): F0 doc-only (bias, gold, doc-length bucket, is_marker: every body token
    # of a document scores the same -> whole-document keep/drop + markers); F1 = F0 + start/end deciles
    "doc_only": dict(emb=False, gold=True, contpos=False, rel=True, pos=False, rel_part="len"),
    "coarse_pos": dict(emb=False, gold=True, contpos=False, rel=True, pos=False),
    # span-level (2026-09-30): each doc body is cut into contiguous spans of S tokens that share ONE score
    # and ONE gate (markers are their own unit); features = bias, gold, doc-length bucket, span index from
    # the doc start / end (one-hot 0-3 + log2 buckets), the span's start / end decile, is_marker
    "span8": dict(emb=False, gold=True, contpos=False, rel=True, rel_part="len", pos=False, span=8),
    "span4": dict(emb=False, gold=True, contpos=False, rel=True, rel_part="len", pos=False, span=4),
}
SPAN_SIZES = (4, 8)
N_SPAN_B = 12  # span-index buckets: 0..3 one-hot, then 4-7, 8-15, ... (capped)
N_SPANF = 2 * N_SPAN_B + 2 * 10  # 44


def span_bucket(s: torch.Tensor) -> torch.Tensor:
    lg = torch.floor(torch.log2(s.clamp(min=1).to(torch.float64))).to(torch.long)
    return torch.where(s < 4, s, 4 + (lg - 2)).clamp(max=N_SPAN_B - 1)

# Relative-position family (FEATURE_VERSION 2): GENERIC bounded one-hots, nothing tailored to a
# particular heuristic. Body token j of a document with body length n:
#   rs<b>  start decile  b = floor(10 j / n)
#   re<b>  end decile    b = floor(10 (n-1-j) / n)
#   nl<b>  document body length bucket b = min(floor(log2 n), 13)
N_DEC = 10
N_LEN = 14
N_REL = 2 * N_DEC + N_LEN  # 34
REL_NAMES = [f"rs{b}" for b in range(N_DEC)] + [f"re{b}" for b in range(N_DEC)] + [f"nl{b}" for b in range(N_LEN)]


def offset_bucket(o: torch.Tensor) -> torch.Tensor:
    """0..15 -> itself; 16..31 -> 16, 32..63 -> 17, ...; capped at N_OFF - 1."""
    o = o.clamp(min=0)
    lg = torch.floor(torch.log2(o.clamp(min=1).to(torch.float64))).to(torch.long)  # 4 for 16..31
    b = torch.where(o < N_ONEHOT, o, N_ONEHOT + (lg - 4))
    return b.clamp(max=N_OFF - 1)


def routed_features(x: torch.Tensor, cid: torch.Tensor, doc_start: int, doc_end: int, n_docs: int,
                    gold: Optional[Sequence[int]], route_markers: bool = False) -> Dict[str, torch.Tensor]:
    """Features of every routed token of ONE row.

    :param x: ``(S,)`` token ids.
    :param cid: ``(S,)`` chunk ids from ``build_chunk_ids_from_tokens(mode="chunked")`` (doc id >= 0
        inside documents, markers included).
    :returns: ``idx`` (N,) routed positions (ascending), ``tok`` (N,) ids, ``pos`` (N, N_POS) float,
        ``gold`` (N,) float, ``doc`` (N,) doc id, ``j`` / ``n`` within-doc index / body length.
    """
    dev = x.device
    cid = cid.to(dev).long()
    markers = (x == doc_start) | (x == doc_end)
    body = (cid >= 0) & ~markers
    idx = torch.nonzero(body).flatten()
    d = cid[idx]
    N = int(idx.numel())
    counts = torch.bincount(d, minlength=max(1, n_docs))
    # documents are contiguous, ascending-id spans, so body tokens of doc d are contiguous in idx
    starts = torch.cumsum(counts, 0) - counts
    j = torch.arange(N, device=dev) - starts[d]
    n = counts[d]
    if N:
        assert bool((j >= 0).all()) and bool((j < n).all()), "documents are not contiguous spans"
    pos = torch.zeros(N, N_POS, dtype=torch.float32, device=dev)
    ar = torch.arange(N, device=dev)
    pos[ar, offset_bucket(j)] = 1.0
    pos[ar, N_OFF + offset_bucket(n - 1 - j)] = 1.0
    pos[:, 2 * N_OFF] = torch.where(n > 1, j.float() / (n - 1).clamp(min=1).float(), torch.zeros_like(j, dtype=torch.float32))
    pos[:, 2 * N_OFF + 1] = d.float() / max(1, n_docs - 1)
    gt = torch.zeros(max(1, n_docs), dtype=torch.bool, device=dev)
    if gold:
        gl = [int(v) for v in gold if 0 <= int(v) < n_docs]
        if gl:
            gt[torch.tensor(gl, device=dev)] = True
    g = gt[d].float()
    rel = torch.zeros(N, N_REL, dtype=torch.float32, device=dev)
    if N:
        rel[ar, (10 * j) // n] = 1.0
        rel[ar, N_DEC + (10 * (n - 1 - j)) // n] = 1.0
        rel[ar, 2 * N_DEC + torch.floor(torch.log2(n.clamp(min=1).to(torch.float64))).to(torch.long).clamp(max=N_LEN - 1)] = 1.0
    spf, skey = {}, {}
    for S in SPAN_SIZES:
        f_ = torch.zeros(N, N_SPANF, dtype=torch.float32, device=dev)
        if N:
            sidx = j // S
            ns = (n + S - 1) // S
            f_[ar, span_bucket(sidx)] = 1.0
            f_[ar, N_SPAN_B + span_bucket(ns - 1 - sidx)] = 1.0
            last = torch.minimum((sidx + 1) * S, n) - 1
            f_[ar, 2 * N_SPAN_B + (10 * sidx * S) // n] = 1.0
            f_[ar, 2 * N_SPAN_B + 10 + (10 * (n - 1 - last)) // n] = 1.0
            skey[S] = d * 1_000_000 + sidx
        else:
            skey[S] = torch.zeros(0, dtype=torch.long, device=dev)
        spf[S] = f_
    out = {"idx": idx, "tok": x[idx], "pos": pos, "rel": rel, "gold": g, "doc": d, "j": j, "n": n,
           "is_marker": torch.zeros(N, dtype=torch.float32, device=dev)}
    for S in SPAN_SIZES:
        out[f"span{S}"] = spf[S]
        out[f"_skey{S}"] = skey[S]
    # marker positions and their document ids (for "markers follow their document", when markers are not routed)
    _midx = torch.nonzero(markers & (cid >= 0)).flatten()
    out["mk_idx"], out["mk_doc"] = _midx, cid[_midx]
    if route_markers:
        # markers are routed too: position features 0, own weight (w_marker), gold flag of their doc
        midx = torch.nonzero(markers & (cid >= 0)).flatten()
        md = cid[midx]
        allidx = torch.cat([idx, midx])
        order = torch.argsort(allidx)
        out = {"idx": allidx[order], "tok": x[allidx[order]],
               "pos": torch.cat([pos, torch.zeros(int(midx.numel()), N_POS, device=dev)])[order],
               "rel": torch.cat([rel, torch.zeros(int(midx.numel()), N_REL, device=dev)])[order],
               "gold": torch.cat([g, gt[md].float()])[order], "doc": torch.cat([d, md])[order],
               "j": torch.cat([j, torch.zeros_like(md)])[order], "n": torch.cat([n, torch.zeros_like(md)])[order],
               "is_marker": torch.cat([torch.zeros(N, device=dev), torch.ones(int(midx.numel()), device=dev)])[order]}
        for S in SPAN_SIZES:
            out[f"span{S}"] = torch.cat([spf[S], torch.zeros(int(midx.numel()), N_SPANF, device=dev)])[order]
            out[f"_skey{S}"] = torch.cat([skey[S], -(midx + 1)])[order]  # every marker is its own unit
    for S in SPAN_SIZES:
        # unit id per routed token (positions are ascending, so a unit's tokens are consecutive)
        k_ = out.pop(f"_skey{S}")
        out[f"unit{S}"] = torch.unique_consecutive(k_, return_inverse=True)[1] if k_.numel() else k_
    return out


def rms_embed(emb_weight: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    e = emb_weight[tok].float()
    return e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)


class LinearRouter(nn.Module):
    """One linear layer over [bias, position features, gold flag, RMS-normalised embedding]."""

    def __init__(self, d_emb: int, variant: str = "full"):
        super().__init__()
        self.d_emb = int(d_emb)
        self.variant = variant
        self.use_emb = VARIANTS[variant]["emb"]
        self.use_gold = VARIANTS[variant]["gold"]
        self.b = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
        self.w_pos = nn.Parameter(torch.zeros(N_POS))
        self.w_gold = nn.Parameter(torch.zeros(1))
        self.w_emb = nn.Parameter(torch.zeros(self.d_emb))
        self.w_marker = nn.Parameter(torch.zeros(1))  # only used when markers are routed
        self.w_rel = nn.Parameter(torch.zeros(N_REL))  # only used by the "relpos" variant
        self.w_span = nn.Parameter(torch.zeros(N_SPANF))  # only used by the span variants
        self.span = int(VARIANTS[variant].get("span", 0))
        self.route_markers = False
        self.marker_follow = False  # markers kept iff >= 1 body token of their document is kept

    def logits(self, feats: Dict[str, torch.Tensor], e_rms: Optional[torch.Tensor]) -> torch.Tensor:
        vv = VARIANTS[self.variant]
        w_pos = self.w_pos
        if not vv.get("contpos", True):
            w_pos = torch.cat([w_pos[: 2 * N_OFF], torch.zeros_like(w_pos[2 * N_OFF :])])
        z = self.b + (feats["pos"] @ w_pos if vv.get("pos", True) else 0.0)
        if vv.get("rel", False):
            w_rel = self.w_rel
            if vv.get("rel_part") == "len":  # doc-length buckets only (no within-doc position)
                w_rel = torch.cat([torch.zeros_like(w_rel[: 2 * N_DEC]), w_rel[2 * N_DEC :]])
            z = z + feats["rel"] @ w_rel
        if self.span:
            z = z + feats[f"span{self.span}"] @ self.w_span
        if self.use_gold:
            z = z + feats["gold"] * self.w_gold
        if "is_marker" in feats:
            z = z + feats["is_marker"] * self.w_marker
        if self.use_emb and e_rms is not None:
            z = z + (e_rms @ self.w_emb) / math.sqrt(self.d_emb)
        return z

    def state(self) -> dict:
        return {
            "feature_version": FEATURE_VERSION,
            "variant": self.variant,
            "d_emb": self.d_emb,
            "b": self.b.detach().cpu(),
            "w_pos": self.w_pos.detach().cpu(),
            "w_gold": self.w_gold.detach().cpu(),
            "w_emb": self.w_emb.detach().cpu(),
            "w_marker": self.w_marker.detach().cpu(),
            "w_rel": self.w_rel.detach().cpu(),
            "w_span": self.w_span.detach().cpu(),
            "route_markers": bool(self.route_markers),
            "marker_follow": self.marker_follow if self.marker_follow == "whole" else bool(self.marker_follow),
        }

    @classmethod
    def from_state(cls, st: dict) -> "LinearRouter":
        assert st["feature_version"] in (1, FEATURE_VERSION), st["feature_version"]
        r = cls(st["d_emb"], st["variant"])
        with torch.no_grad():
            for k in ("b", "w_pos", "w_gold", "w_emb"):
                getattr(r, k).copy_(st[k])
            if "w_marker" in st:
                r.w_marker.copy_(st["w_marker"])
            if "w_rel" in st:
                r.w_rel.copy_(st["w_rel"])
            if "w_span" in st:
                r.w_span.copy_(st["w_span"])
        r.route_markers = bool(st.get("route_markers", False))
        mf = st.get("marker_follow", False)
        r.marker_follow = mf if mf == "whole" else bool(mf)
        return r


def fl20p8_hand_state(d_emb: int, big: float = 20.0) -> dict:
    """A hand-set ``relpos`` + routed-marker weight vector that APPROXIMATES ``gold_fl20p8_noslot`` with
    the generic features: gold documents whole (markers included); every other document keeps its
    first 8 body tokens (start0..start7) plus its first and last body decile (rs0, re0); markers of
    partially kept documents are dropped (a fully kept body frees its markers through
    ``mark_positions_free``'s whole-document rule, as the heuristic does). The heuristic itself keeps
    the first 8 plus first_last's top ceil(0.2 (n - 8)) of the rest, so the two differ by rounding and
    by the prefix overlap. DIAGNOSTIC ONLY -- never an initialisation."""
    r = LinearRouter(d_emb, "relpos")
    r.route_markers = True
    with torch.no_grad():
        r.b.fill_(-big)
        r.w_pos[:8] = 2 * big
        r.w_rel[0] = 2 * big  # rs0
        r.w_rel[N_DEC] = 2 * big  # re0
        r.w_marker.fill_(-2 * big)
        r.w_gold.fill_(4 * big)  # gold body: -b + 4b > 0; gold marker: -b - 2b + 4b > 0
    st = r.state()
    st.update({"keep_rule": "p>0.5", "hand": "gold_fl20p8_noslot~"})
    return st


def unit_mean(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Mean of ``v`` per unit id ``u`` (differentiable)."""
    n_u = int(u.max()) + 1
    cnt = torch.bincount(u, minlength=n_u).to(v.dtype)
    return torch.zeros(n_u, dtype=v.dtype, device=v.device).index_add(0, u, v) / cnt


def topk_keep(z: torch.Tensor, feats: Dict[str, torch.Tensor], k: int, span: int = 0) -> torch.Tensor:
    """Keep the top-``k`` routed tokens by score. For span routers the selection is by whole units (spans,
    markers): units in score order until the token budget is met, a unit being taken when at least half of
    it fits (so the realised T2/T rounds to the nearest span)."""
    N = int(z.numel())
    keep = torch.zeros(N, dtype=torch.bool, device=z.device)
    k = min(N, max(0, int(k)))
    if k == 0:
        return keep
    if not span:
        keep[torch.topk(z, k).indices] = True
        return keep
    u = feats[f"unit{span}"].to(z.device)
    zu = unit_mean(z.float(), u)
    size = torch.bincount(u, minlength=int(zu.numel()))
    order = torch.argsort(zu, descending=True, stable=True)
    before = torch.cumsum(size[order], 0) - size[order]
    take = order[(before + size[order] / 2.0) <= k]
    ku = torch.zeros(int(zu.numel()), dtype=torch.bool, device=z.device)
    ku[take] = True
    return ku[u]


def keep_mask_from(feats: Dict[str, torch.Tensor], keep: torch.Tensor, S: int) -> torch.Tensor:
    """(S,) bool body-token keep mask from a per-routed-token decision ``keep`` (N,) bool."""
    m = torch.zeros(S, dtype=torch.bool, device=feats["idx"].device)
    m[feats["idx"][keep.to(feats["idx"].device)]] = True
    return m


# ------------------------------------------------------------------------------------------------
# Differentiable (relaxed) removal -- train_diff_router.py. Hard-concrete gates (Louizos et al.):
# log_alpha = router logit; s = sigmoid((logit(u) + log_alpha) / beta); z = clamp(s*(ZETA-GAMMA)+GAMMA, 0, 1).
# The deterministic test-time gate keeps a token iff log_alpha > 0, i.e. sigmoid(logit) > 0.5 --
# the same rule the grid driver's router schemes apply.
# ------------------------------------------------------------------------------------------------
HC_GAMMA, HC_ZETA = -0.1, 1.1


def hard_concrete(log_alpha: torch.Tensor, beta: float, generator: Optional[torch.Generator] = None,
                  straight_through: bool = False, st_clamp: bool = True) -> torch.Tensor:
    """Sample a hard-concrete gate per token (differentiable w.r.t. ``log_alpha``). With
    ``straight_through`` the forward value is the binary ``z > 0.5`` and the gradient that of ``z``.
    ``st_clamp`` (default): the [0, 1] clamp is straight-through -- the forward value is exactly the
    clamped gate, but a gate sitting at 0 (or 1) still receives the gradient of the stretched sample,
    so a dropped token keeps getting the (finite) dCE/dz signal and can be revived; with the plain
    clamp such "dead" gates never recover (oolong/rerank collapsed to keep ~0 in the first pilot)."""
    u = torch.rand(log_alpha.shape, generator=generator).to(log_alpha.device).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / beta)
    sbar = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
    z = sbar.clamp(0.0, 1.0)
    if st_clamp:
        z = sbar + (z - sbar).detach()
    if straight_through:
        z = (z > 0.5).float() + z - z.detach()
    return z


def hard_concrete_p_nonzero(log_alpha: torch.Tensor, beta: float) -> torch.Tensor:
    """P(z > 0) per token -- the expected-L0 keep fraction the objective minimises."""
    return torch.sigmoid(log_alpha - beta * math.log(-HC_GAMMA / HC_ZETA))


def soft_keep_row(feats: Dict[str, torch.Tensor], z: torch.Tensor, S: int) -> torch.Tensor:
    """(S,) float keep vector: 1 everywhere (markers, prompt, query, answer) except routed tokens,
    which carry their gate ``z`` (keeps the autograd graph to ``z``)."""
    base = torch.ones(S, dtype=z.dtype, device=z.device)
    out = base.index_put((feats["idx"].to(z.device),), z)
    if feats.get("mk_follow") and feats["mk_idx"].numel():
        # markers follow their document: gate = max of the document's body gates (0 if it has none)
        doc = feats["doc"].to(z.device)
        mdoc = feats["mk_doc"].to(z.device)
        n = int(max(int(doc.max()) if doc.numel() else 0, int(mdoc.max()))) + 1
        if feats.get("mk_follow") == "whole":
            # markers kept only when the WHOLE document body is kept (the bar's semantics). Relaxed gate = MEAN of the
            # body gates (1 iff the whole body is kept; smooth). The min (first try, 2026-10-01) was ~0 for every
            # document under hard-concrete noise + keep-dropout, so markers were always removed in training and
            # the router learned to drop gold (textgroups w_gold < 0, test +0.47)
            dmean = RL_unit_mean(z, doc, n)
            out = out.index_put((feats["mk_idx"].to(z.device),), dmean[mdoc])
        else:
            dmax = torch.zeros(n, dtype=z.dtype, device=z.device).scatter_reduce(0, doc, z, "amax", include_self=False)
            out = out.index_put((feats["mk_idx"].to(z.device),), dmax[mdoc])
    return out


def RL_unit_mean(v: torch.Tensor, u: torch.Tensor, n: int) -> torch.Tensor:
    cnt = torch.bincount(u, minlength=n).clamp(min=1).to(v.dtype)
    return torch.zeros(n, dtype=v.dtype, device=v.device).index_add(0, u, v) / cnt


def follow_keep_budget(z: torch.Tensor, feats: Dict[str, torch.Tensor], budget: int, mode="any") -> torch.Tensor:
    """Markers-follow-doc selection under a TOTAL token budget: body tokens in score order, each newly touched
    document also paying for its markers; the largest prefix whose cost (body + markers) <= ``budget``."""
    N = int(z.numel())
    keep = torch.zeros(N, dtype=torch.bool, device=z.device)
    if N == 0 or budget <= 0:
        return keep
    doc = feats["doc"].to(z.device)
    n = int(max(int(doc.max()), int(feats["mk_doc"].max()) if feats["mk_doc"].numel() else 0)) + 1
    mcnt = torch.bincount(feats["mk_doc"].to(z.device), minlength=n)
    order = torch.argsort(z, descending=True, stable=True)
    d_o = doc[order]
    ar = torch.arange(N, device=z.device)
    if mode == "whole":  # a document pays for its markers when its LAST body token gets kept
        last = torch.full((n,), -1, dtype=torch.long, device=z.device).scatter_reduce(0, d_o, ar, "amax")
        newdoc = (last[d_o] == ar).long() * mcnt[d_o]
    else:
        first = torch.full((n,), N, dtype=torch.long, device=z.device).scatter_reduce(0, d_o, ar, "amin")
        newdoc = (first[d_o] == ar).long() * mcnt[d_o]
    cost = ar + 1 + torch.cumsum(newdoc, 0)
    k = int((cost <= budget).sum())
    keep[order[:k]] = True
    return keep


def follow_comp(z_keep: torch.Tensor, feats: Dict[str, torch.Tensor], T: int, mode="any") -> float:
    """Realised T2/T of a body keep decision under markers-follow-doc."""
    N, M = int(feats["idx"].numel()), int(feats["mk_idx"].numel())
    if M == 0:
        return (T - N + int(z_keep.sum())) / T
    touched = torch.zeros(int(max(int(feats["doc"].max()) if N else 0, int(feats["mk_doc"].max()))) + 1, dtype=torch.bool, device=z_keep.device)
    touched[feats["doc"].to(z_keep.device)[z_keep]] = True
    if mode == "whole":
        d = feats["doc"].to(z_keep.device)
        nd = int(touched.numel())
        touched = torch.bincount(d[z_keep], minlength=nd) == torch.bincount(d, minlength=nd)
        touched &= torch.bincount(d, minlength=nd) > 0
    return (T - N - M + int(z_keep.sum()) + int(touched[feats["mk_doc"].to(z_keep.device)].sum())) / T
