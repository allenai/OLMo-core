"""How similar is the training gradient with token dropping to the full-context gradient (direction and magnitude)?

    python research/token_routing/learned_router/gradprobe/grad_similarity.py --tasks nq,outlier,contradiction --rung 8k \\
        --rows 32 --ckpt /data/prasann/devloss_grid/ckpts/setA32k --ckpt-format olmo_st \\
        --data-root /data/prasann/devloss_grid/data --out-prefix /data/prasann/gradprobe/out/gsim_setA32k

Per row: the parameter gradient g_full = dCE/dtheta of the teacher-forced answer CE on the full context, and g_drop for
the same row with document body tokens HARD-dropped (grid semantics: sequence compacted, original RoPE positions --
``FR.compact`` + the flash forward, here with parameters requiring grad). For each (drop rule, keep ratio rho) we store
per-parameter-group dot(g_full, g_drop), ||g_drop||^2 (and ||g_full||^2 once), so the global and per-module cosine and
norm ratio ||g_drop|| / ||g_full|| can be recomputed offline. Groups: embeddings, lm_head, and per block its
direct children (sequence mixer, feed_forward, the norms); the block's mixer class is logged so attention and GDN
layers can be told apart.

Drop rules (body tokens only; markers / prompt / question / answer are always kept, as in grad_importance.py):
  rand      seeded random
  a0.5_abs  |dCE/dg| at every body gate g = 0.5 (one soft pass, gold-blind; the record's best gold-blind rule)
  gold_ga   gold documents first, then fill by the same |grad|@0.5 (oracle)
  bar       gold_fl20p8_noslot at its own per-row budget
References per row: ``keepall`` (compact path with every token kept: must give cosine ~ 1, checks the explicit-position
path), ``rerun`` (the full gradient recomputed: the GPU-nondeterminism floor of the cosine), and ``xrow`` (cosine of this
row's full gradient with the previous row's: what an unrelated gradient looks like).

Secondary (gate gradients): the per-token gate gradient dCE/dg at g = --gate-alpha over the KEPT body tokens, computed in
the full context (all body gates at alpha) vs in the dropped context (kept gates at alpha, dropped at 0 = exact
deletion) -- Spearman of |grad| and cosine of the signed gradient: are importance scores stable once tokens are gone?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grad_importance as GI  # noqa: E402

FR, TR, G, log = GI.FR, GI.TR, GI.G, GI.log


def group_of(name: str) -> str:
    s = name.split(".")
    if s[0] == "blocks":
        return f"blocks.{s[1]}.{s[2]}"
    return s[0]


class GradSim:
    def __init__(self, model):
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]
        self.params = [p_ for _, p_ in model.named_parameters()]
        self.groups = sorted({group_of(n) for n in self.names})
        self.gidx = [self.groups.index(group_of(n)) for n in self.names]
        self.g_full = None
        self.g_prev = None
        self.g_sum, self.n_sum = None, 0  # --demean: sum of full-context gradients (1) or of unit gradients (2), fp32
        self.demean_mode = 0

    def grad(self, x, pred, tgt, pos=None):
        """(CE, list of per-parameter grads (bf16, None if unused)) for one forward."""
        for p_ in self.params:
            p_.requires_grad_(True)
        try:
            kw = {} if pos is None else {"position_ids": pos}
            lg = self.model(x, logits_to_keep=pred[None], **kw)[0].float()
            ce = F.cross_entropy(lg, tgt)
            gs = torch.autograd.grad(ce, self.params, allow_unused=True)
        finally:
            for p_ in self.params:
                p_.requires_grad_(False)
        return float(ce), [None if g is None else g.detach() for g in gs]

    def stats(self, ga, gb, center=None):
        """Per-group dot(ga, gb), ||ga||^2, ||gb||^2 in fp64 (accumulated from fp32 per-parameter products).
        center = (sum of full gradients over N rows, N, this row's g_full): subtract the leave-one-out mean full gradient
        (sum - g_full_i) / (N - 1) from both sides first (removes the row-independent common direction)."""
        n = len(self.groups)
        dot, na, nb = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, (a, b) in enumerate(zip(ga, gb)):
            j = self.gidx[i]
            af = None if a is None else a.float()
            bf = None if b is None else b.float()
            if center is not None and center[0][i] is not None:
                gs, N, gi = center
                mu = (gs[i] - (0 if gi[i] is None else gi[i].float())) / (N - 1)
                af = -mu if af is None else af - mu
                bf = -mu if bf is None else bf - mu
            if af is not None:
                na[j] += float((af * af).sum())
            if bf is not None:
                nb[j] += float((bf * bf).sum())
            if af is not None and bf is not None:
                dot[j] += float((af * bf).sum())
        return dot, na, nb


def proj_stats(GS, ga, gb, gi, ni):
    """Scale-invariant centering (--demean 2): project the leave-one-out mean DIRECTION of the task's full gradients
    out of both ga and gb, then take their cosine. m = sum_j g_j/|g_j| - g_i/|g_i| (GS.g_sum holds sum of unit vectors);
    cos = (a.b - (a.m)(b.m)/(m.m)) / sqrt((|a|^2 - (a.m)^2/(m.m)) (|b|^2 - (b.m)^2/(m.m))). Unlike subtracting the mean
    gradient, rows whose gradient is 1000x larger (saturated SFT models) cannot dominate it."""
    ab = aa = bb = am = bm = mm = 0.0
    for i, (a, b) in enumerate(zip(ga, gb)):
        if GS.g_sum[i] is None:
            continue
        m = GS.g_sum[i] - (0 if gi[i] is None else gi[i].float() / ni)
        af = None if a is None else a.float()
        bf = None if b is None else b.float()
        mm += float((m * m).sum())
        if af is not None:
            aa += float((af * af).sum()); am += float((af * m).sum())
        if bf is not None:
            bb += float((bf * bf).sum()); bm += float((bf * m).sum())
        if af is not None and bf is not None:
            ab += float((af * bf).sum())
    mm = max(mm, 1e-300)
    num = ab - am * bm / mm
    den = math.sqrt(max((aa - am * am / mm) * (bb - bm * bm / mm), 1e-300))
    return {"cos": num / den, "frac_a_on_mean": (am * am / mm) / max(aa, 1e-300), "frac_b_on_mean": (bm * bm / mm) / max(bb, 1e-300)}


def summ_stats(dot, na, nb):
    D, A, B = dot.sum(), na.sum(), nb.sum()
    return {"cos": float(D / math.sqrt(max(A * B, 1e-300))), "norm_ratio": float(math.sqrt(B / max(A, 1e-300))),
            "norm_full": float(math.sqrt(A)), "norm_drop": float(math.sqrt(B))}


def flk_mask(p, P, k, stop_id, first_only=False):
    """ds64's outlier frontier rule (flk64 / flk64ns, research/token_routing/ds64/launch_ds64.py): GOLD-BLIND, each doc's
    header up to the first ``stop_id`` (Qwen3.5 5491 = ']:', i.e. 'Document [N]:') + first k//2 and last k - k//2 body
    tokens real, everything else dropped (the slot-less twin's hard drop)."""
    from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, mark_doc_headers_free, mark_doc_topk_tokens_free
    from olmo_core.nn.pooled_soft_token import first_last_scores

    ids_ = G.RESERVED_IDS[G.FAMILY]
    xs_, cid = p["x"][0].cpu(), p["cid"]
    n_docs_ = int(cid.max()) + 1
    # first_only: ds64's ck<k> (--st-header-extra-tokens k: the FIRST k body tokens after the header)
    c = mark_doc_headers_free(cid[None].clone(), xs_[None], doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end,
                              stop_id=stop_id, stop_count=1, extra_tokens=k if first_only else 0, cap=32)
    # (stop_id None: no header stop -- the doc keeps only its first k tokens, as nq's nqk<k> arms)
    if not first_only:
        sc = first_last_scores(xs_[None], c, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, n_docs=n_docs_)
        c = mark_doc_topk_tokens_free(c, xs_[None], sc, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, k=float(k),
                                      n_docs=n_docs_)
    real = (c[0] == FREE_CHUNK_ID) | (cid < 0)
    return real[P.body_idx.cpu()].to(P.body_idx.device)


def frac_mask(p, r, P, kind, frac, stop_id, row_seed):
    """ds64's whole-document keep arms, hard-drop (slot-less) form (research/token_routing/ds64/launch_ds64.py):
      kv  F       gold docs + round(F * n_non_gold) (>= 1) random non-gold docs whole; every other doc fully dropped
                  (--st-keep-mode gold_plus_random --st-keep-frac F: kv33 / kv17)
      hdr F:STOP  same whole-doc set, but every other doc keeps its header up to STOP real (hdr33/hdr17/hdr08/hdr03)
      gb  F       GOLD-BLIND: each doc whole with probability F, the rest fully dropped (--st-gold-blind
                  --st-keep-prob F: kvgb)"""
    from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, mark_doc_headers_free

    ids_ = G.RESERVED_IDS[G.FAMILY]
    xs_, cid = p["x"][0].cpu(), p["cid"]
    n_docs_ = int(cid.max()) + 1
    rng = np.random.default_rng(row_seed)
    gold_l = sorted({int(g_) for g_ in (r["gold"] or [])})
    if kind == "gb":
        whole = {d for d in range(n_docs_) if rng.random() < frac}
    else:
        non = [d for d in range(n_docs_) if d not in gold_l]
        n_rand = max(1, int(round(frac * len(non)))) if non else 0
        whole = set(gold_l) | set(rng.choice(non, size=min(n_rand, len(non)), replace=False).tolist())
    if kind == "hdr":
        c = mark_doc_headers_free(cid[None].clone(), xs_[None], doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end,
                                  stop_id=stop_id, stop_count=1, extra_tokens=0, cap=32)
        real = (c[0] == FREE_CHUNK_ID) | (cid < 0)
    else:
        real = cid < 0
    for d in whole:
        real = real | (cid == d)
    return real[P.body_idx.cpu()].to(P.body_idx.device)


def gold_leak_auc(keep, P, r, cid):
    """Mask-only gold leakage: AUC with which a document's KEPT FRACTION ranks gold docs above non-gold docs
    (0.5 = the mask says nothing about which docs are gold; 1.0 = kept-ness identifies gold). Model-free."""
    gold_l = {int(g_) for g_ in (r["gold"] or [])}
    if not gold_l:
        return float("nan")
    d = cid[P.body_idx.cpu()].cpu().numpy().astype(np.int64)
    k = keep.detach().cpu().numpy().astype(np.float64)
    if d.size == 0 or d.min() < 0:
        d = np.where(d < 0, 0, d)
    n = int(d.max()) + 1
    tot = np.bincount(d, minlength=n).astype(np.float64)
    kept = np.bincount(d, weights=k, minlength=n)
    frac = np.where(tot > 0, kept / np.maximum(tot, 1), np.nan)
    g = [frac[i] for i in range(n) if i in gold_l and tot[i] > 0]
    ng = [frac[i] for i in range(n) if i not in gold_l and tot[i] > 0]
    if not g or not ng:
        return float("nan")
    ga, na = np.array(g)[:, None], np.array(ng)[None, :]
    return float(((ga > na).sum() + 0.5 * (ga == na).sum()) / (ga.size * na.size))


def barfam_masks(p, r, P, xs):
    """The bar's family (``grad_importance.py --fl-sweep`` construction, heuristic_real's): [gold docs] + [first 8 tokens of
    every doc] + first/last x% of every doc's body. x = 20 with gold IS the bar (gold_fl20p8_noslot). Yields
    (name, keep_body, x) for the gold ("goldhfl") and gold-blind ("hfl") variants."""
    from olmo_core.nn.attention.chunked_mask import FREE_CHUNK_ID, mark_doc_headers_free, mark_doc_topk_tokens_free
    from olmo_core.nn.pooled_soft_token import first_last_scores

    ids_ = G.RESERVED_IDS[G.FAMILY]
    xs_, cid = p["x"][0].cpu(), p["cid"]
    n_docs_ = int(cid.max()) + 1
    gold_l = [int(g_) for g_ in (r["gold"] or [])]
    bidx = P.body_idx.cpu()
    for xf in [float(v) for v in xs.split(",")]:
        c = mark_doc_headers_free(cid[None].clone(), xs_[None], doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, stop_id=None,
                                  stop_count=1, extra_tokens=8, cap=32)
        if xf > 0:
            sc = first_last_scores(xs_[None], c, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end, n_docs=n_docs_)
            c = mark_doc_topk_tokens_free(c, xs_[None], sc, doc_start_id=ids_.doc_start, doc_end_id=ids_.doc_end,
                                          k=xf / 100.0, n_docs=n_docs_)
        real = (c[0] == FREE_CHUNK_ID) | (cid < 0)
        yield "hfl", real[bidx].to(P.body_idx.device), xf
        if gold_l:
            for g_ in gold_l:
                real = real | (cid == g_)
            yield "goldhfl", real[bidx].to(P.body_idx.device), xf


def run_row(model, GS, p, r, a, dev):
    P = GI.Prober(model, p, dev, ckpt_blocks=False)
    T = p["T"]
    out = {"T": T, "n_body": P.Nb, "gold_docs": sorted(r["gold"] or []), "ce_full": p["ce_full"], "checks": {}, "points": []}
    t0 = time.time()

    def compact(keep_body):
        real = torch.ones(T, dtype=torch.bool, device=dev)
        real[P.body_idx[~keep_body.to(dev)]] = False
        return FR.compact(p, real, torch.zeros(T, dtype=torch.bool))

    # full-context gradient (plain forward, as base_prep's CE) + its rerun + the keep-all compact path
    ce_f, g_full = GS.grad(p["x"], p["pred"], p["tgt"])
    out["checks"]["ce_grad_vs_base_prep"] = ce_f - p["ce_full"]
    dot, na, _ = GS.stats(g_full, g_full)
    out["full_group_norm2"] = na.tolist()
    ctr = (GS.g_sum, GS.n_sum, g_full) if (GS.g_sum is not None and GS.demean_mode == 1) else None
    prj = GS.g_sum is not None and GS.demean_mode == 2
    n_full = math.sqrt(float(np.sum(na))) if prj else None
    if ctr is not None:
        _, nac, _ = GS.stats(g_full, g_full, ctr)
        out["full_group_norm2_centered"] = nac.tolist()
    ce_r, g_r = GS.grad(p["x"], p["pred"], p["tgt"])
    d, n1, n2 = GS.stats(g_full, g_r)
    out["rerun"] = {**summ_stats(d, n1, n2), "dce": ce_r - ce_f}
    del g_r
    it = compact(torch.ones(P.Nb, dtype=torch.bool, device=dev))
    ce_k, g_k = GS.grad(it["x"], it["pred"], it["tgt"], it["pos"])
    d, n1, n2 = GS.stats(g_full, g_k)
    out["keepall"] = {**summ_stats(d, n1, n2), "dce": ce_k - ce_f}
    del g_k
    if GS.g_prev is not None:
        d, n1, n2 = GS.stats(g_full, GS.g_prev)
        out["xrow"] = summ_stats(d, n1, n2)
        if ctr is not None:
            out["xrow_centered"] = summ_stats(*GS.stats(g_full, GS.g_prev, ctr))
        if prj:
            out["xrow_proj"] = proj_stats(GS, g_full, GS.g_prev, g_full, n_full)

    # scoring pass for the gradient rules (soft path, params frozen)
    ones = torch.ones(P.Nb, device=dev)
    al = a.gate_alpha
    _, gz_half, _ = P.soft_ce_grad(torch.full_like(ones, 0.5))
    _, gz_full_ctx, _ = P.soft_ce_grad(torch.full_like(ones, al)) if al != 0.5 else (None, gz_half, None)
    a05 = gz_half.abs()
    gold_t = torch.tensor(P.gold, device=dev, dtype=torch.float32)
    rules = {"rand": torch.rand(P.Nb, generator=torch.Generator().manual_seed(4242 + len(out["gold_docs"]))).to(dev),
             "a0.5_abs": a05}
    if P.gold.any():
        rules["gold_ga"] = 2.0 * gold_t + a05 / a05.max().clamp(min=1e-30)
    bar_body = p["real_bar"].to(dev)[P.body_idx]
    if a.rule_set == "fixed":
        configs = [(nm, rho, P.topk(sc, int(round(rho * P.Nb))), None) for rho in a.keeps for nm, sc in rules.items()]
        configs.append(("bar", float(bar_body.float().mean()), bar_body, None))
    else:  # "barfam": the bar's own family at several x, plus random / |grad|@0.5 at each row's matched budget
        configs = []
        for nm, keep, xf in barfam_masks(p, r, P, a.fl_sweep):
            k = int(keep.sum())
            configs.append((nm, k / max(P.Nb, 1), keep, xf))
            if nm == "goldhfl":  # matched-budget references (per row, same k as the gold variant)
                configs.append(("rand", k / max(P.Nb, 1), P.topk(rules["rand"], k), xf))
                configs.append(("a0.5_abs", k / max(P.Nb, 1), P.topk(a05, k), xf))
                if abs(xf - 20) < 1e-9:
                    out["checks"]["goldhfl20_vs_bar_mismatch"] = int((keep != bar_body).sum())
        specs = [(v, False) for v in a.flk.split(",") if v] + [(v, True) for v in a.ck.split(",") if v]
        for spec, first_only in specs:  # ds64-style gold-blind first/last-k (flk) or first-k (ck) rules + matched refs
            kk_s, sid_s = spec.split(":")
            kk, sid = int(kk_s), (None if sid_s.lower() == "none" else int(sid_s))
            keep = flk_mask(p, P, kk, sid, first_only=first_only)
            k = int(keep.sum())
            nm = f"{'ck' if first_only else 'flk'}{kk}"
            configs.append((nm, k / max(P.Nb, 1), keep, float(kk)))
            if a.matched_refs:
                configs.append((f"rand@{nm}", k / max(P.Nb, 1), P.topk(rules["rand"], k), float(kk)))
                configs.append((f"a0.5_abs@{nm}", k / max(P.Nb, 1), P.topk(a05, k), float(kk)))
        for j, spec in enumerate(v for v in a.frac.split(",") if v):  # ds64 whole-document arms (kv / hdr / gb)
            parts = spec.split(":")
            kind, fr = parts[0], float(parts[1])
            sid = int(parts[2]) if len(parts) > 2 and parts[2].lower() != "none" else None
            keep = frac_mask(p, r, P, kind, fr, sid, row_seed=9100 + 37 * j + p["T"])
            configs.append((f"{kind}{round(fr * 100)}", int(keep.sum()) / max(P.Nb, 1), keep, fr))
    for nm, rho, keep, xf in configs:
        it = compact(keep)
        ce_d, g_d = GS.grad(it["x"], it["pred"], it["tgt"], it["pos"])
        d, n1, n2 = GS.stats(g_full, g_d)
        dc = GS.stats(g_full, g_d, ctr) if ctr is not None else None
        dp = proj_stats(GS, g_full, g_d, g_full, n_full) if prj else None
        del g_d
        pt = {"rule": nm, "rho": rho, "k": int(keep.sum()), "T2": it["T2"], "ce": ce_d, "dce": ce_d - ce_f, "x": xf,
              "gold_kept": float(keep.cpu().numpy()[P.gold].mean()) if P.gold.any() else float("nan"),
              "gold_leak_auc": gold_leak_auc(keep, P, r, p["cid"]),
              **summ_stats(d, n1, n2), "group_dot": d.tolist(), "group_norm2_drop": n2.tolist()}
        if dc is not None:
            pt["centered"] = {**summ_stats(*dc), "group_dot": dc[0].tolist(), "group_norm2_drop": dc[2].tolist()}
        if dp is not None:
            pt["proj"] = dp
        # gate gradients over the kept tokens: full context vs dropped context (soft path at a binary mask = deletion)
        if a.gate and int(keep.sum()) >= 3:
            _, gz_drop, _ = P.soft_ce_grad(keep.float() * al)
            u, v = gz_full_ctx[keep].float(), gz_drop[keep].float()
            pt["gate"] = {"cos_signed": float((u * v).sum() / (u.norm() * v.norm()).clamp(min=1e-30)),
                          "spearman_abs": GI.spearman(u.abs().cpu().numpy(), v.abs().cpu().numpy()),
                          "norm_ratio": float(v.norm() / u.norm().clamp(min=1e-30)),
                          "top10pct_overlap": _topk_overlap(u.abs(), v.abs(), 0.1)}
        out["points"].append(pt)
    GS.g_prev = g_full
    out["sec"] = time.time() - t0
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return out


def _topk_overlap(u, v, frac):
    k = max(1, int(round(frac * u.numel())))
    a_ = set(torch.topk(u, k).indices.tolist())
    b_ = set(torch.topk(v, k).indices.tolist())
    return len(a_ & b_) / k


def summarize(rows, floor):
    """Per (rule, rho): median / mean / quartiles of global cosine and log norm ratio over rows; also restricted to rows
    with CE_full >= floor (saturated finetuned rows have tiny, noisy full gradients)."""
    s = {"eval_size": len(rows)}
    for key in ("rerun", "keepall", "xrow"):
        v = [r[key]["cos"] for r in rows if key in r]
        if v:
            s[key] = {"cos_median": float(np.median(v)), "cos_min": float(np.min(v)), "n": len(v)}
    pts = {}
    for r in rows:
        for q in r["points"]:
            k = (q["rule"], "own" if q["rule"] == "bar" else round(q["rho"], 4))
            pts.setdefault(k, []).append((q, r["ce_full"]))
    for (nm, rho), v in sorted(pts.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        def agg(sel):
            if not sel:
                return None
            c = np.array([q["cos"] for q, _ in sel])
            lr = np.log10(np.array([q["norm_ratio"] for q, _ in sel]))
            o = {"n": len(sel), "cos_median": float(np.median(c)), "cos_q25": float(np.quantile(c, 0.25)),
                 "cos_q75": float(np.quantile(c, 0.75)), "cos_mean": float(c.mean()),
                 "log10_norm_ratio_median": float(np.median(lr)), "dce_mean": float(np.mean([q["dce"] for q, _ in sel])),
                 "rho_mean": float(np.mean([q["rho"] for q, _ in sel])),
                 "gold_leak_auc_median": float(np.nanmedian([q.get("gold_leak_auc", float("nan")) for q, _ in sel]))
                 if any(np.isfinite(q.get("gold_leak_auc", float("nan"))) for q, _ in sel) else None}
            gs = [q["gate"] for q, _ in sel if "gate" in q]
            if gs:
                o["gate_cos_median"] = float(np.median([g["cos_signed"] for g in gs]))
                o["gate_spearman_median"] = float(np.nanmedian([g["spearman_abs"] for g in gs]))
                o["gate_top10_overlap_median"] = float(np.median([g["top10pct_overlap"] for g in gs]))
            return o
        s[f"{nm}@{rho}"] = {"all": agg(v), f"ce_full>={floor}": agg([t for t in v if t[1] >= floor])}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="nq,outlier,contradiction")
    ap.add_argument("--rung", default="8k")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--row-offset", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-format", default="hf")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--keeps", default="0.5,0.2,0.1,0.05,0.02")
    ap.add_argument("--rule-set", default="fixed", choices=["fixed", "barfam"],
                    help="fixed: rand / |grad|@0.5 / gold_ga at --keeps + the bar; barfam: the bar's family (gold + 8 header "
                         "tokens + first/last x%%, x in --fl-sweep; and its gold-blind twin) with random and |grad|@0.5 at "
                         "each row's matched budget")
    ap.add_argument("--fl-sweep", default="0,5,10,20,40", help="barfam: x values (percent); x = 20 with gold is the bar")
    ap.add_argument("--flk", default="", help="barfam: extra ds64-style rules K:STOP_ID (gold-blind header up to STOP_ID + "
                    "first/last K body tokens per doc), e.g. 64:5491 = outlier's flk64ns frontier rule on Qwen3.5")
    ap.add_argument("--ck", default="", help="barfam: like --flk but the FIRST K body tokens (ds64's ck<K>), e.g. 64:5491")
    ap.add_argument("--frac", default="", help="barfam: ds64 whole-document arms KIND:F[:STOP], KIND in kv (gold + F random "
                    "non-gold whole, rest dropped), hdr (same + headers up to STOP real), gb (gold-blind, each doc whole w.p. F), "
                    "e.g. kv:0.333,hdr:0.333:25,gb:0.333")
    ap.add_argument("--matched-refs", type=int, default=1, help="barfam: also run rand/|grad| at each flk/ck rule's budget")
    ap.add_argument("--gate", type=int, default=1, help="secondary gate-gradient stability measurement (0 = off)")
    ap.add_argument("--gate-alpha", type=float, default=0.5)
    ap.add_argument("--ce-floor", type=float, default=0.05)
    ap.add_argument("--demean", type=int, default=0, help="1: also report cosines after subtracting the leave-one-out mean "
                    "full-context gradient over the task's rows; 2: project out the leave-one-out mean DIRECTION of the unit "
                    "full gradients (scale-invariant; use for saturated SFT checkpoints). One extra backward per row, +16 GB fp32")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_<task>_<rung>_r<rows>.json")
    a = ap.parse_args()
    a.keeps = sorted({float(k) for k in a.keeps.split(",")}, reverse=True)
    from transformers import AutoTokenizer

    from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

    ids = G.RESERVED_IDS[G.FAMILY]
    vocab = G.VOCAB_BY_FAMILY[G.FAMILY]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tasks = a.tasks.split(",")
    rows_by = {t: TR.load_split(a.data_root, t, [a.rung], tok, ids, a.row_offset + a.rows)[a.row_offset:] for t in tasks}
    all_ids = np.concatenate([np.asarray(r["ids"], dtype=np.int64) for t in tasks for r in rows_by[t]])
    pieces = tok.convert_ids_to_tokens(list(range(vocab)))
    stop_ids, _, tables = G.build_tables(all_ids, vocab, pieces, ids, lambda t: tok.decode([int(t)]))
    if a.ckpt_format == "olmo_st":
        model = GI.load_olmo_safetensors(a.ckpt, vocab, ids, stop_ids, a.attn_backend)
    else:
        model = G.load_model(a.ckpt, a.ckpt_format, vocab, ids, 42, stop_ids, a.attn_backend)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    freeze_fla_length_autotune()
    dev = next(model.parameters()).device
    TR.configure(model, stop_ids, tables.to(dev))
    model.eval()
    GS = GradSim(model)
    GS.demean_mode = a.demean
    mixers = {k: {n: type(m).__name__ for n, m in b.named_children()} for k, b in model.blocks.items()}
    log(f"{len(GS.params)} params in {len(GS.groups)} groups; block 0 children {mixers[next(iter(mixers))]}")
    t_all = time.time()
    n_tot, n_done = sum(len(v) for v in rows_by.values()), 0
    for task in tasks:
        res, rows = [], rows_by[task]
        GS.g_prev = None
        out = f"{a.out_prefix}_{task}_{a.rung}_r{len(rows)}.json"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        GS.g_sum, GS.n_sum = None, 0
        if a.demean:
            t_d = time.time()
            for r in rows:
                with torch.no_grad():
                    p = FR.base_prep(model, r, ids, dev)
                _, g = GS.grad(p["x"], p["pred"], p["tgt"])
                sc = 1.0
                if a.demean == 2:  # sum of UNIT full gradients (scale-invariant mean direction)
                    sc = 1.0 / max(math.sqrt(sum(float((t.float() ** 2).sum()) for t in g if t is not None)), 1e-30)
                if GS.g_sum is None:
                    GS.g_sum = [None if t is None else t.float() * sc for t in g]
                else:
                    for k, t in enumerate(g):
                        if t is not None:
                            GS.g_sum[k] = t.float() * sc if GS.g_sum[k] is None else GS.g_sum[k].add_(t.float() * sc)
                GS.n_sum += 1
                del g
            log(f"{task}: demean pre-pass over {GS.n_sum} rows ({time.time() - t_d:.0f}s)")
        for i, r in enumerate(rows):
            with torch.no_grad():
                p = FR.base_prep(model, r, ids, dev)
            o = run_row(model, GS, p, r, a, dev)
            res.append(o)
            n_done += 1
            el = time.time() - t_all
            log(f"{task} row {i + 1}/{len(rows)} T {o['T']} body {o['n_body']} CE_full {o['ce_full']:.4f} | rerun cos {o['rerun']['cos']:.4f} "
                f"keepall cos {o['keepall']['cos']:.4f} (dCE {o['keepall']['dce']:+.1e})"
                + (f" xrow cos {o['xrow']['cos']:+.3f}" if "xrow" in o else "")
                + (f" (centered {o['xrow_centered']['cos']:+.3f})" if "xrow_centered" in o else "")
                + (f" (proj {o['xrow_proj']['cos']:+.3f})" if "xrow_proj" in o else "") + f" | {o['sec']:.0f}s, peak {o['peak_gb']:.0f} GB, "
                f"ETA {el / n_done * (n_tot - n_done) / 60:.0f} min")
            if i < 2 or i in (4, 9):
                for q in o["points"]:
                    g = q.get("gate", {})
                    log(f"   {q['rule']:9s} rho {q['rho']:.3f}: cos {q['cos']:+.3f}"
                        + (f" (centered {q['centered']['cos']:+.3f})" if "centered" in q else "")
                        + (f" (proj {q['proj']['cos']:+.3f})" if "proj" in q else "")
                        + f" |g_d|/|g_f| {q['norm_ratio']:.3g} dCE {q['dce']:+.3f}"
                        + (f" | gate cos {g['cos_signed']:+.3f} spearman {g['spearman_abs']:+.3f}" if g else ""))
            json.dump({"rows": res, "argv": sys.argv, "task": task, "rung": a.rung, "groups": GS.groups, "mixers": mixers},
                      open(out, "w"))
        summ = summarize(res, a.ce_floor)
        log(f"SUMMARY {task} " + json.dumps(summ, indent=1))
        json.dump({"summary": summ, "rows": res, "argv": sys.argv, "task": task, "rung": a.rung, "groups": GS.groups,
                   "mixers": mixers}, open(out, "w"))
        GS.g_prev, GS.g_sum = None, None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
