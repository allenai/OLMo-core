"""Extract what the canonical learned token routers keep vs ``gold_fl20p8_noslot``, per token, for a
qualitative check of the learned rules. CPU only, no model forward: the canonical routers
(``relpos_noemb`` / ``doc_only``) use only position, gold and marker features.

    PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
    $PY debug/learned_router/viz/extract_keep_masks.py --data-root <copy of devloss_grid/data> \
        --tokenizer <Qwen3.5-0.8B-Base tokenizer dir>

Writes ``keep_masks.json`` (first 2 grid test rows per task, per-token keep decisions) and
``keep_stats.json`` (per-task region keep fractions over ALL grid test rows + top learned weights)
next to this file.

Router mask = the grid driver's ``router_<name>@c<x>`` rule (``ctc_devloss_grid.router_keep_mask``):
per row, top-k routed tokens by logit with k = round(x*T - (T - N)), then
``mark_positions_free(free_markers="mask")`` (routed markers kept iff selected; a fully kept body
frees its markers). Bar mask = ``gold_fl20p8_noslot`` through the same chunk-id functions the model's
``_compact_pooled_soft_tokens`` calls (headers-free with extra_tokens=8, first_last scores,
fractional top-k 0.2, gold docs kept whole, no slots).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LR = os.path.dirname(HERE)
GRID = os.path.join(os.path.dirname(LR), "devloss_grid")
for p in (GRID, LR):
    if p not in sys.path:
        sys.path.insert(0, p)

import ctc_devloss_grid as G  # noqa: E402
from router_lib import (  # noqa: E402
    N_DEC,
    N_OFF,
    POS_NAMES,
    REL_NAMES,
    VARIANTS,
    LinearRouter,
    keep_mask_from,
    routed_features,
)

from olmo_core.data.document_chunk_landmark import RESERVED_IDS  # noqa: E402
from olmo_core.nn.attention.chunked_mask import (  # noqa: E402
    PAD_CHUNK_ID,
    build_chunk_ids_from_tokens,
    mark_doc_headers_free,
    mark_doc_topk_tokens_free,
    mark_positions_free,
)
from olmo_core.nn.attention.pooled_doc_kv import resolve_keep_docs  # noqa: E402
from olmo_core.nn.pooled_soft_token import first_last_scores  # noqa: E402

BAR = "gold_fl20p8_noslot"
W = os.path.join(LR, "weights")

# (entry id, task key, rung, router name (weights/<task>/<name>.pt), target T2/T = the bar's grid T2/T,
#  grid results file with the same test rows, router scheme in that file, extra router (name, scheme, file))
ENTRIES = [
    ("nq@2k", "nq", "2k", "v5_s0_rhobar", 0.338, "results_router_recipe/nq_2k.json", "router_v5@exact"),
    ("niah@2k", "niah", "2k", "v4kd_s0_rhobar", 0.587, "results_router_recipe/niah_2k.json", "router_v5@exact"),
    ("textgroups@2k", "textgroups", "2k", "v5_s0_rhobar", 0.607, "results_router_recipe/textgroups_2k.json", "router_v5@exact"),
    ("rerank@2k", "rerank", "2k", "v5_s0_rhobar", 0.407, "results_router_recipe/rerank_2k.json", "router_v5@exact"),
    ("scifact@2k", "scifact", "2k", "v5_s0_rhobar", 0.436, "results_router_recipe/scifact_2k.json", "router_v5@exact"),
    ("strmatch@2k", "strmatch", "2k", "v5_s0_rhobar", 0.627, "results_router_recipe/strmatch_2k.json", "router_v5@exact"),
    ("grouping@2k", "grouping", "2k", "v5_s10_rhobar", 0.290, "results_router_recipe/grouping_2k.json", "router_v5@exact"),
    ("contradiction@2k", "contradiction", "2k", "v5_s0_rhobar", 0.480, "results_router_recipe/contradiction_2k.json", "router_v5@exact"),
    ("textgroups@32k", "textgroups", "32k", "v5_s0_rhobar", 0.245, "results_router_v5len/textgroups_32k.json", "router_v5_s0_rhobar@c0.245"),
]
# textgroups: also the v6 F0 doc-only router, shown as a 5th per-token field
F0 = {"2k": ("v6F0_s0_rhobar", "results_router_v6/textgroups_v6F0_s0_2k.json", "router_v6F0_s0_rhobar@c0.607"),
      "32k": ("v6F0_s0_rhobar", "results_router_v6/textgroups_v6F0_s0_32k.json", "router_v6F0_s0_rhobar@c0.245")}
SHOW_ROWS = {"2k": [0, 1], "32k": [1]}  # textgroups 32k row 1 = v5's worst row there (+1.09 vs bar +0.09)
N_ROWS = {"2k": 16, "32k": 8}


def load_router(task, name):
    path = os.path.join(W, task, f"{name}.pt")
    st = torch.load(path, map_location="cpu")
    r = LinearRouter.from_state(st)
    assert not r.use_emb, f"{path}: router uses the embedding; this script has no model"
    return r, path, st


def router_mask(router, x, cid0, n_docs, ids, gold, comp):
    """The grid driver's ``router_<name>@c<comp>`` construction; returns (S,) bool of kept positions."""
    feats = routed_features(x, cid0[0], ids.doc_start, ids.doc_end, n_docs, sorted(gold) if gold else None,
                            route_markers=router.route_markers)
    with torch.no_grad():
        z = router.logits(feats, None)
    N, T = int(z.numel()), int(x.shape[0])
    k = min(N, max(0, int(round(comp * T - (T - N)))))
    keep = torch.zeros(N, dtype=torch.bool)
    if k:
        keep[torch.topk(z, k).indices] = True
    m = keep_mask_from(feats, keep, T)[None]
    c2 = mark_positions_free(cid0, x[None], m, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, n_docs=n_docs,
                             free_markers="mask" if router.route_markers else True)
    return ((c2 != PAD_CHUNK_ID) & (c2 < 0))[0]  # keep="none": no document kept whole via keep_docs


def bar_mask(x, cid0, n_docs, ids, gold, seed, ri):
    """``gold_fl20p8_noslot`` through the model's own chunk-id path; returns (S,) bool of kept positions."""
    c1 = mark_doc_headers_free(cid0, x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, stop_id=None,
                               stop_count=1, extra_tokens=8, cap=32)
    sc = first_last_scores(x[None], c1, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, n_docs=n_docs)
    c2 = mark_doc_topk_tokens_free(c1, x[None], sc, doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, k=0.2,
                                   n_docs=n_docs)
    if gold is None:  # gold-blind twin fl20p8_noslot: keep_prob 0 -> no whole document
        keep = resolve_keep_docs(cid0, n_docs, holder=None, keep_prob=0.0, keep_seed=seed).cpu()
    else:
        keep = G.gold_keep_mask(n_docs, gold, 0.0, seed, ri)
    c = c2[0].long()
    return (c != PAD_CHUNK_ID) & ((c < 0) | keep[0].gather(0, c.clamp(min=0)))


def weight_table(router, st):
    v = VARIANTS[router.variant]
    names, vals = [], []
    if v.get("pos", True):
        names += POS_NAMES[: 2 * N_OFF]
        vals += router.w_pos[: 2 * N_OFF].tolist()
        if v.get("contpos", True):
            names += POS_NAMES[2 * N_OFF:]
            vals += router.w_pos[2 * N_OFF:].tolist()
    if v.get("rel", False):
        lo = 2 * N_DEC if v.get("rel_part") == "len" else 0
        names += REL_NAMES[lo:]
        vals += router.w_rel[lo:].tolist()
    if v["gold"]:
        names.append("gold")
        vals.append(float(router.w_gold))
    if router.route_markers:
        names.append("is_marker")
        vals.append(float(router.w_marker))
    order = np.argsort(vals)
    top_pos = [[names[i], round(vals[i], 3)] for i in order[::-1][:10]]
    top_neg = [[names[i], round(vals[i], 3)] for i in order[:10]]
    return {"variant": router.variant, "route_markers": router.route_markers, "bias_b": round(float(router.b), 3),
            "bias_note": "b cancels under per-row top-k (@exact); every body token sums one start-offset, one "
                         "end-offset, one rs, one re and one nl weight (+ gold); a routed marker's logit is "
                         "b + is_marker (+ gold)",
            "top10_positive": top_pos, "top10_negative": top_neg, "n_features": len(names),
            "polish": st.get("polish")}


class Stats:
    def __init__(self):
        self.c = {}

    def add(self, key, kept, n):
        a = self.c.setdefault(key, [0, 0])
        a[0] += int(kept)
        a[1] += int(n)

    def out(self):
        return {k: {"frac": round(a[0] / a[1], 4) if a[1] else None, "n_tokens": a[1]} for k, a in sorted(self.c.items())}


def accumulate(stats, name, keepm, x, cid0, n_docs, ids, gold):
    """Region keep fractions for one mask (name = router | bar | f0)."""
    cid = cid0[0].long()
    mk = (x == ids.doc_start) | (x == ids.doc_end)
    gs = set(gold or ())
    for d in range(n_docs):
        pos = torch.nonzero(cid == d).flatten()
        mpos = pos[mk[pos]]
        bpos = pos[~mk[pos]]
        n = int(bpos.numel())
        kb = keepm[bpos]
        g = d in gs
        tag = "gold" if g else "nongold"
        stats.add(f"{name}/markers_{tag}", int(keepm[mpos].sum()), int(mpos.numel()))
        stats.add(f"{name}/docs_body_dropped_entirely_{tag}", int(n > 0 and int(kb.sum()) == 0), 1)
        stats.add(f"{name}/docs_fully_dropped_incl_markers_{tag}", int(int(keepm[pos].sum()) == 0), 1)
        stats.add(f"{name}/docs_body_kept_whole_{tag}", int(n > 0 and bool(kb.all())), 1)
        if g:
            stats.add(f"{name}/gold_body", int(kb.sum()), n)
            continue
        j = torch.arange(n)
        idr = j < 8
        stats.add(f"{name}/nongold_id_region_j<8", int(kb[idr].sum()), int(idr.sum()))
        rest = ~idr
        dec = (10 * j) // max(n, 1)
        for b in range(10):
            s = rest & (dec == b)
            if int(s.sum()):
                stats.add(f"{name}/nongold_rest_j>=8_decile{b}", int(kb[s].sum()), int(s.sum()))
        stats.add(f"{name}/nongold_rest_j>=8_all", int(kb[rest].sum()), int(rest.sum()))


def trunc(s, n=300):
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = RESERVED_IDS["qwen3_5"]
    dec1 = lambda t: tok.decode([int(t)])  # noqa: E731

    masks_out = {"meta": {
        "token_format": "[decoded piece, router_keep, bar_keep, is_marker] (+ f0_keep as 5th field on textgroups entries: "
                        "the v6 F0 doc-only router at the same exact T2/T)",
        "elision": "docs longer than 120 tokens (markers included): first 60 + last 30 tokens; the middle is one dict "
                   "{elided, router_kept, bar_kept[, f0_kept]}",
        "router_rule": "grid driver router_<name>@c<x>: per-row top-k over routed tokens (body + markers) so T2/T = x "
                       "(= gold_fl20p8_noslot's grid T2/T for the task+rung); CPU torch.topk, so tie-breaking inside a "
                       "tied logit class can differ from the GPU run (T2/T is identical)",
        "bar_rule": "gold_fl20p8_noslot via the model's chunk-id functions (headers-free extra 8, first_last k=0.2, gold "
                    "whole, no slot, markers of partial docs dropped); exact, not approximated",
        "rows": "first rows of the grid rung file (= the grid's test rows)",
    }, "entries": []}
    stats_out = {"meta": {"regions": "token-level keep fractions pooled over ALL grid test rows (16 at 2k, 8 at 32k); "
                                     "j = body offset within the doc; deciles = floor(10 j / n_body) over j >= 8",
                          "eval_size_note": "16 (2k) / 8 (32k) rows"}, "entries": {}}

    for eid, task, rung, rname, comp, rfile, rscheme in ENTRIES:
        row = G.ROSTER[f"ctc_{task}"]
        exs = G.load_examples(a.data_root, row, rung, N_ROWS[rung])
        router, rpath, rst = load_router(task, rname)
        f0 = None
        if task == "textgroups":
            f0r, f0p, f0st = load_router(task, F0[rung][0])
            f0 = (f0r, f0p, f0st)
        res = json.load(open(os.path.join(GRID, rfile)))
        pr = res["per_row"]
        f0res = json.load(open(os.path.join(GRID, F0[rung][1])))["per_row"] if f0 else None
        stats = Stats()
        shown = []
        comp_chk = []
        for ri, ex in enumerate(exs):
            r_ids, r_mask, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
            assert n_spans == len(ex["documents"]), (eid, ri)
            gold = G.gold_docs(row["spec"], ex)
            x = torch.tensor(r_ids)
            cid0 = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end,
                                               eos_id=ids.eos, mode="chunked")
            n_docs = int(cid0.max()) + 1
            T = int(x.numel())
            km_r = router_mask(router, x, cid0, n_docs, ids, gold, comp)
            km_b = bar_mask(x, cid0, n_docs, ids, gold, a.seed, ri)
            km_f = router_mask(f0[0], x, cid0, n_docs, ids, gold, comp) if f0 else None
            chk = {"router_T2T": round(float(km_r.sum()) / T, 4), "grid_router_T2T": round(pr[rscheme]["compaction"][ri], 4),
                   "bar_T2T": round(float(km_b.sum()) / T, 4), "grid_bar_T2T": round(pr[BAR]["compaction"][ri], 4)}
            if f0:
                chk["f0_T2T"] = round(float(km_f.sum()) / T, 4)
                chk["grid_f0_T2T"] = round(f0res[F0[rung][2]]["compaction"][ri], 4)
            comp_chk.append(chk)
            accumulate(stats, "router", km_r, x, cid0, n_docs, ids, gold)
            accumulate(stats, "bar", km_b, x, cid0, n_docs, ids, gold)
            if f0:
                accumulate(stats, "f0", km_f, x, cid0, n_docs, ids, gold)
            if ri not in SHOW_ROWS[rung]:
                continue
            cid = cid0[0].long()
            mk = (x == ids.doc_start) | (x == ids.doc_end)
            ans = [i for i, m in enumerate(r_mask) if m]
            first_doc = int(torch.nonzero(cid >= 0).flatten()[0])
            last_doc = int(torch.nonzero(cid >= 0).flatten()[-1])
            before = tok.decode(r_ids[:first_doc])
            after = tok.decode(r_ids[last_doc + 1: ans[0]]) if ans else ""
            docs = []
            for d in range(n_docs):
                pos = torch.nonzero(cid == d).flatten().tolist()
                toks = []
                for p in pos:
                    t = [dec1(r_ids[p]), int(km_r[p]), int(km_b[p]), int(mk[p])]
                    if f0:
                        t.append(int(km_f[p]))
                    toks.append(t)
                if len(toks) > 120:
                    mid = toks[60:-30]
                    el = {"elided": len(mid), "router_kept": sum(t[1] for t in mid), "bar_kept": sum(t[2] for t in mid)}
                    if f0:
                        el["f0_kept"] = sum(t[4] for t in mid)
                    toks = toks[:60] + [el] + toks[-30:]
                body = [p for p in pos if not bool(mk[p])]
                dd = {"doc": d, "is_gold": bool(gold and d in gold), "n_body": len(body),
                      "router_body_kept": int(km_r[body].sum()) if body else 0,
                      "bar_body_kept": int(km_b[body].sum()) if body else 0}
                if f0:
                    dd["f0_body_kept"] = int(km_f[body].sum()) if body else 0
                dd["tokens"] = toks
                docs.append(dd)
            full = pr["full"]["ce"][ri]
            sr = {"row": ri, "T": T, "n_docs": n_docs, "gold_docs": sorted(gold) if gold is not None else None,
                  "prompt_before_docs": trunc(before), "prompt_after_docs": trunc(after),
                  "answer": trunc(tok.decode([r_ids[i] for i in ans]), 400),
                  "T2T": chk,
                  "grid_dce": {"router": round(pr[rscheme]["ce"][ri] - full, 4), "bar": round(pr[BAR]["ce"][ri] - full, 4)},
                  "docs": docs}
            if f0:
                sr["grid_dce"]["f0"] = round(f0res[F0[rung][2]]["ce"][ri] - f0res["full"]["ce"][ri], 4)
            shown.append(sr)
        ent = {"id": eid, "task": task, "rung": rung, "router_file": os.path.relpath(rpath, os.path.dirname(os.path.dirname(LR))),
               "target_T2T": comp, "grid_results": rfile, "grid_router_scheme": rscheme, "rows": shown}
        if f0:
            ent["f0_router_file"] = os.path.relpath(f0[1], os.path.dirname(os.path.dirname(LR)))
            ent["f0_grid_scheme"] = F0[rung][2]
        masks_out["entries"].append(ent)
        mx = max(max(abs(c["router_T2T"] - c["grid_router_T2T"]), abs(c["bar_T2T"] - c["grid_bar_T2T"])) for c in comp_chk)
        st = {"router_file": ent["router_file"], "target_T2T": comp, "eval_size": len(exs),
              "max_abs_T2T_mismatch_vs_grid": round(mx, 5), "regions": stats.out(),
              "router_weights": weight_table(router, rst)}
        if f0:
            st["f0_router_file"] = ent["f0_router_file"]
            st["f0_weights"] = weight_table(f0[0], f0[2])
            st["f0_max_abs_T2T_mismatch_vs_grid"] = round(max(abs(c["f0_T2T"] - c["grid_f0_T2T"]) for c in comp_chk), 5)
        stats_out["entries"][eid] = st
        print(f"{eid}: rows {len(exs)}, T2T mismatch vs grid {mx:.4f}, shown {len(shown)}", flush=True)

    json.dump(masks_out, open(os.path.join(HERE, "keep_masks.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(stats_out, open(os.path.join(HERE, "keep_stats.json"), "w"), ensure_ascii=False, indent=1)
    print("sizes:", {f: os.path.getsize(os.path.join(HERE, f)) for f in ("keep_masks.json", "keep_stats.json")})


if __name__ == "__main__":
    main()
