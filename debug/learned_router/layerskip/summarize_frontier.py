"""Compute-vs-loss frontier from ``frontier.py`` runs (``runs/<task>/fr_*.json``, configs merged).

    python debug/learned_router/layerskip/summarize_frontier.py [--tasks nq,outlier] [--export-grid]

Candidates per task: the full model (FLOPs 1, dCE 0) and every (token config, layer keep) point,
keep 1.0 = token router only (soft path). For each tolerance tier tau (paired dCE vs FULL), the
candidate with the lowest mean VAL FLOPs whose mean VAL dCE <= tau is selected (near-ties within 2% of
those FLOPs go to the lower val dCE); its TEST64 numbers
are reported (dCE +- paired SE, FLOPs, and whether test meets tau). References on test64: the bar
(gold_fl20p8_noslot), uniform random layer skip at the selected keep on the same token router, and
the grid's heuristic layer-skip schemes (``devloss_grid/results_layerskip_t64``).
``router_v6a+ls``: on the bar-budget token router (f = 1), the most aggressive layer keep whose VAL
dCE vs token-only is within max(SE, 0.005).

``--export-grid`` writes ``devloss_grid/results_layerskip/<task>_2k.json`` (the grid's 16 test rows at 2k;
schemes ``router_v6a+ls`` and ``router_frontier_t<tau>``, each with a summary ``flops`` field) and
``frontier.json`` next to this file.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(os.path.dirname(HERE)), "devloss_grid")
sys.path.insert(0, GRID)
TIERS = (0.02, 0.05, 0.10)


def se(d):
    d = np.asarray(d, float)
    return float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")


def load(task):
    files = sorted(glob.glob(os.path.join(HERE, "runs", task, "fr_*.json")))
    if not files:
        return None
    base = None
    for f in files:
        r = json.load(open(f))
        if base is None:
            base = r
            base["configs"] = dict(r["configs"])
        else:
            for k, v in r["configs"].items():
                base["configs"].setdefault(k, v)
    return base


def candidates(r):
    """(name, label, keep, {split: {"ce": [...], "flops": [...]}})"""
    full = r["full"]
    out = [("full", None, None, {s: {"ce": full[s], "flops": [1.0] * len(full[s])} for s in full})]
    for label, c in r["configs"].items():
        out.append((f"{label}+k1", label, 1.0, {s: {"ce": c["splits"][s]["tok"], "flops": c["splits"][s]["tok_flops"]} for s in c["splits"]}))
        for k, pt in c["points"].items():
            out.append((f"{label}+k{k}", label, float(k), {s: {"ce": pt[s]["ce"], "flops": pt[s]["flops"]} for s in c["splits"] if s in pt}))
    return out


def stats(r, cand, split):
    d = np.asarray(cand[3][split]["ce"]) - np.asarray(r["full"][split])
    return {"dce": float(d.mean()), "se": se(d), "flops": float(np.mean(cand[3][split]["flops"])), "n": len(d)}


def grid_skip(task):
    f = os.path.join(GRID, "results_layerskip_t64", f"{task}_2k.json")
    if not os.path.exists(f):
        return {}
    import collect_grid as CG

    d = json.load(open(f))
    full = np.asarray(d["per_row"]["full"]["ce"])
    out = {}
    for s, pr in d["per_row"].items():
        if s == "full":
            continue
        dd = np.asarray(pr["ce"]) - full
        out[s] = {"dce": float(dd.mean()), "se": se(dd),
                  "flops": CG.flops_ratio(s, d.get("row_lens"), pr.get("compaction"), pr.get("layer_frac"), 0.0)}
    return out


def v6a_ls(r):
    """router_v6a+ls selection on the f = 1 config (label b1.0)."""
    c = r["configs"].get("b1.0")
    if c is None:
        return None
    tok = np.asarray(c["splits"]["val"]["tok"])
    best = 1.0
    for k in sorted((float(x) for x in c["points"]), reverse=True):
        d = np.asarray(c["points"][f"{k:g}"]["val"]["ce"]) - tok
        if d.mean() <= max(se(d), 0.005):
            best = k
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="nq,outlier,scifact,contradiction")
    ap.add_argument("--export-grid", action="store_true")
    a = ap.parse_args()
    summary = {}
    for task in a.tasks.split(","):
        r = load(task)
        if r is None:
            print(f"== {task}: no runs")
            continue
        cands = [c for c in candidates(r) if all(s in c[3] for s in ("val", "test64"))]
        n64 = len(r["full"]["test64"])
        bar = r["bar"]["test64"]
        db = np.asarray(bar["ce"]) - np.asarray(r["full"]["test64"])
        print(f"\n== {task}  (test64 eval_size {n64} ⚠ <500; val {len(r['full']['val'])}) configs {sorted(r['configs'])}")
        print(f"   bar gold_fl20p8_noslot: test64 dCE {db.mean():+.4f} ± {se(db):.4f}, FLOPs ×{np.mean(bar['flops']):.3f}")
        print(f"   {'candidate':14s} {'val dCE':>9s} {'val FL':>7s} | {'test64 dCE':>17s} {'FLOPs':>7s}")
        for c in sorted(cands, key=lambda c: stats(r, c, "val")["flops"]):
            v, t = stats(r, c, "val"), stats(r, c, "test64")
            print(f"   {c[0]:14s} {v['dce']:+9.4f} {v['flops']:7.3f} | {t['dce']:+8.4f} ± {t['se']:.4f} {t['flops']:7.3f}")
        task_sum = {"bar": {"dce": float(db.mean()), "se": se(db), "flops": float(np.mean(bar["flops"]))}, "tiers": {}, "eval_size": n64}
        gs = grid_skip(task)
        for tau in TIERS:
            ok = [c for c in cands if stats(r, c, "val")["dce"] <= tau]
            # cheapest on val; near-ties (val FLOPs within 2% of the cheapest) go to the lower val dCE
            fmin = min(stats(r, c, "val")["flops"] for c in ok)
            sel = min((c for c in ok if stats(r, c, "val")["flops"] <= 1.02 * fmin), key=lambda c: stats(r, c, "val")["dce"])
            t = stats(r, sel, "test64")
            rec = {"candidate": sel[0], "label": sel[1], "keep": sel[2], **{f"test_{k}": v for k, v in t.items()},
                   "val": stats(r, sel, "val"), "meets": t["dce"] <= tau, "speedup": 1.0 / t["flops"]}
            if sel[1] is not None and sel[2] is not None and sel[2] < 1.0:
                rnd = r["configs"][sel[1]]["random"].get(f"{sel[2]:g}", {}).get("test64")
                if rnd:
                    dr = np.asarray(rnd["ce"]) - np.asarray(r["full"]["test64"])
                    rec["random_same_keep"] = {"dce": float(dr.mean()), "se": se(dr), "flops": float(np.mean(rnd["flops"]))}
            # oracle (selected on TEST) for reference only
            okt = [c for c in cands if stats(r, c, "test64")["dce"] <= tau]
            o = min(okt, key=lambda c: stats(r, c, "test64")["flops"])
            rec["test_oracle"] = {"candidate": o[0], "flops": stats(r, o, "test64")["flops"]}
            task_sum["tiers"][f"{tau:g}"] = rec
            rs = rec.get("random_same_keep")
            print(f"   tier ≤{tau:.2f}: {sel[0]:12s} test64 dCE {t['dce']:+.4f} ± {t['se']:.4f} FLOPs ×{t['flops']:.3f} ({rec['speedup']:.2f}× fewer) "
                  f"{'meets' if rec['meets'] else 'MISSES'}" + (f" | random same keep {rs['dce']:+.3f} ×{rs['flops']:.3f}" if rs else "")
                  + f" | test-oracle {o[0]} ×{rec['test_oracle']['flops']:.3f}")
        k_ls = v6a_ls(r)
        task_sum["v6a_ls_keep"] = k_ls
        task_sum["grid_skip"] = gs
        if gs:
            print("   grid heuristic layer skip (test64): " + "; ".join(f"{s} {v['dce']:+.3f} ×{v['flops']:.3f}" for s, v in gs.items()))
        print(f"   router_v6a+ls: layer keep {k_ls}")
        summary[task] = task_sum
        if a.export_grid:
            export_grid(task, r, task_sum)
    if summary:
        for tau in TIERS:
            fl = [s["tiers"][f"{tau:g}"]["test_flops"] for s in summary.values()]
            print(f"MEAN over {len(fl)} tasks, tier ≤{tau:.2f}: FLOPs ×{np.mean(fl):.3f} (= {1 / np.mean(fl):.2f}× fewer); "
                  f"mean per-task speedup {np.mean([1 / x for x in fl]):.2f}×; meets on test {sum(s['tiers'][f'{tau:g}']['meets'] for s in summary.values())}/{len(fl)}")
        print(f"MEAN bar: FLOPs ×{np.mean([s['bar']['flops'] for s in summary.values()]):.3f}, dCE {np.mean([s['bar']['dce'] for s in summary.values()]):+.4f}")
        if a.export_grid:
            json.dump(summary, open(os.path.join(HERE, "frontier.json"), "w"), indent=1)


def export_grid(task, r, s):
    """Grid rows on the grid's 16 test rows at 2k, with a summary ``flops`` field."""
    full = r["full"]["test16"]
    T = [None] * len(full)
    schemes = {"full": {"ce": full, "flops": [1.0] * len(full), "comp": [1.0] * len(full)}}

    def point(label, keep):
        c = r["configs"][label]
        if keep is None or keep >= 1.0:
            return {"ce": c["splits"]["test16"]["tok"], "flops": c["splits"]["test16"]["tok_flops"], "comp": c["splits"]["test16"]["comp"]}
        pt = c["points"][f"{keep:g}"]["test16"]
        return {"ce": pt["ce"], "flops": pt["flops"], "comp": c["splits"]["test16"]["comp"]}

    if s.get("v6a_ls_keep") is not None and "b1.0" in r["configs"]:
        schemes["router_v6a+ls"] = point("b1.0", s["v6a_ls_keep"])
    for tau, rec in s["tiers"].items():
        schemes[f"router_frontier_t{tau}"] = schemes["full"] if rec["label"] is None else point(rec["label"], rec["keep"])
    summ, per = {}, {}
    for k, v in schemes.items():
        ce = np.asarray(v["ce"], float)
        summ[k] = {"ce": float(ce.mean()), "ce_se": se(ce), "compaction": float(np.mean(v["comp"])), "compaction_se": se(v["comp"]),
                   "flops": float(np.mean(v["flops"])), "layer_frac": 0.0, "sel_cost": 0.0}
        per[k] = {"ce": [float(x) for x in ce], "compaction": [float(x) for x in v["comp"]], "flops": [float(x) for x in v["flops"]]}
    d = {"task": f"ctc_{task}", "rung": "2k", "eval_size": len(full), "argv": ["summarize_frontier.py", "--rows", str(len(full))],
         "git_commit": r.get("git_commit"), "summary": summ, "per_row": per, "row_lens": T if None not in T else None,
         "meta": {"source": "debug/learned_router/layerskip (frontier.py + summarize_frontier.py)", "tiers": list(s["tiers"])}}
    od = os.path.join(GRID, "results_layerskip")
    os.makedirs(od, exist_ok=True)
    json.dump(d, open(os.path.join(od, f"{task}_2k.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
