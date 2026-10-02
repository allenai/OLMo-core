"""Headline for the DIFFERENTIABLE router: per task x tau, does the task-relative tolerance hold out
of sample, and how does the router compare with the REINFORCE router and the grid heuristics?

    python debug/learned_router/summarize_diff.py [--out debug/learned_router/headline_diff.json]

Reads ``debug/devloss_grid/results_router_diff/<task>_<rung>.json`` (``router_diff_tau*`` plus, in the
same file, the REINFORCE router at its val-selected lambda and the heuristic baselines -- paired
per row). Per (task, rung, tau):

* test mean / median dCE, compaction T2/T;
* **relative dCE** = mean dCE / mean CE_full over the test rows of that rung, and whether it is
  within tolerance: mean dCE <= max(tau * mean CE_full, floor) -- the training constraint, now on
  held-out rows (``holds``); ``holds~`` = misses by less than one paired SE;
* verdict vs ``gold_rand20p8_noslot`` and vs the REINFORCE router (same rules as
  summarize_sweep.py: beats / ties / costlier / tradeoff / loses).

Also pulls the train-time selection record (val T2/T, val dCE, eps) from ``runs/<task>/diff_tau*.json``.
Eval sets are 16/16/8 rows -- quote spread, not decimals.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "devloss_grid", "results_router_diff")
RUNGS = ["2k", "8k", "32k"]
FLOOR = 0.005

import summarize_sweep as SW  # noqa: E402  (stats / verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "headline_diff.json"))
    ap.add_argument("--res", default=RES)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.res, "*.json")))
    tasks = sorted({os.path.basename(f)[:-5].rsplit("_", 1)[0] for f in files})
    out = {}
    print("| task | rung | n | CE_full | τ | diff router ×c, ΔCE mean/med | rel ΔCE (≤τ?) | REINFORCE ×c, ΔCE | rand20p8 ×c, ΔCE | vs rand20p8 / vs REINFORCE |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    counts = {"holds": Counter(), "vs_rand": Counter(), "vs_rl": Counter()}
    for t in tasks:
        out[t] = {}
        for r in RUNGS:
            f = os.path.join(a.res, f"{t}_{r}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            cef = float(np.mean(d["per_row"]["full"]["ce"]))
            rl = next((s for s in d["per_row"] if s.startswith("router_l")), None)
            rl_st = SW.stats(d, rl) if rl else None
            rand = SW.stats(d, "gold_rand20p8_noslot")
            for s in sorted(k for k in d["per_row"] if k.startswith("router_diff_tau")):
                tau = float(s[len("router_diff_tau"):])
                st = SW.stats(d, s)
                eps = max(tau * cef, FLOOR)
                rel = st["dce"] / cef if cef > 0 else float("nan")
                holds = "holds" if st["dce"] <= eps else ("holds~" if st["dce"] <= eps + (st["se"] if not math.isnan(st["se"]) else 0) else "misses")
                v_rand = SW.verdict(st, rand)
                v_rl = SW.verdict(st, rl_st) if rl_st else "-"
                counts["holds"][holds] += 1
                counts["vs_rand"][v_rand] += 1
                counts["vs_rl"][v_rl] += 1
                run = os.path.join(HERE, "runs", t, f"diff_tau{s[len('router_diff_tau'):]}.json")
                sel = None
                if os.path.exists(run):
                    rr = json.load(open(run))
                    be = rr["epochs"][rr["best_epoch"]]
                    trd = (be.get("train") or {})
                    sel = {"best_epoch": rr["best_epoch"], "epochs_run": rr["epochs"][-1]["epoch"], "val_comp": rr["best_val"]["comp"],
                           "val_dce": rr["best_val"]["dce"], "train_det_dce": trd.get("det_dce"), "train_det_comp": trd.get("det_comp"),
                           "eps_val": rr["eps_val"], "eps_train": rr["eps_train"], "w_gold": rr["weights_best"]["w_gold"],
                           "dedup_of": rr.get("dedup_of")}
                out[t].setdefault(r, {})[s] = {"tau": tau, "ce_full": cef, "eps": eps, "stats": st, "rel_dce": rel, "holds": holds,
                                               "vs_rand20p8": v_rand, "vs_reinforce": v_rl, "reinforce": rl, "reinforce_stats": rl_st,
                                               "rand20p8": rand, "selection": sel, "eval_size": d["eval_size"]}
                fm = lambda z: "—" if z is None else f"×{z['comp']:.2f} {z['dce']:+.3f}/{z['med']:+.3f}"  # noqa: E731
                print(f"| {t} | {r} | {d['eval_size']} | {cef:.3f} | {tau:g} | {fm(st)} | {rel:+.2f} ({holds}) | "
                      f"{(rl or '').replace('router_', '')} {fm(rl_st)} | {fm(rand)} | {v_rand} / {v_rl} |")
    json.dump(out, open(a.out, "w"), indent=1)
    for k, c in counts.items():
        print(f"{k}: {dict(c)}")
    print("\nselection (val rows) and train-vs-val at the selected epoch:")
    for t in out:
        seen = set()
        for r in out[t]:
            for s_, v in out[t][r].items():
                sl = v["selection"]
                if not sl or s_ in seen:
                    continue
                seen.add(s_)
                tdc = "n/a" if sl["train_det_dce"] is None else f"{sl['train_det_dce']:+.4f}@x{sl['train_det_comp']:.2f}"
                print(f"  {t:14} tau {v['tau']:<5g} eps_tr {sl['eps_train']:.4f} best ep {sl['best_epoch']:2d}/{sl['epochs_run']} "
                      f"train det dCE {tdc}  val dCE {sl['val_dce']:+.4f}@x{sl['val_comp']:.2f}  w_gold {sl['w_gold']:+.2f}"
                      + (f"  (= {sl['dedup_of']})" if sl["dedup_of"] else ""))
    print(f"(eval sets 16/16/8 rows; tolerance = max(tau*CE_full, {FLOOR}) nats) wrote {a.out}")


if __name__ == "__main__":
    main()
