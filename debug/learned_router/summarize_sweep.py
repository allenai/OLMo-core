"""Headline table: per task, the VAL-selected router vs the grid heuristics on the test rows.

    python debug/learned_router/summarize_sweep.py [--out debug/learned_router/headline.json]

lambda selection uses ONLY the held-out router_val rows (never the test rows): among the task's
seed-0 ``full``-variant configs, the one with the smallest val deterministic T2/T whose val
deterministic mean dCE <= PARITY (0.02); if none qualifies, the one with the lowest val dCE. The
selected ``router_l<lam>`` is then read off the test-row results (``results_router``) next to
``gold_rand20p8_noslot`` / ``gold_fl20p8_noslot`` / ``gold_first20`` / ``gold_only_noslot`` (paired,
same file). Prints a markdown table; eval sets are 16/16/8 rows -- quote spread, not decimals.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "devloss_grid", "results_router")
PARITY = 0.02
BASE = ["gold_rand20p8_noslot", "gold_fl20p8_noslot", "gold_first20", "gold_only_noslot"]
RUNGS = ["2k", "8k", "32k"]


def select_lambda(task: str):
    cands = []
    for f in glob.glob(os.path.join(HERE, "runs", task, "l*.json")):
        name = os.path.basename(f)[:-5]
        if not re.fullmatch(r"l[0-9.]+", name):
            continue
        r = json.load(open(f))
        v = r["epochs"][r["best_epoch"]]["val"]["det"]
        cands.append((name, v["dce"], v["comp"], v["keep"]))
    if not cands:
        return None, cands
    ok = [c for c in cands if c[1] <= PARITY]
    pick = min(ok, key=lambda c: c[2]) if ok else min(cands, key=lambda c: c[1])
    return pick[0], sorted(cands, key=lambda c: float(c[0][1:]))


def stats(d, s):
    if s not in d["per_row"]:
        return None
    full = np.asarray(d["per_row"]["full"]["ce"], float)
    diff = np.asarray(d["per_row"][s]["ce"], float) - full
    return {"dce": float(diff.mean()), "med": float(np.median(diff)),
            "se": float(diff.std(ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else float("nan"),
            "max": float(diff.max()), "comp": float(np.mean(d["per_row"][s]["compaction"])), "n": int(diff.size)}


def verdict(r, b, tol_min=PARITY):
    """Val-selected router vs one baseline on the test rows: ``beats`` (lower dCE at <= matched
    compaction, or parity at <= 0.9x the baseline's compaction), ``ties`` (parity at matched
    compaction), ``costlier`` (parity but > 0.02 more compaction), ``tradeoff`` (lower dCE but more
    compaction), ``loses`` (dCE worse by > tol)."""
    if r is None or b is None:
        return "-"
    tol = max(tol_min, b["se"] if not math.isnan(b["se"]) else 0.0)
    if r["dce"] > b["dce"] + tol:
        return "loses"
    if r["comp"] <= 0.9 * b["comp"] or (r["comp"] <= b["comp"] + 0.02 and r["dce"] < b["dce"] - tol):
        return "beats"
    if r["comp"] <= b["comp"] + 0.02:
        return "ties"
    return "tradeoff" if r["dce"] < b["dce"] - tol else "costlier"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "headline.json"))
    a = ap.parse_args()
    tasks = sorted({os.path.basename(f)[:-5].rsplit("_", 1)[0] for f in glob.glob(os.path.join(RES, "*.json"))})
    out = {}
    print("| task | rung | n | router (val-selected λ) | gold_rand20p8_noslot | gold_fl20p8_noslot | gold_only_noslot | vs rand20p8 / fl20p8 |")
    print("|---|---|---|---|---|---|---|---|")
    fmt = lambda s: "—" if s is None else f"×{s['comp']:.2f} {s['dce']:+.3f} / {s['med']:+.3f}"  # noqa: E731
    for t in tasks:
        lam, cands = select_lambda(t)
        out[t] = {"selected": lam, "val_candidates": cands, "rungs": {}}
        for r in RUNGS:
            f = os.path.join(RES, f"{t}_{r}.json")
            if not os.path.exists(f) or lam is None:
                continue
            d = json.load(open(f))
            row = {"router": stats(d, f"router_{lam}"), **{b: stats(d, b) for b in BASE}}
            row["verdict"] = {b: verdict(row["router"], row[b]) for b in BASE}
            out[t]["rungs"][r] = row
            print(f"| {t} | {r} | {d['eval_size']} | {lam}: {fmt(row['router'])} | {fmt(row['gold_rand20p8_noslot'])} | "
                  f"{fmt(row['gold_fl20p8_noslot'])} | {fmt(row['gold_only_noslot'])} | "
                  f"{row['verdict']['gold_rand20p8_noslot']} / {row['verdict']['gold_fl20p8_noslot']} |")
    json.dump(out, open(a.out, "w"), indent=1)
    from collections import Counter

    for b in BASE:
        c = Counter(v["verdict"][b] for t in out.values() for v in t["rungs"].values())
        print(f"verdicts vs {b}: {dict(c)}")
    print(f"\n(cells: compaction T2/T, mean dCE / median dCE; eval sets 16/16/8 rows) wrote {a.out}")


if __name__ == "__main__":
    main()
