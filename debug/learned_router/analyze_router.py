"""Compare the learned router against the grid's fixed heuristics on the test rows.

    python debug/learned_router/analyze_router.py [--tasks nq,outlier,...] [--out debug/learned_router/router_vs_grid.json]

Reads ``debug/devloss_grid/results_router/<task>_<rung>.json`` (router schemes AND the baselines
``gold_rand20p8_noslot`` / ``gold_fl20p8_noslot`` / ``gold_first20`` re-scored in the same file, so
every delta is paired per row). For each (task, rung):

* every scheme's mean / median dCE (vs ``full``), paired SE, mean compaction T2/T;
* the router's Pareto envelope over its deterministic points (all lambdas, variant ``full`` only),
  linearly interpolated at each baseline's compaction -> ``router_at_c`` and the verdict
  ``beats`` / ``ties`` / ``loses`` (tie = within max(0.02, 1 paired SE of the baseline)).
  Outside the router's compaction range the verdict is decided only by dominance (a router point
  with c <= c_b and dCE <= dCE_b), else ``no_match`` (the router never gets that cheap: not on the
  frontier there). Compactions within C_MATCH = 0.02 count as matched. A tie is upgraded to ``beats (cheaper)`` when
  some router point reaches the baseline's dCE (+tol) at <= 0.9x the baseline's compaction
  (``router_c_at_parity``). Verdicts use the DETERMINISTIC router points only (``router_l<lam>``);
  the sampled ``_samp`` points are reported but have heavy per-row tails.

Rows are few (eval_size 16/16/8 per rung): quote spread, not decimals.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "devloss_grid", "results_router")
BASELINES = ["gold_rand20p8_noslot", "gold_fl20p8_noslot", "gold_first20", "gold_only_noslot"]
RUNGS = ["2k", "8k", "32k"]
TIE = 0.02
C_MATCH = 0.02  # compaction difference that still counts as "matched"


def scheme_stats(d, s):
    full = np.asarray(d["per_row"]["full"]["ce"], float)
    ce = np.asarray(d["per_row"][s]["ce"], float)
    diff = ce - full
    comp = np.asarray(d["per_row"][s]["compaction"], float)
    rk = [v for v in d["per_row"][s].get("route_keep", []) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return {
        "dce": float(diff.mean()), "dce_median": float(np.median(diff)),
        "dce_se": float(diff.std(ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else float("nan"),
        "dce_min": float(diff.min()), "dce_max": float(diff.max()),
        "comp": float(comp.mean()), "route_keep": float(np.mean(rk)) if rk else None, "n": int(diff.size),
    }


def envelope(points):
    """Lower (dCE) Pareto envelope over (comp, dce) points, sorted by comp."""
    pts = sorted(points)
    env = []
    for c, v, name in pts:
        if env and v >= env[-1][1]:
            continue  # dominated by a cheaper point
        env.append((c, v, name))
    return env


def at_c(env, c):
    """Envelope value at compaction c (the best dCE reachable with compaction <= c, linearly
    interpolating between adjacent envelope points); None below the cheapest point."""
    if not env or c < env[0][0] - C_MATCH:
        return None, None
    if c < env[0][0]:
        return env[0][1], env[0][2]  # within C_MATCH of the router's cheapest point: matched
    for (ci, vi, ni), (cj, vj, nj) in zip(env, env[1:]):
        if ci <= c < cj:
            t = (c - ci) / (cj - ci)
            return vi + t * (vj - vi), f"{ni}~{nj}"
    return env[-1][1], env[-1][2]  # c at/after the most expensive envelope point


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=os.path.join(HERE, "router_vs_grid.json"))
    a = ap.parse_args()
    only = [t for t in a.tasks.split(",") if t]
    out = {}
    for f in sorted(glob.glob(os.path.join(a.res, "*.json"))):
        d = json.load(open(f))
        task, rung = os.path.basename(f)[:-5].rsplit("_", 1)
        if only and task not in only:
            continue
        cell = {"eval_size": d["eval_size"], "schemes": {}}
        for s in d["per_row"]:
            if s == "full":
                continue
            cell["schemes"][s] = scheme_stats(d, s)
        rpts = [(v["comp"], v["dce"], s) for s, v in cell["schemes"].items()
                if s.startswith("router_l") and "nomark" not in s and not s.endswith("_samp")]
        env = envelope(rpts)
        cell["router_envelope"] = env
        cell["verdicts"] = {}
        for b in BASELINES:
            if b not in cell["schemes"]:
                continue
            bs = cell["schemes"][b]
            val, via = at_c(env, bs["comp"])
            tol = max(TIE, bs["dce_se"] if not math.isnan(bs["dce_se"]) else 0.0)
            if val is None:
                dom = [p for p in rpts if p[0] <= bs["comp"] and p[1] <= bs["dce"]]
                verdict = "beats" if dom else "no_match"
            else:
                verdict = "beats" if val < bs["dce"] - tol else ("loses" if val > bs["dce"] + tol else "ties")
            par = [p[0] for p in rpts if p[1] <= bs["dce"] + tol]
            c_par = min(par) if par else None
            if verdict in ("ties", "loses", "no_match") and c_par is not None and c_par <= 0.9 * bs["comp"]:
                verdict = "beats (cheaper)"
            cell["verdicts"][b] = {"verdict": verdict, "router_at_c": val, "via": via, "baseline_dce": bs["dce"],
                                   "baseline_comp": bs["comp"], "tol": tol, "router_c_at_parity": c_par}
        out.setdefault(task, {})[rung] = cell
    json.dump(out, open(a.out, "w"), indent=1)
    # ---- console table ----
    for task in sorted(out):
        for rung in RUNGS:
            if rung not in out[task]:
                continue
            c = out[task][rung]
            print(f"\n== {task} @ {rung}  eval_size={c['eval_size']}{'  (small eval set)' if c['eval_size'] < 500 else ''}")
            for s, v in sorted(c["schemes"].items(), key=lambda kv: kv[1]["comp"]):
                print(f"   {s:26} x{v['comp']:.2f}  dCE mean {v['dce']:+.3f} ±{v['dce_se']:.3f}  median {v['dce_median']:+.3f}"
                      f"  [{v['dce_min']:+.2f},{v['dce_max']:+.2f}]" + (f"  keep {v['route_keep']:.2f}" if v["route_keep"] is not None else ""))
            for b, v in c["verdicts"].items():
                rv = "--" if v["router_at_c"] is None else f"{v['router_at_c']:+.3f}"
                cp = "--" if v["router_c_at_parity"] is None else f"x{v['router_c_at_parity']:.2f}"
                print(f"   vs {b:22} @x{v['baseline_comp']:.2f}: baseline {v['baseline_dce']:+.3f}, router envelope {rv}, router parity at {cp} -> {v['verdict']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
