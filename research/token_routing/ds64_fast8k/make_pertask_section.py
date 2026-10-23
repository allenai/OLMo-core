"""Per-task (FLOP-fair) section of the tokdrop_lr artifact: each task trained ALONE on its own fast8k g16M pool
(~1,950 rows), so every method's FLOPs go to that one task, as in September's ds64 per-task runs.

Runs: ``f8k-<task>-<arm>-16M[_eg][st<steps>]-tdp-lr<lr>`` from launch_fast8k's state.json (dense curve + ds64's winning
recipes) and from the online-router session's state.json (otr per task). Scores come from ops results, training PF from
the Beaker training log. Each non-dense run is compared with the task's dense curve (per rung, the better of dense LR
2e-6 and 5e-6, linear in log PF) at its own PF. ds64's own matched-FLOP deltas for the same rule are shown beside.

    python research/token_routing/ds64_fast8k/make_pertask_section.py --page PAGE.html
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_pareto_points import ROUTER_STATE, _beaker_train, _ops_scores  # noqa: E402
from make_pareto_section import interp  # noqa: E402

TASKS = ["contradiction", "nq", "outlier", "oolong"]
NAME = re.compile(r"^f8k-(contradiction|nq|outlier|oolong)-(.+?)-16M(?:_eg)?(?:st(\d+))?-tdp-lr(\d[\d.]*(?:e-\d+)?)$")
#: ds64 (September, per-task, 2k-64k mix) matched-FLOP deltas of the same recipes at 8k / 2k, best non-extrapolated
#: budget by 5-rung mean, recomputed per rung from the ds64 artifact's data (FLOPs / dense on the same data, budget)
DS64 = {("contradiction", "chk16"): ("+0.089", "+0.057", "0.28 (64M)"),
        ("contradiction", "chk64"): ("+0.077", "+0.056", "0.52 (32M)"),
        ("nq", "kv33"): ("−0.006", "+0.005", "0.21 (128M)"),
        ("outlier", "flk64"): ("+0.433", "+0.083", "0.26 (64M)"),
        ("outlier", "ck64ds"): ("+0.410", "+0.083", "0.26 (64M)"),
        ("oolong", "ohdr33"): ("+0.055", "−0.003", "0.41 (128M)")}
CACHE = os.path.join(HERE, "results", "pertask_points.json")


def gather():
    srcs = list(json.load(open(os.path.join(HERE, "state.json")))["runs"].items())
    try:
        rs = subprocess.run(["git", "-C", HERE, "show", ":".join(ROUTER_STATE)], capture_output=True, text=True, timeout=60).stdout
        srcs += list(json.loads(rs)["runs"].items())
    except Exception as e:  # noqa: BLE001
        print("router state unavailable:", e)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out = {}
    for run, v in srcs:
        m = NAME.match(run)
        if not m or not v.get("ex"):
            continue
        task, arm, steps, lr = m.groups()
        sc = _ops_scores(run)
        if not sc:
            continue
        c = cache.get(run, {})
        if not c.get("pf"):
            pf, last, wall = _beaker_train(v["ex"])
            c = dict(pf=pf, steps=last, wall=wall)
        out[run] = dict(c, run=run, task=task, arm=arm, lr=float(lr), max_steps=int(steps or 0),
                        scores={f"{t}@{r}": x for (t, r), (x, n) in sc.items()})
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def build(runs):
    rows = []
    for task in TASKS:
        tr = [r for r in runs.values() if r["task"] == task and r.get("pf")]
        dense = [r for r in tr if r["arm"] == "dense"]
        for r in sorted(tr, key=lambda r: (r["arm"] != "dense", r["arm"], r["pf"])):
            cells = {}
            for rung in ("8k", "2k"):
                key = f"{task}@{rung}"
                v = r["scores"].get(key)
                if v is None:
                    continue
                best = None
                if r["arm"] != "dense":
                    for lr in (2e-6, 5e-6):
                        curve = sorted((d["pf"], d["scores"][key]) for d in dense
                                       if abs(d["lr"] - lr) < 1e-12 and key in d["scores"])
                        if len(curve) >= 2:
                            dv, ext = interp(curve, r["pf"])
                            if best is None or dv > best[0]:
                                best = (dv, ext)
                cells[rung] = (v, None if best is None else v - best[0], bool(best and best[1]))
            rows.append(dict(task=task, arm=r["arm"], lr=r["lr"], steps=r["steps"], pf=r["pf"], cells=cells,
                             ds64=DS64.get((task, r["arm"]))))
    return rows


def html_section(rows, updated):
    cls = lambda d: "" if d is None else (" pos" if d >= -0.005 else " mid" if d >= -0.03 else " neg")
    fmt = lambda c: "…" if c is None else (f"{c[0]:.3f}" + ("" if c[1] is None else f" ({c[1]:+.3f}{'*' if c[2] else ''})"))
    h = '<!--PERTASK-BEGIN-->\n<section id="pertask">\n  <h2>Per task, FLOP-fair (each task trained alone)</h2>\n'
    h += ('  <p>Each task trained on its own fast8k pool (~1,950 rows, 16 rows/step, 8k rows, unpacked), so a method\'s FLOPs '
          'all go to that task, as in September\'s ds64 per-task runs. Dense curve per task: LR 5e-6 at 31/61/122 steps and LR 2e-6 '
          'at 61/122; each other run is compared with the per-rung better of the two dense LRs at its own training PF (Δ in '
          'brackets, * = outside the dense range). ds64\'s rules use their verbatim recipes (LR 5e-6); the router is otr10M8 '
          '(LR 2e-6). Right-hand columns: ds64\'s own matched-FLOP Δ for the same rule in its September setting (2k–64k mix, '
          'longer budgets). 500 eval examples per cell, no-cot. Updated ' + updated + '.</p>\n')
    h += ('  <div class="tbl"><table>\n    <tr><th>task</th><th>method</th><th class="n">LR</th><th class="n">steps</th>'
          '<th class="n">PF</th><th class="n">8k (Δ)</th><th class="n">2k (Δ)</th><th class="n">ds64 Δ 8k / 2k</th>'
          '<th class="n">ds64 FLOPs ÷ dense</th></tr>\n')
    for r in rows:
        c8, c2 = r["cells"].get("8k"), r["cells"].get("2k")
        ds = r["ds64"] or ("", "", "")
        h += (f'    <tr><td>{r["task"]}</td><td class="mono">{r["arm"]}</td><td class="n">{r["lr"]:g}</td>'
              f'<td class="n">{r["steps"] or ""}</td><td class="n">{r["pf"]:.0f}</td>'
              f'<td class="n{cls(c8 and c8[1])}">{fmt(c8)}</td><td class="n{cls(c2 and c2[1])}">{fmt(c2)}</td>'
              f'<td class="n">{(ds[0] + " / " + ds[1]) if ds[0] else ""}</td><td class="n">{ds[2]}</td></tr>\n')
    h += '  </table></div>\n</section>\n<!--PERTASK-END-->'
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--updated", default="")
    a = ap.parse_args()
    rows = build(gather())
    h = html_section(rows, a.updated)
    s = open(a.page).read()
    if "<!--PERTASK-BEGIN-->" in s:
        s = re.sub(r"<!--PERTASK-BEGIN-->.*?<!--PERTASK-END-->", lambda m: h, s, flags=re.S)
    else:
        s = s.replace("<!--DS64X-BEGIN-->", h + "\n\n<!--DS64X-BEGIN-->", 1)
    open(a.page, "w").write(s)
    for r in rows:
        if r["arm"] != "dense":
            print(r["task"], r["arm"], round(r["pf"]), {k: (round(v[0], 3), v[1] and round(v[1], 3)) for k, v in r["cells"].items()})


if __name__ == "__main__":
    main()
