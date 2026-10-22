"""Matched-FLOP comparison of keep rules vs a dense curve on a fast8k task mix (Lambda runs; mt4s / mt11s).

For every run: eval scores per (task, rung) from the ladder logs (`[ladder:task@rung] metric=v (n=N`), and
training PF from the run log's last `flop_meter/actual_pflops` (per-example accounting). The dense curve is the
dense runs at several step counts; a dense run without per-example accounting (the old mt4s 122-step run) gets
PF = steps x the per-step PF of the longest per-example dense run. Each rule is compared with dense interpolated (linearly in
log PF) at the rule's PF; outside the dense range it is flagged as extrapolated.

    python research/token_routing/ds64_fast8k/collect_rule_screen.py --runs RUN [RUN ...] --dense RUN:STEPS [...]

Scores are each task's primary metric (f1, pairwise_f1, kendall_tau in [-1, 1], mrr...).
"""

import argparse
import json
import math
import re
import subprocess
from collections import defaultdict

LOGS = "/accounts/projects/sewonm/prasann/ctc_suite/logs"


def ssh(cmd: str) -> str:
    return subprocess.run(["ssh", "lambda", cmd], capture_output=True, text=True, timeout=180).stdout


def scores(run: str):
    out = ssh(f"cd {LOGS}; grep -hoE '\\[ladder:[a-z_]+@[0-9]+k\\] [a-z0-9_]+=-?[0-9.]+ \\(n=[0-9]+' "
              f"f8k_eval_{run}-ev*.log 2>/dev/null | sort -u")
    res = {}
    for m in re.finditer(r"\[ladder:([a-z_]+)@(\d+k)\] [a-z0-9_]+=(-?[0-9.]+) \(n=(\d+)", out):
        res[(m.group(1), m.group(2))] = (float(m.group(3)), int(m.group(4)))
    return res


def train_pf(run: str):
    out = ssh(f"cd {LOGS}; f=$(ls -t ctc_suite_{run}_*.log 2>/dev/null | head -1); "
              f"grep -E 'flop_meter/actual_pflops|flop_meter/actual_over_dense|\\[step=' $f | tail -40")
    pf = [float(x) for x in re.findall(r"flop_meter/actual_pflops[ =]+([0-9.]+)", out)]
    steps = [int(x) for x in re.findall(r"\[step=(\d+)/", out)]
    per_example = "per_example" in ssh(f"cd {LOGS}; f=$(ls -t ctc_suite_{run}_*.log | head -1); "
                                       f"grep -m1 -o 'step_actual_over_dense' $f; grep -m1 -o per_example $f")
    return (pf[-1] if pf else None), (max(steps) if steps else None), per_example


def interp(curve, x):
    """curve: sorted [(pf, f1)]; linear in log pf. Returns (value, extrapolated)."""
    lx = math.log(x)
    pts = [(math.log(p), f) for p, f in curve]
    if lx <= pts[0][0]:
        (a, fa), (b, fb) = pts[0], pts[1]
        ext = lx < pts[0][0]
    elif lx >= pts[-1][0]:
        (a, fa), (b, fb) = pts[-2], pts[-1]
        ext = lx > pts[-1][0]
    else:
        i = max(j for j in range(len(pts)) if pts[j][0] <= lx)
        (a, fa), (b, fb) = pts[i], pts[i + 1]
        ext = False
    return fa + (fb - fa) * (lx - a) / (b - a), ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dense", nargs="+", required=True, help="RUN:STEPS")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dense = []
    for spec in a.dense:
        run, st = spec.rsplit(":", 1)
        pf, steps, pe = train_pf(run)
        dense.append(dict(run=run, steps=int(st), pf=pf, per_example=pe, s=scores(run)))
    ref = max((d for d in dense if d["pf"] and d["per_example"]), key=lambda d: d["steps"], default=None)
    for d in dense:
        if not d["per_example"] and ref is not None:
            d["pf"] = ref["pf"] / ref["steps"] * d["steps"]  # old run: no per-example meter -> scale per-step PF
    rows = []
    for run in a.runs:
        pf, steps, _ = train_pf(run)
        sc = scores(run)
        for (task, rung), (v, n) in sorted(sc.items()):
            curve = sorted((d["pf"], d["s"][(task, rung)][0]) for d in dense if d["pf"] and (task, rung) in d["s"])
            if len(curve) < 2 or pf is None:
                continue
            dv, ext = interp(curve, pf)
            rows.append(dict(run=run, task=task, rung=rung, f1=v, eval_size=n, pf=round(pf, 1),
                             dense_at_pf=round(dv, 4), delta=round(v - dv, 4), extrapolated=ext))
    print("| run | task@rung | f1 | PF | dense @ same PF | Δ | eval_size |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {re.sub(r'^f8k-mt[0-9]+s[0-9]*-', '', r['run']).replace('-lam', '')} | {r['task']}@{r['rung']} | {r['f1']:.3f} | "
              f"{r['pf']} | {r['dense_at_pf']:.3f}{' (extrap)' if r['extrapolated'] else ''} | {r['delta']:+.3f} | {r['eval_size']} |")
    print("\ndense curve:", [(re.sub(r"^f8k-mt[0-9]+s[0-9]*-", "", d["run"]), d["steps"], round(d["pf"] or -1, 1)) for d in dense])
    if a.json:
        json.dump(dict(rows=rows, dense=[{k: v for k, v in d.items() if k != "s"} for d in dense]), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
