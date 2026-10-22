"""Collect online-router training runs: per-task eval scores (Beaker eval logs, `[ladder:task@rung] metric=v (n=N`),
training PF (the run's flops.json line), and the matched-FLOP comparison with the dense curve.

    python research/token_routing/online_router/collect_otr.py [--rows out.jsonl] [--md out.md]

Dense reference: every dense mt4 run of tokdrop_lr in ops results (Lambda, same recipe and base; campaigns lra / lrb / lri /
lre: mt4s488 at 31 / 61 / 122 steps, fresh-data mt4s976 = 244, mt4s1464 = 366), one curve per LR; each cell is compared
with the BEST LR's curve at the run's PF (an LR enters once it has two points). PF for a dense run = steps x the per-step dense PF measured on our own runs
(dense_equivalent_pflops / steps; dense-equivalent charges every token at the full cost), so the dense curve and the
router runs are on the same meter. Rows for `ops result add --json` go to --rows.
"""
import argparse
import json
import math
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BEAKER = os.path.expanduser("~/.local/bin/beaker")
SCORE_RE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] ([a-z0-9_]+)=(-?[0-9.]+) \(n=(\d+)")
FLOP_RE = re.compile(r"'actual_pflops': ([0-9.]+), 'dense_equivalent_pflops': ([0-9.]+).*?'overhead_pflops': (-?[0-9.]+), 'steps': (\d+)")
TASK_METRIC = {"contradiction": "f1", "nq": "f1", "outlier": "f1", "oolong": "score"}


def logs(ex):
    return subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=180).stdout


#: dense run names on Lambda: f8k-mt4s[<rows>]-dense-<campaign>-lr<lr>[-s<steps>]-lam (rows '' = 488, 1 epoch)
DENSE_RE = re.compile(r"^f8k-mt4s(\d*)-dense-(lra|lrb|lre|lri)-lr([0-9.e-]+?)(?:-s(\d+))?-lam$")


def dense_curves():
    """{lr: {run: steps}} for every dense mt4 run in ops results (campaigns lra / lrb / lre / lri, one epoch), and the
    scores {(run, task, rung): value}. Per-cell reference = the best LR at the same PF (Prasann / parent session
    2026-10-22: LR 2e-6 wins outlier@2k and oolong@2k, 5e-6 wins contradiction@8k)."""
    out = subprocess.run(["/accounts/projects/berkeleynlp/prasann/bin/ops", "result", "show", "--experiment",
                          "tokdrop_lr", "--json"], capture_output=True, text=True).stdout
    rows = json.loads(out) if out.strip().startswith("[") else [json.loads(l) for l in out.splitlines() if l.strip()]
    curves, res = {}, {}
    for r in rows:
        m = DENSE_RE.match(r.get("train_run", ""))
        if not m:
            continue
        rows_per_task = int(m.group(1) or 488)
        steps = int(m.group(4)) if m.group(4) else rows_per_task * 4 // 16
        curves.setdefault(float(m.group(3)), {})[r["train_run"]] = steps
        res[(r["train_run"], r["task"], r["rung"])] = float(r["value"])
    return curves, res


def interp(curve, x):
    """curve sorted [(pf, v)], linear in log pf; returns (value, extrapolated)."""
    lx = math.log(x)
    pts = [(math.log(p), v) for p, v in curve]
    if lx <= pts[0][0]:
        (a, fa), (b, fb), ext = pts[0], pts[1], lx < pts[0][0]
    elif lx >= pts[-1][0]:
        (a, fa), (b, fb), ext = pts[-2], pts[-1], lx > pts[-1][0]
    else:
        i = max(j for j in range(len(pts)) if pts[j][0] <= lx)
        (a, fa), (b, fb), ext = pts[i], pts[i + 1], False
    return fa + (fb - fa) * (lx - a) / (b - a), ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    runs = {}
    for name, r in st["runs"].items():
        m = FLOP_RE.search(logs(r["ex"]))
        runs[name] = dict(r, pf=float(m.group(1)) if m else None, dense_eq=float(m.group(2)) if m else None,
                          overhead=float(m.group(3)) if m else None, steps=int(m.group(4)) if m else None, scores={})
    for ename, e in st["evals"].items():
        for m in SCORE_RE.finditer(logs(e["ex"])):
            runs[e["train_run"]]["scores"][(m.group(1), m.group(2))] = (float(m.group(4)), int(m.group(5)), e["ex"])
    per_step = [r["dense_eq"] / r["steps"] for r in runs.values() if r["dense_eq"] and r["steps"]]
    dense_pf_step = sorted(per_step)[len(per_step) // 2] if per_step else None
    curves, dsc = dense_curves()
    rows, lines = [], ["| run | PF | x dense-eq | task@rung | score | dense @ same PF (best LR) | delta | eval_size |",
                       "|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(runs.items()):
        for (task, rung), (v, n, ex) in sorted(r["scores"].items()):
            se = math.sqrt(max(v * (1 - v), 1e-6) / n) if 0 <= v <= 1 else float("nan")
            rows.append(dict(run_id=f"{name}-ev{task}", train_run=name, experiment="learned_router", task=task,
                             rung=rung, metric=TASK_METRIC.get(task, "score"), value=v, eval_size=n, se=round(se, 3),
                             cot_label="no-cot", eval_backend="legacy_native multirung (olmo_core)",
                             ckpt_path=f"/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/{name}",
                             eval_job_id=ex))
            dv, ext, delta = None, False, None
            if r["pf"] and dense_pf_step:
                best = None
                for lr, runs_lr in sorted(curves.items()):
                    curve = sorted((st_ * dense_pf_step, dsc[(d, task, rung)]) for d, st_ in runs_lr.items()
                                   if (d, task, rung) in dsc)
                    if len(curve) < 2:
                        continue
                    val, e_ = interp(curve, r["pf"])
                    if best is None or val > best[0]:
                        best = (val, e_, lr)
                if best is not None:
                    dv, ext = best[0], best[1]
                    delta = v - dv
            lines.append(f"| {name.replace('f8k-mt4-', '').replace('-tdb-lr2e-06', '')} | {r['pf'] or 0:.0f} | "
                         f"{(r['pf'] or 0) / (r['dense_eq'] or 1):.2f} | {task}@{rung} | {v:.3f} | "
                         f"{'' if dv is None else f'{dv:.3f}'}{' (extrap)' if ext else ''}"
                         f"{'' if dv is None else f' @lr{best[2]:g}'} | "
                         f"{'' if delta is None else f'{delta:+.3f}'} | {n} |")
    lines.append(f"\ndense per-step PF (from our runs' dense-equivalent) = {dense_pf_step:.3f}" if dense_pf_step else "")
    print("\n".join(lines))
    if a.rows:
        with open(a.rows, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if a.md:
        open(a.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
