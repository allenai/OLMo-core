"""Every fast8k token-drop training run (Lambda + Beaker) as one table of Pareto points, for the cross-task
plots on the tokdrop_lr artifact.

One row per (run, task, rung): score (the task's primary metric from the ladder log), eval_size, training PF
(``flop_meter/actual_pflops``; a dense run without the per-example meter gets steps x the per-step PF of the
longest per-example dense run on the same cluster), steps, wall-clock train seconds, cluster, and the parsed
arm / data / LR. Lambda runs come from the run registry (``ops runs``) and the Lambda logs; Beaker runs from
``launch_fast8k.py``'s state.json and ``beaker experiment logs``.

    python research/token_routing/ds64_fast8k/collect_pareto_points.py --out research/token_routing/ds64_fast8k/results/pareto_points.json
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_rule_screen import LOGS, scores, ssh, train_pf  # noqa: E402

OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"
BEAKER = os.path.expanduser("~/.local/bin/beaker")
SCORE_RE = re.compile(r"\[ladder:([a-z_]+)@(\d+k)\] [a-z0-9_]+=(-?[0-9.]+) \(n=(\d+)")


def parse_name(run):
    """f8k-mt4s[<rows>]-<arm>[-s<seed>][-lr<c>]-lr<lr>[-ep<E>][-s<steps>][-r<k>]-lam (Lambda) or
    f8k-mt4-<arm>-s<rows>[st<steps>]-tdb-lr<lr> (Beaker). Returns None for names it does not know."""
    m = re.match(r"f8k-(mt\d+)s(\d*)-(.+)$", run)
    if m:
        mix, rows, rest = m.groups()
        lr = re.search(r"-lr(\d[\d.]*(?:e-\d+)?)(?=-|$)", rest)
        if not lr:
            return None
        before, after = rest[: lr.start()].split("-"), rest[lr.end():].split("-")
        camp = before.pop() if re.fullmatch(r"lr[a-z]", before[-1]) else ""
        seed = int(before.pop()[1:]) if len(before) > 1 and re.fullmatch(r"s\d", before[-1]) else 0
        tag = lambda pre: next((int(t[len(pre):]) for t in after if re.fullmatch(pre + r"\d+", t)), 0)
        return dict(mix=mix, rows=int(rows or 488), arm="-".join(before), lr=float(lr.group(1)), campaign=camp,
                    seed=seed, epochs=tag("ep") or 1, max_steps=tag("s"))
    m = re.match(r"f8k-(mt\d+)-(.+)-s(\d+)(?:st(\d+))?-tdb-lr(\d[\d.]*(?:e-\d+)?)$", run)
    if m:
        mix, arm, rows, steps, lr = m.groups()
        return dict(mix=mix, rows=int(rows), arm=arm, lr=float(lr), campaign="tdb", seed=0, epochs=1,
                    max_steps=int(steps or 0))
    return None


def lambda_runs():
    out = subprocess.run([OPS, "runs", "--experiment", "tokdrop_lr", "--cluster", "lambda", "--json"],
                         capture_output=True, text=True, timeout=60).stdout
    rows = json.loads(out) if out.strip().startswith("[") else []
    return sorted({r["run_id"] for r in rows if r.get("kind") == "train" and r["run_id"].startswith("f8k-mt")})


def lambda_wall(run):
    out = ssh(f"cd {LOGS}; f=$(ls -t ctc_suite_{run}_*.log 2>/dev/null | head -1); j=${{f##*_}}; j=${{j%.log}}; "
              f"sacct -X -n -P -o ElapsedRaw,State -j $j 2>/dev/null | head -1")
    try:
        sec, state = out.strip().split("|")[:2]
        return int(sec), state
    except ValueError:
        return None, None


def beaker_points():
    st_path = os.path.join(HERE, "state.json")
    if not os.path.exists(st_path):
        return []
    st = json.load(open(st_path))
    pts = []
    for ename, e in st.get("evals", {}).items():
        run = e.get("train_run")
        if not run or "-tdb" not in run or not e.get("ex"):
            continue
        log = subprocess.run([BEAKER, "experiment", "logs", e["ex"]], capture_output=True, text=True,
                             timeout=120).stdout
        for m in SCORE_RE.finditer(log):
            pts.append(dict(run=run, task=m.group(1), rung=m.group(2), score=float(m.group(3)),
                            eval_size=int(m.group(4))))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-beaker", action="store_true")
    a = ap.parse_args()
    rows, runs = [], {}
    for run in lambda_runs():
        meta = parse_name(run)
        if meta is None:
            continue
        sc = scores(run)
        if not sc:
            continue
        pf, steps, per_example = train_pf(run)
        wall, state = lambda_wall(run)
        # 2026-10-22 option-field bug: lrb runs named tier* kept the keep set WHOLE (--st-kept-doc-k dropped)
        # and lrb *m<N>p8 runs lost their --st-keep-token-min floor. Never plot them under their names.
        if meta["campaign"] == "lrb" and (meta["arm"].startswith("tier") or re.search(r"m\d+p8$", meta["arm"])):
            meta["label_warning"] = "lrb flag-drop bug: ran without --st-kept-doc-k / --st-keep-token-min"
            meta["arm"] = meta["arm"] + "~lrbbug"
        runs[run] = dict(meta, run=run, cluster="lambda-a100", pf=pf, steps=steps, per_example=per_example,
                         train_seconds=wall, gpus=2, dense=meta["arm"].startswith("dense"))
        for (task, rung), (v, n) in sc.items():
            rows.append(dict(run=run, task=task, rung=rung, score=v, eval_size=n))
    # old dense runs without the per-example meter: scale per-step PF from the longest per-example dense run
    ref = max((r for r in runs.values() if r["dense"] and r["per_example"] and r["pf"]),
              key=lambda r: r["steps"] or 0, default=None)
    for r in runs.values():
        if r["dense"] and not r["per_example"] and ref and r["steps"]:
            r["pf"] = ref["pf"] / ref["steps"] * r["steps"]
            r["pf_scaled"] = True
    if not a.no_beaker:
        rows += beaker_points()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(runs=runs, points=rows), open(a.out, "w"), indent=1)
    print(f"{len(runs)} runs, {len(rows)} points -> {a.out}")


if __name__ == "__main__":
    main()
