"""Wall-clock of training runs from their Beaker logs (same cluster: jupiter, 2 x H100, same base / batch / seq).

Per run, from the console logger's ``[step=N/M`` lines (one per optimizer step, timestamped by Beaker):
  s/step      median gap between consecutive steps (steps >= 5; the first steps include compile / autotune warmup)
  train time  last step's timestamp minus first step's (excludes image pull, model load, export)
  dense steps for the router arms, the median gap on full-context steps (step % dense_every == 0) vs the others

    python research/token_routing/online_router/wallclock.py --dense f8k-mt4-dense-s976-tdb-lr5e-06=01M5H825RM46FQA0E7Q92185G4 ...
"""
import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BEAKER = os.path.expanduser("~/.local/bin/beaker")
STEP_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z.*\[step=(\d+)/(\d+)")
DE_RE = re.compile(r"--st-otr-dense-every (\d+)")


def step_times(ex):
    out = subprocess.run([BEAKER, "experiment", "logs", ex], capture_output=True, text=True, timeout=300).stdout
    t = {}
    for ln in out.splitlines():
        m = STEP_RE.match(ln)
        if m:
            t.setdefault(int(m.group(2)), dt.datetime.fromisoformat(m.group(1)))
    return t


def summarize(name, ex, dense_every=0):
    t = step_times(ex)
    if len(t) < 10:
        return dict(run=name, ex=ex, steps=len(t))
    steps = sorted(t)
    gaps = {s: (t[s] - t[s - 1]).total_seconds() for s in steps if s - 1 in t and s >= 5}
    out = dict(run=name, ex=ex, steps=steps[-1], s_per_step=round(statistics.median(gaps.values()), 2),
               train_min=round((t[steps[-1]] - t[steps[0]]).total_seconds() / 60, 1))
    if dense_every:
        d = [g for s, g in gaps.items() if s % dense_every == 0]
        r = [g for s, g in gaps.items() if s % dense_every != 0]
        out.update(s_per_step_dense=round(statistics.median(d), 2) if d else None,
                   s_per_step_routed=round(statistics.median(r), 2) if r else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", nargs="*", default=[], help="NAME=EXPERIMENT_ID of dense runs on the same cluster")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    st = json.load(open(os.path.join(HERE, "state.json")))
    rows = [summarize(*d.split("=", 1)) for d in a.dense]
    from launch_otr import ARMS  # noqa: E402  (same directory)

    for name, r in sorted(st["runs"].items()):
        de = DE_RE.search(ARMS.get(r["arm"], ""))
        rows.append(summarize(name, r["ex"], int(de.group(1)) if de else 0))
    print("| run | steps | s/step | routed s/step | dense-step s/step | train min |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run'].replace('f8k-', '').replace('-tdb-lr2e-06', '').replace('-tdb-lr5e-06', '')} | {r.get('steps')} | "
              f"{r.get('s_per_step', '')} | {r.get('s_per_step_routed', '') or ''} | {r.get('s_per_step_dense', '') or ''} | "
              f"{r.get('train_min', '')} |")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
