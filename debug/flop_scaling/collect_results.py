"""
Collect the FLOP-scaling study into one table + the two plots per task
(records/flop-scaling-ffn-kv-plan.md §1): accuracy vs training tokens, accuracy vs training FLOPs.

Inputs are the harvested artifacts (harvest_to_s3.sh -> `aws s3 sync` here):
  harvest/runs/<run>/flops.json       FlopMeterCallback summary (actual + dense-equivalent PFLOPs)
  harvest/evals/<run>_<task>_multirung.json   flat {"<task>_<rung>": f1, ...} from the ladder eval

Run names: fs-<task>-<arm>-sh<B>[ -s1 | -s2 ]. The ffnmoe arm's stage-2 checkpoint is charged
stage 1 + stage 2 FLOPs (both passes over the data).

    aws s3 sync s3://ai2-llm/checkpoints/prasanns/flop_scaling/harvest results/flop_scaling/harvest
    python debug/flop_scaling/collect_results.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
HARVEST = f"{REPO}/results/flop_scaling/harvest"
OUT = f"{REPO}/results/flop_scaling"
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}
RUNGS = {"contra": ["2k", "8k", "16k", "32k"], "nq": ["3k", "8k", "16k", "32k"],
         "outlier": ["3k", "8k", "16k", "32k"], "oolong": ["8k", "16k", "32k"]}
NAME_RE = re.compile(r"^fs-(?P<task>[a-z]+)-(?P<arm>[a-z0-9]+(?:-s[12])?)-sh(?P<budget>\d+M)$")


def budget_tokens(b: str) -> int:
    return int(b[:-1]) * 1_000_000


def load_flops(run: str):
    p = f"{HARVEST}/runs/{run}/flops.json"
    return json.load(open(p)) if os.path.exists(p) else None


def main() -> None:
    rows = []
    for ev in sorted(glob.glob(f"{HARVEST}/evals/fs-*_multirung.json")):
        base = os.path.basename(ev)[: -len("_multirung.json")]
        run, tkey = base.rsplit("_", 1)
        m = NAME_RE.match(run)
        if not m:
            print("skip (name):", run)
            continue
        task, arm, budget = m["task"], m["arm"], m["budget"]
        res = json.load(open(ev))
        fl = load_flops(run)
        pf = fl["actual_pflops"] if fl else None
        dense_pf = fl["dense_equivalent_pflops"] if fl else None
        if arm == "ffnmoe-s2":  # charge stage 1 too
            fl1 = load_flops(f"fs-{task}-ffnmoe-s1-sh{budget}")
            if fl1 and pf is not None:
                pf += fl1["actual_pflops"]
                dense_pf += fl1["dense_equivalent_pflops"]
        rung_f1 = {r: res.get(f"{tkey}_{r}") for r in RUNGS[tkey]}
        vals = [v for v in rung_f1.values() if v is not None]
        rows.append({
            "task": task, "arm": arm, "budget": budget, "tokens": budget_tokens(budget),
            "actual_pflops": pf, "dense_equiv_pflops": dense_pf,
            "flop_ratio": (pf / dense_pf) if (pf and dense_pf) else None,
            "mean_f1": sum(vals) / len(vals) if vals else None,
            **{f"f1_{r}": v for r, v in rung_f1.items()},
            "run": run,
        })
    os.makedirs(OUT, exist_ok=True)
    if not rows:
        print("no results yet under", HARVEST)
        return
    keys = list(rows[0].keys())
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/results.csv ({len(rows)} rows)")
    for r in sorted(rows, key=lambda x: (x["task"], x["arm"], x["tokens"])):
        print(f"{r['task']:14s} {r['arm']:10s} {r['budget']:>5s}  mean_f1={r['mean_f1'] if r['mean_f1'] is None else round(r['mean_f1'],3)}  "
              f"PF={None if r['actual_pflops'] is None else round(r['actual_pflops'],1)}  ratio={None if r['flop_ratio'] is None else round(r['flop_ratio'],3)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: table only")
        return
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["mean_f1"] is not None:
            by_task[r["task"]][r["arm"]].append(r)
    for task, arms in by_task.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm, rs in sorted(arms.items()):
            rs = sorted(rs, key=lambda x: x["tokens"])
            axes[0].plot([x["tokens"] / 1e6 for x in rs], [x["mean_f1"] for x in rs], "o-", label=arm)
            rs2 = [x for x in rs if x["actual_pflops"]]
            axes[1].plot([x["actual_pflops"] for x in rs2], [x["mean_f1"] for x in rs2], "o-", label=arm)
        axes[0].set_xscale("log"); axes[0].set_xlabel("training tokens (M, short-heavy 2k-32k mix)")
        axes[1].set_xscale("log"); axes[1].set_xlabel("training PFLOPs (method-aware, actual)")
        for ax in axes:
            ax.set_ylabel("mean f1 over eval rungs"); ax.grid(alpha=0.3); ax.legend()
        fig.suptitle(f"{task}: data scaling (left) and FLOP scaling (right), Qwen3-4B")
        fig.tight_layout()
        fig.savefig(f"{OUT}/{task}_scaling.png", dpi=130)
        print("plot:", f"{OUT}/{task}_scaling.png")


if __name__ == "__main__":
    main()
