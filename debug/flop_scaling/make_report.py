"""
Assemble the FLOP-scaling study report (records/flop-scaling-report-<date>.md) from
results/flop_scaling/results35.csv + fits35.md + the orchestrator state, so the morning summary
is a rendering of the data, not a retyping of it.

    python debug/flop_scaling/make_report.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from collections import defaultdict

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
OUT = f"{REPO}/results/flop_scaling"
STATE = f"{REPO}/debug/flop_scaling/orchestrator35_state.json"
REPORT = f"{REPO}/records/flop-scaling-report-{dt.date.today().isoformat()}.md"
ARM_LABEL = {"dense": "dense (prior campaign + 4-10M anchors)", "ffnmoe-s1": "FFN routing, stage 1 (L12+)", "ffnmoe-s2": "FFN routing, stage 2 (all layers)",
             "ffnmoe-t10": "FFN routing, L12+, two-sided target 0.10", "ffnmoe-a10": "FFN routing, all layers, two-sided target 0.10",
             "kv17": "KV soft-token, keep gold + 1/6", "kv33": "KV soft-token, keep gold + 1/3",
             "kvb17": "KV soft-token, gold-blind keep 1/6", "kvb33": "KV soft-token, gold-blind keep 1/3"}
ARM_ORDER = ["dense", "ffnmoe-s1", "ffnmoe-s2", "ffnmoe-t10", "ffnmoe-a10", "kv17", "kv33", "kvb17", "kvb33"]


def main():
    rows = list(csv.DictReader(open(f"{OUT}/results35.csv"))) if os.path.exists(f"{OUT}/results35.csv") else []
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}
    by = defaultdict(lambda: defaultdict(dict))
    f1cols = sorted({k for r in rows for k in r if k.startswith("f1_")}, key=lambda c: int(c[3:-1]))
    for r in rows:
        by[r["task"]][r["arm"]][r["budget"]] = r
    n_runs = len(st["runs"]); done = sum(r["state"] == "DONE" for r in st["runs"].values()); failed = sum(r["state"] == "FAILED" for r in st["runs"].values())
    ev_done = sum(e["state"] == "DONE" for e in st["evals"].values())
    md = [f"# FLOP-scaling study: FFN routing and KV soft tokens vs dense (Qwen3.5-4B)\n",
          f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from `results/flop_scaling/results35.csv`. "
          f"Plan: `records/flop-scaling-ffn-kv-plan.md`. Ledger: `debug/flop_scaling/LAUNCH_LEDGER.tsv`.\n",
          f"Run status at generation: {done}/{n_runs} training runs done ({failed} failed), {ev_done}/{len(st['evals'])} evals done. "
          "Cells missing below are still queued/running on Beaker.\n",
          (open(f"{REPO}/debug/flop_scaling/summary35.md").read() + "\n") if os.path.exists(f"{REPO}/debug/flop_scaling/summary35.md") else "",
          "## Setup\n",
          "- Model/data/optimizer identical to the prior dense campaigns (`debug/taskscale_lengthmix`, `debug/outlier_lengthmix_scaling`): Qwen3.5-4B (`q35-4b-base-markerfix`), short-heavy 2k–32k length mixes as nested-prefix arms, seq 65536 packed (KV: padded single-example rows, same tokens/step), lr 5e-6, 8 rows/step, 1 epoch. Dense points are those campaigns' numbers, not retrained.",
          "- **FFN routing**: nested-width FFN router over the base FFN (Qwen3.5-4B widths 9216/576/144/36/9/1 + null; the slices share weights). Stage 1 routes layers 12+ (20 of 32) with a one-sided budget hinge at target 0.01; stage 2 warm-starts from stage 1 and routes all layers at 0.02 with the hinge on from step 0; the two-sided arms penalize |cost - target| at 0.10 on layers 12+ (t10) or all layers (a10), no exploration. Scored with routing on (the cut carries to inference).",
          "- **KV soft tokens**: a fixed fraction (1/6 or 1/3) of documents keep real tokens, every other document collapses to one projected soft token in the KV; detached soft KV, no distillation, torch attention backend; scored with plain full attention (training-only saving). kv17/kv33 force the gold documents into the kept set (leaks the answer on id-answer tasks); kvb17/kvb33 are gold-blind (random kept set). Oolong has no gold subset, so its kv17/kv33 are gold-blind by construction.",
          "- **FLOPs**: training FLOPs priced per example at its real length (attention quadratic in the example, not the packed window) from the harvested example lengths; FFN arms scale the FFN share by the mean routed cost the trainer measured; KV arms are metered on their compacted rows. Stage 2 is charged stage 1 + stage 2.",
          "- Accuracy = mean f1 over the task's eval rungs that have a value (same fixed eval sets as the dense campaign; eval_size 500-600 per rung). Where a rung is missing for one point (e.g. contradiction dense 28M lacks 32k) its mean covers fewer rungs -- compare per-rung rows for those.\n"]
    for task in sorted(by):
        md.append(f"\n## {task}\n")
        budgets = sorted({b for arm in by[task] for b in by[task][arm]}, key=lambda b: int(b[:-1]))
        md.append("| arm | " + " | ".join(f"{b} tokens<br>mean f1 (PF)" for b in budgets) + " |")
        md.append("|---|" + "---|" * len(budgets))
        for arm in ARM_ORDER:
            if arm not in by[task]:
                continue
            cells = []
            for b in budgets:
                r = by[task][arm].get(b)
                if not r or r["mean_f1"] in ("", "None"):
                    cells.append("–")
                else:
                    pf = r["actual_pflops"]; pf = f"{float(pf):.0f}" if pf not in ("", "None") else "?"
                    cells.append(f"{float(r['mean_f1']):.3f} ({pf})")
            md.append(f"| {ARM_LABEL.get(arm, arm)} | " + " | ".join(cells) + " |")
        # per-rung detail
        md.append(f"\nPer-rung f1 ({', '.join(c[3:] for c in f1cols)}):\n")
        for arm in ARM_ORDER:
            for b in budgets:
                r = by[task].get(arm, {}).get(b)
                if r and r["mean_f1"] not in ("", "None"):
                    vals = " / ".join(f"{float(r[c]):.2f}" if r.get(c) not in ("", "None", None) else "–" for c in f1cols)
                    md.append(f"- {arm} @ {b}: {vals}")
    if os.path.exists(f"{OUT}/fits35.md"):
        md.append("\n## Fitted scaling trends\n")
        md.append(open(f"{OUT}/fits35.md").read().split("\n", 1)[1])
    md.append("\n## Plots\n")
    for task in sorted(by):
        for suffix in ("_flop_fit35.png",):
            p = f"{OUT}/{task}{suffix}"
            if os.path.exists(p):
                md.append(f"- `{p}`")
    open(REPORT, "w").write("\n".join(md) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
