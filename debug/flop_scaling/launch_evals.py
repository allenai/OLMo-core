"""
Launch the multi-rung Beaker evals for FLOP-scaling runs (records/flop-scaling-ffn-kv-plan.md §3).

One Beaker job per run: ``run_q4b_beaker_multirung_eval.py`` with the docchunk (box-marker,
dense-emitter) evaluator, query position AFTER (the shards' layout), no-CoT, Qwen3 tokenizer,
pointed at the run's model-only export on weka (``--ckpt``). The ffnmoe arm's routing is enabled
by the evaluator from the checkpoint's config.json; the soft-token arms evaluate as plain full
attention (the checkpoint's recorded architecture).

    python debug/flop_scaling/launch_evals.py fs-contradiction-dense-sh16M fs-contradiction-kv17-sh16M
    python debug/flop_scaling/launch_evals.py --dry-run fs-oolong-ffnmoe-s2-sh32M
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
LAUNCHER = f"{REPO}/src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py"
CKPTS = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"
LEDGER = f"{REPO}/debug/flop_scaling/LAUNCH_LEDGER.tsv"
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="fs-<task>-<arm>-sh<B> run names")
    ap.add_argument("--cluster", default="ai2/neptune")
    ap.add_argument("--ngpu", type=int, default=2)
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        parts = run.split("-")
        assert parts[0] == "fs", run
        task = parts[1]
        cmd = [
            sys.executable, LAUNCHER, run, args.cluster,
            "--task", TASK_KEY[task], "--variant", "docchunk",
            "--ckpt", f"{CKPTS}/{run}",
            "--query-position", "after", "--cot-mode", "none",
            "--tokenizer", "Qwen/Qwen3-4B", "--ngpu", str(args.ngpu),
            "--max-test", str(args.max_test), "--priority", "urgent",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print(" ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"),
                             capture_output=True, text=True)
        out = (res.stdout + res.stderr).strip().splitlines()
        tail = "\n".join(out[-4:])
        print(f"  -> rc={res.returncode}: {tail[:400]}", flush=True)
        rows.append((run, task, res.returncode, tail.replace("\t", " ")[:200]))
    if not args.dry_run:
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for run, task, rc, tail in rows:
                w.writerow(["beaker-eval", task, run.split("-")[2], run.split("-sh")[-1], "-", run,
                            args.cluster, dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])


if __name__ == "__main__":
    main()
