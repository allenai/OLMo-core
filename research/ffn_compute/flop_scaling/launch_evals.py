"""
Launch the multi-rung Beaker evals for FLOP-scaling runs (records/flop-scaling-ffn-kv-plan.md §3).

One Beaker job per run: ``run_q4b_beaker_multirung_eval.py`` with the docchunk (box-marker,
dense-emitter) evaluator, query position AFTER (the shards' layout), no-CoT, Qwen3 tokenizer,
pointed at the run's model-only export on weka (``--ckpt``). The ffnmoe arm's routing is enabled
by the evaluator from the checkpoint's config.json; the soft-token arms evaluate as plain full
attention (the checkpoint's recorded architecture).

    python research/ffn_compute/flop_scaling/launch_evals.py fs-contradiction-dense-sh16M fs-contradiction-kv17-sh16M
    python research/ffn_compute/flop_scaling/launch_evals.py --dry-run fs-oolong-ffnmoe-s2-sh32M
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[3])
LAUNCHER = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
CKPTS = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts"
# Each eval job is recorded in the ops run registry (kind=eval, parent_run=<run>) by the eval
# launcher itself (--experiment below); LAUNCH_LEDGER.tsv is kept as history only.
EXPERIMENT = "flop_scaling"
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
            "--max-test", str(args.max_test), "--priority", "urgent", "--experiment", EXPERIMENT,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print(" ".join(cmd), flush=True)
        res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src:{REPO}/research/common:{REPO}/recipes/eval/legacy_native"),
                             capture_output=True, text=True)
        out = (res.stdout + res.stderr).strip().splitlines()
        tail = "\n".join(out[-4:])
        for line in out:
            if line.startswith("[ops]"):
                print(f"  {line}", flush=True)
        print(f"  -> rc={res.returncode}: {tail[:400]}", flush=True)
        rows.append((run, task, res.returncode, tail.replace("\t", " ")[:200]))
    failed = [run for run, _task, rc, _tail in rows if rc != 0]
    if failed:
        print(f"LAUNCH-FAILED ({len(failed)}): {' '.join(failed)}", flush=True)


if __name__ == "__main__":
    main()
