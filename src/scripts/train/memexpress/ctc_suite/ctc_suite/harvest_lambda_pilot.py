#!/usr/bin/env python3
"""Harvest a lambda CTC-suite 2k-pilot training run into the pilot_2k result + curve JSONs.

Lambda is air-gapped (WANDB offline), so the training log is the only record. This parses the
olmo_core console log for the per-step ``train/CE loss`` series and writes both
``results/ctc_suite/pilot_2k/<task>.json`` (niah.json schema) and ``curves/<task>.json``.

Usage:
    python harvest_lambda_pilot.py --task obliq_retrieval --jobid 109691
    python harvest_lambda_pilot.py --task obliq_retrieval --jobid 109691 --run-name ctc-1ep-obliq_retrieval-full-lambda
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

LROOT = "/accounts/projects/sewonm/prasann/ctc_suite"
REPO = Path("/accounts/projects/berkeleynlp/prasann/projects/OLMo-core")
OUT = REPO / "results/ctc_suite/pilot_2k"
GIT_COMMIT = "2b4c95c40bffd5634640c97c9a2b66503731ae0e"

STEP_RE = re.compile(r"\[step=(\d+)/(\d+),epoch=")
CE_RE = re.compile(r"train/CE loss=([0-9.]+)")
HDR_RE = re.compile(r"n_examples=(\d+).*?seq_len=(\d+)")


def fetch_log(run: str, jobid: str) -> str:
    """Read the lambda log for this run over ssh (air-gapped -> ssh cat).

    The sbatch output is ``logs/ctc_suite_%x_%j.log`` with ``%x`` = job-name = RUN, so the
    log filename is derived from the run name (not the task) to support re-run suffixes.
    """
    remote = f"{LROOT}/logs/ctc_suite_{run}_{jobid}.log"
    return subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=5",
            "lambda",
            f"cat {remote}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def parse(log: str):
    n_examples = seq_len = None
    m = HDR_RE.search(log)
    if m:
        n_examples, seq_len = int(m.group(1)), int(m.group(2))
    # walk lines: a step= line is followed (a few lines later) by its CE line
    steps, cur_step = [], None
    for line in log.splitlines():
        sm = STEP_RE.search(line)
        if sm:
            cur_step = int(sm.group(1))
            continue
        cm = CE_RE.search(line)
        if cm and cur_step is not None:
            steps.append({"step": cur_step, "ce": float(cm.group(1))})
            cur_step = None
    completed = "Training complete" in log and "DONE" in log
    return n_examples, seq_len, steps, completed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--jobid", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--scale", default="0.8b")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="write result even if training hasn't completed (default: refuse)",
    )
    args = ap.parse_args()
    run = args.run_name or f"ctc-1ep-{args.task}-full-lambda"

    log = fetch_log(run, args.jobid)
    n_examples, seq_len, steps, completed = parse(log)
    if not steps:
        raise SystemExit(
            f"no CE steps parsed for {args.task} (job {args.jobid}) -- crashed? check log"
        )
    if not completed and not args.allow_partial:
        last = steps[-1]["step"]
        raise SystemExit(
            f"{args.task} (job {args.jobid}) still RUNNING (last step {last}, CE {steps[-1]['ce']:.3f}); "
            f"not writing partial result. Re-run once 'Training complete'."
        )

    ce0, cef = steps[0]["ce"], steps[-1]["ce"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curves").mkdir(parents=True, exist_ok=True)

    curve = {
        "task": args.task,
        "scale": args.scale,
        "variant": args.variant,
        "seq_len": seq_len,
        "compute_pool": "lambda-1gpu",
        "steps": steps,
    }
    (OUT / "curves" / f"{args.task}.json").write_text(json.dumps(curve, indent=2))

    result = {
        "ckpt_path": f"{LROOT}/ckpts/{run}",
        "compute_pool": "lambda-1gpu",
        "curve_path": str(OUT / "curves" / f"{args.task}.json"),
        "epochs": 1,
        "eval_metric": None,
        "eval_metric_name": None,
        "eval_size": None,
        "git_commit": GIT_COMMIT,
        "notes": f"lambda air-gapped run (job {args.jobid}); eval: N/A (harvest from log). "
        f"train {'completed' if completed else 'INCOMPLETE'}.",
        "pass": bool(completed and cef < ce0),
        "scale": args.scale,
        "seq_len": seq_len,
        "task": args.task,
        "train_ce_start": ce0,
        "train_ce_end": cef,
        "train_log": f"lambda:{LROOT}/logs/ctc_suite_{run}_{args.jobid}.log",
        "train_steps_logged": len(steps),
        "variant": args.variant,
        "wandb_url_if_any": None,
    }
    (OUT / f"{args.task}.json").write_text(json.dumps(result, indent=2))
    print(
        f"{args.task}: CE {ce0:.3f} -> {cef:.4f}  steps={len(steps)}  n_ex={n_examples} "
        f"seq={seq_len}  completed={completed}  pass={result['pass']}"
    )


if __name__ == "__main__":
    main()
