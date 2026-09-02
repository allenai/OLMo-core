#!/usr/bin/env python3
"""One-off helper for the 2k full-attention validation sweep (not part of the §8 pipeline).

Reads a train_ctc_suite.py log (for train_ce_start/train_ce_end/crash detection) and, if present,
the run_rung_eval.py §8 per-rung JSON (for eval_metric/eval_size), and writes the combined record
into results/ctc_suite/pilot_2k/<task>.json per the pilot-2k sweep's own schema (distinct from the
§8 schema in results_io.py -- this is a plumbing-validation summary, not a science-result record).
"""
import argparse
import json
import os
import re
import subprocess


def parse_train_curve(path):
    """Return list of {"step": int, "ce": float} for every logged [step=N/T] + train/CE loss= pair."""
    if not path or not os.path.exists(path):
        return []
    with open(path, errors="ignore") as f:
        text = f.read()
    points = []
    # Each logging interval looks like:
    #   ...[step=50/312,epoch=1,eta=17m]
    #       ... (a few metric lines) ...
    #       train/CE loss=1.026
    step_re = re.compile(r"\[step=(\d+)/(\d+)")
    ce_re = re.compile(r"train/CE loss=([0-9.eE+-]+)")
    cur_step = None
    for line in text.splitlines():
        sm = step_re.search(line)
        if sm:
            cur_step = int(sm.group(1))
            continue
        cm = ce_re.search(line)
        if cm and cur_step is not None:
            try:
                points.append({"step": cur_step, "ce": float(cm.group(1))})
            except ValueError:
                pass
            cur_step = None  # each step block has exactly one CE loss line
    return points


def parse_train_log(path):
    """Return (ce_start, ce_end, crashed, crash_msg, n_steps_seen)."""
    if not path or not os.path.exists(path):
        return None, None, True, "log not found", 0
    ce_vals = []
    crashed = False
    crash_msg = ""
    with open(path, errors="ignore") as f:
        text = f.read()
    for m in re.finditer(r"train/CE loss=([0-9.eE+-]+)", text):
        try:
            ce_vals.append(float(m.group(1)))
        except ValueError:
            pass
    if re.search(r"\bnan\b", text, re.IGNORECASE) and "CE loss=nan" in text:
        crashed = True
        crash_msg = "NaN in train/CE loss"
    if (
        "Training complete" not in text
        and "DONE" not in text.split("\n")[-1:][0]
        and "=== DONE" not in text
    ):
        # heuristic: if no DONE marker at all, treat as incomplete
        if "=== DONE" not in text:
            crashed = True
            crash_msg = crash_msg or "no '=== DONE' marker in log (job likely killed/still running)"
    if re.search(r"Error|Traceback|FAILED|CUDA out of memory|Segmentation fault", text):
        # only flag as crash if it didn't also reach Training complete
        if "Training complete" not in text:
            crashed = True
            m = re.search(
                r"(OLMoEnvironmentError|RuntimeError|AssertionError|CUDA out of memory)[^\n]*", text
            )
            crash_msg = crash_msg or (m.group(0) if m else "error signature found in log")
    ce_start = ce_vals[0] if ce_vals else None
    ce_end = ce_vals[-1] if ce_vals else None
    return ce_start, ce_end, crashed, crash_msg, len(ce_vals)


def find_ckpt_path(log_path):
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, errors="ignore") as f:
        text = f.read()
    m = re.search(r"saved model-only checkpoint -> (\S+)", text)
    return m.group(1) if m else None


def find_wandb_url(log_path):
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, errors="ignore") as f:
        text = f.read()
    m = re.search(r"View run at (https://wandb\.ai/\S+)", text)
    return m.group(1) if m else None


def git_commit(repo_root):
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scale", default="0.8b")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-log", required=True)
    ap.add_argument(
        "--eval-rung-json",
        default=None,
        help="path to run_rung_eval.py's per-rung JSON, if eval was run",
    )
    ap.add_argument("--eval-na-reason", default=None, help="why eval is N/A (no rung file etc)")
    ap.add_argument("--compute-pool", required=True, help="e.g. cubbins-1gpu, mooney-8gpu")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--curves-dir",
        default=None,
        help="dir to write <task>.json loss curve into (results/ctc_suite/pilot_2k/curves)",
    )
    ap.add_argument(
        "--repo-root", default="/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
    )
    args = ap.parse_args()

    ce_start, ce_end, crashed, crash_msg, n_steps = parse_train_log(args.train_log)
    ckpt_path = find_ckpt_path(args.train_log)
    wandb_url = find_wandb_url(args.train_log)

    eval_metric = None
    eval_size = None
    eval_metric_name = None
    notes = []
    if args.max_steps:
        notes.append(
            f"train early-stopped at --max-steps={args.max_steps} (plumbing check, not full epoch)"
        )

    if args.eval_rung_json and os.path.exists(args.eval_rung_json):
        with open(args.eval_rung_json) as f:
            ev = json.load(f)
        eval_metric = ev.get("metric_value")
        eval_size = ev.get("eval_size")
        eval_metric_name = ev.get("metric_name")
        aux = ev.get("aux_metrics", {})
        parse_rate = aux.get("parse_rate")
        if eval_metric == 0.0 and parse_rate == 1.0:
            notes.append(
                "eval_metric=0.0 at parse_rate=1.0 -- possible maxlen-truncation trap or format bug; DUMP GENERATIONS"
            )
        if ev.get("small_eval_warning"):
            notes.append(ev["small_eval_warning"])
    elif args.eval_na_reason:
        notes.append(f"eval: N/A ({args.eval_na_reason})")

    train_ok = (
        (not crashed) and (ce_end is not None) and (ce_start is not None) and (ce_end < ce_start)
    )
    if crashed:
        notes.append(f"TRAIN CRASH/INCOMPLETE: {crash_msg}")

    if args.eval_rung_json and os.path.exists(args.eval_rung_json):
        eval_ok = eval_metric is not None and not (
            eval_metric == 0.0 and (aux.get("parse_rate") == 1.0)
        )
    else:
        eval_ok = True  # N/A eval doesn't fail the task

    passed = bool(train_ok and eval_ok)

    curve_path = None
    if args.curves_dir:
        os.makedirs(args.curves_dir, exist_ok=True)
        curve_points = parse_train_curve(args.train_log)
        curve_record = {
            "task": args.task,
            "scale": args.scale,
            "variant": args.variant,
            "seq_len": args.seq_len,
            "compute_pool": args.compute_pool,
            "steps": curve_points,
            "wandb_url": wandb_url,
        }
        curve_path = os.path.join(args.curves_dir, f"{args.task}.json")
        with open(curve_path, "w") as f:
            json.dump(curve_record, f, indent=2)
            f.write("\n")

    record = {
        "task": args.task,
        "scale": args.scale,
        "variant": args.variant,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "train_ce_start": ce_start,
        "train_ce_end": ce_end,
        "train_steps_logged": n_steps,
        "eval_metric": eval_metric,
        "eval_size": eval_size,
        "eval_metric_name": eval_metric_name,
        "pass": passed,
        "notes": "; ".join(notes) if notes else "",
        "compute_pool": args.compute_pool,
        "ckpt_path": ckpt_path,
        "wandb_url_if_any": wandb_url,
        "git_commit": git_commit(args.repo_root),
        "train_log": args.train_log,
        "curve_path": curve_path,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"[assemble] wrote {args.out}  pass={passed}  ce {ce_start}->{ce_end}  eval={eval_metric}"
    )


if __name__ == "__main__":
    main()
