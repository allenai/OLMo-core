"""Write a vLLM-backend eval result through the suite's canonical results_io schema.

Takes the output of grade_responses.py (contra) or grade_responses_generic.py (everything
else) plus the run_vllm_eval*.py timing, and emits the same §8 record shape run_rung_eval.py
writes for the native backend -- same TASK_METRIC/TASK_CLASS/cot_label tables (imported
directly, not duplicated), same results/ctc_suite/<task>/<model>_<arm>/rung_<T>.json +
all_results.jsonl layout -- so vLLM and native results are directly comparable in
results-hub. provenance.eval_backend="vllm" is the only backend-identifying field; everything
else about the record is identical in shape to the native path.
"""

import argparse
import datetime
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "common"))  # corpus_reasoning.*
# TODO(port): depends on src/scripts/eval/ctc_suite (results_io, run_rung_eval), not ported

from scripts.eval.ctc_suite import results_io  # noqa: E402
from scripts.eval.ctc_suite.run_rung_eval import (  # noqa: E402
    TASK_ALIASES,
    TASK_CLASS,
    TASK_METRIC,
    _cot_label,
    _git_commit,
)


def find_metrics_dict(summary, task, seg_task):
    """The two graders name their metrics dict differently (grade_responses.py uses the
    task name itself, e.g. "contradiction"; grade_responses_generic.py always uses
    "metrics") -- try both plus the §8 summary_key convention before giving up."""
    candidates = ["metrics", task, seg_task, TASK_METRIC.get(seg_task, (None,))[0]]
    for k in candidates:
        if k and isinstance(summary.get(k), dict):
            return summary[k]
    raise SystemExit(
        f"no metrics dict found in grade summary (tried {candidates}); keys={sorted(summary)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade-json", required=True, help="output of grade_responses[_generic].py")
    ap.add_argument("--task", required=True, help="§8 canonical task name or alias (e.g. hotpotqa)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--model-scale", required=True)
    ap.add_argument("--arm", default="full")
    ap.add_argument("--rung-tokens", type=int, required=True)
    ap.add_argument("--trained-ctx", default="2k-32k-joint-uniform")
    ap.add_argument("--cot-mode", default="none")
    ap.add_argument("--gen-seconds", type=float, required=True, help="vLLM generate() wall-clock")
    ap.add_argument(
        "--load-seconds", type=float, default=None, help="LLM(...) construction wall-clock"
    )
    ap.add_argument("--out-root", default=os.path.join(REPO_ROOT, "results", "ctc_suite"))
    ap.add_argument("--small-eval-ok", action="store_true")
    args = ap.parse_args()

    seg_task = TASK_ALIASES.get(args.task, args.task)
    if seg_task not in TASK_METRIC:
        raise SystemExit(f"--task {args.task!r} -> {seg_task!r} not in TASK_METRIC")
    _summary_key, metric_key, metric_name = TASK_METRIC[seg_task]

    with open(args.grade_json) as f:
        summary = json.load(f)
    res = find_metrics_dict(summary, args.task, seg_task)
    if metric_key not in res:
        raise SystemExit(f"metric_key={metric_key!r} not in metrics dict (keys={sorted(res)})")
    metric_value = float(res[metric_key])
    eval_size = int(summary.get("eval_size", 0))
    parse_rate = res.get("parse_rate")

    if metric_value == 0.0 and parse_rate == 1.0:
        raise SystemExit(
            "GUARDRAIL: metric_value=0.0 at parse_rate=1.0 -- inspect the vLLM generations "
            f"({args.grade_json}) before believing this; nothing was recorded."
        )

    aux_metrics = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    aux_metrics["eval_seconds"] = round(args.gen_seconds, 1)
    if args.load_seconds is not None:
        aux_metrics["load_seconds"] = round(args.load_seconds, 1)

    record = {
        "task": seg_task,
        "complexity_class": TASK_CLASS[seg_task],
        "model": args.model_scale,
        "arm": args.arm,
        "trained_ctx": args.trained_ctx,
        "rung_tokens": args.rung_tokens,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "aux_metrics": aux_metrics,
        "eval_size": eval_size,
        "cot_label": _cot_label(args.cot_mode),
        "provenance": {
            "git_commit": _git_commit(),
            "data_path": args.eval_jsonl,
            "ckpt_path": args.ckpt,
            "eval_backend": "vllm",
            "launcher": os.path.relpath(__file__, REPO_ROOT),
            "evaluator": (
                "research/landmark/ctc_vllm_validation/run_vllm_eval_generic.py"
                if args.task != "contradiction" and args.task != "contra"
                else "research/landmark/ctc_vllm_validation/run_vllm_eval.py"
            ),
            "evaluator_variant": args.arm,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "aux": {
            "grade_json": os.path.abspath(args.grade_json),
        },
    }
    if eval_size < results_io.MIN_EVAL_SIZE:
        if not args.small_eval_ok:
            raise SystemExit(
                f"eval_size={eval_size} < {results_io.MIN_EVAL_SIZE} and --small-eval-ok not "
                "given; refusing to record a small eval silently (project directive)."
            )
        record["small_eval_ok"] = True
        record["small_eval_warning"] = results_io.small_eval_warning(
            min(max(metric_value, 0.0), 1.0), eval_size
        )

    per_rung, jsonl_path = results_io.write_result(args.out_root, record)
    warn = f"  [{record['small_eval_warning']}]" if "small_eval_warning" in record else ""
    print(
        f"[write_vllm_result] {seg_task} {args.model_scale} {args.arm} rung={args.rung_tokens} "
        f"{metric_name}={metric_value:.3f} eval_size={eval_size}{warn} -> {per_rung}",
        flush=True,
    )


if __name__ == "__main__":
    main()
