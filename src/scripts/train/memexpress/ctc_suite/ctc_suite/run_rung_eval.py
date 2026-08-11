"""
Per-rung eval driver for the CTC suite (``records/ctc-suite-scaling-plan.md`` §4 Eval / §8).

Wraps the PROVEN native olmo-core evaluators for one (task, checkpoint, rung):

* ``contradiction`` -> ``src/corpus_reasoning/eval/eval_lc_native_docchunk_contra.py``
* everything else in :data:`NATIVE_TASKS` (all 20 plan §3 canonical task keys) ->
  ``src/corpus_reasoning/eval/eval_lc_native_docchunk.py`` (multi-task via its ``TASK_CFG``).
  Plan/suite catalog names that aren't a canonical ``--task`` key (``hotpotqa``,
  ``niah_contradiction``, ``beir_scifact``, ``beir_fiqa``, ``msmarco``, ``outlier_amazon``,
  ``helmet_qa``, ``helmet_summ``) route through :data:`TASK_ALIASES` to the canonical key that
  carries the right metric (e.g. ``hotpotqa`` -> ``retrieval`` -> ``gold_id_f1``).

Both evaluators have the Qwen3.5 ``--doc-start-id/--doc-end-id/--eos-token-id`` overrides +
``--per-example-out``, so the driver passes the id args and a generations-dump path through for
every task.

Naming trap (deliberate, documented): the DRIVER's ``--variant`` follows the plan's arms --
``dense`` = full-attention eval, ``chunked`` = doc-chunked mask -- while the UNDERLYING evaluators
call full attention ``full`` and the chunked mask ``dense``. The mapping lives in
:data:`VARIANT_TO_EVALUATOR` and nowhere else.

The driver shells out via ``torchrun``, parses the evaluator's summary JSON, and writes the §8
record through :mod:`results_io` (per-rung JSON + ``all_results.jsonl``), keeping the raw evaluator
JSON and the generations dump next to the rung file. Known traps enforced:

* ``--max-length`` defaults to ``rung_tokens + 2048`` and is bumped/refused whenever it cannot fit
  prompt + decode budget (maxlen-truncation trap: an undersized max-length truncates the prompt and
  silently produces metric=0 at parse_rate=1).
* metric==0.0 at parse_rate==1.0 -> nonzero exit with a loud "dump generations" message.
* wall-clock per rung recorded into ``aux_metrics.eval_seconds``.

No GPU on the dev host: use ``--print-cmd`` to see the exact torchrun command without executing::

    PYTHONPATH=src python -m scripts.eval.ctc_suite.run_rung_eval --task contradiction \\
      --ckpt /data/ckpts/q35-08b-contra-full/step1100 --variant dense --arm full \\
      --model-scale qwen3.5-0.8b --rung-tokens 2048 --eval-jsonl data/contra_eval_rung2048.jsonl \\
      --print-cmd
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

try:  # package import (PYTHONPATH=src) or same-directory fallback (direct file execution)
    from scripts.eval.ctc_suite import results_io
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import results_io  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO_ROOT / "src" / "corpus_reasoning" / "eval"

#: Complexity class per canonical ``--task`` name (plan §3 table). ``N`` = O(N), ``NM`` = O(NM),
#: ``N2`` = O(N^2), ``N3`` = O(N^3).
TASK_CLASS = {
    "retrieval": "N",
    "cot_retrieval": "N",
    "oolong": "N",
    "qa": "N",
    "summarization": "N",
    "rerank": "N",
    "outlier": "NM",
    "grouping": "NM",
    "grouping_labeled": "NM",
    "textgroups": "NM",
    "contradiction": "N2",
    "redundancy": "N2",
    "absence": "N2",
    "strmatch": "N2",
    "qdmatch": "N2",
    "xabsence": "N2",
    "mathmatch": "N2",
    "reorder": "N2",
    "cycle": "N3",
    "groups4": "N3",
}
#: Canonical ``--task`` aliases (run-name / launcher / plan-§3 catalog shorthands). Suite/plan
#: names on the left route to the evaluator's canonical task key on the right; the metric that
#: comes along is whatever ``TASK_METRIC[canonical]`` says (e.g. hotpotqa -> retrieval ->
#: gold_id_f1) -- no separate per-alias metric table needed since the alias IS the canonical task.
TASK_ALIASES = {
    "nq": "retrieval",
    "contra": "contradiction",
    "hotpotqa": "retrieval",  # metric gold_id_f1 (plan §3 row 2)
    "niah_contradiction": "retrieval",  # metric gold_id_f1 (row 3; NIAH-contra transform)
    "beir_scifact": "retrieval",  # row 7
    "beir_fiqa": "retrieval",  # row 8
    "msmarco": "retrieval",  # row 9
    "outlier_amazon": "outlier",  # row 12
    "helmet_qa": "qa",  # row 5
    "helmet_summ": "summarization",  # row 6
}

#: Tasks the proven native evaluators support today. Everything else needs an evaluator first.
#: Full plan §3 canonical task set (20 keys; qdmatch's 3 catalog variants -- NQ/HPQA/ObliQ -- and
#: the retrieval-family aliases above all route to one of these via TASK_ALIASES).
NATIVE_TASKS = (
    "oolong",
    "contradiction",
    "retrieval",
    "rerank",
    "outlier",
    "redundancy",
    "absence",
    "xabsence",
    "strmatch",
    "qdmatch",
    "mathmatch",
    "cycle",
    "groups4",
    "textgroups",
    "reorder",
    "grouping",
    "grouping_labeled",
    "qa",
    "summarization",
    "cot_retrieval",
)

#: task -> (summary key in the evaluator JSON, metric key inside it, §8 metric_name). Mirrors
#: eval_lc_native_docchunk.py's TASK_CFG scorer reuse: several canonical tasks share a scorer
#: function there (redundancy/strmatch/mathmatch -> _eval_contradiction's "f1", xabsence ->
#: _eval_absence's "f1", groups4/textgroups -> _eval_cycle's "f1", cot_retrieval ->
#: _eval_retrieval's "f1", grouping_labeled -> _eval_grouping's "pairwise_f1") -- the metric_key
#: read out of the evaluator JSON follows the same reuse; only metric_name (the §8 label) is
#: task-specific for readability in results-hub.
TASK_METRIC = {
    "oolong": ("oolong", "score", "partial_credit"),
    "contradiction": ("contradiction", "f1", "set_f1"),
    "retrieval": ("metrics", "f1", "gold_id_f1"),
    "rerank": ("metrics", "mrr@10", "mrr@10"),
    "outlier": ("metrics", "f1", "set_f1"),
    "redundancy": ("metrics", "f1", "set_f1"),
    "absence": ("metrics", "f1", "set_f1"),
    "xabsence": ("metrics", "f1", "set_f1"),
    "strmatch": ("metrics", "f1", "set_f1"),
    "qdmatch": ("metrics", "f1", "pair_f1"),
    "mathmatch": ("metrics", "f1", "set_f1"),
    "cycle": ("metrics", "f1", "cycle_f1"),
    "groups4": ("metrics", "f1", "cycle_f1"),
    "textgroups": ("metrics", "f1", "textgroups_f1"),
    "reorder": ("metrics", "kendall_tau", "kendall_tau"),
    "grouping": ("metrics", "pairwise_f1", "pairwise_f1"),
    "grouping_labeled": ("metrics", "pairwise_f1", "pairwise_f1"),
    "qa": ("metrics", "f1", "token_f1"),
    "summarization": ("metrics", "rouge1_f", "rouge1_f"),
    "cot_retrieval": ("metrics", "f1", "gold_id_f1"),
}

#: Per-task decode budget defaults (mirroring the evaluators' TASK_CFG, which in turn mirrors
#: evaluate.py's ``--max-tokens`` per-task overrides, ~lines 1171-1231). Contradiction is
#: CoT-dependent: enumerate-CoT walks every claim (~2400 tokens); no-cot is one pair list that
#: early-stops at ``]]`` (512 is generous) -- see ``resolve``.
TASK_MAX_NEW = {
    "oolong": 256,
    "contradiction": 512,  # no-cot; enumerate CoT overrides to 2400 in resolve()
    "retrieval": 64,
    "rerank": 512,
    "outlier": 256,
    "redundancy": 200,
    "absence": 200,
    "xabsence": 200,
    "strmatch": 200,
    "qdmatch": 200,
    "mathmatch": 200,
    "cycle": 200,
    "groups4": 200,
    "textgroups": 200,
    "reorder": 1024,
    "grouping": 2048,
    "grouping_labeled": 2048,
    "qa": 64,
    "summarization": 1024,
    "cot_retrieval": 512,
}
CONTRA_ENUMERATE_MAX_NEW = 2400

#: Driver arm name -> underlying evaluator ``--variant``. The evaluators call the doc-chunked mask
#: "dense" (DocumentChunkedAttention) and plain causal attention "full"; the plan (and this driver)
#: says "dense" for full attention and "chunked" for the mask. Translate exactly once, here.
VARIANT_TO_EVALUATOR = {"dense": "full", "chunked": "dense"}

# Qwen3.5 tokenizer marker ids (driver defaults; plan §4).
QWEN35_IDS = (248049, 248050, 248044)


def _cot_label(cot_mode: str) -> str:
    """Map a converter/evaluator CoT mode to the project's result label (CoT labeling directive)."""
    if cot_mode == "none":
        return "no-cot"
    if cot_mode == "enumerate":
        return "long-cot"
    return "short-cot"


def _git_commit() -> str:
    """The repo HEAD commit (provenance; ``unknown`` if git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (kept separate so ``--help`` stays testable without a GPU)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--task",
        required=True,
        help=f"one of {NATIVE_TASKS} (aliases: nq->retrieval, contra->contradiction)",
    )
    ap.add_argument(
        "--ckpt", required=True, help="distcp step dir (config.json + model_and_optim/)"
    )
    ap.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANT_TO_EVALUATOR),
        help="dense = full-attention eval; chunked = doc-chunked mask "
        "(maps to the evaluators' full/dense respectively)",
    )
    ap.add_argument(
        "--rung-tokens",
        type=int,
        required=True,
        help="the rung's nominal context length in tokens (2048/4096/8192/16384/32768)",
    )
    ap.add_argument("--eval-jsonl", required=True, help="per-rung eval JSONL (unified format)")
    ap.add_argument("--arm", required=True, help="§8 arm label, e.g. full | chunked | chunked-mix")
    ap.add_argument("--model-scale", required=True, help="§8 model label, e.g. qwen3.5-0.8b")
    ap.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "results" / "ctc_suite"),
        help="results root (default results/ctc_suite)",
    )
    ap.add_argument(
        "--backend",
        default="native",
        choices=["native"],
        help="eval backend. TODO(vllm): accept 'vllm' here once the Qwen3.5-hybrid "
        "vLLM validation in flight lands; dispatch would replace the torchrun "
        "command built in build_command().",
    )
    # Tokenizer / marker ids (Qwen3.5 defaults per plan §4).
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-0.8B-Base",
        help="HF tokenizer id; same tokenizer.json across Qwen3.5 sizes",
    )
    ap.add_argument("--doc-start-id", type=int, default=QWEN35_IDS[0], help="<|box_start|> id")
    ap.add_argument("--doc-end-id", type=int, default=QWEN35_IDS[1], help="<|box_end|> id")
    ap.add_argument("--eos-token-id", type=int, default=QWEN35_IDS[2], help="eos / doc-sep id")
    ap.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="evaluator max sequence length. Default rung_tokens + 2048, auto-bumped "
        "to fit the decode budget -- an undersized value truncates the prompt and "
        "silently yields metric=0 at parse_rate=1 (maxlen-truncation trap).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="decode budget override (default: per-task table)",
    )
    ap.add_argument(
        "--cot-mode",
        default=None,
        help="prompt CoT mode; MUST match the training shard (default: none -- the "
        "suite shards are tokenized no-cot unless a task's recipe says otherwise)",
    )
    ap.add_argument(
        "--cot-label", default=None, help="§8 cot_label override (default derived from --cot-mode)"
    )
    ap.add_argument(
        "--max-test-samples",
        type=int,
        default=100000,
        help="cap on eval examples (default: all; the evaluators' own default is 100, "
        "which would silently shrink the eval)",
    )
    ap.add_argument("--nproc", type=int, default=8, help="torchrun --nproc_per_node")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="examples decoded per forward within each rank (left-padded, KV-cached). Default 1 "
        "keeps the original bs=1 behavior. Only 'dense'/'chunked' and 'full' variants support "
        "batching -- see corpus_reasoning.eval.batched_native_decode and "
        "results/ctc_suite/batched_eval_status.md for the parity proof + safe batch sizes.",
    )
    ap.add_argument("--master-port", type=int, default=None, help="torchrun --master_port")
    ap.add_argument(
        "--trained-ctx",
        default="2k-32k-joint-uniform",
        help="§8 trained_ctx label for the checkpoint",
    )
    ap.add_argument(
        "--mask-mix",
        default=None,
        help="optional JSON for the §8 mask_mix field, e.g. "
        '\'{"mode": "curriculum", "start_p": 0.8, "end_p": 0.0}\'',
    )
    ap.add_argument("--n-docs", type=int, default=None, help="optional §8 n_docs for this rung")
    ap.add_argument(
        "--small-eval-ok",
        action="store_true",
        help="allow eval_size < 500 (record gets small_eval_ok + the binomial-SE "
        "warning string; required for the 488-example contradiction set)",
    )
    ap.add_argument(
        "--print-cmd",
        action="store_true",
        help="print the exact torchrun command (with env) and exit without running",
    )
    ap.add_argument(
        "--allow-short-max-length",
        action="store_true",
        help="bypass the max-length >= rung + decode-budget check (only sane when the "
        "eval file's prompts are known to be much shorter than the rung label)",
    )
    return ap


def resolve(args: argparse.Namespace):
    """
    Resolve CLI args into the concrete eval plan (no side effects, no GPU).

    :returns: dict with keys ``task``, ``evaluator``, ``ev_variant``, ``max_new``, ``max_length``,
        ``cot_mode``, ``out_dir``, ``raw_out``, ``gen_out``, ``cmd``, ``env_pythonpath``.

    :raises SystemExit: On unsupported tasks, wrong marker ids for the multi-task evaluator, or a
        max-length that cannot fit prompt + decode budget.
    """
    task = TASK_ALIASES.get(args.task, args.task)
    if task not in NATIVE_TASKS:
        known = task in TASK_CLASS
        raise SystemExit(
            f"--task {args.task!r}: no proven native evaluator yet "
            f"({'in the plan but unported' if known else 'unknown task'}). Native tasks today: "
            f"{NATIVE_TASKS}. Port the task into eval_lc_native_docchunk.py TASK_CFG first."
        )

    ev_variant = VARIANT_TO_EVALUATOR[args.variant]
    if task == "contradiction":
        evaluator = EVAL_DIR / "eval_lc_native_docchunk_contra.py"
    else:
        evaluator = EVAL_DIR / "eval_lc_native_docchunk.py"

    cot_mode = args.cot_mode if args.cot_mode is not None else "none"
    if args.max_new_tokens is not None:
        max_new = args.max_new_tokens
    elif task == "contradiction" and cot_mode == "enumerate":
        max_new = CONTRA_ENUMERATE_MAX_NEW
    else:
        max_new = TASK_MAX_NEW[task]

    # ---- maxlen-truncation trap ----
    # The evaluators skip any prompt longer than (max_length - max_new) and score it as empty --
    # i.e. an undersized max-length reads as a perfectly-parsed 0.0. Default to rung + 2048 and
    # bump until the decode budget fits; refuse an explicit undersized value.
    need = args.rung_tokens + max_new + 512
    if args.max_length is None:
        max_length = args.rung_tokens + 2048
        if max_length < need:
            print(
                f"[run_rung_eval] NOTE: default max_length {max_length} cannot fit rung "
                f"{args.rung_tokens} + max_new {max_new}; bumping to {need} "
                "(maxlen-truncation trap).",
                flush=True,
            )
            max_length = need
    else:
        max_length = args.max_length
        if max_length < need and not args.allow_short_max_length:
            raise SystemExit(
                f"--max-length {max_length} < rung_tokens {args.rung_tokens} + max_new {max_new} "
                f"+ 512 = {need}: prompts would be truncated/skipped and the eval would silently "
                "report metric=0 at parse_rate=1 (see the goldgrad maxlen-truncation record). "
                "Raise --max-length (or pass --allow-short-max-length if the prompts are truly "
                "shorter than the rung label)."
            )

    out_dir = os.path.join(
        args.out_root,
        results_io._sanitize_component(task),
        f"{results_io._sanitize_component(args.model_scale)}"
        f"_{results_io._sanitize_component(args.arm)}",
    )
    raw_out = os.path.join(out_dir, f"rung_{args.rung_tokens}.raw.json")
    gen_out = os.path.join(out_dir, f"rung_{args.rung_tokens}.generations.json")

    cmd = ["torchrun", f"--nproc_per_node={args.nproc}"]
    if args.master_port is not None:
        cmd.append(f"--master_port={args.master_port}")
    cmd += [
        str(evaluator),
        "--variant",
        ev_variant,
        "--model-path",
        args.ckpt,
        "--out",
        raw_out,
        "--tokenizer",
        args.tokenizer,
        "--max-length",
        str(max_length),
        "--max-test-samples",
        str(args.max_test_samples),
        "--cot-mode",
        cot_mode,
        "--doc-start-id",
        str(args.doc_start_id),
        "--doc-end-id",
        str(args.doc_end_id),
        "--eos-token-id",
        str(args.eos_token_id),
        "--per-example-out",
        gen_out,
        "--batch-size",
        str(args.batch_size),
    ]
    if task == "contradiction":
        cmd += [
            "--contra-data",
            args.eval_jsonl,
            "--contra-max-new-tokens",
            str(max_new),
        ]
    else:
        cmd += [
            "--task",
            task,
            "--data",
            args.eval_jsonl,
            "--max-new-tokens",
            str(max_new),
        ]

    return {
        "task": task,
        "evaluator": str(evaluator),
        "ev_variant": ev_variant,
        "max_new": max_new,
        "max_length": max_length,
        "cot_mode": cot_mode,
        "out_dir": out_dir,
        "raw_out": raw_out,
        "gen_out": gen_out,
        "cmd": cmd,
        "env_pythonpath": str(REPO_ROOT / "src"),
    }


def main() -> None:
    """Drive one per-rung eval end to end (or just print the command with ``--print-cmd``)."""
    args = build_parser().parse_args()
    plan = resolve(args)
    shown = f"PYTHONPATH={plan['env_pythonpath']} " + shlex.join(plan["cmd"])
    if args.print_cmd:
        print(shown)
        return

    os.makedirs(plan["out_dir"], exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = plan["env_pythonpath"] + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[run_rung_eval] {shown}", flush=True)
    t0 = time.time()
    proc = subprocess.run(plan["cmd"], env=env)
    eval_seconds = round(time.time() - t0, 1)
    if proc.returncode != 0:
        raise SystemExit(f"evaluator exited {proc.returncode} after {eval_seconds}s")
    if not os.path.exists(plan["raw_out"]):
        raise SystemExit(f"evaluator exited 0 but wrote no summary at {plan['raw_out']}")

    with open(plan["raw_out"]) as f:
        summary = json.load(f)
    summary_key, metric_key, metric_name = TASK_METRIC[plan["task"]]
    res = summary.get(summary_key)
    if not isinstance(res, dict) or metric_key not in res:
        raise SystemExit(
            f"evaluator summary {plan['raw_out']} has no {summary_key!r}[{metric_key!r}] "
            f"(keys: {sorted(summary)}); refusing to invent a metric."
        )
    metric_value = float(res[metric_key])
    eval_size = int(summary.get("eval_size", 0))
    parse_rate = res.get("parse_rate")
    skipped = summary.get("skipped_too_long")

    # ---- guardrails (known traps) ----
    if metric_value == 0.0 and parse_rate == 1.0:
        raise SystemExit(
            "GUARDRAIL: metric_value=0.0 at parse_rate=1.0 -- this exact signature has meant a "
            "TRUNCATED PROMPT (maxlen-truncation trap), not a bad model. DUMP THE GENERATIONS "
            f"({plan['gen_out']}) before believing this number. Nothing was recorded."
        )
    if skipped:
        print(
            f"[run_rung_eval] WARNING: {skipped}/{eval_size} prompts skipped as too long for "
            f"max_length={plan['max_length']} and scored as empty -- the metric is contaminated. "
            "Raise --max-length.",
            flush=True,
        )
        if skipped >= eval_size:
            raise SystemExit("GUARDRAIL: every prompt was skipped; nothing was actually evaluated.")

    aux_metrics = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    aux_metrics["eval_seconds"] = eval_seconds  # eval-wallclock directive: always record timing
    if skipped is not None:
        aux_metrics["skipped_too_long"] = skipped

    record = {
        "task": plan["task"],
        "complexity_class": TASK_CLASS[plan["task"]],
        "model": args.model_scale,
        "arm": args.arm,
        "trained_ctx": args.trained_ctx,
        "rung_tokens": args.rung_tokens,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "aux_metrics": aux_metrics,
        "eval_size": eval_size,
        "cot_label": args.cot_label or _cot_label(plan["cot_mode"]),
        "provenance": {
            "git_commit": _git_commit(),
            "data_path": args.eval_jsonl,
            "ckpt_path": args.ckpt,
            "eval_backend": args.backend,
            "launcher": os.path.relpath(__file__, REPO_ROOT),
            "evaluator": os.path.relpath(plan["evaluator"], REPO_ROOT),
            "evaluator_variant": plan["ev_variant"],
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "aux": {
            "raw_eval_json": plan["raw_out"],
            "generations_path": plan["gen_out"],
            "max_length": plan["max_length"],
            "max_new_tokens": plan["max_new"],
            "nproc": args.nproc,
        },
    }
    if args.mask_mix:
        record["mask_mix"] = json.loads(args.mask_mix)
    if args.n_docs is not None:
        record["n_docs"] = args.n_docs
    if eval_size < results_io.MIN_EVAL_SIZE:
        if not args.small_eval_ok:
            raise SystemExit(
                f"eval_size={eval_size} < {results_io.MIN_EVAL_SIZE} and --small-eval-ok not "
                "given; refusing to record a small eval silently (project directive). The raw "
                f"evaluator JSON is still at {plan['raw_out']}."
            )
        record["small_eval_ok"] = True
        record["small_eval_warning"] = results_io.small_eval_warning(
            min(max(metric_value, 0.0), 1.0), eval_size
        )

    per_rung, jsonl_path = results_io.write_result(args.out_root, record)
    warn = f"  [{record['small_eval_warning']}]" if "small_eval_warning" in record else ""
    print(
        f"[run_rung_eval] {plan['task']} {args.model_scale} {args.arm} rung={args.rung_tokens} "
        f"{metric_name}={metric_value:.3f} eval_size={eval_size}{warn} "
        f"eval_seconds={eval_seconds} -> {per_rung} (+ {jsonl_path})",
        flush=True,
    )


if __name__ == "__main__":
    main()
