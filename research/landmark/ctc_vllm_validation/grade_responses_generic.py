"""Generic (any-CTC-suite-task) version of grade_responses.py.

Dispatches to the same per-task scorer function eval_lc_native_docchunk.py uses
(SCORERS dict, reused from evaluate.py), keyed by TASK_CFG's "scorer" field so e.g.
hotpotqa (-> retrieval) and outlier each get graded with the right metric.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="output of run_vllm_eval_generic.py")
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-test-samples", type=int, default=100000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from corpus_reasoning.eval.eval_lc_native_docchunk import TASK_ALIASES, TASK_CFG
    from corpus_reasoning.eval.evaluate import (
        _eval_absence,
        _eval_contradiction,
        _eval_cycle,
        _eval_grouping,
        _eval_oolong,
        _eval_outlier,
        _eval_qa,
        _eval_qdmatch,
        _eval_reorder,
        _eval_rerank,
        _eval_retrieval,
        _eval_summarization,
        load_unified_examples,
    )

    SCORERS = {
        "oolong": _eval_oolong,
        "contradiction": _eval_contradiction,
        "retrieval": _eval_retrieval,
        "rerank": _eval_rerank,
        "outlier": _eval_outlier,
        "absence": _eval_absence,
        "qdmatch": _eval_qdmatch,
        "cycle": _eval_cycle,
        "reorder": _eval_reorder,
        "grouping": _eval_grouping,
        "qa": _eval_qa,
        "summarization": _eval_summarization,
    }

    seg_task = TASK_ALIASES.get(args.task, args.task)
    cfg = TASK_CFG[seg_task]
    scorer = SCORERS[cfg["scorer"]]

    with open(args.responses) as f:
        pack = json.load(f)

    examples = load_unified_examples(
        args.eval_data,
        args.max_test_samples,
        task=seg_task,
        query_position="both",
        use_alpaca=True,
    )
    assert len(examples) == pack["eval_size"], (len(examples), pack["eval_size"])
    responses = [pack["responses"][str(i)] for i in range(len(examples))]
    res, details = scorer(examples, responses)
    summary = {
        "mode": pack["mode"],
        "hf_model": pack["hf_model"],
        "task": args.task,
        "eval_data": args.eval_data,
        "eval_size": len(examples),
        "patch_debug": pack.get("patch_debug", {}),
        "gen_seconds": pack.get("gen_seconds"),
        "metrics": res,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    prim = res.get("f1", next(iter(res.values()), None))
    print(
        f"[grade:{pack['mode']}:{args.task}] primary~={prim} "
        f"parse_rate={res.get('parse_rate')} (eval_size={len(examples)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
