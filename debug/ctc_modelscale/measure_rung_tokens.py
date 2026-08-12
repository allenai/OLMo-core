#!/usr/bin/env python3
"""Measure the REAL token length of CTC-suite eval rungs, per task.

The rung files are labelled by token budget (``rung_32768.jsonl``), but the label is a build-time
target, not a measurement -- and the contradiction ladder shows the two can diverge (its 32k rung
carries 1423 docs while its 128k rung carries 2503, a 1.76x doc ratio for a nominal 4x token
ratio). Before building any new rung we need the actual tokens/doc for each task, measured through
the same tokenizer the eval uses, so a "64k" rung really lands at 64k.

Reports, per (task, rung): number of examples, docs/example, median total document tokens, and the
implied effective tokens/doc. Document text only -- the query/CoT/answer overhead is small relative
to a >=32k context and is reported separately as the residual against the rung label.

    python debug/ctc_modelscale/measure_rung_tokens.py --tasks contradiction,niah,nq
"""
import argparse
import json
import os
import statistics

RUNG_ROOT = "/scratch/users/prasann/ctc_suite_staged/eval_rungs"
TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"


def doc_text(d):
    if isinstance(d, dict):
        t = d.get("title") or ""
        return (t + " " + d.get("text", "")).strip() if t else d.get("text", "")
    return str(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="contradiction,niah,nq,hotpotqa")
    ap.add_argument("--sample", type=int, default=25, help="examples to tokenize per rung")
    ap.add_argument("--root", default=RUNG_ROOT)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    print(f"{'task':14s} {'rung':>8s} {'ex':>5s} {'docs/ex':>8s} {'med_doc_tok':>12s} "
          f"{'tok/doc':>8s} {'label/actual':>13s}")
    for task in args.tasks.split(","):
        tdir = os.path.join(args.root, task)
        if not os.path.isdir(tdir):
            print(f"{task:14s} -- no rung dir at {tdir}")
            continue
        rungs = sorted(
            (int(f[len("rung_"):-len(".jsonl")]), os.path.join(tdir, f))
            for f in os.listdir(tdir)
            if f.startswith("rung_") and f.endswith(".jsonl")
        )
        for label, path in rungs:
            totals, ndocs = [], []
            with open(path) as fh:
                for i, line in enumerate(fh):
                    if i >= args.sample:
                        break
                    r = json.loads(line)
                    docs = r.get("documents", [])
                    ndocs.append(len(docs))
                    joined = "\n".join(doc_text(d) for d in docs)
                    totals.append(len(tok(joined, add_special_tokens=False)["input_ids"]))
            if not totals:
                continue
            med = statistics.median(totals)
            nd = statistics.median(ndocs)
            print(f"{task:14s} {label:8d} {len(totals):5d} {nd:8.0f} {med:12.0f} "
                  f"{med / max(nd, 1):8.1f} {label / max(med, 1):13.2f}")


if __name__ == "__main__":
    main()
