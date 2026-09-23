"""
Per-(task, context-bucket) TOKEN counts for the setA SFT build.

The build is token-BALANCED by construction (a fixed budget per bucket, so example count falls as
the bucket grows), but the realized token counts are what actually matter and they are not the
bucket label: rung LABELS are not token counts (contradiction runs ~1.5x under its label, niah
~2.9x). This measures them with the tokenizer the run will actually use.

Renders each row through the SAME entry point the converters use --
``corpus_reasoning_prompts.build_prompt(..., use_alpaca=True, query_position=...)`` -- so the counts
are the prompt the model is trained on, not the raw JSON.

    python debug/ctc_sft_setA/count_setA_tokens.py \
        --root /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_max20 \
        --tokenizer Qwen/Qwen3.5-4B --out debug/ctc_sft_setA/setA_token_counts.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np

from olmo_core.data.corpus_reasoning_prompts import build_prompt

#: Ladder name -> the shared *spec* name that ``build_prompt`` renders. Passing a ladder name where
#: a spec is expected raises on every row, which surfaces as a 100% error rate rather than a crash.
TASK_TO_SPEC = {
    "nq": "retrieval",
    "hotpotqa": "cot_retrieval",
    "qdmatch_nq": "qdmatch",
}

TASKS = [
    "nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence",
    "absence", "reorder", "rerank", "strmatch", "textgroups", "grouping",
]
BUCKETS = ["2k", "4k", "8k", "16k", "32k"]


def _count_one(args_tuple) -> dict:
    task, bucket, root, tokenizer_id, query_position, cot_mode, limit = args_tuple
    from transformers import AutoTokenizer

    path = os.path.join(root, "per_task", f"_b{bucket}", task, "train.jsonl")
    rec = {"task": task, "bucket": bucket, "path": path}
    if not os.path.exists(path):
        rec["status"] = "MISSING"
        return rec

    spec = TASK_TO_SPEC.get(task, task)
    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    texts: List[str] = []
    n_rows = n_err = 0
    err_example = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            if limit and n_rows > limit:
                n_rows -= 1
                break
            ex = json.loads(line)
            if "ex" in ex and "documents" not in ex:
                ex = ex["ex"]
            try:
                prompt, output = build_prompt(
                    ex, task=spec, query_position=query_position,
                    use_alpaca=True, cot_mode=cot_mode,
                )
            except Exception as e:  # a bad spec name fails on EVERY row -- report, do not crash
                n_err += 1
                if err_example is None:
                    err_example = f"{type(e).__name__}: {e}"
                continue
            texts.append(prompt + output)

    if not texts:
        rec.update(status="NO_ROWS", n_rows=n_rows, n_err=n_err, error=err_example)
        return rec

    lens: List[int] = []
    for i in range(0, len(texts), 256):
        lens.extend(len(x) for x in tok(texts[i : i + 256], add_special_tokens=False)["input_ids"])
    a = np.asarray(lens, dtype=np.int64)
    rec.update(
        status="ok",
        n_rows=n_rows,
        n_measured=int(a.size),
        n_err=n_err,
        error=err_example,
        total_tokens=int(a.sum()),
        mean=float(a.mean()),
        p50=int(np.percentile(a, 50)),
        p90=int(np.percentile(a, 90)),
        max=int(a.max()),
        over_40960=int((a > 40960).sum()),
    )
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    p.add_argument("--query-position", default="both", choices=("before", "after", "both"))
    p.add_argument("--cot-mode", default="none")
    p.add_argument("--limit", type=int, default=0, help="rows per (task,bucket); 0 = all")
    p.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    jobs = [
        (t, b, args.root, args.tokenizer, args.query_position, args.cot_mode, args.limit)
        for t in TASKS for b in BUCKETS
    ]
    print(f"=== {len(jobs)} (task,bucket) cells | workers={args.workers} | "
          f"tokenizer={args.tokenizer} | query_position={args.query_position} ===", flush=True)

    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(_count_one, jobs), 1):
            results.append(rec)
            if i <= 2 or i % 5 == 0 or i == len(jobs):
                el = time.time() - t0
                eta = el / i * (len(jobs) - i)
                print(f"  [{i}/{len(jobs)}] {rec['task']}@{rec['bucket']} {rec['status']} "
                      f"tok={rec.get('total_tokens', 0):,} p50={rec.get('p50', 0)} "
                      f"| {el:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"root": args.root, "tokenizer": args.tokenizer,
                   "query_position": args.query_position, "cells": results}, f, indent=2)
    print(f"wrote {args.out}", flush=True)

    bad = [r for r in results if r["status"] != "ok" or r.get("n_err")]
    if bad:
        print("\n=== CELLS NEEDING ATTENTION ===", flush=True)
        for r in bad:
            print(f"  {r['task']}@{r['bucket']}: {r['status']} n_err={r.get('n_err')} "
                  f"{r.get('error') or ''}", flush=True)

    print("\n=== TOTAL TOKENS BY TASK x BUCKET ===", flush=True)
    hdr = f"{'task':<18}" + "".join(f"{b:>12}" for b in BUCKETS) + f"{'TOTAL':>14}"
    print(hdr, flush=True)
    grand = 0
    for t in TASKS:
        row = {r["bucket"]: r for r in results if r["task"] == t}
        tot = sum(row.get(b, {}).get("total_tokens", 0) for b in BUCKETS)
        grand += tot
        print(f"{t:<18}" + "".join(f"{row.get(b, {}).get('total_tokens', 0):>12,}" for b in BUCKETS)
              + f"{tot:>14,}", flush=True)
    print(f"{'GRAND TOTAL':<18}" + " " * (12 * len(BUCKETS)) + f"{grand:>14,}", flush=True)


if __name__ == "__main__":
    main()
