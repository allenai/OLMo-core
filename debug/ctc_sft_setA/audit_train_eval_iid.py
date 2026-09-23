"""
Is the setA training data IID with what the CTC suite actually grades — at every context length?

This is the check that has cost this project the most: a train/eval mismatch does not error, it
scores. Realistic-mode contradiction graded on the ``both`` ladder read 0.559 when the true number
was 0.946; a train n-max of 697 against an eval of 1,423 documents read as a long-context collapse.
Both were data bugs wearing a capability result's clothes.

So this audits against **olmo-eval's VENDORED ctc**, not the public package the data was built from.
The vendored copy is spec-only -- ``spec.py`` per task plus ``format/`` and ``data/ladders.py`` --
and it is the code that will actually grade the run. Point ``--vendor`` at
``olmo-eval/src/olmo_eval/evals/tasks/ctc_suite/_vendor``.

Per (task, bucket) it checks five things, each a real failure that has happened:

1. **Format contract.** The row's own gold target, parsed and scored by the EVAL's parser and
   scorer, must come back at the primary metric = 1.0. A target the grader cannot parse trains the
   model to emit unparseable answers and reads as a capability failure.
2. **Gold index base.** Bases differ per task (contradiction 1-indexed, outlier/rerank/nq
   0-indexed). Applying the wrong one silently pools the true gold document away.
3. **Rung coverage.** The bucket must be a rung the eval actually has, else nothing scores there.
4. **Document count.** Train ``n_docs`` must sit on the eval's calibrated ``docs_for_rung`` for that
   rung; a train support that does not reach the eval's n is the "long-context collapse" above.
5. **Prompt renders** under the eval's ``build_prompt`` at the run's ``query_position``.

    python debug/ctc_sft_setA/audit_train_eval_iid.py \
        --root /weka/.../ctc_sft_sets/setA_max20 --vendor <olmo-eval>/..._vendor
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from typing import Dict, List

TASK_TO_SPEC = {"nq": "retrieval", "hotpotqa": "cot_retrieval", "qdmatch_nq": "qdmatch"}
TASKS = ["nq", "hotpotqa", "qdmatch_nq", "outlier", "oolong", "contradiction", "xabsence",
         "reorder", "rerank", "strmatch", "textgroups", "grouping_labeled"]
BUCKETS = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]


def audit_cell(task: str, bucket: str, root: str, sample: int, qpos: str) -> dict:
    from ctc.data import ladders
    from ctc.format import registry

    spec_name = TASK_TO_SPEC.get(task, task)
    rec = {"task": task, "bucket": bucket, "spec": spec_name}
    path = os.path.join(root, "per_task", f"_b{bucket}", task, "train.jsonl")
    if not os.path.exists(path):
        rec["status"] = "absent"
        return rec

    sp = registry.get(spec_name)
    mod = importlib.import_module(f"ctc.tasks.{spec_name}.spec")

    # 3. Does the EVAL have this rung at all?
    rec["eval_has_rung"] = bucket in set(sp.rungs or ())
    # 4. The eval's calibrated document count for this rung.
    try:
        rec["eval_n_docs"] = ladders.docs_for_rung(task, bucket)
    except Exception as e:
        rec["eval_n_docs"] = None
        rec["ladder_error"] = f"{type(e).__name__}: {e}"

    rows: List[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= sample:
                break
            ex = json.loads(line)
            rows.append(ex["ex"] if "ex" in ex and "documents" not in ex else ex)
    if not rows:
        rec["status"] = "empty"
        return rec

    ndocs, scores, bad_gold, render_err, score_err = [], [], 0, None, None
    for ex in rows:
        n = len(ex.get("documents") or [])
        ndocs.append(n)
        # 2. gold index base: every index must land inside the document list under the spec's base.
        base = sp.gold_index_base
        for g in ex.get("gold_doc_indices") or []:
            for idx in (g if isinstance(g, (list, tuple)) else [g]):
                if not (base <= idx < n + base):
                    bad_gold += 1
        # 5 + 1. render, then grade the row's OWN gold through the eval's parser and scorer.
        try:
            sp.build_prompt(ex, query_position=qpos)
        except Exception as e:
            render_err = render_err or f"{type(e).__name__}: {e}"
        try:
            tgt = mod.build_target(ex)
            parsed = sp.parse(tgt, n)
            s = sp.score(parsed, ex.get("gold_doc_indices"))
            scores.append(float(s.get(sp.primary_metric, 0.0)))
        except Exception as e:
            score_err = score_err or f"{type(e).__name__}: {e}"

    rec.update(
        rows=len(rows),
        n_docs_min=min(ndocs), n_docs_med=int(statistics.median(ndocs)), n_docs_max=max(ndocs),
        bad_gold=bad_gold,
        self_score=round(statistics.mean(scores), 4) if scores else None,
        render_error=render_err, score_error=score_err,
        primary_metric=sp.primary_metric, gold_index_base=sp.gold_index_base,
    )

    fails = []
    if render_err:
        fails.append(f"prompt render: {render_err}")
    if score_err:
        fails.append(f"grade: {score_err}")
    if rec["self_score"] is not None and rec["self_score"] < 1.0:
        fails.append(f"gold target self-scores {rec['self_score']} under the EVAL's grader, not 1.0")
    if bad_gold:
        fails.append(f"{bad_gold} gold index/indices out of range at base {sp.gold_index_base}")
    if not rec["eval_has_rung"]:
        fails.append(f"the eval has no {bucket} rung for this task ({sorted(sp.rungs or ())})")
    if rec.get("eval_n_docs") and not (
            rec["n_docs_min"] <= rec["eval_n_docs"] <= rec["n_docs_max"]):
        fails.append(f"train n_docs [{rec['n_docs_min']},{rec['n_docs_max']}] does not cover the "
                     f"eval's calibrated {rec['eval_n_docs']} for {bucket}")
    rec["status"] = "ok" if not fails else "FAIL"
    rec["fails"] = fails
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the set root holding per_task/")
    ap.add_argument("--vendor", required=True,
                    help="olmo-eval .../ctc_suite/_vendor -- the code that will GRADE the run")
    ap.add_argument("--sample", type=int, default=50, help="rows per (task,bucket)")
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sys.path.insert(0, args.vendor)
    import ctc.tasks as T

    T.load_all()
    print(f"grading through the VENDORED spec at {args.vendor}\n", flush=True)

    results = []
    for t in TASKS:
        for b in BUCKETS:
            results.append(audit_cell(t, b, args.root, args.sample, args.query_position))

    present = [r for r in results if r["status"] not in ("absent",)]
    fails = [r for r in present if r["status"] == "FAIL"]

    print(f"{'task':<18}{'bucket':>7}{'rows':>6}{'n_docs (min/med/max)':>24}"
          f"{'eval_n':>8}{'self':>7}  status")
    for r in results:
        if r["status"] == "absent":
            continue
        nd = f"{r['n_docs_min']}/{r['n_docs_med']}/{r['n_docs_max']}"
        print(f"{r['task']:<18}{r['bucket']:>7}{r['rows']:>6}{nd:>24}"
              f"{str(r.get('eval_n_docs') or '-'):>8}{str(r.get('self_score')):>7}  {r['status']}")

    if fails:
        print(f"\n=== {len(fails)} FAILING CELL(S) ===")
        for r in fails:
            for f in r["fails"]:
                print(f"  {r['task']}@{r['bucket']}: {f}")
    print(f"\n{len(present) - len(fails)}/{len(present)} cells IID-clean"
          + ("" if not fails else "  -- DO NOT TRAIN until the above are understood"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"vendor": args.vendor, "cells": results}, f, indent=2)
        print(f"wrote {args.out}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
