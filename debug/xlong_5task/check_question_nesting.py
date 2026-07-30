"""Verify the nested-ladder property: same questions at every rung, only distractors change.

The v2 ladder design is that a task's rungs share ONE question set (same queries, same gold
documents, same answers) and differ only in how many distractor documents are packed around them --
so a metric change across rungs is attributable to length alone, not to a different question mix.

This checks that property holds ACROSS the join of the original 2k-32k rungs and the newly built
64k/128k/256k rungs, which were produced by a separate script (`build_xlong_rungs.py`) at a
different `--count` and with a different tokenizer calibration.

Per-example identity is the multiset of GOLD document texts (plus the query when the task has a
real one) -- the one thing that must survive expansion. Index fields are deliberately NOT used:
expansion reshuffles documents and remaps every index, which is expected.
"""

import glob
import json
import os

E5 = os.environ.get("EVAL500_ROOT", "/data/prasann/xlong5/eval")

# task -> (subdir, ordered rung patterns). Globs so calibrated doc counts can drift.
TASKS = {
    "contradiction": ("contra", [
        ("2k", "contradiction_eval_pubmed_both_n100_k3.jsonl"),
        ("32k", "contradiction_eval_pubmed_both_n765_k3.jsonl"),
        ("64k", "contradiction_eval_pubmed_both_n*_k3_xlong_64k.jsonl"),
        ("256k", "contradiction_eval_pubmed_both_n*_k3_xlong_256k.jsonl"),
    ], True),
    "nq": ("nq", [
        ("3k", "nq_validation_k20_600.jsonl"),
        ("32k", "nq_validation_k200_600.jsonl"),
        ("64k", "nq_validation_k*_xlong_64k.jsonl"),
        ("256k", "nq_validation_k*_xlong_256k.jsonl"),
    ], False),
    "outlier": ("outlier", [
        ("3k", "outlier_wiki100w_n22_k3_eval_600.jsonl"),
        ("32k", "outlier_wiki100w_n220_k3_eval_600.jsonl"),
        ("64k", "outlier_wiki100w_n*_k3_eval_xlong_64k.jsonl"),
        ("256k", "outlier_wiki100w_n*_k3_eval_xlong_256k.jsonl"),
    ], True),
    "rerank": ("rerank", [
        ("3k", "msmarco_trainhn_eval_k20_500.jsonl"),
        ("16k", "msmarco_trainhn_eval_k100_500.jsonl"),
        ("64k", "msmarco_trainhn_eval_k*_xlong_64k.jsonl"),
        ("256k", "msmarco_trainhn_eval_k*_xlong_256k.jsonl"),
    ], False),
}
GOLD_BASE = {"contradiction": 1, "nq": 0, "outlier": 0, "rerank": 0}


def identity(ex: dict, task: str, gold_only: bool) -> str:
    base = GOLD_BASE[task]
    golds = ex.get("gold_doc_indices") or []
    flat = []
    for g in golds:
        flat.extend(g if isinstance(g, (list, tuple)) else [g])
    docs = ex["documents"]
    texts = sorted(docs[i - base]["text"][:120] for i in flat if 0 <= i - base < len(docs))
    q = "" if gold_only else (ex.get("queries") or [""])[0]
    return q + "||" + "|".join(texts)


def main() -> None:
    for task, (sub, rungs, gold_only) in TASKS.items():
        print(f"[{task}]")
        sets = []
        for lab, pat in rungs:
            hits = sorted(glob.glob(os.path.join(E5, sub, pat)))
            if not hits:
                print(f"   {lab:>5}  MISSING ({pat})")
                continue
            rows = [json.loads(x) for x in open(hits[0]) if x.strip()]
            ids = [identity(r, task, gold_only) for r in rows]
            sets.append((lab, ids, os.path.basename(hits[0])))
        if len(sets) < 2:
            print("   not enough rungs to compare\n")
            continue
        ref_lab, ref_ids, ref_name = sets[0]
        ref = set(ref_ids)
        for lab, ids, name in sets:
            s = set(ids)
            inter = len(ref & s)
            # A shorter rung file may hold MORE examples (600 vs 500); the nested property is that
            # the smaller set is a SUBSET of the larger, not that they are equal.
            sub_of_ref = inter == len(s)
            ref_sub = inter == len(ref)
            verdict = "IDENTICAL" if (sub_of_ref and ref_sub) else (
                "SUBSET of " + ref_lab if sub_of_ref else (
                    "SUPERSET of " + ref_lab if ref_sub else "DIVERGENT"))
            print(f"   {lab:>5}  n={len(s):<5} shared_with_{ref_lab}={inter:<5} {verdict:22s} {name}")
        print()


if __name__ == "__main__":
    main()
