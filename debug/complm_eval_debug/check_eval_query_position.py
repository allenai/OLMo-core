"""Prove ``--query-position`` reaches the v2 ladder AND the OOD probes in ``eval_lc_native.py``.

The flag is only useful if it changes the bytes the model actually sees, on *both* paths. The OOD
probes in particular used to bypass the shared ``_load`` helper, so a flag added to ``_load`` alone
would have silently missed them. This calls ``load_unified_examples`` exactly as the driver does and
diffs the rendered prompt under ``both`` vs ``after``.

Expected: for a retrieval-style task the ask moves from before+after the corpus to after only, so
the ``after`` prompt is SHORTER, its tail is unchanged, and its head is the first document rather
than the question.

    PYTHONPATH=src python debug/complm_eval_debug/check_eval_query_position.py
"""

import sys

from corpus_reasoning.eval.evaluate import load_unified_examples

DATA = "/net/horton/data/prasann/corpus-reasoning/data"

CASES = [
    # (label, path, task) -- a v2-ladder-style retrieval file and a contradiction file.
    ("OOD probe (scifact, retrieval)", f"{DATA}/beir_scifact_test_k20_300_spliteval.jsonl", "retrieval"),
    ("ladder (contradiction n100)", f"{DATA}/contradiction_eval_pubmed_both_n100_k3.jsonl", "contradiction"),
]


def main():
    failures = []
    for label, path, task in CASES:
        both = load_unified_examples(path, 1, task=task, query_position="both", use_alpaca=True)
        after = load_unified_examples(path, 1, task=task, query_position="after", use_alpaca=True)
        pb, pa = both[0]["prompt"], after[0]["prompt"]
        print(f"=== {label}")
        print(f"    chars both={len(pb):,}  after={len(pa):,}  delta={len(pb)-len(pa):,}")
        print(f"    changed: {pb != pa}   tail preserved: {pb[-200:] == pa[-200:]}")
        print(f"    both  head: {pb[:110].replace(chr(10), ' | ')}")
        print(f"    after head: {pa[:110].replace(chr(10), ' | ')}")
        if pb == pa:
            failures.append(f"{label}: query_position had NO effect")
        elif len(pa) >= len(pb):
            failures.append(f"{label}: 'after' should be shorter than 'both'")
        elif pb[-200:] != pa[-200:]:
            failures.append(f"{label}: tail changed -- 'after' should only drop the leading ask")
    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("OK -- query_position changes the rendered prompt on both the ladder and OOD paths")


if __name__ == "__main__":
    main()
