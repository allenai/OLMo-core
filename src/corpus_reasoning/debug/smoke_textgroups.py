"""Smoke test for the textgroups task: generate a few examples per feature,
verify exact-K + exact counts, and render train prompts/outputs for each CoT mode.
"""
import json
import random
import sys
from itertools import combinations
 # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from corpus_reasoning.data import generate_textgroups_data as G
from corpus_reasoning.lib.data_format import build_prompt


class Args:
    num_docs = 24
    num_groups = 2
    group_size = 3
    target = 30
    tolerance = 0
    cmin = 4
    cmax = 18
    feature = "mixed"


def check(ex, args):
    meta = ex["_meta"]
    counts = meta["counts"]
    # recompute feature counts straight from the rendered text
    for i, d in enumerate(ex["documents"]):
        c = G._count_feature(d["text"], meta["feature"], meta["word"])
        assert c == counts[i], f"count mismatch doc {i}: text={c} meta={counts[i]}"
    # exactly K triples sum to target (1-indexed gold)
    T, W = meta["target"], meta["tolerance"]
    hits = [list(c) for c in combinations(range(len(counts)), 3)
            if abs(sum(counts[j] for j in c) - T) <= W]
    gold0 = sorted(sorted(j - 1 for j in g) for g in ex["gold_doc_indices"])
    assert sorted(hits) == gold0, f"K mismatch: brute={len(hits)} gold={len(gold0)}"
    # every gold group sums to T
    for g in ex["gold_doc_indices"]:
        assert abs(sum(counts[j - 1] for j in g) - T) <= W


rng = random.Random(7)
for feat in G.FEATURES:
    args = Args()
    args.feature = feat
    for _ in range(8):
        ex = G.build_example(args, rng)
        check(ex, args)
    print(f"[ok] feature={feat}: 8 examples, exact-K + exact-counts verified")

# render one example end-to-end through every cot mode
args = Args(); args.feature = "nouns"
ex = G.build_example(args, rng)
print("\n=== sample passages (first 3) ===")
for i, d in enumerate(ex["documents"][:3], 1):
    print(f"[{i}] ({ex['_meta']['counts'][i-1]} nouns) {d['text']}")
print("\nquery:", ex["queries"][0])
print("gold:", ex["gold_doc_indices"])
for mode in ("label", "template", "scan"):
    prompt, out = build_prompt(ex, task="textgroups", cot_mode=mode,
                               query_position="after", use_alpaca=True)
    print(f"\n=== cot_mode={mode} OUTPUT ===\n{out}")
print("\n=== prompt tail ===\n", prompt[-600:])
print("\nALL SMOKE CHECKS PASSED")
