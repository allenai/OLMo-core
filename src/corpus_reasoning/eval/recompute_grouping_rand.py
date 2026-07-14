"""Recompute Rand Index (pairwise accuracy) for grouping eval results offline.

Joins saved results (`raw_output` strings) with the eval dataset's gold partitions
and computes the unadjusted Rand Index over all C(n,2) pairs — the fraction of
pairs where predicted same/different cluster agrees with gold. Random baseline
is ~0.5 for moderate k.
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations

from corpus_reasoning.lib.io import load_jsonl
from corpus_reasoning.lib.eval_tasks import parse_partition, partition_to_labels


def rand_index(pred_labels, gold_labels):
    n = len(pred_labels)
    agree = 0
    total = 0
    for i, j in combinations(range(n), 2):
        total += 1
        if (pred_labels[i] == pred_labels[j]) == (gold_labels[i] == gold_labels[j]):
            agree += 1
    return agree / total if total else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--eval-data", required=True)
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    examples = load_jsonl(args.eval_data)
    raw = data["results"]
    # Old task-script format: list of per-example records with "raw_output".
    # New unified evaluate.py format: {<label>: {"metrics": ..., "details":
    # [per-example records with "prediction"]}}.
    if isinstance(raw, dict):
        (label, payload), = raw.items()
        records = [{"raw_output": d["prediction"], "level": None}
                   for d in payload["details"]]
    else:
        records = raw
    assert len(records) == len(examples), f"{len(records)} vs {len(examples)}"

    per_level = defaultdict(list)
    all_ri = []
    for rec, ex in zip(records, examples):
        n = len(ex["documents"])
        gold_clusters_1 = [[i + 1 for i in c] for c in ex["gold_doc_indices"]]
        gold_labels = partition_to_labels(gold_clusters_1, n)
        pred = parse_partition(rec["raw_output"], n) or []
        pred_labels = partition_to_labels(pred, n)
        ri = rand_index(pred_labels, gold_labels)
        all_ri.append(ri)
        per_level[ex["level"]].append(ri)

    print(f"\n{args.results}")
    print(f"  overall rand_index: {sum(all_ri)/len(all_ri):.3f}  (eval_size={len(all_ri)})")
    for lvl in sorted(per_level):
        v = per_level[lvl]
        print(f"    L{lvl} (eval_size={len(v)}): {sum(v)/len(v):.3f}")


if __name__ == "__main__":
    main()
