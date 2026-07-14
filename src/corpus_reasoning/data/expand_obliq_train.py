"""Expand permutation count of existing OBLIQ train JSONL(s) by re-shuffling.

Takes one or more existing OBLIQ train files (each example carries a fixed
gold + BM25-distractor set in some shuffled order) and emits a single merged
file with --target-perms examples per (source, query) pair. New permutations
are produced by re-shuffling the document list of an existing example (cycled
across the available content variants), which is essentially free — no BM25,
no HF download, no Lucene indexing.

The existing content variants from `generate_obliq_data.py --num-train-perms N`
already differ in *which* BM25 rank slice supplied the distractors; this script
adds *positional* variation on top so each query contributes more training
signal without re-mining negatives.

Usage:
    python scripts/data/expand_obliq_train.py \\
        --input data/obliq_writing_train_k30_p3_153.jsonl \\
        --input data/obliq_congress_train_k30_p3_75.jsonl \\
        --target-perms 10 \\
        --output data/obliq_writing_congress_train_k30_p10_760.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl


def _reshuffle(example, rng):
    """Return a deep copy of `example` with documents reshuffled.

    Remaps `gold_doc_indices` and `hard_neg_indices` to the new positions.
    Handles both flat-list and per-query nested gold lists.
    """
    docs = example["documents"]
    n = len(docs)
    perm = list(range(n))
    rng.shuffle(perm)
    rev = {old: new for new, old in enumerate(perm)}

    new = dict(example)
    new["documents"] = [docs[i] for i in perm]

    gold = example["gold_doc_indices"]
    if gold and isinstance(gold[0], list):
        new["gold_doc_indices"] = [sorted(rev[g] for g in gids) for gids in gold]
    else:
        new["gold_doc_indices"] = sorted(rev[g] for g in gold)

    new["hard_neg_indices"] = sorted(rev[h] for h in example.get("hard_neg_indices", []))
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="Input train JSONL; repeat to merge multiple subsets.")
    ap.add_argument("--target-perms", type=int, required=True,
                    help="Number of examples per (source, query) in the output.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle-mix", action="store_true",
                    help="Don't cross-shuffle examples across queries in output.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Group by (source, query) so we keep subsets distinguishable even after merge.
    groups = defaultdict(list)
    for path in args.input:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                key = (ex.get("source"), ex["queries"][0])
                groups[key].append(ex)

    out = []
    truncated = 0
    expanded_groups = 0
    for key, exs in groups.items():
        if len(exs) >= args.target_perms:
            out.extend(exs[: args.target_perms])
            truncated += len(exs) - args.target_perms
            continue
        cur = list(exs)
        i = 0
        while len(cur) < args.target_perms:
            base = exs[i % len(exs)]
            cur.append(_reshuffle(base, rng))
            i += 1
        out.extend(cur)
        expanded_groups += 1

    if not args.no_shuffle_mix:
        rng.shuffle(out)

    save_jsonl(args.output, out)
    by_src = defaultdict(int)
    for ex in out:
        by_src[ex.get("source")] += 1

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    n_queries = len(groups)
    print(f"Wrote {len(out)} examples to {args.output}  ({size_mb:.1f} MB)")
    print(f"  Queries: {n_queries}  target_perms: {args.target_perms}  "
          f"expanded: {expanded_groups}  truncated: {truncated}")
    for s, n in sorted(by_src.items()):
        print(f"  source={s}: {n} examples")


if __name__ == "__main__":
    main()
