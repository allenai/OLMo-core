"""Prepare sentence-pair JSONL for the contradiction atomic-op classifier.

Reads a contradiction task JSONL (format produced by generate_pubmed_contradiction_data.py)
where each line has `documents` (list of sentences) and `gold_doc_indices` (1-indexed
pairs of contradictory sentences). Emits one JSONL line per (sentence_a, sentence_b)
pair with a binary `label`:
    label=1 — pair is in gold_doc_indices (contradictory)
    label=0 — random non-gold pair from the same example

Usage:
    python scripts/data/prepare_contradiction_pair_classifier.py \
        --input data/contradiction_train_pubmed_both_n20_k3.jsonl \
        --output data/contradiction_pair_train.jsonl \
        --neg-ratio 3
"""

import argparse
import random

from corpus_reasoning.lib.io import load_jsonl, save_jsonl


def _norm_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def build_pairs(example: dict, ex_id: int, neg_ratio: float, rng: random.Random) -> list[dict]:
    docs = example["documents"]
    n = len(docs)
    # gold_doc_indices is 1-indexed in the source data
    pos_pairs = {
        _norm_pair(a - 1, b - 1)
        for a, b in example["gold_doc_indices"]
        if 1 <= a <= n and 1 <= b <= n and a != b
    }
    if not pos_pairs:
        return []

    out: list[dict] = []
    for i, j in pos_pairs:
        out.append({
            "text_a": docs[i]["text"],
            "text_b": docs[j]["text"],
            "label": 1,
            "_meta": {"source_example_id": ex_id, "pair": [i, j]},
        })

    n_neg = int(round(neg_ratio * len(pos_pairs)))
    # All candidate negatives = unordered pairs (i<j) not in pos_pairs
    max_attempts = n_neg * 50 + 100
    neg: set[tuple[int, int]] = set()
    attempts = 0
    while len(neg) < n_neg and attempts < max_attempts:
        i = rng.randrange(n)
        j = rng.randrange(n)
        attempts += 1
        if i == j:
            continue
        p = _norm_pair(i, j)
        if p in pos_pairs or p in neg:
            continue
        neg.add(p)

    for i, j in neg:
        out.append({
            "text_a": docs[i]["text"],
            "text_b": docs[j]["text"],
            "label": 0,
            "_meta": {"source_example_id": ex_id, "pair": [i, j]},
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Contradiction task JSONL")
    p.add_argument("--output", required=True, help="Pair JSONL output path")
    p.add_argument("--neg-ratio", type=float, default=3.0,
                   help="Negative pairs per positive (default 3)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    examples = load_jsonl(args.input)

    pairs: list[dict] = []
    for ex_id, ex in enumerate(examples):
        pairs.extend(build_pairs(ex, ex_id, args.neg_ratio, rng))

    rng.shuffle(pairs)
    save_jsonl(args.output, pairs)

    n_pos = sum(1 for x in pairs if x["label"] == 1)
    n_neg = len(pairs) - n_pos
    print(f"Wrote {len(pairs)} pairs to {args.output}  (pos={n_pos}, neg={n_neg})")
    print(f"From {len(examples)} source examples in {args.input}")


if __name__ == "__main__":
    main()
