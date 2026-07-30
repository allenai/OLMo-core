"""Prepare (query, doc) pair JSONL for the NQ atomic-op relevance classifier.

Reads NQ task JSONL (format produced by generate_nq_training_data.py) where each
line has `queries`, `documents` (with optional `title`), `gold_doc_indices`
(0-indexed) and `hard_neg_indices`. Emits one JSONL line per (query, doc) pair
with a binary `label`:
    label=1 — doc is the gold doc for this query
    label=0 — doc is a negative (hard from BM25 or random)

For training, negatives are subsampled per-query: --neg-hard from hard_neg_indices
and --neg-random from the remaining non-gold pool.

For eval, pass --emit-all to emit ALL (query, doc) pairs in each example. This
lets the evaluator compute ranking metrics (Recall@1, MRR) by grouping on
_meta.query_id.

Usage:
    # Training set (subsampled negatives)
    python scripts/data/prepare_nq_pair_classifier.py \
        --input data/nq_train_k100_hn10_2500.jsonl \
        --output data/nq_pair_train.jsonl --neg-hard 2 --neg-random 2

    # Eval set (all pairs, for ranking metrics)
    python scripts/data/prepare_nq_pair_classifier.py \
        --input data/nq_validation_k100_hn10_500.jsonl \
        --output data/nq_pair_eval.jsonl --emit-all
"""

import argparse
import random

from corpus_reasoning.lib.io import load_jsonl, save_jsonl


def _format_doc(doc: dict) -> str:
    title = doc.get("title")
    text = doc.get("text", "")
    if title:
        return f"{title}\n{text}"
    return text


def build_pairs(
    example: dict,
    ex_id: int,
    neg_hard: int,
    neg_random: int,
    emit_all: bool,
    rng: random.Random,
) -> list[dict]:
    queries = example.get("queries") or []
    if not queries:
        return []
    query = queries[0]
    docs = example["documents"]
    n = len(docs)
    gold = set(example.get("gold_doc_indices") or [])
    if not gold:
        return []
    hard = [i for i in (example.get("hard_neg_indices") or []) if i not in gold and 0 <= i < n]

    if emit_all:
        chosen_idx = list(range(n))
    else:
        hard_pool = list(hard)
        rng.shuffle(hard_pool)
        hard_pick = hard_pool[:neg_hard]
        used = set(gold) | set(hard_pick)
        random_pool = [i for i in range(n) if i not in used]
        rng.shuffle(random_pool)
        random_pick = random_pool[:neg_random]
        chosen_idx = sorted(set(gold) | set(hard_pick) | set(random_pick))

    out: list[dict] = []
    for i in chosen_idx:
        out.append({
            "text_a": query,
            "text_b": _format_doc(docs[i]),
            "label": 1 if i in gold else 0,
            "_meta": {
                "query_id": ex_id,
                "doc_idx": i,
                "is_hard_neg": i in hard and i not in gold,
            },
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--neg-hard", type=int, default=2,
                   help="Number of hard negatives per query (training)")
    p.add_argument("--neg-random", type=int, default=2,
                   help="Number of random negatives per query (training)")
    p.add_argument("--emit-all", action="store_true",
                   help="Emit all (query, doc) pairs per example "
                        "(use for eval to enable ranking metrics)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=None,
                   help="Optional cap on number of source examples")
    args = p.parse_args()

    rng = random.Random(args.seed)
    examples = load_jsonl(args.input)
    if args.max_examples:
        examples = examples[: args.max_examples]

    pairs: list[dict] = []
    for ex_id, ex in enumerate(examples):
        pairs.extend(build_pairs(
            ex, ex_id, args.neg_hard, args.neg_random, args.emit_all, rng,
        ))

    # Keep grouping intact when emit-all (evaluator needs per-query groups);
    # shuffle otherwise for training.
    if not args.emit_all:
        rng.shuffle(pairs)

    save_jsonl(args.output, pairs)
    n_pos = sum(1 for x in pairs if x["label"] == 1)
    n_neg = len(pairs) - n_pos
    print(f"Wrote {len(pairs)} pairs to {args.output}  (pos={n_pos}, neg={n_neg})")
    print(f"From {len(examples)} source queries in {args.input}")


if __name__ == "__main__":
    main()
