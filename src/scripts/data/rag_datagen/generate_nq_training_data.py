"""Generate NQ training data in unified structured format.

Loads `nq_open` (Lee et al. 2019, the open-domain NQ used by DPR / RAG / FiD)
and produces BOTH a train and a validation file in one run. Every document —
gold, BM25 hard negative, and random distractor — is drawn from pyserini's
`wikipedia-dpr-100w` index, so all docs share one surface format (no NQ HF
contexts, no tokenization mismatch).

For each (question, answers) pair the script:
  1. BM25-searches `wikipedia-dpr-100w` for a passage that contains an answer
     string — that becomes the gold doc. Examples with no such passage are
     skipped (the script overshoots to hit the requested count).
  2. Takes the top BM25 hits that do NOT contain the answer as hard negatives.
  3. Fills the rest of the doc window with random passages from the corpus.
  4. Shuffles the final list and records `gold_doc_indices` +
     `hard_neg_indices` so provenance is recoverable.

HF dataset: https://huggingface.co/datasets/google-research-datasets/nq_open
  train split:       87,925 questions
  validation split:  3,610 questions (canonical NQ-open test set)

Usage:
    python scripts/data/generate_nq_training_data.py --num-train 10000 --num-eval 500 --num-docs 100 --num-hard-negatives 10
"""

import argparse
import random

from datasets import load_dataset
from tqdm import tqdm

from lib.io import save_jsonl, print_dataset_stats


def _assemble_example(question, answers, gold_doc, hard_negs, pool_distractors, rng):
    """Tag, shuffle, and emit a unified-format NQ example."""
    tagged = (
        [(gold_doc, "gold")]
        + [(d, "hard") for d in hard_negs]
        + [(d, "pool") for d in pool_distractors]
    )
    rng.shuffle(tagged)
    documents = [d for d, _ in tagged]
    gold_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "gold"]
    hard_neg_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]
    return {
        "documents": documents,
        "queries": [question],
        "answers": [answers[0]],
        "gold_doc_indices": gold_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": "nq",
    }


def _generate_split(ds, num_wanted, args, rng, bm25_searcher, split_name):
    """Build up to `num_wanted` examples from a nq_open split.

    Uses batched BM25 search (via Lucene's Java thread pool) to amortize the
    search cost over many queries at once. Each batch:
      1. `batch_find_gold_and_hard_negs` → one fused BM25 call per query,
         yielding gold passage + hard negs together.
      2. Per example: sample random-pool passages from the corpus, then
         shuffle gold + hard + random into the final doc list.

    Shuffles example order before processing so batches are drawn uniformly.
    Iterates until we hit the target count or run out of questions.
    """
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    n_hard = min(args.num_hard_negatives, args.num_docs - 1)
    n_random = args.num_docs - 1 - n_hard

    hn_label = (
        f", {args.num_hard_negatives} BM25 hard neg"
        if args.num_hard_negatives > 0 else ""
    )
    desc = f"NQ {split_name} k{args.num_docs}{hn_label}"

    examples: list[dict] = []
    skipped = 0
    pbar = tqdm(total=num_wanted, desc=desc)

    for batch_start in range(0, len(indices), args.batch_size):
        if len(examples) >= num_wanted:
            break
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        batch = [ds[i] for i in batch_indices]
        questions = [ex["question"] for ex in batch]
        answers_list = [ex["answer"] for ex in batch]

        golds, hard_neg_lists = bm25_searcher.batch_find_gold_and_hard_negs(
            questions, answers_list,
            num_hard=n_hard,
            max_pair_overlap=args.max_hard_neg_pair_overlap,
            threads=args.bm25_threads,
        )

        for ex, (gold, gold_docid), hard_negs in zip(batch, golds, hard_neg_lists):
            if len(examples) >= num_wanted:
                break
            if gold is None:
                skipped += 1
                continue
            pool_distractors = bm25_searcher.sample_random_passages(
                n_random,
                exclude_docids={gold_docid} | {d["text"] for d in hard_negs},
                exclude_answers=ex["answer"],
                rng=rng,
            )
            examples.append(_assemble_example(
                ex["question"], ex["answer"], gold, hard_negs, pool_distractors, rng,
            ))
            pbar.update(1)
    pbar.close()

    if skipped > 0:
        print(f"  Skipped {skipped} examples (gold passage not found in BM25 results)")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate NQ train + validation data")
    parser.add_argument("--num-train", type=int, default=10000,
                        help="Number of training examples to generate")
    parser.add_argument("--num-eval", type=int, default=500,
                        help="Number of validation examples to generate")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="Total documents per example (1 gold + N-1 distractors)")
    parser.add_argument("--num-hard-negatives", type=int, default=0,
                        help="Number of BM25 hard negatives per example "
                             "(0 = all random from the corpus)")
    parser.add_argument("--max-hard-neg-pair-overlap", type=float, default=0.5,
                        help="Reject a hard-neg candidate whose word-level "
                             "Jaccard with an already-selected hard neg "
                             "exceeds this fraction. Prevents near-duplicate "
                             "chunks. Set to 1.0 to disable.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of queries to send to Lucene per "
                             "batch_search call")
    parser.add_argument("--bm25-threads", type=int, default=8,
                        help="Java thread count Lucene uses inside each "
                             "batch_search call")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-context", action="store_true",
                        help="Generate closed-book examples (no documents)")
    args = parser.parse_args()

    if args.no_context:
        args.num_docs = 0

    rng = random.Random(args.seed)

    from lib.bm25 import BM25Searcher
    bm25_searcher = BM25Searcher()

    hn_tag = f"_hn{args.num_hard_negatives}" if args.num_hard_negatives > 0 else ""

    for split_name, num_wanted in [("train", args.num_train), ("validation", args.num_eval)]:
        if num_wanted <= 0:
            continue

        print(f"\nLoading nq_open ({split_name})...")
        ds = load_dataset("nq_open", split=split_name)
        print(f"  Loaded {len(ds)} questions")

        examples = _generate_split(
            ds, num_wanted, args, rng, bm25_searcher, split_name,
        )

        if args.no_context:
            path = f"{args.output_dir}/nq_{split_name}_nocontext_{len(examples)}.jsonl"
        else:
            path = f"{args.output_dir}/nq_{split_name}_k{args.num_docs}{hn_tag}_{len(examples)}.jsonl"

        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)


if __name__ == "__main__":
    main()
