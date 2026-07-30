"""Generate matching-n-gram data.

Each example is a list of N short n-grams (n words each, n=2 by default)
sampled from Wikipedia (wiki100w). Among the N snippets, m disjoint pairs are
character-identical duplicates; the remaining N - 2m snippets are unique
fillers. The model is asked to identify the matching pairs.

Mirrors the contradiction task's structure (`gold_doc_indices: [[a, b], ...]`)
so the same prompt builder, parser and metrics apply.

Usage:
    python scripts/data/generate_matching_ngram_data.py \\
        --num-docs 100 --n 2 --num-matches 3 \\
        --num-train 0 --num-eval 50 \\
        --pool-passages 5000
"""

import argparse
import random
import re

from tqdm import tqdm

from corpus_reasoning.lib.bm25 import BM25Searcher
from corpus_reasoning.lib.wiki100w_sample import extract_title_and_body
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


# Token regex: words of length >= 2, alphabetic only. Avoids splits introducing
# punctuation glyphs that would make matching depend on tokenizer quirks.
_WORD = re.compile(r"[A-Za-z]{2,}")


def extract_ngrams(text: str, n: int) -> list[str]:
    """Lower-cased word n-grams from a passage body."""
    words = [w.lower() for w in _WORD.findall(text)]
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def build_ngram_pool(searcher: BM25Searcher, n: int, pool_passages: int,
                     seed: int) -> list[str]:
    """Sample passages, extract all n-grams, dedupe to a unique pool."""
    print(f"Pre-fetching {pool_passages} random wiki100w passages...")
    docs = searcher.prefetch_random_pool(pool_passages, threads=16, seed=seed)
    seen = set()
    pool = []
    for d in tqdm(docs, desc="Extracting n-grams"):
        _, body = extract_title_and_body(d["text"])
        for g in extract_ngrams(body, n):
            if g in seen:
                continue
            seen.add(g)
            pool.append(g)
    print(f"  Unique n-grams in pool: {len(pool):,}")
    return pool


def build_example(pool: list[str], num_docs: int, num_matches: int,
                  cursor: int) -> tuple[dict, int]:
    """Assemble one example. `cursor` advances through the (already shuffled)
    pool so successive examples consume disjoint slices — guarantees no
    accidental cross-example reuse and no within-example duplicates beyond the
    intended gold pairs.
    """
    needed_uniques = (num_docs - 2 * num_matches) + num_matches
    if cursor + needed_uniques > len(pool):
        raise RuntimeError(
            f"Pool exhausted at cursor={cursor}; need {needed_uniques} more "
            f"unique n-grams. Increase --pool-passages."
        )
    block = pool[cursor:cursor + needed_uniques]
    cursor += needed_uniques

    gold_grams = block[:num_matches]
    fillers = block[num_matches:]
    snippets = list(fillers) + [g for g in gold_grams for _ in range(2)]

    rng = random.Random(cursor)  # deterministic per-example shuffle
    order = list(range(len(snippets)))
    rng.shuffle(order)
    shuffled = [snippets[i] for i in order]

    pairs = []
    for g in gold_grams:
        positions = [i + 1 for i, s in enumerate(shuffled) if s == g]
        assert len(positions) == 2, (
            f"Expected exactly 2 occurrences of {g!r}, got {len(positions)}"
        )
        pairs.append(sorted(positions))
    pairs.sort()

    ex = {
        "documents": [{"text": s} for s in shuffled],
        "queries": [],
        "answers": [],
        "gold_doc_indices": pairs,
        "source": "matching_ngram_wiki100w",
    }
    return ex, cursor


def main():
    ap = argparse.ArgumentParser(
        description="Generate matching-n-gram data from wiki100w"
    )
    ap.add_argument("--num-docs", type=int, default=100,
                    help="Total snippets per example (N).")
    ap.add_argument("--n", type=int, default=2,
                    help="Tokens per snippet (n-gram length).")
    ap.add_argument("--num-matches", type=int, default=3,
                    help="Number of matching pairs per example (m).")
    ap.add_argument("--num-train", type=int, default=0)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--pool-passages", type=int, default=5000,
                    help="Random wiki passages to pre-fetch for the n-gram pool.")
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert args.num_matches * 2 <= args.num_docs, \
        "num_docs must hold all matching-pair slots"

    searcher = BM25Searcher()
    pool = build_ngram_pool(searcher, args.n, args.pool_passages, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    cursor = 0
    splits = []
    for split_label, count in [("train", args.num_train), ("eval", args.num_eval)]:
        if count <= 0:
            continue
        examples = []
        for _ in tqdm(range(count), desc=f"Building {split_label} N{args.num_docs} m{args.num_matches}"):
            ex, cursor = build_example(pool, args.num_docs, args.num_matches, cursor)
            examples.append(ex)
        splits.append((split_label, examples))

    tag = f"wiki100w_n{args.n}_N{args.num_docs}_m{args.num_matches}"
    for split_label, examples in splits:
        path = f"{args.output_dir}/matching_ngram_{split_label}_{tag}.jsonl"
        save_jsonl(path, examples)
        print_dataset_stats(examples, split_label.capitalize(), path)


if __name__ == "__main__":
    main()
