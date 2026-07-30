"""Generate narrative-reordering task data from the Wikipedia 100w corpus.

Wikipedia version of generate_reorder_data.py. Source documents are
consecutive 100w chunks from a single Wikipedia article (sampled via
scripts.lib.wiki100w_sample.sample_article_run, which exploits the
property that adjacent Lucene ids in the wikipedia-dpr-100w index belong
to the same article).

Caveat: 100w chunks within a Wikipedia article are less narrative-sequential
than Gutenberg novel paragraphs. Topical continuity is the main signal here,
not story progression — expect this to be a harder task than the Gutenberg
variant.

Unified schema (matches generate_reorder_data.py output):
    {
      "documents":      [{"title": null, "text": "..."}, ...],   # shuffled
      "queries":        [],
      "answers":        ["[17, 3, 42, ...]"],                     # JSON permutation
      "gold_order":     [17, 3, 42, ...],                         # 1-indexed display IDs
      "source_type":    "wiki100w",
      "source_id":      "wiki100w/<article-title>/<start-lid>",
      "n_chunks":       N,
      "chunk_word_lens":[...],
      "chunk_start_lid": <Lucene id of first chunk in run>,
      "split":          "train" | "eval",
    }

Usage:
    python scripts/data/generate_wiki_reorder_data.py \\
        --num-examples 200 --n-chunks 20 --out-dir data/
"""

import argparse
import json
import random
from pathlib import Path

from corpus_reasoning.lib.bm25 import BM25Searcher
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.lib.wiki100w_sample import sample_article_run


def word_count(s: str) -> int:
    return len(s.split())


def make_example(run: list[dict], n: int, rng: random.Random, split: str) -> dict:
    """Build a unified-format reorder example from an in-article chunk run.

    `run` is a list of n chunk dicts (lid, title, body, ...) in original
    article order. We shuffle them and emit the gold permutation.
    """
    title = run[0]["title"]
    start_lid = run[0]["lid"]
    bodies = [d["body"] for d in run]

    order = list(range(n))
    rng.shuffle(order)
    documents = [{"title": None, "text": bodies[order[i]]} for i in range(n)]
    gold_order = [0] * n
    for display_pos, orig_idx in enumerate(order):
        gold_order[orig_idx] = display_pos + 1

    return {
        "documents": documents,
        "queries": [],
        "answers": [json.dumps(gold_order)],
        "gold_order": gold_order,
        "source_type": "wiki100w",
        "source_id": f"wiki100w/{title}/{start_lid}",
        "n_chunks": n,
        "chunk_word_lens": [word_count(d["text"]) for d in documents],
        "chunk_start_lid": start_lid,
        "split": split,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--num-examples", type=int, default=200,
                    help="Total examples (train + eval).")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--n-chunks", type=int, default=20,
                    help="Chunks per example (N). Default 20 is lower than "
                         "the Gutenberg default (50) because Wikipedia "
                         "articles have fewer ~100w chunks.")
    ap.add_argument("--max-attempts-per-example", type=int, default=100,
                    help="Retry budget when an article doesn't have a "
                         "long-enough run.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.n_chunks

    print(f"Loading wikipedia-dpr-100w index...")
    searcher = BM25Searcher()
    print(f"Target: {args.num_examples} examples, N={n} chunks/example")

    examples: list[dict] = []
    n_eval = max(1, int(round(args.num_examples * args.eval_frac)))
    seen_titles: set[str] = set()
    n_attempts = 0

    while len(examples) < args.num_examples:
        n_attempts += 1
        run = sample_article_run(
            searcher, n, rng=rng,
            max_attempts=args.max_attempts_per_example,
            exclude_titles=seen_titles,
        )
        if run is None:
            print(f"  warning: failed to find an unused article with ≥{n} chunks "
                  f"after attempt {n_attempts}; continuing")
            continue
        seen_titles.add(run[0]["title"])
        split = "eval" if len(examples) < n_eval else "train"
        examples.append(make_example(run, n, rng, split))
        if len(examples) % 50 == 0:
            print(f"  built {len(examples)}/{args.num_examples}")

    print(f"\nProduced {len(examples)} examples from {len(seen_titles)} distinct articles.")

    rng.shuffle(examples)
    train = [e for e in examples if e["split"] == "train"]
    evald = [e for e in examples if e["split"] == "eval"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"reorder_wiki100w_n{n}"
    train_path = out_dir / f"{base}_train_{len(train)}.jsonl"
    eval_path = out_dir / f"{base}_eval_{len(evald)}.jsonl"
    save_jsonl(str(train_path), train)
    save_jsonl(str(eval_path), evald)
    print_dataset_stats(
        [{"input": json.dumps(e["documents"])[:1], "output": e["answers"][0]}
         for e in train], "train", str(train_path))
    print_dataset_stats(
        [{"input": json.dumps(e["documents"])[:1], "output": e["answers"][0]}
         for e in evald], "eval", str(eval_path))


if __name__ == "__main__":
    main()
