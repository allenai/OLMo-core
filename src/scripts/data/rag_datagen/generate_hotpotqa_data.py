"""Generate HotpotQA training data in unified structured format.

Supports both single-query and multi-query modes:
  - Single-query (default): Each example has 1 question with 2 supporting docs
    among distractors. Model must reason across both to answer.
  - Multi-query (--num-queries > 1): Each example bundles N independent questions
    with all their supporting docs shuffled together, optionally padded with
    distractors.

Supports BM25 hard negatives from pyserini's wikipedia-dpr-100w index:
    --num-hard-negatives N uses N BM25-mined hard negatives per example
    (each ~100 words, matching NQ corpus chunk size), filling remaining
    distractor slots with the original HotpotQA pool.

Output is unified JSONL. Documents are shuffled at generation time; the
`gold_doc_indices` and `hard_neg_indices` fields record where each doc's
provenance (supporting / BM25 hard negative / pool distractor) landed.

Usage:
    # Single-query
    python scripts/data/generate_hotpotqa_data.py --num-examples 1000 --num-docs 20
    python scripts/data/generate_hotpotqa_data.py --num-examples 500 --num-docs 30 --question-type bridge

    # With BM25 hard negatives (100w chunks from wikipedia-dpr-100w)
    python scripts/data/generate_hotpotqa_data.py --num-examples 1000 --num-docs 20 --num-hard-negatives 5

    # Multi-query
    python scripts/data/generate_hotpotqa_data.py --num-queries 10 --num-examples 1000
    python scripts/data/generate_hotpotqa_data.py --num-queries 5 --total-docs 40 --num-examples 500
"""

import argparse
import random
from datasets import load_dataset
from tqdm import tqdm

from lib.io import save_jsonl, print_dataset_stats
from align_hn_doc_lengths import (
    collect_gold_lens,
    distribution_match_truncate,
    lens_by_kind,
    normalize_example,
)


def _align_examples(examples):
    """Normalize surface format + distribution-match hard/rand to gold lengths.

    Gold paragraphs come from HotpotQA's Wikipedia snapshot while BM25 hard
    negatives come from wikipedia-dpr-100w. This function applies the same
    post-processing as `scripts/data/align_hn_doc_lengths.py`: strip the
    leading `"Title"\\n` from hard-neg text, blank title fields, then
    quantile-match hard and random doc lengths to the gold distribution
    (gold docs are left untouched).
    """
    import statistics

    def _log(label, exs):
        gold, hard, rand = lens_by_kind(exs)
        parts = []
        for name, vs in [("gold", gold), ("hard", hard), ("rand", rand)]:
            if vs:
                parts.append(
                    f"{name} mean={statistics.mean(vs):.0f} "
                    f"median={statistics.median(vs):.0f}"
                )
        print(f"  {label:<18} {' | '.join(parts)}")

    print("  Aligning gold/hard/rand length distributions...")
    _log("BEFORE:", examples)
    normalized = [normalize_example(ex) for ex in examples]
    _log("AFTER normalize:", normalized)
    gold_lens = collect_gold_lens(normalized)
    aligned = distribution_match_truncate(normalized, gold_lens)
    _log("AFTER align:", aligned)
    return aligned


def paragraphs_from_context(context):
    """Convert HotpotQA context dict to list of {title, text} dicts."""
    titles = context["title"]
    sentences_list = context["sentences"]
    docs = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(s.strip() for s in sentences)
        docs.append({"title": title, "text": text})
    return docs


def get_supporting_titles(example):
    """Return set of supporting document titles."""
    return set(example["supporting_facts"]["title"])


def build_distractor_pool(ds, indices, rng):
    """Build external distractor pool from non-supporting docs of given examples."""
    pool = []
    for idx in indices:
        ex = ds[idx]
        sup_titles = get_supporting_titles(ex)
        for doc in paragraphs_from_context(ex["context"]):
            if doc["title"] not in sup_titles:
                pool.append(doc)
    rng.shuffle(pool)
    return pool


def _gather_distractors(local_distractors, distractor_pool, num_needed, rng):
    """Select distractors: local first, then external pool, with repetition if needed."""
    distractors = list(local_distractors)
    rng.shuffle(distractors)
    if len(distractors) < num_needed:
        extra = rng.sample(distractor_pool,
                           min(num_needed - len(distractors), len(distractor_pool)))
        distractors.extend(extra)
    while len(distractors) < num_needed:
        distractors.append(rng.choice(distractor_pool))
    return distractors[:num_needed]


def _split_supporting(example):
    """Return (supporting_paragraphs, local_distractors) for one HotpotQA example."""
    all_docs = paragraphs_from_context(example["context"])
    supporting_titles = get_supporting_titles(example)
    sf_titles_ordered = list(dict.fromkeys(example["supporting_facts"]["title"]))
    supporting = []
    for t in sf_titles_ordered:
        for d in all_docs:
            if d["title"] == t and d not in supporting:
                supporting.append(d)
                break
    local_distractors = [d for d in all_docs if d["title"] not in supporting_titles]
    return supporting, local_distractors


def build_single_example(example, distractor_pool, num_docs, rng, use_titles=True,
                         bm25_searcher=None, num_hard_negatives=0,
                         max_pair_overlap=None, pre_mined_hard_negs=None):
    """Build one unified-format example from a single HotpotQA question.

    Docs are shuffled at generation time so the model sees gold at a random
    position. `gold_doc_indices` and `hard_neg_indices` track where each
    provenance class landed after the shuffle.

    When `pre_mined_hard_negs` is supplied, the BM25 call is skipped — this
    lets the caller mine hard negs in batch via `batch_mine_hard_negs` and
    pass them through.
    """
    if num_docs == 0:
        return {
            "documents": [],
            "queries": [example["question"]],
            "answers": [example["answer"]],
            "gold_doc_indices": [],
            "hard_neg_indices": [],
            "source": "hotpotqa",
        }

    supporting, local_distractors = _split_supporting(example)

    num_distractors_needed = max(0, num_docs - len(supporting))
    if pre_mined_hard_negs is not None:
        hard_negs = pre_mined_hard_negs[: min(num_hard_negatives, num_distractors_needed)]
        n_hard = len(hard_negs)
    else:
        n_hard = min(num_hard_negatives, num_distractors_needed) if bm25_searcher else 0
        hard_negs = []
        if n_hard > 0:
            hard_negs = bm25_searcher.mine_distractors(
                example["question"], num_hard=n_hard,
                exclude_answers=[example["answer"]],
                max_pair_overlap=max_pair_overlap,
            )
    n_pool = num_distractors_needed - n_hard

    pool_distractors = []
    if n_pool > 0:
        pool_distractors = _gather_distractors(
            local_distractors, distractor_pool, n_pool, rng
        )

    # Tag each doc with its provenance, then shuffle and record where each kind landed.
    tagged = (
        [(d, "gold") for d in supporting]
        + [(d, "hard") for d in hard_negs]
        + [(d, "pool") for d in pool_distractors]
    )
    rng.shuffle(tagged)
    all_paragraphs = [d for d, _ in tagged]
    gold_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "gold"]
    hard_neg_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]

    return {
        "documents": all_paragraphs,
        "queries": [example["question"]],
        "answers": [example["answer"]],
        "gold_doc_indices": gold_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": "hotpotqa",
    }


def build_multi_example(examples_group, distractor_pool, total_docs, rng,
                        use_titles=True, bm25_searcher=None, num_hard_negatives=0,
                        max_pair_overlap=None):
    """Build one unified-format multi-query example from N HotpotQA questions.

    Collects supporting docs from all N queries (deduplicated by title), then
    shuffles everything together. `gold_doc_indices` (per-query) and
    `hard_neg_indices` track where each provenance class landed.

    Returns dict with: documents, queries, answers, gold_doc_indices,
    hard_neg_indices, source.
    """
    all_supporting = []
    all_local_distractors = []
    queries = []
    answers = []
    seen_titles = set()
    per_query_supporting_titles = []

    for ex in examples_group:
        all_docs = paragraphs_from_context(ex["context"])
        supporting_titles = get_supporting_titles(ex)
        per_query_supporting_titles.append(supporting_titles)

        for d in all_docs:
            if d["title"] in supporting_titles and d["title"] not in seen_titles:
                all_supporting.append(d)
                seen_titles.add(d["title"])

        for d in all_docs:
            if d["title"] not in supporting_titles and d["title"] not in seen_titles:
                all_local_distractors.append(d)
                seen_titles.add(d["title"])

        queries.append(ex["question"])
        answers.append(ex["answer"])

    num_distractors_needed = max(0, total_docs - len(all_supporting)) if total_docs > 0 else 0
    n_hard = min(num_hard_negatives, num_distractors_needed) if bm25_searcher else 0
    n_pool = num_distractors_needed - n_hard

    hard_negs = []
    if n_hard > 0:
        hard_negs = bm25_searcher.mine_distractors(
            " ".join(queries), num_hard=n_hard,
            exclude_answers=answers,
            max_pair_overlap=max_pair_overlap,
        )

    pool_distractors = []
    if n_pool > 0:
        pool_available = [d for d in distractor_pool if d["title"] not in seen_titles]
        pool_distractors = _gather_distractors(
            all_local_distractors, pool_available, n_pool, rng
        )

    tagged = (
        [(d, "gold") for d in all_supporting]
        + [(d, "hard") for d in hard_negs]
        + [(d, "pool") for d in pool_distractors]
    )
    rng.shuffle(tagged)
    all_paragraphs = [d for d, _ in tagged]
    hard_neg_indices = [i for i, (_, tag) in enumerate(tagged) if tag == "hard"]

    gold_doc_indices = []
    for sup_titles in per_query_supporting_titles:
        indices = [i for i, d in enumerate(all_paragraphs) if d["title"] in sup_titles]
        gold_doc_indices.append(indices)

    return {
        "documents": all_paragraphs,
        "queries": queries,
        "answers": answers,
        "gold_doc_indices": gold_doc_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": "hotpotqa",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HotpotQA training data")
    parser.add_argument("--num-examples", type=int, default=1000,
                        help="Number of examples to generate per split")
    parser.add_argument("--num-queries", type=int, default=1,
                        help="Queries per example (1=single-query, >1=multi-query)")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="Total documents in context (single-query mode, must be >= 2)")
    parser.add_argument("--total-docs", type=int, default=0,
                        help="Total documents in context (multi-query mode, 0=supporting only)")
    parser.add_argument("--question-type", type=str, default="bridge",
                        choices=["bridge", "comparison", "all"],
                        help="Filter by question type")
    parser.add_argument("--level", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Filter by difficulty level")
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "both"],
                        help="Which dataset split(s) to use")
    parser.add_argument("--num-eval", type=int, default=500,
                        help="Number of eval examples (only used with --split both)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-context", action="store_true",
                        help="Generate closed-book examples (no documents)")
    parser.add_argument("--num-hard-negatives", type=int, default=0,
                        help="Number of BM25 hard negatives per example from "
                             "wikipedia-dpr-100w (~100 word passages). 0=all from HotpotQA pool")
    parser.add_argument("--max-hard-neg-pair-overlap", type=float, default=0.5,
                        help="Reject a hard-neg candidate whose word-level "
                             "Jaccard with an already-selected hard neg "
                             "exceeds this fraction. Prevents near-duplicate "
                             "chunks from the same article. Set to 1.0 to disable.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of queries to send to Lucene per "
                             "batch_search call")
    parser.add_argument("--bm25-threads", type=int, default=8,
                        help="Java thread count Lucene uses inside each "
                             "batch_search call")
    parser.add_argument("--align", dest="align", action="store_true",
                        default=True,
                        help="Post-process examples so hard/rand doc lengths "
                             "match gold's distribution (default: on)")
    parser.add_argument("--no-align", dest="align", action="store_false",
                        help="Skip the format/length alignment step")
    args = parser.parse_args()

    multi_query = args.num_queries > 1

    if args.no_context:
        args.num_docs = 0
    elif not multi_query and args.num_docs < 2:
        parser.error("Need at least 2 docs for multi-hop QA (or use --no-context)")

    rng = random.Random(args.seed)

    # Initialize BM25 searcher if needed
    bm25_searcher = None
    if args.num_hard_negatives > 0:
        from lib.bm25 import BM25Searcher
        bm25_searcher = BM25Searcher()

    if args.split == "both":
        splits_to_process = [("train", args.num_examples), ("validation", args.num_eval)]
    else:
        splits_to_process = [(args.split, args.num_examples)]

    for split_name, num_wanted in splits_to_process:
        print(f"\nLoading hotpotqa/hotpot_qa distractor ({split_name})...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)
        print(f"  Loaded {len(ds)} examples")

        if args.question_type != "all":
            ds = ds.filter(lambda ex: ex["type"] == args.question_type)
            print(f"  After filtering to type={args.question_type}: {len(ds)} examples")

        if args.level != "all":
            ds = ds.filter(lambda ex: ex["level"] == args.level)
            print(f"  After filtering to level={args.level}: {len(ds)} examples")

        if len(ds) == 0:
            print(f"  No examples remaining for {split_name}, skipping.")
            continue

        indices = list(range(len(ds)))
        rng.shuffle(indices)

        if multi_query:
            # Multi-query: each example needs num_queries source examples
            source_needed = num_wanted * args.num_queries
            if len(ds) < source_needed:
                num_wanted = len(ds) // args.num_queries
                print(f"  Warning: only enough for {num_wanted} multi-query examples")
            if num_wanted == 0:
                continue

            selected_count = num_wanted * args.num_queries
            selected = indices[:selected_count]
            pool_indices = indices[selected_count:selected_count + selected_count] or indices
            distractor_pool = build_distractor_pool(ds, pool_indices, rng)
            print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

            # Group into bundles of num_queries
            groups = []
            for i in range(0, selected_count, args.num_queries):
                groups.append([ds[idx] for idx in selected[i:i + args.num_queries]])

            docs_label = f"total_docs={args.total_docs}" if args.total_docs > 0 else "supporting_only"
            hn_label = f", {args.num_hard_negatives} BM25 hard neg" if args.num_hard_negatives > 0 else ""
            print(f"  Generating {len(groups)} multi-query examples "
                  f"(N={args.num_queries}, {docs_label}{hn_label})...")

            examples = []
            for group in tqdm(groups, desc=f"HotpotQA multi-query {split_name}"):
                examples.append(build_multi_example(
                    group, distractor_pool, args.total_docs, rng,
                    use_titles=not args.no_titles,
                    bm25_searcher=bm25_searcher,
                    num_hard_negatives=args.num_hard_negatives,
                    max_pair_overlap=args.max_hard_neg_pair_overlap,
                ))
        else:
            # Single-query mode — batch BM25 calls across multiple examples.
            n = min(num_wanted, len(ds))
            selected = indices[:n]
            pool_indices = indices[n:n + n * 3] if n < len(ds) else indices
            distractor_pool = build_distractor_pool(ds, pool_indices, rng) if args.num_docs > 0 else []
            if distractor_pool:
                print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

            hn_label = f", {args.num_hard_negatives} BM25 hard neg" if args.num_hard_negatives > 0 else ""
            desc = f"HotpotQA {split_name} k{args.num_docs}{hn_label}"

            # Per-example we need hard negs = min(num_hard_negatives, num_docs - |supporting|);
            # |supporting| is always 2 for bridge questions so a single cap is fine.
            n_hard_cap = min(args.num_hard_negatives, max(0, args.num_docs - 2))

            examples = []
            pbar = tqdm(total=len(selected), desc=desc)
            for batch_start in range(0, len(selected), args.batch_size):
                batch_ids = selected[batch_start:batch_start + args.batch_size]
                batch = [ds[i] for i in batch_ids]

                if bm25_searcher is not None and n_hard_cap > 0:
                    queries = [ex["question"] for ex in batch]
                    exclude_answers_list = [[ex["answer"]] for ex in batch]
                    hard_neg_batch = bm25_searcher.batch_mine_hard_negs(
                        queries, num_hard=n_hard_cap,
                        exclude_answers_list=exclude_answers_list,
                        max_pair_overlap=args.max_hard_neg_pair_overlap,
                        threads=args.bm25_threads,
                    )
                else:
                    hard_neg_batch = [[] for _ in batch]

                for ex, hard_negs in zip(batch, hard_neg_batch):
                    examples.append(build_single_example(
                        ex, distractor_pool, args.num_docs, rng,
                        use_titles=not args.no_titles,
                        bm25_searcher=bm25_searcher,
                        num_hard_negatives=args.num_hard_negatives,
                        max_pair_overlap=args.max_hard_neg_pair_overlap,
                        pre_mined_hard_negs=hard_negs,
                    ))
                    pbar.update(1)
            pbar.close()

        # Build filename
        if multi_query:
            flags = [f"n{args.num_queries}"]
            flags.append(f"k{args.total_docs}" if args.total_docs > 0 else "suponly")
        elif args.num_docs == 0:
            flags = ["nocontext"]
        else:
            flags = [f"k{args.num_docs}"]

        if args.question_type != "all":
            flags.append(args.question_type)
        if args.level != "all":
            flags.append(args.level)
        if args.no_titles:
            flags.append("notitle")
        if args.num_hard_negatives > 0:
            flags.append(f"hn{args.num_hard_negatives}")
        tag = "_".join(flags)

        label = "train" if split_name == "train" else "eval"
        prefix = "multi_hotpotqa" if multi_query else "hotpotqa"
        path = f"{args.output_dir}/{prefix}_{label}_{tag}_{len(examples)}.jsonl"

        if args.align and args.num_hard_negatives > 0 and args.num_docs > 0:
            examples = _align_examples(examples)

        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)


if __name__ == "__main__":
    main()
