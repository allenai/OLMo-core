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

from corpus_reasoning.lib.bm25 import sample_from_pool
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


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
        # Store the FULL nq_open alias list, not just answers[0]. Gold selection
        # and distractor exclusion both run over the full list, so storing only
        # the first alias left ~18% of golds not containing their own stored
        # answer (the gold matched a different alias). answers[0] stays first,
        # so downstream "The answer is {answers[0]}" CoT injection is unchanged.
        "answers": list(answers),
        "gold_doc_indices": gold_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": "nq",
    }


def _ce_gold_keep(scorer, questions, golds, hard_neg_lists,
                  ce_gold_min, ce_margin, ce_ref_negs):
    """Decide, per example, whether the gold is a trustworthy relevant passage.

    Returns a list of bools (one per example; entries with gold=None are False).
    Keep iff the gold's cross-encoder relevance score is (a) >= `ce_gold_min`
    in absolute terms AND (b) >= every reference hard-negative's score +
    `ce_margin`, i.e. the answer-bearing gold is genuinely the most
    query-relevant passage, not an incidental answer-string match that a
    distractor out-ranks.

    The gold is compared against only the top `ce_ref_negs` hard negatives
    (highest BM25, a stable prefix). This keeps the decision independent of the
    pool size k — so the same questions survive across k20/k50/k100/k200 — and
    avoids over-filtering large pools where "beat all 199 hard negs" is far
    stricter than "beat all 19".

    All (query, passage) pairs for the batch are scored in one CE call.
    """
    pairs = []
    spans = []  # (gold_pair_idx, [hard_pair_idx...]) per example, or None
    for q, (gold, _docid), hard_negs in zip(questions, golds, hard_neg_lists):
        if gold is None:
            spans.append(None)
            continue
        gi = len(pairs)
        pairs.append((q, gold["text"]))
        his = []
        for hn in hard_negs[:ce_ref_negs]:
            his.append(len(pairs))
            pairs.append((q, hn["text"]))
        spans.append((gi, his))
    if not pairs:
        return [False] * len(questions)
    scores = scorer.score(pairs)
    keep = []
    for span in spans:
        if span is None:
            keep.append(False)
            continue
        gi, his = span
        g = scores[gi]
        best_hn = max((scores[i] for i in his), default=float("-inf"))
        keep.append(g >= ce_gold_min and g >= best_hn + ce_margin)
    return keep


def _generate_split(ds, num_wanted, args, rng, bm25_searcher, split_name,
                    ce_scorer=None, random_pool=None):
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
    # Stride-shard the (identically-shuffled) question pool across parallel workers so
    # each shard draws a disjoint set of questions. All shards must share --seed.
    if getattr(args, "num_shards", 1) > 1:
        indices = indices[args.shard_index::args.num_shards]

    # Continuous per-example context-length sampling. When --num-docs-min/-max
    # are set, request the MAX hard-negs once per batch (one deep BM25 search,
    # reused) and then sample a per-example doc count k in [min, max], slicing
    # the hard-neg list to k-1. Keeps n_random=0 (all distractors are BM25
    # hard negs, matching the ladder's num_hard_negatives = num_docs-1 setup).
    continuous = args.num_docs_min is not None or args.num_docs_max is not None
    if continuous:
        if args.num_docs_min is None or args.num_docs_max is None:
            raise ValueError(
                "--num-docs-min and --num-docs-max must both be set together"
            )
        if args.num_docs_min < 1 or args.num_docs_max < args.num_docs_min:
            raise ValueError(
                "require 1 <= --num-docs-min <= --num-docs-max"
            )
        # The deep BM25 search only needs the MAX hard-negs any example will use
        # (~frac*(max_docs-1)), NOT all (max_docs-1). Fetching ~frac x fewer hard
        # negs shrinks Lucene's search depth ~1/frac x -> big BM25 speedup at frac<1.
        _max_hard_used = max(1, int(args.hard_neg_frac * (args.num_docs_max - 1)) + 1)
        n_hard = min(args.num_docs_max - 1, _max_hard_used + 10)
        n_random = 0
    else:
        n_hard = min(args.num_hard_negatives, args.num_docs - 1)
        n_random = args.num_docs - 1 - n_hard

    if continuous:
        desc = f"NQ {split_name} k[{args.num_docs_min},{args.num_docs_max}] BM25 hard"
    else:
        hn_label = (
            f", {args.num_hard_negatives} BM25 hard neg"
            if args.num_hard_negatives > 0 else ""
        )
        desc = f"NQ {split_name} k{args.num_docs}{hn_label}"

    examples: list[dict] = []
    skipped = 0
    ce_dropped = 0
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

        # Gold-quality filter: keep only examples whose answer-bearing gold is
        # genuinely the most query-relevant passage (drops incidental matches).
        if ce_scorer is not None:
            ce_keep = _ce_gold_keep(
                ce_scorer, questions, golds, hard_neg_lists,
                args.ce_gold_min, args.ce_margin, args.ce_ref_negs,
            )
        else:
            ce_keep = [True] * len(batch)

        for ex, (gold, gold_docid), hard_negs, keep in zip(
                batch, golds, hard_neg_lists, ce_keep):
            if len(examples) >= num_wanted:
                break
            if gold is None:
                skipped += 1
                continue
            if not keep:
                ce_dropped += 1
                continue
            if continuous:
                # Per-example doc count k in [min, max]; 1 gold + (k-1) negatives.
                # --hard-neg-frac controls the hard/random split: n_hard = frac*(k-1)
                # BM25 hard negs (>=1 when frac>0) + the rest random corpus distractors.
                # frac=1.0 -> legacy all-hard; frac=0.1 -> 10% hard + 90% random.
                k = rng.randint(args.num_docs_min, args.num_docs_max)
                n_neg = k - 1
                n_hard_k = min(n_neg, max(1, round(args.hard_neg_frac * n_neg)) if args.hard_neg_frac > 0 else 0)
                ex_hard_negs = hard_negs[:n_hard_k]
                n_rand_k = n_neg - len(ex_hard_negs)
                if n_rand_k > 0 and random_pool is not None:
                    # Pre-fetched pool + pure-Python sampling: avoids up to
                    # ~180 serial Lucene `doc()` lookups per example (the
                    # dominant cost at n_docs up to 200 w/ hard-neg-frac 0.1
                    # -- see BM25Searcher.prefetch_random_pool docstring).
                    pool_distractors = sample_from_pool(
                        random_pool, n_rand_k,
                        exclude_answers=ex["answer"],
                        exclude_texts={gold["text"]} | {d["text"] for d in ex_hard_negs},
                        rng=rng,
                    )
                elif n_rand_k > 0:
                    pool_distractors = bm25_searcher.sample_random_passages(
                        n_rand_k,
                        exclude_docids={gold_docid} | {d["text"] for d in ex_hard_negs},
                        exclude_answers=ex["answer"],
                        rng=rng,
                    )
                else:
                    pool_distractors = []
            else:
                ex_hard_negs = hard_negs
                pool_distractors = bm25_searcher.sample_random_passages(
                    n_random,
                    exclude_docids={gold_docid} | {d["text"] for d in hard_negs},
                    exclude_answers=ex["answer"],
                    rng=rng,
                )
            examples.append(_assemble_example(
                ex["question"], ex["answer"], gold, ex_hard_negs, pool_distractors, rng,
            ))
            pbar.update(1)
    pbar.close()

    if skipped > 0:
        print(f"  Skipped {skipped} examples (gold passage not found in BM25 results)")
    if ce_dropped > 0:
        kept = len(examples)
        print(f"  CE-filter dropped {ce_dropped} examples "
              f"(gold not top-relevant); kept {kept} "
              f"({100*kept/(kept+ce_dropped):.1f}% of answer-found candidates)")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate NQ train + validation data")
    parser.add_argument("--num-train", type=int, default=10000,
                        help="Number of training examples to generate")
    parser.add_argument("--num-eval", type=int, default=500,
                        help="Number of validation examples to generate")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="Total documents per example (1 gold + N-1 distractors)")
    parser.add_argument("--num-docs-min", type=int, default=None,
                        help="Lower bound (inclusive) for per-example continuous "
                             "doc-count sampling. Set together with --num-docs-max "
                             "to sample k ~ U[min, max] docs per example (all "
                             "distractors BM25 hard negs); overrides --num-docs.")
    parser.add_argument("--num-docs-max", type=int, default=None,
                        help="Upper bound (inclusive) for per-example continuous "
                             "doc-count sampling. The deep BM25 search requests "
                             "num_docs_max-1 hard negs once per batch and reuses it.")
    parser.add_argument("--num-hard-negatives", type=int, default=0,
                        help="Number of BM25 hard negatives per example "
                             "(0 = all random from the corpus)")
    parser.add_argument("--hard-neg-frac", type=float, default=1.0,
                        help="CONTINUOUS mode only: fraction of the (k-1) negatives that are BM25 hard "
                             "negs; the rest are random corpus distractors. 1.0 = all-hard (legacy); "
                             "0.1 = 10%% hard + 90%% random (the CE-filtered 'latest' pipeline).")
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
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Stride-partition the shuffled question pool into N disjoint shards "
                             "for parallel generation (all shards share --seed).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which shard in [0, num-shards) this process generates.")
    parser.add_argument("--no-context", action="store_true",
                        help="Generate closed-book examples (no documents)")
    # Cross-encoder gold-quality filter (mainly for eval data). NQ golds are
    # the first BM25 hit *containing* an answer string, which is sometimes an
    # incidental match the model is unfairly penalized for missing. With
    # --ce-filter we keep an example only when the gold is genuinely the most
    # query-relevant passage in the pool.
    parser.add_argument("--ce-filter", action="store_true",
                        help="Drop examples whose answer-bearing gold is not the "
                             "top CE-relevant passage vs its hard negatives")
    parser.add_argument("--ce-model", default=None,
                        help="Cross-encoder model (default ms-marco-MiniLM-L-6-v2)")
    parser.add_argument("--ce-gold-min", type=float, default=0.0,
                        help="Minimum absolute CE relevance logit for the gold")
    parser.add_argument("--ce-margin", type=float, default=0.0,
                        help="Gold CE score must exceed every reference hard "
                             "negative's score by at least this margin")
    parser.add_argument("--ce-ref-negs", type=int, default=20,
                        help="Compare the gold against only the top-N BM25 hard "
                             "negatives, so the keep decision is stable across "
                             "pool sizes k (same questions survive at k20..k200)")
    parser.add_argument("--ce-batch-size", type=int, default=128)
    parser.add_argument("--pool-passages", type=int, default=0,
                        help="If > 0, pre-fetch this many random passages ONCE "
                             "(threaded) and sample random distractors from that "
                             "pool in pure Python instead of one Lucene doc() "
                             "lookup per distractor per example. Continuous mode "
                             "(--num-docs-min/-max) only. 0 = old per-example path.")
    parser.add_argument("--pool-fetch-threads", type=int, default=16)
    args = parser.parse_args()

    if args.no_context:
        args.num_docs = 0

    rng = random.Random(args.seed)

    from corpus_reasoning.lib.bm25 import BM25Searcher
    bm25_searcher = BM25Searcher()

    ce_scorer = None
    if args.ce_filter:
        from corpus_reasoning.lib.cross_encoder import CrossEncoderScorer, DEFAULT_CE_MODEL
        ce_scorer = CrossEncoderScorer(
            args.ce_model or DEFAULT_CE_MODEL, batch_size=args.ce_batch_size,
        )
        print(f"  CE gold filter ON (min={args.ce_gold_min}, margin={args.ce_margin})")

    random_pool = None
    if args.pool_passages > 0:
        print(f"Pre-fetching random pool: {args.pool_passages} passages "
              f"x {args.pool_fetch_threads} threads...")
        random_pool = bm25_searcher.prefetch_random_pool(
            args.pool_passages, threads=args.pool_fetch_threads, seed=args.seed,
        )
        print(f"  Pool ready: {len(random_pool)} passages")

    hn_tag = f"_hn{args.num_hard_negatives}" if args.num_hard_negatives > 0 else ""

    for split_name, num_wanted in [("train", args.num_train), ("validation", args.num_eval)]:
        if num_wanted <= 0:
            continue

        print(f"\nLoading nq_open ({split_name})...")
        ds = load_dataset("nq_open", split=split_name)
        print(f"  Loaded {len(ds)} questions")

        examples = _generate_split(
            ds, num_wanted, args, rng, bm25_searcher, split_name,
            ce_scorer=ce_scorer, random_pool=random_pool,
        )

        shard_tag = f"_s{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
        if args.no_context:
            path = f"{args.output_dir}/nq_{split_name}_nocontext{shard_tag}_{len(examples)}.jsonl"
        elif args.num_docs_min is not None:
            path = (f"{args.output_dir}/nq_{split_name}"
                    f"_k{args.num_docs_min}-{args.num_docs_max}{shard_tag}_{len(examples)}.jsonl")
        else:
            path = f"{args.output_dir}/nq_{split_name}_k{args.num_docs}{hn_tag}_{len(examples)}.jsonl"

        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)


if __name__ == "__main__":
    main()
