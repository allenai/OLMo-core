"""Generate topic-outlier detection data from the Wikipedia 100w corpus.

Wikipedia analog of generate_review_outlier_data.py. Article titles play the
role that product categories play in the Amazon-Reviews task: the majority
of passages come from one (or several) Wikipedia article(s); a minority
come from a different unrelated article — the outliers.

Two modes (mixed in one dataset by --simple-ratio):

  simple v1: 1 majority article + 1 outlier article.
  mixed  v2: K majority articles (imbalanced counts) + 1 outlier article.

There is no rating analog — Wikipedia chunks don't carry star ratings — so
this generator does category-style only.

Important: passage titles are intentionally hidden (documents[i].title=None)
because Wikipedia article titles would trivially leak the outlier identity.
The model must read the body to identify outliers.

Output schema (mirrors generate_review_outlier_data.py):
    documents:         list of {title: null, text: <chunk body>}
    queries:           [<topic-outlier query string>]
    answers:           ["<1-indexed positions joined by ';'>"]
    gold_doc_indices:  list of 0-indexed outlier positions
    source:            "wiki_outlier_topic"
    meta:              {majority_label, minority_label, num_outliers, ...}

Usage:
    python scripts/data/generate_wiki_outlier_data.py \\
        --num-examples 200 --num-docs 20 --num-outliers 3 \\
        --simple-ratio 0.5 --out-dir data/
"""

import argparse
import json
import math
import random
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats
from corpus_reasoning.lib.wiki100w_pool import ArticlePool, build_article_pool

TOPIC_QUERY = (
    "Can you find passages that are about a different topic than the rest "
    "of these passages?"
)


def _docs_and_gold(majority_chunks: list[dict], outlier_chunks: list[dict],
                   rng: random.Random) -> tuple[list[dict], list[int]]:
    """Combine majority + outlier chunks, shuffle, return (docs, gold_idx)."""
    items = [(c, False) for c in majority_chunks] + \
            [(c, True) for c in outlier_chunks]
    rng.shuffle(items)
    docs = [{"title": None, "text": c["body"]} for c, _ in items]
    gold_idx = [i for i, (_, is_out) in enumerate(items) if is_out]
    return docs, gold_idx


def _answers_from_gold(gold_idx: list[int]) -> str:
    """1-indexed gold positions joined by '; ' — the `answers` field is metadata only;
    the prompt-side answer is built from gold_doc_indices in data_format.py."""
    return "; ".join(str(g + 1) for g in sorted(gold_idx))


def _run_titles(run: list[dict]) -> set[str]:
    """All distinct article titles touched by a run (a run may be composed from
    several articles for long lengths, so exclude every one, not just run[0])."""
    return {c["title"] for c in run}


def build_simple_example(pool: ArticlePool, num_docs: int, num_outliers: int,
                         rng: random.Random) -> dict | None:
    """Simple v1: one majority article + one outlier article.

    The majority is required to be a SINGLE article (``single_only=True``); if no
    single article is long enough this returns None so the caller can fall back
    to mixed mode.
    """
    n_maj = num_docs - num_outliers
    majority = pool.sample_run(n_maj, rng=rng, single_only=True)
    if majority is None:
        return None
    outlier = pool.sample_run(num_outliers, rng=rng,
                              exclude_titles=_run_titles(majority))
    if outlier is None:
        return None
    docs, gold_idx = _docs_and_gold(majority, outlier, rng)
    return {
        "documents": docs,
        "queries": [TOPIC_QUERY],
        "answers": [_answers_from_gold(gold_idx)],
        "gold_doc_indices": gold_idx,
        "source": "wiki_outlier_topic",
        "meta": {
            "majority_label": majority[0]["title"],
            "minority_label": outlier[0]["title"],
            "num_outliers": num_outliers,
            "num_docs": num_docs,
            "mode": "simple",
        },
    }


def _partition_with_min(total: int, k: int, minimum: int,
                        rng: random.Random) -> list[int]:
    """Split `total` into k parts each >= minimum, with non-uniform proportions.

    Mirrors generate_review_outlier_data._partition_with_min: exponential-weight
    draw (≈ Dirichlet(α=1)) so distributions vary markedly across examples.
    """
    remainder = total - k * minimum
    if remainder < 0:
        return []
    weights = [rng.expovariate(1.0) for _ in range(k)]
    s = sum(weights) or 1.0
    counts = [int(remainder * w / s) for w in weights]
    leftover = remainder - sum(counts)
    for i in range(leftover):
        counts[i % k] += 1
    return [c + minimum for c in counts]


def build_mixed_example(pool: ArticlePool, num_docs: int, num_outliers: int,
                        min_k: int, max_k: int, rng: random.Random,
                        maj_outlier_gap: int = 1) -> dict | None:
    """Mixed v2: K majority articles with imbalanced counts + 1 outlier article.

    `maj_outlier_gap` enforces that every majority topic has at least
    `num_outliers + maj_outlier_gap` documents — so the outlier topic is the
    smallest by at least `maj_outlier_gap`. Default of 1 reproduces v1 behavior.
    """
    min_per_maj = num_outliers + maj_outlier_gap
    n_maj = num_docs - num_outliers
    hard_k_max = 1 + n_maj // min_per_maj
    k_max = min(max_k, hard_k_max)
    k_min = max(min_k, 2)
    if k_max < k_min:
        return None
    K = rng.randint(k_min, k_max)

    counts = _partition_with_min(n_maj, K - 1, min_per_maj, rng)
    if not counts:
        return None

    majority_runs: list[list[dict]] = []
    used_titles: set[str] = set()
    for c in counts:
        run = pool.sample_run(c, rng=rng, exclude_titles=used_titles)
        if run is None:
            return None
        used_titles.update(_run_titles(run))
        majority_runs.append(run)

    outlier = pool.sample_run(num_outliers, rng=rng, exclude_titles=used_titles)
    if outlier is None:
        return None

    majority_chunks = [c for run in majority_runs for c in run]
    docs, gold_idx = _docs_and_gold(majority_chunks, outlier, rng)
    distribution = {run[0]["title"]: len(run) for run in majority_runs}
    distribution[outlier[0]["title"]] = num_outliers
    return {
        "documents": docs,
        "queries": [TOPIC_QUERY],
        "answers": [_answers_from_gold(gold_idx)],
        "gold_doc_indices": gold_idx,
        "source": "wiki_outlier_topic",
        "meta": {
            "majority_label": None,  # signals mixed-majority (v2)
            "minority_label": outlier[0]["title"],
            "num_outliers": num_outliers,
            "num_categories": K,
            "category_distribution": distribution,
            "num_docs": num_docs,
            "mode": "mixed",
        },
    }


def build_articles_example(pool: ArticlePool, num_docs: int, num_outliers: int,
                           min_run: int, max_run: int, rng: random.Random,
                           max_tries: int = 400) -> dict | None:
    """Mirror of `build_v2_outlier_ladder.build_example`'s majority construction.

    WHY THIS EXISTS. The v2 EVAL ladder does NOT sample a category count. It fills
    the corpus with WHOLE articles of U[min_run, max_run] chunks (mean 9) until the
    rung is full, so K is *emergent* and lands in a tight band: measured K = 3/7/13/25
    at n = 22/55/110/220. `build_mixed_example` instead samples K ~ U[min_k, max_k]
    and then partitions, which at chunks-per-article=25 gave K ~ U[2,10] with ~40 docs
    per topic. Training on that and scoring on the eval is the M-axis mismatch in
    `records/contradiction-train-eval-non-iid.md` §2 — at n=220 the eval asks for 25
    categories where training's cap could not exceed 16 and its observed max was 10.

    Matching by re-deriving the eval's K through a different mechanism (fitting
    --chunks-per-article) is exactly how that bug arose. This shares the construction
    instead: same fill rule, same backoff, same parameters.
    """
    n_maj = num_docs - num_outliers
    outlier = pool.sample_run(num_outliers, rng=rng, single_only=True)
    if outlier is None:
        return None
    used = _run_titles(outlier)

    runs: list[list[dict]] = []
    cum = 0
    for _ in range(max_tries):
        if cum >= n_maj:
            break
        size = rng.randint(min_run, max_run)
        run = pool.sample_run(size, rng=rng, exclude_titles=used, single_only=True)
        if run is None:
            continue
        used |= _run_titles(run)
        runs.append(run)
        cum += len(run)
    if cum < n_maj:
        return None

    # Exact-length prefix, backing off so the trailing article never contributes
    # fewer than num_outliers+1 chunks (every majority topic stays strictly larger
    # than the outlier) -- identical to the eval builder's prefix_chunks().
    flat, art_of = [], []
    for ai, run in enumerate(runs):
        for c in run:
            flat.append(c)
            art_of.append(ai)
    m = min(n_maj, len(flat))
    min_maj = num_outliers + 1
    if m > 0:
        last_ai = art_of[m - 1]
        art_start = art_of.index(last_ai)
        partial = m - art_start
        if 0 < partial < min_maj:
            m = art_start
    maj = flat[:m]
    if not maj:
        return None

    docs, gold_idx = _docs_and_gold(maj, outlier, rng)
    maj_titles = sorted({c["title"] for c in maj})
    distribution: dict[str, int] = {}
    for c in maj:
        distribution[c["title"]] = distribution.get(c["title"], 0) + 1
    distribution[outlier[0]["title"]] = num_outliers
    return {
        "documents": docs,
        "queries": [TOPIC_QUERY],
        "answers": [_answers_from_gold(gold_idx)],
        "gold_doc_indices": gold_idx,
        "source": "wiki_outlier_topic",
        "meta": {
            "majority_label": None,
            "minority_label": outlier[0]["title"],
            "num_outliers": num_outliers,
            "num_categories": len(maj_titles) + 1,
            "category_distribution": distribution,
            "num_docs": len(docs),
            "mode": "articles",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=200)
    p.add_argument("--num-docs", type=int, default=20,
                   help="Total docs per example. Smaller than the Amazon "
                        "default (100) because Wikipedia articles often "
                        "have <30 chunks.")
    p.add_argument("--num-outliers", type=int, default=3)
    p.add_argument("--simple-ratio", type=float, default=0.5,
                   help="Fraction of examples that use the simple v1 mode "
                        "(one majority article). The rest use mixed v2.")
    p.add_argument("--mixed-min-k", type=int, default=2,
                   help="Min number of majority articles in mixed v2 "
                        "(K = majority articles + 1 outlier).")
    p.add_argument("--mixed-max-k", type=int, default=5)
    p.add_argument("--majority-mode", choices=["partition", "articles"],
                   default="partition",
                   help="'partition' (default, historical): sample K then split n_maj "
                        "into K-1 imbalanced runs. 'articles': fill the corpus with "
                        "WHOLE articles of U[--min-run,--max-run] chunks until full and "
                        "let K emerge -- the construction build_v2_outlier_ladder.py "
                        "uses, so training matches the scale-K eval by sharing the rule "
                        "rather than by fitting --chunks-per-article to it.")
    p.add_argument("--min-run", type=int, default=4,
                   help="articles mode: min chunks per majority article "
                        "(must exceed num_outliers)")
    p.add_argument("--max-run", type=int, default=14,
                   help="articles mode: max chunks per majority article")
    p.add_argument("--mixed-min-k-frac", type=float, default=0.0,
                   help="If >0, floor the sampled K at ceil(frac*max_k) so the "
                        "category count lands in a tight band near max_k instead "
                        "of U[min_k, max_k]. Needed to match the v2 EVAL ladder, "
                        "which fills rungs with whole articles (n=220 -> K 23-28). "
                        "0.0 = historical uniform behaviour.")
    p.add_argument("--mixed-max-k-cap", type=int, default=16,
                   help="Hard cap on the per-example mixed K for long examples. "
                        "For large num_docs, K is auto-grown so each majority "
                        "topic gets ~`chunks-per-article` chunks (findable as a "
                        "single article), up to this cap.")
    p.add_argument("--chunks-per-article", type=int, default=25,
                   help="Target chunks per majority article when auto-growing K "
                        "for long examples. Keeps each majority run inside one "
                        "article (max article ~87 chunks).")
    p.add_argument("--simple-max-docs", type=int, default=60,
                   help="Examples with num_docs above this always use mixed mode "
                        "(no single article is long enough for a clean simple "
                        "majority).")
    p.add_argument("--maj-outlier-gap", type=int, default=1,
                   help="Min gap between outlier doc count and any majority "
                        "topic's doc count. Each majority topic gets at least "
                        "(num_outliers + maj_outlier_gap) documents.")
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--max-build-retries", type=int, default=20,
                   help="Per-example retry budget when build_*_example "
                        "returns None (e.g. failed to find K distinct "
                        "long-enough articles).")
    # Per-example continuous length: when both set, num_docs is drawn uniformly
    # in [min_docs, max_docs] per example (overrides --num-docs).
    p.add_argument("--min-docs", type=int, default=None,
                   help="If set with --max-docs, sample num_docs ~ U[min,max] "
                        "per example for a continuous length distribution.")
    p.add_argument("--max-docs", type=int, default=None)
    # Pool options.
    p.add_argument("--pool-cache", default="data/wiki100w_article_pool.pkl",
                   help="Precomputed article-pool pickle. Built on first use.")
    p.add_argument("--min-article-chunks", type=int, default=4)
    p.add_argument("--pool-shards", type=int, default=8,
                   help="Parallel scan processes when building the pool.")
    p.add_argument("--pool-max-lids", type=int, default=None,
                   help="Cap the index scan to the first N Lucene ids when "
                        "building the pool (partial pool; for smoke tests).")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--out-name", default=None,
                   help="Output JSONL basename (without extension). Defaults to "
                        "an auto name encoding the doc-count range.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    continuous = args.min_docs is not None and args.max_docs is not None
    if continuous and args.min_docs > args.max_docs:
        raise ValueError("--min-docs must be <= --max-docs")

    print("Loading / building wiki100w article pool...")
    build_article_pool(min_article_chunks=args.min_article_chunks,
                        shards=args.pool_shards, cache=args.pool_cache,
                        max_lids=args.pool_max_lids)
    pool = ArticlePool(args.pool_cache)
    print(f"  pool: {len(pool):,} articles, max {pool.max_chunks} chunks/article")

    def _per_example_max_k(num_docs: int) -> int:
        """Grow mixed K for long examples so each majority run fits in one
        article (~chunks-per-article chunks), capped at --mixed-max-k-cap."""
        n_maj = num_docs - args.num_outliers
        grown = 1 + -(-n_maj // max(args.chunks_per_article, 1))  # ceil div
        return max(args.mixed_max_k, min(grown, args.mixed_max_k_cap))

    def _per_example_min_k(max_k: int) -> int:
        """Floor for the sampled K, optionally tied to max_k.

        K is drawn uniformly from [min_k, max_k]. With a constant min_k that
        makes the category count roughly U[2, max_k] — but the v2 EVAL ladder
        (`build_v2_outlier_ladder.py`) fills each rung with whole articles, so
        its K lands in a TIGHT band near the top (measured: n=220 -> 23-28 with
        max_k=26, not 2-26). Training on the uniform version and scoring on the
        eval is the M-axis train/eval mismatch documented in
        `records/contradiction-train-eval-non-iid.md` §2.

        --mixed-min-k-frac ties the floor to max_k so the band matches. It
        defaults to 0.0, which reproduces the historical uniform behaviour
        exactly, so no existing build changes.
        """
        if args.mixed_min_k_frac <= 0:
            return args.mixed_min_k
        return max(args.mixed_min_k, math.ceil(args.mixed_min_k_frac * max_k))

    examples: list[dict] = []
    n_simple = 0
    n_mixed = 0

    while len(examples) < args.num_examples:
        num_docs = (rng.randint(args.min_docs, args.max_docs)
                    if continuous else args.num_docs)
        # Simple mode only when a single article can plausibly cover the
        # majority; otherwise force mixed.
        use_simple = (rng.random() < args.simple_ratio
                      and num_docs <= args.simple_max_docs)
        max_k = _per_example_max_k(num_docs)
        ex = None
        actually_simple = use_simple
        if args.majority_mode == "articles":
            # Shares build_v2_outlier_ladder's fill rule; K is emergent, so the
            # simple/mixed split and the K-sampling knobs do not apply.
            for _ in range(args.max_build_retries):
                ex = build_articles_example(
                    pool, num_docs, args.num_outliers,
                    max(args.min_run, args.num_outliers + 1), args.max_run, rng)
                if ex is not None:
                    break
            if ex is None:
                print(f"  warning: gave up on an articles example "
                      f"(num_docs={num_docs}) after {args.max_build_retries} retries")
                continue
            examples.append(ex)
            n_mixed += 1
            continue
        for _ in range(args.max_build_retries):
            if use_simple:
                ex = build_simple_example(
                    pool, num_docs, args.num_outliers, rng)
                if ex is None:
                    # No single article long enough -> fall back to mixed.
                    actually_simple = False
                    ex = build_mixed_example(
                        pool, num_docs, args.num_outliers,
                        _per_example_min_k(max_k), max_k, rng,
                        maj_outlier_gap=args.maj_outlier_gap)
            else:
                ex = build_mixed_example(
                    pool, num_docs, args.num_outliers,
                    _per_example_min_k(max_k), max_k, rng,
                    maj_outlier_gap=args.maj_outlier_gap)
            if ex is not None:
                break
        if ex is None:
            mode = "simple" if use_simple else "mixed"
            print(f"  warning: gave up on a {mode} example (num_docs={num_docs}) "
                  f"after {args.max_build_retries} retries; trying again")
            continue
        examples.append(ex)
        if actually_simple and ex["meta"].get("mode") == "simple":
            n_simple += 1
        else:
            n_mixed += 1
        if len(examples) % 50 == 0:
            print(f"  built {len(examples)}/{args.num_examples} "
                  f"(simple={n_simple}, mixed={n_mixed})")

    print(f"\nProduced {len(examples)} examples "
          f"(simple={n_simple}, mixed={n_mixed}).")

    rng.shuffle(examples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        base = args.out_name
    elif continuous:
        base = (f"outlier_wiki100w_n{args.min_docs}-{args.max_docs}"
                f"_k{args.num_outliers}_{len(examples)}")
    else:
        base = f"outlier_wiki100w_n{args.num_docs}_k{args.num_outliers}_{len(examples)}"
    out_path = out_dir / f"{base}.jsonl"
    save_jsonl(str(out_path), examples)
    print_dataset_stats(
        [{"input": json.dumps(e["documents"])[:1], "output": e["answers"][0]}
         for e in examples], "all", str(out_path))
    print(f"\nWrote {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
