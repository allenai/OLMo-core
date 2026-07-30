"""Generate review outlier detection data from Amazon-Reviews-2023.

Two example types, mixed in one dataset via --rating-ratio:

  (a) rating: majority of reviews share one star rating (1-5); a few outliers
      have a different star rating. Query: "Can you find outliers with the
      least common rating in this data?"

  (b) category: majority of reviews come from one product category; a few
      outliers are from a different category. Query: "Can you find outliers
      from the least common category?"

No LLM labeling. Ground truth comes from structured metadata only:
  - rating   := the star rating field, normalized to an int in {1,2,3,4,5}.
  - category := HF config name (e.g. "Books", "Electronics", "Beauty_and_Personal_Care").

Documents shown to the model contain only review title + body (no rating,
no category label), so the model must read the content to identify outliers.

Output: unified structured JSONL with fields:
    documents:         list of {title, text}
    queries:           list of query strings (length 1)
    answers:           list of list of outlier doc titles (gold)
    gold_doc_indices:  list of list of 0-indexed outlier positions
    source:            "review_outlier_rating" | "review_outlier_category"
    meta:              {majority_label, minority_label, num_outliers}

Usage (corpus-reasoning-eval env):
    python scripts/data/generate_review_outlier_data.py \\
        --num-examples 200 --num-docs 100 --num-outliers 4 \\
        --sentiment-ratio 0.5 \\
        --categories Books Electronics Beauty_and_Personal_Care Home_and_Kitchen \\
        --out data/review_outlier_200.jsonl
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


RATING_QUERY = "Can you find outliers with the least common rating in this data?"
CATEGORY_QUERY = "Can you find outliers from the least common category?"

VALID_RATINGS = (1, 2, 3, 4, 5)

# Stop-scanning cap per category when pooling reviews.
DEFAULT_PER_CATEGORY_POOL = 20000


def normalize_rating(r) -> int | None:
    try:
        r = int(round(float(r)))
    except (TypeError, ValueError):
        return None
    return r if r in VALID_RATINGS else None


def clean_text(text: str, title: str) -> tuple[str, str]:
    text = (text or "").strip()
    title = (title or "").strip()
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)
    title = re.sub(r"\s+", " ", title)
    return title, text


def pool_from_category(hf_dataset: str, category: str, pool_size: int,
                        min_len: int, max_len: int, seed: int) -> list[dict]:
    """Stream a category and collect up to pool_size usable reviews."""
    rng = random.Random(seed)
    # Reservoir-style: take first pool_size, then randomly replace.
    # The Amazon-Reviews-2023 dataset's loading script is unsupported on newer
    # `datasets`. Load the raw jsonl.gz files directly.
    path = f"hf://datasets/{hf_dataset}/raw/review_categories/{category}.jsonl"
    ds = load_dataset("json", data_files=path, split="train", streaming=True)
    kept: list[dict] = []
    pbar = tqdm(desc=f"scan {category}", unit="rev")
    for i, row in enumerate(ds):
        pbar.update(1)
        title, text = clean_text(row.get("text", ""), row.get("title", ""))
        if len(text) < min_len or len(text) > max_len:
            continue
        rating = normalize_rating(row.get("rating"))
        if rating is None:
            continue
        item = {
            "title": title,
            "text": text,
            "rating": rating,
            "category": category,
            "parent_asin": row.get("parent_asin"),
        }
        kept.append(item)
        if len(kept) >= pool_size:
            break
        if i > 500000 and len(kept) < pool_size // 2:
            # Category is sparse; bail early.
            break
    pbar.close()
    rng.shuffle(kept)
    return kept


def build_rating_example(pool: list[dict], num_docs: int, num_outliers: int,
                          rng: random.Random) -> dict | None:
    """Pick num_docs reviews from one category; majority one star rating (1-5), outliers a different rating."""
    by_rating = defaultdict(list)
    for r in pool:
        by_rating[r["rating"]].append(r)

    n_maj = num_docs - num_outliers
    # Valid majority ratings: those with >= n_maj reviews.
    maj_candidates = [r for r in VALID_RATINGS if len(by_rating[r]) >= n_maj]
    if not maj_candidates:
        return None
    majority = rng.choice(maj_candidates)
    min_candidates = [r for r in VALID_RATINGS if r != majority and len(by_rating[r]) >= num_outliers]
    if not min_candidates:
        return None
    minority = rng.choice(min_candidates)

    maj_sample = rng.sample(by_rating[majority], n_maj)
    min_sample = rng.sample(by_rating[minority], num_outliers)
    docs = maj_sample + min_sample
    rng.shuffle(docs)

    outlier_ids = {id(d) for d in min_sample}
    gold_idx = [i for i, d in enumerate(docs) if id(d) in outlier_ids]

    return {
        "documents": [{"title": d["title"], "text": d["text"]} for d in docs],
        "queries": [RATING_QUERY],
        "answers": ["; ".join(docs[i]["title"] for i in gold_idx)],
        "gold_doc_indices": gold_idx,
        "source": "review_outlier_rating",
        "meta": {
            "majority_label": majority,
            "minority_label": minority,
            "num_outliers": num_outliers,
            "category": pool[0]["category"] if pool else None,
        },
    }


def build_category_example(pools_by_cat: dict[str, list[dict]], num_docs: int,
                            num_outliers: int, rng: random.Random) -> dict | None:
    cats = [c for c, p in pools_by_cat.items() if len(p) >= num_docs]
    if len(cats) < 2:
        return None
    majority = rng.choice(cats)
    other_cats = [c for c in pools_by_cat if c != majority and len(pools_by_cat[c]) >= num_outliers]
    if not other_cats:
        return None
    minority = rng.choice(other_cats)

    n_maj = num_docs - num_outliers
    maj_sample = rng.sample(pools_by_cat[majority], n_maj)
    min_sample = rng.sample(pools_by_cat[minority], num_outliers)
    docs = maj_sample + min_sample
    rng.shuffle(docs)

    outlier_ids = {id(d) for d in min_sample}
    gold_idx = [i for i, d in enumerate(docs) if id(d) in outlier_ids]

    return {
        "documents": [{"title": d["title"], "text": d["text"]} for d in docs],
        "queries": [CATEGORY_QUERY],
        "answers": ["; ".join(docs[i]["title"] for i in gold_idx)],
        "gold_doc_indices": gold_idx,
        "source": "review_outlier_category",
        "meta": {
            "majority_label": majority,
            "minority_label": minority,
            "num_outliers": num_outliers,
        },
    }


def _partition_with_min(total: int, k: int, minimum: int, rng: random.Random) -> list[int]:
    """Split `total` into k parts each >= minimum, with non-uniform proportions.

    Uses an exponential-weights draw (≈ Dirichlet(α=1)) so distributions vary
    markedly from call to call. Assumes k * minimum <= total.
    """
    remainder = total - k * minimum
    weights = [rng.expovariate(1.0) for _ in range(k)]
    s = sum(weights) or 1.0
    counts = [int(remainder * w / s) for w in weights]
    # Fix rounding so they sum to remainder exactly
    leftover = remainder - sum(counts)
    for i in range(leftover):
        counts[i % k] += 1
    return [c + minimum for c in counts]


def build_category_example_mixed(pools_by_cat: dict[str, list[dict]], num_docs: int,
                                  num_outliers: int, min_k: int, max_k: int,
                                  rng: random.Random) -> dict | None:
    """Category-variant example with variable #categories and non-uniform counts.

    - Pick K in [min_k, max_k] total categories present.
    - One is the outlier (exactly num_outliers docs).
    - Remaining K-1 majority categories each get >= num_outliers+1 docs
      (keeps the outlier uniquely-smallest), with counts drawn from an
      exponential-weights partition (high cross-example variance).
    """
    min_per_maj = num_outliers + 1
    n_maj = num_docs - num_outliers
    hard_k_max = 1 + n_maj // min_per_maj  # feasibility cap
    k_max = min(max_k, hard_k_max, len(pools_by_cat))
    k_min = max(min_k, 2)
    if k_max < k_min:
        return None

    # Need enough per-category pool size for sampling without replacement.
    eligible = [c for c, p in pools_by_cat.items() if len(p) >= max(num_outliers, min_per_maj)]
    if len(eligible) < k_min:
        return None
    K = rng.randint(k_min, min(k_max, len(eligible)))

    outlier_cat = rng.choice(eligible)
    other_eligible = [c for c in eligible if c != outlier_cat]
    majority_cats = rng.sample(other_eligible, K - 1)

    counts = _partition_with_min(n_maj, K - 1, min_per_maj, rng)
    # Ensure every majority category has enough pool to draw count docs.
    # If not, bail (caller will retry).
    for cat, n in zip(majority_cats, counts):
        if len(pools_by_cat[cat]) < n:
            return None

    maj_samples = []
    for cat, n in zip(majority_cats, counts):
        maj_samples.extend(rng.sample(pools_by_cat[cat], n))
    min_sample = rng.sample(pools_by_cat[outlier_cat], num_outliers)

    docs = maj_samples + min_sample
    rng.shuffle(docs)
    outlier_ids = {id(d) for d in min_sample}
    gold_idx = [i for i, d in enumerate(docs) if id(d) in outlier_ids]

    distribution = {cat: n for cat, n in zip(majority_cats, counts)}
    distribution[outlier_cat] = num_outliers

    return {
        "documents": [{"title": d["title"], "text": d["text"]} for d in docs],
        "queries": [CATEGORY_QUERY],
        "answers": ["; ".join(docs[i]["title"] for i in gold_idx)],
        "gold_doc_indices": gold_idx,
        "source": "review_outlier_category",
        "meta": {
            "majority_label": None,  # signals mixed-majority v2 example
            "minority_label": outlier_cat,
            "num_outliers": num_outliers,
            "num_categories": K,
            "category_distribution": distribution,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", default="McAuley-Lab/Amazon-Reviews-2023")
    p.add_argument(
        "--categories", nargs="+",
        default=["Books", "Electronics", "Beauty_and_Personal_Care", "Home_and_Kitchen"],
        help="HF subset suffixes (raw_review_<CAT>). Need >= 2 for category examples.",
    )
    p.add_argument("--num-examples", type=int, default=200)
    p.add_argument("--num-docs", type=int, default=100)
    p.add_argument("--num-outliers", type=int, default=4)
    p.add_argument("--rating-ratio", type=float, default=0.5,
                   help="Fraction of examples that are rating-skew type (rest are category-skew type).")
    p.add_argument("--pool-size", type=int, default=DEFAULT_PER_CATEGORY_POOL,
                   help="Target reviews pooled per category.")
    p.add_argument("--min-text-chars", type=int, default=120)
    p.add_argument("--max-text-chars", type=int, default=1500)
    p.add_argument("--min-categories-per-ex", type=int, default=2,
                   help="Min # distinct categories in a category-variant example. "
                        "Default 2 reproduces v1 (one majority + one outlier).")
    p.add_argument("--max-categories-per-ex", type=int, default=2,
                   help="Max # distinct categories in a category-variant example. "
                        "If >2, uses the mixed-majority v2 builder with non-uniform "
                        "per-category counts.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/review_outlier.jsonl")
    args = p.parse_args()

    assert 0.0 <= args.rating_ratio <= 1.0
    assert args.num_outliers < args.num_docs

    rng = random.Random(args.seed)

    # Pool reviews per category once.
    pools_by_cat: dict[str, list[dict]] = {}
    for i, cat in enumerate(args.categories):
        pools_by_cat[cat] = pool_from_category(
            args.hf_dataset, cat, args.pool_size,
            args.min_text_chars, args.max_text_chars, args.seed + i,
        )
        rating_hist = Counter(x["rating"] for x in pools_by_cat[cat])
        hist_str = " ".join(f"{r}★:{rating_hist[r]}" for r in VALID_RATINGS)
        print(f"  pool[{cat}]: {len(pools_by_cat[cat])} reviews ({hist_str})")

    n_rating = int(round(args.num_examples * args.rating_ratio))
    n_category = args.num_examples - n_rating
    print(f"\nTargets: {n_rating} rating examples, {n_category} category examples")

    examples: list[dict] = []
    cats_with_pool = list(pools_by_cat.keys())

    # (a) rating-skew examples.
    attempts = 0
    while len([e for e in examples if e["source"] == "review_outlier_rating"]) < n_rating and attempts < n_rating * 5:
        cat = rng.choice(cats_with_pool)
        ex = build_rating_example(pools_by_cat[cat], args.num_docs, args.num_outliers, rng)
        if ex is not None:
            examples.append(ex)
        attempts += 1

    # (b) category examples.
    use_mixed = args.max_categories_per_ex > 2
    attempts = 0
    while len([e for e in examples if e["source"] == "review_outlier_category"]) < n_category and attempts < n_category * 5:
        if use_mixed:
            ex = build_category_example_mixed(
                pools_by_cat, args.num_docs, args.num_outliers,
                args.min_categories_per_ex, args.max_categories_per_ex, rng,
            )
        else:
            ex = build_category_example(pools_by_cat, args.num_docs, args.num_outliers, rng)
        if ex is not None:
            examples.append(ex)
        attempts += 1

    rng.shuffle(examples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(out), examples)
    print(f"\nWrote {len(examples)} examples to {out}")
    print_dataset_stats(examples, "review_outlier", str(out))

    # Write a markdown preview: 2 full examples per kind, with gold marked.
    preview_path = out.with_suffix(".preview.md")
    lines = [f"# Review outlier preview — {out.name}", ""]
    for kind in ("review_outlier_rating", "review_outlier_category"):
        kind_examples = [e for e in examples if e["source"] == kind][:2]
        for ex_i, ex in enumerate(kind_examples):
            gold = set(ex["gold_doc_indices"])
            lines.append(f"## {kind} — example {ex_i}")
            lines.append(f"- query: {ex['queries'][0]}")
            lines.append(f"- meta: `{json.dumps(ex['meta'])}`")
            lines.append(f"- gold indices: {sorted(gold)}")
            lines.append("")
            for i, d in enumerate(ex["documents"]):
                marker = " **[GOLD OUTLIER]**" if i in gold else ""
                lines.append(f"### doc {i}{marker} — {d['title']}")
                lines.append(d["text"])
                lines.append("")
    preview_path.write_text("\n".join(lines))
    print(f"Wrote preview to {preview_path}")

    for kind in ("review_outlier_rating", "review_outlier_category"):
        ex = next((e for e in examples if e["source"] == kind), None)
        if ex is None:
            continue
        print(f"\n--- preview: {kind} ---")
        print(f"  query: {ex['queries'][0]}")
        print(f"  meta:  {ex['meta']}")
        print(f"  gold:  {ex['gold_doc_indices']}")


if __name__ == "__main__":
    main()
