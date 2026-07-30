"""Build the **v2** outlier-detection eval ladder (scale-K, comparable rungs).

The generic build_v2_eval_ladders shrink path is WRONG for outlier: it drops
random distractors, which can shrink a majority topic below the outlier count and
destroy the "outlier is the rarest topic" invariant (mixed-v2 / scale-K). This
builder instead fixes each example's OUTLIER article across all rungs and scales
only the MAJORITY:

  * per example, sample ONE outlier article (num_outliers chunks) -- FIXED across
    every rung, so the question/answer content is identical rung to rung (the v2
    goal), only the distractor majority grows;
  * sample an ORDERED list of single-article majority runs up to the largest rung;
  * each shorter rung is a nested CHUNK-PREFIX of that majority pool, backed off so
    the last (partial) majority article keeps >= num_outliers+1 chunks -- i.e. every
    majority topic in every rung is strictly larger than the outlier, so the outlier
    is always the unique rarest topic (scale-K preserved).

Rungs (docs/ex) mirror the v2 filenames the native eval reads:
    3k=n22, 8k=n55, 16k=n110, 32k=n220   (num_outliers=3)

gold_doc_indices are 0-indexed outlier positions (matching the generator +
_eval_outlier, which does int(g)+1). answers is the 1-indexed metadata string.

Usage:
    PYTHONPATH=. python scripts/data/build_v2_outlier_ladder.py \
        --num-examples 600 --out-root /scratch/users/prasann/cpt_data/eval500_v2
"""

import argparse
import os
import random

from corpus_reasoning.lib.io import save_jsonl
from corpus_reasoning.lib.wiki100w_pool import ArticlePool

TOPIC_QUERY = (
    "Can you find passages that are about a different topic than the rest "
    "of these passages?"
)
RUNGS = {22: "3k", 55: "8k", 110: "16k", 220: "32k"}
OUT_TMPL = "outlier_wiki100w_n{n}_k3_eval_600.jsonl"
SEED = 1234


def _titles(run):
    return {c["title"] for c in run}


def build_majority_pool(pool, need_chunks, num_outliers, exclude, rng,
                        min_run, max_run, max_tries=400):
    """Ordered list of single-article majority runs whose total chunk count is
    >= need_chunks. Each run is one article of >= num_outliers+1 chunks (so it
    can never be the rarest topic). Returns (runs, chunk_article_ids) or None."""
    runs, used, cum, tries = [], set(exclude), 0, 0
    min_run = max(min_run, num_outliers + 1)   # every majority topic > outlier
    while cum < need_chunks and tries < max_tries:
        tries += 1
        size = rng.randint(min_run, max_run)
        run = pool.sample_run(size, rng=rng, exclude_titles=used, single_only=True)
        if run is None:
            continue
        used |= _titles(run)
        runs.append(run)
        cum += len(run)
    if cum < need_chunks:
        return None
    return runs


def prefix_chunks(runs, target_maj, min_maj):
    """Nested chunk-prefix of the flattened majority runs totaling ~target_maj
    chunks, but never ending mid-article with fewer than `min_maj` chunks of that
    article (so no majority topic is <= the outlier). Returns list of chunks."""
    # article id per chunk (0-based over the ordered runs)
    flat, art_of = [], []
    for ai, run in enumerate(runs):
        for c in run:
            flat.append(c)
            art_of.append(ai)
    m = min(target_maj, len(flat))
    if m == 0:
        return []
    # If the prefix ends partway through an article with < min_maj chunks of it,
    # back off to that article's start (drop the too-small partial topic).
    last_ai = art_of[m - 1]
    art_start = art_of.index(last_ai)
    partial = m - art_start
    if 0 < partial < min_maj:
        m = art_start          # drop the undersized trailing topic
    return flat[:m]


def build_example(pool, ei, num_outliers, min_run, max_run):
    rng = random.Random(SEED * 1_000_003 + ei)
    max_n = max(RUNGS)
    # Fixed outlier article (one coherent topic), shared by every rung.
    outlier = pool.sample_run(num_outliers, rng=rng, single_only=True)
    if outlier is None:
        return None
    runs = build_majority_pool(pool, max_n - num_outliers, num_outliers,
                               _titles(outlier), rng, min_run, max_run)
    if runs is None:
        return None
    min_maj = num_outliers + 1
    per_rung = {}
    for n, lab in RUNGS.items():
        maj = prefix_chunks(runs, n - num_outliers, min_maj)
        items = [(c, False) for c in maj] + [(c, True) for c in outlier]
        r2 = random.Random(SEED * 7 + ei * 101 + n)
        r2.shuffle(items)
        docs = [{"title": None, "text": c["body"]} for c, _ in items]
        gold = [i for i, (_, is_out) in enumerate(items) if is_out]
        # distinct majority topics in this rung (for meta / sanity)
        maj_titles = sorted({c["title"] for c in maj})
        per_rung[n] = {
            "documents": docs,
            "queries": [TOPIC_QUERY],
            "answers": ["; ".join(str(g + 1) for g in sorted(gold))],
            "gold_doc_indices": gold,
            "source": "wiki_outlier_topic",
            "meta": {
                "majority_label": None,          # mixed-majority (v2)
                "minority_label": outlier[0]["title"],
                "num_outliers": num_outliers,
                "num_categories": len(maj_titles) + 1,
                "num_docs": len(docs),
                "mode": "mixed",
                "rung": lab,
            },
        }
    return per_rung


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=600)
    ap.add_argument("--num-outliers", type=int, default=3)
    ap.add_argument("--min-run", type=int, default=4,
                    help="min chunks per majority article (must be > num_outliers)")
    ap.add_argument("--max-run", type=int, default=14,
                    help="max chunks per majority article")
    ap.add_argument("--pool-cache",
                    default="data/wiki100w_article_pool_smoke.pkl")
    ap.add_argument("--out-root",
                    default="/scratch/users/prasann/cpt_data/eval500_v2")
    ap.add_argument("--rungs", default=None,
                    help="Override the default 3k/8k/16k/32k (n=22/55/110/220) ladder. "
                         "Comma-separated n:label pairs, e.g. "
                         "'14:2048,28:4096,57:8192,111:16384,220:32768' -- lets this "
                         "scale-K-preserving builder target the CTC-suite's token-calibrated "
                         "doc counts instead of the original wiki100w v2 ladder's own labels.")
    ap.add_argument("--out-tmpl", default=None,
                    help="Override OUT_TMPL. Supports {n} (doc count) and {lab} (rung label) "
                         "placeholders, e.g. 'rung_{lab}.jsonl' to match the CTC-suite's "
                         "eval_rungs/<task>/rung_<tokens>.jsonl naming.")
    args = ap.parse_args()

    global RUNGS, OUT_TMPL
    if args.rungs:
        RUNGS = {}
        for pair in args.rungs.split(","):
            n_str, lab = pair.split(":")
            RUNGS[int(n_str)] = lab
    if args.out_tmpl:
        OUT_TMPL = args.out_tmpl

    print(f"Loading article pool {args.pool_cache} ...")
    pool = ArticlePool(args.pool_cache)
    print(f"  pool: {len(pool):,} articles, max {pool.max_chunks} chunks/article")

    rung_rows = {n: [] for n in RUNGS}
    ei = 0
    built = 0
    while built < args.num_examples:
        per = build_example(pool, ei, args.num_outliers, args.min_run, args.max_run)
        ei += 1
        if per is None:
            continue
        for n in RUNGS:
            rung_rows[n].append(per[n])
        built += 1
        if built % 100 == 0:
            print(f"  built {built}/{args.num_examples}")

    out_dir = os.path.join(args.out_root, "outlier")
    os.makedirs(out_dir, exist_ok=True)
    for n, lab in RUNGS.items():
        path = os.path.join(out_dir, OUT_TMPL.format(n=n, lab=lab))
        rows = rung_rows[n]
        save_jsonl(path, rows)
        nd = [len(r["documents"]) for r in rows]
        ncat = [r["meta"]["num_categories"] for r in rows]
        print(f"  [{lab:>3}] target n={n:<4} {len(rows)} ex "
              f"ndocs[min/med/max]={min(nd)}/{sorted(nd)[len(nd)//2]}/{max(nd)} "
              f"cats[med]={sorted(ncat)[len(ncat)//2]} -> {path}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
