"""Generate OBLIQ-Bench in-context retrieval data.

OBLIQ-Bench (https://huggingface.co/datasets/dianetc/OBLIQ-Bench) ships as
BEIR-style triples (corpus.jsonl + queries.jsonl + qrels.tsv) over five
subsets. The OBLIQ paper evaluates large-corpus retrievers (NDCG@10 etc.),
but this codebase trains/evaluates *in-context* retrieval: ~100 documents
stuffed into a prompt with the model picking the gold by 1-indexed ID.

This script bridges the two. Per query it:
  1. BM25-searches the OBLIQ subset's own corpus (not wikipedia-dpr-100w)
     via `scripts.lib.bm25_local.LocalBM25Searcher` to mine candidate
     distractors.
  2. Forces all gold passages (from `qrels.tsv` with score > 0) into the
     example, filters out per_query_excluded_ids (self-source masking
     for writing / math), and fills the remaining doc slots from the
     BM25 hit list.
  3. Shuffles documents and emits a unified-format example matching
     `scripts.lib.data_format` (compatible with `--task retrieval` /
     `cot_retrieval` everywhere downstream).

Train / test split. Queries are split deterministically: `--train-frac`
(default 10%) become training queries; the rest form the held-out test
split. Each training query produces `--num-train-perms` examples, where
permutation i draws distractors from BM25 rank window
[i * rank_slice_step, i * rank_slice_step + need]. This diversifies both
content (different hard negatives) and position (independent shuffle per
permutation). Test queries get a single canonical example each (top BM25
distractors, single shuffle).

Usage (run inside `corpus-reasoning-eval` env — needs pyserini):
    python scripts/data/generate_obliq_data.py \\
        --subset writing --num-docs 100 --num-train-perms 3 --train-frac 0.1

Outputs (relative to --output-dir, default data/):
    obliq_<subset>_train_k<K>_p<P>_<count>.jsonl
    obliq_<subset>_test_k<K>_<count>.jsonl
"""

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

# NOTE: `from corpus_reasoning.lib.bm25_local import LocalBM25Searcher` is
# deliberately NOT at module top -- pyserini spins up an Anserini JVM at import
# time, which hangs/core-dumps on some compute nodes. It's imported lazily inside
# main() (the only place it's used), so importing this module's helpers
# (_load_subset, _assemble_example, ...) elsewhere never triggers the JVM.
from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


HF_BASE = "https://huggingface.co/datasets/dianetc/OBLIQ-Bench/resolve/main"

SUBSET_PATHS = {
    "writing": "analogues/writing",
    "math": "analogues/math",
    "twitter": "descriptive/twitter",
    "wildchat": "descriptive/wildchat",
    "congress": "tip-of-tongue/congress",
}

# Per-subset query framing. The `analogues` subsets (writing, math) ship the
# query as a RAW passage with no instruction, so the retrieval intent (find a
# stylistically / strategically analogous document) is implicit and the query
# just "looks like a document". We prepend an explicit instruction. The
# `descriptive` subsets (twitter, wildchat) already phrase the query as an
# explicit "Find ..." description, so they pass through unframed (prefix "").
# Applied at emit time only — BM25 distractor mining still uses the raw query.
SUBSET_QUERY_PREFIX = {
    "writing": "Find a document written in a similar style and voice to the "
               "following writing passage: ",
    "math": "Find a problem solved with a similar approach or proof strategy "
            "to the following problem: ",
    "congress": "Find the congressional-hearing record that matches the "
                "following recollection: ",
    "twitter": "",
    "wildchat": "",
}


def _download(url, dest):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    r = requests.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def _load_subset(subset, cache_dir):
    """Download (cached) + parse corpus / queries / qrels / excluded_ids."""
    base = SUBSET_PATHS[subset]
    cache_dir = Path(cache_dir) / subset
    corpus_path = _download(
        f"{HF_BASE}/{base}/corpus/corpus.jsonl", cache_dir / "corpus.jsonl")
    queries_path = _download(
        f"{HF_BASE}/{base}/queries%2Bqrels/queries.jsonl",
        cache_dir / "queries.jsonl")
    qrels_path = _download(
        f"{HF_BASE}/{base}/queries%2Bqrels/qrels.tsv",
        cache_dir / "qrels.tsv")
    # Optional — only writing and math have per-query exclusions.
    excluded_path = cache_dir / "per_query_excluded_ids.json"
    try:
        _download(
            f"{HF_BASE}/{base}/queries%2Bqrels/per_query_excluded_ids.json",
            excluded_path)
    except requests.HTTPError:
        excluded_path = None

    corpus = [json.loads(l) for l in open(corpus_path)]
    queries = [json.loads(l) for l in open(queries_path)]

    qrels = defaultdict(list)
    with open(qrels_path) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if i == 0 and line.lower().startswith("query"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], parts[2]
            try:
                if int(score) > 0:
                    qrels[qid].append(cid)
            except ValueError:
                continue

    excluded = {}
    if excluded_path and excluded_path.exists():
        excluded = json.load(open(excluded_path))

    return corpus, queries, dict(qrels), excluded


_WHITESPACE = re.compile(r"\s+")


def _clean_text(text):
    """Flatten OBLIQ snippets to a single line of clean prose.

    OBLIQ source text carries formatting noise (runs of blank lines, tabs,
    trailing spaces, multi-space gaps). We remove ALL newlines and collapse
    every run of whitespace to a single space, so each document/query is one
    continuous line — easier to read in dumps and free of layout artifacts.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _assemble_example(query_text, gold_docs, distractor_docs, source, rng,
                      query_prefix=""):
    """Shuffle gold + distractors, emit a unified-format retrieval example."""
    tagged = ([(d, "gold") for d in gold_docs]
              + [(d, "hard") for d in distractor_docs])
    rng.shuffle(tagged)
    documents = [{"title": None, "text": _clean_text(d["text"])} for d, _ in tagged]
    gold_indices = [i for i, (_, t) in enumerate(tagged) if t == "gold"]
    hard_neg_indices = [i for i, (_, t) in enumerate(tagged) if t == "hard"]
    return {
        "documents": documents,
        "queries": [query_prefix + _clean_text(query_text)],
        "answers": [""],
        "gold_doc_indices": gold_indices,
        "hard_neg_indices": hard_neg_indices,
        "source": source,
    }


def _build_examples(queries, hits_per_query, qrels, excluded, corpus_by_id,
                    num_docs, source, rank_slices, rng, desc, max_gold=0,
                    query_prefix=""):
    """Assemble (query × permutation) examples.

    rank_slices: list of (start, need_offset) tuples. For each (start, _)
    we take distractors from the filtered candidate list at positions
    [start : start + need], where need = num_docs - len(gold_docs).

    max_gold (>0): cap the number of gold docs per example. OBLIQ's `writing`
    subset has 4-19 qrels golds per query, which clutters the context and
    makes the retrieval target an unwieldy 15-ID list. We keep the `max_gold`
    golds that rank highest in the query's own BM25 results (the most clearly
    relevant ones), and backfill the freed slots with distractors so the pool
    stays at `num_docs`. The dropped golds are simply excluded from the
    example — never relabeled as distractors, which would make them false
    negatives.
    """
    examples = []
    skipped = {"no_gold": 0, "no_gold_in_corpus": 0, "insufficient_pool": 0}
    truncated_gold = 0
    capped_gold = 0

    for q, hits in tqdm(list(zip(queries, hits_per_query)), desc=desc):
        qid = str(q["_id"])
        gold_ids = qrels.get(qid, [])
        if not gold_ids:
            skipped["no_gold"] += 1
            continue
        gold_ids = [cid for cid in gold_ids if cid in corpus_by_id]
        if not gold_ids:
            skipped["no_gold_in_corpus"] += 1
            continue

        # Exclude ALL original golds from the distractor pool, even the ones a
        # cap drops — a dropped gold relabeled as a distractor is a false
        # negative.
        all_gold_set = set(gold_ids)

        # Cap golds: keep those ranking best in the query's BM25 hits (golds
        # not in the hit list are appended last, in qrels order).
        if max_gold and len(gold_ids) > max_gold:
            hit_rank = {h["docid"]: i for i, h in enumerate(hits)}
            gold_ids = sorted(gold_ids, key=lambda c: hit_rank.get(c, len(hits)))
            gold_ids = gold_ids[:max_gold]
            capped_gold += 1

        gold_docs = [corpus_by_id[cid] for cid in gold_ids]

        excl = set(excluded.get(qid, []))
        # Filter BM25 hits down to the candidate distractor pool.
        cand = [h for h in hits
                if h["docid"] not in all_gold_set
                and h["docid"] not in excl
                and h["docid"] in corpus_by_id]

        # If golds outnumber the doc budget, truncate (rare for writing/congress).
        if len(gold_docs) >= num_docs:
            gold_docs = gold_docs[:num_docs]
            truncated_gold += 1
            need = 0
        else:
            need = num_docs - len(gold_docs)

        for start, _ in rank_slices:
            distractor_hits = cand[start : start + need]
            if len(distractor_hits) < need:
                # Pad from elsewhere in the candidate pool (anywhere not
                # already used in this slice). Avoids dropping the example
                # just because we ran out of slice room.
                used = {h["docid"] for h in distractor_hits}
                extras = [h for h in cand if h["docid"] not in used]
                distractor_hits = distractor_hits + extras[: need - len(distractor_hits)]
            if len(distractor_hits) < need:
                skipped["insufficient_pool"] += 1
                continue
            distractors = [corpus_by_id[h["docid"]] for h in distractor_hits]
            examples.append(_assemble_example(
                q["text"], gold_docs, distractors, source, rng,
                query_prefix=query_prefix,
            ))

    if any(skipped.values()):
        print(f"  Skipped: {skipped}")
    if capped_gold:
        print(f"  Capped gold to {max_gold} for {capped_gold} queries "
              f"(had > {max_gold} qrels golds)")
    if truncated_gold:
        print(f"  Truncated gold for {truncated_gold} queries (>= num_docs golds)")
    return examples


def _print_obliq_stats(examples, label, path):
    """Char-count stats — useful for picking sequence_len in training configs."""
    if not examples:
        print(f"\n{label}: 0 examples")
        return
    n_gold = [len(ex["gold_doc_indices"]) for ex in examples]
    doc_chars = [sum(len(d["text"]) for d in ex["documents"]) for ex in examples]
    q_chars = [len(ex["queries"][0]) for ex in examples]
    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\n{label}: {len(examples)} examples -> {path}")
    print(f"  File size:        {size_mb:.1f} MB")
    print(f"  Golds/example:    median {statistics.median(n_gold):.0f}  "
          f"min {min(n_gold)}  max {max(n_gold)}")
    print(f"  Query chars:      median {statistics.median(q_chars):,.0f}  "
          f"p95 {sorted(q_chars)[int(0.95 * len(q_chars))]:,}")
    print(f"  All-doc chars:    median {statistics.median(doc_chars):,.0f}  "
          f"p95 {sorted(doc_chars)[int(0.95 * len(doc_chars))]:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate OBLIQ-Bench in-context retrieval data")
    parser.add_argument("--subset", choices=list(SUBSET_PATHS), required=True)
    parser.add_argument("--num-docs", type=int, default=100,
                        help="Total documents per example (gold + distractors).")
    parser.add_argument("--max-gold", type=int, default=3,
                        help="Cap gold docs per example at this many (0=no cap); "
                             "keeps the highest BM25-ranked golds and backfills "
                             "distractors so the pool stays at --num-docs. "
                             "OBLIQ writing has 4-19 golds/query by default.")
    parser.add_argument("--num-train-perms", type=int, default=3,
                        help="Permutations per training query (different "
                             "BM25 rank slices, independent shuffles).")
    parser.add_argument("--rank-slice-step", type=int, default=50,
                        help="BM25 rank offset between consecutive train "
                             "permutations.")
    parser.add_argument("--train-frac", type=float, default=0.1,
                        help="Fraction of queries used for training; the "
                             "rest form the held-out test split.")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--cache-dir", type=str, default="data/.cache/obliq",
                        help="Cache for downloaded OBLIQ files.")
    parser.add_argument("--index-dir", type=str,
                        default="data/.cache/obliq/_indices",
                        help="Cache for built Lucene indices.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bm25-threads", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading OBLIQ subset: {args.subset}")
    corpus, queries, qrels, excluded = _load_subset(args.subset, args.cache_dir)
    print(f"  corpus={len(corpus):,}  queries={len(queries)}  "
          f"qrels={len(qrels)}  excluded={len(excluded)}")

    corpus_by_id = {str(doc["_id"]): doc for doc in corpus}

    from corpus_reasoning.lib.bm25_local import LocalBM25Searcher  # lazy: JVM

    index_dir = Path(args.index_dir) / args.subset
    searcher = LocalBM25Searcher(corpus, str(index_dir),
                                 threads=args.bm25_threads)

    # Deterministic shuffle so train/test split is reproducible across runs.
    qsplit = list(queries)
    random.Random(args.seed).shuffle(qsplit)
    n_train = max(1, int(round(args.train_frac * len(qsplit))))
    train_queries = qsplit[:n_train]
    test_queries = qsplit[n_train:]
    print(f"  Split: {len(train_queries)} train queries  "
          f"{len(test_queries)} test queries")

    train_need = args.num_docs  # upper bound (assumes few golds; safe overshoot)
    max_train_rank = (args.num_train_perms - 1) * args.rank_slice_step + train_need
    # Overshoot to absorb gold + excluded filtering.
    train_search_k = max(args.num_docs * 3, max_train_rank + 200)
    test_search_k = max(args.num_docs * 3, args.num_docs + 200)

    print(f"  Train BM25 search k={train_search_k}  Test BM25 search k={test_search_k}")
    train_hits = searcher.batch_search(
        [q["text"] for q in train_queries], k=train_search_k,
        threads=args.bm25_threads)
    test_hits = searcher.batch_search(
        [q["text"] for q in test_queries], k=test_search_k,
        threads=args.bm25_threads)

    query_prefix = SUBSET_QUERY_PREFIX.get(args.subset, "")
    if query_prefix:
        print(f"  Query prefix: {query_prefix!r}")

    train_slices = [(i * args.rank_slice_step, args.num_docs)
                    for i in range(args.num_train_perms)]
    train_examples = _build_examples(
        train_queries, train_hits, qrels, excluded, corpus_by_id,
        args.num_docs, f"obliq_{args.subset}", train_slices, rng,
        desc=f"OBLIQ {args.subset} train", max_gold=args.max_gold,
        query_prefix=query_prefix,
    )

    test_slices = [(0, args.num_docs)]
    test_examples = _build_examples(
        test_queries, test_hits, qrels, excluded, corpus_by_id,
        args.num_docs, f"obliq_{args.subset}", test_slices, rng,
        desc=f"OBLIQ {args.subset} test", max_gold=args.max_gold,
        query_prefix=query_prefix,
    )

    out_dir = Path(args.output_dir)
    train_path = out_dir / (
        f"obliq_{args.subset}_train_k{args.num_docs}"
        f"_p{args.num_train_perms}_{len(train_examples)}.jsonl"
    )
    test_path = out_dir / (
        f"obliq_{args.subset}_test_k{args.num_docs}_{len(test_examples)}.jsonl"
    )
    save_jsonl(str(train_path), train_examples)
    save_jsonl(str(test_path), test_examples)

    print_dataset_stats(train_examples, "Train", str(train_path))
    print_dataset_stats(test_examples, "Test", str(test_path))
    _print_obliq_stats(train_examples, "Train (OBLIQ)", str(train_path))
    _print_obliq_stats(test_examples, "Test (OBLIQ)", str(test_path))


if __name__ == "__main__":
    main()
