"""Generate scientific-abstract grouping data using OpenAlex concept levels.

Task: given N abstracts and an integer k, partition the abstracts into k
groups. The query provides only the document set and k — it does NOT tell
the model which axis to group along. The model has to infer the right
granularity from the corpus and k.

Granularity is controlled at example-construction time via a "level" L in
{L0, L1, L2, L3} from the OpenAlex concept hierarchy. A higher level (L0)
means coarser categories (fewer groups, more abstracts each); a lower
level (L3) means finer categories (more groups, fewer abstracts each).
We only sample papers that have concepts annotated down to L3, so every
paper has a value at all four levels.

For a given example: pick L, sample k distinct concept values at level L,
pull abstracts tagged with each, shuffle, emit. k is drawn from a per-level
range: L0 has 2-3 groups, L3 has 6-10 groups (configurable).

Data source: OpenAlex bulk works snapshot (free, ~300GB; or the smaller
"works" filtered subset). Each work record includes:
    - title, abstract_inverted_index (must be reconstructed)
    - primary_topic: {id, display_name, subfield, field, domain}
    - topics: list of secondary topics with the same shape
    - concepts: list of {display_name, level, score}
    - publication_year

Schema additions vs. unified format:
    - `level`: int in {0, 1, 2, 3}, the OpenAlex concept level used for gold
    - `gold_doc_indices`: list[list[int]] (one sub-list per cluster, 0-indexed)
    - `cluster_labels`: list[str], parallel to gold_doc_indices (model is NOT
       asked to predict labels — they're recorded for inspection/eval only)
    - `queries`: ["Group these documents into k groups."]
    - `answers`: [JSON {"groups": [{"doc_ids": [1-indexed]}, ...]}]

Usage (lightweight: API fetch, ~50MB total, no snapshot needed):
    python scripts/data/generate_arxiv_grouping_data.py \\
        --api-fetch --api-email you@example.com \\
        --api-per-field 2000 --compact-out data/openalex_compact.jsonl
    python scripts/data/generate_arxiv_grouping_data.py \\
        --compact-in data/openalex_compact.jsonl \\
        --num-train 5000 --num-eval 500 \\
        --docs-per-example 20 --clusters-per-example 4

Usage (heavyweight: full snapshot, ~300GB):
    python scripts/data/generate_arxiv_grouping_data.py \\
        --preprocess --openalex-dir data/openalex/works \\
        --compact-out data/openalex_compact.jsonl
"""

import argparse
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

from corpus_reasoning.lib.io import save_jsonl, print_dataset_stats


# ---------- Level configuration ----------

# Levels we consider for grouping. We require every sampled paper to have a
# concept at each of these levels (so any level can be used as the grouping
# axis with no missing data).
LEVELS = [0, 1, 2, 3]

# Per-level k range, expressed as (min_frac, max_frac) of n_docs.
# Higher level -> coarser -> fewer groups; finer levels can go up to ~all-singletons.
# Absolute minimum of 2 groups is enforced; max is capped at n_docs // 2 so
# every group still has >=2 docs (single-doc groups are degenerate).
DEFAULT_K_FRAC_PER_LEVEL = {
    0: (0.05, 0.15),   # ~2-3 of 20, ~5-15 of 100
    1: (0.10, 0.30),   # ~3-5 of 20, ~10-30 of 100
    2: (0.10, 0.85),   # ~2-17 of 20, ~10-85 of 100
    3: (0.25, 0.95),   # ~5-19 of 20, ~25-95 of 100 (near-singleton clusters)
}

QUERY_PHRASINGS = [
    "Group these documents into {k} groups.",
    "Partition the following abstracts into {k} categories.",
    "Cluster these {n} papers into {k} groups.",
]


# ---------- OpenAlex preprocessing ----------

def reconstruct_abstract(inv_index):
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the string."""
    if not inv_index:
        return ""
    pos_to_word = {}
    for word, positions in inv_index.items():
        for p in positions:
            pos_to_word[p] = word
    return " ".join(pos_to_word[i] for i in sorted(pos_to_word))


def project_work(rec, max_words=120):
    """Pull the fields we need; drop bad rows. Returns None if unusable."""
    abs_idx = rec.get("abstract_inverted_index")
    abstract = reconstruct_abstract(abs_idx)
    if len(abstract.split()) < 30:
        return None
    title = (rec.get("title") or "").strip()
    if not title:
        return None
    pt = rec.get("primary_topic") or {}
    if not pt.get("display_name"):
        return None
    year = rec.get("publication_year")
    if not isinstance(year, int):
        return None

    # Pick best (highest-score) concept at each level for stable axis values.
    concepts_by_level = defaultdict(list)
    for c in rec.get("concepts", []) or []:
        lvl = c.get("level")
        name = c.get("display_name")
        score = c.get("score", 0.0)
        if lvl is None or not name:
            continue
        concepts_by_level[lvl].append((score, name))
    best_concept = {f"concept_L{lvl}": max(items)[1]
                    for lvl, items in concepts_by_level.items()}

    words = abstract.split()
    if len(words) > max_words:
        abstract = " ".join(words[:max_words]) + " ..."

    return {
        "id":        rec.get("id"),
        "title":     title,
        "abstract":  abstract,
        "year":      year,
        "topic":     pt.get("display_name"),
        "subfield":  (pt.get("subfield") or {}).get("display_name"),
        "field":     (pt.get("field")    or {}).get("display_name"),
        "domain":    (pt.get("domain")   or {}).get("display_name"),
        **best_concept,
    }


def stream_openalex(works_dir):
    """Yield raw work records from OpenAlex .gz JSONL shards."""
    for path in sorted(Path(works_dir).rglob("*.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def preprocess(works_dir, out_path, max_words):
    """One pass over the OpenAlex snapshot -> compact JSONL of usable papers."""
    n_in, n_out = 0, 0
    with open(out_path, "w") as out:
        for rec in tqdm(stream_openalex(works_dir), desc="preprocess"):
            n_in += 1
            p = project_work(rec, max_words=max_words)
            if p is None:
                continue
            out.write(json.dumps(p) + "\n")
            n_out += 1
    print(f"projected {n_out:,} / {n_in:,} works -> {out_path}")


# ---------- API-based fetch (no snapshot needed) ----------

OPENALEX_FIELDS_API = "https://api.openalex.org/fields"
OPENALEX_WORKS_API  = "https://api.openalex.org/works"

# Per-page cap is 200; OpenAlex paginates beyond 10k results only via cursor.
PAGE_SIZE = 200


def api_get(url, params, email):
    params = {**params, "mailto": email}
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenAlex API failed: {url} {params}")


def list_fields(email):
    """Return all ~26 OpenAlex field IDs as short numeric strings (e.g., '17').

    The API returns full URLs ("https://openalex.org/fields/17") but the filter
    endpoint only accepts the short tail.
    """
    data = api_get(OPENALEX_FIELDS_API, {"per_page": 200}, email)
    return [f["id"].rsplit("/", 1)[-1] for f in data["results"]]


def fetch_works_for_field(field_id, n_target, email, year_min, year_max,
                          max_words):
    """Pull up to n_target usable papers from one field via cursor pagination."""
    out = []
    cursor = "*"
    pbar = tqdm(total=n_target, desc=field_id.split("/")[-1], leave=False)
    while len(out) < n_target and cursor:
        params = {
            "filter": (f"primary_topic.field.id:{field_id},"
                       f"has_abstract:true,"
                       f"publication_year:{year_min}-{year_max}"),
            "per_page": PAGE_SIZE,
            "cursor": cursor,
            "select": ("id,title,abstract_inverted_index,publication_year,"
                       "primary_topic,concepts"),
        }
        data = api_get(OPENALEX_WORKS_API, params, email)
        for rec in data["results"]:
            p = project_work(rec, max_words=max_words)
            if p is None:
                continue
            out.append(p)
            pbar.update(1)
            if len(out) >= n_target:
                break
        cursor = data.get("meta", {}).get("next_cursor")
    pbar.close()
    return out


def fetch_via_api(out_path, email, per_field, year_min, year_max, max_words):
    """Stratified sample: per_field papers from each OpenAlex field."""
    fields = list_fields(email)
    print(f"fetching from {len(fields)} fields, {per_field} papers each "
          f"(~{len(fields) * per_field:,} total)")
    seen = set()
    n_written = 0
    with open(out_path, "w") as out:
        for fid in tqdm(fields, desc="fields"):
            for p in fetch_works_for_field(fid, per_field, email,
                                           year_min, year_max, max_words):
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                out.write(json.dumps(p) + "\n")
                n_written += 1
    print(f"wrote {n_written:,} papers -> {out_path}")


def load_compact(path):
    papers = []
    with open(path) as f:
        for line in tqdm(f, desc="load compact"):
            papers.append(json.loads(line))
    return papers


# ---------- Index by level ----------

def has_all_levels(paper, levels=LEVELS):
    return all(paper.get(f"concept_L{lvl}") for lvl in levels)


def build_level_index(papers, level, min_per_value=2):
    """Return {concept_value: [papers]} for one level (filtered to deep papers)."""
    key = f"concept_L{level}"
    idx = defaultdict(list)
    for p in papers:
        if not has_all_levels(p):
            continue
        v = p.get(key)
        if v:
            idx[v].append(p)
    return {v: ps for v, ps in idx.items() if len(ps) >= min_per_value}


# ---------- Example construction ----------

def sample_partition_sizes(total, k, rng, min_per=1):
    """Random partition of `total` into k parts, each >= min_per."""
    assert total >= k * min_per
    extra = total - k * min_per
    sizes = [min_per] * k
    for _ in range(extra):
        sizes[rng.randrange(k)] += 1
    rng.shuffle(sizes)
    return sizes


def sample_k_for_level(level, n_docs, rng,
                       k_frac=DEFAULT_K_FRAC_PER_LEVEL, min_per_group=1):
    """Sample k (number of groups) for the given level."""
    lo_f, hi_f = k_frac[level]
    lo = max(2, int(round(lo_f * n_docs)))
    hi = min(n_docs // min_per_group, max(lo, int(round(hi_f * n_docs))))
    return rng.randint(lo, hi)


def build_example(level, level_idx, n_docs, k_groups, rng, source_tag,
                  min_per_group=1):
    """One grouping example at concept level `level` with `k_groups` clusters."""
    eligible = list(level_idx.keys())
    if len(eligible) < k_groups:
        return None
    if n_docs < k_groups * min_per_group:
        return None
    chosen = rng.sample(eligible, k_groups)
    sizes = sample_partition_sizes(n_docs, k_groups, rng, min_per=min_per_group)
    # Capacity-aware: largest size goes to value with largest pool.
    sizes_sorted   = sorted(sizes, reverse=True)
    chosen_sorted  = sorted(chosen, key=lambda v: -len(level_idx[v]))

    docs, gold_clusters, labels = [], [], []
    for value, size in zip(chosen_sorted, sizes_sorted):
        pool = level_idx[value]
        if len(pool) < size:
            return None
        picks = rng.sample(pool, size)
        idxs = []
        for p in picks:
            idxs.append(len(docs))
            docs.append({"title": p["title"], "text": p["abstract"]})
        gold_clusters.append(idxs)
        labels.append(value)

    perm = list(range(n_docs))
    rng.shuffle(perm)
    inv = {old: new for new, old in enumerate(perm)}
    docs = [docs[old] for old in perm]
    gold_clusters = [sorted(inv[i] for i in c) for c in gold_clusters]

    query = rng.choice(QUERY_PHRASINGS).format(k=k_groups, n=n_docs)
    answer = json.dumps({
        "groups": [
            {"doc_ids": [i + 1 for i in idxs]}
            for idxs in gold_clusters
        ]
    })

    return {
        "documents": docs,
        "queries": [query],
        "answers": [answer],
        "gold_doc_indices": gold_clusters,
        "cluster_labels": labels,
        "level": level,
        "k": k_groups,
        "source": source_tag,
    }


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", action="store_true",
                    help="Snapshot -> compact JSONL projection and exit.")
    ap.add_argument("--api-fetch", action="store_true",
                    help="Fetch papers via OpenAlex API (no snapshot needed).")
    ap.add_argument("--openalex-dir", default="data/openalex/works",
                    help="Directory of OpenAlex .gz work shards (snapshot mode).")
    ap.add_argument("--compact-out", default="data/openalex_compact.jsonl")
    ap.add_argument("--compact-in",  default="data/openalex_compact.jsonl")
    ap.add_argument("--max-abstract-words", type=int, default=120)
    ap.add_argument("--api-email", default=None,
                    help="Your email for OpenAlex polite-pool (required for --api-fetch).")
    ap.add_argument("--api-per-field", type=int, default=2000,
                    help="Papers per field. ~26 fields x 2000 = ~52k papers (~50MB).")
    ap.add_argument("--api-year-min", type=int, default=2018)
    ap.add_argument("--api-year-max", type=int, default=2025)

    ap.add_argument("--num-train", type=int, default=5000)
    ap.add_argument("--num-eval",  type=int, default=500)
    ap.add_argument("--docs-per-example", type=int, default=20)
    ap.add_argument("--levels", nargs="+", type=int, default=LEVELS,
                    help="Concept levels to sample examples from.")
    ap.add_argument("--min-per-group", type=int, default=1,
                    help="Smallest allowed cluster size (1 = singletons allowed).")
    ap.add_argument("--eval-year-min", type=int, default=2024,
                    help="Papers from this year onward go to eval (leak prevention).")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.preprocess:
        preprocess(args.openalex_dir, args.compact_out, args.max_abstract_words)
        return

    if args.api_fetch:
        if not args.api_email:
            ap.error("--api-fetch requires --api-email (OpenAlex polite-pool requirement)")
        fetch_via_api(args.compact_out, args.api_email, args.api_per_field,
                      args.api_year_min, args.api_year_max, args.max_abstract_words)
        return

    rng = random.Random(args.seed)
    papers = load_compact(args.compact_in)
    print(f"loaded {len(papers):,} papers")

    train_pool = [p for p in papers if p["year"] <  args.eval_year_min]
    eval_pool  = [p for p in papers if p["year"] >= args.eval_year_min]
    print(f"train pool: {len(train_pool):,}   eval pool: {len(eval_pool):,}")

    print("building per-level concept indices (papers needing all of L0-L3)...")
    train_idx = {lvl: build_level_index(train_pool, lvl,
                                        min_per_value=args.min_per_group)
                 for lvl in args.levels}
    eval_idx  = {lvl: build_level_index(eval_pool,  lvl,
                                        min_per_value=args.min_per_group)
                 for lvl in args.levels}
    for lvl in args.levels:
        print(f"  L{lvl}  train values: {len(train_idx[lvl]):>6,}   "
              f"eval values: {len(eval_idx[lvl]):>6,}")

    def gen(per_level, n_examples, tag):
        examples, attempts = [], 0
        pbar = tqdm(total=n_examples, desc=f"gen {tag}")
        while len(examples) < n_examples and attempts < n_examples * 20:
            attempts += 1
            level = rng.choice(args.levels)
            k = sample_k_for_level(level, args.docs_per_example, rng,
                                   min_per_group=args.min_per_group)
            ex = build_example(level, per_level[level],
                               args.docs_per_example, k,
                               rng, source_tag=f"openalex_grouping_{tag}",
                               min_per_group=args.min_per_group)
            if ex is None:
                continue
            examples.append(ex)
            pbar.update(1)
        pbar.close()
        return examples

    train = gen(train_idx, args.num_train, "train")
    evals = gen(eval_idx,  args.num_eval,  "eval")

    base = f"openalex_grouping_n{args.docs_per_example}_levels"
    train_path = Path(args.out_dir) / f"{base}_train_{args.num_train}.jsonl"
    eval_path  = Path(args.out_dir) / f"{base}_eval_{args.num_eval}.jsonl"
    save_jsonl(train_path, train)
    save_jsonl(eval_path, evals)
    print_dataset_stats(train, "train", train_path)
    print_dataset_stats(evals, "eval", eval_path)
    print(f"wrote {train_path}\nwrote {eval_path}")


if __name__ == "__main__":
    main()
