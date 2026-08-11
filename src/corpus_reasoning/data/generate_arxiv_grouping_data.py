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
#
# NOTE on the fix (2026-07): the old `sample_k_for_level` picked k as a pure
# fraction of n_docs with no awareness of how many distinct concept values
# exist at that level, or how many docs each value's pool can actually supply.
# At small n_docs this rarely mattered; at large n_docs (esp. the 32k rung,
# n_docs=176) the *coarse* levels (L0 has only ~19 distinct top-level fields
# in the whole OpenAlex concept taxonomy -- a hard ceiling that MORE DATA
# cannot raise) started requesting k beyond what's feasible, and
# `build_example` silently returned None. The outer retry loop then just
# filled the quota from whichever level happened to still be easy (finer
# levels, with thousands of distinct values), so the realized level-mix and
# gold-cluster granularity silently drifted with N -- conflating "more
# documents" with "harder/finer grouping" across the ladder.
#
# Fix has two parts:
#  1. `sample_k_for_level` now takes the actual `level_idx` and clamps the
#     frac-derived [lo, hi] range to what's *provably buildable*: k can't
#     exceed the number of distinct values, and can't exceed the largest k
#     for which the sum of the k biggest pools covers n_docs (a necessary
#     condition for the capacity-aware partition below to succeed). This
#     makes build_example deterministic-success instead of retry-and-hope,
#     for a given (level, n_docs, level_idx) it never silently drops.
#  2. `build_example` selects group values and partition sizes in a
#     capacity-aware way (biased sample toward larger pools + iterative
#     overflow redistribution) instead of a pure random partition that gets
#     rejected whenever a randomly large slice lands on a small pool.
#  3. (in `main()`/`gen()`) the outer loop now draws a FIXED quota of
#     examples per level (stratified), instead of `rng.choice` + hoping
#     differential per-level success rates happen to cancel out -- which is
#     the second half of why the level-mix drifted (even before the L0
#     ceiling bites, L1/L2/L3 always had lower raw accept rates than L0).

def sample_partition_sizes(total, k, rng, min_per=1):
    """Random partition of `total` into k parts, each >= min_per."""
    assert total >= k * min_per
    extra = total - k * min_per
    sizes = [min_per] * k
    for _ in range(extra):
        sizes[rng.randrange(k)] += 1
    rng.shuffle(sizes)
    return sizes


def _min_k_by_capacity(pool_sizes_desc, n_docs):
    """Smallest k such that the k biggest pools can jointly cover n_docs docs.

    This is a LOWER bound on k, not an upper one: sum(top-k) is
    non-decreasing in k (adding another value only adds capacity), so once
    k reaches this threshold every larger k (up to the eligible-value count)
    remains feasible too -- more groups only ever adds total capacity. The
    binding failure mode for coarse levels is the opposite of what it looks
    like at first glance: too few groups (their pools, even the biggest
    ones, can't jointly hold n_docs), not too many.
    """
    cum = 0
    for i, s in enumerate(pool_sizes_desc, start=1):
        cum += s
        if cum >= n_docs:
            return i
    return 0  # 0 if even ALL eligible values together can't cover n_docs


def sample_k_for_level(level, n_docs, rng, level_idx,
                       k_frac=DEFAULT_K_FRAC_PER_LEVEL, min_per_group=1):
    """Sample k (number of groups) for the given level, clamped to what the
    actual concept-value index can support at this n_docs (see module note).

    Returns None if the level is infeasible for this n_docs at all (e.g. even
    the single largest pool can't supply min_per_group docs, or no k >= 2 is
    buildable) -- callers should treat that as "skip this level this draw",
    not silently retry-until-something-else-works.
    """
    lo_f, hi_f = k_frac[level]
    lo = max(2, int(round(lo_f * n_docs)))
    hi = max(lo, int(round(hi_f * n_docs)))
    hi = min(hi, n_docs // min_per_group)

    pool_sizes_desc = sorted((len(v) for v in level_idx.values()), reverse=True)
    max_k_eligible = len(pool_sizes_desc)
    min_k_capacity = _min_k_by_capacity(pool_sizes_desc, n_docs)
    if min_k_capacity == 0:
        return None  # even every eligible value combined can't cover n_docs
    feasible_lo = max(lo, min_k_capacity)
    feasible_hi = min(hi, max_k_eligible)
    if feasible_lo > feasible_hi:
        # frac target and feasibility don't overlap -- prefer the smallest
        # feasible k over silently failing (keeps the level alive at this N
        # rather than dropping it from the mix; only reachable when the frac
        # target's hi is below what feasibility requires, i.e. an extremely
        # thin pool -- flagged via the caller's "only got X/quota" printout
        # if it also fails downstream).
        feasible_lo = feasible_hi = min_k_capacity if min_k_capacity <= max_k_eligible else None
        if feasible_hi is None:
            return None
    return rng.randint(feasible_lo, feasible_hi)


def _choose_group_values(level_idx, k, n_docs, rng, diversity_mult=3):
    """Pick k distinct concept values with combined capacity >= n_docs.

    Biases toward larger pools (so the pick is very likely capacity-feasible
    on the first try, matching the clamp in `sample_k_for_level`) while still
    giving example-to-example variety: sample from the top
    `min(len(eligible), max(k * diversity_mult, k + 10))` values by pool
    size, weighted by pool size, then verify; fall back to the strict top-k
    (guaranteed feasible per `sample_k_for_level`'s clamp) if the weighted
    draw came up short on capacity.
    """
    values_by_size = sorted(level_idx.keys(), key=lambda v: -len(level_idx[v]))
    if len(values_by_size) < k:
        return None
    candidate_pool = values_by_size[:min(len(values_by_size),
                                          max(k * diversity_mult, k + 10))]
    weights = [len(level_idx[v]) for v in candidate_pool]
    # NOTE: `chosen` must be an order-preserving list, not a set -- `pool.pop(i)`
    # already guarantees no duplicates, and a set's iteration order depends on
    # Python's per-process string-hash randomization (PYTHONHASHSEED), which
    # would silently make output non-deterministic across runs/processes
    # despite a fixed `rng` seed (the resulting group order feeds back into
    # how many further rng.sample/rng.uniform calls happen and in what order).
    chosen = []
    pool = list(zip(candidate_pool, weights))
    for _ in range(k):
        if not pool:
            break
        total_w = sum(w for _, w in pool)
        r = rng.uniform(0, total_w)
        acc = 0.0
        for i, (v, w) in enumerate(pool):
            acc += w
            if acc >= r:
                chosen.append(v)
                pool.pop(i)
                break
        else:
            chosen.append(pool.pop()[0])
    if sum(len(level_idx[v]) for v in chosen) >= n_docs:
        return chosen
    # Weighted draw came up short (can happen near the feasibility boundary)
    # -- fall back to the deterministic, guaranteed-feasible top-k.
    return values_by_size[:k]


def _partition_with_capacity(total, caps, rng, min_per=1):
    """Random partition of `total` into len(caps) parts, each in
    [min_per, caps[i]]. Feasible (and found) whenever sum(caps) >= total and
    min_per * len(caps) <= total: start from an uncapped random partition and
    iteratively move overflow off any over-capacity group onto groups with
    spare room (order determined by `rng`, so still randomized).
    """
    k = len(caps)
    if total < k * min_per or sum(caps) < total:
        return None
    sizes = sample_partition_sizes(total, k, rng, min_per=min_per)
    for _ in range(4 * k + 20):
        over = [i for i in range(k) if sizes[i] > caps[i]]
        if not over:
            return sizes
        i = over[0]
        excess = sizes[i] - caps[i]
        sizes[i] = caps[i]
        under = [j for j in range(k) if sizes[j] < caps[j]]
        rng.shuffle(under)
        for j in under:
            if excess == 0:
                break
            room = caps[j] - sizes[j]
            add = min(room, excess)
            sizes[j] += add
            excess -= add
        if excess > 0:
            sizes[i] += excess  # couldn't place it (shouldn't happen); retry loop
    return None


def build_example(level, level_idx, n_docs, k_groups, rng, source_tag,
                  min_per_group=1):
    """One grouping example at concept level `level` with `k_groups` clusters.

    `k_groups` is expected to already be feasibility-clamped by
    `sample_k_for_level` against this same `level_idx` -- this function still
    defends (returns None) if it isn't, so it's safe to call standalone.
    """
    eligible = list(level_idx.keys())
    if len(eligible) < k_groups:
        return None
    if n_docs < k_groups * min_per_group:
        return None

    chosen = _choose_group_values(level_idx, k_groups, n_docs, rng)
    if chosen is None:
        return None
    caps = [len(level_idx[v]) for v in chosen]
    sizes = _partition_with_capacity(n_docs, caps, rng, min_per=min_per_group)
    if sizes is None:
        return None
    # Capacity-aware: largest size goes to value with largest pool (both
    # already respect each other's cap by construction, this just pairs them
    # sensibly for readability of the resulting label sizes).
    order = sorted(range(k_groups), key=lambda i: -caps[i])
    chosen_sorted = [chosen[i] for i in order]
    sizes_sorted = [sizes[i] for i in order]

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
    ap.add_argument("--eval-compact-in", default=None,
                    help="Optional separate compact pool to draw EVAL examples from "
                         "(still filtered by --eval-year-min). Use this when the "
                         "--compact-in pool's held-out (>= --eval-year-min) slice is too "
                         "small/thin per concept value to support large --docs-per-example "
                         "at coarse levels (L0 has only ~19 distinct values total -- fetch "
                         "a bigger, year-restricted pool with --api-fetch "
                         "--api-year-min/--api-year-max and point this at it). Defaults to "
                         "--compact-in (old behavior: single pool, split by year).")
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
    ap.add_argument("--level-mix", type=float, nargs="+", default=None,
                    help="Fixed fraction of examples to draw from each of --levels, same "
                         "order (must sum to ~1). Default: uniform over --levels. This is "
                         "sampled as an explicit per-level QUOTA -- not `rng.choice` per "
                         "attempt -- so the realized level-mix no longer depends on each "
                         "level's differential accept rate (which is what silently drifted "
                         "the granularity mix across the N ladder before this fix; see the "
                         "module note above `sample_partition_sizes`).")
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
    print(f"loaded {len(papers):,} papers from {args.compact_in}")
    eval_papers = (load_compact(args.eval_compact_in)
                   if args.eval_compact_in else papers)
    if args.eval_compact_in:
        print(f"loaded {len(eval_papers):,} papers from {args.eval_compact_in} (eval-only pool)")

    train_pool = [p for p in papers      if p["year"] <  args.eval_year_min]
    eval_pool  = [p for p in eval_papers if p["year"] >= args.eval_year_min]
    print(f"train pool: {len(train_pool):,}   eval pool: {len(eval_pool):,}")

    print("building per-level concept indices (papers needing all of L0-L3)...")
    train_idx = {lvl: build_level_index(train_pool, lvl,
                                        min_per_value=args.min_per_group)
                 for lvl in args.levels}
    eval_idx  = {lvl: build_level_index(eval_pool,  lvl,
                                        min_per_value=args.min_per_group)
                 for lvl in args.levels}
    for lvl in args.levels:
        train_cap = sum(len(v) for v in train_idx[lvl].values())
        eval_cap  = sum(len(v) for v in eval_idx[lvl].values())
        print(f"  L{lvl}  train values: {len(train_idx[lvl]):>6,} (cap {train_cap:>7,})   "
              f"eval values: {len(eval_idx[lvl]):>6,} (cap {eval_cap:>7,})")

    if args.level_mix is not None:
        if len(args.level_mix) != len(args.levels):
            ap.error("--level-mix must have one weight per --levels entry")
        mix = dict(zip(args.levels, args.level_mix))
    else:
        mix = {lvl: 1.0 / len(args.levels) for lvl in args.levels}
    mix_total = sum(mix.values())
    mix = {lvl: w / mix_total for lvl, w in mix.items()}  # normalize
    print(f"level mix (fixed, independent of N): "
          f"{ {f'L{l}': round(w, 3) for l, w in mix.items()} }")

    def level_quotas(n_examples):
        """Largest-remainder rounding so per-level quotas sum to n_examples exactly."""
        raw = {lvl: mix[lvl] * n_examples for lvl in args.levels}
        quotas = {lvl: int(raw[lvl]) for lvl in args.levels}
        remainder = n_examples - sum(quotas.values())
        # hand out leftover slots to the levels with the largest fractional part
        order = sorted(args.levels, key=lambda l: -(raw[l] - quotas[l]))
        for lvl in order[:remainder]:
            quotas[lvl] += 1
        return quotas

    def gen(per_level, n_examples, tag):
        if n_examples == 0:
            return []
        quotas = level_quotas(n_examples)
        examples = []
        pbar = tqdm(total=n_examples, desc=f"gen {tag}")
        for level, quota in quotas.items():
            got, attempts = 0, 0
            max_attempts = max(200, quota * 20)
            while got < quota and attempts < max_attempts:
                attempts += 1
                k = sample_k_for_level(level, args.docs_per_example, rng,
                                       per_level[level],
                                       min_per_group=args.min_per_group)
                if k is None:
                    continue  # this level is infeasible at this n_docs -- don't spin
                ex = build_example(level, per_level[level],
                                   args.docs_per_example, k,
                                   rng, source_tag=f"openalex_grouping_{tag}",
                                   min_per_group=args.min_per_group)
                if ex is None:
                    continue
                examples.append(ex)
                got += 1
                pbar.update(1)
            if got < quota:
                print(f"  ⚠ L{level} {tag}: only got {got}/{quota} "
                      f"(level infeasible/too-thin at docs-per-example="
                      f"{args.docs_per_example}; see cap printout above)")
        pbar.close()
        rng.shuffle(examples)  # de-block the per-level generation order
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
