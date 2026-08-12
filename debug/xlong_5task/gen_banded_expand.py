"""Generate 2k->256k short-skewed TRAIN pools for nq / outlier / rerank.

Reuses the **audited** expansion core of ``corpus_reasoning/data/build_xlong_rungs.py``
(``build_pool`` + ``expand_example``) so the training data is constructed the same way as the eval
rungs it will be scored against -- same ``self_nongold`` distractor harvesting, same global
gold-text exclusion, same index/ce remapping. The only change is that the target document count is
drawn per example from the banded short-heavy plan in ``bands.py`` instead of being a fixed rung.

Why this and not a fresh retrieval/CE run:
  * ``keep_all=True`` retains every original document -- gold **and** its CE-mined hard negatives
    (nq) or its real CE scores (rerank) -- and only pads with non-gold fillers. So the expensive,
    trap-prone parts (pyserini BM25 over MS MARCO, cross-encoder scoring) are already baked into the
    audited source pool and are preserved exactly.
  * The eval xlong rungs are built this way, so train and eval agree. Fillers carry ``ce=None``,
    which the rerank grader already treats as gain 0 and excludes from the Kendall-tau reference.

⚠ Consequence to report with any rerank number above ~32k: the added negatives are random non-gold
documents, not CE-mined hard negatives. That is an approximation, and it is the *same*
approximation the eval rung makes -- but it means "rerank @128k" measures surfacing CE-relevant
docs among more noise, not among harder noise.
"""

import argparse
import json
import os
import random
import sys
import time

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, f"{REPO}/src/corpus_reasoning/data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_xlong_rungs as xl  # noqa: E402

from bands import band_label, draw_plan, n_for_tokens  # noqa: E402

#: Train-pool configs. Gold conventions mirror ``build_xlong_rungs.XTASKS`` exactly; only the
#: source file changes (train pool instead of the v2 eval canonical).
TRAIN_TASKS = {
    "nq": {
        "src": "/scratch/users/prasann/nq_p10_20k/nq_train_k25-202_clean.jsonl",
        "load_task": "retrieval", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": False, "index_base": 0,
        "extra_index_fields": ["hard_neg_indices"], "keep_all": True,
        "pool": "self_nongold",
    },
    "outlier": {
        "src": "/data/prasann/ctc_suite_data/outlier_pool/"
               "outlier_wiki100w_contin_n14-220_k3_20000.jsonl",
        "load_task": "outlier", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": False, "index_base": 0,
        "extra_index_fields": [], "answers_from_gold": True, "keep_all": True,
        "pool": "self_nongold",
    },
    "rerank": {
        "src": "/data/prasann/ctc_suite_data/rerank_pool/"
               "msmarco_trainhn_train_k20-315_20000.jsonl",
        "load_task": "rerank", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": False, "index_base": 0,
        "extra_index_fields": ["hard_neg_indices"], "has_ce": True, "keep_all": True,
        "pool": "self_nongold",
    },
}


def shrink_example(ex, cfg, tgt_docs, rng):
    """Subset an example down to ``tgt_docs`` documents, always keeping gold.

    ``build_xlong_rungs.expand_example`` only ever GROWS: with ``keep_all=True`` it computes
    ``need = max(0, tgt - len(kept))``, so when the target is smaller than the source example's
    document count it silently keeps everything and the example stays at its natural length. The
    nq / outlier / rerank pools have median 117-166 documents (~14-18k tokens), so without this
    step the entire 2k-16k half of the band plan is unreachable -- the realized distribution piles
    up at 16-32k (measured: 40 instances at 2-4k instead of 6000).

    Priority when choosing what to keep: gold first (never dropped), then the CE-mined hard
    negatives (they carry the task's difficulty), then ordinary distractors.

    :returns: A new example dict with exactly ``min(tgt_docs, len(documents))`` documents and every
        index field / ``ce_scores`` remapped, or ``None`` if the target cannot hold the gold set.
    """
    docs = ex["documents"]
    if tgt_docs >= len(docs):
        return ex

    gidx = sorted(xl.v2.gold_index_set(ex, cfg))
    gold_set = set(gidx)
    if len(gidx) > tgt_docs:
        return None  # target too small to even hold the gold documents

    hard = [i for i in (ex.get("hard_neg_indices") or []) if i not in gold_set]
    hard_set = set(hard)
    rest = [i for i in range(len(docs)) if i not in gold_set and i not in hard_set]
    rng.shuffle(hard)
    rng.shuffle(rest)

    keep = list(gidx)
    for bucket in (hard, rest):
        for i in bucket:
            if len(keep) >= tgt_docs:
                break
            keep.append(i)
    keep = sorted(keep)
    old2new = {old: new for new, old in enumerate(keep)}

    out = {}
    has_ce = cfg.get("has_ce") and "ce_scores" in ex
    base = cfg.get("index_base", 0)
    for k, val in ex.items():
        if k == "documents":
            out[k] = [docs[i] for i in keep]
        elif k == cfg["gold_field"]:
            if cfg.get("gold_is_pairs"):
                out[k] = [sorted(old2new[a] + base for a in pair) for pair in gidx_pairs(ex, cfg)]
            else:
                out[k] = sorted(old2new[i] + base for i in gidx)
        elif k == "hard_neg_indices" and "hard_neg_indices" in cfg.get("extra_index_fields", []):
            out[k] = sorted(old2new[i] for i in (val or []) if i in old2new)
        elif k == "ce_scores" and has_ce:
            out[k] = [val[i] for i in keep]
        else:
            out[k] = val
    return out


def gidx_pairs(ex, cfg):
    """Gold pairs as 0-indexed tuples (only used for pair-gold tasks)."""
    base = cfg.get("index_base", 0)
    return [[a - base for a in pair] for pair in ex[cfg["gold_field"]]]


def load_head(path: str, limit: int) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TRAIN_TASKS))
    ap.add_argument("--src", default="",
                    help="Override TRAIN_TASKS[task]['src']. Needed to band a REBUILT pool: the "
                         "outlier entry points at outlier_wiki100w_contin_n14-220_k3_20000.jsonl, "
                         "whose K is 2-10, while the eval's xlong rungs sit at K=25 for every n. "
                         "The iid rebuild reaches K=24 at n=220 and is the correct seed.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-src", type=int, default=3000,
                    help="source examples loaded to harvest the distractor pool / cycle bases")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(TRAIN_TASKS[args.task])
    if args.src:
        cfg["src"] = args.src
        print(f"[{args.task}] src OVERRIDE -> {args.src}", flush=True)
    plan = draw_plan(args.seed)
    mine = [(i, t) for i, t in enumerate(plan) if i % args.num_shards == args.shard_index]
    print(f"[{args.task}] shard {args.shard_index}/{args.num_shards}: "
          f"{len(mine)} of {len(plan)} examples", flush=True)

    canon = load_head(cfg["src"], args.pool_src)
    # Same normalization build_xlong_rungs.main() applies before expanding: force 0-based indices
    # internally and drop examples whose gold does not survive sanitation.
    for ex in canon:
        xl.v2.normalize_to_zero_indexed(ex, cfg)
    for ex in canon:
        xl.v2.sanitize_gold(ex, cfg)
    canon = [ex for ex in canon if ex[cfg["gold_field"]]]
    print(f"  loaded {len(canon)} source examples from {cfg['src']}", flush=True)

    pool = xl.build_pool(cfg, canon)
    random.Random(args.seed).shuffle(pool)
    max_n = max(n_for_tokens(args.task, t) for _, t in mine) if mine else 0
    print(f"  distractor pool: {len(pool)} distinct non-gold docs; max n_docs needed {max_n}",
          flush=True)
    if len(pool) < max_n:
        raise SystemExit(
            f"distractor pool ({len(pool)}) smaller than the largest target n ({max_n}); "
            f"raise --pool-src (currently {args.pool_src}) so no example repeats a filler"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hist: dict = {}
    n_skipped = 0
    t0 = time.time()
    with open(args.out, "w") as f:
        for j, (gi, target_tok) in enumerate(mine):
            n = n_for_tokens(args.task, target_tok)
            base = canon[gi % len(canon)]
            # SHRINK first when the target is below the source's natural size, then expand.
            # expand_example only grows, so without this the short bands are unreachable.
            shrunk = shrink_example(base, cfg, n, random.Random(args.seed * 7919 + gi))
            if shrunk is None:
                # Target cannot hold this example's gold set; cycle to the next source example
                # rather than emitting a malformed record.
                for step in range(1, len(canon)):
                    cand = canon[(gi + step) % len(canon)]
                    shrunk = shrink_example(cand, cfg, n, random.Random(args.seed * 7919 + gi))
                    if shrunk is not None:
                        break
            if shrunk is None:
                n_skipped += 1
                continue
            base = shrunk
            rec = xl.expand_example(base, cfg, pool, n, xl.off_for(gi, pool, n))
            f.write(json.dumps(rec) + "\n")
            lab = band_label(target_tok)
            hist[lab] = hist.get(lab, 0) + 1
            if (j + 1) % 250 == 0:
                el = time.time() - t0
                print(f"  {j + 1}/{len(mine)}  {el:.0f}s  {el / (j + 1):.2f}s/ex  last_n={n}",
                      flush=True)

    print(f"wrote {len(mine) - n_skipped} examples (skipped {n_skipped}) -> {args.out}", flush=True)
    print(f"band histogram: {json.dumps(hist, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
