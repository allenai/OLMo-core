"""Convert our CE-scored MS MARCO pools into HELMET-native rerank jsonl.

HELMET's `load_msmarco_rerank` (data.py) consumes one record per query:

    {"qid": ..., "query": ..., "ctxs": [{"id": ..., "text": ..., "label": ...}, ...]}

and builds the prompt with explicit per-doc IDs, a 2-shot demo block, and a
full-permutation "Ranking: ID3 > ID1 > ..." target; relevance is the graded
integer `label` (TREC-style 0-3 — the "dep3" in HELMET's filenames), gold order
= label descending, metric = graded NDCG@10. See results/msmarco_rerank.md.

This script reformats the unified CE-scored files produced by
generate_msmarco_trainhn_data.py (each doc already carries a cross-encoder
`ce_scores` value, model cross-encoder/ms-marco-MiniLM-L-6-v2) into that schema:

  * label  = CE score bucketed to 0-3 by sigmoid-probability thresholds
             (default 0.1/0.5/0.8). CE = relevance truth, consistent with the
             in-repo `rerank` task.
  * id     = a per-example random permutation of 1..K rendered as strings, so the
             displayed order carries no positional signal about relevance.
  * qid    = stable 12-hex hash of the query (scopes the per-record ids; lets
             HELMET drop demos that share a test query).
  * title  = OMITTED (MS MARCO passages are untitled) so HELMET uses its
             "[ID: {id}] Document: {text}" passage template.

The output drops straight into HELMET as test_files / demo_files, and is read
in-repo by `evaluate.py --task rerank_helmet` (and trainable on the identical
template). No GPU and no cross-encoder run here — the scores already exist.

Usage (see jobs/gen_msmarco_helmet_rerank.sh):
    python scripts/data/generate_msmarco_helmet_rerank_data.py \
        --inputs data/msmarco_trainhn_train_k20_2000.jsonl ... \
        --out-dir data --demo-k 10
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random


def ce_to_grade(ce, thresholds):
    """Bucket a cross-encoder logit into a graded relevance label in 0..len(thr).

    Bucketing is on the sigmoid probability so the cutoffs are interpretable
    (default 0.1/0.5/0.8 -> grades 1/2/3); a None score (unscored) -> 0."""
    if ce is None:
        return 0
    p = 1.0 / (1.0 + math.exp(-ce))
    g = 0
    for t in thresholds:
        if p >= t:
            g += 1
    return g


def _qid(query):
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]


def to_helmet_record(ex, idx, thresholds, seed):
    """Unified CE-scored example -> HELMET-native {qid, query, ctxs[id,text,label]}."""
    docs = ex["documents"]
    ce = ex.get("ce_scores") or [None] * len(docs)
    k = len(docs)
    rng = random.Random(seed + idx)
    ids = list(range(1, k + 1))
    rng.shuffle(ids)
    # `score` = the raw cross-encoder logit — the ground-truth relevance signal:
    # both the rerank TARGET order and our in-repo NDCG (reciprocal CE rank) come
    # from it, with no bucketing. `label` = a coarse 0-3 bucket kept ONLY so the
    # files stay loadable by HELMET's own `load_msmarco_rerank` (which requires an
    # integer label); our `--task rerank_helmet` metric ignores it. HELMET in turn
    # ignores the extra `score` field (it reads only id/title/text/label).
    ctxs = [{"id": str(ids[j]),
             "text": docs[j]["text"],
             "label": ce_to_grade(ce[j], thresholds),
             "score": round(float(ce[j]), 4) if ce[j] is not None else -1e9}
            for j in range(k)]
    return {"qid": _qid(ex["queries"][0]), "query": ex["queries"][0], "ctxs": ctxs}


def _label_hist(records):
    h = {}
    for r in records:
        for c in r["ctxs"]:
            h[c["label"]] = h.get(c["label"], 0) + 1
    return dict(sorted(h.items()))


def convert_file(path, thresholds, seed):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                records.append(to_helmet_record(json.loads(line), i, thresholds, seed))
    return records


def _out_name(in_path):
    """data/msmarco_trainhn_eval_k50_500.jsonl -> msmarco_helmet_rerank_dev_k50_500.jsonl

    'train' stays 'train'; 'eval' is renamed 'dev' (HELMET terminology)."""
    base = os.path.basename(in_path).replace("msmarco_trainhn_", "")
    base = base.replace("eval_", "dev_", 1)
    return "msmarco_helmet_rerank_" + base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Unified CE-scored jsonl files (msmarco_trainhn_*).")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--grade-thresholds", default="0.1,0.5,0.8",
                    help="Ascending sigmoid-prob cutoffs for grades 1..N (default "
                         "0.1,0.5,0.8 -> 0-3).")
    ap.add_argument("--seed", type=int, default=42)
    # Few-shot demo file (HELMET uses a small-pool demo file; default k=10).
    ap.add_argument("--demo-from", default="",
                    help="Input file to build the demo pool from (default: the "
                         "smallest-k train input).")
    ap.add_argument("--demo-k", type=int, default=10,
                    help="Pool size per demo record (HELMET uses 10).")
    ap.add_argument("--demo-n", type=int, default=300,
                    help="Number of demo records to emit.")
    ap.add_argument("--no-demos", action="store_true")
    args = ap.parse_args()

    thresholds = sorted(float(t) for t in args.grade_thresholds.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    # Expand any globs the shell didn't.
    inputs = []
    for p in args.inputs:
        inputs.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    print(f"grade thresholds (sigmoid p): {thresholds} -> grades 0..{len(thresholds)}")
    for in_path in inputs:
        records = convert_file(in_path, thresholds, args.seed)
        out_path = os.path.join(args.out_dir, _out_name(in_path))
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        k = len(records[0]["ctxs"]) if records else 0
        print(f"{in_path} -> {out_path}: {len(records)} recs, k={k}, "
              f"labels {_label_hist(records)}")

    if args.no_demos:
        return
    # Demo file: subsample each train pool to demo-k (deterministic), keeping the
    # graded labels. Mirrors HELMET's small-pool demo_files.
    demo_src = args.demo_from
    if not demo_src:
        trains = [p for p in inputs if "_train_" in os.path.basename(p)]
        demo_src = min(trains or inputs,
                       key=lambda p: len(json.loads(open(p).readline())["documents"]))
    src_records = convert_file(demo_src, thresholds, args.seed)
    rng = random.Random(args.seed + 1)
    demos = []
    for r in src_records[:args.demo_n]:
        ctxs = r["ctxs"]
        sample = ctxs if len(ctxs) <= args.demo_k else rng.sample(ctxs, args.demo_k)
        demos.append({"qid": r["qid"], "query": r["query"], "ctxs": sample})
    demo_path = os.path.join(args.out_dir,
                             f"msmarco_helmet_rerank_demos_k{args.demo_k}.jsonl")
    with open(demo_path, "w") as f:
        for d in demos:
            f.write(json.dumps(d) + "\n")
    print(f"{demo_src} -> {demo_path}: {len(demos)} demo recs, k={args.demo_k}, "
          f"labels {_label_hist(demos)}")


if __name__ == "__main__":
    main()
