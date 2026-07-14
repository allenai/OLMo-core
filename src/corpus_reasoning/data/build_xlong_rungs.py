#!/usr/bin/env python3
"""
Build ULTRA-LONG (64k / 128k / 256k) eval rungs -- the OPT-IN extension of the
v2 long-context ladders (which top out at 32k). These files are NOT built by the
normal pipeline; you run this script by hand and then flip the eval's --xlong
flag (see run_beaker_multirung_eval.sh LADDER_XLONG) to score them.

Design (mirrors build_v2_eval_ladders.py, "expand" mode, no LLM / no network):

  * Question set = the v2 32k canonical file for the task (shared 500 questions,
    gold docs + query + answer already intact/clean).
  * Each example is grown to the xlong doc-count `n` by keeping GOLD docs + query
    + answer fixed and appending distractor documents from a harvested pool, then
    shuffling and remapping every index field (gold_doc_indices / hard_neg_indices /
    outlier answers). Nested-prefix across sizes: 128k's distractors are a superset
    of 64k's for the same example.
  * `n` per size is CALIBRATED at runtime: we tokenize the canonical file through
    the REAL eval load path (load_unified_examples -> Qwen3 chat template) to get
    the true effective tokens/doc, then n = round(target_tokens / eff_tok_per_doc).
    This reproduces the coordinator's measure (median of the full built prompt).

Distractor pools (never inject a doc that is a gold-member anywhere -> would plant
a stray contradiction / a foreign answer):
  * contra  : cross-file harvest of neutral PubMed filler sentences
              (build_v2_eval_ladders.harvest_fillers -- gold-member excluded).
  * nq/outlier : the canonical file's own non-gold documents, deduped, with every
              global gold-member text excluded.

rerank is SKIPPED: it is CE-graded and no CE-scored pool larger than k100 (~16k)
exists, so it cannot be extended past 16k (documented, not built).
oolong is SKIPPED here: it is a single packed document of labeled items, not a
doc pool, so it scales via generate_oolong_ladder_data.py --len-min/--len-max
<target> (the ctx label == token budget). The calibrated "n" for oolong IS the
token budget; this script prints the command instead of building it.

By default this ONLY prints the calibrated counts (a dry calibration). Pass
--sizes / --tasks to actually WRITE files, and keep --count small (default 150)
-- a 256k x 600 file is enormous. Files land in $OUT_ROOT/<task>/ with an
`_xlong` marker in the name so they never collide with the 32k v2 files.

    # calibrate only (no files written) -- prints n for every task/size
    python scripts/data/build_xlong_rungs.py

    # build the proof rung: contradiction @ 64k, 150 examples
    python scripts/data/build_xlong_rungs.py --tasks contra --sizes 64k --count 150

    # build everything cleanly-scalable at 64k+128k (256k is huge -- opt in explicitly)
    python scripts/data/build_xlong_rungs.py --tasks contra,nq,outlier --sizes 64k,128k --count 150
"""
import argparse
import json
import os
import random
import statistics
import sys

# reuse the v2 builder's gold-handling + harvest helpers (no LLM, offline)
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
import build_v2_eval_ladders as v2  # noqa: E402

SIZES = {"64k": 65536, "128k": 131072, "256k": 262144}
E5 = "/scratch/users/prasann/cpt_data/eval500_v2"
OUT_ROOT_DEFAULT = E5

# Per-task config. Reuses v2.TASKS gold conventions; overrides the canonical to
# the v2 32k file (shared-question, clean gold) and forces expand mode.
XTASKS = {
    "contra": {
        "canonical": f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl",
        "load_task": "contradiction", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": True, "index_base": 1,
        "extra_index_fields": [],
        "pool": "contra_harvest",
        "out_tmpl": "contradiction_eval_pubmed_both_n{n}_k3_xlong_{size}.jsonl",
    },
    # keep_all=True => every doc of the v2-32k canonical (gold + its original hard/CE-mined
    # negatives) is RETAINED and only PADDED with random fillers to reach the xlong length.
    # This makes each xlong rung a strict nested SUPERSET of the v2-32k rung for the same
    # question (same gold, same original negatives), so length is the only added variable.
    "nq": {
        "canonical": f"{E5}/nq/nq_validation_k200_600.jsonl",
        "load_task": "retrieval", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": False, "index_base": 0,
        "extra_index_fields": ["hard_neg_indices"], "keep_all": True,
        "pool": "self_nongold",
        "out_tmpl": "nq_validation_k{n}_xlong_{size}.jsonl",
    },
    "outlier": {
        "canonical": f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl",
        "load_task": "outlier", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": False, "index_base": 0,
        "extra_index_fields": [], "answers_from_gold": True, "keep_all": True,
        "pool": "self_nongold",
        "out_tmpl": "outlier_wiki100w_n{n}_k3_eval_xlong_{size}.jsonl",
    },
    # APPROXIMATE rerank xlong: no CE-scored pool > k100 exists, so we KEEP all 100 real-CE
    # docs (gold + CE-scored negatives) and bury them under random-negative fillers that carry
    # ce=None. The grader (_eval_rerank) already treats ce=None as gain 0 for NDCG@k and EXCLUDES
    # them from the Kendall-tau reference set, so NDCG@10 still measures "can the model surface the
    # CE-relevant docs among far more noise" and tau stays defined over the true CE ordering. This
    # is an approximation (the added negatives are un-scored random docs, not CE-mined hard negs).
    "rerank": {
        "canonical": f"{E5}/rerank/msmarco_trainhn_eval_k100_500.jsonl",
        "load_task": "rerank", "qp": "both",
        "gold_field": "gold_doc_indices", "gold_is_pairs": False, "index_base": 0,
        "extra_index_fields": ["hard_neg_indices"], "has_ce": True, "keep_all": True,
        "pool": "self_nongold",
        "out_tmpl": "msmarco_trainhn_eval_k{n}_xlong_{size}.jsonl",
    },
}
SEED = 1234


def _measure_ctx(rows, cfg, tok):
    """Median full-prompt (chat-templated) token length of a set of built examples."""
    import tempfile
    from ctc_eval.eval.evaluate import load_unified_examples
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for r in rows:
            tf.write(json.dumps(r) + "\n")
        tmp = tf.name
    ex = load_unified_examples(tmp, len(rows), task=cfg["load_task"],
                               query_position=cfg["qp"], use_alpaca=False)
    os.unlink(tmp)
    lens = [len(tok(tok.apply_chat_template([{"role": "user", "content": e["prompt"]}],
                    tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)["input_ids"]) for e in ex]
    return statistics.median(lens)


def calibrate(cfg, canon, pool, tok, probe_n=1000, probe_ex=6):
    """Pool-aware calibration: expand a few canonical examples with the SHUFFLED
    xlong pool to probe_n docs, measure the true chat-templated ctx, derive the
    effective tokens/doc of THIS pool, and solve n = round(target / eff). This is
    accurate even when the harvested fillers are shorter than the canonical docs."""
    probe_rows = [expand_example(canon[i], cfg, pool, probe_n, off_for(i, pool, probe_n))
                  for i in range(min(probe_ex, len(canon)))]
    ndocs = statistics.median(len(r["documents"]) for r in probe_rows)
    ctx = _measure_ctx(probe_rows, cfg, tok)
    eff = ctx / ndocs
    ns = {lab: max(len(v2.gold_index_set(canon[0], cfg)) + 1, round(SIZES[lab] / eff))
          for lab in SIZES}
    return int(ndocs), ctx, eff, ns


def gold_texts_of(rows, cfg):
    """Every text that is a gold-member in the canonical (0-indexed already)."""
    out = set()
    for ex in rows:
        for i in v2.gold_index_set(ex, cfg):
            if 0 <= i < len(ex["documents"]):
                out.add(ex["documents"][i].get("text", ""))
    return out


def build_pool(cfg, canon):
    """Distinct distractor docs, excluding every global gold-member text."""
    if cfg["pool"] == "contra_harvest":
        return v2.harvest_fillers({"filler_glob": f"{v2.DATA}/contradiction_*_k3.jsonl",
                                   "gold_field": cfg["gold_field"],
                                   "gold_is_pairs": cfg["gold_is_pairs"]},
                                  gold_texts_of(canon, cfg))
    # self_nongold: pool the canonical file's own non-gold docs, dedup, exclude golds.
    gtext = gold_texts_of(canon, cfg)
    seen, pool = set(), []
    for ex in canon:
        gset = v2.gold_index_set(ex, cfg)
        for i, d in enumerate(ex["documents"]):
            if i in gset:
                continue
            t = d.get("text", "")
            if not t or t in seen or t in gtext:
                continue
            seen.add(t)
            pool.append(d)
    return pool


def off_for(ei, pool, need):
    """Deterministic per-example offset into the (shuffled) pool. FIXED per example
    across sizes so a larger rung's distractors are a superset of a smaller rung's
    (nested prefix): 128k's slice starts at the same offset and just runs longer."""
    if not pool:
        return 0
    return (ei * 7919) % len(pool)


def take_slice(pool, off, need):
    """`need` docs from the shuffled pool starting at `off`, wrapping around the end.
    need <= len(pool) => no within-example repeats (guaranteed by the pool>=need check)."""
    if need <= 0 or not pool:
        return []
    if off + need <= len(pool):
        return pool[off:off + need]
    return pool[off:] + pool[:need - (len(pool) - off)]


def expand_example(ex, cfg, pool, tgt_docs, off):
    """Grow one canonical example to `tgt_docs` documents. `kept` original docs are
    preserved (gold-only for most tasks; ALL original docs for keep_all=rerank so its
    real CE scores survive); the rest are random fillers drawn from the shuffled pool
    at `off`. Everything is shuffled and every index field / ce_scores is remapped.
    Filler docs carry ce=None (random negatives) when the task is CE-graded."""
    rng = random.Random(SEED * 1_000_003 + off)
    docs = ex["documents"]
    gidx = sorted(v2.gold_index_set(ex, cfg))
    has_ce = cfg.get("has_ce") and "ce_scores" in ex
    ce = ex.get("ce_scores") if has_ce else None
    kept = list(range(len(docs))) if cfg.get("keep_all") else gidx
    kept_docs = [docs[i] for i in kept]
    kept_ce = [ce[i] for i in kept] if has_ce else None
    old2keptpos = {old: p for p, old in enumerate(kept)}     # original idx -> position in kept_docs
    need = max(0, tgt_docs - len(kept_docs))
    fillers = [dict(d) for d in take_slice(pool, off, need)]
    combined = kept_docs + fillers
    combined_ce = (kept_ce + [None] * len(fillers)) if has_ce else None
    perm = list(range(len(combined)))
    rng.shuffle(perm)
    new_docs = [combined[p] for p in perm]
    newpos_of_combined = {old: new for new, old in enumerate(perm)}

    def new_pos(old_idx):
        return newpos_of_combined[old2keptpos[old_idx]]

    rec = {}
    for k, val in ex.items():
        if k == "documents":
            rec["documents"] = new_docs
        elif k == cfg["gold_field"]:
            base = cfg.get("index_base", 0)
            if cfg.get("gold_is_pairs"):
                rec[k] = [sorted(new_pos(a) + base for a in pair) for pair in val]
            else:
                rec[k] = sorted(new_pos(a) + base for a in val)
        elif k in cfg.get("extra_index_fields", []):
            continue  # rebuilt below
        elif k == "ce_scores" and has_ce:
            continue  # rebuilt below (parallel array, must follow docs)
        elif k == "answers" and cfg.get("answers_from_gold"):
            continue  # rebuilt below
        else:
            rec[k] = val
    if has_ce:
        rec["ce_scores"] = [combined_ce[p] for p in perm]
    if "hard_neg_indices" in cfg.get("extra_index_fields", []):
        # keep_all (rerank): remap the original hard-neg positions (they are among `kept`).
        # otherwise (nq): canonical hard-negs were dropped distractors -> no hard-neg flags.
        if cfg.get("keep_all"):
            rec["hard_neg_indices"] = sorted(new_pos(i) for i in ex.get("hard_neg_indices", [])
                                             if i in old2keptpos)
        else:
            rec["hard_neg_indices"] = []
    if cfg.get("answers_from_gold"):
        gp = sorted(new_pos(a) for a in gidx)
        rec["answers"] = ["; ".join(str(i + 1) for i in gp)]
    return rec


def build_size(task, cfg, size, n, count, pool, canon, out_root, verbose=True):
    rows_out = [expand_example(ex, cfg, pool, n, off_for(ei, pool, n))
                for ei, ex in enumerate(canon[:count])]
    out_path = os.path.join(out_root, task, cfg["out_tmpl"].format(n=n, size=size))
    v2.save_jsonl(out_path, rows_out)
    if verbose:
        need = n - len(v2.gold_index_set(canon[0], cfg))
        warn = "  [WARN pool<need -> within-example repeats]" if len(pool) < need else ""
        print(f"    [{size:>4}] n={n:<5} {len(rows_out)} ex -> {out_path}"
              f"  (pool={len(pool)}, need/ex={need}){warn}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="contra,nq,outlier,rerank",
                    help="comma list from contra,nq,outlier,rerank. rerank is APPROXIMATE "
                         "(random-negative fillers, ce=None). oolong is not doc-pool scalable.")
    ap.add_argument("--sizes", default="",
                    help="comma list from 64k,128k,256k. EMPTY => calibrate only (no files written).")
    ap.add_argument("--count", type=int, default=300,
                    help="examples per xlong rung (256k x 600 is enormous; 300 balances CI vs disk).")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    for s in sizes:
        if s not in SIZES:
            raise SystemExit(f"unknown size {s}; choose from {list(SIZES)}")

    print(f"=== xlong calibration (count={args.count}, out_root={args.out_root}) ===")
    for task in tasks:
        if task not in XTASKS:
            print(f"skip unknown/unsupported task {task}"); continue
        cfg = XTASKS[task]
        canon = v2.load_jsonl(cfg["canonical"])
        # normalize + sanitize gold exactly like v2 so indexing is uniform 0-based internally
        for ex in canon:
            v2.normalize_to_zero_indexed(ex, cfg)
        for ex in canon:
            v2.sanitize_gold(ex, cfg)
        canon = [ex for ex in canon if ex[cfg["gold_field"]]]
        pool = build_pool(cfg, canon)
        random.Random(SEED).shuffle(pool)   # representative slices + de-cluster by length
        nd, ctx, eff, ns = calibrate(cfg, canon, pool, tok)
        print(f"\n[{task}] canonical={os.path.basename(cfg['canonical'])} "
              f"canon_ndocs={len(canon[0]['documents'])} pool={len(pool)}")
        print(f"        probe({nd} docs)->{ctx:.0f} tok  eff_tok/doc={eff:.2f}   "
              f"calibrated n:  64k={ns['64k']}  128k={ns['128k']}  256k={ns['256k']}")
        if not sizes:
            continue
        for size in sizes:
            build_size(task, cfg, size, ns[size], args.count, pool, canon, args.out_root)

    # oolong + rerank guidance (not built here)
    print("\n[oolong] NOT a doc pool -> regenerate packed-item ladder at the token budget, e.g.:")
    for lab, t in SIZES.items():
        print(f"        {lab}: python scripts/data/generate_oolong_ladder_data.py "
              f"--num-examples {args.count} --len-min {int(t*0.9)} --len-max {t} "
              f"--output-dir {OUT_ROOT_DEFAULT}/oolong")
    print("\nDONE.")


if __name__ == "__main__":
    main()
