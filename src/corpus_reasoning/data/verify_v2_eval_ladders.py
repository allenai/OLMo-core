#!/usr/bin/env python3
"""Verify the v2 eval ladders: >=500/rung, identical questions across rungs,
index-remap integrity, distractor nesting, and (optional) Qwen token lengths."""
import argparse
import glob
import json
import os

V2 = "/scratch/users/prasann/cpt_data/eval500_v2"

TASKS = {
    "contra": {"gold_pairs": True, "gold_field": "gold_doc_indices",
               "glob": "contra/contradiction_eval_pubmed_both_n*_k3.jsonl"},
    "nq": {"gold_pairs": False, "gold_field": "gold_doc_indices",
           "extra": ["hard_neg_indices"], "glob": "nq/nq_validation_k*_600.jsonl"},
    "outlier": {"gold_pairs": False, "gold_field": "gold_doc_indices",
                "answers_from_gold": True,
                "glob": "outlier/outlier_wiki100w_n*_k3_eval_600.jsonl"},
    "rerank": {"gold_pairs": False, "gold_field": "gold_doc_indices",
               "extra": ["hard_neg_indices"], "has_ce": True,
               "glob": "rerank/msmarco_trainhn_eval_k*_500.jsonl"},
}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def ndocs_of(name):  # extract the integer n from a filename for sorting
    import re
    m = re.search(r"[nk](\d+)", os.path.basename(name))
    return int(m.group(1)) if m else 0


def gold_texts(ex, cfg):
    docs = ex["documents"]
    gf = ex[cfg["gold_field"]]
    idxs = [i for pair in gf for i in pair] if cfg.get("gold_pairs") else list(gf)
    return sorted(docs[i]["text"] for i in idxs)


def query_key(ex):
    q = ex.get("queries") or []
    return q[0] if q else None


def verify_task(task, cfg, tok=None):
    files = sorted(glob.glob(os.path.join(V2, cfg["glob"])), key=ndocs_of)
    assert files, f"{task}: no v2 files matched {cfg['glob']}"
    rungs = {ndocs_of(f): load(f) for f in files}
    ns = sorted(rungs)
    print(f"\n### {task}: rungs n={ns}")
    problems = []

    # (1) counts
    for n in ns:
        c = len(rungs[n])
        flag = "" if c >= 500 else "  <<< BELOW 500"
        print(f"    n={n:<4} {c} examples{flag}")
        if c < 500:
            problems.append(f"n={n} has {c}<500")

    counts = {len(rungs[n]) for n in ns}
    if len(counts) != 1:
        problems.append(f"rung counts differ: {counts}")
    m = min(len(rungs[n]) for n in ns)

    # (2) per-example identity across rungs + (3) index integrity
    base_n = ns[-1]  # largest rung
    bad_gold_text = bad_query = bad_idx = bad_ans = bad_nest = bad_ce = 0
    for ei in range(m):
        gt_ref = gold_texts(rungs[base_n][ei], cfg)
        q_ref = query_key(rungs[base_n][ei])
        distr_sets = {}
        for n in ns:
            ex = rungs[n][ei]
            nd = len(ex["documents"])
            # index bounds
            gf = ex[cfg["gold_field"]]
            flat = [i for pair in gf for i in pair] if cfg.get("gold_pairs") else list(gf)
            if any(i < 0 or i >= nd for i in flat):
                bad_idx += 1
            for f in cfg.get("extra", []):
                if any(i < 0 or i >= nd for i in ex.get(f, [])):
                    bad_idx += 1
            if cfg.get("has_ce") and len(ex.get("ce_scores", [])) != nd:
                bad_ce += 1
            # gold-text identity
            if gold_texts(ex, cfg) != gt_ref:
                bad_gold_text += 1
            # query identity
            if query_key(ex) != q_ref:
                bad_query += 1
            # outlier answers recomputed from gold positions
            if cfg.get("answers_from_gold"):
                want = "; ".join(str(i + 1) for i in sorted(gf))
                if (ex.get("answers") or [None])[0] != want:
                    bad_ans += 1
            # distractor set (non-gold texts) for nesting check
            goldset = set(flat)
            distr_sets[n] = frozenset(
                d["text"] for i, d in enumerate(ex["documents"]) if i not in goldset)
        # (5) nesting: smaller rung distractors subset of next-larger
        for a, b in zip(ns, ns[1:]):
            if not distr_sets[a] <= distr_sets[b]:
                bad_nest += 1

    def ok(x):
        return "OK" if x == 0 else f"FAIL({x})"
    print(f"    gold-text identical across rungs: {ok(bad_gold_text)}")
    print(f"    query identical across rungs:     {ok(bad_query)}")
    print(f"    gold/hardneg indices in-range:    {ok(bad_idx)}")
    if cfg.get("answers_from_gold"):
        print(f"    outlier answers match gold pos:   {ok(bad_ans)}")
    print(f"    distractor nesting (subset):      {ok(bad_nest)}")
    if cfg.get("has_ce"):
        print(f"    ce_scores length == #docs:        {ok(bad_ce)}")
    for nm, v in [("gold-text", bad_gold_text), ("query", bad_query),
                  ("idx", bad_idx), ("ans", bad_ans), ("nest", bad_nest), ("ce", bad_ce)]:
        if v:
            problems.append(f"{nm} mismatch x{v}")

    # (6) token lengths
    if tok is not None:
        def render(ex):
            parts = []
            for i, d in enumerate(ex["documents"]):
                t = d.get("title")
                parts.append((f"{t}\n" if t else "") + d.get("text", ""))
            q = query_key(ex) or ""
            return "\n\n".join(parts) + "\n\n" + q
        for n in ns:
            import statistics
            sample = rungs[n][:40]
            lens = [len(tok(render(ex))["input_ids"]) for ex in sample]
            print(f"    n={n:<4} median tokens ~ {int(statistics.median(lens)):>6} "
                  f"(min {min(lens)}, max {max(lens)}, sample={len(sample)})")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="contra,nq,outlier,rerank")
    ap.add_argument("--tokens", action="store_true", help="measure Qwen token lengths")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    args = ap.parse_args()
    tok = None
    if args.tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    allprob = {}
    for t in args.tasks.split(","):
        t = t.strip()
        allprob[t] = verify_task(t, TASKS[t], tok)
    print("\n=== SUMMARY ===")
    bad = False
    for t, probs in allprob.items():
        if probs:
            bad = True
            print(f"  {t}: PROBLEMS -> {probs}")
        else:
            print(f"  {t}: all checks passed")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
