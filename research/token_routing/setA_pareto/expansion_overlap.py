"""How new is a fresh-seed setA build? Per (task, bucket), compare its rows with the shipped setA.

A new ``--seed`` gives a new ctc stream, but supply-bound pools (contradiction's gold pairs,
rerank's scored queries, qdmatch_nq) are finite, so a "fresh" row can re-use the answer-bearing
material of a seed-42 row with new distractors -- closer to a second epoch than to new data.
Reported per cell:

* ``rows``: rows in the new build.
* ``question_seen``: fraction whose question text occurs in the reference build (same task, any
  bucket); ``None`` when the task's question is a fixed template (one distinct value).
* ``gold_seen``: fraction whose gold-document set (texts, resolved with the converter's own index
  base) occurs in the reference; ``None`` for tasks without gold.
* ``row_seen``: fraction whose (question, full document list) is identical to a reference row.

    python research/token_routing/setA_pareto/expansion_overlap.py --new <root> --ref <root> \
        --buckets 2k 4k 8k 16k 32k --out overlap.json
"""

import argparse
import hashlib
import json
import os
import sys


def _h(x) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _doc_text(d) -> str:
    return str(d.get("text", "")) if isinstance(d, dict) else str(d)


def _question(ex) -> str:
    for k in ("question", "query", "prompt_question", "instruction"):
        if ex.get(k):
            return str(ex[k])
    return ""


def _keys(ex, base: int):
    from olmo_core.nn.attention.gold_grad_mask import gold_chunks_from_gold_doc_indices

    docs = [_doc_text(d) for d in ex.get("documents") or []]
    q = _question(ex)
    gold = None
    gdi = ex.get("gold_doc_indices")
    if gdi:
        idx = sorted(i for i in gold_chunks_from_gold_doc_indices(gdi, base=base) if 0 <= i < len(docs))
        gold = _h(sorted(docs[i] for i in idx)) if idx else None
    return q, gold, _h([q, docs])


def _rows(root: str, task: str, bucket: str):
    p = os.path.join(root, "per_task", f"_b{bucket}", task, "train.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            ex = json.loads(line)
            out.append(ex["ex"] if "ex" in ex and "documents" not in ex else ex)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--buckets", nargs="+", default=["2k", "4k", "8k", "16k", "32k"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from olmo_core.nn.attention.gold_grad_mask import gold_index_base_for_task

    man = json.load(open(os.path.join(a.new, "manifest.json")))
    report = {}
    for t in man["roster"]:
        name, base = t["name"], gold_index_base_for_task(t["spec"])
        ref_q, ref_g, ref_r = set(), set(), set()
        for b in a.buckets:
            for ex in _rows(a.ref, name, b):
                q, g, r = _keys(ex, base)
                ref_q.add(q); ref_r.add(r)
                if g:
                    ref_g.add(g)
        templ = len(ref_q) <= 1
        for b in a.buckets:
            rows = _rows(a.new, name, b)
            if not rows:
                continue
            ks = [_keys(ex, base) for ex in rows]
            n = len(ks)
            golds = [g for _, g, _ in ks if g]
            cell = {
                "rows": n,
                "question_seen": None if templ else round(sum(q in ref_q for q, _, _ in ks) / n, 4),
                "gold_seen": round(sum(g in ref_g for g in golds) / len(golds), 4) if golds else None,
                "row_seen": round(sum(r in ref_r for _, _, r in ks) / n, 4),
                "distinct_questions_new": len({q for q, _, _ in ks}),
            }
            report[f"{name}@{b}"] = cell
            print(f"{name:14s} {b:>4s} rows={n:6d} q_seen={cell['question_seen']} "
                  f"gold_seen={cell['gold_seen']} row_seen={cell['row_seen']}", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
