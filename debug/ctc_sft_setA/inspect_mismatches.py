"""
Side-by-side of setA train rows and olmo-eval rows for the tasks the IID audit flagged on
something other than document count: rerank (train target self-scores 0.9), qdmatch_nq (train
documents ~1.7x longer at some rungs), grouping (gold cardinality), outlier (train rows lack
``meta``) and xabsence (two-sided vs one-sided). Prints structure, not whole rows.

    python debug/ctc_sft_setA/inspect_mismatches.py --root /weka/.../setA_max20 --vendor /vendor
"""

import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def short(x, n=160):
    s = json.dumps(x, default=str) if not isinstance(x, str) else x
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--vendor", required=True)
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    sys.path.insert(0, args.vendor)
    sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "src"))
    import audit_train_eval_iid as A
    import ctc.tasks as T
    from ctc.format import registry

    T.load_all()
    R = json.load(open(os.path.join(HERE, "olmo_eval_roster.json")))

    def pair(task, bucket, k=8):
        row = R["roster"][A.TASK_TO_ROW[task]]
        return (A._train_rows(args.root, task, bucket, k),
                A._eval_rows(R["hf_dataset"], row["subset"], "r" + bucket, k) or [])

    def scalars(r):
        return {k: short(v, 80) for k, v in r.items() if k not in ("documents",)}

    for task, bucket in [("rerank", "2k"), ("qdmatch_nq", "2k"), ("qdmatch_nq", "16k"),
                         ("grouping_labeled", "16k"), ("outlier", "8k"), ("xabsence", "2k"),
                         ("reorder", "2k"), ("oolong", "8k"), ("strmatch", "2k"),
                         ("contradiction", "2k")]:
        tr, ev = pair(task, bucket)
        print(f"\n################ {task}@{bucket}  (train {len(tr)} / eval {len(ev)})")
        for name, rows in (("TRAIN", tr), ("EVAL", ev)):
            if not rows:
                continue
            r = rows[0]
            print(f"--- {name} fields: {scalars(r)}")
            docs = r.get("documents") or []
            lens = [len(A._doc_text(d)) for d in docs]
            print(f"    docs {len(docs)}; chars min/med/max {min(lens)}/{statistics.median(lens)}/"
                  f"{max(lens)}; doc keys {sorted(docs[0].keys()) if docs and isinstance(docs[0], dict) else type(docs[0]).__name__}")
            for d in docs[:2]:
                print(f"    doc: {short(A._doc_text(d), 200)}")
            if task == "qdmatch_nq":
                # queries interleaved as documents? bimodal lengths would show it
                hist = collections.Counter(min(l // 100, 9) * 100 for l in lens)
                print(f"    length histogram (100-char bins): {sorted(hist.items())}")
        if task == "rerank":
            sp = registry.get("rerank")
            for i, ex in enumerate(tr):
                _, ans = A._render_train(ex, "rerank", "both")
                s = A._self_score(sp, ex, ans)
                if s < 1.0:
                    ce = ex.get("ce_scores")
                    print(f"  train row {i}: self {s:.3f}; target {short(ans, 200)}")
                    print(f"     ce_scores {short(ce, 300)}")
                    print(f"     gold {ex.get('gold_doc_indices')}  n_docs {len(ex['documents'])}")
                    print(f"     metrics {sp.score(sp.parse(ans, len(ex['documents'])), ex)}")
                    break
        if task == "grouping_labeled":
            for name, rows in (("TRAIN", tr), ("EVAL", ev)):
                print(f"  {name} gold shape: " + "; ".join(
                    f"{len(r.get('gold_doc_indices') or [])} groups, k={r.get('k')}, level={r.get('level')}"
                    for r in rows[:4]))


if __name__ == "__main__":
    main()
