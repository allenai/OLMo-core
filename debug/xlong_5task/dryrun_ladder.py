"""Dry-run the ctc_eval v2 ladder against the new eval bundle -- no model, no GPU.

Replicates the LADDERS construction in ``src/scripts/ctc_eval/eval/eval_lc_native.py`` (v2 branch +
the oolong short rungs + the ``--xlong`` block) and reports, per task, which rungs resolve to a file
that actually exists and how many examples each holds. This is the check that the long rungs are
reachable from the easy-use eval commands before any GPU time is spent.
"""

import os

E5 = os.environ.get("EVAL500_ROOT", "/data/prasann/xlong5/eval")

# The v2 base ladder for the 5 tasks (mirrors eval_lc_native.py). Only the entries whose files we
# care about here; the point is the LONG end plus oolong's new short end.
BASE = {
    "contradiction": [
        ("2k", "contra/contradiction_eval_pubmed_both_n100_k3.jsonl"),
        ("8k", "contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
        ("16k", "contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
        ("32k", "contra/contradiction_eval_pubmed_both_n765_k3.jsonl"),
    ],
    "nq": [
        ("3k", "nq/nq_validation_k20_600.jsonl"),
        ("8k", "nq/nq_validation_k50_600.jsonl"),
        ("16k", "nq/nq_validation_k100_600.jsonl"),
        ("32k", "nq/nq_validation_k200_600.jsonl"),
    ],
    "outlier": [
        ("3k", "outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
        ("8k", "outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
        ("16k", "outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
        ("32k", "outlier/outlier_wiki100w_n220_k3_eval_600.jsonl"),
    ],
    "rerank": [
        ("3k", "rerank/msmarco_trainhn_eval_k20_500.jsonl"),
        ("8k", "rerank/msmarco_trainhn_eval_k50_500.jsonl"),
        ("16k", "rerank/msmarco_trainhn_eval_k100_500.jsonl"),
    ],
    "oolong": [
        ("8k", "oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
        ("16k", "oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
        ("32k", "oolong/oolong_test_synth_ctx32768_spliteval.jsonl"),
    ],
}

XL = {
    "contradiction": ("contra", "contradiction_eval_pubmed_both_n*_k3_xlong_{s}.jsonl"),
    "nq": ("nq", "nq_validation_k*_xlong_{s}.jsonl"),
    "outlier": ("outlier", "outlier_wiki100w_n*_k3_eval_xlong_{s}.jsonl"),
    "rerank": ("rerank", "msmarco_trainhn_eval_k*_xlong_{s}.jsonl"),
}
XL_OOLONG = {"64k": 65536, "128k": 131072, "256k": 262144}


def count(p: str) -> int:
    with open(p) as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    import glob

    ladders = {t: [(lab, os.path.join(E5, rel)) for lab, rel in v] for t, v in BASE.items()}

    short = []
    for lab, ctx in (("2k", 2048), ("4k", 4096)):
        p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{ctx}_spliteval.jsonl")
        if os.path.exists(p):
            short.append((lab, p))
    ladders["oolong"] = short + ladders["oolong"]

    for t, (sub, pat) in XL.items():
        for s in ("64k", "128k", "256k"):
            hits = sorted(glob.glob(os.path.join(E5, sub, pat.format(s=s))))
            if hits:
                ladders[t].append((s, hits[0]))
    for s, ctx in XL_OOLONG.items():
        p = os.path.join(E5, "oolong", f"oolong_test_synth_ctx{ctx}_spliteval.jsonl")
        if os.path.exists(p):
            ladders["oolong"].append((s, p))

    print(f"EVAL500_ROOT = {E5}\n")
    total_ok = total_missing = 0
    for t in ("contradiction", "nq", "outlier", "rerank", "oolong"):
        print(f"[{t}]")
        for lab, p in ladders[t]:
            if os.path.exists(p):
                n = count(p)
                flag = "" if n >= 500 else f"  ⚠ eval_size={n} < 500"
                print(f"   {lab:>5}  eval_size={n:<6} {os.path.basename(p)}{flag}")
                total_ok += 1
            else:
                print(f"   {lab:>5}  MISSING       {os.path.basename(p)}")
                total_missing += 1
        print()
    print(f"resolved={total_ok}  missing={total_missing}")


if __name__ == "__main__":
    main()
