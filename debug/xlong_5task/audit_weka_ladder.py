"""Audit the LIVE weka v2 eval bundle: per rung, eval_size + FEVER/wiki contamination + ambiguity.

Runs ON BEAKER (see audit_weka_ladder.sh) because the bundle the evals actually read lives on weka,
unreachable from Berkeley. The local /scratch copies are a DIFFERENT build -- different ``n`` in the
filenames -- so auditing those would answer the wrong question.

Answers, per (task, rung): does it exist, is eval_size >= 500, and is it PubMed-only?

``contra_fever`` is intentionally FEVER and is excluded: it is a separate experimental setting, not
a leak into the PubMed contradiction ladder.
"""

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os

ORDER = ["2k", "3k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M", "2M"]

XL = {  # per-task glob for the size-labelled xlong rungs
    "contra": "contradiction_*_xlong_{s}.jsonl",
    "nq": "nq_*_xlong_{s}.jsonl",
    "outlier": "outlier_*_xlong_{s}.jsonl",
    "rerank": "msmarco_*_xlong_{s}.jsonl",
}
BASE = {
    "contra": ("contra", {"2k": "contradiction_eval_pubmed_both_n100_k3.jsonl",
                          "8k": "contradiction_eval_pubmed_both_n190_k3.jsonl",
                          "16k": "contradiction_eval_pubmed_both_n385_k3.jsonl",
                          "32k": "contradiction_eval_pubmed_both_n765_k3.jsonl"}),
    "nq": ("nq", {"3k": "nq_validation_k20_600.jsonl", "8k": "nq_validation_k50_600.jsonl",
                  "16k": "nq_validation_k100_600.jsonl", "32k": "nq_validation_k200_600.jsonl"}),
    "outlier": ("outlier", {"3k": "outlier_wiki100w_n22_k3_eval_600.jsonl",
                            "8k": "outlier_wiki100w_n55_k3_eval_600.jsonl",
                            "16k": "outlier_wiki100w_n110_k3_eval_600.jsonl",
                            "32k": "outlier_wiki100w_n220_k3_eval_600.jsonl"}),
    "rerank": ("rerank", {"3k": "msmarco_trainhn_eval_k20_500.jsonl",
                          "8k": "msmarco_trainhn_eval_k50_500.jsonl",
                          "16k": "msmarco_trainhn_eval_k100_500.jsonl"}),
    "oolong": ("oolong", {lab: "oolong_test_synth_ctx%d_spliteval.jsonl" % c for lab, c in
                          (("2k", 2048), ("4k", 4096), ("8k", 8192), ("16k", 16384),
                           ("32k", 32768), ("64k", 65536), ("128k", 131072), ("256k", 262144),
                           ("512k", 524288), ("1M", 1048576), ("2M", 2097152))}),
}


def fp(text):
    return hashlib.blake2b(text.strip().encode("utf-8", "replace"), digest_size=8).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/weka/oe-training-default/ai2-llm/checkpoints/"
                                        "prasanns/_eval_bundle_eval500_v2")
    ap.add_argument("--hashes", default="/tmp/fw.txt.gz")
    ap.add_argument("--sample-docs", type=int, default=2500)
    ap.add_argument("--min-eval-size", type=int, default=500)
    args = ap.parse_args()

    with gzip.open(args.hashes, "rt") as f:
        FW = set(line.strip() for line in f if line.strip())
    print("FEVER/wiki fingerprints: %d" % len(FW))
    print("bundle: %s\n" % args.bundle)

    ladders = {}
    for task, (sub, base) in BASE.items():
        rungs = dict(base)
        if task in XL:
            for s in ("64k", "128k", "256k", "512k", "1M", "2M"):
                rungs[s] = XL[task].format(s=s)
        ladders[task] = (sub, rungs)

    print("%-8s %-5s %9s %9s  %s" % ("task", "rung", "eval_size", "fever_%", "status / file"))
    gaps = collections.defaultdict(list)
    for task in ("contra", "nq", "outlier", "rerank", "oolong"):
        sub, rungs = ladders[task]
        for lab in ORDER:
            pat = rungs.get(lab)
            if not pat:
                continue
            hits = sorted(glob.glob(os.path.join(args.bundle, sub, pat)))
            if not hits:
                print("%-8s %-5s %9s %9s  MISSING" % (task, lab, "-", "-"))
                gaps[task].append((lab, "MISSING"))
                continue
            if len(hits) > 1:
                names = ", ".join(os.path.basename(x) for x in hits)
                print("%-8s %-5s %9s %9s  AMBIGUOUS (%d): %s"
                      % (task, lab, "?", "?", len(hits), names))
                gaps[task].append((lab, "AMBIGUOUS"))
                continue
            path = hits[0]
            n = seen = fev = 0
            with open(path) as f:
                for line in f:
                    n += 1
                    if seen >= args.sample_docs:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for d in ex.get("documents", []):
                        t = d.get("text", "")
                        if not t:
                            continue
                        seen += 1
                        if fp(t) in FW:
                            fev += 1
                        if seen >= args.sample_docs:
                            break
            pct = (100.0 * fev / seen) if seen else 0.0
            bad = []
            if n < args.min_eval_size:
                bad.append("eval_size<%d" % args.min_eval_size)
            if pct > 0.5:
                bad.append("FEVER %.1f%%" % pct)
            status = "OK" if not bad else "BAD: " + ", ".join(bad)
            if bad:
                gaps[task].append((lab, status))
            print("%-8s %-5s %9d %8.2f%%  %s  %s"
                  % (task, lab, n, pct, status, os.path.basename(path)))

    print("\n=== GAPS (need building / rebuilding) ===")
    if not gaps:
        print("  none -- clean v2 ladder to 2M for every task")
    for t in ("contra", "nq", "outlier", "rerank", "oolong"):
        if gaps.get(t):
            print("  %-8s %s" % (t, gaps[t]))


if __name__ == "__main__":
    main()
