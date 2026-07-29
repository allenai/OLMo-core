"""Which contradiction eval rungs contain FEVER / wiki_mix documents?

`build_xlong_rungs.py` harvested distractors with the glob `contradiction_*_k3.jsonl`, which matches
the FEVER and wiki_mix corpora as well as PubMed -- so xlong rungs could carry Wikipedia-style
claims as distractors in a file named `contradiction_eval_pubmed_both_*`. contra_fever is a separate
experimental setting and must not bleed into the PubMed contradiction ladder.
`build_v2_eval_ladders.py` already restricts to `*pubmed*`, so the <=32k rungs are expected clean --
this script CHECKS that rather than trusting it, since files on disk may predate that fix.

Method: hash every document text in the FEVER + wiki_mix sources, then report what fraction of each
eval rung's documents hit that set. Exact-text membership, no heuristics.

Run: python debug/xlong_5task/audit_contra_fever_leak.py [--sample-docs N]
"""

import argparse
import glob
import hashlib
import json
import os

DATA = "/scratch/users/prasann/corpus-reasoning/data"
E5 = "/scratch/users/prasann/cpt_data/eval500_v2"
CTC = "/scratch/users/prasann/ctc_suite_staged/eval_rungs/contradiction"


def h(text: str) -> str:
    return hashlib.blake2b(text.strip().encode("utf-8", "replace"), digest_size=16).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-docs", type=int, default=3000,
                    help="documents sampled per rung (from the first few examples)")
    args = ap.parse_args()

    # 1. Fingerprint every FEVER / wiki_mix document.
    foreign = set()
    srcs = sorted(set(glob.glob(f"{DATA}/contradiction_*fever*_k3.jsonl")
                      + glob.glob(f"{DATA}/contradiction_*wiki*_k3.jsonl")))
    print(f"fingerprinting {len(srcs)} FEVER/wiki source files ...", flush=True)
    for p in srcs:
        with open(p) as f:
            for line in f:
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for d in ex.get("documents", []):
                    t = d.get("text", "")
                    if t:
                        foreign.add(h(t))
    print(f"  {len(foreign):,} distinct FEVER/wiki document texts\n", flush=True)

    # 2. Every contradiction rung we can reach locally.
    rungs = []
    for pat, label in ((f"{E5}/contra/contradiction_*_k3.jsonl", "eval500_v2"),
                       (f"{CTC}/rung_*.jsonl", "ctc_suite")):
        for p in sorted(glob.glob(pat)):
            rungs.append((label, p))

    print(f"{'bundle':11s} {'rung file':62s} {'docs':>7} {'FEVER/wiki':>11} {'pct':>7}")
    for label, p in rungs:
        seen = 0
        hits = 0
        with open(p) as f:
            for line in f:
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for d in ex.get("documents", []):
                    t = d.get("text", "")
                    if not t:
                        continue
                    seen += 1
                    if h(t) in foreign:
                        hits += 1
                if seen >= args.sample_docs:
                    break
        if not seen:
            continue
        pct = 100.0 * hits / seen
        flag = "  <-- CONTAMINATED" if pct > 0.5 else ""
        print(f"{label:11s} {os.path.basename(p)[:62]:62s} {seen:7d} {hits:11d} {pct:6.2f}%{flag}")


if __name__ == "__main__":
    main()
