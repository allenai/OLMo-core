"""Audit the contra xlong FILLER pool: which source files feed it, and how long their docs are.

Why this exists: ``build_v2_eval_ladders.harvest_fillers`` builds the contradiction distractor pool
by globbing ``contradiction_*_k3.jsonl`` over a MUTABLE data directory. The pool is therefore
"whatever files happen to be on disk at build time" -- so two builds months apart draw different
filler distributions and the same rung label means different corpora. That is what made the live
root's 64k/128k/256k rungs average 36.5 tok/doc while the 2026-07-29 512k/1M/2M rungs average 47.1.

It also shows the pool is domain-HETEROGENEOUS: the glob matches ``*_wiki_mix_*`` files, so
Wikipedia/FEVER-style claims get mixed into a nominally PubMed contradiction eval.

Run: python debug/xlong_5task/audit_filler_pool.py [--data DIR] [--probe N]
"""

import argparse
import glob
import json
import os
import statistics

CHARS_PER_TOK = 4.2  # Qwen3.5 on this corpus; only used for a rough per-file tok/doc estimate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/scratch/users/prasann/corpus-reasoning/data")
    ap.add_argument("--probe", type=int, default=40,
                    help="examples per file to sample document lengths from")
    ap.add_argument("--live-build-date", default="2026-07-03",
                    help="files newer than this were NOT in the pool when the live "
                         "64k/128k/256k rungs were built")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.data, "contradiction_*_k3.jsonl"))):
        lens = []
        with open(p) as f:
            for i, line in enumerate(f):
                if i >= args.probe:
                    break
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lens += [len(d.get("text", "")) for d in ex.get("documents", []) if d.get("text")]
        if not lens:
            continue
        mtime = os.popen(f"date -r '{p}' +%Y-%m-%d").read().strip()
        rows.append({
            "file": os.path.basename(p),
            "mb": os.path.getsize(p) / 1e6,
            "tok_per_doc": statistics.mean(lens) / CHARS_PER_TOK,
            "mtime": mtime,
            "is_wiki": "wiki" in os.path.basename(p),
            "added_after_live": mtime >= args.live_build_date,
        })

    rows.sort(key=lambda r: -r["mb"])
    print(f"{'file':58s} {'MB':>7} {'~tok/doc':>9} {'mtime':>11}  note")
    for r in rows:
        note = ""
        if r["is_wiki"]:
            note = "WIKI/FEVER -> domain-foreign distractors"
        elif r["added_after_live"]:
            note = "ADDED after the live rungs were built"
        print(f"{r['file']:58s} {r['mb']:7.1f} {r['tok_per_doc']:9.1f} {r['mtime']:>11}  {note}")

    tot = sum(r["mb"] for r in rows)
    new = sum(r["mb"] for r in rows if r["added_after_live"])
    wiki = sum(r["mb"] for r in rows if r["is_wiki"])
    print(f"\ntotal matched: {tot:.0f} MB across {len(rows)} files")
    print(f"  added after {args.live_build_date}: {new:.0f} MB ({100*new/tot:.0f}% of the pool by bytes)"
          f"  <- present for the 512k/1M/2M build, absent for the live 64k/128k/256k build")
    print(f"  wiki/FEVER sources:               {wiki:.0f} MB ({100*wiki/tot:.0f}% of the pool by bytes)")


if __name__ == "__main__":
    main()
