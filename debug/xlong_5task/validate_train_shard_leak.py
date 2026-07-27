"""Scan built TRAINING shards for free-token leakage between document chunks.

The chunked mask isolates each document: every token is FREE (attends globally) unless it lies
inside a matched ``<|doc_start|> ... <|doc_end|>`` span. The design invariant is that the ONLY free
tokens are the instruction/question PREFIX and the query/answer SUFFIX.

**A LEAK = any token strictly between the end of one chunk and the start of the next.** Such a token
bridges two supposedly-isolated documents.

This is the training-shard counterpart of ``debug/ctc_vllm_validation/validate_chunk_leak.py``
(which covers eval prefills). It reads the raw ``token_ids_part_*.npy`` uint32 stream directly and
is EOS-aware: a gap containing the EOS id is a legitimate example boundary, not a leak.

This is the check that catches the oolong ``--item-regex`` bug: with the bare ``'||'`` regex the
instruction/question/header lines were wrapped as their own chunks with free ``\\n\\n`` between them,
which showed up as ~5 inter-chunk free tokens per example (2019 total in the audited shard).
"""

import argparse
import glob
import os

import numpy as np

DOC_START, DOC_END, EOS = 248049, 248050, 248044


def scan(path: str, max_instances: int) -> dict:
    """Count inter-chunk FREE tokens across the instances in one shard part."""
    a = np.fromfile(path, dtype=np.uint32)
    starts = np.flatnonzero(a == DOC_START)
    ends = np.flatnonzero(a == DOC_END)
    eos = set(np.flatnonzero(a == EOS).tolist())

    n = min(len(starts), len(ends))
    leak_tokens = 0
    leak_gaps = 0
    examples_spanned = set()
    # Walk consecutive (end_k, start_k+1) pairs. A gap that contains an EOS separates two
    # examples -- expected, not a leak.
    for k in range(n - 1):
        gap_lo, gap_hi = int(ends[k]) + 1, int(starts[k + 1])
        if gap_hi <= gap_lo:
            continue
        if any(i in eos for i in range(gap_lo, gap_hi)):
            continue  # example boundary
        leak_tokens += gap_hi - gap_lo
        leak_gaps += 1
        examples_spanned.add(k)
        if max_instances and len(examples_spanned) >= max_instances:
            break
    return {
        "chunks": n,
        "leak_tokens": leak_tokens,
        "leak_gaps": leak_gaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/prasann/xlong5/shards")
    ap.add_argument("--parts", type=int, default=1, help="shard parts to scan per task")
    ap.add_argument("--max-instances", type=int, default=0)
    args = ap.parse_args()

    print(f"{'task':16s}{'chunks':>12}{'leak_gaps':>12}{'leak_tokens':>14}   verdict")
    all_clean = True
    for d in sorted(glob.glob(os.path.join(args.root, "*_train"))):
        task = os.path.basename(d).replace("_train", "")
        parts = sorted(glob.glob(os.path.join(d, "token_ids_part_*.npy")))[: args.parts]
        if not parts:
            print(f"{task:16s}  (no shards)")
            all_clean = False
            continue
        tot = {"chunks": 0, "leak_tokens": 0, "leak_gaps": 0}
        for p in parts:
            r = scan(p, args.max_instances)
            for k in tot:
                tot[k] += r[k]
        clean = tot["leak_tokens"] == 0
        all_clean = all_clean and clean
        print(
            f"{task:16s}{tot['chunks']:>12,}{tot['leak_gaps']:>12,}{tot['leak_tokens']:>14,}"
            f"   {'CLEAN' if clean else 'LEAK'}"
        )
    print("\nALL SHARDS CLEAN" if all_clean else "\nLEAK DETECTED")


if __name__ == "__main__":
    main()
