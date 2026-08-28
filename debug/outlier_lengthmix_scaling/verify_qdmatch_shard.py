"""Verify a tokenized qdmatch_nq arm is COMPLETE and correctly terminated.

This is the check that the truncated CTC-suite qdmatch_nq shard failed for a month
([[qdmatch-nq-shard-truncated]]): metadata alone lies, so verify the bytes.

  1. part counts: #token_ids parts == #labels_mask parts
  2. sum(token bytes) == num_tokens * 4  and  sum(mask bytes) == num_tokens
  3. instance count recovered by splitting on EOS == metadata num_instances
  4. detokenize instance --index: it must end with the gold-pair answer + EOS, and contain
     ALL M queries and N docs of the corresponding source row (no context truncation)
  5. the loss mask must cover exactly the answer span (+ the EOS)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--source", required=True, help="the arm jsonl this shard was built from")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    meta = json.load(open(os.path.join(args.shard, "metadata.json")))
    tok_parts = sorted(glob.glob(os.path.join(args.shard, "token_ids_part_*.npy")))
    lab_parts = sorted(glob.glob(os.path.join(args.shard, "labels_mask_*.npy")))
    tb = sum(os.path.getsize(p) for p in tok_parts)
    lb = sum(os.path.getsize(p) for p in lab_parts)
    fails = []
    print(f"shard          = {args.shard}")
    print(f"token parts    = {len(tok_parts)}  bytes={tb}  expected={meta['num_tokens'] * 4}")
    print(f"mask parts     = {len(lab_parts)}  bytes={lb}  expected={meta['num_tokens']}")
    if len(tok_parts) != len(lab_parts):
        fails.append(f"part-count mismatch {len(tok_parts)} vs {len(lab_parts)}")
    if tb != meta["num_tokens"] * 4:
        fails.append("token bytes != num_tokens*4  (TRUNCATED SHARD)")
    if lb != meta["num_tokens"]:
        fails.append("mask bytes != num_tokens  (TRUNCATED SHARD)")

    ids = np.concatenate([np.fromfile(p, dtype=np.uint32) for p in tok_parts])
    mask = np.concatenate([np.fromfile(p, dtype=bool) for p in lab_parts])
    eos = meta["eos"]
    ends = np.flatnonzero(ids == eos)
    print(f"instances(EOS) = {len(ends)}  metadata num_instances={meta['num_instances']}")
    if len(ends) != meta["num_instances"]:
        fails.append(f"EOS-split instance count {len(ends)} != {meta['num_instances']}")

    i = args.index
    start = 0 if i == 0 else int(ends[i - 1]) + 1
    stop = int(ends[i]) + 1
    inst, imask = ids[start:stop], mask[start:stop]
    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    text = tk.decode(inst[:-1].tolist())
    answer = tk.decode(inst[imask][:-1].tolist())

    with open(args.source) as f:
        for j, line in enumerate(f):
            if j == i:
                row = json.loads(line)
                break
    want = json.dumps([[int(a), int(b)] for a, b in row["gold_pairs"]])
    n_q = sum(1 for d in row["documents"] if d.get("type") == "query")
    n_d = len(row["documents"]) - n_q
    got_q, got_d = text.count("] Query: "), text.count("] Document: ")

    print("--- instance", i, "---")
    print(f"tokens         = {len(inst)}  (last id {int(inst[-1])}, eos={eos})")
    print(f"rendered items = {got_q} Query / {got_d} Document   (source: {n_q} / {n_d})")
    print(f"answer(masked) = {answer!r}")
    print(f"answer(source) = {want!r}")
    print(f"head           = {text[:180]!r}")
    print(f"tail           = {text[-220:]!r}")
    if int(inst[-1]) != eos:
        fails.append("instance does not end with EOS")
    if not bool(imask[-1]):
        fails.append("EOS is not in the loss mask")
    # the masked span is the assistant turn: the answer plus any chat-template suffix
    if not answer.strip().startswith(want):
        fails.append(f"masked answer {answer.strip()!r} does not start with gold {want!r}")
    if (got_q, got_d) != (n_q, n_d):
        fails.append(f"context truncated: rendered {got_q}/{got_d}, source {n_q}/{n_d}")
    if want not in text:
        fails.append("rendered instance does not contain the answer string")
    # loss must be answer-only: every masked token after the first masked one, none before
    first = int(np.flatnonzero(imask)[0])
    if not imask[first:].all():
        fails.append("loss mask is not a contiguous suffix (should be assistant turn + EOS)")
    print(f"loss tokens    = {int(imask.sum())} of {len(inst)} (suffix from {first})")

    print()
    if fails:
        print("VERIFY FAILED:")
        for f_ in fails:
            print("  !!!", f_)
        sys.exit(1)
    print("VERIFY OK: complete shard, full context, answer+EOS terminated, answer-only loss")


if __name__ == "__main__":
    main()
