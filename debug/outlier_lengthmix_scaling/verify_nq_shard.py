"""Verify a tokenized **nq** arm is COMPLETE and correctly terminated.

Same contract as verify_qdmatch_shard.py -- metadata alone lies, so verify the bytes
([[qdmatch-nq-shard-truncated]]: a shard that was short 2 of 5 token parts read as "underfitting"
for a month).

  1. part counts: #token_ids parts == #labels_mask parts
  2. sum(token bytes) == num_tokens * 4  and  sum(mask bytes) == num_tokens
  3. instance count recovered by splitting on EOS == metadata num_instances
  4. detokenize instance --index: it must contain ALL k rendered `Document [i]` blocks of the
     corresponding source row (no context truncation) and the question, and the assistant turn
     must be exactly the 1-indexed gold id list + EOS. NOTE the answer is the BARE id string
     ("[7]", data_format._build_retrieval_ids) even though the instruction shows
     "Relevant Document: [id]" -- that is the shipped convention `_eval_retrieval` grades against.
     Gold in the source is 0-indexed ([[gold-doc-indices-per-task-base]]), prompt ids are 1-indexed.
  5. the loss mask must be a contiguous suffix covering exactly the answer span (+ the EOS)
"""

import argparse
import glob
import json
import os
import re
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
    gold = sorted(row["gold_doc_indices"])
    want = ", ".join(f"[{g + 1}]" for g in gold)   # bare ids: _build_retrieval_ids()
    n_d = len(row["documents"])
    got_d = len(re.findall(r"Document \[\d+\]", text))
    question = row["queries"][0]

    print("--- instance", i, "---")
    print(f"tokens         = {len(inst)}  (last id {int(inst[-1])}, eos={eos})")
    print(f"rendered docs  = {got_d}   (source: {n_d})")
    print(f"answer(masked) = {answer!r}")
    print(f"answer(source) = {want!r}")
    print(f"question       = {question!r}  present={question in text}")
    print(f"head           = {text[:200]!r}")
    print(f"tail           = {text[-220:]!r}")
    if int(inst[-1]) != eos:
        fails.append("instance does not end with EOS")
    if not bool(imask[-1]):
        fails.append("EOS is not in the loss mask")
    if not answer.strip().startswith(want):
        fails.append(f"masked answer {answer.strip()!r} does not start with gold {want!r}")
    if got_d != n_d:
        fails.append(f"context truncated: rendered {got_d} docs, source {n_d}")
    if question not in text:
        fails.append("rendered instance does not contain the question")
    assistant_seg = text.split("<|im_start|>assistant")[-1]
    if want not in assistant_seg:
        fails.append(f"assistant turn {assistant_seg[-60:]!r} does not carry the answer {want!r}")
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
