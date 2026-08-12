#!/usr/bin/env python3
"""Measure the chunk layout of a tokenized shard, to detect the oolong `--item-regex` leak.

A bare `--item-regex '||'` is an alternation of empty branches, so it matches EVERY line: in
`--chunk-by line` mode it wrapped oolong's instruction / question / header lines as their own
chunks and left the blank lines between them FREE, bridging chunks that the mask is supposed to
isolate (`debug/ctc_vllm_validation/CHUNK_LEAK_AUDIT.md`).

`metadata.json` cannot settle this for shards built before the converter started recording
`item_regex` -- a bad shard looks identical to a good one. So this reads the raw token stream and
reproduces the mask's own definition of FREE: a token is FREE unless it lies inside a matched
`<|box_start|>` ... `<|box_end|>` span (the same rule as
`corpus_reasoning/lib/vllm_chunked_patch._build_chunk_ids_for_batch`). The leak metric is the
number of FREE tokens strictly BETWEEN consecutive wrapped chunks -- FREE tokens before the first
chunk or after the last are the legitimate prefix/suffix.

EOS-aware, so cross-example boundaries are not counted as gaps.

    python debug/ctc_modelscale/scan_oolong_chunks.py --shard <dir> [--max-examples 400]
"""
import argparse
import glob
import json
import os

import numpy as np

DOC_START, DOC_END, EOS = 248049, 248050, 248044


def scan_example(toks):
    """Return (n_chunks, inter_chunk_free_tokens) for one example's token array."""
    starts = np.flatnonzero(toks == DOC_START)
    ends = np.flatnonzero(toks == DOC_END)
    n = min(len(starts), len(ends))
    if n == 0:
        return 0, 0
    spans = []
    ei = 0
    for s in starts:
        while ei < len(ends) and ends[ei] < s:
            ei += 1
        if ei >= len(ends):
            break
        spans.append((s, ends[ei]))
        ei += 1
    if len(spans) < 2:
        return len(spans), 0
    gap = 0
    for (_, e_prev), (s_next, _) in zip(spans, spans[1:]):
        gap += max(0, int(s_next) - int(e_prev) - 1)
    return len(spans), gap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--max-examples", type=int, default=400)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.shard, "metadata.json")))
    parts = sorted(glob.glob(os.path.join(args.shard, "token_ids_part_*.npy")))
    if not parts:
        raise SystemExit(f"no token_ids_part_*.npy under {args.shard}")

    # These are RAW memmaps that merely carry a .npy extension -- there is no .npy header (which is
    # why `sum(token bytes) == num_tokens * 4` exactly, with no 128-byte-per-part overhead), so
    # np.load misreads them as pickled object data. Read them the way the trainer's memmap dataset
    # does: a flat array of the dtype metadata.json declares.
    dtype = np.dtype(meta.get("dtype", "uint32"))
    stream = np.memmap(parts[0], dtype=dtype, mode="r")
    eos_at = np.flatnonzero(np.asarray(stream[: min(len(stream), 60_000_000)]) == EOS)
    bounds = [0] + [int(i) + 1 for i in eos_at[: args.max_examples]]

    chunks, gaps, n_ex = [], [], 0
    for a, b in zip(bounds, bounds[1:]):
        toks = np.asarray(stream[a:b])
        if len(toks) < 8:
            continue
        c, g = scan_example(toks)
        chunks.append(c)
        gaps.append(g)
        n_ex += 1

    if not n_ex:
        raise SystemExit("no complete examples found (no EOS in the scanned prefix?)")

    mean_chunks = sum(chunks) / n_ex
    mean_gap = sum(gaps) / n_ex
    leaky = sum(1 for g in gaps if g > 0)
    print(f"shard        : {args.shard}")
    print(f"  metadata   : instances={meta.get('num_instances')} "
          f"chunk_by={meta.get('chunk_by')!r} item_regex={meta.get('item_regex', 'NOT RECORDED')!r} "
          f"query_position={meta.get('query_position', 'NOT RECORDED')!r}")
    print(f"  examples   : {n_ex}")
    print(f"  chunks/ex  : {mean_chunks:.1f}")
    print(f"  FREE gaps/ex (between chunks): {mean_gap:.3f}   "
          f"[{leaky}/{n_ex} examples affected]")
    print(f"  VERDICT    : {'LEAK' if mean_gap > 0.01 else 'CLEAN'}")


if __name__ == "__main__":
    main()
