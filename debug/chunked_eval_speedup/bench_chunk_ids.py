"""Microbenchmark: how much of the chunked-eval per-step cost is chunk_ids rebuilding?

The vLLM FlexAttention backend calls the metadata builder EVERY decode step, and our patch
re-derives chunk_ids from scratch each time. This measures that CPU-side cost alone (no GPU),
for realistic (num_reqs, seq_len, n_docs) shapes taken from the actual CTC rungs.

Run: python debug/chunked_eval_speedup/bench_chunk_ids.py
"""
import time

import numpy as np

_FREE, _PAD = -1, -2
DOC_START, DOC_END = 248049, 248050


def build_original(token_ids_cpu, seq_lens, num_reqs):
    """Verbatim port of vllm_chunked_patch._build_chunk_ids_for_batch's inner loop."""
    max_len = int(seq_lens.max())
    chunk_ids = np.full((num_reqs, max_len), _PAD, dtype=np.int32)
    for ri in range(num_reqs):
        slen = int(seq_lens[ri])
        ids = token_ids_cpu[ri, :slen]
        chunk_ids[ri, :slen] = _FREE
        starts = np.flatnonzero(ids == DOC_START)
        ends = np.flatnonzero(ids == DOC_END)
        if starts.size == 0 or ends.size == 0:
            continue
        ei = 0
        chunk_idx = 0
        for s in starts:
            while ei < ends.size and ends[ei] < s:
                ei += 1
            if ei >= ends.size:
                break
            e = ends[ei]
            chunk_ids[ri, s : e + 1] = chunk_idx
            chunk_idx += 1
            ei += 1
    return chunk_ids


def make_batch(num_reqs, prompt_len, n_docs, decoded=0):
    """A prompt of n_docs wrapped documents + `decoded` appended free tokens."""
    total = prompt_len + decoded
    tok = np.full((num_reqs, total), 7, dtype=np.int32)
    per = max(3, (prompt_len - 64) // max(n_docs, 1))
    for ri in range(num_reqs):
        p = 32
        for _ in range(n_docs):
            if p + per >= prompt_len:
                break
            tok[ri, p] = DOC_START
            tok[ri, p + per - 1] = DOC_END
            p += per
    return tok, np.full(num_reqs, total, dtype=np.int64)


CASES = [
    # (label, num_reqs, prompt_len, n_docs)  -- from the real contradiction rungs
    ("contra 2k   n=44",   16,  1925,  44),
    ("contra 8k   n=187",  16,  8052, 187),
    ("contra 32k  n=762",   8, 32397, 762),
    ("grouping 32k n=176",  8, 32000, 176),
]

print(f"{'case':22s} {'reqs':>5s} {'len':>7s} {'docs':>5s} {'ms/call':>9s} "
      f"{'calls/ex':>9s} {'s/500ex':>9s}")
for label, nr, plen, ndoc in CASES:
    tok, slens = make_batch(nr, plen, ndoc, decoded=32)
    build_original(tok, slens, nr)  # warm
    t0 = time.perf_counter()
    REP = 5
    for _ in range(REP):
        build_original(tok, slens, nr)
    ms = (time.perf_counter() - t0) / REP * 1000.0
    # ~65 builder calls per example measured on the real runs (32374 calls / 500 ex)
    calls_per_ex = 65
    total_s = ms / 1000.0 * calls_per_ex * 500 / nr
    print(f"{label:22s} {nr:5d} {plen:7d} {ndoc:5d} {ms:9.2f} {calls_per_ex:9d} {total_s:9.1f}")

print("\nBreakdown for the 32k case (where it hurts most):")
tok, slens = make_batch(8, 32397, 762, decoded=32)
nr = 8
t0 = time.perf_counter()
for _ in range(20):
    for ri in range(nr):
        ids = tok[ri, : int(slens[ri])]
        np.flatnonzero(ids == DOC_START)
        np.flatnonzero(ids == DOC_END)
print(f"  flatnonzero scans      : {(time.perf_counter()-t0)/20*1000:7.2f} ms")
t0 = time.perf_counter()
for _ in range(20):
    np.full((nr, 32429), _PAD, dtype=np.int32)
print(f"  array alloc+fill       : {(time.perf_counter()-t0)/20*1000:7.2f} ms")
t0 = time.perf_counter()
for _ in range(20):
    build_original(tok, slens, nr)
print(f"  full build (incl. loop): {(time.perf_counter()-t0)/20*1000:7.2f} ms")
