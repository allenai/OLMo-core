"""Round-2 CPU microbenchmark for `_build_chunk_ids_for_batch`.

WHY. The 8k sweep (job 3437892) profiled `build_chunk_ids` at 24.9 ms/call with 8 resident
sequences but 0.75 ms/call with 2 -- a 33x jump for a 4x change in work. That smells like a
measurement artifact, not an algorithmic cliff: in the canonical patch module the stage is timed
with ``_prof_add(..., sync=True)``, so its elapsed time ABSORBS a ``torch.cuda.synchronize()`` and
therefore the tail of the previous step's GPU work. With 8 resident sequences the GPU has ~4x more
decode work outstanding, so the sync waits longer -- and the wait is billed to chunk_ids.

This script runs the SAME function with no CUDA in the picture, so whatever it reports is the real
CPU cost. Pure numpy; runs anywhere.
"""

import argparse
import time

import numpy as np

_FREE, _PAD = -1, -2


def build_chunk_ids(token_ids_cpu, seq_lens, doc_start, doc_end):
    """Verbatim copy of the numpy half of `_build_chunk_ids_for_batch` (no .to(device))."""
    num_reqs = len(seq_lens)
    max_len = int(seq_lens.max())
    chunk_ids = np.full((num_reqs, max_len), _PAD, dtype=np.int32)
    for ri in range(num_reqs):
        slen = int(seq_lens[ri])
        if slen <= 0:
            continue
        ids = token_ids_cpu[ri, :slen]
        chunk_ids[ri, :slen] = _FREE
        starts = np.flatnonzero(ids == doc_start)
        ends = np.flatnonzero(ids == doc_end)
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


def synth(num_reqs, seq_len, n_chunks, doc_start, doc_end):
    buf = np.zeros((num_reqs, seq_len + 8), dtype=np.int32)
    per = seq_len // (n_chunks + 1)
    for ri in range(num_reqs):
        rng = np.random.default_rng(ri)
        buf[ri, :] = rng.integers(1000, 50000, size=seq_len + 8)
        for c in range(n_chunks):
            s = c * per
            e = min(s + per - 2, seq_len - 1)
            buf[ri, s] = doc_start
            buf[ri, e] = doc_end
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8875)
    ap.add_argument("--n-chunks", type=int, default=187)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    doc_start, doc_end = 151648, 151649

    print(f"seq_len={args.seq_len} n_chunks={args.n_chunks} iters={args.iters}")
    print(f"{'num_reqs':>9s} {'ms/call':>9s}")
    for num_reqs in (1, 2, 4, 8, 16, 32):
        buf = synth(num_reqs, args.seq_len, args.n_chunks, doc_start, doc_end)
        seq_lens = np.full(num_reqs, args.seq_len, dtype=np.int64)
        build_chunk_ids(buf, seq_lens, doc_start, doc_end)  # warm
        t0 = time.perf_counter()
        for _ in range(args.iters):
            build_chunk_ids(buf, seq_lens, doc_start, doc_end)
        dt = (time.perf_counter() - t0) / args.iters * 1000.0
        print(f"{num_reqs:9d} {dt:9.3f}")


if __name__ == "__main__":
    main()
