"""
Parallel-read every `.distcp` shard of a checkpoint's `model_and_optim/` directory to warm the
node's page cache before the real load.

Root cause this works around: these checkpoints are saved as 128-256 separate `.distcp` shards
(50-55GB total -- one shard per rank at training time, e.g. HSDP shard_degree x CP degree).
compute_loss.py loads them on a single GPU with no torch.distributed initialized, so PyTorch's
distributed-checkpoint reader -- built for N ranks reading N shards in parallel -- reads all of them
sequentially from one process instead. Measured effective throughput doing that: ~50-100 MB/s
(542-1108s wall time for 50-55GB), which is weka-under-a-single-reader speed, not weka's real
aggregate bandwidth. Reading the same shards with many concurrent threads first hits weka with
many parallel connections (its actual scaling axis) and leaves the shards in the node's page cache
(2 TiB RAM on these nodes vs 50-55GB per checkpoint, so it comfortably fits); the subsequent
sequential dcp.load() inside compute_loss.py then reads from RAM instead of the network.

Needs weka mounted (run as a prefix step in the same GPU job, right before compute_loss.py). No
GPU needed for this step itself, but it's cheap enough to just run at the start of the GPU job
rather than as a separate CPU job + hoping the same node gets reused.

Usage:
    python prewarm_checkpoint.py /weka/.../checkpoints/.../stepNNNN [--workers 48]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

READ_CHUNK = 16 * 1024 * 1024


def read_file(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(READ_CHUNK)
            if not chunk:
                break
            n += len(chunk)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    model_dir = os.path.join(args.checkpoint_dir, "model_and_optim")
    files = [
        os.path.join(model_dir, f) for f in sorted(os.listdir(model_dir)) if f.endswith(".distcp")
    ]
    total_bytes = sum(os.path.getsize(f) for f in files)
    print(
        f"[prewarm] {len(files)} shards, {total_bytes / 1e9:.1f} GB, "
        f"{args.workers} parallel readers: {model_dir}",
        flush=True,
    )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        read_bytes = sum(ex.map(read_file, files))
    dt = time.time() - t0
    print(
        f"[prewarm] read {read_bytes / 1e9:.1f} GB in {dt:.1f}s "
        f"({read_bytes / 1e6 / max(dt, 1e-6):.0f} MB/s aggregate)",
        flush=True,
    )


if __name__ == "__main__":
    main()
