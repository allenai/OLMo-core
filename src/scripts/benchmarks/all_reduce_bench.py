"""
Run an NCCL all-reduce bandwidth benchmark.

Launch under torchrun (or via the Beaker launcher), e.g.:

    torchrun --nproc-per-node=8 src/scripts/benchmarks/all_reduce_bench.py
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_local_rank, get_world_size
from olmo_core.train import prepare_training_environment, teardown_training_environment

log = logging.getLogger(__name__)


def timed_allreduce(mat, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    size = mat.numel() * mat.element_size()
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(get_local_rank())

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= get_world_size()

    return algbw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NCCL all-reduce bandwidth.")
    parser.add_argument(
        "--rows", type=int, default=500000, help="Rows (N) of the fp32 payload tensor."
    )
    parser.add_argument(
        "--cols", type=int, default=2000, help="Columns (M) of the fp32 payload tensor."
    )
    parser.add_argument("--trials", type=int, default=5, help="Number of timed trials to average.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    mat = torch.rand(args.rows, args.cols, dtype=torch.float32).cuda(get_local_rank())
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm up.
    for _ in range(args.warmup):
        timed_allreduce(mat, start_event, end_event)

    algbw_gather = []
    for i in range(args.trials):
        log.info(f"trial {i + 1}/{args.trials}")
        algbw_gather += timed_allreduce(mat, start_event, end_event)

    algbw = torch.mean(torch.stack(algbw_gather))

    # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw reflects how optimally the hardware is used
    n = dist.get_world_size()
    busbw = algbw * (2 * (n - 1) / n)

    payload_gb = args.rows * args.cols * 4 / 1e9
    log.info(
        f"The average bandwidth of all_reduce with a {payload_gb}GB payload "
        f"({args.trials} trials, {n} ranks):\n"
        f"algbw: {algbw / 1e9:.3f} GBps ({algbw * 8 / 1e9:.1f} Gbps)\n"
        f"busbw: {busbw / 1e9:.3f} GBps ({busbw * 8 / 1e9:.1f} Gbps)\n"
    )


if __name__ == "__main__":
    prepare_training_environment()
    try:
        main()
    finally:
        teardown_training_environment()
