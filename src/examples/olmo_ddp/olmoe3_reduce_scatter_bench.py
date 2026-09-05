"""Two-GPU qualification of the production FP32 gradient-bucket packing operation."""

import gc
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from olmo_core.nn.parallel import MultiGroupDistributedDataParallel


def main():
    """Compare actual DDP reduction paths on 1/2-GiB single-parameter buckets."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "gpus": dist.get_world_size(),
        "torch": torch.__version__,
        "nccl": torch.cuda.nccl.version(),
        "cases": [],
        "caveat": "Isolated bucket timing on one node, not a full-model or 64-GPU speedup.",
    }
    try:
        for gib in (1, 2):
            numel = gib * 1024**3 // 4
            for label, fast_path in (
                ("packed-before", False),
                ("single-param", True),
                ("packed-after", False),
            ):
                model = nn.ParameterDict(
                    {
                        "weight": nn.Parameter(
                            torch.empty(numel, device=device, dtype=torch.bfloat16)
                        )
                    }
                )
                ddp = MultiGroupDistributedDataParallel(
                    model,
                    init_sync=False,
                    accumulate_grads_in_fp32=True,
                    reduce_grads_in_fp32=True,
                    use_reduce_scatter=True,
                )
                ddp.configure_reduce_scatter_params(set(model.values()))
                ddp._reduce_scatter_single_param_fast_path = fast_path
                assert len(ddp._grad_buckets) == 1
                bucket = ddp._grad_buckets[0]
                milliseconds = []
                for iteration in range(25):
                    bucket.flat_storage.fill_(dist.get_rank() + 1)
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    ddp._launch_bucket_grad_reduce(0)
                    handle, _ = ddp._grad_reduce_hooks.pop()
                    handle.wait()
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(start.elapsed_time(end), device=device)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    if iteration >= 5:
                        milliseconds.append(elapsed.item())
                expected = (dist.get_world_size() + 1) / 2
                assert bool((bucket.flat_reduced_storage == expected).all().item())
                assert bool(ddp._reduce_scatter_pack_scratch) == (not fast_path)
                entry = {
                    "gradient_gib": gib,
                    "arm": label,
                    "mean_ms": statistics.mean(milliseconds),
                    "median_ms": statistics.median(milliseconds),
                    "all_ms": milliseconds,
                    "correct": True,
                }
                report["cases"].append(entry)
                if dist.get_rank() == 0:
                    print("REDUCE_SCATTER_BENCH", json.dumps(entry), flush=True)
                    Path("/results/reduce-scatter-bench.json").write_text(
                        json.dumps(report, indent=2)
                    )
                del model, ddp, bucket, handle
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
