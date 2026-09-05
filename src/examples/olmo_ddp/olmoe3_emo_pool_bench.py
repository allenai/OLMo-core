"""One-GPU exact-mask qualification and timing at the small model's routing shape."""

import json
import os
import statistics
from pathlib import Path

import torch

from olmo_core.ops.moe import pool_keep_mask, pool_keep_mask_inverse_scatter


def measure(function, scores, pools):
    """CUDA-event timings after compilation and ten warmup calls."""
    for _ in range(10):
        function(scores, pools)
    torch.cuda.synchronize()
    values = []
    for _ in range(30):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function(scores, pools)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values)}


def main():
    """Test eager and compiled alternatives with unchanged inputs and random seeds."""
    torch.manual_seed(123)
    device = torch.cuda.get_device_name()
    scores = torch.rand(4, 8192, 512, device="cuda", dtype=torch.float32)
    pools = torch.randint(16, 513, (4, 8192), device="cuda")
    tied_scores = torch.randint(-2, 3, scores.shape, device="cuda").float()
    results = {
        "device": device,
        "shape": list(scores.shape),
        "git_commit": os.environ.get("GIT_REF"),
    }
    reference = pool_keep_mask(scores, pools)
    tied_reference = pool_keep_mask(tied_scores, pools)
    for mode in ("eager", "compiled"):
        for name, function in (
            ("double-sort", pool_keep_mask),
            ("inverse-scatter", pool_keep_mask_inverse_scatter),
        ):
            if mode == "compiled":
                function = torch.compile(function, fullgraph=True)
            assert torch.equal(function(scores, pools), reference), (mode, name, "random")
            assert torch.equal(function(tied_scores, pools), tied_reference), (mode, name, "ties")
            measurement = measure(function, scores, pools)
            results[f"{mode}:{name}"] = {"exact_masks": True, **measurement}
            print("EMO_POOL_BENCH", mode, name, json.dumps(measurement), flush=True)
    destination = Path(os.environ.get("RESULTS_DIR", "/results"))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "emo-pool-bench.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
