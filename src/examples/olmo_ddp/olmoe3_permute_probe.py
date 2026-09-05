"""Forward-only token packing screen; include stable sorting and map construction.

No training imports this file. Exact reference row-map order is mandatory: an
equivalent permutation with different rows is not an acceptable comparison.
"""

import json
import os
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.permutation import moe_permute


@triton.jit
def _pack(X, Order, Y, Map, T: tl.constexpr, K: tl.constexpr, D: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    slots = tl.load(Order + rows, rows < T * K, other=0)
    tokens = slots // K
    cols = tl.arange(0, D)
    values = tl.load(X + tokens[:, None] * D + cols[None, :], rows[:, None] < T * K, other=0)
    tl.store(Y + rows[:, None] * D + cols[None, :], values, rows[:, None] < T * K)
    tl.store(Map + (slots % K) * T + tokens, rows.to(tl.int32), rows < T * K)


def candidate(x, indices, rows_per_program):
    """Compute the whole forward permutation, not a copy with a free precomputed map."""
    t, d = x.shape
    k = indices.shape[1]
    order = torch.argsort(indices.reshape(-1), stable=True)
    output = torch.empty((t * k, d), dtype=x.dtype, device=x.device)
    mapping = torch.empty(t * k, dtype=torch.int32, device=x.device)
    _pack[(triton.cdiv(t * k, rows_per_program),)](
        x, order, output, mapping, t, k, d, rows_per_program, num_warps=4
    )
    return output, mapping


def measure(fn):
    """Bracket candidates with the existing full forward operation on the same device."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples)}


def main():
    """Screen exact forward copies at production token count/latent width/top-k."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    assert torch.cuda.get_device_capability(0)[0] == 10
    output = Path("/results/permute-probe")
    output.mkdir(parents=True, exist_ok=True)
    report = {"source": os.environ.get("GIT_REF"), "torch": torch.__version__, "cases": []}
    x = torch.randn(32768, 512, device="cuda", dtype=torch.bfloat16)
    for experts in (512, 64):
        indices = torch.rand(32768, experts, device="cuda").topk(16, dim=-1).indices.int()

        def baseline():
            return moe_permute(
                inp=x, routing_map=indices, num_out_tokens=32768 * 16, map_type="index"
            )

        expected, mapping = baseline()
        row = {
            "active_experts": experts,
            "map_shape": list(mapping.shape),
            "map_dtype": str(mapping.dtype),
            "baseline_before": measure(baseline),
            "candidates": [],
        }
        for r in (4, 8, 16):
            found, found_map = candidate(x, indices, r)
            exact_output = torch.equal(expected, found)
            exact_map = torch.equal(mapping.reshape(-1), found_map.reshape(-1))
            result = {
                "rows_per_program": r,
                "exact_output": exact_output,
                "exact_map": exact_map,
            }
            if exact_output and exact_map:
                result.update(measure(lambda: candidate(x, indices, r)))
            row["candidates"].append(result)
            print("PERMUTE_PROBE", json.dumps(result), flush=True)
        row["baseline_after"] = measure(baseline)
        report["cases"].append(row)
        report["caveat"] = "Forward-only synthetic screen; no autograd/optimizer/training sign-off"
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    if not all(
        c["exact_output"] and c["exact_map"] for r in report["cases"] for c in r["candidates"]
    ):
        raise RuntimeError("A candidate did not reproduce the exact reference permutation")


if __name__ == "__main__":
    main()
