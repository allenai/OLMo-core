"""Probe a one-kernel 512/top16 selector, including the native tie contract.

Benchmark-only: a routing-index mismatch disqualifies integration even when the
selected values agree. Inputs match EMO's nonnegative scores / masked -infinity.
"""

import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _top16(scores, output, ROWS: tl.constexpr, LOWER_INDEX_FIRST: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, 512)
    value = tl.load(scores + row * 512 + index)
    # Positive finite float bits preserve numeric order; -inf sorts below zero.
    bits = tl.where(value >= 0, value.to(tl.uint32, bitcast=True), 0)
    order = 511 - index if LOWER_INDEX_FIRST else index
    key = (bits.to(tl.uint64) << 32) | order.to(tl.uint64)
    selected = tl.topk(key, 16).to(tl.uint32)
    result = 511 - selected if LOWER_INDEX_FIRST else selected
    tl.store(output + row * 16 + tl.arange(0, 16), result.to(tl.int64))


def main():
    """Compare indices and selected values, then time with bracketing controls."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(590)
    output = Path("/results/topk")
    output.mkdir(parents=True, exist_ok=True)
    reference = torch.compile(lambda x: x.topk(16, dim=-1).indices, fullgraph=True)
    report = []

    def timing(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        events = []
        for _ in range(30):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        values = [a.elapsed_time(b) for a, b in events]
        return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values)}

    for case in ("random", "tied", "pool16", "pool17", "all_equal"):
        x = torch.rand(32768, 512, device="cuda")
        if case == "tied":
            x = (x * 8).floor()
        elif case == "all_equal":
            x.fill_(1)
        elif case.startswith("pool"):
            pool = int(case[4:])
            x[:, pool:] = float("-inf")
        else:
            x.masked_fill_(torch.rand_like(x) < 0.3, float("-inf"))
        expected = reference(x)
        reference_before = timing(lambda: reference(x))
        for lower_first in (True, False):
            actual = torch.empty_like(expected)

            def candidate():
                _top16[(x.shape[0],)](x, actual, x.shape[0], lower_first, num_warps=4)

            candidate()
            row = {
                "case": case,
                "lower_index_first": lower_first,
                "mismatched_indices": int((expected != actual).sum()),
                "mismatched_values": int((x.gather(1, expected) != x.gather(1, actual)).sum()),
                "reference_before": reference_before,
                "candidate": timing(candidate),
                "reference_after": timing(lambda: reference(x)),
            }
            report.append(row)
            (output / "summary.json").write_text(json.dumps(report, indent=2))
            print("TOPK_PROBE", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
