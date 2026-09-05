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


@triton.jit
def _native_tie_top16(scores, output):
    row = tl.program_id(0)
    index = tl.arange(0, 512)
    value = tl.load(scores + row * 512 + index)
    bits = tl.where(value >= 0, value.to(tl.uint32, bitcast=True), 0)
    key = (bits.to(tl.uint64) << 32) | (511 - index).to(tl.uint64)
    selected = 511 - tl.topk(key, 16).to(tl.uint32)
    selected_values = tl.gather(value, selected.to(tl.int32), 0)
    threshold = tl.min(selected_values, 0)
    # Reconstruct CUDA gatherTopK order: >threshold in source order, then the
    # first remaining threshold ties in source order. No floating perturbations.
    priority = (selected_values > threshold).to(tl.uint32) * 512 + 511 - selected
    ordered = tl.sort(priority, descending=True) % 512
    selected = 511 - ordered
    # CUDA small-sort uses a 32-element bitonic network with 16 invalid tails.
    # Reproduce its comparator/swap directions, including equal-key swaps.
    lane = tl.arange(0, 32)
    valid = lane < 16
    ids = tl.gather(selected, lane % 16, 0).to(tl.int32)
    values = tl.gather(value, ids, 0)
    for log_size in tl.static_range(1, 6):
        for log_stride in tl.static_range(log_size - 1, -1, -1):
            stride = 1 << log_stride
            partner = lane ^ stride
            other_values = tl.gather(values, partner, 0)
            other_ids = tl.gather(ids, partner, 0)
            other_valid = tl.gather(valid, partner, 0)
            lower = (lane & stride) == 0
            left = tl.where(lower, values, other_values)
            right = tl.where(lower, other_values, values)
            valid_left = tl.where(lower, valid, other_valid)
            valid_right = tl.where(lower, other_valid, valid)
            direction = ((lane & (1 << log_size)) != 0) if log_size < 5 else False
            swap = (((left > right) & valid_left) | ~valid_right) == direction
            values = tl.where(swap, other_values, values)
            ids = tl.where(swap, other_ids, ids)
            valid = tl.where(swap, other_valid, valid)
    tl.store(output + row * 16 + lane, ids.to(tl.int64), lane < 16)


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
        for lower_first in (True, False, "native_tie"):
            actual = torch.empty_like(expected)

            def candidate():
                if lower_first == "native_tie":
                    _native_tie_top16[(x.shape[0],)](x, actual, num_warps=4)
                else:
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
