"""Exact-shape Torch 2.11 CUDA top16 index selection; default-off profiling use.

Reimplements TensorTopK gather ordering and SortUtils 32-lane tie swaps:
https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/SortUtils.cuh
No score perturbations; preserve CUDA radix ordering and NaN handling.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _native_tie_top16(scores, output):
    row = tl.program_id(0)
    index = tl.arange(0, 512)
    value = tl.load(scores + row * 512 + index)
    raw_bits = value.to(tl.uint32, bitcast=True)
    bits = tl.where((raw_bits & 0x80000000) != 0, ~raw_bits, raw_bits ^ 0x80000000)
    bits = tl.where(value != value, 0xFFFFFFFF, bits).to(tl.uint32)
    key = (bits.to(tl.uint64) << 32) | (511 - index).to(tl.uint64)
    selected = 511 - tl.topk(key, 16).to(tl.uint32)
    selected_bits = tl.gather(bits, selected.to(tl.int32), 0)
    threshold = tl.min(selected_bits, 0)
    # Reconstruct CUDA gatherTopK order: >threshold in source order, then the
    # first remaining threshold ties in source order. No floating perturbations.
    priority = (selected_bits > threshold).to(tl.uint32) * 512 + 511 - selected
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
            greater = (left > right) | ((left != left) & (right == right))
            swap = ((greater & valid_left) | ~valid_right) == direction
            values = tl.where(swap, other_values, values)
            ids = tl.where(swap, other_ids, ids)
            valid = tl.where(swap, other_valid, valid)
    tl.store(output + row * 16 + lane, ids.to(tl.int64), lane < 16)


def top16_native_indices(scores):
    """Return indices with the native CUDA topk(16, sorted=True) tie policy."""
    if (
        not scores.is_cuda
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or scores.shape[-1] != 512
        or not torch.__version__.startswith("2.11.")
    ):
        raise ValueError("Qualified only for contiguous CUDA FP32 512-expert Torch 2.11 scores")
    output = torch.empty((*scores.shape[:-1], 16), device=scores.device, dtype=torch.int64)
    _native_tie_top16[(scores.numel() // 512,)](scores, output, num_warps=4)
    return output
