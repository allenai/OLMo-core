"""Experimental vectorized mixed-dtype gradient addition; callers opt in explicitly."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gradient_add(destination, source, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    x = tl.load(destination + offsets, mask=mask, other=0)
    y = tl.load(source + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(destination + offsets, x + y, mask=mask)


def gradient_add(destination, source, block=2048, warps=4):
    """Add BF16 gradients into an existing contiguous FP32 buffer on the current stream."""
    if (
        not destination.is_cuda
        or destination.device != source.device
        or destination.dtype != torch.float32
        or source.dtype != torch.bfloat16
        or destination.shape != source.shape
        or not destination.is_contiguous()
        or not source.is_contiguous()
    ):
        raise ValueError("Expected same-shape contiguous CUDA FP32 destination and BF16 source")
    if destination.numel():
        _gradient_add[(triton.cdiv(destination.numel(), block),)](
            destination, source, destination.numel(), BLOCK=block, num_warps=warps
        )
    return destination
