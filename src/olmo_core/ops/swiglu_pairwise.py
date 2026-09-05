"""Experimental paired-output SwiGLU backward matching the compiled BF16 path.

Default-off profiling experiment. The baseline Inductor kernel carries intermediate
values in FP32 and rounds at the BF16 output stores. This is intentionally not a
claim of bitwise agreement with eager BF16 autograd, which has intermediate stores.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu_backward_pair(x, dy, dx, pairs, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < pairs
    base = index // HIDDEN * (2 * HIDDEN) + index % HIDDEN
    up = tl.load(x + base, mask=mask, other=0).to(tl.float32)
    gate = tl.load(x + base + HIDDEN, mask=mask, other=0).to(tl.float32)
    grad = tl.load(dy + index, mask=mask, other=0).to(tl.float32)
    # Preserve the operation order and libdevice exponential of the saved
    # PyTorch 2.11/SM103 generated code; do not use the approximate tl.sigmoid.
    denom = libdevice.exp(-gate) + 1.0
    grad_up = grad * (gate / denom)
    sigmoid = 1.0 / denom
    grad_gate = ((grad * up) * sigmoid) * (gate * (1.0 - sigmoid) + 1.0)
    tl.store(dx + base, grad_up, mask=mask)
    tl.store(dx + base + HIDDEN, grad_gate, mask=mask)


def swiglu_backward_pair(x, dy, *, block=1024, warps=4):
    """Compute contiguous BF16 input gradients without changing input or upstream gradient."""
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or dy.dtype != x.dtype
        or dy.device != x.device
        or x.ndim != 2
        or x.shape[1] % 2
        or tuple(dy.shape) != (x.shape[0], x.shape[1] // 2)
        or not x.is_contiguous()
        or not dy.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous same-device CUDA BF16 [M,2H] input and [M,H] gradient"
        )
    out = torch.empty_like(x)
    if dy.numel():
        _swiglu_backward_pair[(triton.cdiv(dy.numel(), block),)](
            x,
            dy,
            out,
            dy.numel(),
            x.shape[1] // 2,
            block,
            num_warps=warps,
            enable_fp_fusion=True,
        )
    return out


class _PairwiseSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        up, gate = x.chunk(2, dim=-1)
        return up * F.silu(gate)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return swiglu_backward_pair(x, grad.contiguous())


def pairwise_swiglu(x):
    """Use with torch.compile: unchanged PyTorch forward and a qualified paired backward."""
    return _PairwiseSwiGLU.apply(x)
