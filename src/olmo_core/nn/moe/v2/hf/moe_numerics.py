"""Numerics of OLMoDDP's no-EP expert forward, without a TransformerEngine dependency."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only Transformers installations.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _swiglu_kernel(x, y, rows: tl.constexpr, hidden: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = index // hidden
        col = index % hidden
        up = tl.load(x + row * (2 * hidden) + col, row < rows, 0).to(tl.float32)
        gate = tl.load(x + row * (2 * hidden) + hidden + col, row < rows, 0).to(tl.float32)
        value = up * gate * tl.sigmoid(gate)
        tl.store(y + index, value, row < rows)


def core_swiglu(up_gate: torch.Tensor) -> torch.Tensor:
    """Apply the core inference kernel's single-rounding SwiGLU to packed [up, gate]."""
    if up_gate.is_cuda and not torch.is_grad_enabled():
        if triton is None:
            raise RuntimeError("OLMo-core expert inference numerics require Triton on CUDA")
        up_gate = up_gate.contiguous()
        rows, width = up_gate.shape
        hidden = width // 2
        output = up_gate.new_empty((rows, hidden))
        if rows:
            _swiglu_kernel[(triton.cdiv(rows * hidden, 1024),)](
                up_gate, output, rows, hidden, BLOCK=1024
            )
        return output
    up, gate = up_gate.float().chunk(2, dim=-1)
    return (up * gate * torch.sigmoid(gate)).to(up_gate.dtype)


@torch.compiler.disable
def core_combine(values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Reproduce TE's index-map combination in routing-slot order.

    TE 2.18 converts probabilities to the activation dtype and rounds each fused
    multiply-add to that dtype. Keeping these explicit casts outside compilation
    prevents pointwise fusion from retaining an FP32 accumulator across slots.
    ``values`` is [tokens, top_k, hidden]; probabilities remain FP32 up to this op.
    """
    dtype = values.dtype
    probabilities = probabilities.to(dtype).float()
    values = values.float()
    output = values.new_zeros((values.shape[0], values.shape[2])).to(dtype)
    for slot in range(values.shape[1]):
        output = torch.addcmul(output.float(), values[:, slot], probabilities[:, slot, None]).to(
            dtype
        )
    return output
