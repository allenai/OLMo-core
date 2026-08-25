"""KDA — Kimi Delta Attention, the gated delta rule with a per-dimension gate.

    from kernel_fun.kda import chunk_kda

Same signature and return contract as `fla.ops.kda.chunk_kda`; anything this package does
not implement is forwarded to fla, so the swap is one import line and reverting it is one
line back.

Measured on B300 at the production shape (B=16, T=8192, H=HV=16, K=128, V=256, chunk 64):

    forward+backward, gate and q/k norm in-op   24.07 ms vs fla's 37.20   1.545x
    forward+backward, pre-computed gate         22.14 ms vs fla's 35.57   1.606x
    forward only                                 5.75 ms vs fla's  7.55   1.313x

Ours are the forward scan+readout and four of the backward's seven stages; the rest are
fla's own kernels, called at fla's own stage boundaries. What each stage runs, and why, is
in chain.py.
"""

from .ops import chunk_kda, is_supported, warmup

__all__ = ["chunk_kda", "is_supported", "warmup"]
