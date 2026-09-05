"""cconv — the KDA short convolution: width-4 depthwise causal conv + silu.

    from kernel_fun.cconv import causal_conv1d

Same signature and return contract as `fla.modules.convolution.causal_conv1d`; anything this
package does not implement is forwarded to fla, so the swap is one import line and
reverting it is one line back.

A KDA layer makes three of these calls per forward (q, k at D = n_heads*128, v at 2x that),
and the 810m/B300 step trace put them at ~50 ms of a ~520 ms step — the largest non-GEMM
item in the step, against an ~8-10 ms bandwidth roofline. Measured isolated on B300 at the
production call (B=16, T=8192, D=2048, W=4, bf16 x / fp32 weight; 2026-09-01, the ladder's
cconv/001 record 002):

    backward (dx, dw; no forward re-run)   1.597 ms -> 0.344 ms   4.65x   (4.7 TB/s)
    forward  (silu fused)                  0.307 ms -> 0.190 ms   1.62x   (5.6 TB/s)

The same ratios hold at D=1024 and D=4096. Both kernels are Triton; see ops.py for the
supported box and _kernels/strip.py for the design.
"""

from .ops import causal_conv1d, is_supported, warmup

__all__ = ["causal_conv1d", "is_supported", "warmup"]
