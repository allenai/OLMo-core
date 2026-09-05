"""`causal_conv1d` — a drop-in for `fla.modules.convolution.causal_conv1d`.

Same signature, same `(y, final_state)` return, same validation. On a supported call it runs
the strip kernels in _kernels/strip.py; on anything else it forwards the call to fla
verbatim, which is why the fallback is bit-identical rather than merely close.

The gate is a WHITELIST, like kda's. Every fla parameter is either implemented here or
forces the fallback, so an fla flag this package has never seen degrades to fla instead of
being silently dropped. The supported box is exactly what the KDA layer calls in mainline
pretraining: `activation` silu/swish, no bias, no residual, no state, no `cu_seqlens`,
`backend="triton"`, W <= 4, bf16/fp16 `x` of shape [B, T, D] (any strides), a [D, W] weight.

Two things this family does differently from kda, both on purpose:

  - No CuTe, no sm100 gate. Both kernels are Triton, so the arch floor is sm90. They have
    only been TIMED on sm100 (B300); on an H100 they compute the same thing at an
    unmeasured speed.
  - `torch.compiler.disable`, but for a different reason than kda's. This family used to
    go undecorated on the theory that fla's `causal_conv1d` is plain Python around an
    `autograd.Function` and so is this one, so Dynamo would treat the two alike and the
    graph-break count of a compiled block would not change. Production falsified that on
    the 30M mainline ladder (2026-09-04): Dynamo took OUR `autograd.Function` down its
    `trace_backward_graph` path — two tensor arguments and nothing else is exactly the
    shape it agrees to speculate, where fla's eleven-argument Function is not — and
    speculating `cconv_bwd` died on `dy.stride(0)` with a symbolic stride
    (`AssertionError: Cannot construct ConstantVariable for value of type torch.SymInt`),
    a Dynamo bug we cannot fix from here. The decorator makes the entry point opaque, so
    the backward runs eagerly under the autograd engine like kda's does. It also REDUCES
    the break count rather than raising it: the undecorated form already broke twice
    inside this function (at `is_supported`, then at `log_once`) and now breaks once at
    the call. Keep every tensor operation inside the Function anyway — a `.float()` in
    the caller would split a compiled block for no reason.
"""

from __future__ import annotations

import logging

import torch

from .._common import support

__all__ = ["causal_conv1d", "is_supported", "warmup"]

log = logging.getLogger(__name__)

# Written for W <= 4 taps (every ladder config); narrower W zero-weights the missing taps.
MAX_W = 4

# Arguments this package understands. Anything else present and non-default -> fla.
_HANDLED = frozenset({"x", "weight", "activation", "backend", "output_final_state"})
# Present in fla's signature, and each one non-None/non-False forces the fallback.
_UNSUPPORTED = frozenset({
    "bias", "residual", "initial_state", "cu_seqlens", "cu_seqlens_cpu", "chunk_indices",
    "cp_context",
})
_ACTIVATIONS = ("silu", "swish")


def is_supported(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    backend: str | None = "triton",
    output_final_state: bool | None = False,
    **kwargs,
) -> tuple[bool, str | None]:
    """Can this call use our kernels? Returns (ok, reason-if-not).

    The reason strings are meant to end up in a training log verbatim.
    """
    reason = support.triton_unsupported_reason(x, "cconv", min_major=9)
    if reason is not None:
        return False, reason

    for name in _UNSUPPORTED:
        val = kwargs.get(name)
        if val is not None:
            return False, f"{name}={type(val).__name__} is not implemented here"
    for name in kwargs:
        if name not in _HANDLED and name not in _UNSUPPORTED:
            return False, f"unrecognized argument {name!r} (fla may have grown a flag)"

    if output_final_state:
        return False, "output_final_state=True (the training path keeps no conv state)"
    if activation not in _ACTIVATIONS:
        return False, f"activation={activation!r} (only silu/swish, the KDA layer's contract)"
    if backend != "triton":
        return False, f"backend={backend!r} was asked for explicitly"
    if weight is None:
        return False, "weight=None"
    if x.dim() != 3:
        return False, f"x must be [B, T, D], got {x.dim()} dims"
    D = x.shape[-1]
    if weight.dim() != 2 or weight.shape[0] != D:
        return False, f"weight must be [D={D}, W], got {list(weight.shape)}"
    W = weight.shape[1]
    if W > MAX_W:
        return False, f"W={W} (the register ring is written for W <= {MAX_W})"
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {x.dtype} (only bf16 and fp16 x)"
    return True, None


def _make_fn():
    """Built lazily: the kernel module imports triton, and importing this package must stay
    cheap on a machine that will never call it."""
    from ._kernels import strip

    class CausalConv1dStrip(torch.autograd.Function):
        """Residuals are fla's: x and the weight. The backward recomputes the pre-activation
        in registers from the x it needs for dw anyway, so unlike fla it saves nothing
        else and re-runs nothing."""

        @staticmethod
        def forward(ctx, x, w):
            y = strip.cconv_fwd(x, w)
            ctx.save_for_backward(x, w)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, w = ctx.saved_tensors
            dx, dw = strip.cconv_bwd(x, w, dy)
            return dx, dw

    return CausalConv1dStrip


_FN = None


@torch.compiler.disable
def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    backend: str | None = "triton",
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context=None,
    **kwargs,
):
    """Drop-in for ``fla.modules.convolution.causal_conv1d``. See that function for the
    full argument docs. Returns ``(y, None)`` on our path, exactly fla's contract when
    ``output_final_state`` is False.

    ``@torch.compiler.disable``: Dynamo must not speculate this family's backward — see
    the module docstring for the crash that taught us so. Unlike kda's decorator this one
    ADDS a break relative to fla (whose conv is traceable), while removing the two the
    undecorated form already cost inside this function. Keep every tensor operation
    inside the Function."""
    global _FN
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

    fla_kwargs = dict(
        x=x, weight=weight, bias=bias, residual=residual, initial_state=initial_state,
        output_final_state=output_final_state, activation=activation, backend=backend,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=chunk_indices,
        cp_context=cp_context, **kwargs,
    )
    ok, reason = is_supported(
        x, weight, activation=activation, backend=backend,
        output_final_state=output_final_state, bias=bias, residual=residual,
        initial_state=initial_state, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices, cp_context=cp_context, **kwargs,
    )
    if not ok:
        support.log_once(
            f"kernel-fun cconv: falling back to fla — {reason}", logging.WARNING
        )
        return fla_causal_conv1d(**fla_kwargs)

    assert weight is not None
    # fla's input_guard makes every tensor but x contiguous; the squeezed nn.Conv1d weight
    # already is, so this is a no-op in production and a correctness guard elsewhere.
    w = weight if weight.is_contiguous() else weight.contiguous()
    B, T, D = x.shape
    support.log_versions_once()
    support.log_once(
        f"kernel-fun cconv: engaged (B={B} T={T} D={D} W={w.shape[1]} {x.dtype} "
        f"weight={w.dtype})"
    )
    if _FN is None:
        _FN = _make_fn()
    # fla's input_guard also enters the tensor's device; Triton launches on the current one.
    with torch.cuda.device(x.device):
        y = _FN.apply(x, w)
    return y, None


def warmup(
    *,
    B: int,
    T: int,
    D: tuple[int, ...] = (2048, 4096),
    W: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> float:
    """Autotune both kernels for the training shapes, so step 1 does not.

    The autotune key is (D, W, B, T): pass the real microbatch and sequence length, and
    every channel count the layer convolves (q/k at n_heads*head_dim, v at expand_v times
    that). Each new key costs ~24 trial launches per kernel — a few seconds — which is
    exactly what a reported "regression" at step 1 looks like. Also runs the fla
    compatibility probe. Returns the elapsed seconds — log it.
    """
    import time

    from .._common.compat import check_fla
    from ._kernels import strip

    check_fla()
    t0 = time.perf_counter()
    torch.manual_seed(0)
    for d in D:
        x = torch.randn(B, T, d, device=device, dtype=dtype, requires_grad=True)
        w = (torch.rand(d, W, device=device, dtype=torch.float32) * 2 - 1) * W ** -0.5
        w.requires_grad_(True)
        y, _ = causal_conv1d(x, w, activation="silu")
        y.float().square().sum().backward()
    torch.cuda.synchronize()

    # A warmup that autotuned nothing hides the cost it exists to move.
    empty = [
        k.fn.__name__ for k in (strip.cconv_fwd_strip, strip.cconv_bwd_strip)
        if not getattr(k, "cache", None)
    ]
    if empty:
        raise RuntimeError(
            f"kernel-fun cconv warmup autotuned nothing for {empty} — the call fell back "
            f"to fla (KERNEL_FUN_DEBUG=1 logs why), so step 1 will still pay it."
        )
    return time.perf_counter() - t0
