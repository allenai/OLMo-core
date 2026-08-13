"""
The gated RMS norm, with a CuTe DSL backend and a
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_ fallback.

:func:`rms_norm_gated` is the entry point. Per row of ``D`` features it computes

.. code-block::

    y = x * rsqrt(mean(x^2) + eps) * weight * act(g)

with ``act`` either ``swish`` (``g * sigmoid(g)``) or ``sigmoid`` — i.e. exactly what fla's
``FusedRMSNormGated`` computes, and what :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet`
applies to the delta-rule output. It routes to one of:

- ``GNormBackend.cute`` — the CuTe DSL kernels in :mod:`olmo_core.kernels.gnorm_cute`, which
  replace both directions of fla's ``fused_norm_gate``. Measured on a B300 at
  ``B=16, T=8192, HV=16, D=256`` in bf16: forward 1.10x, backward 1.26x in isolation.
- ``GNormBackend.fla`` — fla's Triton kernels, unchanged.

``GNormBackend.auto`` (the default) picks ``cute`` when the CuTe DSL is installed, the GPU is
Blackwell or newer, and the shapes are in the supported envelope (see
:func:`gnorm_cute_unsupported_reason`); otherwise it falls back to ``fla``.

Which one a given shape resolved to is logged at ``INFO`` by this module's logger
(``olmo_core.ops.gnorm``) the first time it dispatches, and a fallback under ``auto`` is logged
at ``WARNING`` with the reason. Grep a run's logs for ``gated RMS norm kernels``.

The scope here is narrower than fla's module: RMS norm only (no mean subtraction), a weight and
no bias, no residual and no ``prenorm``. Those are the settings the gated DeltaNet path uses,
and the CuTe kernels implement only those; anything else has to keep calling fla directly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

from olmo_core.config import StrEnum
from olmo_core.kernels.gnorm_cute import has_gnorm_cute

log = logging.getLogger(__name__)

__all__ = [
    "GNormBackend",
    "GNORM_CUTE_ACTIVATIONS",
    "GNORM_CUTE_MAX_DIM",
    "GNORM_CUTE_ROWS_PER_BLOCK",
    "GNORM_CUTE_MIN_ROW_BLOCKS",
    "rms_norm_gated",
    "gnorm_cute_unsupported_reason",
]


#: The activations the CuTe kernels implement, matching fla's own set for this module.
GNORM_CUTE_ACTIVATIONS = ("swish", "silu", "sigmoid")

#: The largest row length (``D``) the kernels handle: the row is held in registers, one warp
#: wide, so ``D`` also has to be a multiple of 32.
GNORM_CUTE_MAX_DIM = 512

#: Rows per CTA, i.e. warps per CTA. A copy of ``kernel_fwd.RPB``; the row count has to be a
#: multiple of it because the kernels' only predicate is the block-stride loop bound.
GNORM_CUTE_ROWS_PER_BLOCK = 8

#: The first-level fold width of the ``dw`` reduction, a copy of ``kernel_bwd._RED_CTAS``. The
#: backward assumes at least this many row-blocks exist, so the row count must be at least
#: ``GNORM_CUTE_ROWS_PER_BLOCK * GNORM_CUTE_MIN_ROW_BLOCKS``.
GNORM_CUTE_MIN_ROW_BLOCKS = 32


class GNormBackend(StrEnum):
    """
    An enumeration of the available gated RMS norm kernel backends.
    """

    auto = "auto"
    """
    Use :data:`cute` where it's available and supports the shapes, otherwise :data:`fla`.
    """

    cute = "cute"
    """
    The CuTe DSL kernels in :mod:`olmo_core.kernels.gnorm_cute`. Requires a Blackwell
    (``sm_100+``) GPU and ``nvidia-cutlass-dsl``, and raises if the shapes aren't supported.
    """

    fla = "fla"
    """
    fla's Triton kernels.
    """


def gnorm_cute_unsupported_reason(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "swish",
) -> Optional[str]:
    """
    Check whether the CuTe gated RMS norm kernels can handle these inputs.

    :param x: The input of shape ``(..., D)``.
    :param g: The gate, same shape as ``x``.
    :param weight: The norm weight of shape ``(D,)``.
    :param activation: The gate activation.

    :returns: ``None`` if the inputs are supported, otherwise a human-readable reason why not.
    """
    if not has_gnorm_cute(x.device):
        return "requires nvidia-cutlass-dsl and a Blackwell (sm_100+) GPU"
    if activation not in GNORM_CUTE_ACTIVATIONS:
        return f"activation must be one of {GNORM_CUTE_ACTIVATIONS}, got {activation!r}"
    if x.dtype not in (torch.bfloat16, torch.float16):
        return f"x must be bf16 or fp16, got {x.dtype}"
    if g.dtype != x.dtype:
        return f"x and g must share a dtype, got {x.dtype}/{g.dtype}"
    if g.shape != x.shape:
        return f"x and g must have the same shape, got {tuple(x.shape)}/{tuple(g.shape)}"

    D = x.shape[-1]
    if weight.shape != (D,):
        return f"weight must have shape ({D},), got {tuple(weight.shape)}"
    if D % 32 != 0 or D > GNORM_CUTE_MAX_DIM:
        return f"the row length must be a multiple of 32 and at most {GNORM_CUTE_MAX_DIM}, got {D}"

    # One warp per row, `GNORM_CUTE_ROWS_PER_BLOCK` rows per CTA, and the backward's `dw` fold
    # assumes a full first level, so short inputs are fla's.
    rows = x.numel() // D
    min_rows = GNORM_CUTE_ROWS_PER_BLOCK * GNORM_CUTE_MIN_ROW_BLOCKS
    if rows % GNORM_CUTE_ROWS_PER_BLOCK != 0:
        return f"the row count must be a multiple of {GNORM_CUTE_ROWS_PER_BLOCK}, got {rows}"
    if rows < min_rows:
        return f"the row count must be at least {min_rows}, got {rows}"
    return None


_KERNELS: Optional[Tuple[Any, Any]] = None


def _kernels() -> Tuple[Any, Any]:
    """
    The vendored forward and backward modules, imported lazily so this module imports on
    machines without the CuTe DSL.

    Also the one place that checks the constants this module mirrors from those files. They
    decide which row counts :func:`gnorm_cute_unsupported_reason` lets through, and a
    re-vendoring is a wholesale file copy that could retune either of them, so assert rather
    than silently gate on stale numbers.
    """
    global _KERNELS
    if _KERNELS is None:
        from olmo_core.kernels.gnorm_cute import kernel_bwd, kernel_fwd

        assert kernel_fwd.RPB == kernel_bwd.RPB == GNORM_CUTE_ROWS_PER_BLOCK, (
            f"the vendored kernels changed their rows-per-CTA to "
            f"{kernel_fwd.RPB}/{kernel_bwd.RPB}; update GNORM_CUTE_ROWS_PER_BLOCK"
        )
        assert kernel_bwd._RED_CTAS == GNORM_CUTE_MIN_ROW_BLOCKS, (
            f"the vendored backward changed its dw fold width to {kernel_bwd._RED_CTAS}; "
            f"update GNORM_CUTE_MIN_ROW_BLOCKS"
        )
        _KERNELS = (kernel_fwd, kernel_bwd)
    return _KERNELS


def _kernel_ready(t: torch.Tensor) -> torch.Tensor:
    """
    Make a tensor safe to hand to the kernels: contiguous, and 16B-aligned.

    The kernels index gmem through CuTe views built from shape/stride and re-point them with
    raw pointer writes that assert 16B alignment, so a contiguous *view* into the middle of
    someone else's storage — a slice of a fused activation, say — would trip the assert rather
    than run slowly. Copying it is both rare and cheap next to being wrong.
    """
    t = t.contiguous()
    if t.data_ptr() % 16 != 0:
        t = t.clone()
    return t


class RMSNormGatedCute(torch.autograd.Function):
    """
    The autograd wrapper around the CuTe gated RMS norm kernels.

    Mirrors fla's ``LayerNormGatedFunction`` for the ``is_rms_norm=True``, no-bias,
    no-residual, no-prenorm case: the forward saves ``x``, ``g``, the weight and the per-row
    ``rstd``, and the backward recomputes everything else from them. Call
    :func:`rms_norm_gated` rather than this directly.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        activation: str,
        eps: float,
    ):
        kernel_fwd, _ = _kernels()

        x = _kernel_ready(x)
        g = _kernel_ready(g)
        # The kernels want an fp32 weight. Under mixed precision the parameter already is one,
        # and when it isn't this is a D-element cast, so `dw` comes back fp32 either way and is
        # cast to the parameter's dtype in the backward — the same contract as fla's, which
        # ends its backward with `dw.sum(0).to(weight.dtype)`.
        w32 = _kernel_ready(weight if weight.dtype == torch.float32 else weight.float())

        y, rstd = kernel_fwd.gnorm_cute_fwd(x, g, w32, activation, float(eps))

        ctx.save_for_backward(x, g, w32, rstd)
        ctx.activation = activation
        ctx.weight_dtype = weight.dtype
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        _, kernel_bwd = _kernels()
        x, g, w32, rstd = ctx.saved_tensors

        dx, dg, dw = kernel_bwd.gnorm_cute_bwd(_kernel_ready(dy), x, g, w32, rstd, ctx.activation)

        return (
            dx.view(x.shape),
            dg.view(g.shape),
            dw.to(ctx.weight_dtype),
            None,  # activation
            None,  # eps
        )


_LOGGED: set = set()


def _log_once(key: Any, level: int, msg: str, *args) -> None:
    """
    Log a dispatch decision the first time it's made.

    Once per distinct decision, not per call: this sits in the forward of every GDN layer of
    every step. Logging is rank0-only by default (see
    :func:`~olmo_core.utils.setup_logging`), so one line per shape per run.
    """
    if key not in _LOGGED:
        _LOGGED.add(key)
        log.log(level, msg, *args)


# Not traceable: the CuTe path re-points memref descriptors with ctypes pointer writes, and the
# fla path disables the compiler for itself anyway. Graph-break here rather than at some
# arbitrary depth inside.
@torch.compiler.disable
def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    activation: str = "swish",
    backend: GNormBackend = GNormBackend.auto,
) -> torch.Tensor:
    """
    The gated RMS norm, dispatched to the best available backend.

    :param x: The input of shape ``(..., D)``, bf16 or fp16 for the CuTe backend.
    :param g: The gate, same shape and dtype as ``x``.
    :param weight: The norm weight of shape ``(D,)``.
    :param eps: The epsilon inside the ``rsqrt``.
    :param activation: The gate activation, ``"swish"`` (a.k.a. ``"silu"``) or ``"sigmoid"``.
    :param backend: Which kernel backend to use.

    :returns: The output, same shape and dtype as ``x``.

    :raises RuntimeError: If ``backend`` names a backend that can't run these inputs.
    """
    # Imported here, not at module scope, for the same reason `olmo_core.ops.gdn` does it:
    # `olmo_core.nn.attention.recurrent` imports this module.
    from olmo_core.nn.attention.flash_linear_attn_api import (
        dispatch_rms_norm_gated,
        has_fla,
    )

    if backend != GNormBackend.fla:
        reason = gnorm_cute_unsupported_reason(x, g, weight, activation=activation)
        if reason is None:
            _log_once(
                ("cute", tuple(x.shape), x.dtype, activation),
                logging.INFO,
                "Using the CuTe gated RMS norm kernels for %s %s rows of %d, activation=%s",
                tuple(x.shape),
                x.dtype,
                x.shape[-1],
                activation,
            )
            return RMSNormGatedCute.apply(x, g, weight, activation, eps)  # type: ignore[return-value]
        if backend == GNormBackend.cute:
            raise RuntimeError(f"the CuTe gated RMS norm kernels cannot be used here: {reason}")
        # auto: fall back, but say so once per distinct reason. Silently running the slow path
        # is how a "faster" branch turns out to have been fla all along.
        _log_once(
            ("fla", reason),
            logging.WARNING,
            "Falling back to the fla gated RMS norm kernels: %s",
            reason,
        )
    else:
        _log_once(
            ("fla", "requested"),
            logging.INFO,
            "Using the fla gated RMS norm kernels (kernel_backend='fla')",
        )

    assert has_fla(), "the gated RMS norm requires flash-linear-attention (fla)"
    return dispatch_rms_norm_gated(x, g, weight, activation=activation, eps=eps)
