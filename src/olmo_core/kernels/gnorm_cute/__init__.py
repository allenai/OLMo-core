"""
CuTe DSL kernels for the gated RMS norm (fla's ``FusedRMSNormGated``).

These replace `flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_'s
Triton ``fused_norm_gate`` kernels for the one configuration the gated DeltaNet path uses: RMS
norm (no mean subtraction), a weight but no bias, no residual, ``prenorm=False``. They were
developed and benchmarked in the `kernel-fun-2 <https://github.com/allenai/kernel-fun-2>`_ repo
(``kernels/gnorm/ideas/002-cute-bwd``) and are copied here verbatim.

The files are:

- ``kernel_fwd.py`` — ``y = rmsnorm(x) * weight * act(g)``, plus the per-row fp32 ``rstd`` the
  backward consumes. One warp per row of the flattened ``(R, D)`` view.
- ``kernel_bwd.py`` — ``dx``, ``dg`` and ``dw`` in one compiled callable (the streaming pass
  plus two ``dw`` fold kernels, three launches on one host dispatch). ``dw`` is reduced through
  a fixed ``[num_ctas, D]`` split and fixed-order folds, never atomics, so it is deterministic.

These are plain SIMT kernels — no ``tcgen05``, no TMEM, no TMA — so the arch requirement is
only whatever the CuTe DSL itself needs. They are
gated to Blackwell anyway (see :data:`GNORM_CUTE_MIN_COMPUTE_CAPABILITY`) because that is where
they were measured; :mod:`olmo_core.ops.gnorm` falls back to fla everywhere else.

Measured on a B300 at the production shape (``B=16, T=8192, HV=16, D=256``, bf16): forward
1.10x over fla and backward 1.26x in isolation, both at ~95% of streaming bandwidth. At shorter
sequences the forward reaches 1.66-1.77x, where fla is launch-bound and the layout-keyed call
cache in these wrappers is doing the work. Full forward+backward through autograd lands at
1.09-1.14x on every shape measured: a ~0.9ms fixed per-backward cost in torch's autograd engine
(measured with a no-op backward ``Function``) sits in both arms and dominates the ladder rows.

**Re-vendoring.** Take these from kernel-fun-2 at ``main`` b57a2b3 or later, as-is — isort,
black, ruff and mypy are pointed away from this directory so they stay byte-identical to the
source repo and re-copying stays a plain ``cp``. Two constants there are load-bearing here:
``kernel_fwd.RPB`` (rows per CTA) and ``kernel_bwd._RED_CTAS`` (the ``dw`` fold width) decide
which row counts the kernels accept, and :mod:`olmo_core.ops.gnorm` asserts its copies of them
still match on first use, so a re-vendoring that retunes either fails loudly rather than
mis-gating.
"""

import torch

__all__ = [
    "GNORM_CUTE_MIN_COMPUTE_CAPABILITY",
    "has_cutlass_dsl",
    "has_gnorm_cute",
]

#: Minimum CUDA compute capability. The kernels themselves are plain SIMT and would very likely
#: run on older architectures, but Blackwell is where they were validated and tuned (the grid
#: caps in both files come from sweeps on a B300), so that is what the ``auto`` backend claims.
GNORM_CUTE_MIN_COMPUTE_CAPABILITY = 10


def has_cutlass_dsl() -> bool:
    """
    Check if the CuTe DSL (``nvidia-cutlass-dsl``) is installed.
    """
    try:
        import cutlass.cute  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def has_gnorm_cute(device: torch.device | None = None) -> bool:
    """
    Check if the CuTe gated RMS norm kernels can run here: the CuTe DSL is installed and the
    device is Blackwell or newer.

    :param device: The device to check. Defaults to the current CUDA device.
    """
    if not torch.cuda.is_available() or not has_cutlass_dsl():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= GNORM_CUTE_MIN_COMPUTE_CAPABILITY
