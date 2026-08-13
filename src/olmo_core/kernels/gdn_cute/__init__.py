"""
CuTe DSL kernels for the chunked gated delta rule (gated delta net / GDN).

These are drop-in replacements for a subset of `flash-linear-attention
<https://github.com/fla-org/flash-linear-attention>`_'s Triton kernels, written against the
CuTe DSL for Blackwell (``sm_100+``, i.e. ``tcgen05`` MMA + TMEM). They were developed and
benchmarked in the `kernel-fun-2 <https://github.com/allenai/kernel-fun-2>`_ repo
(``kernels/gdn/ideas/003-cute-bwd``) and are copied here verbatim apart from the
output-buffer change described below.

The files are:

- ``kernel_fwd.py`` — the fused forward (state scan + output readout), i.e. fla's
  ``chunk_gated_delta_rule_fwd_o`` + ``chunk_fwd_h`` in one kernel. The chunk-local WY
  representation (gate cumsum, ``A``, ``w``, ``u``) is still fla's Triton code.
- ``kernel_fwdh.py`` — fla's ``chunk_gated_delta_rule_fwd_h``, the backward's state-scan
  recompute.
- ``kernel_dhu.py`` — fla's ``chunk_gated_delta_rule_bwd_dhu``, the reverse state scan.
- ``kernel_dqkwg.py`` — fla's ``chunk_bwd_dqkwg``.
- ``kernel_wy_bwd.py`` — fla's ``prepare_wy_repr_bwd``.

Every backward stage wrapper falls back to fla's own kernel on shapes it does not cover, so
the stage table in :mod:`olmo_core.ops.gdn` is always safe to call. The forward has no such
fallback — :func:`olmo_core.ops.gdn.gdn_cute_supported` gates it instead.

Measured on a B300 at the production shape (``B=16, T=8192, H=HV=16, K=128, V=256``,
``chunk_size=64``, bf16): forward 1.62x over fla, forward+backward 1.27x.

**Divergence from the source repo.** In kernel-fun-2 each wrapper's output tensors were
allocated once and owned by the marshaling cache, so two calls with the same input layouts
returned the *same* buffer. That is fine for a benchmark harness that builds fresh inputs per
arm, but fatal in a real model: every GDN layer in a stack has identical shapes, so layer 1's
forward would overwrite the ``o`` that layer 0 saved for backward. Here the outputs are
allocated fresh per call and only their memref descriptors are re-pointed, which keeps the
~0.28ms/call marshaling saving without the aliasing. (kernel-fun-2 has since fixed this the
same way, so the two copies agree again; the regression test is
``test_chunk_gated_delta_rule_cute_does_not_alias_outputs``.)

**Re-vendoring.** Take these from kernel-fun-2 at ``cute-gdn`` e711e32 or later. Earlier
commits carry a race in ``kernel_wy_bwd``'s ``dg`` epilogue that only fires when a CTA owns
more than one chunk — which never happens at a shape small enough to unit-test, and always
happens at production sizes, where it put ``dg`` three orders of magnitude outside its
tolerance and made it nondeterministic run to run. See
``test_chunk_gated_delta_rule_cute_matches_fla_with_many_chunks_per_cta``, which forces the
multi-chunk path at a small shape via the ``GDN_WYBWD_CTAS`` override.
"""

import torch

__all__ = [
    "GDN_CUTE_MIN_COMPUTE_CAPABILITY",
    "has_cutlass_dsl",
    "has_gdn_cute",
]

#: Minimum CUDA compute capability. The kernels use ``tcgen05`` MMA and TMEM, which are
#: Blackwell-only; on anything older they will not even compile.
GDN_CUTE_MIN_COMPUTE_CAPABILITY = 10


def has_cutlass_dsl() -> bool:
    """
    Check if the CuTe DSL (``nvidia-cutlass-dsl``) is installed.
    """
    try:
        import cutlass.cute  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def has_gdn_cute(device: torch.device | None = None) -> bool:
    """
    Check if the CuTe GDN kernels can run here: the CuTe DSL is installed and the device is
    Blackwell or newer.

    :param device: The device to check. Defaults to the current CUDA device.
    """
    if not torch.cuda.is_available() or not has_cutlass_dsl():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= GDN_CUTE_MIN_COMPUTE_CAPABILITY
