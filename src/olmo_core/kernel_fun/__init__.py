"""A vendored copy of the ``kernel-fun`` package.

Source: ``allenai/kernel-fun`` @ ``0c5f7f2`` (2026-09-04),
``packages/kernel-fun/src/kernel_fun/``. The subtrees below — ``_common/``, ``kda/`` and
``cconv/`` — are **byte-identical** to that package; only this file differs, and only to
record where the copy came from. This directory exists so a training run can use these
kernels without installing a private git dependency; when OLMo-core takes the dependency
instead (branch ``caleb/cute-kda``) it is deleted and ``from olmo_core.kernel_fun.kda
import ...`` becomes ``from kernel_fun.kda import ...``, which is the only wiring
difference between the two branches.

Do not edit anything under ``_common/``, ``kda/`` or ``cconv/`` here: those edits are lost
on the next vendor, and the fix belongs upstream where the benchmark ladder can measure
it. To refresh, re-copy the tree at a new commit and update ``VENDORED_FROM``.

--- the package's own docstring follows ---

kernel-fun — CuTe/Triton kernels for linear-attention ops on Blackwell.

Each op family is a subpackage exporting a drop-in for its flash-linear-attention
counterpart, with the same signature and the same return contract:

    from olmo_core.kernel_fun.kda import chunk_kda        # the KDA chunk kernel (CuTe + Triton)
    from olmo_core.kernel_fun.cconv import causal_conv1d  # the KDA short conv + silu (Triton)

A call the package does not implement — wrong architecture, wrong shape, a flag we have
never seen, a CUDA graph capture — is forwarded to fla verbatim, so installing this can
change how fast a model trains but not what it computes beyond kernel-level rounding.

Nothing is imported eagerly: importing this package must stay cheap on a machine with no
GPU, so torch, triton and the CuTe DSL are pulled in by the family that needs them.

Two environment switches, read per call:
    KERNEL_FUN_DISABLE=1        forward everything to fla (per family: _KDA_, _CCONV_, ...)
    KERNEL_FUN_DEBUG=1          log why a call fell back
"""

from __future__ import annotations

__all__ = ["__version__", "VENDORED_FROM", "versions"]

__version__ = "0.2.0.dev0"

#: The ``allenai/kernel-fun`` commit this tree was copied from. Reported by
#: :func:`versions`, so a training log says which copy ran and not just which release.
VENDORED_FROM = "0c5f7f24f03a32f1708135a3b890eeec67f64ff8"


def versions() -> dict[str, str]:
    """Everything that decides what this package computes and how fast.

    Worth logging once per training run: two of these (the CuTe DSL and cuda-python) are
    not pinned by any requirement and arrive with the base image, and no bench row in the
    research repo records them either. When a run is slower than the last one, this is the
    first thing to diff.
    """
    import importlib

    out: dict[str, str] = {"kernel_fun": f"{__version__} (vendored @ {VENDORED_FROM[:7]})"}
    for name, mod in (
        ("torch", "torch"),
        ("triton", "triton"),
        ("fla", "fla"),
        ("cutlass", "cutlass"),
        ("cuda-python", "cuda.bindings"),
    ):
        try:
            m = importlib.import_module(mod)
            out[name] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            out["device"] = f"{torch.cuda.get_device_name()} sm{cap[0]}{cap[1]}"
    except Exception:
        pass
    return out
