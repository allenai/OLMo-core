"""A vendored copy of the ``kernel-fun`` package, kda family only.

Source: ``allenai/kernel-fun-2`` @ ``5b95aaf`` (2026-08-24),
``packages/kernel-fun/src/kernel_fun/``. The subtree below — ``_common/`` and ``kda/`` —
is **byte-identical** to that package, so when OLMo-core takes a dependency on it this
directory is deleted and the import below becomes ``from kernel_fun.kda import ...``.
Do not edit anything under ``_common/`` or ``kda/`` here: those edits are lost on the next
vendor, and the fix belongs upstream where the benchmark ladder can measure it.

What it does, and what it costs
-------------------------------
:func:`cute_chunk_kda` has ``fla.ops.kda.chunk_kda``'s signature and return contract. Any
call it does not implement — wrong architecture, wrong shape, packed documents, a flag it
has never seen, a CUDA graph capture — is forwarded to fla verbatim, so it changes how
fast a model trains and not what it computes beyond kernel-level rounding.

Measured on B300 at the production shape (B16 x T8192 x H=HV16, K128/V256, chunk 64):

    fwd+bwd, gate and q/k norm in-op (what this layer calls)   24.07ms vs fla 37.20  1.545x
    fwd+bwd, pre-computed gate                                 22.14ms vs fla 35.57  1.606x
    forward only                                                5.75ms vs fla  7.55  1.313x

Ours are the forward scan+readout and four of the backward's seven stages; the rest are
fla's own kernels at fla's own stage boundaries. ``kda/chain.py`` is the stage table and
``ALGORITHM.md`` is the math each one implements.

Switches, all read per call
---------------------------
``KERNEL_FUN_DISABLE=1`` (or ``KERNEL_FUN_KDA_DISABLE=1``) forwards everything to fla —
the 2am switch, and the control arm that separates "the wrapper" from "the kernels".
``KERNEL_FUN_DEBUG=1`` logs why a call fell back. The old ``OLMO_CUTE_KDA_*`` bisection
knobs are gone: they selected between kernels that no longer exist, and bisecting a stage
now means reaching for the research ladder, which keeps all of them.

One INFO line per process says whether the kernels engaged and, if not, exactly why —
a silent fallback reads as a correct 1.00x, and "did they actually run?" is the most
expensive question a kernel port can leave open.
"""

from .kda import chunk_kda as cute_chunk_kda
from .kda import is_supported, warmup

__all__ = ["cute_chunk_kda", "is_supported", "warmup", "versions"]


def versions() -> dict[str, str]:
    """Everything that decides what these kernels compute and how fast.

    Worth logging once per run: two of these (the CuTe DSL and cuda-python) are pinned by
    nothing in ``pyproject.toml`` and arrive with the base image. When a run is slower than
    the last one, this is the first thing to diff.
    """
    import importlib

    out: dict[str, str] = {"kernel_fun": "0.1.0.dev0 (vendored @ 5b95aaf)"}
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
