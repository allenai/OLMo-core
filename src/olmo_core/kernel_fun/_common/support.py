"""Can this call use our kernels — and if not, exactly why?

Every public entry point in this package answers that question before doing anything, and
delegates to flash-linear-attention when the answer is no. The reason string is the
important half: a silent fallback reads as a correct 1.00x result, and the last time
kernels from this repo landed in training the single most expensive question was "did they
actually run?".

Probes that touch the driver or import cutlass are cached — the previous port called
`import cutlass.cute` on every forward. Probe functions are module-level and cached so a
test can monkeypatch one and clear its cache to exercise the fallback on any GPU.
"""

from __future__ import annotations

import logging
import os
from functools import cache

import torch

log = logging.getLogger(__name__)

# Serial-scan kernels need a full GPU: a one-CTA-per-(chunk, b*hv) kernel at a 64-CTA grid
# lost 1.4ms to fla in the gdn ladder. Below this the fla path is genuinely faster, so the
# gate is a performance decision, not a capability one.
MIN_CTAS = 256

_logged: set[str] = set()


def log_once(message: str, level: int = logging.INFO) -> None:
    if message in _logged:
        return
    _logged.add(message)
    log.log(level, message)


def log_versions_once() -> None:
    """Log `versions()` the first time any family engages.

    This belongs HERE, not in the caller's module: every entry point that calls it is
    already `torch.compiler.disable`d, so the line costs no graph break, whereas the same
    call from a compiled `forward` splits the block (and, in olmo-core, hit `lru_cache` on
    a dict return and raised `TypeError: unhashable type: 'dict'`). Two of these versions
    — the CuTe DSL and cuda-python — are pinned by nothing and ride in with the image,
    so when a run is slower than the last one this is the first thing to diff.
    """
    from .. import versions

    log_once(f"kernel-fun {versions()}")


@cache
def has_cute() -> bool:
    """Is the CuTe DSL importable? Cached: the import pulls MLIR and costs seconds."""
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except Exception:  # pragma: no cover - environment-dependent
        return False
    return True


@cache
def arch_at_least(device_index: int, major: int) -> bool:
    """For the Triton-only families: any Blackwell-or-newer datacenter part, or Hopper.

    Those kernels need no tcgen05 and no CuTe DSL, so the sm100 gate below would deny them
    for no reason; but nobody has timed them below sm100, so callers state their floor.
    """
    return torch.cuda.get_device_capability(device_index)[0] >= major


@cache
def arch_ok(device_index: int) -> bool:
    """sm100 exactly — Blackwell datacenter (B200/B300).

    Not `major >= 10`: sm_120 is consumer Blackwell and has no tcgen05, so the MMA kernels
    would fail inside cute.compile rather than fall back. Not `major == 10 and minor == 0`
    either: B300 reports (10, 3).
    """
    return torch.cuda.get_device_capability(device_index)[0] == 10


def disabled(family: str) -> bool:
    """KERNEL_FUN_DISABLE=1 kills every family; KERNEL_FUN_<FAMILY>_DISABLE=1 kills one.

    Read per call, never at import: the point of a kill switch is that someone can set it
    on a run that is already failing, and an import-time read would depend on which module
    got imported first.
    """
    return (
        os.environ.get("KERNEL_FUN_DISABLE", "0") == "1"
        or os.environ.get(f"KERNEL_FUN_{family.upper()}_DISABLE", "0") == "1"
    )


def debug() -> bool:
    return os.environ.get("KERNEL_FUN_DEBUG", "0") == "1"


def capturing() -> bool:
    """Is a CUDA graph being captured on this stream?

    The call cache pokes a host-side descriptor at launch time, so a captured graph would
    replay the capture-time pointers against whatever tensors a later replay passes —
    silently wrong answers, not a crash. Anything under capture goes to fla.
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # pragma: no cover - older torch without the query
        return False


def _basic_unsupported_reason(t: torch.Tensor, family: str) -> str | None:
    if disabled(family):
        return f"KERNEL_FUN_{family.upper()}_DISABLE / KERNEL_FUN_DISABLE is set"
    if not t.is_cuda:
        return "not a CUDA tensor"
    if capturing():
        # The CuTe call cache pokes pointers at launch time, and the Triton autotuners
        # (ours and fla's) run trial launches — neither belongs inside a capture.
        return "CUDA graph capture"
    return None


def common_unsupported_reason(t: torch.Tensor, family: str) -> str | None:
    """The gates every CuTe family shares. Family-specific shape checks live with the family."""
    reason = _basic_unsupported_reason(t, family)
    if reason is not None:
        return reason
    if not arch_ok(t.device.index or 0):
        cap = torch.cuda.get_device_capability(t.device.index or 0)
        return f"device capability sm{cap[0]}{cap[1]} is not sm100 (B200/B300)"
    if not has_cute():
        return "the CUTLASS CuTe DSL is not installed"
    return None


def triton_unsupported_reason(t: torch.Tensor, family: str, min_major: int = 9) -> str | None:
    """The same, for a family whose kernels are all Triton: no CuTe, a lower arch floor."""
    reason = _basic_unsupported_reason(t, family)
    if reason is not None:
        return reason
    if not arch_at_least(t.device.index or 0, min_major):
        cap = torch.cuda.get_device_capability(t.device.index or 0)
        return (
            f"device capability sm{cap[0]}{cap[1]} is below sm{min_major}0 "
            f"(measured on sm100; untimed below it)"
        )
    return None
