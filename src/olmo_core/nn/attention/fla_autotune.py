"""
Stop FLA's Triton kernels from re-autotuning at every new sequence length.

Several flash-linear-attention kernels on the Gated DeltaNet path (``causal_conv1d``, ``l2norm``,
the gated RMSNorm) autotune with ``NB`` -- a block count derived from the sequence length -- in
their autotune key. A new ``NB`` means a fresh autotune: about 25 s of kernel compiles on an H100
for a Qwen3.5-4B forward. Training sees a handful of fixed lengths, so it never notices; generation
over an eval set of natural-length prompts sees a new ``NB`` on almost every example. Measured on a
fast-compressive-landmark Qwen3.5-4B: prefill+decode 0.5 s per example, plus ~10-25 s of
re-autotuning, i.e. ~95% of the eval's wall-clock.

``NB`` only picks among launch configs whose grids are computed from the tuned meta-parameters, so
dropping it from the key changes speed, never results: the first tuning is reused at every length.
"""

import logging
import sys

log = logging.getLogger(__name__)

__all__ = ["freeze_fla_length_autotune"]

#: autotune-key entries that are derived from sequence length
_LENGTH_KEYS = frozenset({"NB"})


def freeze_fla_length_autotune() -> int:
    """
    Remove length-derived entries from the autotune keys of every loaded FLA kernel.

    Idempotent, and a no-op when FLA or Triton is not importable.

    :returns: Number of autotuners whose key was changed.
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return 0
    changed = 0
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "fla" or name.startswith("fla.")):
            continue
        for obj in list(vars(module).values()):
            if isinstance(obj, Autotuner) and _LENGTH_KEYS & set(obj.keys):
                obj.keys = [k for k in obj.keys if k not in _LENGTH_KEYS]
                changed += 1
    if changed:
        log.info(f"FLA: dropped length-derived autotune keys on {changed} kernel(s)")
    return changed
