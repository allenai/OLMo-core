"""Opt-in fixed existing FLA launch configurations for offline inference diagnostics."""

import importlib
import os

_INSTALLED = False


def install():
    """Remove autotuner choice as a numerical variable; never introduce a new kernel."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("FLA_CACHE_MODE", "disabled") != "disabled":
        raise RuntimeError("Fixed diagnostic launches require FLA_CACHE_MODE=disabled")
    from triton.runtime.autotuner import Autotuner

    settings = (
        ("fla.modules.l2norm", "l2norm_fwd_kernel", "BT", 8, 8),
        ("fla.ops.kda.gate", "kda_gate_chunk_cumsum_vector_kernel", "BS", 32, 4),
        (
            "fla.ops.kda.chunk_intra_token_parallel",
            "chunk_kda_fwd_kernel_intra_token_parallel",
            "BH",
            1,
            1,
        ),
    )
    for module, name, key, value, warps in settings:
        operator = getattr(importlib.import_module(module), name)
        while not isinstance(operator, Autotuner):
            operator = operator.fn
        choices = [
            c for c in operator.configs if c.kwargs.get(key) == value and c.num_warps == warps
        ]
        if len(choices) != 1:
            raise RuntimeError(f"Missing exact existing FLA configuration: {name}")
        operator.configs = choices
        operator.cache.clear()
        print("HERO_FIXED_FLA_CONFIG", name, str(choices[0]), flush=True)
    _INSTALLED = True
