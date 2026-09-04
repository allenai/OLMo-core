"""Are the flash-linear-attention internals we call still the ones we were built against?

This package is a partial replacement: several stages of every chain are fla's own kernels,
called by private path, and one of them (`chunk_kda_fwd_kernel_inter_solve_fused`) is a raw
`triton.jit` kernel we launch with our own grid and constexpr list. That is a deliberate
trade — reusing fla's kernels is why our numbers are comparable to fla's launch for launch —
but it means an fla upgrade can change our results without changing our code.

So: check at warmup, not at import. A raise here happens before a training run spends any
time; the same drift discovered at step 40,000 costs a day.

Policy: warn on a version we have not tested, raise on a symbol or signature that moved.
KERNEL_FUN_FALLBACK=1 downgrades the raise to a warning plus a fallback, for whoever needs
the cluster running now rather than correct attribution.
"""

from __future__ import annotations

import inspect
import logging
import os
from functools import cache

log = logging.getLogger(__name__)

# Versions this package has actually been run against, not a guess at compatibility.
TESTED_FLA = {"0.5.2"}

# module path -> symbols we import from it. Kept as data so the check and the docs cannot
# drift apart, and so a new family only has to extend this table.
FLA_SYMBOLS: dict[str, tuple[str, ...]] = {
    "fla.ops.kda.chunk_intra": (
        "chunk_kda_fwd_intra",
        "chunk_kda_bwd_intra",
        "chunk_kda_fwd_kernel_inter_solve_fused",
    ),
    "fla.ops.kda.chunk_intra_token_parallel": ("chunk_kda_fwd_intra_token_parallel",),
    "fla.ops.kda.chunk_bwd": ("chunk_kda_bwd_dAv", "chunk_kda_bwd_wy_dqkg_fused"),
    "fla.ops.kda.wy_fast": ("recompute_w_u_fwd",),
    "fla.ops.kda.gate": ("kda_gate_chunk_cumsum", "kda_gate_bwd"),
    "fla.ops.common.chunk_delta_h": (
        "chunk_gated_delta_rule_fwd_h",
        "chunk_gated_delta_rule_bwd_dhu",
    ),
    "fla.ops.common.gate": ("fused_beta_sigmoid", "fused_beta_sigmoid_bwd"),
    "fla.ops.gla.chunk": ("chunk_gla_fwd_o_gk",),
    "fla.ops.utils": ("chunk_local_cumsum",),
    "fla.ops.utils.constant": ("RCP_LN2",),
    "fla.ops.utils.cache": ("fla_cache_autotune",),
    "fla.ops.utils.op": ("exp2",),
    "fla.utils": (
        "autotune_cache_kwargs",
        "check_shared_mem",
        "input_guard",
        "autocast_custom_fwd",
        "autocast_custom_bwd",
    ),
    "fla.modules.l2norm": ("l2norm_fwd", "l2norm_bwd"),
    # cconv: the public entry point is also the fallback, and its parameter list is what
    # the whitelist in kernel_fun.cconv.ops classifies.
    "fla.modules.convolution": ("causal_conv1d",),
}

# Keywords we pass by name. A rename upstream is a TypeError at the worst possible moment
# otherwise; a reorder of the raw triton kernel's args would be worse — wrong answers.
FLA_KWARGS: dict[tuple[str, str], tuple[str, ...]] = {
    ("fla.ops.kda.wy_fast", "recompute_w_u_fwd"): ("q", "k", "v", "beta", "A", "gk"),
    ("fla.ops.kda.gate", "kda_gate_chunk_cumsum"): (
        "g", "A_log", "dt_bias", "scale", "chunk_size",
    ),
    ("fla.ops.kda.gate", "kda_gate_bwd"): ("g", "A_log", "dt_bias", "dyg"),
    ("fla.ops.kda.chunk_bwd", "chunk_kda_bwd_dAv"): (
        "q", "k", "v", "do", "A", "scale", "chunk_size",
    ),
    ("fla.ops.utils", "chunk_local_cumsum"): ("g", "chunk_size", "reverse"),
    ("fla.modules.convolution", "causal_conv1d"): (
        "x", "weight", "bias", "residual", "initial_state", "output_final_state",
        "activation", "backend", "cu_seqlens", "cu_seqlens_cpu", "chunk_indices",
        "cp_context",
    ),
}


class FlaDriftError(RuntimeError):
    """An fla internal we depend on moved. See the message for which one."""


def _fail(msg: str) -> None:
    if os.environ.get("KERNEL_FUN_FALLBACK", "0") == "1":
        log.warning("kernel-fun: %s (KERNEL_FUN_FALLBACK=1: falling back to fla)", msg)
        return
    raise FlaDriftError(
        f"{msg}\n"
        f"kernel-fun calls flash-linear-attention internals directly; tested against "
        f"{sorted(TESTED_FLA)}. Set KERNEL_FUN_FALLBACK=1 to degrade to fla's own kernels "
        f"instead of raising."
    )


@cache
def check_fla() -> bool:
    """Verify every fla symbol and keyword this package uses. Cached; call from warmup.

    Returns False when drift was found and KERNEL_FUN_FALLBACK downgraded it, so callers
    can route to fla wholesale.
    """
    import importlib

    import fla

    version = getattr(fla, "__version__", "unknown")
    if version not in TESTED_FLA:
        log.warning(
            "kernel-fun: flash-linear-attention %s is untested here (tested: %s); "
            "verify numerics before trusting a training run",
            version, sorted(TESTED_FLA),
        )

    ok = True
    for mod_path, names in FLA_SYMBOLS.items():
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            _fail(f"cannot import {mod_path}: {e}")
            ok = False
            continue
        for name in names:
            if not hasattr(mod, name):
                _fail(f"{mod_path}.{name} is gone in fla {version}")
                ok = False

    for (mod_path, name), kwargs in FLA_KWARGS.items():
        try:
            fn = getattr(importlib.import_module(mod_path), name)
        except (ImportError, AttributeError):
            continue  # already reported above
        params = _parameters(fn)
        if params is None:
            continue
        missing = [kw for kw in kwargs if kw not in params]
        if missing:
            _fail(f"{mod_path}.{name} no longer accepts {missing} in fla {version}")
            ok = False
    return ok


def _parameters(fn) -> set[str] | None:
    """Parameter names of a python function or a triton.jit kernel.

    A JITFunction is not introspectable by `inspect.signature`; it carries `arg_names`.
    That list is also ORDERED, which matters for the one kernel we launch ourselves — a
    reorder upstream would silently feed our arguments to the wrong parameters.
    """
    arg_names = getattr(fn, "arg_names", None)
    if arg_names is not None:
        return set(arg_names)
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None


def kernel_arg_names(mod_path: str, name: str) -> tuple[str, ...]:
    """Ordered parameter names of a raw triton kernel, for an exact-match assertion."""
    import importlib

    fn = getattr(importlib.import_module(mod_path), name)
    return tuple(getattr(fn, "arg_names", ()))
