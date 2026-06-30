"""
Shared nvtx annotation helper for the fused MoE modules.

Centralizes the optional-``nvtx`` import (degrading to the no-op
:mod:`olmo_core._nvtx` fallback when the ``profiling`` extra isn't installed) and the
range-coloring convention, so call sites pick a *subsystem* rather than a raw color and every
range is labeled/colored consistently.
"""

from __future__ import annotations

try:
    import nvtx as _nvtx
except ImportError:
    from olmo_core._nvtx import nvtx as _nvtx

# Range color per subsystem.
_SUBSYSTEM_COLORS = {
    "routing": "blue",  # token routing + per-block forward orchestration
    "experts": "purple",  # expert compute and expert-weight preparation
    "comm": "green",  # communication / token movement (permute, all-to-all, drop/restore)
    "tbo": "orange",  # two-batch-overlap orchestration
}

# Used when a range isn't tied to a known subsystem.
_DEFAULT_COLOR = "gray"


def annotate(label: str, subsystem: str | None = None):
    """
    Create an nvtx range following the shared annotation convention.

    Usable as either a decorator (``@annotate("MoERouter.forward", "routing")``) or a context
    manager (``with annotate("permute", "comm"): ...``), and a no-op when nvtx isn't installed.

    :param label: The range label — the qualified ``ClassName.method`` / ``module_function`` name
        for a whole callable, or a ``snake_case`` phase name for an inner block.
    :param subsystem: One of ``"routing"``, ``"experts"``, ``"comm"``, ``"tbo"``; selects the color.
        Omit it for ranges that don't belong to a subsystem — they get a neutral default color.

    :raises ValueError: If ``subsystem`` is given but not a known subsystem.
    """
    if subsystem is None:
        color = _DEFAULT_COLOR
    else:
        try:
            color = _SUBSYSTEM_COLORS[subsystem]
        except KeyError:
            raise ValueError(
                f"unknown nvtx subsystem {subsystem!r}; expected one of {sorted(_SUBSYSTEM_COLORS)}"
            )
    return _nvtx.annotate(label, color=color)
