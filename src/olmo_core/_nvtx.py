"""
Optional ``nvtx`` profiling support: a no-op fallback plus the shared annotation helper.

``nvtx`` (NVIDIA Tools Extension) is only needed to emit profiler ranges under a profiler such as
Nsight Systems, so it is declared as an optional dependency (the ``profiling`` extra). Modules that
annotate hot paths should go through :func:`maybe_nvtx_annotate`, which emits a real nvtx range when
nvtx is installed and otherwise returns a no-op range.

Modules that need the raw API can still import it defensively::

    try:
        import nvtx
    except ImportError:
        from olmo_core._nvtx import nvtx

so the ``@nvtx.annotate(...)`` annotations become no-ops when nvtx is not installed.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any, Optional

__all__ = ["nvtx", "maybe_nvtx_annotate"]


class _NoOpRange(ContextDecorator):
    """A do-nothing range usable as both a decorator and a context manager."""

    def __enter__(self) -> "_NoOpRange":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Returning None (falsy) means we never suppress exceptions.
        return None


class _NoOpNvtx:
    """Drop-in stand-in exposing the (sole) ``nvtx.annotate`` API as a no-op."""

    @staticmethod
    def annotate(*args: Any, **kwargs: Any) -> _NoOpRange:
        return _NoOpRange()


nvtx = _NoOpNvtx()

# The active nvtx used by maybe_nvtx_annotate: real nvtx when installed, else the no-op above.
try:
    import nvtx as _active_nvtx
except ImportError:
    _active_nvtx = nvtx  # type: ignore[assignment]

# Optional convention: map a subsystem to a consistent range color. Ranges without a subsystem get
# no color (nvtx's default), so maybe_nvtx_annotate is a drop-in for a bare nvtx.annotate(label).
_SUBSYSTEM_COLORS = {
    "routing": "blue",  # token routing + per-block forward orchestration
    "experts": "purple",  # expert compute and expert-weight preparation
    "comm": "green",  # communication / token movement (permute, all-to-all, drop/restore)
    "tbo": "orange",  # two-batch-overlap orchestration
}


def maybe_nvtx_annotate(label: str, subsystem: Optional[str] = None):
    """
    Create an nvtx range (a no-op when nvtx isn't installed).

    Usable as either a decorator (``@maybe_nvtx_annotate("MoERouter.forward", "routing")``) or a
    context manager (``with maybe_nvtx_annotate("permute", "comm"): ...``).

    :param label: The range label — the qualified ``ClassName.method`` / ``module_function`` name
        for a whole callable, or a ``snake_case`` phase name for an inner block.
    :param subsystem: One of ``"routing"``, ``"experts"``, ``"comm"``, ``"tbo"``; selects the range
        color. Omit it for ranges that don't belong to a subsystem — they get no color.

    :raises ValueError: If ``subsystem`` is given but not a known subsystem.
    """
    if subsystem is None:
        return _active_nvtx.annotate(label)
    try:
        color = _SUBSYSTEM_COLORS[subsystem]
    except KeyError:
        raise ValueError(
            f"unknown nvtx subsystem {subsystem!r}; expected one of {sorted(_SUBSYSTEM_COLORS)}"
        )
    return _active_nvtx.annotate(label, color=color)
