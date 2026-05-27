"""
No-op fallback for the optional ``nvtx`` profiling dependency.

``nvtx`` (NVIDIA Tools Extension) is only needed to emit profiler ranges under a
profiler such as Nsight Systems, so it is declared as an optional dependency (the
``profiling`` extra). Modules that annotate hot paths import it defensively::

    try:
        import nvtx
    except ImportError:
        from olmo_core._nvtx import nvtx

so the ``@nvtx.annotate(...)`` annotations become no-ops when nvtx is not installed.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any

__all__ = ["nvtx"]


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
