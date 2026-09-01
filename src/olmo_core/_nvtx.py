"""
Optional ``nvtx`` profiling support: a no-op fallback plus the shared annotation helper.

``nvtx`` (NVIDIA Tools Extension) is only needed to emit profiler ranges under a profiler such as
Nsight Systems, so it is declared as an optional dependency (the ``profiling`` extra). Modules that
annotate hot paths should go through :func:`maybe_nvtx_annotate`, which emits a real nvtx range when
nvtx is installed and otherwise returns a no-op range.

Modules that want the raw API should just do::

    from olmo_core._nvtx import nvtx

which is the real ``nvtx`` module when it is installed and a no-op stand-in otherwise, so
``@nvtx.annotate(...)`` annotations cost nothing when nvtx is absent. A defensive
``try: import nvtx / except ImportError:`` around it is unnecessary.
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


_no_op_nvtx = _NoOpNvtx()

try:
    import nvtx as _active_nvtx
except ImportError:
    _active_nvtx = _no_op_nvtx  # type: ignore[assignment]

# The exported 'nvtx': the real module when installed, else the no-op above.
#
# NOTE: this binding is load-bearing. ~20 modules annotate their hot paths with
# 'from olmo_core._nvtx import nvtx' + '@nvtx.annotate(...)' / 'with nvtx.annotate(...)'.
# Binding this name to the no-op unconditionally would make every one of those ranges a
# permanent no-op, so installing the 'profiling' extra would buy nothing -- the ranges would
# be dead code. Point it at the real module so the extra is what actually decides.
nvtx = _active_nvtx


def maybe_nvtx_annotate(label: str, color: Optional[str] = None):
    """
    Create an nvtx range (a no-op when nvtx isn't installed).

    Usable as either a decorator (``@maybe_nvtx_annotate("MoERouter.forward", ROUTING_COLOR)``) or a
    context manager (``with maybe_nvtx_annotate("permute", COMM_COLOR): ...``).

    This helper is subsystem-agnostic: it just forwards ``color`` to nvtx. The convention of mapping
    a subsystem to a consistent color lives with the calling code — each domain declares its own
    color constants (e.g. :mod:`olmo_core.nn.moe.v2._nvtx_colors`) and passes one in — so this
    module needn't know about any particular subsystem.

    :param label: The range label — the qualified ``ClassName.method`` / ``module_function`` name
        for a whole callable, or a ``snake_case`` phase name for an inner block.
    :param color: The nvtx range color (any color name nvtx accepts). Omit it for ranges that don't
        need a color — they get nvtx's default.
    """
    if color is None:
        return _active_nvtx.annotate(label)
    return _active_nvtx.annotate(label, color=color)
