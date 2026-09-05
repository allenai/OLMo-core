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

import os
from contextlib import ContextDecorator, nullcontext
from typing import Any, Optional

import torch

__all__ = ["nvtx", "maybe_nvtx_annotate"]

_COMPILE_SAFE_NOOP_RANGES = os.environ.get("OLMO_PROFILE_SAFE_NOOP_NVTX", "0") == "1"


class _NoOpRange(ContextDecorator):
    """A do-nothing range usable as both a decorator and a context manager."""

    def __enter__(self) -> "_NoOpRange":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Returning None (falsy) means we never suppress exceptions.
        return None

    def __call__(self, func):
        if _COMPILE_SAFE_NOOP_RANGES:
            # A disabled annotation need not wrap the function in a context manager.
            # Such a wrapper can prevent Dynamo resuming after an intentional graph break.
            return func
        return super().__call__(func)


class _NoOpNvtx:
    """Drop-in stand-in exposing the (sole) ``nvtx.annotate`` API as a no-op."""

    @staticmethod
    def annotate(*args: Any, **kwargs: Any):
        if _COMPILE_SAFE_NOOP_RANGES and torch.compiler.is_compiling():
            # Dynamo understands nullcontext across graph breaks. A custom context
            # can instead cause the entire surrounding region to remain eager.
            return nullcontext()
        return _NoOpRange()


nvtx = _NoOpNvtx()

# The active nvtx used by maybe_nvtx_annotate: real nvtx when installed, else the no-op above.
try:
    import nvtx as _active_nvtx
except ImportError:
    _active_nvtx = nvtx  # type: ignore[assignment]


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
