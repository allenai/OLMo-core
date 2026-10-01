from __future__ import annotations

import os
from typing import Any, Literal

MXFP8ScaleMode = Literal["floor", "rceil"]

MXFP8_SCALE_MODE_FLOOR: MXFP8ScaleMode = "floor"
MXFP8_SCALE_MODE_RCEIL: MXFP8ScaleMode = "rceil"
MXFP8_SCALE_MODE_ENV = "OLMO_MXFP8_SCALE_MODE"


def normalize_mxfp8_scale_mode_value(scale_mode: Any) -> MXFP8ScaleMode:
    """
    Coerce a scale-mode value (a string or an enum-like object with a ``.value``) to a
    canonical :data:`MXFP8ScaleMode` literal.

    :raises ValueError: If the value is not ``"floor"`` or ``"rceil"``.
    """
    value = getattr(scale_mode, "value", scale_mode)
    if isinstance(value, str):
        value = value.strip().lower()
    if value == MXFP8_SCALE_MODE_FLOOR:
        return MXFP8_SCALE_MODE_FLOOR
    if value == MXFP8_SCALE_MODE_RCEIL:
        return MXFP8_SCALE_MODE_RCEIL
    raise ValueError(
        f"{MXFP8_SCALE_MODE_ENV}/scale_mode must be 'floor' or 'rceil', " f"got {scale_mode!r}"
    )


_MXFP8_DEFAULT_SCALE_MODE = normalize_mxfp8_scale_mode_value(
    os.environ.get(MXFP8_SCALE_MODE_ENV, MXFP8_SCALE_MODE_RCEIL)
)


def normalize_mxfp8_scale_mode(scale_mode: Any) -> MXFP8ScaleMode:
    """
    Like :func:`normalize_mxfp8_scale_mode_value`, but resolve ``None`` to the import-time
    default (from ``OLMO_MXFP8_SCALE_MODE``).
    """
    if scale_mode is None:
        return _MXFP8_DEFAULT_SCALE_MODE
    return normalize_mxfp8_scale_mode_value(scale_mode)


def get_mxfp8_default_scale_mode() -> MXFP8ScaleMode:
    """Return the MXFP8 scale mode resolved once from ``OLMO_MXFP8_SCALE_MODE`` at import time."""
    return _MXFP8_DEFAULT_SCALE_MODE
