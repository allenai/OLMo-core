"""
Reading raw CLI overrides *before* :meth:`~olmo_core.config.Config.merge` runs.

Most training-script settings can be read off the merged config. A few cannot: anything
that shapes a structure built *during* ``build_config`` — optimizer param groups, the
scheduler's group table, ``freeze_params`` — has to be known before the merge happens,
because by the time ``merge`` applies the override the structure is already built. Those
settings are read straight out of ``sys.argv`` with the helpers here, and the script is
then expected to re-validate after the merge that the two agree (see
``Molmo2-Stage1.py``'s ``train_vit`` handling for the canonical example).

Kept in one place because the parsing has a non-obvious rule: ``Config.merge`` normalizes
hyphens to underscores via ``_clean_opt``, so ``--train-vit=false`` and
``--train_vit=false`` are the same override to it. A pre-merge reader that compared the
raw name would miss the dashed spelling, build the config from the default, and let
``merge`` apply the requested value to the top-level field anyway — a silent divergence
between what the config says and what the optimizer groups were built from.
"""

from __future__ import annotations

from typing import Dict, List

from ..exceptions import OLMoConfigurationError

__all__ = ["read_bool_override", "read_float_override", "read_int_override", "read_override"]

_BOOLS: Dict[str, bool] = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
}


def read_override(overrides: List[str], key: str, default: str) -> str:
    """Read a top-level scalar out of the raw overrides. Later occurrences win, matching
    ``Config.merge``."""
    value = default
    for override in overrides:
        name, _, raw = override.lstrip("-").partition("=")
        if name.replace("-", "_") == key and raw:
            value = raw
    return value


def read_bool_override(overrides: List[str], key: str, default: bool) -> bool:
    """Read a boolean top-level scalar out of the raw overrides."""
    raw = read_override(overrides, key, str(default)).strip().lower()
    if raw not in _BOOLS:
        raise OLMoConfigurationError(f"{key}={raw!r} is not a boolean")
    return _BOOLS[raw]


def read_float_override(overrides: List[str], key: str, default: float) -> float:
    """Read a float top-level scalar out of the raw overrides."""
    raw = read_override(overrides, key, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        raise OLMoConfigurationError(f"{key}={raw!r} is not a float") from None


def read_int_override(overrides: List[str], key: str, default: int) -> int:
    """Read an integer top-level scalar out of the raw overrides."""
    raw = read_override(overrides, key, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        raise OLMoConfigurationError(f"{key}={raw!r} is not an integer") from None
