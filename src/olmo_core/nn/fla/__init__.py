"""
Backwards-compatible imports for configs serialized with ``olmo_core.nn.fla``.

The GatedDeltaNet implementation now lives under ``olmo_core.nn.attention``.
Older checkpoints may still contain fully-qualified class names from the old
``olmo_core.nn.fla`` package.
"""

from .layer import (
    FLA,
    FLAConfig,
    GatedDeltaNet,
    GatedDeltaNetConfig,
    dispatch_causal_conv1d,
    dispatch_chunk_gated_delta_rule,
    has_fla,
)
from .model import FLAModel, FLAModelConfig

__all__ = [
    "FLA",
    "FLAConfig",
    "FLAModel",
    "FLAModelConfig",
    "GatedDeltaNet",
    "GatedDeltaNetConfig",
    "dispatch_causal_conv1d",
    "dispatch_chunk_gated_delta_rule",
    "has_fla",
]
