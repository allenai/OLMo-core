"""
Backwards-compatible imports for configs serialized with ``olmo_core.nn.fla``.

The FLA/GatedDeltaNet implementation now lives under ``olmo_core.nn.attention``.
Older checkpoints may still contain fully-qualified class names from this module.
"""

from olmo_core.nn.attention.flash_linear_attn_api import (
    dispatch_causal_conv1d,
    dispatch_chunk_gated_delta_rule,
    has_fla,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNetConfig

__all__ = [
    "GatedDeltaNet",
    "GatedDeltaNetConfig",
    "dispatch_causal_conv1d",
    "dispatch_chunk_gated_delta_rule",
    "has_fla",
]
