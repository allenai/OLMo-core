"""
Compatibility module for old ``olmo_core.nn.fla.gated_deltanet`` imports.
"""

from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNetConfig

__all__ = ["GatedDeltaNet", "GatedDeltaNetConfig"]
