"""
DDP-stack model entry points.
"""

from .moe.v2.model import OLMoDDPModel
from .transformer.config import OLMoDDPModelConfig

__all__ = ["OLMoDDPModel", "OLMoDDPModelConfig"]
