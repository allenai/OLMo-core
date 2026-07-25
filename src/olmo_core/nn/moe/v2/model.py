"""
Compatibility re-export. The fused MoE-v2 transformer moved to :mod:`olmo_core.nn.ddp.model`
(:class:`~olmo_core.nn.ddp.model.OLMoDDPModel`); it remains importable from this former module path.
"""

from olmo_core.nn.ddp.model import OLMoDDPModel

__all__ = ["OLMoDDPModel"]
