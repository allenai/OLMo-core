"""
Back-compat shim. The fused MoE-v2 transformer moved to :mod:`olmo_core.nn.ddp.model` and was
renamed ``MoEFusedV2Transformer`` -> :class:`~olmo_core.nn.ddp.model.OLMoDDPModel`. The old name
remains importable from here as an alias.
"""

from olmo_core.nn.ddp.model import OLMoDDPModel

MoEFusedV2Transformer = OLMoDDPModel

__all__ = ["OLMoDDPModel", "MoEFusedV2Transformer"]
