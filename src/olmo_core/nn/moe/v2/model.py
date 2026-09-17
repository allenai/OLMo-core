"""
Compatibility alias for :class:`~olmo_core.nn.ddp.model.OLMoDDPModel`.

``MoEFusedV2Transformer`` remains available for serialized model configs.
"""

from olmo_core.nn.ddp.model import OLMoDDPModel

MoEFusedV2Transformer = OLMoDDPModel

__all__ = ["OLMoDDPModel", "MoEFusedV2Transformer"]
