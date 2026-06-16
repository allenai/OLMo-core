"""
Compatibility imports for the old MoE V2 model path.
"""

from olmo_core.nn.ddp.model import MoEFusedV2Transformer, OLMoDDPModel

__all__ = ["OLMoDDPModel", "MoEFusedV2Transformer"]
