"""
Compatibility imports for the old MoE V2 transformer-block path.
"""

from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock, OLMoDDPTransformerBlockConfig

MoEFusedV2TransformerBlock = OLMoDDPTransformerBlock
MoEFusedV2TransformerBlockConfig = OLMoDDPTransformerBlockConfig

__all__ = [
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
    "MoEFusedV2TransformerBlock",
    "MoEFusedV2TransformerBlockConfig",
]
