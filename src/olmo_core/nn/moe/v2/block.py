"""
Compatibility imports for the old MoE V2 transformer-block path.
"""

from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock, OLMoDDPTransformerBlockConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig

MoEFusedV2TransformerBlock = OLMoDDPTransformerBlock
MoEFusedV2TransformerBlockConfig = OLMoDDPTransformerBlockConfig

__all__ = [
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
    "MoEFusedV2TransformerBlock",
    "MoEFusedV2TransformerBlockConfig",
    "MoERouterConfigV2",
    "RoutedExpertsConfig",
    "SharedExpertsConfig",
]
