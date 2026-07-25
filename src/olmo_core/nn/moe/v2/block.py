"""
Compatibility re-export. The fused MoE-v2 transformer block moved to
:mod:`olmo_core.nn.ddp.block` (:class:`~olmo_core.nn.ddp.block.OLMoDDPTransformerBlock`); it and the
MoE-v2 sub-configs remain importable from this former module path.
"""

from olmo_core.nn.ddp.block import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig

__all__ = [
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
    "MoERouterConfigV2",
    "RoutedExpertsConfig",
    "SharedExpertsConfig",
]
