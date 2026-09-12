"""
The OLMo DDP model stack: the fused expert-parallel MoE-v2 transformer, trained with the OLMo
multi-group DDP parallelism strategy.
"""

from olmo_core.nn.ddp.block import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.ddp.model import OLMoDDPModel

__all__ = [
    "OLMoDDPModel",
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
]
