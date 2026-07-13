"""
MoE V2 compatibility exports.
"""

from typing import Any

from ...output_discard_checkpoint import OutputDiscardCheckpoint

_BLOCK_EXPORTS = {
    "MoEFusedV2TransformerBlock",
    "MoEFusedV2TransformerBlockConfig",
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
}

_MODEL_EXPORTS = {
    "MoEFusedV2Transformer",
    "OLMoDDPModel",
}


def __getattr__(name: str) -> Any:
    if name in _BLOCK_EXPORTS:
        from . import block

        value = getattr(block, name)
        globals()[name] = value
        return value
    if name in _MODEL_EXPORTS:
        from . import model

        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OutputDiscardCheckpoint",
    "MoEFusedV2TransformerBlock",
    "MoEFusedV2TransformerBlockConfig",
    "OLMoDDPTransformerBlock",
    "OLMoDDPTransformerBlockConfig",
    "MoEFusedV2Transformer",
    "OLMoDDPModel",
]
