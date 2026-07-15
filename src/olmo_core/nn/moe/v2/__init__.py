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

_QWEN_EXPORTS = {
    "QWEN3_MOE_LAYER_PATTERN",
    "QWEN3_DENSE_MOE_LAYER_TYPE",
    "build_qwen3_moe_config",
    "build_qwen3_moe_config_from_hf_config",
    "build_debug_qwen3_moe_config",
    "get_qwen3_moe_text_config_overrides",
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
    if name in _QWEN_EXPORTS:
        from . import qwen

        value = getattr(qwen, name)
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
    "QWEN3_MOE_LAYER_PATTERN",
    "QWEN3_DENSE_MOE_LAYER_TYPE",
    "build_qwen3_moe_config",
    "build_qwen3_moe_config_from_hf_config",
    "build_debug_qwen3_moe_config",
    "get_qwen3_moe_text_config_overrides",
]
