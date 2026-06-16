"""
Common :class:`torch.nn.Module` implementations.
"""

from typing import Any, TYPE_CHECKING

from .mxfp8_linear import MXFP8Linear
from .output_discard_checkpoint import OutputDiscardCheckpoint

if TYPE_CHECKING:
    from .vision import (
        ImagePoolingType,
        ImageProjectorType,
        MultimodalLM,
        MultimodalLMConfig,
        VisionConnector,
        VisionConnectorConfig,
        VisionEncoderConfig,
        VisionEncoderType,
        VisionTransformer,
    )

_VISION_EXPORTS = {
    "ImagePoolingType",
    "ImageProjectorType",
    "MultimodalLM",
    "MultimodalLMConfig",
    "VisionConnector",
    "VisionConnectorConfig",
    "VisionEncoderConfig",
    "VisionEncoderType",
    "VisionTransformer",
}

_DDP_EXPORTS = {
    "OLMoDDPModel",
    "OLMoDDPModelConfig",
}


def __getattr__(name: str) -> Any:
    if name in _DDP_EXPORTS:
        from . import ddp

        value = getattr(ddp, name)
        globals()[name] = value
        return value
    if name in _VISION_EXPORTS:
        from . import vision

        value = getattr(vision, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MXFP8Linear",
    "OutputDiscardCheckpoint",
    "OLMoDDPModel",
    "OLMoDDPModelConfig",
    "VisionEncoderType",
    "VisionEncoderConfig",
    "VisionTransformer",
    "ImagePoolingType",
    "ImageProjectorType",
    "VisionConnectorConfig",
    "VisionConnector",
    "MultimodalLMConfig",
    "MultimodalLM",
]
