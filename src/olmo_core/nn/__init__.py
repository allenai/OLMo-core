"""
Common :class:`torch.nn.Module` implementations.
"""

from .output_discard_checkpoint import OutputDiscardCheckpoint
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

__all__ = [
    "OutputDiscardCheckpoint",
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
