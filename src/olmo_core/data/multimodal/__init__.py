"""
Multimodal data preprocessing for inference and evaluation.

Provides image preprocessing (resize / normalize / crop) and token-sequence
building.

Benchmark task definitions and their scoring logic live in
``olmo-eval-internal`` (the ``olmo_eval`` package), which OLMo-core installs
as its ``[eval]`` extra.  The training pipeline (datasets, collator,
DataLoader) is added in the next PR.
"""

from .image_preprocessor import (
    ImagePreprocessor,
    ImagePreprocessorConfig,
    NormalizeStyle,
)
from .multicrop import (
    CropMode,
    MultiCropOutput,
    MultiCropPreprocessor,
    MultiCropPreprocessorConfig,
)
from .tokens import IMAGE_SPECIAL_TOKENS, MultimodalTokenizerConfig

__all__ = [
    "IMAGE_SPECIAL_TOKENS",
    "MultimodalTokenizerConfig",
    "NormalizeStyle",
    "ImagePreprocessor",
    "ImagePreprocessorConfig",
    "CropMode",
    "MultiCropOutput",
    "MultiCropPreprocessor",
    "MultiCropPreprocessorConfig",
]
