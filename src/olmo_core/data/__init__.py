"""
Dataset, data loaders, and config builders for use with the :class:`~olmo_core.train.Trainer`.

Overview
--------

Prepare your data by writing token IDs to numpy arrays on disk, using the
`Dolma toolkit <https://allenai.github.io/dolma/>`_ for example.

Configure and build your dataset using the :class:`~olmo_core.data.numpy_dataset.NumpyDatasetConfig`
builder and pass it to :meth:`TrainerConfig.build() <olmo_core.train.TrainerConfig.build>`.
"""

from .collator import DataCollator, PaddingDirection
from .data_loader import DataLoaderBase, FSLDataLoader, VSLDataLoader
from .mixes import DataMix
from .numpy_dataset import (
    NumpyDatasetBase,
    NumpyDatasetConfig,
    NumpyDatasetDType,
    NumpyDatasetType,
    NumpyFSLDataset,
    NumpyPaddedFSLDataset,
    NumpyVSLDataset,
    VSLCurriculum,
    VSLCurriculumConfig,
    VSLCurriculumType,
    VSLGrowLinearCurriculum,
    VSLGrowP2Curriculum,
    VSLGrowthCurriculum,
    VSLNaturalCurriculum,
)
from .tokenizer import TokenizerConfig, TokenizerName

__all__ = [
    "NumpyDatasetBase",
    "NumpyFSLDataset",
    "NumpyPaddedFSLDataset",
    "NumpyVSLDataset",
    "VSLCurriculum",
    "VSLNaturalCurriculum",
    "VSLGrowthCurriculum",
    "VSLGrowP2Curriculum",
    "VSLGrowLinearCurriculum",
    "NumpyDatasetConfig",
    "NumpyDatasetType",
    "VSLCurriculumType",
    "VSLCurriculumConfig",
    "NumpyDatasetDType",
    "TokenizerConfig",
    "TokenizerName",
    "DataMix",
    "DataCollator",
    "PaddingDirection",
    "DataLoaderBase",
    "FSLDataLoader",
    "VSLDataLoader",
]
