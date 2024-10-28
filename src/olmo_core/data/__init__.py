"""
Dataset, data loaders, and config builders for use with the :class:`~olmo_core.train.Trainer`.

Overview
--------

Prepare your data by writing token IDs to numpy arrays on disk, using the
`Dolma toolkit <https://allenai.github.io/dolma/>`_ for example.

Configure and build your dataset using the :class:`~olmo_core.data.numpy_dataset.NumpyDatasetConfig`
builder, build your data loader with the :class:`~olmo_core.data.data_loader.NumpyDataLoaderConfig`
builder, then pass it to :meth:`TrainerConfig.build() <olmo_core.train.TrainerConfig.build>`.
"""

from .collator import DataCollator, PaddingDirection
from .data_loader import (
    DataLoaderBase,
    NumpyDataLoaderBase,
    NumpyDataLoaderConfig,
    NumpyFSLDataLoader,
    NumpyVSLDataLoader,
)
from .mixes import DataMix, DataMixBase
from .numpy_dataset import (
    NumpyDatasetBase,
    NumpyDatasetConfig,
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
from .types import NumpyDatasetDType, NumpyDatasetType

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
    "DataMixBase",
    "DataMix",
    "DataCollator",
    "PaddingDirection",
    "DataLoaderBase",
    "NumpyDataLoaderBase",
    "NumpyFSLDataLoader",
    "NumpyVSLDataLoader",
    "NumpyDataLoaderConfig",
]
