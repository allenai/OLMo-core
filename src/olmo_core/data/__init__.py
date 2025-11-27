"""
Dataset, data loaders, and config builders for use with the :class:`~olmo_core.train.Trainer`.

Overview
--------

For text-based data you should prepare your data by writing token IDs to numpy arrays on disk, using the
`Dolma toolkit <https://allenai.github.io/dolma/>`_ for example.
Then configure and build your dataset and data loader using either the :mod:`olmo_core.data.composable` API or one of the
:class:`~olmo_core.data.numpy_dataset.NumpyDatasetConfigBase` builders
(e.g. :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig`) with the
:class:`~olmo_core.data.data_loader.NumpyDataLoaderConfig`
builder. Data loaders can be passed to :meth:`TrainerConfig.build() <olmo_core.train.TrainerConfig.build>`.
"""

from . import composable
from .collator import DataCollator, PaddingDirection
from .data_loader import (
    DataLoaderBase,
    NumpyDataLoaderBase,
    NumpyDataLoaderConfig,
    NumpyFSLDataLoader,
    NumpyVSLDataLoader,
    TextDataLoaderBase,
)
from .mixes import DataMix, DataMixBase
from .numpy_dataset import (
    InstanceFilterConfig,
    NumpyDatasetBase,
    NumpyDatasetConfig,
    NumpyFSLDataset,
    NumpyFSLDatasetBase,
    NumpyFSLDatasetConfig,
    NumpyInterleavedFSLDatasetConfig,
    NumpyPackedFSLDataset,
    NumpyPackedFSLDatasetConfig,
    NumpyPaddedFSLDataset,
    NumpyPaddedFSLDatasetConfig,
    NumpyVSLDataset,
    NumpyVSLDatasetConfig,
    VSLCurriculum,
    VSLCurriculumConfig,
    VSLCurriculumType,
    VSLGrowLinearCurriculum,
    VSLGrowP2Curriculum,
    VSLGrowthCurriculum,
    VSLNaturalCurriculum,
)
from .tokenizer import TokenizerConfig, TokenizerName
from .types import LongDocStrategy, NumpyDatasetDType

__all__ = [
    "composable",
    "NumpyDatasetBase",
    "NumpyFSLDatasetBase",
    "NumpyFSLDataset",
    "NumpyPaddedFSLDataset",
    "NumpyPackedFSLDataset",
    "NumpyVSLDataset",
    "VSLCurriculum",
    "VSLNaturalCurriculum",
    "VSLGrowthCurriculum",
    "VSLGrowP2Curriculum",
    "VSLGrowLinearCurriculum",
    "NumpyDatasetConfig",
    "NumpyFSLDatasetConfig",
    "NumpyPaddedFSLDatasetConfig",
    "NumpyPackedFSLDatasetConfig",
    "NumpyInterleavedFSLDatasetConfig",
    "NumpyVSLDatasetConfig",
    "InstanceFilterConfig",
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
    "TextDataLoaderBase",
    "NumpyDataLoaderBase",
    "NumpyFSLDataLoader",
    "NumpyVSLDataLoader",
    "NumpyDataLoaderConfig",
    "LongDocStrategy",
]
