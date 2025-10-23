"""
Dataset, data loaders, and config builders for use with the :class:`~olmo_core.train.Trainer`.

Overview
--------

For text-based data you should prepare your data by writing token IDs to numpy arrays on disk, using the
`Dolma toolkit <https://allenai.github.io/dolma/>`_ for example.
Then configure and build your dataset using one of the
:class:`~olmo_core.data.numpy_dataset.NumpyDatasetConfigBase` builders (for example
``NumpyFSLDatasetConfig``), build your data loader with the
:class:`~olmo_core.data.data_loader.NumpyDataLoaderConfig`
builder, and pass it to :meth:`TrainerConfig.build() <olmo_core.train.TrainerConfig.build>`.
"""

from .collator import DataCollator, PaddingDirection, ByteDataCollator
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
    NumpyByteFSLDataset,
    NumpyByteFSLDatasetConfig,
    NumpyBytePaddedFSLDataset,
    NumpyBytePaddedFSLDatasetConfig,
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
from .tokenizer import ByteTokenizerConfig, ByteTokenizer, TokenizerConfig, TokenizerName
from .types import LongDocStrategy, NumpyDatasetDType

__all__ = [
    "NumpyDatasetBase",
    "NumpyFSLDatasetBase",
    "NumpyFSLDataset",
    "NumpyByteFSLDataset",
    "NumpyPaddedFSLDataset",
    "NumpyBytePaddedFSLDataset",
    "NumpyPackedFSLDataset",
    "NumpyVSLDataset",
    "VSLCurriculum",
    "VSLNaturalCurriculum",
    "VSLGrowthCurriculum",
    "VSLGrowP2Curriculum",
    "VSLGrowLinearCurriculum",
    "NumpyDatasetConfig",
    "NumpyFSLDatasetConfig",
    "NumpyByteFSLDatasetConfig",
    "NumpyPaddedFSLDatasetConfig",
    "NumpyBytePaddedFSLDatasetConfig",
    "NumpyPackedFSLDatasetConfig",
    "NumpyInterleavedFSLDatasetConfig",
    "NumpyVSLDatasetConfig",
    "InstanceFilterConfig",
    "VSLCurriculumType",
    "VSLCurriculumConfig",
    "NumpyDatasetDType",
    "TokenizerConfig",
    "ByteDataCollator",
    "ByteTokenizerConfig",
    "ByteTokenizer",
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
