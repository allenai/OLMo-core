"""
Datasets classes and config builders for use with the :class:`~olmo_core.trainer.Trainer`.

Overview
--------

Prepare your data by writing token IDs to numpy arrays on disk, using the
`Dolma toolkit <https://allenai.github.io/dolma/>`_ for example.

Configure and build your dataset using the :class:`NumpyDatasetConfig` builder and pass it
to :meth:`TrainerConfig.build() <olmo_core.train.TrainerConfig.build>`.

API Reference
-------------
"""

from .collator import DataCollator, PaddingDirection
from .iterable_dataset import (
    IterableDatasetBase,
    IterableFSLDataset,
    IterableVSLDataset,
)
from .mixes import DataMix
from .numpy_dataset import (
    NumpyDatasetBase,
    NumpyDatasetConfig,
    NumpyDatasetDType,
    NumpyDatasetType,
    NumpyFSLDataset,
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
    "IterableDatasetBase",
    "IterableFSLDataset",
    "IterableVSLDataset",
]
