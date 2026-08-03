"""
Multimodal (vision-language) training data: datasets and collation for Molmo2.

This subpackage provides a standalone, ``mm_olmo``-free pipeline for Molmo2 "stage 1"
caption pretraining:

* :class:`~olmo_core.data.multimodal.pixmo_cap.PixMoCapDataset` — map-style dataset
  yielding packed image + caption/transcript training examples.
* :class:`~olmo_core.data.multimodal.collator.MultimodalCollator` — pads/stacks them
  into batches for :class:`~olmo_core.nn.vision.MultimodalLM`.
* :func:`~olmo_core.data.multimodal.sequence_builder.build_packed_sequence` — the
  core multi-annotation (branch-packing) sequence assembly with float loss weights.

Unlike the text-only :mod:`olmo_core.data.composable` pipeline (a token-stream
packer), this carries variable-shape image tensors alongside the token sequence.
"""

from .collator import MultimodalCollator, MultimodalCollatorConfig
from .data_loader import MultimodalDataLoader
from .mixture_data_loader import MixtureDataLoader
from .packing import pack_examples
from .pixmo_cap import PixMoCapDataset, PixMoCapDatasetConfig
from .pixmo_points import (
    CoSynPointDataset,
    CoSynPointDatasetConfig,
    PixMoCountDataset,
    PixMoCountDatasetConfig,
    PixMoPointsDataset,
    PixMoPointsDatasetConfig,
)
from .sequence_builder import (
    ATTEND_ALL_SUBSEGMENT_ID,
    build_branched_sequence,
    build_packed_sequence,
)
from .paths import (
    ACADEMIC_DATASETS,
    MOLMO_DATA_DIR,
    PIXMO_DATASETS,
    TORCH_DATASETS,
    TULU4_DATA,
)
from .academic_dataset import AcademicDataset, AcademicDatasetConfig
from .pixmo_ama import PixMoAmaDataset, PixMoAmaDatasetConfig
from .pixmo_cap_qa import PixMoCapQaDataset, PixMoCapQaDatasetConfig
from .message_weight import MessageWeight, apply_message_weight_to_loss_masks
from .mixture_weights import DatasetSource, SubMixture, compute_flat_mixture_weights
from .sft_formatter import SftFormatter
from .tulu import Tulu4Dataset, Tulu4DatasetConfig

__all__ = [
    "PixMoCapDataset",
    "PixMoCapDatasetConfig",
    "PixMoPointsDataset",
    "PixMoPointsDatasetConfig",
    "PixMoCountDataset",
    "PixMoCountDatasetConfig",
    "CoSynPointDataset",
    "CoSynPointDatasetConfig",
    "Tulu4Dataset",
    "Tulu4DatasetConfig",
    "AcademicDataset",
    "AcademicDatasetConfig",
    "PixMoAmaDataset",
    "PixMoAmaDatasetConfig",
    "PixMoCapQaDataset",
    "PixMoCapQaDatasetConfig",
    "SftFormatter",
    "MessageWeight",
    "apply_message_weight_to_loss_masks",
    "DatasetSource",
    "SubMixture",
    "compute_flat_mixture_weights",
    "PIXMO_DATASETS",
    "TULU4_DATA",
    "ACADEMIC_DATASETS",
    "MOLMO_DATA_DIR",
    "TORCH_DATASETS",
    "MultimodalCollator",
    "MultimodalCollatorConfig",
    "MultimodalDataLoader",
    "MixtureDataLoader",
    "build_packed_sequence",
    "build_branched_sequence",
    "ATTEND_ALL_SUBSEGMENT_ID",
    "pack_examples",
]
