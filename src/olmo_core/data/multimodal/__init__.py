"""
Multimodal (vision-language) training data, replay, packing, and collation.

This subpackage provides a standalone, ``mm_olmo``-free pipeline shared by Molmo2 recipes and
the separate vision-alignment continued-pretraining recipe:

* :class:`~olmo_core.data.multimodal.pixmo_cap.PixMoCapDataset` — map-style dataset
  yielding packed image + caption/transcript training examples.
* :class:`~olmo_core.data.multimodal.collator.MultimodalCollator` — pads/stacks them
  into batches for :class:`~olmo_core.nn.vision.MultimodalLM`.
* :func:`~olmo_core.data.multimodal.sequence_builder.build_packed_sequence` — the
  core multi-annotation (branch-packing) sequence assembly with float loss weights.
* :class:`~olmo_core.data.multimodal.native_text_replay.NativeTextReplayDataset` — bounded,
  exact-token replay from a pinned parent-pretraining manifest.

Unlike the text-only :mod:`olmo_core.data.composable` pipeline (a token-stream
packer), this carries variable-shape image tensors alongside the token sequence.
"""

from .academic_dataset import AcademicDataset, AcademicDatasetConfig
from .collator import MultimodalCollator, MultimodalCollatorConfig
from .data_loader import MultimodalDataLoader
from .finevision import (
    FINEVISION_ROOT,
    FineVisionDataset,
    FineVisionDatasetConfig,
    VisualWebInstructDataset,
    VisualWebInstructDatasetConfig,
)
from .message_weight import MessageWeight, apply_message_weight_to_loss_masks
from .mixture_data_loader import MixtureDataLoader
from .mixture_weights import DatasetSource, SubMixture, compute_flat_mixture_weights
from .mmfinereason import (
    MMFineReasonDataset,
    MMFineReasonDatasetConfig,
    extract_answer_text,
)
from .native_text_replay import NativeTextReplayDataset, NativeTextReplayDatasetConfig
from .numpy_fsl_text import NumpyFSLTextDataset, NumpyFSLTextDatasetConfig
from .packing import pack_examples
from .paths import (
    ACADEMIC_DATASETS,
    MOLMO_DATA_DIR,
    PIXMO_DATASETS,
    TORCH_DATASETS,
    TULU4_DATA,
)
from .pixmo_ama import PixMoAmaDataset, PixMoAmaDatasetConfig
from .pixmo_cap import PixMoCapDataset, PixMoCapDatasetConfig
from .pixmo_cap_qa import PixMoCapQaDataset, PixMoCapQaDatasetConfig
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
from .sft_formatter import SftFormatter
from .tulu import Tulu4Dataset, Tulu4DatasetConfig

__all__ = [
    "FineVisionDataset",
    "FineVisionDatasetConfig",
    "VisualWebInstructDataset",
    "VisualWebInstructDatasetConfig",
    "FINEVISION_ROOT",
    "MMFineReasonDataset",
    "MMFineReasonDatasetConfig",
    "extract_answer_text",
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
    "NativeTextReplayDataset",
    "NativeTextReplayDatasetConfig",
    "NumpyFSLTextDataset",
    "NumpyFSLTextDatasetConfig",
    "build_packed_sequence",
    "build_branched_sequence",
    "ATTEND_ALL_SUBSEGMENT_ID",
    "pack_examples",
]
