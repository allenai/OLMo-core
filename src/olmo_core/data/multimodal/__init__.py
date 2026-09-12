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
* :class:`~olmo_core.data.multimodal.pretraining_replay.PretrainingReplayDataset` — native
  text replay resolved from the parent checkpoint or an explicit dataset config.

Unlike the text-only :mod:`olmo_core.data.composable` pipeline (a token-stream
packer), this carries variable-shape image tensors alongside the token sequence.
"""

from typing import Any

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
from .vision_alignment_perception import (
    VISION_ALIGNMENT_OCR_SOURCES,
    VisionAlignmentAuditedAlignmentDataset,
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDataset,
    VisionAlignmentOcrDocumentDatasetConfig,
)

_LEGACY_REPLAY_EXPORTS = {
    "NativeTextReplayDataset",
    "NativeTextReplayDatasetConfig",
    "NativeTextReplayManifest",
    "NativeTextReplaySource",
    "NativeTextReplayVerificationReceipt",
}
_LEGACY_PERCEPTION_EXPORTS = {
    "VisionAlignmentPerceptionSourceSpec",
    "build_vision_alignment_perception_dataset",
    "build_vision_alignment_perception_dataset_config",
}


def __getattr__(name: str) -> Any:
    """Load retained legacy exports only on explicit access."""
    if name in _LEGACY_REPLAY_EXPORTS:
        from . import native_text_replay

        value = getattr(native_text_replay, name)
    elif name in _LEGACY_PERCEPTION_EXPORTS:
        from . import vision_alignment_perception_sources

        value = getattr(vision_alignment_perception_sources, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


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
    "NativeTextReplayManifest",
    "NativeTextReplaySource",
    "NativeTextReplayVerificationReceipt",
    "build_packed_sequence",
    "build_branched_sequence",
    "ATTEND_ALL_SUBSEGMENT_ID",
    "pack_examples",
    "VISION_ALIGNMENT_OCR_SOURCES",
    "VisionAlignmentAuditedAlignmentDataset",
    "VisionAlignmentAuditedAlignmentDatasetConfig",
    "VisionAlignmentOcrDocumentDataset",
    "VisionAlignmentOcrDocumentDatasetConfig",
    "VisionAlignmentPerceptionSourceSpec",
    "build_vision_alignment_perception_dataset",
    "build_vision_alignment_perception_dataset_config",
]
