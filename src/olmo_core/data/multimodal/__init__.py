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
from .pixmo_cap import PixMoCapDataset, PixMoCapDatasetConfig
from .sequence_builder import ATTEND_ALL_SUBSEGMENT_ID, build_packed_sequence

__all__ = [
    "PixMoCapDataset",
    "PixMoCapDatasetConfig",
    "MultimodalCollator",
    "MultimodalCollatorConfig",
    "MultimodalDataLoader",
    "build_packed_sequence",
    "ATTEND_ALL_SUBSEGMENT_ID",
]
