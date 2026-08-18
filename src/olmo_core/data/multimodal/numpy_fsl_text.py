"""Adapt fixed-length numpy text datasets to the multimodal example schema."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from olmo_core.config import Config
from olmo_core.data.numpy_dataset import (
    NumpyFSLDatasetBase,
    NumpyFSLDatasetConfig,
    NumpyFSLDatasetMixture,
)
from olmo_core.io import get_file_size
from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

__all__ = ["NumpyFSLTextDataset", "NumpyFSLTextDatasetConfig"]

_CONTENT_FINGERPRINT_VERSION = "numpy-fsl-text-adapter-v1"
_CONTENT_FINGERPRINT_DOMAIN = b"olmo-core-numpy-fsl-text-adapter-v1\0"


@dataclass
class NumpyFSLTextDatasetConfig(Config):
    """Configuration for :class:`NumpyFSLTextDataset`.

    :param dataset: Fixed-length numpy dataset configuration to adapt.
    """

    dataset: NumpyFSLDatasetConfig

    def build(self) -> "NumpyFSLTextDataset":
        """Build the child dataset and wrap it for multimodal collation."""
        if self.dataset.source_mixture_config is not None:
            raise ValueError(
                "NumpyFSLTextDatasetConfig does not support source_mixture_config; "
                "use paths or an official DataMix"
            )
        if self.dataset.generate_doc_lengths:
            raise ValueError("NumpyFSLTextDatasetConfig does not support generate_doc_lengths=True")
        dataset = self.dataset.build()
        if not isinstance(dataset, NumpyFSLDatasetBase):
            raise TypeError(
                "NumpyFSLTextDatasetConfig requires a NumpyFSLDatasetBase child, "
                f"got {type(dataset).__name__}"
            )
        return NumpyFSLTextDataset(dataset)


class NumpyFSLTextDataset:
    """Expose a :class:`NumpyFSLDatasetBase` as a text-only multimodal source.

    The adapter applies the same next-token shift and target masking as
    :func:`olmo_core.data.utils.get_labels`. Repetition-filtered instances have all labels
    ignored while retaining ``sequence_length - 1`` loss weight, matching the standard OLMo
    DDP loss divisor. No tokens are added, removed, or retokenized.

    :param dataset: The fixed-length numpy dataset to adapt.
    """

    content_fingerprint_version = _CONTENT_FINGERPRINT_VERSION
    """Version of :attr:`content_fingerprint` and the adapter's output semantics."""

    def __init__(self, dataset: NumpyFSLDatasetBase):
        if isinstance(dataset, NumpyFSLDatasetMixture):
            raise TypeError(
                "NumpyFSLTextDataset does not support NumpyFSLDatasetMixture because that "
                "child does not yet advertise a complete semantic fingerprint"
            )
        if dataset.generate_doc_lengths:
            raise ValueError("NumpyFSLTextDataset does not support children that generate doc_lens")
        self.dataset = dataset

    @property
    def sequence_length(self) -> int:
        """Return the fixed number of tokens in every child instance."""
        return self.dataset.sequence_length

    @property
    def content_fingerprint(self) -> str:
        """Return a stable digest of the adapter contract and child dataset identity."""
        instance_filter = self.dataset.instance_filter_config
        payload = {
            "adapter_version": self.content_fingerprint_version,
            "child_fingerprint": self.dataset.fingerprint,
            "child_fingerprint_version": self.dataset.fingerprint_version,
            # The child fingerprint deliberately omits these fields for historical
            # compatibility, but they change this adapter's labels / loss weights.
            "instance_filter_config": (
                None if instance_filter is None else instance_filter.as_config_dict()
            ),
            "label_mask_files": [
                {
                    "basename": os.path.basename(str(path)),
                    "size": get_file_size(path),
                }
                for path in (self.dataset.label_mask_paths or ())
            ],
            # NumpyFSLDataset's fingerprint does not currently include this property.
            "sequence_length": self.sequence_length,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(_CONTENT_FINGERPRINT_DOMAIN + encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Alias :attr:`content_fingerprint` for numpy data-loader compatibility."""
        return self.content_fingerprint

    @property
    def fingerprint_version(self) -> str:
        """Alias :attr:`content_fingerprint_version` for numpy data-loader compatibility."""
        return self.content_fingerprint_version

    def prepare(self) -> None:
        """Delegate all dataset preparation to the wrapped numpy dataset."""
        self.dataset.prepare()

    def __len__(self) -> int:
        """Return the number of fixed-length child instances."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Materialize one epoch-zero text-only multimodal example."""
        return self.get(index, epoch=0)

    def get(self, index: int, epoch: int = 0) -> Dict[str, Any]:
        """Materialize one text-only multimodal example.

        ``epoch`` is accepted for compatibility with
        :class:`~olmo_core.data.multimodal.MixtureDataLoader`; numpy instances are invariant
        to source epoch.

        :param index: Child dataset index.
        :param epoch: Ignored source epoch.
        :returns: Shifted language-model fields and empty vision arrays.
        """
        del epoch
        item = self.dataset[index]
        input_ids = np.asarray(item["input_ids"], dtype=np.int64).copy()
        if input_ids.shape != (self.sequence_length,):
            raise ValueError(
                "NumpyFSLTextDataset child input_ids must have shape "
                f"({self.sequence_length},), got {input_ids.shape}"
            )

        label_mask_value = item.get("label_mask")
        if label_mask_value is None:
            label_mask = np.ones(self.sequence_length, dtype=np.bool_)
        else:
            label_mask = np.asarray(label_mask_value, dtype=np.bool_).copy()
            if label_mask.shape != input_ids.shape:
                raise ValueError(
                    "NumpyFSLTextDataset child label_mask must match input_ids shape, "
                    f"got {label_mask.shape} and {input_ids.shape}"
                )

        labels = np.full(self.sequence_length, -100, dtype=np.int64)
        loss_masks = np.zeros(self.sequence_length, dtype=np.float32)
        valid_instance = bool(item.get("instance_mask", True))
        if valid_instance:
            labels[:-1] = np.where(label_mask[1:], input_ids[1:], -100)
            loss_masks[:-1] = label_mask[1:]
        else:
            # The standard OLMo DDP path ignores every label but adds L-1 back to the
            # divisor for a repetition-filtered instance, irrespective of label_mask.
            loss_masks[:-1] = 1.0

        example: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_masks": loss_masks,
            "position_ids": np.arange(self.sequence_length, dtype=np.int64),
            "token_type_ids": np.zeros(self.sequence_length, dtype=np.int64),
            "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
            "pooled_patches_idx": np.full((0, POOL_H * POOL_W), -1, dtype=np.int64),
        }
        if "metadata" in item:
            example["metadata"] = item["metadata"]
        return example
