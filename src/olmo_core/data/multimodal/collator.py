"""Collator for Molmo2 multimodal training batches.

Right-pads variable-length token / image fields produced by
:class:`~olmo_core.data.multimodal.pixmo_cap.PixMoCapDataset` into dense batch
tensors consumable by :class:`~olmo_core.nn.vision.MultimodalLM`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from olmo_core.config import Config

__all__ = ["MultimodalCollator", "MultimodalCollatorConfig"]


@dataclass
class MultimodalCollatorConfig(Config):
    """Configuration for :class:`MultimodalCollator`."""

    pad_token_id: int
    """Token id used to pad ``input_ids`` (typically EOS)."""

    label_ignore_index: int = -100
    """Pad value for ``labels`` (ignored by the loss)."""

    pad_sequence_length: Optional[int] = None
    """If set, pad every batch's token fields to this fixed length instead of the
    per-batch max. Use this to give every batch a constant token count (required by
    the token-based :class:`~olmo_core.train.Trainer` batching)."""

    def build(self) -> "MultimodalCollator":
        return MultimodalCollator(
            pad_token_id=self.pad_token_id,
            label_ignore_index=self.label_ignore_index,
            pad_sequence_length=self.pad_sequence_length,
        )


class MultimodalCollator:
    """Pad and stack a list of per-example dicts into a training batch.

    Each example is a dict of ``np.ndarray`` with (at least) ``input_ids``,
    ``labels``, ``loss_masks``, ``position_ids``, ``token_type_ids``, ``images``,
    ``pooled_patches_idx`` and optionally ``subsegment_ids``.

    Token fields are right-padded to the batch's max sequence length; ``images`` is
    padded along the crop axis and ``pooled_patches_idx`` along the pooled-token axis
    (with ``-1``, which the connector treats as padding).
    """

    def __init__(
        self,
        pad_token_id: int,
        label_ignore_index: int = -100,
        pad_sequence_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.label_ignore_index = label_ignore_index
        self.pad_sequence_length = pad_sequence_length

    def _pad_1d(self, arrays: List[np.ndarray], value, max_len: int, dtype) -> torch.Tensor:
        out = np.full((len(arrays), max_len), value, dtype=dtype)
        for i, a in enumerate(arrays):
            out[i, : len(a)] = a
        return torch.from_numpy(out)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        if self.pad_sequence_length is not None:
            if max_len > self.pad_sequence_length:
                raise ValueError(
                    f"example length {max_len} exceeds pad_sequence_length "
                    f"{self.pad_sequence_length}"
                )
            max_len = self.pad_sequence_length
        max_crops = max(ex["images"].shape[0] for ex in examples)
        max_pool = max(ex["pooled_patches_idx"].shape[0] for ex in examples)
        n_patches = examples[0]["images"].shape[1]
        patch_dim = examples[0]["images"].shape[2]
        pool_size = examples[0]["pooled_patches_idx"].shape[1]

        batch: Dict[str, torch.Tensor] = {
            "input_ids": self._pad_1d(
                [ex["input_ids"] for ex in examples], self.pad_token_id, max_len, np.int64
            ),
            "labels": self._pad_1d(
                [ex["labels"] for ex in examples], self.label_ignore_index, max_len, np.int64
            ),
            "loss_masks": self._pad_1d(
                [ex["loss_masks"] for ex in examples], 0.0, max_len, np.float32
            ),
            "position_ids": self._pad_1d(
                [ex["position_ids"] for ex in examples], 0, max_len, np.int64
            ),
            "token_type_ids": self._pad_1d(
                [ex["token_type_ids"] for ex in examples], 0, max_len, np.int64
            ),
        }

        # Images: (B, max_crops, n_patches, patch_dim), pad new crops with 0.
        images = np.zeros((len(examples), max_crops, n_patches, patch_dim), dtype=np.float32)
        # Pooled patch indices: (B, max_pool, pool_size), pad with -1 (connector ignores).
        pooled = np.full((len(examples), max_pool, pool_size), -1, dtype=np.int64)
        for i, ex in enumerate(examples):
            im = ex["images"]
            images[i, : im.shape[0]] = im
            pp = ex["pooled_patches_idx"]
            pooled[i, : pp.shape[0]] = pp
        batch["images"] = torch.from_numpy(images)
        batch["pooled_patches_idx"] = torch.from_numpy(pooled)

        # Subsegment ids only when at least one example is multi-branch (packed). For
        # padded / single-branch positions a uniform id leaves attention unrestricted.
        subseg_arrays: Optional[List[np.ndarray]] = None
        if any("subsegment_ids" in ex for ex in examples):
            subseg_arrays = [
                ex.get("subsegment_ids", np.zeros(len(ex["input_ids"]), dtype=np.int64))
                for ex in examples
            ]
            batch["subsegment_ids"] = self._pad_1d(subseg_arrays, 0, max_len, np.int64)

        return batch
