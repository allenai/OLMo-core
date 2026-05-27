"""
Multimodal collator.

Stacks a list of per-example dicts (produced by
:class:`~olmo_core.data.multimodal.preprocessor.MultimodalPreprocessor`) into
the batched tensor dict that :class:`~olmo_core.nn.vision.MultimodalTransformer.forward`
consumes.

The tricky part is variable image-layout across the batch (different
``n_crops`` and ``n_pooled`` per example, which happens in
``overlap_and_resize`` mode). The model's splice asserts that the total
number of ``<im_patch>`` tokens in ``input_ids`` equals the total number of
pooled image features, so we:

1. Pad ``pooled_patches_idx`` to the max ``n_pooled`` in the batch with all-``-1``
   rows. These rows produce zero features (since every patch index is masked
   to zero in the connector).
2. Append ``(max_n_pooled - n_pooled)`` dummy ``<im_patch>`` tokens to each
   example's ``input_tokens`` (with ``loss_mask = 0``) so the count matches.
   These dummy positions receive ``+= 0`` from the splice, so they're
   functionally invisible to training.
3. Pad ``images`` to the max ``n_crops`` with zeros and ``image_masks`` with
   zeros. These extra crops aren't referenced by any non-``-1`` pool index, so
   the vision tower processes them but the connector never gathers them.

The base tokenizer's ``pad_token_id`` is used to pad ``input_tokens`` and
``loss_masks`` to the max sequence length in the batch.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ...config import Config
from .tokens import MultimodalTokenizerConfig

__all__ = [
    "MultimodalCollatorConfig",
    "MultimodalCollator",
]


@dataclass
class MultimodalCollatorConfig(Config):
    """Configuration for :class:`MultimodalCollator`."""

    tokenizer: MultimodalTokenizerConfig
    """Tokenizer providing ``image_patch_id`` and ``base.pad_token_id``."""

    pad_to_multiple_of: int = 1
    """If > 1, pad the sequence length up to the next multiple. Useful for
    hardware alignment (e.g. flash-attn) but optional."""

    def build(self) -> "MultimodalCollator":
        return MultimodalCollator(self)


class MultimodalCollator:
    """Stack per-example dicts into a single batched dict of ``torch.Tensor``."""

    def __init__(self, cfg: MultimodalCollatorConfig):
        self.cfg = cfg
        self.image_patch_id = cfg.tokenizer.image_patch_id
        self.pad_id = cfg.tokenizer.base.pad_token_id

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Batch ``examples`` into tensors.

        :param examples: List of dicts with keys ``input_tokens``, ``loss_masks``,
            ``images``, ``image_masks``, ``pooled_patches_idx`` (the output of
            :class:`MultimodalPreprocessor`).
        :returns: Dict with keys ``input_ids``, ``loss_masks``, ``images``,
            ``image_masks``, ``pooled_patches_idx`` — all ``torch.Tensor``.
        """
        if not examples:
            raise ValueError("collator received an empty batch")

        B = len(examples)
        cfg = self.cfg

        # Determine output shapes.
        max_n_pooled = max(e["pooled_patches_idx"].shape[0] for e in examples)
        pool_size = examples[0]["pooled_patches_idx"].shape[1]
        max_n_crops = max(e["images"].shape[0] for e in examples)
        # Per-crop dimensions come from any example's `images` shape — they're
        # fixed by the preprocessor config and present even when n_crops=0.
        n_patches_per_crop = examples[0]["images"].shape[1]
        patch_dim = examples[0]["images"].shape[2]
        for e in examples[1:]:
            if e["images"].shape[1:] != (n_patches_per_crop, patch_dim):
                raise ValueError(
                    "All examples must share patch shape; got "
                    f"{e['images'].shape[1:]} vs ({n_patches_per_crop}, {patch_dim})"
                )

        # Sequence length includes any dummy <im_patch> tokens we'll append.
        raw_lengths = [
            e["input_tokens"].shape[0] + (max_n_pooled - e["pooled_patches_idx"].shape[0])
            for e in examples
        ]
        max_seq = max(raw_lengths)
        if cfg.pad_to_multiple_of > 1:
            mult = cfg.pad_to_multiple_of
            max_seq = mult * ((max_seq + mult - 1) // mult)

        # Allocate batched tensors.
        input_ids = torch.full((B, max_seq), self.pad_id, dtype=torch.long)
        loss_masks = torch.zeros((B, max_seq), dtype=torch.float32)
        images = torch.zeros((B, max_n_crops, n_patches_per_crop, patch_dim), dtype=torch.float32)
        image_masks = torch.zeros((B, max_n_crops, n_patches_per_crop), dtype=torch.float32)
        pooled_patches_idx = torch.full((B, max_n_pooled, pool_size), -1, dtype=torch.long)

        # Fill.
        for i, e in enumerate(examples):
            tokens = e["input_tokens"]
            n_pooled = e["pooled_patches_idx"].shape[0]
            n_dummy = max_n_pooled - n_pooled

            seq_len = tokens.shape[0]
            input_ids[i, :seq_len] = torch.from_numpy(tokens)
            loss_masks[i, :seq_len] = torch.from_numpy(e["loss_masks"])
            # Dummy <im_patch> tokens after the real content, before pad tokens.
            if n_dummy > 0:
                input_ids[i, seq_len : seq_len + n_dummy] = self.image_patch_id
                # loss_mask stays 0 for these positions.

            n_crops = e["images"].shape[0]
            if n_crops > 0:
                images[i, :n_crops, : e["images"].shape[1]] = torch.from_numpy(e["images"])
                image_masks[i, :n_crops, : e["image_masks"].shape[1]] = torch.from_numpy(
                    e["image_masks"]
                )
            if n_pooled > 0:
                pooled_patches_idx[i, :n_pooled] = torch.from_numpy(e["pooled_patches_idx"])

        return {
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "images": images,
            "image_masks": image_masks,
            "pooled_patches_idx": pooled_patches_idx,
        }
