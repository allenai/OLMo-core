"""Bounded same-geometry image-content diagnostics for in-loop evaluation."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from olmo_core.exceptions import OLMoConfigurationError

from .matched_wrong_image import (
    MultimodalMatchedWrongImageDataset,
    _array_descriptor,
    _geometry,
    _get_example,
)


def build_bounded_image_pairs(
    dataset: Any, *, examples: int, max_candidates: int, seed: int = 0
) -> list[tuple[int, int]]:
    """Pair distinct image tensors within exact geometry groups in a bounded validation prefix.

    Only the first ``min(max_candidates, len(dataset))`` rows are preprocessed, at source
    epoch zero. Image digests reject duplicate pixel tensors; geometry uses the same exact
    image-shape and pooling-index contract as
    :func:`~olmo_core.eval.matched_wrong_image.build_matched_wrong_image_pairing`.
    No training population, annotation inventory, or source files are scanned.

    :param examples: Required number of paired recipients.
    :param max_candidates: Maximum validation rows to inspect for donors and recipients.
    :param seed: Seed for reproducible recipient and donor ordering.
    :returns: Recipient/donor dataset indices, with unique recipients and donors.
    """
    if examples <= 0 or max_candidates < examples or seed < 0:
        raise OLMoConfigurationError("Invalid bounded image-pairing size or seed")
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    seen: set[tuple[tuple[Any, ...], str]] = set()
    for index in range(min(max_candidates, len(dataset))):
        example = _get_example(dataset, index, 0)
        descriptors = {}
        for name in ("images", "pooled_patches_idx"):
            value = example.get(name)
            if not isinstance(value, (np.ndarray, torch.Tensor)) or 0 in value.shape:
                raise OLMoConfigurationError(
                    f"Image-pairing row {index} requires a nonempty {name!r} array"
                )
            descriptors[name] = _array_descriptor(value, field_name=name)
        geometry = _geometry({"example": descriptors})
        identity = (geometry, descriptors["images"]["sha256"])
        if identity not in seen:
            seen.add(identity)
            groups[geometry].append(index)

    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        rng.shuffle(group)
        pairs.extend(zip(group, group[1:] + group[:1]))
    if len(pairs) < examples:
        raise OLMoConfigurationError(
            f"Only {len(pairs)} validation rows have distinct exact-geometry image donors "
            f"within {min(max_candidates, len(dataset))} candidates; requested {examples}. "
            "Increase matched_image_candidates or reduce matched_image_examples."
        )
    rng.shuffle(pairs)
    return pairs[:examples]


class MultimodalImagePairDataset:
    """Read fixed correct-image or wrong-image recipients for an in-loop diagnostic.

    Pairing indices are prepared once by :func:`build_bounded_image_pairs`. Every replay
    reads source epoch zero. Wrong-image examples replace only the image tensor, retaining
    the recipient's labels, loss weights, geometry, and attention metadata.
    """

    def __init__(self, dataset: Any, pairs: Sequence[tuple[int, int]], *, wrong_images: bool):
        self.dataset = dataset
        self.pairs = tuple(pairs)
        self.wrong_images = wrong_images

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        """Read a recipient, checking donor geometry and distinct pixels before replacement."""
        if epoch != 0:
            raise OLMoConfigurationError("Image-pair evaluation requires source epoch zero")
        recipient_index, donor_index = self.pairs[index]
        recipient = dict(_get_example(self.dataset, recipient_index, 0))
        if self.wrong_images:
            donor = _get_example(self.dataset, donor_index, 0)
            equal = MultimodalMatchedWrongImageDataset._arrays_equal
            if (
                recipient["images"].shape != donor["images"].shape
                or recipient["images"].dtype != donor["images"].dtype
                or not equal(recipient["pooled_patches_idx"], donor["pooled_patches_idx"])
                or equal(recipient["images"], donor["images"])
            ):
                raise OLMoConfigurationError("Matched image geometry or distinctness changed")
            images = donor["images"]
            recipient["images"] = (
                images.clone() if isinstance(images, torch.Tensor) else images.copy()
            )
        return recipient
