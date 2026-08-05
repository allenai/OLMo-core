"""Sequence packing for Molmo2 multimodal training.

Most stage-1 examples are far shorter than the training sequence length (caption/pointing
real lengths are ~1.1-1.5k tokens vs a 4096 pad length → ~65% of every sequence is
padding, i.e. wasted compute). Packing concatenates several whole examples into one
sequence so there is little/no padding, recovering that compute as useful tokens — the
OLMo-core analogue of mm_olmo's dynamic packer.

Cross-example isolation reuses the same machinery as intra-example branch isolation:

* a per-token ``example_ids`` vector marks which packed example each token belongs to;
  :class:`~olmo_core.nn.vision.MultimodalLM` ANDs ``example_ids[q] == example_ids[k]`` into
  the attention keep-mask so a token never attends across an example boundary.
* per-example ``position_ids`` are preserved (each example keeps its own 0-based RoPE
  positions / branch overlap), so packing is invisible to RoPE.
* ``subsegment_ids`` are concatenated (examples without branches get a constant id, which
  is unrestrictive within the example); the existing ``subseg[q] <= subseg[k]`` rule then
  applies *within* each example, gated by the example-id equality above.
* image crops are concatenated along the crop axis and each example's
  ``pooled_patches_idx`` is offset by the running crop-patch count so the connector still
  gathers the right patches.

Examples are never split, so the ``#<im_patch> == #pooled-features`` invariant is preserved
per example (and hence for the pack).
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

__all__ = ["pack_examples", "greedy_pack_indices", "iter_packs", "example_has_images"]


def example_has_images(ex: Dict[str, np.ndarray]) -> bool:
    """True when the example carries real image crops (not text-only NLP)."""
    return ex["images"].shape[0] > 0


def example_crop_count(ex: Dict[str, np.ndarray]) -> int:
    """Number of image crops in an example (0 for text-only)."""
    return int(ex["images"].shape[0])


def greedy_pack_indices(
    lengths: Sequence[int],
    seq_len: int,
    *,
    crop_counts: Optional[Sequence[int]] = None,
    max_crops_per_pack: Optional[int] = None,
) -> List[List[int]]:
    """Greedily group example indices so each group's total length ``<= seq_len``.

    A simple next-fit over the given order keeps the sampling order stable and remains the
    default packing policy for backwards compatibility.

    :param lengths: real token length of each example, in the order they should be packed.
    :param seq_len: maximum packed length.
    :param crop_counts: per-example crop counts (required when ``max_crops_per_pack`` is set).
    :param max_crops_per_pack: maximum total crops in one pack (mm_olmo ``image_weight`` parity).
    :returns: a list of groups, each a list of indices into ``lengths``.
    """
    if max_crops_per_pack is not None and crop_counts is None:
        raise ValueError("crop_counts is required when max_crops_per_pack is set")
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_len = 0
    cur_crops = 0
    for i, n in enumerate(lengths):
        n = int(n)
        crops = int(crop_counts[i]) if crop_counts is not None else 0
        over_tokens = cur and cur_len + n > seq_len
        over_crops = (
            max_crops_per_pack is not None and cur and cur_crops + crops > max_crops_per_pack
        )
        if over_tokens or over_crops:
            groups.append(cur)
            cur, cur_len, cur_crops = [], 0, 0
        cur.append(i)
        cur_len += n
        cur_crops += crops
    if cur:
        groups.append(cur)
    return groups


def _select_buffered_pack_indices(
    lengths: Sequence[int],
    crop_counts: Sequence[int],
    seq_len: int,
    max_crops_per_pack: int,
) -> List[int]:
    """Select one pack with Molmo2's two-constraint dynamic program."""
    token_granularity = max(1, seq_len // 512)
    token_values = np.asarray(
        [(int(n) + token_granularity - 1) // token_granularity for n in lengths],
        dtype=np.int64,
    )
    crop_values = np.asarray(crop_counts, dtype=np.int64)
    # Use a conservative quantized capacity so selected examples are guaranteed to fit even
    # when ``seq_len`` is not divisible by the granularity.
    max_tokens = seq_len // token_granularity

    # Match Molmo2's objective: useful text tokens plus image crops. The crop dimension is
    # also a hard constraint, so the small image term only resolves otherwise similar packs.
    objective = np.asarray(lengths, dtype=np.float32) + crop_values.astype(np.float32)
    n_items = len(lengths)
    dp = np.zeros((n_items + 1, max_tokens + 1, max_crops_per_pack + 1), dtype=np.float32)

    for item in range(1, n_items + 1):
        token_value = int(token_values[item - 1])
        crop_value = int(crop_values[item - 1])
        dp[item] = dp[item - 1]
        if token_value <= max_tokens and crop_value <= max_crops_per_pack:
            take = (
                dp[
                    item - 1,
                    : max_tokens + 1 - token_value,
                    : max_crops_per_pack + 1 - crop_value,
                ]
                + objective[item - 1]
            )
            dp[item, token_value:, crop_value:] = np.maximum(
                dp[item, token_value:, crop_value:], take
            )

    selected: List[int] = []
    tokens_left, crops_left = max_tokens, max_crops_per_pack
    for item in range(n_items, 0, -1):
        token_value = int(token_values[item - 1])
        crop_value = int(crop_values[item - 1])
        if (
            token_value <= tokens_left
            and crop_value <= crops_left
            and dp[item, tokens_left, crops_left] != dp[item - 1, tokens_left, crops_left]
        ):
            selected.append(item - 1)
            tokens_left -= token_value
            crops_left -= crop_value
    return selected


def _pop_buffered_pack(
    buffer: List[Dict[str, np.ndarray]],
    seq_len: int,
    max_crops_per_pack: int,
) -> List[Dict[str, np.ndarray]]:
    selected = _select_buffered_pack_indices(
        [len(ex["input_ids"]) for ex in buffer],
        [example_crop_count(ex) for ex in buffer],
        seq_len,
        max_crops_per_pack,
    )
    if not selected:
        raise RuntimeError("Buffered packer could not select an example within its constraints")
    packed = [buffer[i] for i in selected]
    for i in sorted(selected, reverse=True):
        buffer.pop(i)
    return packed


def pack_examples(examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate several example dicts into one packed example.

    Each input is a dict as produced by the stage-1 datasets (``input_ids``, ``labels``,
    ``loss_masks``, ``position_ids``, ``token_type_ids``, optional ``subsegment_ids``,
    ``images`` ``(n_crops, n_patches, patch_dim)``, ``pooled_patches_idx``
    ``(n_pooled, pool_size)``). The result is the concatenation along the token axis (and
    crop axis for images), plus an ``example_ids`` vector. It is **not** padded — the
    collator pads to the batch/seq length.

    :raises ValueError: if ``examples`` is empty.
    """
    if not examples:
        raise ValueError("pack_examples requires at least one example")

    tok_keys = ["input_ids", "labels", "loss_masks", "position_ids", "token_type_ids"]
    out: Dict[str, np.ndarray] = {}
    for k in tok_keys:
        out[k] = np.concatenate([ex[k] for ex in examples], axis=0)

    # Subsegment ids: keep each example's own (branch isolation); examples without branches
    # get a constant id so `subseg[q] <= subseg[k]` is unrestrictive within them. The values
    # need not be globally unique — example-id equality gates cross-example attention.
    subseg_parts = []
    for ex in examples:
        n = len(ex["input_ids"])
        if "subsegment_ids" in ex:
            subseg_parts.append(ex["subsegment_ids"])
        else:
            subseg_parts.append(np.zeros(n, dtype=np.int64))
    out["subsegment_ids"] = np.concatenate(subseg_parts, axis=0)

    # Per-token example id (0, 1, 2, ...).
    out["example_ids"] = np.concatenate(
        [np.full(len(ex["input_ids"]), i, dtype=np.int64) for i, ex in enumerate(examples)],
        axis=0,
    )

    if any("_source_name" in ex for ex in examples):
        out["pack_source_names"] = [ex.get("_source_name", "?") for ex in examples]

    # Images: concatenate crops; offset each example's pooled indices by the running
    # crop-patch count so they index into the concatenated (total_crops * n_patches) axis.
    images = [ex["images"] for ex in examples]
    n_patches = next((im.shape[1] for im in images if im.shape[0]), images[0].shape[1])
    patch_dim = next((im.shape[2] for im in images if im.shape[0]), images[0].shape[2])
    pool_size = examples[0]["pooled_patches_idx"].shape[1]

    pooled_parts: List[np.ndarray] = []
    crop_offset = 0
    for ex in examples:
        im = ex["images"]
        pp = ex["pooled_patches_idx"].copy()
        if pp.shape[0]:
            valid = pp >= 0
            pp[valid] = pp[valid] + crop_offset * n_patches
            pooled_parts.append(pp)
        crop_offset += im.shape[0]

    out["images"] = (
        np.concatenate([im for im in images if im.shape[0]], axis=0)
        if any(im.shape[0] for im in images)
        else np.zeros((0, n_patches, patch_dim), dtype=np.float32)
    )
    out["pooled_patches_idx"] = (
        np.concatenate(pooled_parts, axis=0)
        if pooled_parts
        else np.full((0, pool_size), -1, dtype=np.int64)
    )
    return out


def iter_packs(
    examples: Iterable[Dict[str, np.ndarray]],
    seq_len: int,
    *,
    max_crops_per_pack: Optional[int] = None,
    buffer_size: int = 0,
) -> Iterator[Dict[str, np.ndarray]]:
    """Pack a stream of example dicts into ``<= seq_len`` sequences.

    ``examples`` is an iterator of example dicts — typically an infinite, cycled (and
    optionally prefetched) stream, in which case this yields indefinitely and the caller
    caps the number of batches. Cycling in the caller keeps every data-parallel rank
    yielding the same number of batches regardless of how its examples pack (no collective
    desync). An example longer than ``seq_len`` is emitted alone (the collator tail-truncates
    it; the image block at the front is preserved). A finite stream flushes all examples.

    The default ``buffer_size=0`` retains the original deterministic next-fit behavior.
    A positive buffer enables the same two-constraint knapsack policy used by Molmo2: it
    selects examples that maximize useful tokens while respecting the token and crop
    budgets. Buffered packing may safely mix image and text-only examples because selected
    packs are guaranteed not to be tail-truncated.

    :param examples: iterator of per-example dicts (the heavy loading happens upstream, so it
        can be prefetched off the training thread).
    :param seq_len: target packed length.
    :param max_crops_per_pack: cap total image crops per pack (mm_olmo crop-budget parity).
    :param buffer_size: number of examples considered by the dynamic packing solver; zero
        keeps the original next-fit policy.
    """
    if buffer_size < 0:
        raise ValueError("buffer_size must be non-negative")
    if buffer_size:
        if max_crops_per_pack is None:
            raise ValueError("max_crops_per_pack is required when buffer_size is positive")
        if max_crops_per_pack <= 0:
            raise ValueError("max_crops_per_pack must be positive")

        buffer: List[Dict[str, np.ndarray]] = []
        for ex in examples:
            length = len(ex["input_ids"])
            crops = example_crop_count(ex)
            # Preserve the existing handling of an individually oversized example. Stage-1
            # datasets are bounded by these constraints, but emitting alone is safer than
            # allowing an invalid item to stall an infinite stream.
            if length > seq_len or crops > max_crops_per_pack:
                yield pack_examples([ex])
                continue
            if len(buffer) < buffer_size:
                buffer.append(ex)
                continue
            packed = _pop_buffered_pack(buffer, seq_len, max_crops_per_pack)
            buffer.append(ex)
            yield pack_examples(packed)

        while buffer:
            yield pack_examples(_pop_buffered_pack(buffer, seq_len, max_crops_per_pack))
        return

    cur: List[Dict[str, np.ndarray]] = []
    cur_len = 0
    cur_crops = 0
    for ex in examples:
        length = len(ex["input_ids"])
        crops = example_crop_count(ex)
        # Do not mix text-only NLP (e.g. Tulu4) with image-bearing examples in one pack:
        # head-truncating a cross-modal pack can orphan <im_patch> tokens from their
        # pooled rows (mm_olmo uses separate packing constraints when NLP is enabled).
        over_tokens = cur and cur_len + length > seq_len
        over_crops = (
            max_crops_per_pack is not None
            and cur
            and example_has_images(cur[0])
            and cur_crops + crops > max_crops_per_pack
        )
        if cur and (
            over_tokens or over_crops or example_has_images(ex) != example_has_images(cur[0])
        ):
            yield pack_examples(cur)
            cur, cur_len, cur_crops = [], 0, 0
        cur.append(ex)
        cur_len += length
        cur_crops += crops
    if cur:
        yield pack_examples(cur)
