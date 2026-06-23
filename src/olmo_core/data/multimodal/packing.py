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

from typing import Any, Callable, Dict, Iterator, List, Sequence

import numpy as np

__all__ = ["pack_examples", "greedy_pack_indices", "iter_packs"]


def greedy_pack_indices(lengths: Sequence[int], seq_len: int) -> List[List[int]]:
    """Greedily group example indices so each group's total length ``<= seq_len``.

    First-fit-decreasing is overkill here; a simple next-fit over the given order keeps the
    sampling order stable (important for mixture proportions) and is what mm_olmo does.

    :param lengths: real token length of each example, in the order they should be packed.
    :param seq_len: maximum packed length.
    :returns: a list of groups, each a list of indices into ``lengths``.
    """
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_len = 0
    for i, n in enumerate(lengths):
        n = int(n)
        if cur and cur_len + n > seq_len:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += n
    if cur:
        groups.append(cur)
    return groups


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
    refs: Sequence[Any],
    get_example: Callable[[Any], Dict[str, np.ndarray]],
    seq_len: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """Greedily next-fit-pack examples from ``refs`` into ``<= seq_len`` sequences forever.

    Cycles ``refs`` indefinitely (the caller caps the number of batches), so every data-
    parallel rank yields the same number of batches regardless of how its examples happen to
    pack — avoiding a collective desync. An example longer than ``seq_len`` is emitted alone
    (the collator tail-truncates it; the image block at the front is preserved).

    :param refs: example references (indices, ``(src, idx)`` tuples, …) for this rank.
    :param get_example: maps a ref to its example dict.
    :param seq_len: target packed length.
    """
    if len(refs) == 0:
        raise ValueError("iter_packs requires a non-empty `refs`")
    cur: List[Dict[str, np.ndarray]] = []
    cur_len = 0
    i = 0
    n = len(refs)
    while True:
        ex = get_example(refs[i % n])
        i += 1
        length = len(ex["input_ids"])
        if cur and cur_len + length > seq_len:
            yield pack_examples(cur)
            cur, cur_len = [], 0
        cur.append(ex)
        cur_len += length
