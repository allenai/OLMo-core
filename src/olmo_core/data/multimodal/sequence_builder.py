"""Build packed training sequences for Molmo2 multi-annotation multimodal data.

This is a dependency-free port of the sequence-assembly performed by ``mm_olmo``'s
``RefactoredExamplePreprocessor`` (``flatten_tree`` + ``build_sequence`` in
``olmo/models/molmo2/example_preprocessor.py``) for the common case of a single
shared prefix (BOS + image block + user prompt + assistant header) that branches
into one or more assistant responses (e.g. a caption plus a transcript).

Multiple branches that share one image are packed into a single sequence where
each branch is *isolated* from its siblings via subsegment attention and the
branches share an overlapping RoPE position range (each branch continues from the
end of the shared prefix). The loss is a float per-token weight, response-only,
scaled by ``1/sqrt(n_branches)`` under ``root_subsegments`` weighting so the packed
sequence is equivalent in expectation to sampling one annotation.

See :class:`~olmo_core.nn.vision.MultimodalLM` for how ``subsegment_ids`` /
``position_ids`` / ``loss_masks`` are consumed at training time.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from olmo_core.nn.vision.molmo2_tokens import IMAGE_TOKEN_IDS

# Subsegment id of the shared prefix. Larger than any branch id so that, under the
# ``subseg[q] <= subseg[k]`` rule, prefix tokens only attend to other prefix tokens
# while every branch attends the prefix. Matches mm_olmo's ``ATTEND_ALL_SUBSEGMENT_ID``.
ATTEND_ALL_SUBSEGMENT_ID = 10000

LOSS_TOKEN_WEIGHTINGS = ("none", "root_subsegments", "root_subsegments_root_tokens")


def build_packed_sequence(
    prefix_ids: Sequence[int],
    response_id_lists: Sequence[Sequence[int]],
    *,
    eos_id: int,
    image_token_ids: frozenset = IMAGE_TOKEN_IDS,
    loss_token_weighting: str = "root_subsegments",
) -> Dict[str, np.ndarray]:
    """Assemble a packed training example from a shared prefix and response branches.

    :param prefix_ids: Token IDs of the shared prefix, ending with the assistant
        header (e.g. ``…<|im_start|>assistant\\n``). The final token is "carried over"
        into each branch as a non-loss token so each branch copy can predict its own
        first response token (matching mm_olmo).
    :param response_id_lists: One token-ID list per annotation (e.g. caption,
        transcript). Each is the assistant response **without** a trailing EOS — the
        EOS is supplied as the target at the branch's final position.
    :param eos_id: End-of-sequence token id used as the target at each branch end.
    :param image_token_ids: Token IDs that count as image tokens for ``token_type_ids``.
    :param loss_token_weighting: ``"none"`` (binary response weights),
        ``"root_subsegments"`` (each branch scaled by ``1/sqrt(n_branches)``), or
        ``"root_subsegments_root_tokens"`` (additionally ``2/sqrt(n_response_tokens)``
        per branch).

    :returns: A dict of 1-D ``np.ndarray`` with keys ``input_ids``, ``labels``,
        ``loss_masks`` (float32), ``position_ids``; plus ``subsegment_ids`` when there
        is more than one branch. ``labels``/``loss_masks`` are already shifted to align
        with next-token prediction. ``token_type_ids`` is also returned.

    :raises ValueError: If no responses are given or the weighting is unknown.
    """
    if loss_token_weighting not in LOSS_TOKEN_WEIGHTINGS:
        raise ValueError(
            f"Unknown loss_token_weighting {loss_token_weighting!r}; "
            f"expected one of {LOSS_TOKEN_WEIGHTINGS}"
        )
    n_branches = len(response_id_lists)
    if n_branches == 0:
        raise ValueError("`response_id_lists` must contain at least one response")

    prefix_ids = list(prefix_ids)
    if len(prefix_ids) == 0:
        raise ValueError("`prefix_ids` must be non-empty")

    root_length = loss_token_weighting == "root_subsegments_root_tokens"

    # ``parts`` mirror mm_olmo's flatten_tree output: each is
    # (tokens, weight, start_position, subsegment_id, is_segment_end_token[]).
    parts: List[dict] = []

    if n_branches == 1:
        # No branching: a single causal sequence (prefix + response), sequential
        # positions, no subsegments. Weight is binary unless root_length.
        response = list(response_id_lists[0])
        tokens = prefix_ids + response
        weight = 1.0
        if root_length:
            n_resp = len(response) + 1  # +1 for EOS
            weight = 2.0 / np.sqrt(n_resp) if n_resp else 0.0
        loss = np.zeros(len(tokens), dtype=np.float32)
        loss[len(prefix_ids) :] = weight  # response tokens get loss
        seg_end = np.zeros(len(tokens), dtype=bool)
        seg_end[-1] = True
        parts.append(
            dict(
                tokens=np.asarray(tokens, dtype=np.int64),
                loss=loss,
                position=np.arange(len(tokens), dtype=np.int64),
                subsegment_id=None,
                seg_end=seg_end,
            )
        )
    else:
        # Branching: shared prefix (minus carry-over) + N isolated branches that each
        # start with the carried-over last prefix token and continue from position
        # ``len(prefix) - 1``.
        carry_over = prefix_ids[-1]
        prefix_body = prefix_ids[:-1]
        start_position = len(prefix_ids) - 1

        prefix_tokens = np.asarray(prefix_body, dtype=np.int64)
        parts.append(
            dict(
                tokens=prefix_tokens,
                loss=np.zeros(len(prefix_body), dtype=np.float32),
                position=np.arange(len(prefix_body), dtype=np.int64),
                subsegment_id=ATTEND_ALL_SUBSEGMENT_ID,
                seg_end=np.zeros(len(prefix_body), dtype=bool),
            )
        )
        for branch_idx, response in enumerate(response_id_lists):
            response = list(response)
            branch_tokens = [carry_over] + response
            branch_weight = 1.0
            if root_length:
                n_resp = len(response) + 1  # +1 for EOS
                branch_weight = 2.0 / np.sqrt(n_resp) if n_resp else 0.0
            loss = np.zeros(len(branch_tokens), dtype=np.float32)
            loss[1:] = branch_weight  # carry-over (idx 0) is non-loss; response gets loss
            seg_end = np.zeros(len(branch_tokens), dtype=bool)
            seg_end[-1] = True
            parts.append(
                dict(
                    tokens=np.asarray(branch_tokens, dtype=np.int64),
                    loss=loss,
                    position=np.arange(
                        start_position, start_position + len(branch_tokens), dtype=np.int64
                    ),
                    subsegment_id=branch_idx,
                    seg_end=seg_end,
                )
            )

    input_ids = np.concatenate([p["tokens"] for p in parts], 0)
    loss_mask = np.concatenate([p["loss"] for p in parts], 0)
    position_ids = np.concatenate([p["position"] for p in parts], 0)
    seg_ends = np.concatenate([p["seg_end"] for p in parts], 0)

    # Labels via shift; segment ends predict EOS rather than the next segment's token.
    labels = np.zeros_like(input_ids)
    labels[:-1] = input_ids[1:]
    labels[seg_ends] = eos_id

    # Loss masks shifted to align with labels; segment ends keep the unshifted weight.
    loss_mask_shifted = np.zeros_like(loss_mask)
    loss_mask_shifted[:-1] = loss_mask[1:]
    loss_mask_shifted[seg_ends] = loss_mask[seg_ends]

    # root_subsegments: scale every loss weight by 1/sqrt(n_branches) (no-op for 1 branch).
    if loss_token_weighting in ("root_subsegments", "root_subsegments_root_tokens"):
        if n_branches > 1:
            loss_mask_shifted = loss_mask_shifted / np.sqrt(n_branches)

    token_type_ids = np.isin(input_ids, np.fromiter(image_token_ids, dtype=np.int64)).astype(
        np.int64
    )

    out: Dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_masks": loss_mask_shifted.astype(np.float32),
        "position_ids": position_ids,
        "token_type_ids": token_type_ids,
    }
    if n_branches > 1:
        out["subsegment_ids"] = np.concatenate(
            [np.full(len(p["tokens"]), p["subsegment_id"], dtype=np.int64) for p in parts],
            0,
        )
    return out
