"""Encode SFT chat turns + image into Molmo2 training tensors."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2
from olmo_core.nn.vision.molmo2_tokens import DEFAULT_MOLMO2_TOKEN_IDS, Molmo2TokenIds

from .message_weight import (
    MessageWeight,
    apply_message_weight_to_loss_masks,
    loss_token_weighting_for_build,
)
from .qwen3_layout import branch_context_ids, image_prefix_ids
from .sequence_builder import build_branched_sequence

__all__ = ["encode_sft_example", "encode_text_only_sft"]


def encode_sft_example(
    tokenizer,
    pil_image,
    turns: Sequence[Tuple[str, str]],
    *,
    max_crops: int = 8,
    loss_token_weighting: str = "root_subsegments_root_tokens",
    token_ids: Optional[Molmo2TokenIds] = None,
    message_weight: Optional[MessageWeight] = None,
    seed: int = 0,
    shuffle_rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    """Build a branched Molmo2 SFT example from (user, assistant) turn pairs."""
    import torch

    images_t, pooling_t, image_grid = preprocess_image_molmo2(
        pil_image, dtype=torch.float32, device=torch.device("cpu"), max_crops=max_crops
    )
    turn_pairs = [(q, a) for q, a in turns if a]
    if not turn_pairs:
        raise ValueError("No valid (question, answer) branches")

    if len(turn_pairs) > 1:
        order = np.arange(len(turn_pairs))
        rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)
        rng.shuffle(order)
        turn_pairs = [turn_pairs[i] for i in order]

    resolved_token_ids = token_ids or DEFAULT_MOLMO2_TOKEN_IDS
    prefix = image_prefix_ids(tokenizer, image_grid, token_ids=resolved_token_ids)
    multi_branch = len(turn_pairs) > 1
    branches = [
        (
            branch_context_ids(tokenizer, q, branch_index=i, multi_branch=multi_branch),
            tokenizer.encode(a, add_special_tokens=False),
        )
        for i, (q, a) in enumerate(turn_pairs)
    ]

    mw = MessageWeight.from_string(loss_token_weighting).with_overrides(
        message_weight.weight if isinstance(message_weight, MessageWeight) else message_weight
    )
    seq = build_branched_sequence(
        prefix,
        branches,
        eos_id=tokenizer.eos_token_id,
        image_token_ids=resolved_token_ids.image_token_ids,
        loss_token_weighting=loss_token_weighting_for_build(mw),
    )
    subsegment_ids = seq.get("subsegment_ids")
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"],
        subsegment_ids,
        mw,
        branch_scaling_already_applied=True,
    )
    seq["images"] = images_t[0].numpy()
    seq["pooled_patches_idx"] = pooling_t[0].numpy()
    return seq


def encode_text_only_sft(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    loss_token_weighting: str = "root_subsegments_root_tokens",
) -> Dict[str, np.ndarray]:
    """Text-only multi-turn SFT (Tulu4-style)."""
    from olmo_core.nn.vision.molmo2_tokens import N_PATCHES_SQ, PATCH_DIM, POOL_H, POOL_W

    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    input_ids: List[int] = [bos]
    loss_masks: List[float] = [0.0]
    for i, msg in enumerate(messages):
        is_assistant = msg["role"] == "assistant"
        if is_assistant:
            ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        else:
            from .qwen3_layout import user_turn_ids

            ids = user_turn_ids(tokenizer, msg["content"])
        input_ids.extend(ids)
        loss_masks.extend([1.0 if is_assistant else 0.0] * len(ids))

    labels = np.array([-100] * len(input_ids), dtype=np.int64)
    for i, m in enumerate(loss_masks):
        if m > 0:
            labels[i] = input_ids[i]

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "labels": labels,
        "loss_masks": np.array(loss_masks, dtype=np.float32),
        "position_ids": np.arange(len(input_ids), dtype=np.int64),
        "images": np.zeros((0, N_PATCHES_SQ, PATCH_DIM), dtype=np.float32),
        "pooled_patches_idx": np.zeros((0, POOL_H * POOL_W), dtype=np.int64),
    }
