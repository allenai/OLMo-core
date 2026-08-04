"""Encode SFT chat turns + image into Molmo2 training tensors."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.nn.vision.molmo2_image_processor import preprocess_image_molmo2

from .message_weight import (
    MessageWeight,
    apply_message_weight_to_loss_masks,
    loss_token_weighting_for_build,
)
from .qwen3_layout import branch_context_ids
from .sequence_builder import build_branched_sequence

__all__ = ["encode_sft_example", "encode_text_only_sft"]


def encode_sft_example(
    tokenizer,
    pil_image,
    turns: Sequence[Tuple[str, str]],
    *,
    max_crops: int = 8,
    max_images: int = 5,
    p_high_res: float = 0.0,
    high_res_max_crops: int = 24,
    loss_token_weighting: str = "root_subsegments_root_tokens",
    message_weight: Optional[MessageWeight] = None,
    seed: int = 0,
    shuffle_rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    """Build a branched Molmo2 SFT example from (user, assistant) turn pairs.

    ``pil_image`` may be a single image or a **list** of images (multi-image
    example). Multi-image handling follows mm_olmo's ``MultiImagePreprocessor`` +
    ``build_sequence``: at most ``max_images`` images, each preprocessed with the
    same crop budget, ``"Image {i+1}"`` text prefixes when there is more than one
    image, crops concatenated along the crop axis, and each image's pooled patch
    indices offset by the running crop-patch count.
    """
    import torch

    from .qwen3_layout import multi_image_prefix_ids

    rng = shuffle_rng if shuffle_rng is not None else np.random.RandomState(seed)

    pil_images = pil_image if isinstance(pil_image, (list, tuple)) else [pil_image]
    pil_images = list(pil_images)[:max_images]

    grids = []
    crops_list = []
    pooling_list = []
    pooled_offset = 0
    for img in pil_images:
        images_t, pooling_t, image_grid = preprocess_image_molmo2(
            img,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_crops=max_crops,
            p_high_res=p_high_res,
            high_res_max_crops=high_res_max_crops,
            is_training=True,
            rng=rng,
        )
        crops = images_t[0].numpy()  # (n_crops, n_patches, patch_dim)
        pooled = pooling_t[0].numpy()  # (n_pool, pool_size), indices local to this image
        # Offset into the concatenated (total_crops * n_patches) axis
        # (mm_olmo build_sequence: token_pooling_offset += prod(images.shape[:2])).
        pooled = np.where(pooled >= 0, pooled + pooled_offset, pooled)
        pooled_offset += int(np.prod(crops.shape[:2]))
        grids.append(image_grid)
        crops_list.append(crops)
        pooling_list.append(pooled)

    turn_pairs = [(q, a) for q, a in turns if a]
    if not turn_pairs:
        raise ValueError("No valid (question, answer) branches")

    if len(turn_pairs) > 1:
        order = np.arange(len(turn_pairs))
        rng.shuffle(order)
        turn_pairs = [turn_pairs[i] for i in order]

    prefix = multi_image_prefix_ids(tokenizer, grids)
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
        loss_token_weighting=loss_token_weighting_for_build(mw),
    )
    subsegment_ids = seq.get("subsegment_ids")
    seq["loss_masks"] = apply_message_weight_to_loss_masks(
        seq["loss_masks"],
        subsegment_ids,
        mw,
        branch_scaling_already_applied=True,
    )
    seq["images"] = (
        np.concatenate(crops_list, axis=0) if len(crops_list) > 1 else crops_list[0]
    )
    seq["pooled_patches_idx"] = (
        np.concatenate(pooling_list, axis=0) if len(pooling_list) > 1 else pooling_list[0]
    )
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
