"""Tests for the multimodal collator."""

import numpy as np
import torch

from olmo_core.data.multimodal.collator import (
    MultimodalCollator,
    MultimodalCollatorConfig,
)
from olmo_core.data.multimodal.tokens import MultimodalTokenizerConfig


def _example(
    seq_len: int,
    n_crops: int,
    n_patches_per_crop: int,
    patch_dim: int,
    n_pooled: int,
    pool_size: int,
    rng: np.random.Generator,
) -> dict:
    return {
        "input_tokens": rng.integers(0, 100, size=seq_len, dtype=np.int64),
        "loss_masks": rng.random(seq_len, dtype=np.float32),
        "images": rng.random((n_crops, n_patches_per_crop, patch_dim), dtype=np.float32),
        "image_masks": np.ones((n_crops, n_patches_per_crop), dtype=np.float32),
        "pooled_patches_idx": rng.integers(
            0, n_crops * n_patches_per_crop, size=(n_pooled, pool_size), dtype=np.int64
        ),
    }


def _collator() -> MultimodalCollator:
    tok = MultimodalTokenizerConfig.dolma2()
    return MultimodalCollatorConfig(tokenizer=tok).build()


# ---------------------------------------------------------------------------
# Uniform batches
# ---------------------------------------------------------------------------


def test_uniform_batch_basic_shapes():
    rng = np.random.default_rng(0)
    coll = _collator()
    batch = [_example(16, 1, 4, 12, 1, 4, rng) for _ in range(3)]
    out = coll(batch)
    assert out["input_ids"].shape == (3, 16)
    assert out["loss_masks"].shape == (3, 16)
    assert out["images"].shape == (3, 1, 4, 12)
    assert out["image_masks"].shape == (3, 1, 4)
    assert out["pooled_patches_idx"].shape == (3, 1, 4)


def test_dtypes_match_model_contract():
    rng = np.random.default_rng(0)
    coll = _collator()
    batch = [_example(8, 1, 4, 12, 1, 4, rng)]
    out = coll(batch)
    assert out["input_ids"].dtype == torch.long
    assert out["loss_masks"].dtype == torch.float32
    assert out["images"].dtype == torch.float32
    assert out["image_masks"].dtype == torch.float32
    assert out["pooled_patches_idx"].dtype == torch.long


# ---------------------------------------------------------------------------
# Variable sequence length — text padding
# ---------------------------------------------------------------------------


def test_variable_seq_len_pads_to_max():
    rng = np.random.default_rng(0)
    coll = _collator()
    batch = [
        _example(
            seq_len=8,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=1,
            pool_size=4,
            rng=rng,
        ),
        _example(
            seq_len=12,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=1,
            pool_size=4,
            rng=rng,
        ),
    ]
    out = coll(batch)
    assert out["input_ids"].shape == (2, 12)
    # First example's trailing positions should be pad_id.
    tok = MultimodalTokenizerConfig.dolma2()
    assert (out["input_ids"][0, 8:] == tok.base.pad_token_id).all()
    # Loss mask is zero at padded positions.
    assert (out["loss_masks"][0, 8:] == 0.0).all()


def test_pad_to_multiple_of():
    rng = np.random.default_rng(0)
    tok = MultimodalTokenizerConfig.dolma2()
    coll = MultimodalCollatorConfig(tokenizer=tok, pad_to_multiple_of=8).build()
    batch = [
        _example(
            seq_len=5,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=1,
            pool_size=4,
            rng=rng,
        )
    ]
    out = coll(batch)
    assert out["input_ids"].shape[1] % 8 == 0
    assert out["input_ids"].shape[1] >= 5


# ---------------------------------------------------------------------------
# Variable image layout — crop padding + dummy patch tokens
# ---------------------------------------------------------------------------


def test_variable_n_crops_pads_to_max():
    rng = np.random.default_rng(0)
    coll = _collator()
    batch = [
        _example(
            seq_len=16,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=1,
            pool_size=4,
            rng=rng,
        ),
        _example(
            seq_len=16,
            n_crops=3,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=1,
            pool_size=4,
            rng=rng,
        ),
    ]
    out = coll(batch)
    assert out["images"].shape == (2, 3, 4, 12)
    # First example's last two crops are zero-padded.
    assert (out["images"][0, 1:].sum() == 0).item()
    assert (out["image_masks"][0, 1:].sum() == 0).item()


def test_variable_n_pooled_pads_and_adds_dummy_patches():
    """Variable n_pooled: pad pooled_patches_idx with -1 rows AND append
    matching dummy <im_patch> tokens to input_ids so the model contract
    (count of <im_patch> tokens == B * max_n_pooled) holds."""
    rng = np.random.default_rng(0)
    tok = MultimodalTokenizerConfig.dolma2()
    coll = _collator()
    batch = [
        _example(
            seq_len=8,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=1,
            pool_size=4,
            rng=rng,
        ),
        _example(
            seq_len=8,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=3,
            pool_size=4,
            rng=rng,
        ),
    ]
    # Force input_tokens to contain the right number of real <im_patch> tokens.
    batch[0]["input_tokens"] = np.array([tok.image_patch_id, 5, 6, 7, 0, 0, 0, 0], dtype=np.int64)
    batch[1]["input_tokens"] = np.array([tok.image_patch_id] * 3 + [5, 6, 7, 0, 0], dtype=np.int64)
    out = coll(batch)
    # pooled_patches_idx padded to max=3.
    assert out["pooled_patches_idx"].shape == (2, 3, 4)
    # Padding rows are all -1.
    assert (out["pooled_patches_idx"][0, 1:] == -1).all()
    # Each example's <im_patch> count should equal max_n_pooled=3.
    for i in range(2):
        n_image_patch = (out["input_ids"][i] == tok.image_patch_id).sum().item()
        assert n_image_patch == 3, f"example {i} has {n_image_patch} <im_patch> tokens, expected 3"
    # Loss mask is zero at dummy positions for the example that received them.
    # Example 0: original n_pooled=1; needs +2 dummy. The 2 extra <im_patch>
    # tokens land at positions [seq_len, seq_len+1] = [8, 9] (no original loss
    # information there because seq_len=8). Loss mask at those positions is 0.
    assert out["loss_masks"][0, 8] == 0.0
    assert out["loss_masks"][0, 9] == 0.0


def test_model_contract_total_image_patches_matches_pooled_size():
    """Most important: total <im_patch> across batch == B * max_n_pooled."""
    rng = np.random.default_rng(0)
    tok = MultimodalTokenizerConfig.dolma2()
    coll = _collator()
    batch = [
        _example(
            seq_len=8,
            n_crops=1,
            n_patches_per_crop=4,
            patch_dim=12,
            n_pooled=k,
            pool_size=4,
            rng=rng,
        )
        for k in (1, 2, 3)
    ]
    # Each example needs exactly k real <im_patch> tokens.
    for i, k in enumerate((1, 2, 3)):
        toks = np.full(8, 99, dtype=np.int64)
        toks[:k] = tok.image_patch_id
        batch[i]["input_tokens"] = toks
    out = coll(batch)
    total_image_patch = (out["input_ids"] == tok.image_patch_id).sum().item()
    B, max_n_pooled = out["pooled_patches_idx"].shape[:2]
    assert total_image_patch == B * max_n_pooled


# ---------------------------------------------------------------------------
# Empty / text-only batch
# ---------------------------------------------------------------------------


def test_text_only_batch():
    """A batch with no images at all should still produce valid tensors."""
    rng = np.random.default_rng(0)
    coll = _collator()
    batch = [
        {
            "input_tokens": rng.integers(0, 100, size=8, dtype=np.int64),
            "loss_masks": np.ones(8, dtype=np.float32),
            "images": np.zeros((0, 4, 12), dtype=np.float32),
            "image_masks": np.zeros((0, 4), dtype=np.float32),
            "pooled_patches_idx": np.zeros((0, 4), dtype=np.int64),
        }
        for _ in range(2)
    ]
    out = coll(batch)
    assert out["input_ids"].shape == (2, 8)
    assert out["images"].shape == (2, 0, 4, 12)
    assert out["pooled_patches_idx"].shape == (2, 0, 4)
