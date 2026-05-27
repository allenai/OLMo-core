"""Tests for the multi-crop preprocessor."""

import numpy as np
import pytest

from olmo_core.data.multimodal.image_preprocessor import ImagePreprocessorConfig
from olmo_core.data.multimodal.multicrop import (
    CropMode,
    MultiCropPreprocessorConfig,
    arange_for_pooling,
    select_tiling,
)
from olmo_core.data.multimodal.tokens import MultimodalTokenizerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_image(side: int = 56) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)


def _wide_image(h: int = 28, w: int = 84) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _resize_cfg(base: int = 28, patch: int = 14) -> MultiCropPreprocessorConfig:
    return MultiCropPreprocessorConfig(
        base_image_input_size=(base, base),
        crop_mode=CropMode.resize,
        pool_h=2,
        pool_w=2,
        image_preprocessor=ImagePreprocessorConfig(patch_size=patch),
    )


def _overlap_cfg(
    base: int = 28, patch: int = 14, max_crops: int = 4
) -> MultiCropPreprocessorConfig:
    return MultiCropPreprocessorConfig(
        base_image_input_size=(base, base),
        crop_mode=CropMode.overlap_and_resize,
        max_crops=max_crops,
        overlap_margins=(0, 0),  # no overlap to keep tests deterministic
        pool_h=1,
        pool_w=1,  # 1x1 pooling so each patch is its own group → easy to count
        image_preprocessor=ImagePreprocessorConfig(patch_size=patch),
    )


# ---------------------------------------------------------------------------
# select_tiling
# ---------------------------------------------------------------------------


def test_select_tiling_square_image_prefers_square_layout():
    assert select_tiling(224, 224, 224, 4) == (1, 1)


def test_select_tiling_wide_image_picks_wide_layout():
    rows, cols = select_tiling(224, 672, 224, 6)  # 1:3 aspect
    assert cols >= rows


def test_select_tiling_respects_max_crops():
    rows, cols = select_tiling(2240, 2240, 224, 4)  # too big — should still be ≤ 4 crops
    assert rows * cols <= 4


# ---------------------------------------------------------------------------
# arange_for_pooling
# ---------------------------------------------------------------------------


def test_arange_for_pooling_divides_evenly():
    idx = np.arange(16, dtype=np.int64).reshape(4, 4)
    out = arange_for_pooling(idx, 2, 2)
    assert out.shape == (2, 2, 4)
    # First pool group should contain the top-left 2x2 patches: [0, 1, 4, 5].
    assert sorted(out[0, 0].tolist()) == [0, 1, 4, 5]


def test_arange_for_pooling_pads_with_minus_one():
    idx = np.arange(9, dtype=np.int64).reshape(3, 3)
    out = arange_for_pooling(idx, 2, 2)
    # 3x3 padded to 4x4 with one row and one col of -1 → 4 groups of 4.
    assert out.shape == (2, 2, 4)
    # Some pool slots must be -1 since we padded.
    assert (out == -1).any()


# ---------------------------------------------------------------------------
# MultiCropPreprocessor — resize mode
# ---------------------------------------------------------------------------


class TestResizeMode:
    def setup_method(self):
        self.tok = MultimodalTokenizerConfig.dolma2()
        self.cfg = _resize_cfg(base=28, patch=14)
        self.pp = self.cfg.build(self.tok)

    def test_output_shapes(self):
        out = self.pp(_square_image(56))
        # 28x28 image / 14 = 2x2 patches → 4 patches, pool 2x2 → 1 group.
        assert out.images.shape == (1, 4, 14 * 14 * 3)
        assert out.image_masks.shape == (1, 4)
        assert out.pooled_patches_idx.shape == (1, 4)

    def test_image_token_layout(self):
        out = self.pp(_square_image(56))
        # Expected: <im_start> <im_patch> <im_col> <im_end>
        # (only 1 pool row, 1 pool col → 1 patch token + 1 col token)
        assert out.image_tokens[0] == self.tok.image_start_id
        assert out.image_tokens[-1] == self.tok.image_end_id
        # Body contains exactly 1 <im_patch> and 1 <im_col>.
        body = out.image_tokens[1:-1]
        assert (body == self.tok.image_patch_id).sum() == 1
        assert (body == self.tok.image_col_id).sum() == 1

    def test_pooled_idx_covers_all_patches_when_divisible(self):
        out = self.pp(_square_image(56))
        # With a divisible grid and no padding, every patch index 0..3 appears once.
        valid = out.pooled_patches_idx[out.pooled_patches_idx >= 0]
        assert sorted(valid.tolist()) == [0, 1, 2, 3]

    def test_n_patch_tokens_matches_n_pooled(self):
        """The number of <im_patch> tokens should equal n_pooled (model contract)."""
        out = self.pp(_square_image(56))
        n_pooled = out.pooled_patches_idx.shape[0]
        # In resize mode all pool groups use <im_patch> (we ignore col / start / end).
        n_patch_tokens = (out.image_tokens == self.tok.image_patch_id).sum()
        assert n_patch_tokens == n_pooled


def test_resize_mode_larger_grid():
    """A 56x56 base with 14-px patches → 4x4 patch grid → 2x2 pool groups."""
    tok = MultimodalTokenizerConfig.dolma2()
    cfg = MultiCropPreprocessorConfig(
        base_image_input_size=(56, 56),
        crop_mode=CropMode.resize,
        pool_h=2,
        pool_w=2,
        image_preprocessor=ImagePreprocessorConfig(patch_size=14),
    )
    pp = cfg.build(tok)
    out = pp(_square_image(112))
    assert out.images.shape == (1, 16, 14 * 14 * 3)
    assert out.pooled_patches_idx.shape == (4, 4)


# ---------------------------------------------------------------------------
# MultiCropPreprocessor — overlap_and_resize mode
# ---------------------------------------------------------------------------


class TestOverlapMode:
    def setup_method(self):
        self.tok = MultimodalTokenizerConfig.dolma2()
        self.cfg = _overlap_cfg(base=28, patch=14, max_crops=4)
        self.pp = self.cfg.build(self.tok)

    def test_wide_image_produces_multiple_crops(self):
        out = self.pp(_wide_image(28, 84))
        # 1 global + N regional crops, each 28x28 with 2x2=4 patches.
        n_crops = out.images.shape[0]
        assert n_crops >= 2  # at least one regional crop in addition to the global
        assert out.images.shape[1] == 4

    def test_token_layout_has_two_im_start_tokens(self):
        """With Molmo2 defaults both sections open with <im_start>."""
        out = self.pp(_wide_image(28, 84))
        # 2 <im_start> (one per section) + 2 <im_end>.
        assert (out.image_tokens == self.tok.image_start_id).sum() == 2
        assert (out.image_tokens == self.tok.image_end_id).sum() == 2
        # <low_res_im_start> is unused with the default flag.
        assert (out.image_tokens == self.tok.low_res_image_start_id).sum() == 0

    def test_global_uses_im_patch_by_default(self):
        """With Molmo2's default (use_low_res_token_for_global=False) the global
        view also uses <im_patch>, so the model splice can be triggered by a
        single token ID."""
        out = self.pp(_wide_image(28, 84))
        assert (out.image_tokens == self.tok.image_low_id).sum() == 0
        assert (out.image_tokens == self.tok.image_patch_id).sum() > 0

    def test_n_image_patch_tokens_equals_n_pooled(self):
        """Number of <im_patch> tokens must match n_pooled (model splice contract)."""
        out = self.pp(_wide_image(28, 84))
        n_patch_tokens = (out.image_tokens == self.tok.image_patch_id).sum()
        assert n_patch_tokens == out.pooled_patches_idx.shape[0]

    def test_pooled_idx_in_bounds(self):
        out = self.pp(_wide_image(28, 84))
        n_total_patches = out.images.shape[0] * out.images.shape[1]
        idx = out.pooled_patches_idx
        # Valid indices fall within [0, n_total_patches); padding marked -1.
        assert (idx[idx >= 0] < n_total_patches).all()


def test_overlap_mode_square_image_single_crop():
    """A square image that fits the base size should produce one regional crop."""
    tok = MultimodalTokenizerConfig.dolma2()
    cfg = _overlap_cfg(base=28, patch=14, max_crops=4)
    pp = cfg.build(tok)
    out = pp(_square_image(28))
    # 1 global + 1 regional = 2 crops minimum.
    assert out.images.shape[0] == 2


# ---------------------------------------------------------------------------
# Both modes — patch dtype / sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crop_mode", [CropMode.resize, CropMode.overlap_and_resize])
def test_outputs_are_float32(crop_mode):
    tok = MultimodalTokenizerConfig.dolma2()
    if crop_mode == CropMode.resize:
        cfg = _resize_cfg()
    else:
        cfg = _overlap_cfg()
    pp = cfg.build(tok)
    out = pp(_square_image(56))
    assert out.images.dtype == np.float32
    assert out.image_masks.dtype == np.float32
    assert out.pooled_patches_idx.dtype == np.int64
    assert out.image_tokens.dtype == np.int64
