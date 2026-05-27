"""Tests for the multimodal image preprocessor."""

import numpy as np
import pytest

from olmo_core.data.multimodal.image_preprocessor import (
    ImagePreprocessorConfig,
    NormalizeStyle,
)


def _random_image(h: int = 56, w: int = 56) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _gradient_image(h: int = 28, w: int = 28) -> np.ndarray:
    """An RGB image with distinct per-channel gradients so flatten order is testable."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = np.linspace(0, 1, h)[:, None]  # R varies with row
    img[..., 1] = np.linspace(0, 1, w)[None, :]  # G varies with col
    img[..., 2] = 0.5  # B constant
    return img


# ---------------------------------------------------------------------------
# Resize + pad
# ---------------------------------------------------------------------------


class TestResizeAndPad:
    def setup_method(self):
        self.prep = ImagePreprocessorConfig(patch_size=14).build()

    def test_square_image_no_padding(self):
        img = _random_image(56, 56)
        out, mask = self.prep.resize_and_pad(img, (28, 28))
        assert out.shape == (28, 28, 3)
        assert mask.shape == (28, 28)
        assert mask.all()  # no padding for matching aspect ratio

    def test_wide_image_pads_vertically(self):
        img = _random_image(28, 56)  # 1:2 aspect ratio
        out, mask = self.prep.resize_and_pad(img, (56, 56))
        assert out.shape == (56, 56, 3)
        # The image fits the width; height has padding above and below.
        assert mask[0, :].sum() == 0  # top row is all pad
        assert mask[-1, :].sum() == 0  # bottom row is all pad
        assert mask.sum() > 0  # but something is unmasked

    def test_tall_image_pads_horizontally(self):
        img = _random_image(56, 28)  # 2:1 aspect
        out, mask = self.prep.resize_and_pad(img, (56, 56))
        assert out.shape == (56, 56, 3)
        assert mask[:, 0].sum() == 0  # left col padded
        assert mask[:, -1].sum() == 0  # right col padded

    def test_pad_value_used(self):
        cfg = ImagePreprocessorConfig(patch_size=14, pad_value=0.5)
        prep = cfg.build()
        img = _random_image(28, 56)
        out, mask = prep.resize_and_pad(img, (56, 56))
        # All pixels where mask is 0 should equal pad_value.
        pad_pixels = out[mask == 0]
        assert np.allclose(pad_pixels, 0.5)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


def test_normalize_siglip_maps_to_unit_range():
    prep = ImagePreprocessorConfig(normalize=NormalizeStyle.siglip).build()
    img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = prep.normalize(img)
    np.testing.assert_allclose(out, np.array([[[-1.0, 0.0, 1.0]]]))


def test_normalize_openai_zero_mean_unit_std():
    prep = ImagePreprocessorConfig(normalize=NormalizeStyle.openai).build()
    # An image with each channel equal to its CLIP mean should normalize to ~0.
    from olmo_core.data.multimodal.image_preprocessor import OPENAI_CLIP_MEAN

    img = np.tile(np.asarray(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :], (4, 4, 1))
    out = prep.normalize(img)
    assert np.allclose(out, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Patchify (C-first flatten)
# ---------------------------------------------------------------------------


def test_patchify_shape():
    prep = ImagePreprocessorConfig(patch_size=14).build()
    img = np.zeros((28, 28, 3), dtype=np.float32)
    patches = prep.patchify(img)
    # 28/14 = 2 → 4 patches, each 14*14*3 = 588 features.
    assert patches.shape == (4, 588)


def test_patchify_c_first_order():
    """Verify that flat index i = c*p*p + kh*p + kw."""
    prep = ImagePreprocessorConfig(patch_size=2).build()
    img = _gradient_image(2, 2)  # single patch
    patches = prep.patchify(img)
    assert patches.shape == (1, 12)

    p = 2
    # Construct expected flat values via the documented index formula.
    expected = np.zeros(12, dtype=np.float32)
    for c in range(3):
        for kh in range(p):
            for kw in range(p):
                expected[c * p * p + kh * p + kw] = img[kh, kw, c]
    np.testing.assert_allclose(patches[0], expected)


def test_patchify_mask_partial_padding():
    prep = ImagePreprocessorConfig(patch_size=14).build()
    # Mask with half the pixels valid → patches should have ~0.5 weight.
    mask = np.zeros((14, 28), dtype=np.float32)
    mask[:, :14] = 1.0  # left patch fully valid; right patch fully padded
    patch_mask = prep.patchify_mask(mask)
    assert patch_mask.shape == (2,)
    assert patch_mask[0] == 1.0
    assert patch_mask[1] == 0.0


# ---------------------------------------------------------------------------
# End-to-end preprocess
# ---------------------------------------------------------------------------


def test_preprocess_returns_consistent_shapes():
    prep = ImagePreprocessorConfig(patch_size=14, normalize=NormalizeStyle.siglip).build()
    img = _random_image(56, 56)
    patches, mask = prep.preprocess(img, target_size=(28, 28))
    assert patches.shape == (4, 588)
    assert mask.shape == (4,)
    assert mask.dtype == np.float32


def test_preprocess_normalized_range_within_siglip():
    """After siglip normalize, values must lie in [-1, 1]."""
    prep = ImagePreprocessorConfig(patch_size=14, normalize=NormalizeStyle.siglip).build()
    img = _random_image(56, 56)
    patches, _ = prep.preprocess(img, target_size=(28, 28))
    assert patches.min() >= -1.0 - 1e-5
    assert patches.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# PIL compatibility
# ---------------------------------------------------------------------------


def test_accepts_pil_image():
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.fromarray(_random_image(56, 56))
    prep = ImagePreprocessorConfig(patch_size=14).build()
    patches, mask = prep.preprocess(img, target_size=(28, 28))
    assert patches.shape == (4, 588)
    assert mask.shape == (4,)
