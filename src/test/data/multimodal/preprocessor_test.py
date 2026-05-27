"""Tests for the top-level multimodal preprocessor."""

import numpy as np
import pytest

from olmo_core.data.multimodal.image_preprocessor import ImagePreprocessorConfig
from olmo_core.data.multimodal.multicrop import CropMode, MultiCropPreprocessorConfig
from olmo_core.data.multimodal.preprocessor import (
    MultimodalPreprocessor,
    MultimodalPreprocessorConfig,
)
from olmo_core.data.multimodal.tokens import MultimodalTokenizerConfig

transformers = pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_tokenizer():
    tok_cfg = MultimodalTokenizerConfig.dolma2()
    try:
        return tok_cfg.load_hf_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load dolma2 HF tokenizer: {e}")


def _resize_cfg(seq_len: int = 256) -> MultimodalPreprocessorConfig:
    return MultimodalPreprocessorConfig(
        max_sequence_length=seq_len,
        multicrop=MultiCropPreprocessorConfig(
            base_image_input_size=(28, 28),
            crop_mode=CropMode.resize,
            pool_h=2,
            pool_w=2,
            image_preprocessor=ImagePreprocessorConfig(patch_size=14),
        ),
    )


def _overlap_cfg(seq_len: int = 512) -> MultimodalPreprocessorConfig:
    return MultimodalPreprocessorConfig(
        max_sequence_length=seq_len,
        multicrop=MultiCropPreprocessorConfig(
            base_image_input_size=(28, 28),
            crop_mode=CropMode.overlap_and_resize,
            max_crops=4,
            overlap_margins=(0, 0),
            pool_h=1,
            pool_w=1,
            image_preprocessor=ImagePreprocessorConfig(patch_size=14),
        ),
    )


def _random_image(h: int = 56, w: int = 56):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Shapes / dtypes
# ---------------------------------------------------------------------------


def test_text_with_image_returns_expected_keys(hf_tokenizer):
    cfg = _resize_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp(prompt="What is in this image?", response="A cat.", image=_random_image())
    assert set(out.keys()) == {
        "input_tokens",
        "loss_masks",
        "images",
        "image_masks",
        "pooled_patches_idx",
    }


def test_text_with_image_dtypes(hf_tokenizer):
    pp = MultimodalPreprocessor(_resize_cfg(), hf_tokenizer)
    out = pp("Caption:", "A cat.", _random_image())
    assert out["input_tokens"].dtype == np.int64
    assert out["loss_masks"].dtype == np.float32
    assert out["images"].dtype == np.float32
    assert out["image_masks"].dtype == np.float32
    assert out["pooled_patches_idx"].dtype == np.int64


def test_input_tokens_and_loss_masks_same_length(hf_tokenizer):
    pp = MultimodalPreprocessor(_resize_cfg(), hf_tokenizer)
    out = pp("Question.", "Answer.", _random_image())
    assert out["input_tokens"].shape == out["loss_masks"].shape


def test_image_tensors_match_multicrop(hf_tokenizer):
    """The image tensors in the preprocessor output should match running
    multicrop directly on the same image."""
    cfg = _resize_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    img = _random_image()
    out = pp("Q", "A", img)
    mc_out = pp.multicrop(img)
    np.testing.assert_array_equal(out["images"], mc_out.images)
    np.testing.assert_array_equal(out["image_masks"], mc_out.image_masks)
    np.testing.assert_array_equal(out["pooled_patches_idx"], mc_out.pooled_patches_idx)


# ---------------------------------------------------------------------------
# Loss masks
# ---------------------------------------------------------------------------


def test_loss_mask_only_on_response(hf_tokenizer):
    """Tokens that come from prompt/image text must have loss_mask = 0."""
    cfg = _resize_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp(prompt="prompt text", response="response text", image=_random_image())
    # Find the response section by looking at where loss_mask flips to 1.
    assert (out["loss_masks"] == 1.0).any()
    assert (out["loss_masks"] == 0.0).any()
    # Loss mask should be contiguous: zeros first, then ones.
    flip_points = np.where(np.diff(out["loss_masks"]) != 0)[0]
    assert len(flip_points) == 1, f"loss_masks should flip exactly once, got {len(flip_points)}"


def test_eos_added_to_response(hf_tokenizer):
    """When add_eos=True, the response should end with the base EOS token."""
    cfg = _resize_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp(prompt="Q", response="A", image=_random_image())
    base_eos = cfg.tokenizer.base.eos_token_id
    assert out["input_tokens"][-1] == base_eos
    assert out["loss_masks"][-1] == 1.0


def test_no_eos_when_disabled(hf_tokenizer):
    cfg = _resize_cfg()
    cfg.add_eos = False
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp(prompt="Q", response="A", image=_random_image())
    # The response template prepends a space, so encode " A" to get the
    # expected trailing tokens.
    expected_resp = hf_tokenizer.encode(
        cfg.response_template.format(response="A"), add_special_tokens=False
    )
    assert out["input_tokens"][-len(expected_resp) :].tolist() == expected_resp
    # And the base EOS should not be the final token.
    assert out["input_tokens"][-1] != cfg.tokenizer.base.eos_token_id


# ---------------------------------------------------------------------------
# Image-patch token count contract (load-bearing for the model splice)
# ---------------------------------------------------------------------------


def test_n_image_patch_tokens_equals_n_pooled(hf_tokenizer):
    """The number of <im_patch> tokens in input_tokens must equal
    pooled_patches_idx.shape[0] — model splice contract."""
    cfg = _resize_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp("Q", "A", _random_image())
    tok = cfg.tokenizer
    n_image_tokens = int((out["input_tokens"] == tok.image_patch_id).sum())
    assert n_image_tokens == out["pooled_patches_idx"].shape[0]


def test_n_image_patch_tokens_equals_n_pooled_overlap(hf_tokenizer):
    """Same contract holds in overlap-and-resize mode (with Molmo2 defaults
    the global view also uses <im_patch>)."""
    cfg = _overlap_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp("Describe this.", "It is wide.", _random_image(28, 84))
    tok = cfg.tokenizer
    n_image_tokens = int((out["input_tokens"] == tok.image_patch_id).sum())
    assert n_image_tokens == out["pooled_patches_idx"].shape[0]


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_caps_to_max_sequence_length(hf_tokenizer):
    cfg = _resize_cfg(seq_len=16)  # very short
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp(prompt="Q", response="A" * 200, image=_random_image())
    assert out["input_tokens"].shape[0] <= 16
    assert out["loss_masks"].shape[0] == out["input_tokens"].shape[0]


# ---------------------------------------------------------------------------
# Text-only path
# ---------------------------------------------------------------------------


def test_text_only_example(hf_tokenizer):
    cfg = _resize_cfg()
    pp = MultimodalPreprocessor(cfg, hf_tokenizer)
    out = pp(prompt="Just text.", response="More text.", image=None)
    assert out["images"].shape[0] == 0
    assert out["image_masks"].shape[0] == 0
    assert out["pooled_patches_idx"].shape[0] == 0
    # input_tokens must still be non-empty.
    assert out["input_tokens"].shape[0] > 0
    # Loss must still cover the response.
    assert (out["loss_masks"] == 1.0).any()
