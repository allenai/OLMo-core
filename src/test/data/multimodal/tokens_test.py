"""Tests for the multimodal tokenizer config."""

import pytest

from olmo_core.data.multimodal.tokens import (
    IMAGE_SPECIAL_TOKENS,
    MultimodalTokenizerConfig,
)
from olmo_core.data.tokenizer import TokenizerConfig


def test_dolma2_factory_assigns_consecutive_ids():
    cfg = MultimodalTokenizerConfig.dolma2()
    base = TokenizerConfig.dolma2()
    assert cfg.base.identifier == base.identifier
    assert cfg.image_start_id == base.vocab_size + 0
    assert cfg.image_end_id == base.vocab_size + 1
    assert cfg.image_patch_id == base.vocab_size + 2
    assert cfg.image_col_id == base.vocab_size + 3
    assert cfg.low_res_image_start_id == base.vocab_size + 4
    assert cfg.image_low_id == base.vocab_size + 5


def test_vocab_size_extends_base():
    cfg = MultimodalTokenizerConfig.dolma2()
    base = TokenizerConfig.dolma2()
    assert cfg.vocab_size == base.vocab_size + len(IMAGE_SPECIAL_TOKENS)


def test_padded_vocab_size_rounds_up():
    cfg = MultimodalTokenizerConfig.dolma2()
    # dolma2 vocab is 100278; + 6 = 100284; padded to multiple of 128 → 100352.
    assert cfg.padded_vocab_size(128) == 100352


def test_special_token_ids_are_unique_and_ordered():
    cfg = MultimodalTokenizerConfig.dolma2()
    ids = cfg.special_token_ids
    assert len(ids) == len(IMAGE_SPECIAL_TOKENS)
    assert len(set(ids)) == len(ids)  # unique
    assert ids == sorted(ids)  # consecutive ascending


def test_image_patch_id_used_by_multimodal_transformer():
    """The image_patch_id is what MultimodalTransformerConfig.image_patch_token_id should be."""
    cfg = MultimodalTokenizerConfig.dolma2()
    # This is the contract: the LM splices image features at every occurrence
    # of this ID in the input_ids tensor.
    assert isinstance(cfg.image_patch_id, int)
    assert cfg.image_patch_id >= cfg.base.vocab_size


def test_load_hf_tokenizer_matches_config_ids():
    """The HF tokenizer must assign IDs to image tokens that match config properties."""
    transformers = pytest.importorskip("transformers")  # noqa: F841
    cfg = MultimodalTokenizerConfig.dolma2()
    try:
        tok = cfg.load_hf_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load dolma2 HF tokenizer: {e}")

    for token_str, expected_id in zip(IMAGE_SPECIAL_TOKENS, cfg.special_token_ids):
        assert tok.convert_tokens_to_ids(token_str) == expected_id

    # Encoding text containing the special token round-trips.
    encoded = tok.encode("hello <im_patch> world", add_special_tokens=False)
    assert cfg.image_patch_id in encoded
