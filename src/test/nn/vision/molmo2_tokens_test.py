import pytest

from olmo_core.nn.vision.molmo2_tokens import (
    IMAGE_SPECIAL_TOKENS,
    Molmo2TokenIds,
    build_image_token_ids,
    prepare_molmo2_tokenizer,
)


class _AppendOnlyTokenizer:
    def __init__(self, next_id: int = 100278):
        self.next_id = next_id
        self.vocab = {"<|im_end|>": 100265}

    def add_tokens(self, tokens, special_tokens=False):
        assert special_tokens
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
                added += 1
        return added

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


def test_prepare_molmo2_tokenizer_uses_padded_s002_rows():
    tokenizer = _AppendOnlyTokenizer()
    ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)

    assert [tokenizer.vocab[token] for token in IMAGE_SPECIAL_TOKENS] == list(range(100278, 100284))
    assert ids == Molmo2TokenIds(
        im_start_id=100278,
        im_end_id=100279,
        im_patch_id=100280,
        im_col_id=100281,
        low_res_im_start_id=100282,
        image_placeholder_id=100283,
        im_end_turn_id=100265,
    )


def test_prepare_molmo2_tokenizer_rejects_embedding_resize():
    with pytest.raises(ValueError, match="model vocabulary"):
        prepare_molmo2_tokenizer(_AppendOnlyTokenizer(), model_vocab_size=100283)


def test_build_image_tokens_honors_custom_ids():
    ids = Molmo2TokenIds(
        im_start_id=10,
        im_end_id=11,
        im_patch_id=12,
        im_col_id=13,
        low_res_im_start_id=14,
        image_placeholder_id=15,
        im_end_turn_id=16,
    )
    result = build_image_token_ids(1, 2, 2, 1, token_ids=ids)
    assert result == [14, 12, 12, 11, 10, 12, 13, 12, 13, 11]
    assert ids.image_token_ids == frozenset({10, 11, 12, 13, 14})
