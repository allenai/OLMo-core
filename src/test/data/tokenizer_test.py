from olmo_core.data import TokenizerConfig


def test_padded_vocab_size():
    assert TokenizerConfig.dolma2().padded_vocab_size() == 100352
    assert TokenizerConfig.gpt_neox_olmo_dolma_v1_5().padded_vocab_size() == 50304


def test_hf_tokenizer():
    tokenizer = TokenizerConfig.dolma2().build_hf()
    token_ids = tokenizer.encode("Hello, World!", add_special_tokens=True)
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == "Hello, World!"
