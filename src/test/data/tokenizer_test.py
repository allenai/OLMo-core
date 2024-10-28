from olmo_core.data import TokenizerConfig


def test_padded_vocab_size():
    assert TokenizerConfig.dolma2().padded_vocab_size() == 100352
    assert TokenizerConfig.gpt_neox_olmo_dolma_v1_5().padded_vocab_size() == 50304
