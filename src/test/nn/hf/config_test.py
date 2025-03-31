import pytest
from transformers import Olmo2Config

from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.transformer.config import TransformerConfig


def test_get_hf_config():
    vocab_size = 4096
    model_config = TransformerConfig.olmo2_190M(vocab_size)
    model = model_config.build()

    hf_config = get_hf_config(model)
    assert isinstance(hf_config, Olmo2Config)
    assert hf_config.hidden_size == model_config.d_model
    assert hf_config.intermediate_size == 3072
    assert hf_config.num_hidden_layers == model_config.n_layers


def test_get_hf_config_default_block():
    vocab_size = 4096
    model_config = TransformerConfig.llama2_271M(vocab_size)
    model = model_config.build()

    with pytest.raises(NotImplementedError):
        get_hf_config(model)


def test_get_hf_config_moe():
    vocab_size = 4096
    model_config = TransformerConfig.smallmoe(vocab_size)
    model = model_config.build()

    with pytest.raises(NotImplementedError):
        get_hf_config(model)
