import json

import pytest
from cached_path import cached_path

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)


def _block(**norm_kwargs) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        name=TransformerBlockType.default,
        sequence_mixer=AttentionConfig(name=AttentionType.default, n_heads=8),
        feed_forward=FeedForwardConfig(hidden_size=512),
        **norm_kwargs,
    )


def test_block_config_layer_norm_back_compat():
    """Legacy ``layer_norm=`` is mirrored into both split norm fields and not stored itself."""
    ln = LayerNormConfig()
    block = _block(layer_norm=ln)
    assert block.attention_norm is ln
    assert block.feed_forward_norm is ln

    # ``layer_norm`` is an InitVar, so it must not survive serialization.
    serialized = block.as_config_dict()
    assert "layer_norm" not in serialized
    assert "attention_norm" in serialized and "feed_forward_norm" in serialized

    # Round-trip should be lossless and still resolve to the split fields.
    roundtripped = TransformerBlockConfig.from_dict(serialized)
    assert roundtripped.attention_norm is not None
    assert roundtripped.feed_forward_norm is not None


def test_block_config_split_norms_independent():
    """The two norm fields can be configured independently."""
    attn_norm = LayerNormConfig(bias=False)
    ff_norm = LayerNormConfig(bias=True)
    block = _block(attention_norm=attn_norm, feed_forward_norm=ff_norm)
    assert block.attention_norm is attn_norm
    assert block.feed_forward_norm is ff_norm


def test_block_config_layer_norm_conflict_raises():
    """Specifying both the legacy and split fields is an error."""
    with pytest.raises(OLMoConfigurationError):
        _block(layer_norm=LayerNormConfig(), attention_norm=LayerNormConfig())


def test_llama_factory_sets_split_norms():
    """Factory methods populate the split norm fields (not the legacy alias)."""
    config = TransformerConfig.llama2_271M(vocab_size=1024, dtype=DType.float32)
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.attention_norm is not None
    assert config.block.feed_forward_norm is not None
    assert "layer_norm" not in config.block.as_config_dict()


# OLMO3_7B_CHECKPOINT = "https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage1/step0"
OLMO3_7B_CHECKPOINT = "https://storage.googleapis.com/ai2-llm/checkpoints/OLMo25/step0"


def test_load_olmo3_7b_config():
    """Verify that old checkpoint configs with a single block (not a dict) still load correctly."""
    config_path = cached_path(f"{OLMO3_7B_CHECKPOINT}/config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = TransformerConfig.from_dict(config_dict["model"])

    assert config.d_model == 4096
    assert config.n_layers == 32
    assert config.vocab_size == 100352
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.name == "reordered_norm"

    # Round-trip through as_config_dict / from_dict should be lossless.
    roundtripped = TransformerConfig.from_dict(config.as_config_dict())
    assert roundtripped.as_config_dict() == config.as_config_dict()
