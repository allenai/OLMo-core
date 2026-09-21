"""NoPE remains explicit across config construction and HF serialization."""

import json

import pytest

from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig


@pytest.mark.parametrize("parameters", [None, {"rope_theta": None}, {"rope_theta": 10000.0}])
def test_nope_roundtrip(tmp_path, parameters):
    """Explicit use_rope=False wins over absent, legacy, or contradictory metadata."""
    config = Olmo3MoeConfig(use_rope=False, rope_parameters=parameters)
    for _ in range(2):
        config.save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "config.json").read_text())
        assert saved["use_rope"] is False
        assert saved["rope_theta"] is None
        # Some Transformers versions add rope_type='default'; an explicit null
        # theta must survive, rather than being replaced by default theta=10000.
        assert isinstance(saved["rope_parameters"], dict)
        assert saved["rope_parameters"]["rope_theta"] is None
        config = Olmo3MoeConfig.from_pretrained(tmp_path)
        assert config.rope_parameters["rope_theta"] is None
        assert config.use_rope is False


def test_rope_enabled_unchanged(tmp_path):
    """The NoPE normalization must not disable genuinely rotary models."""
    config = Olmo3MoeConfig(use_rope=True, rope_theta=500000.0)
    config.save_pretrained(tmp_path)
    restored = Olmo3MoeConfig.from_pretrained(tmp_path)
    assert restored.use_rope is True
    assert restored.rope_theta == 500000.0
