"""Independent projection bias preserves the default and parameter accounting."""

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("qkv_bias", [None, False, True])
def test_qkv_bias_and_parameter_count(bias, qkv_bias):
    config = AttentionConfig(n_heads=4, n_kv_heads=2, bias=bias, qkv_bias=qkv_bias)
    attention = config.build(layer_idx=0, n_layers=1, d_model=32, init_device="meta")
    expected_qkv = bias if qkv_bias is None else qkv_bias
    for projection in (attention.w_q, attention.w_k, attention.w_v):
        assert (projection.bias is not None) == expected_qkv
    assert (attention.w_out.bias is not None) == bias
    assert config.num_params(32) == sum(p.numel() for p in attention.parameters())
    assert AttentionConfig.from_dict(config.as_dict()).qkv_bias == qkv_bias


@pytest.mark.parametrize("name", [AttentionType.fused_v2, AttentionType.normalized])
@pytest.mark.parametrize("qkv_bias", [False, True])
def test_nondefault_attention_rejects_explicit_qkv_bias(name, qkv_bias):
    with pytest.raises(OLMoConfigurationError, match="qkv_bias"):
        AttentionConfig(name=name, n_heads=4, qkv_bias=qkv_bias).build(
            layer_idx=0, n_layers=1, d_model=32
        )
