import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType


def test_legacy_d_attn_sets_head_dim_and_does_not_reach_module():
    config = AttentionConfig(
        name=AttentionType.fused_v2,
        n_heads=32,
        n_kv_heads=8,
        d_attn=4096,
        bias=False,
    )

    assert config.head_dim == 128
    attention = config.build(d_model=2048, layer_idx=0, n_layers=1)
    assert attention.head_dim == 128
    assert attention.w_qkv.out_features == 6144


@pytest.mark.parametrize("attention_type", [AttentionType.fused_v2, AttentionType.normalized])
def test_disabled_scalable_softmax_does_not_reach_unsupported_modules(attention_type):
    config = AttentionConfig(name=attention_type, n_heads=8, scalable_softmax=False)

    config.build(d_model=64, layer_idx=0, n_layers=1)


@pytest.mark.parametrize(
    "attention_type", [AttentionType.fused, AttentionType.fused_v2, AttentionType.normalized]
)
def test_enabled_scalable_softmax_rejects_unsupported_modules(attention_type):
    config = AttentionConfig(name=attention_type, n_heads=8, scalable_softmax=True)

    with pytest.raises(
        OLMoConfigurationError, match="scalable_softmax is only supported by default attention"
    ):
        config.build(d_model=64, layer_idx=0, n_layers=1)
