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
