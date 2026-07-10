from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.v2.qwen import build_qwen3_moe_config_from_hf_config


def test_hf_qwen_config_matches_checkpoint_router_precision() -> None:
    config = build_qwen3_moe_config_from_hf_config(
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "rope_theta": 1_000_000,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "rms_norm_eps": 1e-6,
        },
        dtype=DType.bfloat16,
        attention_backend=AttentionBackendName.torch,
    )

    assert not isinstance(config.block, dict)
    assert config.block.routed_experts_router is not None
    assert config.block.routed_experts_router.router_logits_in_fp32 is False
