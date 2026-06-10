import pytest
import torch
from transformers import Olmo2Config, Qwen3_5Config

from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None


def _get_olmo2_config() -> Olmo2Config:
    return Olmo2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        eos_token_id=42,
    )


def _get_qwen3_5_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config={
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 6,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 2,
            "layer_types": ["linear_attention"],
        },
    )


def _get_qwen3_5_hf_state() -> dict[str, torch.Tensor]:
    return {
        "model.language_model.embed_tokens.weight": torch.randn(8, 4),
        "model.language_model.norm.weight": torch.randn(4),
        "lm_head.weight": torch.randn(8, 4),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.randn(6, 4),
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": torch.randn(2, 4),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": torch.randn(2, 4),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": torch.randn(2, 4),
        "model.language_model.layers.0.linear_attn.out_proj.weight": torch.randn(4, 2),
        "model.language_model.layers.0.linear_attn.conv1d.weight": torch.randn(6, 1, 4),
        "model.language_model.layers.0.linear_attn.norm.weight": torch.randn(2),
        "model.language_model.layers.0.linear_attn.A_log": torch.randn(2),
        "model.language_model.layers.0.linear_attn.dt_bias": torch.randn(2),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(4),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(6, 4),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(4, 6),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(6, 4),
        "model.visual.patch_embed.proj.weight": torch.randn(4, 4),
    }


def test_convert_state_from_hf():
    hf_config = _get_olmo2_config()

    hf_state = {
        "model.embed_tokens.weight": torch.randn(
            hf_config.max_position_embeddings, hf_config.vocab_size
        ),
        "lm_head.weight": torch.randn(hf_config.hidden_size),
    }

    converted_state = convert_state_from_hf(hf_config, hf_state)

    torch.testing.assert_close(
        converted_state["embeddings.weight"], hf_state["model.embed_tokens.weight"]
    )
    torch.testing.assert_close(converted_state["lm_head.w_out.weight"], hf_state["lm_head.weight"])


def test_convert_layers_from_hf():
    hf_config = _get_olmo2_config()

    hf_state = {}
    for i in range(hf_config.num_hidden_layers):
        hf_state.update(
            {
                f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(
                    hf_config.hidden_size, hf_config.hidden_size
                ),
                f"model.layers.{i}.mlp.gate_proj.weight": torch.randn(
                    hf_config.hidden_size, hf_config.intermediate_size
                ),
                f"model.layers.{i}.post_attention_layernorm.weight": torch.randn(
                    hf_config.hidden_size
                ),
            }
        )

    converted_state = convert_state_from_hf(hf_config, hf_state)

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.attention.w_q.weight"],
            hf_state[f"model.layers.{i}.self_attn.q_proj.weight"],
        )
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.feed_forward.w1.weight"],
            hf_state[f"model.layers.{i}.mlp.gate_proj.weight"],
        )
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.attention_norm.weight"],
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"],
        )


def test_convert_model_type_specific_from_hf():
    hf_config = _get_olmo2_config()

    hf_state = {}
    for i in range(hf_config.num_hidden_layers):
        hf_state.update(
            {
                f"model.layers.{i}.post_attention_layernorm.weight": torch.randn(
                    hf_config.hidden_size
                ),
            }
        )

    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="llama")

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.feed_forward_norm.weight"],
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"],
        )


def test_convert_qwen3_5_top_level_config_from_hf():
    wrapper_config = _get_qwen3_5_config()
    hf_state = _get_qwen3_5_hf_state()

    converted_state = convert_state_from_hf(
        wrapper_config,
        hf_state,
        model_type="qwen3_5",
    )

    torch.testing.assert_close(
        converted_state["blocks.0.attention.w_q.weight"],
        hf_state["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"][:2],
    )
    torch.testing.assert_close(
        converted_state["blocks.0.attention.w_k.weight"],
        hf_state["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"][2:4],
    )
    torch.testing.assert_close(
        converted_state["blocks.0.attention.w_v.weight"],
        hf_state["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"][4:6],
    )
    torch.testing.assert_close(
        converted_state["blocks.0.attention.w_out.weight"],
        hf_state["model.language_model.layers.0.linear_attn.out_proj.weight"],
    )
    assert all("visual" not in key for key in converted_state)


def test_convert_state_from_hf_and_flatten():
    hf_config = _get_olmo2_config()

    hf_state = {}
    for i in range(hf_config.num_hidden_layers):
        hf_state.update(
            {
                f"model.layers.{i}.mlp.gate.weight": torch.randn(5, 6),
            }
        )

    converted_state = convert_state_from_hf(hf_config, hf_state)

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.feed_forward_moe.router.weight"],
            hf_state[f"model.layers.{i}.mlp.gate.weight"].flatten(),
        )


def test_convert_state_to_hf():
    hf_config = _get_olmo2_config()

    olmo_core_state = {
        "embeddings.weight": torch.randn(hf_config.max_position_embeddings, hf_config.vocab_size),
        "lm_head.w_out.weight": torch.randn(hf_config.hidden_size),
    }

    converted_state = convert_state_to_hf(hf_config, olmo_core_state)

    torch.testing.assert_close(
        converted_state["model.embed_tokens.weight"],
        olmo_core_state["embeddings.weight"],
    )
    torch.testing.assert_close(
        converted_state["lm_head.weight"], olmo_core_state["lm_head.w_out.weight"]
    )


def test_convert_layers_to_hf():
    hf_config = _get_olmo2_config()

    olmo_core_state = {}
    for i in range(hf_config.num_hidden_layers):
        olmo_core_state.update(
            {
                f"blocks.{i}.attention.w_q.weight": torch.randn(
                    hf_config.hidden_size, hf_config.hidden_size
                ),
                f"blocks.{i}.feed_forward.w1.weight": torch.randn(
                    hf_config.hidden_size, hf_config.intermediate_size
                ),
                f"blocks.{i}.attention_norm.weight": torch.randn(hf_config.hidden_size),
            }
        )

    converted_state = convert_state_to_hf(hf_config, olmo_core_state)

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"model.layers.{i}.self_attn.q_proj.weight"],
            olmo_core_state[f"blocks.{i}.attention.w_q.weight"],
        )
        torch.testing.assert_close(
            converted_state[f"model.layers.{i}.mlp.gate_proj.weight"],
            olmo_core_state[f"blocks.{i}.feed_forward.w1.weight"],
        )
        torch.testing.assert_close(
            converted_state[f"model.layers.{i}.post_attention_layernorm.weight"],
            olmo_core_state[f"blocks.{i}.attention_norm.weight"],
        )


def test_convert_state_to_hf_and_unflatten():
    hf_config = _get_olmo2_config()
    hf_config.num_experts = hf_config.hidden_size // 2

    olmo_core_state = {}
    for i in range(hf_config.num_hidden_layers):
        olmo_core_state.update(
            {
                f"blocks.{i}.feed_forward_moe.router.weight": torch.randn(
                    hf_config.num_experts * hf_config.hidden_size
                ),
            }
        )

    converted_state = convert_state_to_hf(hf_config, olmo_core_state)

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"model.layers.{i}.mlp.gate.weight"].flatten(),
            olmo_core_state[f"blocks.{i}.feed_forward_moe.router.weight"],
        )


def test_convert_state_to_flex_olmo_hf():
    if FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")

    hf_config = FlexOlmoConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_experts=2,
        max_position_embeddings=64,
        eos_token_id=42,
    )

    olmo_core_state = {}
    for i in range(hf_config.num_hidden_layers):
        olmo_core_state.update(
            {
                f"blocks.{i}.feed_forward_moe.router.weight": torch.randn(
                    hf_config.num_experts * hf_config.hidden_size
                ),
            }
        )

    converted_state = convert_state_to_hf(hf_config, olmo_core_state)

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"model.layers.{i}.mlp.gate.weight"].flatten(),
            olmo_core_state[f"blocks.{i}.feed_forward_moe.router.weight"],
        )
