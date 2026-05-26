import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Olmo2Config

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


def test_convert_state_from_hf_ties_word_embeddings():
    hf_config = AutoConfig.for_model(
        "qwen3",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )

    # A tied HF checkpoint omits `lm_head.weight`.
    hf_state = {
        "model.embed_tokens.weight": torch.randn(hf_config.vocab_size, hf_config.hidden_size)
    }

    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="qwen3")

    assert "lm_head.w_out.weight" in converted_state
    torch.testing.assert_close(
        converted_state["lm_head.w_out.weight"], converted_state["embeddings.weight"]
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


def _roundtrip_norm_keys(model_id, model_type, norm_suffixes, forbidden=()):
    hf_config = AutoConfig.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_state = {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}
    del hf_model

    oc_state = convert_state_from_hf(hf_config, hf_state, model_type=model_type)
    hf_roundtrip = convert_state_to_hf(hf_config, oc_state)

    for i in range(hf_config.num_hidden_layers):
        for suffix in norm_suffixes:
            key = f"model.layers.{i}.{suffix}"
            assert key in hf_roundtrip, f"missing {key} in round-tripped state"
            torch.testing.assert_close(hf_roundtrip[key], hf_state[key])
        for suffix in forbidden:
            assert (
                f"model.layers.{i}.{suffix}" not in hf_roundtrip
            ), f"unexpected key model.layers.{i}.{suffix}"


def test_qwen3_0_6b_roundtrip_pre_norm():
    _roundtrip_norm_keys(
        "Qwen/Qwen3-0.6B",
        model_type="qwen3",
        norm_suffixes=("input_layernorm.weight", "post_attention_layernorm.weight"),
        forbidden=("post_feedforward_layernorm.weight",),
    )


def test_llama_tiny_roundtrip_pre_norm():
    hf_config = AutoConfig.for_model(
        "llama",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    n, h = hf_config.num_hidden_layers, hf_config.hidden_size
    olmo_core_state = {}
    for i in range(n):
        olmo_core_state[f"blocks.{i}.attention_norm.weight"] = torch.full((h,), 1.0 + i)
        olmo_core_state[f"blocks.{i}.feed_forward_norm.weight"] = torch.full((h,), 100.0 + i)

    hf_roundtrip = convert_state_to_hf(hf_config, olmo_core_state)
    for i in range(n):
        torch.testing.assert_close(
            hf_roundtrip[f"model.layers.{i}.input_layernorm.weight"],
            olmo_core_state[f"blocks.{i}.attention_norm.weight"],
        )
        torch.testing.assert_close(
            hf_roundtrip[f"model.layers.{i}.post_attention_layernorm.weight"],
            olmo_core_state[f"blocks.{i}.feed_forward_norm.weight"],
        )
        assert f"model.layers.{i}.post_feedforward_layernorm.weight" not in hf_roundtrip


def test_gemma3_270m_roundtrip_pre_norm():
    _roundtrip_norm_keys(
        "google/gemma-3-270m",
        model_type="gemma3_text",
        norm_suffixes=(
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ),
    )


def _assert_logprobs_match_after_roundtrip(model_id: str, model_type: str):
    hf_config = AutoConfig.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    input_ids = torch.randint(
        0, hf_config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        ref_logits = hf_model(input_ids).logits
    ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

    hf_state = {k: v.detach().cpu().clone() for k, v in hf_model.state_dict().items()}
    oc_state = convert_state_from_hf(hf_config, hf_state, model_type=model_type)
    hf_roundtrip = convert_state_to_hf(hf_config, oc_state)

    hf_model.load_state_dict(hf_roundtrip, strict=True)
    hf_model.eval()
    with torch.no_grad():
        rt_logits = hf_model(input_ids).logits
    rt_logprobs = torch.log_softmax(rt_logits, dim=-1)

    torch.testing.assert_close(rt_logprobs, ref_logprobs, rtol=1e-5, atol=1e-5)


def test_qwen3_0_6b_logprobs_roundtrip():
    _assert_logprobs_match_after_roundtrip("Qwen/Qwen3-0.6B", model_type="qwen3")


def test_gemma3_270m_logprobs_roundtrip():
    _assert_logprobs_match_after_roundtrip("google/gemma-3-270m", model_type="gemma3_text")
