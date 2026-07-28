import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.moe.v2.qwen_hf_export import load_qwen3_moe_from_olmo_state


@pytest.fixture
def qwen_model_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
    )


def test_load_qwen3_moe_from_olmo_state(qwen_model_config):
    source_model = Qwen3MoeForCausalLM(qwen_model_config)
    source_state = {name: value.clone() for name, value in source_model.state_dict().items()}
    native_state = convert_state_from_hf(
        qwen_model_config,
        source_state,
        model_type=qwen_model_config.model_type,
    )
    checkpoint_state = {f"module.{name}.main": value for name, value in native_state.items()}
    target_model = Qwen3MoeForCausalLM(qwen_model_config)
    for parameter in target_model.parameters():
        parameter.data.zero_()

    load_qwen3_moe_from_olmo_state(target_model, checkpoint_state)

    torch.testing.assert_close(target_model.state_dict(), source_state)


def test_verify_qwen3_moe_from_olmo_state_detects_difference(qwen_model_config):
    model = Qwen3MoeForCausalLM(qwen_model_config)
    native_state = convert_state_from_hf(
        qwen_model_config,
        model.state_dict(),
        model_type=qwen_model_config.model_type,
    )
    checkpoint_state = {
        f"module.{name}.main": value.clone() for name, value in native_state.items()
    }
    checkpoint_state["module.embeddings.weight.main"][0, 0] += 1

    with pytest.raises(ValueError, match="exported tensor differs"):
        load_qwen3_moe_from_olmo_state(model, checkpoint_state, verify_only=True)
