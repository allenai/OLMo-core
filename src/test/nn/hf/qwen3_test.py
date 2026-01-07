import os

import pytest
import torch
import transformers

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig


def test_qwen3_forward_pass():
    config = TransformerConfig.qwen3_0_6B(vocab_size=1024, n_layers=2)
    model = config.build(init_device="cpu")
    model.eval()

    input_ids = torch.randint(0, 1024, (2, 16))
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (2, 16, 1024)


def test_qwen3_has_head_qk_norm():
    config = TransformerConfig.qwen3_8B(vocab_size=1024, n_layers=2)
    assert config.block.attention.qk_norm is not None
    assert config.block.attention.use_head_qk_norm is True
    assert config.block.attention.head_dim == 128

    model = config.build(init_device="cpu")

    head_dim = config.block.attention.head_dim
    assert model.blocks["0"].attention.q_norm.weight.shape == (head_dim,)
    assert model.blocks["0"].attention.k_norm.weight.shape == (head_dim,)


def test_qwen3_conversion_mappings():
    hf_config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
    )

    hf_state = {}
    for i in range(hf_config.num_hidden_layers):
        hf_state[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(
            hf_config.hidden_size
        )

    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="qwen3")

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.feed_forward_norm.weight"],
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"],
        )


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set",
)
def test_qwen3_matches_huggingface():
    model_name = "Qwen/Qwen3-0.6B"

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        token=os.environ.get("HF_TOKEN"),
    )
    hf_model.eval()
    hf_config = hf_model.config

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
    )

    olmo_config = TransformerConfig.qwen3_0_6B(
        vocab_size=hf_config.vocab_size,
    )
    olmo_model = olmo_config.build(init_device="cpu")

    converted_state = convert_state_from_hf(
        hf_config,
        hf_model.state_dict(),
        model_type="qwen3",
    )
    olmo_model.load_state_dict(converted_state)
    olmo_model.eval()

    hf_q_ptr = hf_model.model.layers[0].self_attn.q_proj.weight.data_ptr()
    olmo_q_ptr = olmo_model.blocks["0"].attention.w_q.weight.data_ptr()
    print(f"HF q_proj data_ptr: {hf_q_ptr}")
    print(f"OLMo w_q data_ptr: {olmo_q_ptr}")
    print(f"Weights aliased: {hf_q_ptr == olmo_q_ptr}")
    print(f"HF model id: {id(hf_model)}")
    print(f"OLMo model id: {id(olmo_model)}")

    prompt = "Hello, Qwen3! This is a test. Please generate 64 tokens."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        generated = hf_model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

    with torch.no_grad():
        hf_logits = hf_model(generated).logits
        olmo_logits = olmo_model(generated)

    diff = (hf_logits - olmo_logits).abs()
    print(f"Logits diff mean: {diff.mean().item():.2e}, std: {diff.std().item():.2e}")

    max_diff_per_pos = diff.max(dim=-1).values
    print(f"Max diff per position (first 10): {max_diff_per_pos[0, :10].tolist()}")

    with torch.no_grad():
        original_weight = olmo_model.blocks["0"].attention.w_q.weight.clone()
        olmo_model.blocks["0"].attention.w_q.weight.add_(1e-6)
        perturbed_logits = olmo_model(generated)
        olmo_model.blocks["0"].attention.w_q.weight.copy_(original_weight)
    perturbed_diff = (hf_logits - perturbed_logits).abs().mean()
    print(f"Perturbed logits diff: {perturbed_diff:.2e}")

    torch.testing.assert_close(hf_logits, olmo_logits, rtol=1e-6, atol=1e-6)
