import os

import pytest
import torch
import transformers

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

HF_TOKEN = os.environ.get("HF_TOKEN")


def _get_gemma3_config(hf_config) -> TransformerConfig:
    padded_vocab_size = (hf_config.vocab_size + 255) // 256 * 256
    return TransformerConfig.gemma3_like(
        d_model=hf_config.hidden_size,
        vocab_size=padded_vocab_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_size=hf_config.intermediate_size,
        head_dim=hf_config.head_dim,
        local_window_size=hf_config.sliding_window or 1024,
        local_rope_theta=getattr(hf_config, "rope_local_base_freq", None) or 10_000,
        global_rope_theta=hf_config.rope_theta or 1_000_000,
        layer_norm_eps=hf_config.rms_norm_eps,
        attn_backend=AttentionBackendName.torch,
    )


def _get_qwen3_config(hf_config) -> TransformerConfig:
    return TransformerConfig.qwen3_0_6B(vocab_size=hf_config.vocab_size)


MODEL_CONFIGS = [
    pytest.param(
        "google/gemma-3-270m", "gemma3_text", _get_gemma3_config, 5e-5, 5e-5, id="gemma3-270m"
    ),
    pytest.param("Qwen/Qwen3-0.6B", "qwen3", _get_qwen3_config, 5e-5, 5e-5, id="qwen3-0.6B"),
]


@pytest.mark.skipif(not HF_TOKEN, reason="HF_TOKEN not set")
@pytest.mark.parametrize("model_id,model_type,config_fn,rtol,atol", MODEL_CONFIGS)
def test_generation_logits_match(
    model_id: str, model_type: str, config_fn, rtol: float, atol: float
):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, token=HF_TOKEN, torch_dtype=torch.float32, attn_implementation="eager"
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

    olmo_config = config_fn(hf_model.config)
    olmo_model = olmo_config.build(init_device="cpu")
    converted_state = convert_state_from_hf(
        hf_model.config, hf_model.state_dict(), model_type=model_type
    )
    olmo_model.load_state_dict(converted_state)
    olmo_model.eval()

    prompt = "Hello! I am a test prompt. Please generate 64 tokens for me."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        generated = hf_model.generate(
            input_ids, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )

    print(f"Generated: {tokenizer.decode(generated[0], skip_special_tokens=True)}")

    with torch.no_grad():
        hf_logits = hf_model(generated).logits
        olmo_logits = olmo_model(generated)

    diff = (hf_logits - olmo_logits).abs()
    print(f"Logits diff mean: {diff.mean().item():.2e}, std: {diff.std().item():.2e}")

    torch.testing.assert_close(hf_logits, olmo_logits, rtol=rtol, atol=atol)
