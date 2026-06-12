import os

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing.utils import requires_fla, requires_gpu

HF_TOKEN = os.environ.get("HF_TOKEN")


def _has_qwen3_5_transformers() -> bool:
    try:
        import transformers

        return hasattr(transformers, "Qwen3_5ForCausalLM")
    except ImportError:
        return False


def _get_qwen3_5_config(hf_config) -> TransformerConfig:
    text_config = getattr(hf_config, "text_config", hf_config)
    return TransformerConfig.qwen3_5_0_8B(
        vocab_size=hf_config.vocab_size,
        attn_backend=AttentionBackendName.torch,
        tie_word_embeddings=text_config.tie_word_embeddings,
    )


@pytest.mark.skipif(not HF_TOKEN, reason="HF_TOKEN not set")
@pytest.mark.skipif(not _has_qwen3_5_transformers(), reason="transformers lacks Qwen3.5 support")
@requires_gpu
@requires_fla
def test_qwen3_5_matches_huggingface():
    import transformers

    device = torch.device("cuda")

    hf_model = transformers.Qwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        token=HF_TOKEN,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", token=HF_TOKEN)

    olmo_config = _get_qwen3_5_config(hf_model.config)
    olmo_model = olmo_config.build(init_device="cpu").eval().to(device)
    converted_state = convert_state_from_hf(
        hf_model.config,
        hf_model.state_dict(),
        model_type="qwen3_5_text",
    )
    olmo_model.load_state_dict(converted_state, strict=True)

    prompt = "Hello! I am a test prompt."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        olmo_logits = olmo_model(input_ids)

    diff = (hf_logits - olmo_logits).abs()
    print(f"Logits diff mean: {diff.mean().item():.2e}, max: {diff.max().item():.2e}")

    torch.testing.assert_close(hf_logits, olmo_logits, rtol=1e-3, atol=5e-3)
