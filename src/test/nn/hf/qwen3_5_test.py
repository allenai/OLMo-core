import os

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

HF_TOKEN = os.environ.get("HF_TOKEN")


def _has_qwen3_5_transformers() -> bool:
    try:
        import transformers

        return hasattr(transformers, "Qwen3_5ForConditionalGeneration")
    except ImportError:
        return False


def _get_qwen3_5_config(hf_config) -> TransformerConfig:
    text_config = getattr(hf_config, "text_config", hf_config)
    return TransformerConfig.qwen3_5_0_8B(
        vocab_size=text_config.vocab_size,
        attn_backend=AttentionBackendName.torch,
    )


@pytest.mark.skipif(not HF_TOKEN, reason="HF_TOKEN not set")
@pytest.mark.skipif(not _has_qwen3_5_transformers(), reason="transformers lacks Qwen3.5 support")
@pytest.mark.skipif(not has_fla(), reason="flash-linear-attention (fla) not available")
def test_qwen3_5_matches_huggingface():
    import transformers

    hf_model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        token=HF_TOKEN,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", token=HF_TOKEN)

    olmo_config = _get_qwen3_5_config(hf_model.config)
    olmo_model = olmo_config.build(init_device="cpu")
    converted_state = convert_state_from_hf(
        hf_model.config,
        hf_model.state_dict(),
        model_type="qwen3_5_text",
    )
    olmo_model.load_state_dict(converted_state, strict=True)
    olmo_model.eval()

    prompt = "Hello! I am a test prompt."
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        olmo_logits = olmo_model(input_ids)

    diff = (hf_logits - olmo_logits).abs()
    print(f"Logits diff mean: {diff.mean().item():.2e}, max: {diff.max().item():.2e}")

    torch.testing.assert_close(hf_logits, olmo_logits, rtol=1e-4, atol=1e-4)
