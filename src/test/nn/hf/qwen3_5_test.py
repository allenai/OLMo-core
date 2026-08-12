import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing.utils import requires_fla, requires_gpu

QWEN3_5_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def _has_qwen3_5_transformers() -> bool:
    try:
        import transformers

        return hasattr(transformers, "Qwen3_5ForCausalLM") and hasattr(
            transformers, "Qwen3_5TextConfig"
        )
    except ImportError:
        return False


def _get_qwen3_5_config(hf_config) -> TransformerConfig:
    text_config = getattr(hf_config, "text_config", hf_config)
    rope_params = getattr(text_config, "rope_parameters", {}) or {}
    return TransformerConfig.qwen3_5_like(
        d_model=text_config.hidden_size,
        vocab_size=text_config.vocab_size,
        n_layers=text_config.num_hidden_layers,
        n_heads=text_config.num_attention_heads,
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        intermediate_size=text_config.intermediate_size,
        linear_num_key_heads=text_config.linear_num_key_heads,
        linear_num_value_heads=text_config.linear_num_value_heads,
        linear_key_head_dim=text_config.linear_key_head_dim,
        linear_value_head_dim=text_config.linear_value_head_dim,
        linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
        rope_theta=rope_params.get("rope_theta", 10_000_000),
        partial_rotary_factor=text_config.partial_rotary_factor,
        attn_backend=AttentionBackendName.torch,
        tie_word_embeddings=text_config.tie_word_embeddings,
    )


def _load_pretrained(model_cls, **kwargs):
    """Load the official Qwen3.5 checkpoint, skipping the test if it can't be obtained.

    CI serves the checkpoint from an ``HF_HOME`` cache on Weka, so this loads from
    the local mount rather than downloading (a download here can blow past the GPU
    job's task timeout). Local runs fall back to a normal download.
    """
    try:
        return model_cls.from_pretrained(QWEN3_5_MODEL_ID, **kwargs)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load {QWEN3_5_MODEL_ID}: {e}")


@pytest.mark.skipif(not _has_qwen3_5_transformers(), reason="transformers lacks Qwen3.5 support")
@requires_gpu
@requires_fla
def test_qwen3_5_matches_huggingface():
    import transformers

    torch.manual_seed(0)
    device = torch.device("cuda")

    hf_model = _load_pretrained(
        transformers.Qwen3_5ForCausalLM,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval().to(device)
    tokenizer = _load_pretrained(transformers.AutoTokenizer)

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
