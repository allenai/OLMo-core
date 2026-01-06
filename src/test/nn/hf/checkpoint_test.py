import logging
import os
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3TextConfig,
    Olmo2Config,
)

from olmo_core.nn.hf.checkpoint import load_hf_model, save_hf_model
from olmo_core.nn.transformer.config import TransformerConfig

log = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
requires_hf_token = pytest.mark.skipif(
    HF_TOKEN is None,
    reason="HF_TOKEN environment variable not set - required for accessing gated models",
)


def test_load_hf_model(tmp_path: Path):
    vocab_size = 200
    padded_vocab_size = 256
    model_config = TransformerConfig.olmo2_190M(padded_vocab_size)

    hf_config = Olmo2Config(
        vocab_size=vocab_size,
        hidden_size=model_config.d_model,
        intermediate_size=3072,
        num_hidden_layers=model_config.n_layers,
        num_attention_heads=12,
        rope_theta=500_000,
        rms_norm_eps=1e-6,
    )
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    hf_model.save_pretrained(tmp_path / "hf")

    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=padded_vocab_size,
    )
    model.load_state_dict(model_state_dict)

    rand_input = torch.randint(0, vocab_size, (2, 3))
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=rand_input)

    assert hf_logits.shape[-1] == vocab_size
    assert logits.shape[-1] == padded_vocab_size
    torch.testing.assert_close(hf_logits, logits[..., :vocab_size])


def test_save_hf_model(tmp_path: Path):
    vocab_size = 200
    padded_vocab_size = 256
    model_config = TransformerConfig.olmo2_190M(padded_vocab_size)
    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    save_hf_model(
        tmp_path / "hf",
        model_state_dict,
        model,
        vocab_size=vocab_size,
    )
    model.load_state_dict(model_state_dict)

    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path / "hf")

    rand_input = torch.randint(0, vocab_size, (2, 3))
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=rand_input)

    assert hf_logits.shape[-1] == vocab_size
    assert logits.shape[-1] == padded_vocab_size
    torch.testing.assert_close(hf_logits, logits[..., :vocab_size])


def test_load_hf_gemma3(tmp_path: Path):
    vocab_size = 200
    padded_vocab_size = 256
    n_layers = 2
    model_config = TransformerConfig.gemma3_1B(padded_vocab_size, n_layers=n_layers)

    n_heads = model_config.block.attention.n_heads
    head_dim = model_config.d_model // n_heads
    assert model_config.block.feed_forward is not None
    hf_config = Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=model_config.d_model,
        intermediate_size=model_config.block.feed_forward.hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=model_config.block.attention.n_kv_heads,
        head_dim=head_dim,
        rope_theta=10_000,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=head_dim,
        sliding_window=1024,
        hidden_activation="gelu_pytorch_tanh",
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
    )
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    hf_model.save_pretrained(tmp_path / "hf")

    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=padded_vocab_size,
    )
    model.load_state_dict(model_state_dict)

    rand_input = torch.randint(0, vocab_size, (2, 3))
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=rand_input, return_dict=False)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=rand_input)

    assert hf_logits.shape[-1] == vocab_size
    assert logits.shape[-1] == padded_vocab_size
    torch.testing.assert_close(hf_logits, logits[..., :vocab_size])


def test_gemma3_debug_layer_by_layer(tmp_path: Path):
    """Debug test to find where HF and olmo-core diverge layer by layer."""
    vocab_size = 200
    padded_vocab_size = 256
    n_layers = 2

    model_config = TransformerConfig.gemma3_1B(padded_vocab_size, n_layers=n_layers)

    n_heads = model_config.block.attention.n_heads
    head_dim = model_config.d_model // n_heads
    assert model_config.block.feed_forward is not None
    hf_config = Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=model_config.d_model,
        intermediate_size=model_config.block.feed_forward.hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=model_config.block.attention.n_kv_heads,
        head_dim=head_dim,
        rope_theta=10_000,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=head_dim,
        sliding_window=1024,
        hidden_activation="gelu_pytorch_tanh",
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
    )
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    hf_model.save_pretrained(tmp_path / "hf")

    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=padded_vocab_size,
    )
    model.load_state_dict(model_state_dict)

    hf_intermediates = {}
    olmo_intermediates = {}

    def make_hf_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hf_intermediates[name] = output[0].detach().clone()
            else:
                hf_intermediates[name] = output.detach().clone()

        return hook

    def make_olmo_hook(name):
        def hook(module, input, output):
            olmo_intermediates[name] = output.detach().clone()

        return hook

    hf_model.model.embed_tokens.register_forward_hook(make_hf_hook("embed"))
    for i, layer in enumerate(hf_model.model.layers):
        layer.input_layernorm.register_forward_hook(make_hf_hook(f"layer{i}_input_norm"))
        layer.self_attn.register_forward_hook(make_hf_hook(f"layer{i}_attn"))
        layer.post_attention_layernorm.register_forward_hook(
            make_hf_hook(f"layer{i}_post_attn_norm")
        )
        layer.pre_feedforward_layernorm.register_forward_hook(make_hf_hook(f"layer{i}_pre_ff_norm"))
        layer.mlp.register_forward_hook(make_hf_hook(f"layer{i}_ff"))
        layer.post_feedforward_layernorm.register_forward_hook(
            make_hf_hook(f"layer{i}_post_ff_norm")
        )
    hf_model.model.norm.register_forward_hook(make_hf_hook("final_norm"))
    hf_model.lm_head.register_forward_hook(make_hf_hook("lm_head"))

    model.embeddings.register_forward_hook(make_olmo_hook("embed"))
    for i, (key, block) in enumerate(model.blocks.items()):
        block.attention_norm.register_forward_hook(make_olmo_hook(f"layer{i}_input_norm"))
        block.attention.register_forward_hook(make_olmo_hook(f"layer{i}_attn"))
        block.feed_forward_norm.register_forward_hook(make_olmo_hook(f"layer{i}_ff_norm"))
        block.feed_forward.register_forward_hook(make_olmo_hook(f"layer{i}_ff"))
    assert model.lm_head.norm is not None
    model.lm_head.norm.register_forward_hook(make_olmo_hook("final_norm"))
    model.lm_head.w_out.register_forward_hook(make_olmo_hook("lm_head"))

    rand_input = torch.randint(0, vocab_size, (1, 4))
    with torch.no_grad():
        hf_model(input_ids=rand_input)
        model.eval()
        model(input_ids=rand_input)

    log.info("=" * 60)
    log.info("Layer-by-layer comparison (HF Gemma3 vs OLMo-core)")
    log.info("=" * 60)

    all_keys = sorted(set(hf_intermediates.keys()) | set(olmo_intermediates.keys()))
    for key in all_keys:
        hf_val = hf_intermediates.get(key)
        olmo_val = olmo_intermediates.get(key)

        if hf_val is None:
            log.info(f"{key}: MISSING in HF")
            continue
        if olmo_val is None:
            log.info(f"{key}: MISSING in OLMo")
            continue

        if hf_val.shape != olmo_val.shape:
            log.info(f"{key}: SHAPE MISMATCH - HF {hf_val.shape} vs OLMo {olmo_val.shape}")
            continue

        max_diff = (hf_val - olmo_val).abs().max().item()
        mean_diff = (hf_val - olmo_val).abs().mean().item()
        hf_norm = hf_val.abs().mean().item()
        olmo_norm = olmo_val.abs().mean().item()

        status = "OK" if max_diff < 1e-4 else "DIVERGED"
        log.info(
            f"{key}: {status} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
            f"hf_mean={hf_norm:.4f} olmo_mean={olmo_norm:.4f}"
        )

    log.info("=" * 60)


def _get_gemma3_config_for_hf_model(hf_config) -> TransformerConfig:
    """Create an OLMo TransformerConfig that matches an HF Gemma3 config."""
    vocab_size = hf_config.vocab_size
    padded_vocab_size = (vocab_size + 255) // 256 * 256

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
    )


@requires_hf_token
def test_load_hf_gemma3_270m_pretrained(tmp_path: Path):
    """Test loading google/gemma-3-270m pretrained model."""
    model_id = "google/gemma-3-270m"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_config = hf_model.config

    log.info(f"HF model config: {hf_config}")
    log.info(f"HF model: {hf_model}")

    model_config = _get_gemma3_config_for_hf_model(hf_config)
    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    hf_model.save_pretrained(tmp_path / "hf")
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=model_config.vocab_size,
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    test_input = tokenizer("Hello, world!", return_tensors="pt")
    input_ids = test_input["input_ids"]

    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)
        olmo_logits = model(input_ids=input_ids)

    vocab_size = hf_config.vocab_size
    log.info(f"HF logits shape: {hf_logits.shape}, OLMo logits shape: {olmo_logits.shape}")

    max_diff = (hf_logits - olmo_logits[..., :vocab_size]).abs().max().item()
    mean_diff = (hf_logits - olmo_logits[..., :vocab_size]).abs().mean().item()
    log.info(f"Max diff: {max_diff}, Mean diff: {mean_diff}")

    torch.testing.assert_close(hf_logits, olmo_logits[..., :vocab_size], atol=1e-4, rtol=1e-4)


@requires_hf_token
def test_load_hf_gemma3_1b_pretrained(tmp_path: Path):
    """Test loading google/gemma-3-1b-pt pretrained model."""
    model_id = "google/gemma-3-1b-pt"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_config = hf_model.config

    log.info(f"HF model config: {hf_config}")

    model_config = _get_gemma3_config_for_hf_model(hf_config)
    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    hf_model.save_pretrained(tmp_path / "hf")
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=model_config.vocab_size,
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    test_input = tokenizer("Hello, world!", return_tensors="pt")
    input_ids = test_input["input_ids"]

    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)
        olmo_logits = model(input_ids=input_ids)

    vocab_size = hf_config.vocab_size
    log.info(f"HF logits shape: {hf_logits.shape}, OLMo logits shape: {olmo_logits.shape}")

    max_diff = (hf_logits - olmo_logits[..., :vocab_size]).abs().max().item()
    mean_diff = (hf_logits - olmo_logits[..., :vocab_size]).abs().mean().item()
    log.info(f"Max diff: {max_diff}, Mean diff: {mean_diff}")

    torch.testing.assert_close(hf_logits, olmo_logits[..., :vocab_size], atol=1e-4, rtol=1e-4)


@requires_hf_token
@pytest.mark.parametrize(
    "model_id",
    [
        "google/gemma-3-270m",
        "google/gemma-3-1b-pt",
    ],
)
def test_load_hf_gemma3_models(tmp_path: Path, model_id: str):
    """Parametrized test for loading various Gemma 3 HF models."""
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_config = hf_model.config

    log.info(f"Testing model: {model_id}")
    log.info(
        f"HF config: hidden_size={hf_config.hidden_size}, n_layers={hf_config.num_hidden_layers}"
    )

    model_config = _get_gemma3_config_for_hf_model(hf_config)
    model = model_config.build()

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    hf_model.save_pretrained(tmp_path / "hf")
    load_hf_model(
        tmp_path / "hf",
        model_state_dict,
        num_embeddings=model_config.vocab_size,
    )
    model.load_state_dict(model_state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    test_input = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    input_ids = test_input["input_ids"]

    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)
        olmo_logits = model(input_ids=input_ids)

    vocab_size = hf_config.vocab_size
    max_diff = (hf_logits - olmo_logits[..., :vocab_size]).abs().max().item()
    mean_diff = (hf_logits - olmo_logits[..., :vocab_size]).abs().mean().item()
    log.info(f"Model {model_id}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    torch.testing.assert_close(hf_logits, olmo_logits[..., :vocab_size], atol=1e-4, rtol=1e-4)
