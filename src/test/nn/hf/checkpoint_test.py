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
from olmo_core.testing.utils import HF_TOKEN, requires_hf_token


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


def _get_gemma3_config_for_hf_model(hf_config) -> TransformerConfig:
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
    )


@requires_hf_token
@pytest.mark.parametrize(
    "model_id",
    [
        "google/gemma-3-270m",
        "google/gemma-3-1b-pt",
    ],
)
def test_load_hf_gemma3_pretrained(tmp_path: Path, model_id: str):
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    hf_config = hf_model.config

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
    input_ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")[
        "input_ids"
    ]

    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)
        olmo_logits = model(input_ids=input_ids)

    torch.testing.assert_close(
        hf_logits, olmo_logits[..., : hf_config.vocab_size], atol=1e-4, rtol=1e-4
    )
