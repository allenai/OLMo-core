from pathlib import Path

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from transformers import AutoModelForCausalLM, Olmo2Config

from olmo_core.config import DType
from olmo_core.nn.hf.checkpoint import _cast_hybrid_export_dtype, load_hf_model, save_hf_model
from olmo_core.nn.transformer.config import TransformerConfig


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


def test_cast_hybrid_export_dtype_preserves_fp32_router():
    router = torch.randn(4, 8, dtype=torch.float32)
    dense = torch.randn(8, 8, dtype=torch.float32)
    state = {
        "model.layers.1.mlp.router.gate.weight": router,
        "model.layers.1.self_attn.q_proj.weight": dense,
        "metadata": "unchanged",
    }

    converted = _cast_hybrid_export_dtype(
        state,
        DType.bfloat16,
        preserve_router_precision=True,
    )

    assert converted["model.layers.1.mlp.router.gate.weight"].dtype == torch.float32
    assert converted["model.layers.1.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert converted["metadata"] == "unchanged"


def test_cast_hybrid_export_dtype_can_cast_router():
    state = {"model.layers.1.mlp.router.gate.weight": torch.randn(4, 8)}

    converted = _cast_hybrid_export_dtype(
        state,
        DType.bfloat16,
        preserve_router_precision=False,
    )

    assert converted["model.layers.1.mlp.router.gate.weight"].dtype == torch.bfloat16
