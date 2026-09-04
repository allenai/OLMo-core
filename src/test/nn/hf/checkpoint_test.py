from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Olmo2Config

from olmo_core.config import DType
from olmo_core.nn.hf.checkpoint import (
    _cast_hybrid_export_dtype,
    load_hf_model,
    save_hf_model,
    save_hf_model_with_native_router_overlay,
)
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


def test_native_router_overlay_preserves_template_and_replaces_only_routers(tmp_path: Path):
    template = tmp_path / "template"
    output = tmp_path / "output"
    template.mkdir()
    (template / "config.json").write_text('{"model_type": "olmo3moe"}\n')
    (template / "modeling_olmo3moe.py").write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "class Olmo3MoeRouter:\n"
        "    def forward(self, x):\n"
        "        logits = self.gate(x)\n"
        "        return logits\n"
    )
    hf_router_name = "model.layers.1.mlp.router.gate.weight"
    hf_dense_name = "model.layers.1.self_attn.q_proj.weight"
    original_dense = torch.randn(3, 3, dtype=torch.bfloat16)
    save_file(
        {
            hf_router_name: torch.zeros(4, 3, dtype=torch.bfloat16),
            hf_dense_name: original_dense,
        },
        template / "model.safetensors",
        metadata={"format": "pt"},
    )
    native_router = torch.randn(4 * 3, dtype=torch.float32)

    save_hf_model_with_native_router_overlay(
        output,
        template,
        {"blocks.1.routed_experts_router.weight": native_router},
    )

    exported = load_file(output / "model.safetensors")
    assert torch.equal(exported[hf_router_name], native_router.reshape(4, 3).to(torch.bfloat16))
    assert exported[hf_router_name].dtype == torch.bfloat16
    assert torch.equal(exported[hf_dense_name], original_dense)
    assert (output / "config.json").read_text() == (template / "config.json").read_text()
    modeling = (output / "modeling_olmo3moe.py").read_text()
    assert "with torch.autocast(device_type=x.device.type, enabled=False):" in modeling
    assert "logits = F.linear(" in modeling
    assert "self.gate.weight.float()" in modeling
    assert "logits = self.gate(x)" in (template / "modeling_olmo3moe.py").read_text()
    template_weights = load_file(template / "model.safetensors")
    assert torch.equal(template_weights[hf_router_name], torch.zeros(4, 3, dtype=torch.bfloat16))
    with safe_open(output / "model.safetensors", framework="pt") as checkpoint:
        assert checkpoint.metadata() == {"format": "pt"}


def test_native_router_overlay_rejects_mismatched_element_count(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    save_file(
        {"model.layers.1.mlp.router.gate.weight": torch.zeros(4, 3)},
        template / "model.safetensors",
    )

    with pytest.raises(RuntimeError, match="Router shape mismatch"):
        save_hf_model_with_native_router_overlay(
            tmp_path / "output",
            template,
            {"blocks.1.routed_experts_router.weight": torch.ones(11)},
        )


def test_native_router_overlay_rejects_missing_template_router(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    save_file({"dense.weight": torch.ones(2, 2)}, template / "model.safetensors")

    with pytest.raises(RuntimeError, match="missing native router"):
        save_hf_model_with_native_router_overlay(
            tmp_path / "output",
            template,
            {"blocks.1.routed_experts_router.weight": torch.ones(4, 3)},
        )
