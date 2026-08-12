"""Tests for the Qwen3 MoE OLMo-to-Hugging-Face converter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "convert_qwen3_moe_olmo_to_hf",
    Path(__file__).resolve().parents[3] / "src/scripts/convert_qwen3_moe_olmo_to_hf.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load convert_qwen3_moe_olmo_to_hf.py")
convert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_module)


def test_default_max_position_embeddings() -> None:
    config = SimpleNamespace(max_position_embeddings=32_768)

    convert_module._set_max_position_embeddings(
        config,
        convert_module._DEFAULT_MAX_POSITION_EMBEDDINGS,
    )

    assert config.max_position_embeddings == 65_536


def test_max_position_embeddings_must_be_positive() -> None:
    config = SimpleNamespace(max_position_embeddings=32_768)

    with pytest.raises(ValueError, match="must be positive"):
        convert_module._set_max_position_embeddings(config, 0)


def test_qwen35_reverse_mapping_repacks_hybrid_attention_and_experts() -> None:
    config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        layer_types=["linear_attention", "full_attention"],
        num_hidden_layers=2,
        num_experts=2,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=2,
        hidden_size=4,
        head_dim=2,
        num_attention_heads=2,
    )
    olmo_state: dict[str, torch.Tensor] = {}
    hf_state: dict[str, torch.Tensor] = {}
    next_value = 1

    def source(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal next_value
        count = int(torch.tensor(shape).prod().item())
        value = torch.arange(next_value, next_value + count, dtype=torch.float32).reshape(shape)
        next_value += count
        olmo_state[name] = value
        return value

    def target(name: str, shape: tuple[int, ...]) -> None:
        hf_state[name] = torch.zeros(shape, dtype=torch.float32)

    source("module.embeddings.weight.main", (7, 4))
    source("module.lm_head.norm.weight.main", (4,))
    source("module.lm_head.w_out.weight.main", (7, 4))
    target("model.embed_tokens.weight", (7, 4))
    target("model.norm.weight", (4,))
    target("lm_head.weight", (7, 4))

    for layer_idx, layer_type in enumerate(config.layer_types):
        hp = f"model.layers.{layer_idx}"
        op = f"module.blocks.{layer_idx}"
        source(f"{op}.attention_norm.weight.main", (4,))
        source(f"{op}.feed_forward_norm.weight.main", (4,))
        target(f"{hp}.input_layernorm.weight", (4,))
        target(f"{hp}.post_attention_layernorm.weight", (4,))

        if layer_type == "linear_attention":
            for suffix, shape in (
                ("w_q.weight.main", (2, 4)),
                ("w_k.weight.main", (2, 4)),
                ("w_v.weight.main", (4, 4)),
                ("q_conv1d.weight.main", (2, 1, 2)),
                ("k_conv1d.weight.main", (2, 1, 2)),
                ("v_conv1d.weight.main", (4, 1, 2)),
                ("w_a.weight.main", (2, 4)),
                ("w_b.weight.main", (2, 4)),
                ("w_g.weight.main", (4, 4)),
                ("w_out.weight.main", (4, 4)),
                ("o_norm.weight.main", (4,)),
                ("A_log.main", (2,)),
                ("dt_bias.main", (2,)),
            ):
                source(f"{op}.attention.{suffix}", shape)
            for suffix, shape in (
                ("in_proj_qkv.weight", (8, 4)),
                ("conv1d.weight", (8, 1, 2)),
                ("in_proj_a.weight", (2, 4)),
                ("in_proj_b.weight", (2, 4)),
                ("in_proj_z.weight", (4, 4)),
                ("out_proj.weight", (4, 4)),
                ("norm.weight", (4,)),
                ("A_log", (2,)),
                ("dt_bias", (2,)),
            ):
                target(f"{hp}.linear_attn.{suffix}", shape)
        else:
            for suffix, shape in (
                ("w_q.weight.main", (4, 4)),
                ("w_g.weight.main", (4, 4)),
                ("w_k.weight.main", (2, 4)),
                ("w_v.weight.main", (2, 4)),
                ("w_out.weight.main", (4, 4)),
                ("q_norm.weight.main", (2,)),
                ("k_norm.weight.main", (2,)),
            ):
                source(f"{op}.attention.{suffix}", shape)
            for suffix, shape in (
                ("q_proj.weight", (8, 4)),
                ("k_proj.weight", (2, 4)),
                ("v_proj.weight", (2, 4)),
                ("o_proj.weight", (4, 4)),
                ("q_norm.weight", (2,)),
                ("k_norm.weight", (2,)),
            ):
                target(f"{hp}.self_attn.{suffix}", shape)

        source(f"{op}.routed_experts_router.weight.main", (2, 4))
        source(f"{op}.routed_experts.w_up_gate.main", (2, 6, 4))
        source(f"{op}.routed_experts.w_down.main", (2, 3, 4))
        source(f"{op}.shared_experts.w_up_gate.main", (4, 4))
        source(f"{op}.shared_experts.w_down.main", (1, 2, 4))
        source(f"{op}.shared_experts_router.weight.main", (1, 4))
        target(f"{hp}.mlp.gate.weight", (2, 4))
        target(f"{hp}.mlp.experts.gate_up_proj", (2, 6, 4))
        target(f"{hp}.mlp.experts.down_proj", (2, 4, 3))
        target(f"{hp}.mlp.shared_expert.up_proj.weight", (2, 4))
        target(f"{hp}.mlp.shared_expert.gate_proj.weight", (2, 4))
        target(f"{hp}.mlp.shared_expert.down_proj.weight", (4, 2))
        target(f"{hp}.mlp.shared_expert_gate.weight", (1, 4))

    model = SimpleNamespace(config=config, state_dict=lambda: hf_state)
    convert_module.load_qwen3_moe_from_olmo_state(model, olmo_state)

    op = "module.blocks.0"
    hp = "model.layers.0"
    assert torch.equal(
        hf_state[f"{hp}.linear_attn.in_proj_qkv.weight"],
        torch.cat(
            [
                olmo_state[f"{op}.attention.w_q.weight.main"],
                olmo_state[f"{op}.attention.w_k.weight.main"],
                olmo_state[f"{op}.attention.w_v.weight.main"],
            ]
        ),
    )
    op = "module.blocks.1"
    hp = "model.layers.1"
    q = olmo_state[f"{op}.attention.w_q.weight.main"].reshape(2, 2, 4)
    gate = olmo_state[f"{op}.attention.w_g.weight.main"].reshape(2, 2, 4)
    assert torch.equal(
        hf_state[f"{hp}.self_attn.q_proj.weight"],
        torch.cat((q, gate), dim=1).reshape(8, 4),
    )
    routed = olmo_state[f"{op}.routed_experts.w_up_gate.main"]
    up, gate = routed.split(3, dim=1)
    assert torch.equal(hf_state[f"{hp}.mlp.experts.gate_up_proj"], torch.cat((gate, up), dim=1))
    assert torch.equal(
        hf_state[f"{hp}.mlp.shared_expert.down_proj.weight"],
        olmo_state[f"{op}.shared_experts.w_down.main"].squeeze(0).t(),
    )
