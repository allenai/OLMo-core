"""
Roundtrip tests for the ``olmo3moe`` <-> OLMo-core MoE-v2 state-dict conversion.

These exercise the conversion functions directly on synthetic tensor dicts with a stub config, so
they don't require ``transformers`` or a built model — they verify the HF<->OLMo-core mapping is a
faithful bijection (including the fused ``w_up_gate`` split and the mixed dense/MoE layer layout).
"""

import types

import pytest
import torch

from olmo_core.nn.hf.convert import (
    convert_olmo3moe_state_from_hf,
    convert_olmo3moe_state_to_hf,
)


def _fake_config():
    # Distinct hidden sizes (dense/moe/shared) so a transpose or shape slip can't roundtrip by
    # accident. Layer 0 is dense, layer 1 is MoE.
    return types.SimpleNamespace(
        num_hidden_layers=2,
        n_routed_experts=3,
        hidden_size=8,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=4,
        dense_layers_indices=[0],
        embed_norm=True,
        use_peri_ln=False,
    )


def _synthetic_hf_state(config):
    torch.manual_seed(0)
    d = config.hidden_size
    e = config.n_routed_experts
    h = config.moe_intermediate_size
    hs = config.shared_expert_intermediate_size
    vocab, dense_h = 6, 7
    dense = set(config.dense_layers_indices)

    hf = {
        "model.embed_tokens.weight": torch.randn(vocab, d),
        "model.norm.weight": torch.randn(d),
        "lm_head.weight": torch.randn(vocab, d),
        "model.embed_norm.weight": torch.randn(d),
    }
    for layer in range(config.num_hidden_layers):
        p = f"model.layers.{layer}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hf[f"{p}self_attn.{proj}.weight"] = torch.randn(d, d)
        for norm in ("q_norm", "k_norm"):
            hf[f"{p}self_attn.{norm}.weight"] = torch.randn(d)
        hf[f"{p}post_attention_layernorm.weight"] = torch.randn(d)
        hf[f"{p}post_feedforward_layernorm.weight"] = torch.randn(d)
        if layer in dense:
            hf[f"{p}mlp.gate_proj.weight"] = torch.randn(dense_h, d)
            hf[f"{p}mlp.up_proj.weight"] = torch.randn(dense_h, d)
            hf[f"{p}mlp.down_proj.weight"] = torch.randn(d, dense_h)
        else:
            hf[f"{p}mlp.router.gate.weight"] = torch.randn(e, d)
            for expert in range(e):
                hf[f"{p}mlp.experts.{expert}.up_proj.weight"] = torch.randn(h, d)
                hf[f"{p}mlp.experts.{expert}.gate_proj.weight"] = torch.randn(h, d)
                hf[f"{p}mlp.experts.{expert}.down_proj.weight"] = torch.randn(d, h)
            hf[f"{p}mlp.shared_expert.up_proj.weight"] = torch.randn(hs, d)
            hf[f"{p}mlp.shared_expert.gate_proj.weight"] = torch.randn(hs, d)
            hf[f"{p}mlp.shared_expert.down_proj.weight"] = torch.randn(d, hs)
    return hf


def test_olmo3moe_hf_conversion_roundtrips():
    config = _fake_config()
    hf = _synthetic_hf_state(config)

    olmo = convert_olmo3moe_state_from_hf(config, hf)
    hf_roundtrip = convert_olmo3moe_state_to_hf(config, olmo)

    assert set(hf_roundtrip.keys()) == set(hf.keys())
    for key, tensor in hf.items():
        assert torch.equal(hf_roundtrip[key], tensor), f"roundtrip mismatch for '{key}'"


def test_olmo3moe_conversion_rejects_peri_ln():
    config = _fake_config()
    config.use_peri_ln = True
    with pytest.raises(NotImplementedError, match="use_peri_ln"):
        convert_olmo3moe_state_from_hf(config, {})
