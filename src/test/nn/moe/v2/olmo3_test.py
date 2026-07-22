import pytest
import torch

transformers = pytest.importorskip("transformers")

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM
from olmo_core.nn.moe.v2.olmo3 import (
    build_olmo3_moe_config_from_hf_config,
    gather_olmo3_moe_hf_state,
    load_olmo3_moe_hf_state,
)


def small_config(**kwargs):
    values = dict(
        vocab_size=64,
        hidden_size=32,
        attention_hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        max_position_embeddings=32,
        use_head_qk_norm=True,
        dense_layers_indices=[],
        layer_types=["full_attention", "full_attention"],
    )
    values.update(kwargs)
    return Olmo3MoeConfig(**values)


def test_factory_builds_all_moe_olmo_ddp_model():
    native_config = build_olmo3_moe_config_from_hf_config(
        small_config(), attention_backend=AttentionBackendName.torch
    )
    model = native_config.build(init_device="cpu")
    assert model.__class__.__name__ == "OLMoDDPModel"
    assert len(list(model.routed_blocks())) == 2


def test_factory_rejects_legacy_dense_layer_layout():
    with pytest.raises(NotImplementedError, match="dense_layers_indices"):
        build_olmo3_moe_config_from_hf_config(small_config(dense_layers_indices=[0]))


def test_full_hf_state_load_and_gather_roundtrip_without_ep():
    config = small_config()
    hf_model = Olmo3MoeForCausalLM(config)
    hf_state = {name: value.detach().clone() for name, value in hf_model.state_dict().items()}
    native_config = build_olmo3_moe_config_from_hf_config(
        config, dtype=DType.float32, attention_backend=AttentionBackendName.torch
    )
    native_model = native_config.build(init_device="cpu")

    load_olmo3_moe_hf_state(native_model, config, hf_state)
    roundtrip = gather_olmo3_moe_hf_state(native_model, config)

    assert roundtrip.keys() == hf_state.keys()
    for name in hf_state:
        torch.testing.assert_close(roundtrip[name], hf_state[name])
