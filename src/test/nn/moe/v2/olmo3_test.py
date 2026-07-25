import pytest
import torch

transformers = pytest.importorskip("transformers")

from olmo_core.nn.attention import AttentionBackendName  # noqa: E402
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes  # noqa: E402
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig  # noqa: E402
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM, Olmo3MoeModel  # noqa: E402
from olmo_core.nn.moe.v2.olmo3 import (  # noqa: E402
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
        small_config(), ep_capacity_factor=2.0, attention_backend=AttentionBackendName.torch
    )
    model = native_config.build(init_device="cpu")
    assert model.__class__.__name__ == "OLMoDDPModel"
    assert len(list(model.routed_blocks())) == 2
    assert all(block.ep.capacity_factor == 2.0 for block in native_config.resolved_block_configs)


def test_registers_base_model_for_transformers_auto_model():
    _register_olmo3moe_auto_classes()
    config = small_config()

    assert config.auto_map["AutoModel"] == "modeling_olmo3moe.Olmo3MoeModel"
    assert isinstance(transformers.AutoModel.from_config(config), Olmo3MoeModel)


def test_factory_builds_mixed_dense_moe_peri_ln_model():
    config = small_config(
        dense_layers_indices=[0],
        dense_mlp_intermediate_size=24,
        use_peri_ln=True,
    )
    native_config = build_olmo3_moe_config_from_hf_config(
        config, attention_backend=AttentionBackendName.torch
    )
    model = native_config.build(init_device="cpu")

    blocks = list(model.blocks.values())
    assert blocks[0].is_shared_only
    assert blocks[0].shared_experts.hidden_size == 24
    assert blocks[1].has_routed_experts
    assert len(list(model.routed_blocks())) == 1
    assert all(block.use_peri_norm for block in blocks)
    assert all(block.attention_input_norm is not None for block in blocks)
    assert all(block.feed_forward_input_norm is not None for block in blocks)


def test_full_hf_state_load_and_gather_roundtrip_without_ep():
    config = small_config(
        dense_layers_indices=[0],
        dense_mlp_intermediate_size=24,
        use_peri_ln=True,
    )
    hf_model = Olmo3MoeForCausalLM(config).to(dtype=torch.bfloat16)
    hf_state = {name: value.detach().clone() for name, value in hf_model.state_dict().items()}
    native_config = build_olmo3_moe_config_from_hf_config(
        config, attention_backend=AttentionBackendName.torch
    )
    native_model = native_config.build(init_device="cpu")

    load_olmo3_moe_hf_state(native_model, config, hf_state)
    roundtrip = gather_olmo3_moe_hf_state(native_model, config)

    assert roundtrip.keys() == hf_state.keys()
    for name in hf_state:
        torch.testing.assert_close(roundtrip[name], hf_state[name])
