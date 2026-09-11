"""Exercise the adapter factory against gdn2's real hybrid modules and HF tensors."""

from copy import deepcopy

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionType,
    KimiDeltaAttention,
)
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM


def hybrid_config(latent_dim=16):
    return Olmo3MoeConfig(
        vocab_size=32,
        hidden_size=32,
        attention_hidden_size=32,
        head_dim=8,
        dense_mlp_intermediate_size=24,
        dense_layers_use_shared_expert=True,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_allow_neg_eigval=True,
        latent_moe_dim=latent_dim,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        embed_norm=True,
        use_peri_ln=True,
    )


@pytest.mark.parametrize("latent_dim", [None, 16])
def test_hybrid_factory_uses_native_components_and_roundtrips_weights(latent_dim):
    hf = hybrid_config(latent_dim)
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=AttentionBackendName.torch
    )
    native = config.build(init_device="cpu")
    assert isinstance(native.blocks["0"].attention, KimiDeltaAttention)
    sparse = native.blocks["1"]
    assert sparse.routed_experts.d_model == (latent_dim or hf.hidden_size)
    assert sparse.routed_experts_router.d_model == hf.hidden_size
    assert sparse.shared_experts.d_model == hf.hidden_size
    assert (sparse.latent_down_proj is not None) == (latent_dim is not None)
    assert sparse.attention.rope is None
    assert sparse.attention.gate.full_precision
    reference = Olmo3MoeForCausalLM(hf).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
    state = reference.state_dict()
    olmo3.load_olmo3_moe_hf_state(native, hf, state)
    exported = olmo3.gather_olmo3_moe_hf_state(native, hf, cpu=True)
    assert set(exported) == set(state)
    for name in state:
        torch.testing.assert_close(exported[name], state[name], rtol=0, atol=0, check_dtype=False)


@pytest.mark.parametrize("latent_dim", [None, 16])
def test_streaming_export_matches_complete_export_and_is_lazy(latent_dim):
    hf = hybrid_config(latent_dim)
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=AttentionBackendName.torch
    )
    native = config.build(init_device="cpu")
    reference = Olmo3MoeForCausalLM(hf).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
    olmo3.load_olmo3_moe_hf_state(native, hf, reference.state_dict())
    expected = olmo3.gather_olmo3_moe_hf_state(native, hf, cpu=True)
    actual = dict(olmo3.iter_olmo3_moe_hf_state(native, hf))
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    # A later missing weight must not be touched by the first yield. Full
    # consumption still rejects incomplete state, retaining strict conversion.
    del native.blocks["1"].attention.w_out.weight
    stream = olmo3.iter_olmo3_moe_hf_state(native, hf)
    assert next(stream)[0] == "model.embed_tokens.weight"
    with pytest.raises(KeyError):
        dict(stream)


def test_streaming_export_splits_fused_attention_weights():
    hf = hybrid_config(None)
    hf.layer_types = ["full_attention", "full_attention"]
    hf.attention_gate_type = None
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf,
        dtype=DType.bfloat16,
        attention_backend=AttentionBackendName.torch,
        attention_type=AttentionType.fused_v2,
    )
    native = config.build(init_device="cpu")
    reference = Olmo3MoeForCausalLM(hf).to(torch.bfloat16)
    olmo3.load_olmo3_moe_hf_state(native, hf, reference.state_dict())
    expected = reference.state_dict()
    actual = dict(olmo3.iter_olmo3_moe_hf_state(native, hf))
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("hidden,heads,kv,latent", [(32, 4, 2, 16), (48, 6, 3, 24)])
@pytest.mark.parametrize("head_gains,ssmax", [(False, False), (True, False), (True, True)])
def test_hero_config_and_streaming_state_roundtrip(hidden, heads, kv, latent, head_gains, ssmax):
    hf = hybrid_config(latent)
    hf.hidden_size = hidden
    hf.attention_hidden_size = hidden
    hf.num_attention_heads = heads
    hf.num_key_value_heads = kv
    hf.qk_norm_per_head_gains = head_gains
    hf.scalable_softmax = ssmax
    hf.latent_moe_bias = True
    hf.latent_moe_up_proj_input_norm = True
    hf.layer_types = ["linear_attention", "full_attention", "linear_attention"]
    hf.num_hidden_layers = 3
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=AttentionBackendName.torch
    )
    reverse = olmo3.build_olmo3_moe_hf_config_from_native_config(
        config,
        max_position_embeddings=hf.max_position_embeddings,
        pad_token_id=hf.pad_token_id,
        bos_token_id=hf.bos_token_id,
        eos_token_id=hf.eos_token_id,
    )
    fields = [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "qk_norm_per_head_gains",
        "scalable_softmax",
        "use_rope",
        "attention_gate_type",
        "attention_gate_full_precision",
        "layer_types",
        "dense_layers_indices",
        "dense_layers_use_shared_expert",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_allow_neg_eigval",
        "linear_conv_kernel_dim",
        "linear_norm_eps",
        "latent_moe_dim",
        "latent_moe_bias",
        "latent_moe_up_proj_input_norm",
        "n_routed_experts",
        "num_experts_per_tok",
    ]
    assert {k: getattr(reverse, k) for k in fields} == {k: getattr(hf, k) for k in fields}
    native = config.build(init_device="cpu")
    attention = native.blocks["1"].attention
    assert tuple(attention.q_norm.weight.shape) == ((heads, 8) if head_gains else (8,))
    assert tuple(attention.k_norm.weight.shape) == ((kv, 8) if head_gains else (8,))
    if ssmax:
        assert tuple(attention.ssmax_scale.shape) == (heads,)
    reference = Olmo3MoeForCausalLM(hf).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                parameter.zero_()
            elif name.endswith(("q_norm.weight", "k_norm.weight", "ssmax_scale")):
                parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) / 32 + 1)
    expected = reference.state_dict()
    olmo3.load_olmo3_moe_hf_state(native, reverse, expected)
    actual = dict(olmo3.iter_olmo3_moe_hf_state(native, reverse))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, check_dtype=False)
    if ssmax:
        del native.blocks["1"].attention.ssmax_scale
        with pytest.raises(KeyError, match="ssmax_scale"):
            dict(olmo3.iter_olmo3_moe_hf_state(native, reverse))


def test_reverse_config_rejects_heterogeneous_attention_features():
    hf = hybrid_config()
    hf.num_hidden_layers = 3
    hf.layer_types = ["linear_attention", "full_attention", "full_attention"]
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, attention_backend=AttentionBackendName.torch
    )
    config.block["other_full"] = deepcopy(config.block["full_attention"])
    config.block["other_full"].sequence_mixer.scalable_softmax = True
    config.block_pattern[-1] = "other_full"
    with pytest.raises(ValueError, match="consistent across layers"):
        olmo3.build_olmo3_moe_hf_config_from_native_config(
            config,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=1,
        )


@pytest.mark.parametrize("feature", ["qk_norm_per_head_gains", "scalable_softmax"])
def test_fused_attention_rejects_unsupported_hero_features(feature):
    hf = hybrid_config()
    setattr(hf, feature, True)
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf,
        attention_backend=AttentionBackendName.torch,
        attention_type=AttentionType.fused_v2,
    )
    with pytest.raises(
        OLMoConfigurationError, match=f"'{feature}' is only supported by default attention"
    ):
        config.build(init_device="meta")


@pytest.mark.parametrize("global_lb", [False, True])
def test_hf_ordinary_dense_layout_imports_without_mutating_config(global_lb):
    hf = hybrid_config()
    hf.dense_layers_use_shared_expert = False
    hf.global_load_balancing = global_lb
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf,
        dtype=DType.bfloat16,
        attention_backend=AttentionBackendName.torch,
    )
    model = config.build(init_device="cpu")
    assert model.blocks["1"].routed_experts_router.global_load_balancing == global_lb
    reference = Olmo3MoeForCausalLM(hf).to(torch.bfloat16)
    with torch.no_grad():
        for name, value in reference.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                value.zero_()
    expected = reference.state_dict()
    olmo3.load_olmo3_moe_hf_state(model, hf, expected)
    assert hf.dense_layers_use_shared_expert is False
    streamed = dict(olmo3.iter_olmo3_moe_hf_state(model, hf))
    gathered = olmo3.gather_olmo3_moe_hf_state(model, hf)
    assert hf.dense_layers_use_shared_expert is False
    assert streamed.keys() == gathered.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(streamed[key], expected[key], rtol=0, atol=0, check_dtype=False)
        torch.testing.assert_close(gathered[key], expected[key], rtol=0, atol=0, check_dtype=False)
    reverse = olmo3.build_olmo3_moe_hf_config_from_native_config(
        config,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=1,
    )
    assert reverse.dense_layers_use_shared_expert is True
    assert reverse.global_load_balancing == global_lb


def test_sliding_window_size_matches_hf_in_both_config_directions():
    hf = hybrid_config()
    hf.layer_types = ["linear_attention", "sliding_attention"]
    hf.sliding_window = 4
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, attention_backend=AttentionBackendName.torch
    )
    native = config.build(init_device="cpu")
    assert native.blocks["1"].attention.backend.window_size == (3, 0)
    reverse = olmo3.build_olmo3_moe_hf_config_from_native_config(
        config,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=1,
    )
    assert reverse.sliding_window == 4
    assert reverse.layer_types == hf.layer_types
