from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from olmo_core.nn.hf.convert import (
    convert_qwen3_5_state_from_hf,
    convert_qwen3_5_state_to_hf,
    convert_state_from_hf,
    convert_state_to_hf,
)


def _text_config(*, tie_word_embeddings: bool = False) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=tie_word_embeddings,
    )


def _tensor(shape: tuple[int, ...], value: float) -> torch.Tensor:
    numel = 1
    for size in shape:
        numel *= size
    return torch.arange(numel, dtype=torch.float32).reshape(shape) / 100 + value


def _olmo_state(*, tied: bool = False) -> dict[str, torch.Tensor]:
    state = {
        "embeddings.weight": _tensor((16, 8), 1),
        "lm_head.norm.weight": _tensor((8,), 1),
        "lm_head.w_out.weight": _tensor((16, 8), 2),
        # GDN layer.
        "blocks.0.attention.w_q.weight": _tensor((2, 8), 10),
        "blocks.0.attention.w_k.weight": _tensor((2, 8), 20),
        "blocks.0.attention.w_v.weight": _tensor((4, 8), 30),
        "blocks.0.attention.w_g.weight": _tensor((4, 8), 40),
        "blocks.0.attention.w_a.weight": _tensor((2, 8), 50),
        "blocks.0.attention.w_b.weight": _tensor((2, 8), 60),
        "blocks.0.attention.w_out.weight": _tensor((8, 4), 70),
        "blocks.0.attention.q_conv1d.weight": _tensor((2, 1, 4), 80),
        "blocks.0.attention.k_conv1d.weight": _tensor((2, 1, 4), 90),
        "blocks.0.attention.v_conv1d.weight": _tensor((4, 1, 4), 100),
        "blocks.0.attention.o_norm.weight": _tensor((2,), 110),
        "blocks.0.attention.A_log": _tensor((2,), 120),
        "blocks.0.attention.dt_bias": _tensor((2,), 130),
        "blocks.0.attention_norm.weight": _tensor((8,), 1),
        "blocks.0.feed_forward_norm.weight": _tensor((8,), 1.25),
        "blocks.0.feed_forward.w1.weight": _tensor((12, 8), 140),
        "blocks.0.feed_forward.w2.weight": _tensor((8, 12), 150),
        "blocks.0.feed_forward.w3.weight": _tensor((12, 8), 160),
        # Full-attention layer.
        "blocks.1.attention.w_q.weight": _tensor((4, 8), 170),
        "blocks.1.attention.w_g.weight": _tensor((4, 8), 180),
        "blocks.1.attention.w_k.weight": _tensor((2, 8), 190),
        "blocks.1.attention.w_v.weight": _tensor((2, 8), 200),
        "blocks.1.attention.w_out.weight": _tensor((8, 4), 210),
        "blocks.1.attention.q_norm.weight": _tensor((2,), 1.5),
        "blocks.1.attention.k_norm.weight": _tensor((2,), 1.75),
        "blocks.1.attention_norm.weight": _tensor((8,), 2),
        "blocks.1.feed_forward_norm.weight": _tensor((8,), 2.25),
        "blocks.1.feed_forward.w1.weight": _tensor((12, 8), 220),
        "blocks.1.feed_forward.w2.weight": _tensor((8, 12), 230),
        "blocks.1.feed_forward.w3.weight": _tensor((12, 8), 240),
    }
    if tied:
        # DCP reconstruction preserves logical values but may not preserve storage aliasing.
        state["lm_head.w_out.weight"] = state["embeddings.weight"].clone()
    return state


def _regular_norm_keys() -> set[str]:
    return {
        "lm_head.norm.weight",
        "blocks.0.attention_norm.weight",
        "blocks.0.feed_forward_norm.weight",
        "blocks.1.attention.q_norm.weight",
        "blocks.1.attention.k_norm.weight",
        "blocks.1.attention_norm.weight",
        "blocks.1.feed_forward_norm.weight",
    }


def _regular_hf_norm_keys(state: dict[str, torch.Tensor]) -> set[str]:
    patterns = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "model.norm.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    return {key for key in state if any(pattern in key for pattern in patterns)}


def test_qwen3_5_to_hf_uses_official_fused_layouts_and_is_non_mutating():
    config = _text_config()
    olmo_state = _olmo_state()
    original = deepcopy(olmo_state)

    hf_state = convert_qwen3_5_state_to_hf(config, olmo_state)

    torch.testing.assert_close(
        hf_state["model.layers.0.linear_attn.in_proj_qkv.weight"],
        torch.cat(
            [
                original["blocks.0.attention.w_q.weight"],
                original["blocks.0.attention.w_k.weight"],
                original["blocks.0.attention.w_v.weight"],
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        hf_state["model.layers.0.linear_attn.conv1d.weight"],
        torch.cat(
            [
                original["blocks.0.attention.q_conv1d.weight"],
                original["blocks.0.attention.k_conv1d.weight"],
                original["blocks.0.attention.v_conv1d.weight"],
            ]
        ),
        rtol=0,
        atol=0,
    )

    q_proj = hf_state["model.layers.1.self_attn.q_proj.weight"]
    for head_idx in range(config.num_attention_heads):
        olmo_start = head_idx * config.head_dim
        hf_start = head_idx * 2 * config.head_dim
        torch.testing.assert_close(
            q_proj[hf_start : hf_start + config.head_dim],
            original["blocks.1.attention.w_q.weight"][olmo_start : olmo_start + config.head_dim],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            q_proj[hf_start + config.head_dim : hf_start + 2 * config.head_dim],
            original["blocks.1.attention.w_g.weight"][olmo_start : olmo_start + config.head_dim],
            rtol=0,
            atol=0,
        )

    torch.testing.assert_close(
        hf_state["model.layers.0.linear_attn.norm.weight"],
        original["blocks.0.attention.o_norm.weight"],
        rtol=0,
        atol=0,
    )
    for olmo_key, hf_key in [
        ("lm_head.norm.weight", "model.norm.weight"),
        ("blocks.0.attention_norm.weight", "model.layers.0.input_layernorm.weight"),
        (
            "blocks.1.attention.q_norm.weight",
            "model.layers.1.self_attn.q_norm.weight",
        ),
    ]:
        torch.testing.assert_close(hf_state[hf_key], original[olmo_key] - 1, rtol=0, atol=0)

    expected_q_proj = torch.cat(
        [
            tensor
            for head_idx in range(config.num_attention_heads)
            for tensor in (
                original["blocks.1.attention.w_q.weight"][
                    head_idx * config.head_dim : (head_idx + 1) * config.head_dim
                ],
                original["blocks.1.attention.w_g.weight"][
                    head_idx * config.head_dim : (head_idx + 1) * config.head_dim
                ],
            )
        ]
    )
    expected = {
        "model.embed_tokens.weight": original["embeddings.weight"],
        "model.norm.weight": original["lm_head.norm.weight"] - 1,
        "lm_head.weight": original["lm_head.w_out.weight"],
        "model.layers.0.linear_attn.in_proj_qkv.weight": torch.cat(
            [
                original["blocks.0.attention.w_q.weight"],
                original["blocks.0.attention.w_k.weight"],
                original["blocks.0.attention.w_v.weight"],
            ]
        ),
        "model.layers.0.linear_attn.in_proj_z.weight": original["blocks.0.attention.w_g.weight"],
        "model.layers.0.linear_attn.in_proj_a.weight": original["blocks.0.attention.w_a.weight"],
        "model.layers.0.linear_attn.in_proj_b.weight": original["blocks.0.attention.w_b.weight"],
        "model.layers.0.linear_attn.out_proj.weight": original["blocks.0.attention.w_out.weight"],
        "model.layers.0.linear_attn.conv1d.weight": torch.cat(
            [
                original["blocks.0.attention.q_conv1d.weight"],
                original["blocks.0.attention.k_conv1d.weight"],
                original["blocks.0.attention.v_conv1d.weight"],
            ]
        ),
        "model.layers.0.linear_attn.norm.weight": original["blocks.0.attention.o_norm.weight"],
        "model.layers.0.linear_attn.A_log": original["blocks.0.attention.A_log"],
        "model.layers.0.linear_attn.dt_bias": original["blocks.0.attention.dt_bias"],
        "model.layers.0.input_layernorm.weight": original["blocks.0.attention_norm.weight"] - 1,
        "model.layers.0.post_attention_layernorm.weight": original[
            "blocks.0.feed_forward_norm.weight"
        ]
        - 1,
        "model.layers.0.mlp.gate_proj.weight": original["blocks.0.feed_forward.w1.weight"],
        "model.layers.0.mlp.down_proj.weight": original["blocks.0.feed_forward.w2.weight"],
        "model.layers.0.mlp.up_proj.weight": original["blocks.0.feed_forward.w3.weight"],
        "model.layers.1.self_attn.q_proj.weight": expected_q_proj,
        "model.layers.1.self_attn.k_proj.weight": original["blocks.1.attention.w_k.weight"],
        "model.layers.1.self_attn.v_proj.weight": original["blocks.1.attention.w_v.weight"],
        "model.layers.1.self_attn.o_proj.weight": original["blocks.1.attention.w_out.weight"],
        "model.layers.1.self_attn.q_norm.weight": original["blocks.1.attention.q_norm.weight"] - 1,
        "model.layers.1.self_attn.k_norm.weight": original["blocks.1.attention.k_norm.weight"] - 1,
        "model.layers.1.input_layernorm.weight": original["blocks.1.attention_norm.weight"] - 1,
        "model.layers.1.post_attention_layernorm.weight": original[
            "blocks.1.feed_forward_norm.weight"
        ]
        - 1,
        "model.layers.1.mlp.gate_proj.weight": original["blocks.1.feed_forward.w1.weight"],
        "model.layers.1.mlp.down_proj.weight": original["blocks.1.feed_forward.w2.weight"],
        "model.layers.1.mlp.up_proj.weight": original["blocks.1.feed_forward.w3.weight"],
    }
    assert hf_state.keys() == expected.keys()
    for key, expected_value in expected.items():
        torch.testing.assert_close(hf_state[key], expected_value, rtol=0, atol=0)

    assert olmo_state.keys() == original.keys()
    for key in olmo_state:
        torch.testing.assert_close(olmo_state[key], original[key], rtol=0, atol=0)


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_5_bidirectional_roundtrips_complete_state(tied: bool):
    config = _text_config(tie_word_embeddings=tied)
    olmo_state = _olmo_state(tied=tied)

    hf_state = convert_state_to_hf(config, olmo_state)
    olmo_roundtrip = convert_state_from_hf(config, hf_state, model_type=config.model_type)

    assert olmo_roundtrip.keys() == olmo_state.keys()
    for key, expected in olmo_state.items():
        torch.testing.assert_close(olmo_roundtrip[key], expected, rtol=0, atol=0)

    hf_roundtrip = convert_qwen3_5_state_to_hf(
        config,
        convert_qwen3_5_state_from_hf(config, hf_state),
    )
    assert hf_roundtrip.keys() == hf_state.keys()
    for key, expected in hf_state.items():
        torch.testing.assert_close(hf_roundtrip[key], expected, rtol=0, atol=0)


def test_qwen3_5_multimodal_export_is_out_of_scope():
    wrapper_config = Qwen3_5Config(text_config=_text_config(tie_word_embeddings=True))

    with pytest.raises(ValueError, match="multimodal Qwen3.5 export is not supported"):
        convert_state_to_hf(wrapper_config, _olmo_state(tied=True))


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_5_strict_load(tied: bool):
    config = _text_config(tie_word_embeddings=tied)
    model = Qwen3_5ForCausalLM(config)
    hf_state = convert_state_to_hf(config, _olmo_state(tied=tied))

    incompatible = model.load_state_dict(hf_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_5_save_pretrained_and_reload(tmp_path: Path, tied: bool):
    config = _text_config(tie_word_embeddings=tied)
    hf_state = convert_state_to_hf(config, _olmo_state(tied=tied))
    model = Qwen3_5ForCausalLM(config)
    model.load_state_dict(hf_state, strict=True)

    model.save_pretrained(tmp_path)
    reloaded = Qwen3_5ForCausalLM.from_pretrained(tmp_path)
    saved_state = load_file(tmp_path / "model.safetensors")

    assert reloaded.config.model_type == "qwen3_5_text"
    expected_saved_keys = set(hf_state)
    if tied:
        # Transformers deduplicates the aliased LM head during serialization.
        expected_saved_keys.remove("lm_head.weight")
    assert saved_state.keys() == expected_saved_keys
    for key, expected in hf_state.items():
        if key in saved_state:
            assert saved_state[key].shape == expected.shape
        torch.testing.assert_close(reloaded.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen3_5_olmo_origin_low_precision_norm_roundtrip_is_exact(dtype: torch.dtype):
    config = _text_config()
    olmo_state = {key: value.to(dtype) for key, value in _olmo_state().items()}
    for key in _regular_norm_keys():
        olmo_state[key] = torch.linspace(0.75, 1.25, olmo_state[key].numel(), dtype=dtype)

    roundtrip = convert_qwen3_5_state_from_hf(
        config,
        convert_qwen3_5_state_to_hf(config, olmo_state),
    )

    for key, expected in olmo_state.items():
        torch.testing.assert_close(roundtrip[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen3_5_hf_origin_low_precision_norm_roundtrip_matches_native_arithmetic(
    dtype: torch.dtype,
):
    config = _text_config()
    hf_state = {
        key: value.to(dtype)
        for key, value in convert_qwen3_5_state_to_hf(config, _olmo_state()).items()
    }
    for key in _regular_hf_norm_keys(hf_state):
        # Include the magnitude observed in official Qwen3.5 norm tensors. BF16
        # spacing grows with magnitude, so a constant epsilon is not a valid
        # global bound for the add-one/subtract-one roundtrip.
        hf_state[key] = torch.linspace(-0.75, 5.5, hf_state[key].numel(), dtype=dtype)

    roundtrip = convert_qwen3_5_state_to_hf(
        config,
        convert_qwen3_5_state_from_hf(config, hf_state),
    )

    changed_norm_keys = set()
    for key, expected in hf_state.items():
        if key in _regular_hf_norm_keys(hf_state):
            native_arithmetic = (expected + 1.0) - 1.0
            torch.testing.assert_close(roundtrip[key], native_arithmetic, rtol=0, atol=0)
            if not torch.equal(native_arithmetic, expected):
                changed_norm_keys.add(key)
        else:
            torch.testing.assert_close(roundtrip[key], expected, rtol=0, atol=0)
    # The existing importer stores the affine transform in the source dtype, so
    # precision discarded by that intermediate representation cannot be recovered.
    assert changed_norm_keys


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("missing", KeyError, "Missing required"),
        ("extra", RuntimeError, "Some state keys were not converted"),
        ("shape", ValueError, "Unexpected shape"),
        ("value", TypeError, "to be a tensor"),
        ("wrapper", ValueError, "canonical unwrapped"),
    ],
)
def test_qwen3_5_rejects_malformed_state(mutation, error, message):
    config = _text_config()
    state = _olmo_state()
    if mutation == "missing":
        del state["blocks.0.attention.w_a.weight"]
    elif mutation == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif mutation == "shape":
        state["blocks.0.attention.w_a.weight"] = torch.ones(3, 8)
    elif mutation == "value":
        state["blocks.0.attention.w_a.weight"] = "not a tensor"  # type: ignore[assignment]
    elif mutation == "wrapper":
        state["module.unexpected.weight"] = torch.ones(1)

    with pytest.raises(error, match=message):
        convert_qwen3_5_state_to_hf(config, state)


def test_qwen3_5_rejects_unknown_or_incomplete_layer_types():
    config = _text_config()
    config.layer_types[0] = "unknown"
    with pytest.raises(ValueError, match="Unknown layer type"):
        convert_qwen3_5_state_to_hf(config, _olmo_state())

    config = _text_config()
    config.layer_types.pop()
    with pytest.raises(ValueError, match="one Qwen3.5 layer type per layer"):
        convert_qwen3_5_state_to_hf(config, _olmo_state())


def test_qwen3_5_tied_embeddings_must_agree():
    config = _text_config(tie_word_embeddings=True)
    state = _olmo_state(tied=True)
    assert state["embeddings.weight"].data_ptr() != state["lm_head.w_out.weight"].data_ptr()
    hf_state = convert_qwen3_5_state_to_hf(config, state)
    torch.testing.assert_close(
        hf_state["model.embed_tokens.weight"], hf_state["lm_head.weight"], rtol=0, atol=0
    )

    state["lm_head.w_out.weight"] = state["embeddings.weight"]
    convert_qwen3_5_state_to_hf(config, state)

    with pytest.raises(ValueError, match="ties word embeddings.*differ"):
        convert_qwen3_5_state_to_hf(config, _olmo_state(tied=False))

    state = _olmo_state(tied=True)
    state["lm_head.w_out.weight"] = state["lm_head.w_out.weight"].double()
    with pytest.raises(ValueError, match="different dtype or device"):
        convert_qwen3_5_state_to_hf(config, state)


def test_qwen3_5_tied_embeddings_support_meta_preflight():
    config = _text_config(tie_word_embeddings=True)
    expected = convert_state_to_hf(config, _olmo_state(tied=True))
    meta_state = {key: tensor.to(device="meta") for key, tensor in _olmo_state(tied=True).items()}

    actual = convert_state_to_hf(config, meta_state)

    assert actual.keys() == expected.keys()
    for key, tensor in actual.items():
        assert tensor.is_meta
        assert tensor.shape == expected[key].shape
        assert tensor.dtype == expected[key].dtype


def test_qwen3_5_tied_embeddings_reject_mixed_meta_state():
    config = _text_config(tie_word_embeddings=True)
    state = {key: tensor.to(device="meta") for key, tensor in _olmo_state(tied=True).items()}
    state["lm_head.w_out.weight"] = torch.empty_like(state["lm_head.w_out.weight"], device="cpu")

    with pytest.raises(ValueError, match="different dtype or device"):
        convert_qwen3_5_state_to_hf(config, state)


def test_qwen3_5_does_not_implicitly_resize_vocabulary():
    config = _text_config()
    state = _olmo_state()
    state["embeddings.weight"] = torch.ones(17, 8)
    state["lm_head.w_out.weight"] = torch.ones(17, 8)

    with pytest.raises(ValueError, match=r"expected \(16, 8\), found \(17, 8\)"):
        convert_qwen3_5_state_to_hf(config, state)


def test_qwen3_5_accepts_non_contiguous_query_and_gate_weights():
    state = _olmo_state()
    for key in ("blocks.1.attention.w_q.weight", "blocks.1.attention.w_g.weight"):
        state[key] = state[key].T.contiguous().T
        assert not state[key].is_contiguous()

    hf_state = convert_qwen3_5_state_to_hf(_text_config(), state)

    assert hf_state["model.layers.1.self_attn.q_proj.weight"].shape == (8, 8)
