import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from examples.huggingface.convert_checkpoint_to_hf_hybrid import (
    ATTN_KEY_MAP,
    GDN_KEY_MAP,
    SHARED_KEY_MAP,
    build_hf_config,
    convert_checkpoint_to_hf,
    convert_state_dict,
    get_layer_types,
    is_gdn_layer,
)
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNetConfig
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock, TransformerBlock
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.model import Transformer


def _can_build_gdn() -> bool:
    """Check that fla *and* its runtime dependencies (triton) are importable."""
    if not has_fla():
        return False
    try:
        from fla.modules import FusedRMSNormGated  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


requires_fla = pytest.mark.skipif(
    not _can_build_gdn(), reason="flash-linear-attention (fla) or triton not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


@pytest.fixture
def hybrid_model_config(tokenizer_config: TokenizerConfig) -> TransformerConfig:
    """
    Build a small hybrid model config:  4 layers with pattern [gdn, gdn, gdn, attn].
    Based on the OLMo2-190M config but with named blocks and a GDN block.
    """
    config = TransformerConfig.olmo2_190M(
        tokenizer_config.padded_vocab_size(),
        n_layers=4,
        n_heads=12,
        attn_backend=AttentionBackendName.torch,
    )
    assert isinstance(config.block, TransformerBlockConfig)

    attn_block = config.block
    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=12,
            head_dim=config.d_model // 12,
            allow_neg_eigval=True,
        ),
    )

    config.block = {"gdn": gdn_block, "attn": attn_block}
    config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
    return config


@pytest.fixture
def hybrid_model(hybrid_model_config: TransformerConfig) -> Transformer:
    return hybrid_model_config.build()


@pytest.fixture
def olmo_core_model_path(
    tmp_path: Path, hybrid_model_config: TransformerConfig, hybrid_model: Transformer
) -> Iterator[Path]:
    """Save a hybrid model checkpoint and return the checkpoint directory."""
    model_path = tmp_path / "olmo_core_hybrid"
    save_model_and_optim_state(model_path / "model_and_optim", hybrid_model)
    yield model_path
    shutil.rmtree(model_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


@requires_fla
def test_is_gdn_layer(hybrid_model: Transformer):
    blocks = list(hybrid_model.blocks.values())
    assert is_gdn_layer(blocks[0]) is True
    assert is_gdn_layer(blocks[1]) is True
    assert is_gdn_layer(blocks[2]) is True
    assert is_gdn_layer(blocks[3]) is False


@requires_fla
def test_get_layer_types(hybrid_model: Transformer):
    layer_types = get_layer_types(hybrid_model)
    assert layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


@requires_fla
def test_convert_state_dict_no_missing_keys(hybrid_model: Transformer):
    """Every key in the OLMo-core state dict should be mapped to an HF key."""
    state_dict = {k: v for k, v in hybrid_model.named_parameters()}
    layer_types = get_layer_types(hybrid_model)

    hf_state = convert_state_dict(state_dict, layer_types)

    # No keys should have been dropped.
    assert len(hf_state) == len(state_dict)


@requires_fla
def test_convert_state_dict_gdn_keys(hybrid_model: Transformer):
    """GDN layers should get ``linear_attn.*`` HF keys."""
    state_dict = {k: v for k, v in hybrid_model.named_parameters()}
    layer_types = get_layer_types(hybrid_model)

    hf_state = convert_state_dict(state_dict, layer_types)

    gdn_layer_keys = [k for k in hf_state if "layers.0." in k]
    assert any("linear_attn." in k for k in gdn_layer_keys)
    assert not any("self_attn." in k for k in gdn_layer_keys)


@requires_fla
def test_convert_state_dict_attn_keys(hybrid_model: Transformer):
    """Attention layers should get ``self_attn.*`` HF keys."""
    state_dict = {k: v for k, v in hybrid_model.named_parameters()}
    layer_types = get_layer_types(hybrid_model)

    hf_state = convert_state_dict(state_dict, layer_types)

    attn_layer_keys = [k for k in hf_state if "layers.3." in k]
    assert any("self_attn." in k for k in attn_layer_keys)
    assert not any("linear_attn." in k for k in attn_layer_keys)


@requires_fla
def test_convert_state_dict_shared_keys(hybrid_model: Transformer):
    """Non-block keys should be correctly mapped."""
    state_dict = {k: v for k, v in hybrid_model.named_parameters()}
    layer_types = get_layer_types(hybrid_model)

    hf_state = convert_state_dict(state_dict, layer_types)

    assert "model.embed_tokens.weight" in hf_state
    assert "model.norm.weight" in hf_state
    assert "lm_head.weight" in hf_state


@requires_fla
def test_convert_state_dict_preserves_values(hybrid_model: Transformer):
    """Tensor values should be preserved (not cloned or modified)."""
    state_dict = {k: v for k, v in hybrid_model.named_parameters()}
    layer_types = get_layer_types(hybrid_model)

    hf_state = convert_state_dict(state_dict, layer_types)

    # Check a GDN weight.
    assert torch.equal(
        hf_state["model.layers.0.linear_attn.q_proj.weight"],
        state_dict["blocks.0.attention.w_q.weight"],
    )
    # Check an attention weight.
    assert torch.equal(
        hf_state["model.layers.3.self_attn.q_proj.weight"],
        state_dict["blocks.3.attention.w_q.weight"],
    )


# ---------------------------------------------------------------------------
# HF config tests
# ---------------------------------------------------------------------------


@requires_fla
def test_build_hf_config_model_type(hybrid_model: Transformer):
    layer_types = get_layer_types(hybrid_model)
    hf_config = build_hf_config(hybrid_model, layer_types, max_seq_len=256)

    assert hf_config["model_type"] == "olmo_hybrid"
    assert hf_config["architectures"] == ["OlmoHybridForCausalLM"]


@requires_fla
def test_build_hf_config_layer_types(hybrid_model: Transformer):
    layer_types = get_layer_types(hybrid_model)
    hf_config = build_hf_config(hybrid_model, layer_types, max_seq_len=256)

    assert hf_config["layer_types"] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]


@requires_fla
def test_build_hf_config_standard_fields(
    hybrid_model: Transformer, hybrid_model_config: TransformerConfig
):
    layer_types = get_layer_types(hybrid_model)
    hf_config = build_hf_config(hybrid_model, layer_types, max_seq_len=256)

    assert hf_config["hidden_size"] == hybrid_model_config.d_model
    assert hf_config["num_hidden_layers"] == 4
    assert hf_config["num_attention_heads"] == 12
    assert hf_config["max_position_embeddings"] == 256
    assert hf_config["hidden_act"] == "silu"
    assert hf_config["tie_word_embeddings"] is False


@requires_fla
def test_build_hf_config_gdn_fields(hybrid_model: Transformer):
    layer_types = get_layer_types(hybrid_model)
    hf_config = build_hf_config(hybrid_model, layer_types, max_seq_len=256)

    # GDN-specific fields should be extracted from the first GDN block.
    gdn: GatedDeltaNet = list(hybrid_model.blocks.values())[0].attention
    assert hf_config["linear_num_key_heads"] == gdn.n_heads
    assert hf_config["linear_num_value_heads"] == gdn.n_v_heads
    assert hf_config["linear_key_head_dim"] == gdn.head_k_dim
    assert hf_config["linear_value_head_dim"] == gdn.head_v_dim
    assert hf_config["linear_conv_kernel_dim"] == gdn.conv_size
    assert hf_config["linear_allow_neg_eigval"] == gdn.allow_neg_eigval


@requires_fla
def test_build_hf_config_rope(hybrid_model: Transformer):
    layer_types = get_layer_types(hybrid_model)
    hf_config = build_hf_config(hybrid_model, layer_types, max_seq_len=256)

    assert "rope_parameters" in hf_config
    assert hf_config["rope_parameters"]["rope_theta"] == 500_000.0
    assert hf_config["rope_parameters"]["rope_type"] == "default"


# ---------------------------------------------------------------------------
# End-to-end conversion test
# ---------------------------------------------------------------------------


@requires_fla
def test_convert_checkpoint_to_hf_produces_valid_output(
    tmp_path: Path,
    olmo_core_model_path: Path,
    hybrid_model_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """
    Full end-to-end test: save an OLMo-core hybrid checkpoint, convert it to HF format,
    and verify the output files and config are correct.
    """
    output_dir = tmp_path / "hf-output-hybrid"

    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path,
        output_path=output_dir,
        transformer_config_dict=hybrid_model_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
    )

    # Check that the output files exist.
    assert (output_dir / "config.json").exists()
    assert (output_dir / "model.safetensors").exists()

    # Load and validate config.json.
    with open(output_dir / "config.json") as f:
        config = json.load(f)

    assert config["model_type"] == "olmo_hybrid"
    assert config["architectures"] == ["OlmoHybridForCausalLM"]
    assert config["num_hidden_layers"] == 4
    assert config["hidden_size"] == hybrid_model_config.d_model
    assert config["layer_types"] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert config["max_position_embeddings"] == 256

    # Load and validate the state dict.
    hf_state = load_file(output_dir / "model.safetensors")

    # Should have the shared keys.
    assert "model.embed_tokens.weight" in hf_state
    assert "model.norm.weight" in hf_state
    assert "lm_head.weight" in hf_state

    # GDN layer 0 should have linear_attn keys.
    assert "model.layers.0.linear_attn.q_proj.weight" in hf_state
    assert "model.layers.0.linear_attn.A_log" in hf_state
    assert "model.layers.0.input_layernorm.weight" in hf_state

    # Attention layer 3 should have self_attn keys.
    assert "model.layers.3.self_attn.q_proj.weight" in hf_state
    assert "model.layers.3.self_attn.q_norm.weight" in hf_state
    assert "model.layers.3.post_attention_layernorm.weight" in hf_state
    assert "model.layers.3.post_feedforward_layernorm.weight" in hf_state

    shutil.rmtree(output_dir)


@requires_fla
def test_convert_checkpoint_to_hf_weights_match_original(
    tmp_path: Path,
    olmo_core_model_path: Path,
    hybrid_model_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """
    Verify that the converted HF weights are numerically identical to the original
    OLMo-core weights (after key remapping and dtype cast).
    """
    output_dir = tmp_path / "hf-output-hybrid-match"

    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path,
        output_path=output_dir,
        transformer_config_dict=hybrid_model_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
        dtype=None,  # Keep original dtype to ensure exact match.
    )

    # Reload the original model.
    original_model = hybrid_model_config.build()
    load_model_and_optim_state(olmo_core_model_path / "model_and_optim", model=original_model)
    original_state = {k: v for k, v in original_model.named_parameters()}

    # Load converted weights.
    hf_state = load_file(output_dir / "model.safetensors")

    # Spot-check: shared keys.
    assert torch.equal(hf_state["model.embed_tokens.weight"], original_state["embeddings.weight"])
    assert torch.equal(hf_state["model.norm.weight"], original_state["lm_head.norm.weight"])
    assert torch.equal(hf_state["lm_head.weight"], original_state["lm_head.w_out.weight"])

    # Spot-check: GDN layer 0.
    assert torch.equal(
        hf_state["model.layers.0.linear_attn.q_proj.weight"],
        original_state["blocks.0.attention.w_q.weight"],
    )
    assert torch.equal(
        hf_state["model.layers.0.linear_attn.A_log"],
        original_state["blocks.0.attention.A_log"],
    )
    assert torch.equal(
        hf_state["model.layers.0.mlp.gate_proj.weight"],
        original_state["blocks.0.feed_forward.w1.weight"],
    )

    # Spot-check: attention layer 3.
    assert torch.equal(
        hf_state["model.layers.3.self_attn.q_proj.weight"],
        original_state["blocks.3.attention.w_q.weight"],
    )
    assert torch.equal(
        hf_state["model.layers.3.self_attn.q_norm.weight"],
        original_state["blocks.3.attention.q_norm.weight"],
    )
    assert torch.equal(
        hf_state["model.layers.3.mlp.down_proj.weight"],
        original_state["blocks.3.feed_forward.w2.weight"],
    )

    shutil.rmtree(output_dir)


@requires_fla
def test_convert_checkpoint_to_hf_vocab_truncation(
    tmp_path: Path,
    olmo_core_model_path: Path,
    hybrid_model_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """Embedding and LM head weights should be truncated to the tokenizer vocab size."""
    output_dir = tmp_path / "hf-output-hybrid-vocab"

    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path,
        output_path=output_dir,
        transformer_config_dict=hybrid_model_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
    )

    hf_state = load_file(output_dir / "model.safetensors")
    vocab_size = tokenizer_config.vocab_size

    # The padded vocab size is larger than the actual tokenizer vocab size.
    # Both embeddings and lm_head should be truncated to the actual vocab size.
    assert hf_state["model.embed_tokens.weight"].shape[0] == vocab_size
    assert hf_state["lm_head.weight"].shape[0] == vocab_size

    shutil.rmtree(output_dir)


@requires_fla
def test_convert_checkpoint_to_hf_dtype_cast(
    tmp_path: Path,
    olmo_core_model_path: Path,
    hybrid_model_config: TransformerConfig,
    tokenizer_config: TokenizerConfig,
):
    """Weights should be cast to the requested dtype."""
    from olmo_core.config import DType

    output_dir = tmp_path / "hf-output-hybrid-dtype"

    convert_checkpoint_to_hf(
        original_checkpoint_path=olmo_core_model_path,
        output_path=output_dir,
        transformer_config_dict=hybrid_model_config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
        dtype=DType.bfloat16,
    )

    hf_state = load_file(output_dir / "model.safetensors")
    for key, tensor in hf_state.items():
        assert tensor.dtype == torch.bfloat16, f"{key} has dtype {tensor.dtype}, expected bfloat16"

    shutil.rmtree(output_dir)


def test_convert_state_dict_with_mock_data():
    """Test convert_state_dict with synthetic keys (no model build required)."""
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]

    mock_state = {
        # Shared keys
        "embeddings.weight": torch.zeros(1),
        "lm_head.norm.weight": torch.zeros(1),
        "lm_head.w_out.weight": torch.zeros(1),
        # GDN layer 0
        "blocks.0.attention.w_q.weight": torch.ones(1),
        "blocks.0.attention.A_log": torch.ones(1) * 2,
        "blocks.0.attention_norm.weight": torch.zeros(1),
        "blocks.0.feed_forward_norm.weight": torch.zeros(1),
        "blocks.0.feed_forward.w1.weight": torch.zeros(1),
        "blocks.0.feed_forward.w2.weight": torch.zeros(1),
        "blocks.0.feed_forward.w3.weight": torch.zeros(1),
        # Attention layer 3
        "blocks.3.attention.w_q.weight": torch.ones(1) * 3,
        "blocks.3.attention.q_norm.weight": torch.zeros(1),
        "blocks.3.attention_norm.weight": torch.zeros(1),
        "blocks.3.feed_forward_norm.weight": torch.zeros(1),
        "blocks.3.feed_forward.w1.weight": torch.zeros(1),
        "blocks.3.feed_forward.w2.weight": torch.zeros(1),
        "blocks.3.feed_forward.w3.weight": torch.zeros(1),
    }

    hf = convert_state_dict(mock_state, layer_types)

    # No keys dropped.
    assert len(hf) == len(mock_state)

    # GDN layer 0: should use linear_attn prefix.
    assert torch.equal(hf["model.layers.0.linear_attn.q_proj.weight"], torch.ones(1))
    assert torch.equal(hf["model.layers.0.linear_attn.A_log"], torch.ones(1) * 2)
    assert "model.layers.0.input_layernorm.weight" in hf

    # Attention layer 3: should use self_attn prefix.
    assert torch.equal(hf["model.layers.3.self_attn.q_proj.weight"], torch.ones(1) * 3)
    assert "model.layers.3.self_attn.q_norm.weight" in hf
    assert "model.layers.3.post_attention_layernorm.weight" in hf
    assert "model.layers.3.post_feedforward_layernorm.weight" in hf


def test_key_maps_are_consistent():
    """The GDN and attention key maps should map the same OLMo-core MLP suffixes identically."""
    # MLP suffixes appear in both maps and should map to the same HF suffix.
    gdn_mlp = {k: v for k, v in GDN_KEY_MAP.items() if k.startswith("feed_forward.")}
    attn_mlp = {k: v for k, v in ATTN_KEY_MAP.items() if k.startswith("feed_forward.")}
    assert gdn_mlp == attn_mlp

    # Sequence mixer keys should be completely disjoint (linear_attn vs self_attn).
    gdn_mixer = {v for k, v in GDN_KEY_MAP.items() if k.startswith("attention.")}
    attn_mixer = {v for k, v in ATTN_KEY_MAP.items() if k.startswith("attention.")}
    assert gdn_mixer.isdisjoint(attn_mixer), (
        f"Overlapping mixer HF keys: {gdn_mixer & attn_mixer}"
    )
