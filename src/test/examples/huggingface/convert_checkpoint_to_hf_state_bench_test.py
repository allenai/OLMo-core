"""HF conversion tests for the StateBench 275M model variants.

The StateBench suite (``src/scripts/train/ladder/state_bench.py``) trains five variants of
the hybrid-small-suite backbone (peri-norm blocks, gated attention with head QK-norm,
embed scale + embedding norm): two pure transformers (with/without RoPE), a GDN/attention
hybrid, and two pure-GDN models. All five must convert through the HF ``olmo3_5_hybrid``
format — the dense OLMo2/OLMo3 configs cannot express the backbone's features.

These tests build small versions of each variant and check detection, config emission,
and state-dict mapping. The attention-only variants run on CPU; GDN-bearing variants
require flash-linear-attention.
"""

import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.hf import convert_checkpoint_to_hf
from olmo_core.nn.hf.config import (
    get_hybrid_hf_config,
    get_hybrid_layer_types,
    requires_hybrid_hf_format,
)
from olmo_core.nn.hf.convert import convert_hybrid_state_to_hf
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)


def _can_build_gdn() -> bool:
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

MODEL_TYPES = ["transformer-rope", "transformer-nope", "hybrid", "gdn-sdp", "gdn-full"]
GDN_MODEL_TYPES = {"hybrid", "gdn-sdp", "gdn-full"}

N_LAYERS = 4
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = 16


def fla_param(model_type: str):
    """Mark GDN-bearing variants as requiring fla."""
    if model_type in GDN_MODEL_TYPES:
        return pytest.param(model_type, marks=requires_fla)
    return pytest.param(model_type)


PARAMETRIZED_MODEL_TYPES = [fla_param(mt) for mt in MODEL_TYPES]


def state_bench_model_config(model_type: str, vocab_size: int) -> TransformerConfig:
    """A miniature of ``StateBenchModelConfigurator.configure_model``."""
    dtype = DType.float32
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    feed_forward = FeedForwardConfig(
        hidden_size=D_MODEL * 2, bias=False, dtype=dtype, activation=ActivationFunction.silu
    )

    def attention_block(rope: RoPEConfig | None) -> TransformerBlockConfig:
        return TransformerBlockConfig(
            name=TransformerBlockType.peri_norm,
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=N_HEADS,
                n_kv_heads=N_HEADS,
                head_dim=HEAD_DIM,
                bias=False,
                rope=rope,
                gate=GateConfig(granularity=GateGranularity.elementwise, full_precision=True),
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                backend=AttentionBackendName.torch,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            layer_norm=layer_norm,
        )

    def gdn_block(allow_neg_eigval: bool) -> TransformerBlockConfig:
        return TransformerBlockConfig(
            name=TransformerBlockType.peri_norm,
            sequence_mixer=GatedDeltaNetConfig(
                n_heads=N_HEADS,
                n_v_heads=N_HEADS,
                head_dim=HEAD_DIM,
                expand_v=2.0,
                allow_neg_eigval=allow_neg_eigval,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            layer_norm=layer_norm,
        )

    block: TransformerBlockConfig
    block_overrides: dict[int, TransformerBlockConfig] | None = None
    if model_type == "transformer-rope":
        block = attention_block(RoPEConfig())
    elif model_type == "transformer-nope":
        block = attention_block(None)
    elif model_type == "hybrid":
        block = gdn_block(allow_neg_eigval=True)
        block_overrides = {N_LAYERS - 1: attention_block(None)}
    elif model_type == "gdn-sdp":
        block = gdn_block(allow_neg_eigval=False)
    elif model_type == "gdn-full":
        block = gdn_block(allow_neg_eigval=True)
    else:
        raise ValueError(model_type)

    return TransformerConfig(
        d_model=D_MODEL,
        vocab_size=vocab_size,
        n_layers=N_LAYERS,
        block=block,
        lm_head=LMHeadConfig(
            loss_implementation=LMLossImplementation.default,
            layer_norm=layer_norm,
            bias=False,
            dtype=dtype,
        ),
        dtype=dtype,
        block_overrides=block_overrides,
        embed_scale=math.sqrt(D_MODEL),
        embedding_norm=LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False),
    )


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig.dolma2()


def build_model(model_type: str, tokenizer_config: TokenizerConfig):
    config = state_bench_model_config(model_type, tokenizer_config.padded_vocab_size())
    model = config.build()
    model.init_weights()
    return config, model


EXPECTED_LAYER_TYPES = {
    "transformer-rope": ["full_attention"] * N_LAYERS,
    "transformer-nope": ["full_attention"] * N_LAYERS,
    "hybrid": ["linear_attention"] * (N_LAYERS - 1) + ["full_attention"],
    "gdn-sdp": ["linear_attention"] * N_LAYERS,
    "gdn-full": ["linear_attention"] * N_LAYERS,
}


@pytest.mark.parametrize("model_type", PARAMETRIZED_MODEL_TYPES)
def test_requires_hybrid_hf_format(model_type: str, tokenizer_config: TokenizerConfig):
    """Every StateBench variant must route through the hybrid HF format."""
    _, model = build_model(model_type, tokenizer_config)
    assert requires_hybrid_hf_format(model)


def test_plain_olmo2_does_not_require_hybrid_format(tokenizer_config: TokenizerConfig):
    config = TransformerConfig.olmo2_190M(
        tokenizer_config.padded_vocab_size(),
        n_layers=2,
        n_heads=4,
        attn_backend=AttentionBackendName.torch,
    )
    model = config.build()
    assert not requires_hybrid_hf_format(model)


@pytest.mark.parametrize("model_type", PARAMETRIZED_MODEL_TYPES)
def test_layer_types(model_type: str, tokenizer_config: TokenizerConfig):
    _, model = build_model(model_type, tokenizer_config)
    assert get_hybrid_layer_types(model) == EXPECTED_LAYER_TYPES[model_type]


@pytest.mark.parametrize("model_type", PARAMETRIZED_MODEL_TYPES)
def test_hybrid_hf_config(model_type: str, tokenizer_config: TokenizerConfig):
    """The hybrid config must be buildable for every variant, including single-mixer models."""
    _, model = build_model(model_type, tokenizer_config)
    layer_types = get_hybrid_layer_types(model)
    hf_config = get_hybrid_hf_config(model, layer_types, max_seq_len=256)

    assert hf_config["model_type"] == "olmo3_5_hybrid"
    assert hf_config["layer_types"] == EXPECTED_LAYER_TYPES[model_type]
    assert hf_config["num_hidden_layers"] == N_LAYERS
    assert hf_config["hidden_size"] == D_MODEL
    assert hf_config["use_peri_norm"] is True
    assert hf_config["use_embedding_norm"] is True
    assert hf_config["embed_scale"] == pytest.approx(math.sqrt(D_MODEL))

    # RoPE is present only for the RoPE transformer.
    if model_type == "transformer-rope":
        assert hf_config["rope_parameters"]["rope_theta"] > 0
    else:
        assert "rope_parameters" not in hf_config
        assert hf_config["rope_theta"] is None

    # Attention-derived fields: real for attention-bearing variants, inert otherwise.
    assert hf_config["num_attention_heads"] == N_HEADS
    assert hf_config["head_dim"] == HEAD_DIM
    if model_type in ("transformer-rope", "transformer-nope", "hybrid"):
        assert hf_config["use_attention_gate"] is True
        assert hf_config["use_head_qk_norm"] is True
    else:
        assert hf_config["use_attention_gate"] is False
        assert hf_config["use_head_qk_norm"] is False

    # GDN-derived fields: real for GDN-bearing variants, inert otherwise.
    assert hf_config["linear_num_key_heads"] == N_HEADS
    if model_type == "gdn-sdp":
        assert hf_config["linear_allow_neg_eigval"] is False
    elif model_type in ("gdn-full", "hybrid"):
        assert hf_config["linear_allow_neg_eigval"] is True
    if model_type in GDN_MODEL_TYPES:
        assert hf_config["linear_value_head_dim"] == 2 * HEAD_DIM  # expand_v=2.0


@pytest.mark.parametrize("model_type", PARAMETRIZED_MODEL_TYPES)
def test_state_dict_maps_completely(model_type: str, tokenizer_config: TokenizerConfig):
    """Every parameter of every variant must map to an HF key (peri-norm maps)."""
    _, model = build_model(model_type, tokenizer_config)
    layer_types = get_hybrid_layer_types(model)
    state_dict = {k: v for k, v in model.named_parameters()}

    hf_state = convert_hybrid_state_to_hf(state_dict, layer_types, peri_norm=True)

    assert len(hf_state) == len(state_dict)
    assert "model.embed_tokens.weight" in hf_state
    assert "model.embedding_norm.weight" in hf_state
    assert "model.norm.weight" in hf_state
    assert "lm_head.weight" in hf_state
    if model_type in GDN_MODEL_TYPES:
        # The olmo3_5_hybrid modeling code declares o_proj/o_norm; the out_proj/norm
        # spellings load as randomly initialized parameters.
        assert "model.layers.0.linear_attn.o_proj.weight" in hf_state
        assert "model.layers.0.linear_attn.o_norm.weight" in hf_state
    # Peri-norm blocks emit all four per-block norms.
    for norm in (
        "pre_attention_norm",
        "post_attention_layernorm",
        "pre_feedforward_norm",
        "post_feedforward_layernorm",
    ):
        assert f"model.layers.0.{norm}.weight" in hf_state


@pytest.mark.parametrize("model_type", ["transformer-rope", "transformer-nope"])
def test_end_to_end_transformer_conversion(
    model_type: str, tmp_path: Path, tokenizer_config: TokenizerConfig
):
    """
    Full conversion of the attention-only variants (CPU-runnable): the converter must
    route them through the hybrid format, not the dense OLMo2/OLMo3 path.
    """
    config, model = build_model(model_type, tokenizer_config)
    checkpoint_dir = tmp_path / "olmo_core"
    save_model_and_optim_state(checkpoint_dir / "model_and_optim", model)

    output_dir = tmp_path / "hf-output"
    convert_checkpoint_to_hf(
        original_checkpoint_path=checkpoint_dir,
        output_path=output_dir,
        transformer_config_dict=config.as_config_dict(),
        tokenizer_config_dict=tokenizer_config.as_config_dict(),
        max_sequence_length=256,
        validate=False,
    )

    with open(output_dir / "config.json") as f:
        hf_config = json.load(f)
    assert hf_config["model_type"] == "olmo3_5_hybrid"
    assert hf_config["layer_types"] == ["full_attention"] * N_LAYERS

    hf_state = load_file(output_dir / "model.safetensors")
    assert "model.layers.0.self_attn.q_proj.weight" in hf_state
    assert "model.layers.0.self_attn.g_proj.weight" in hf_state
    assert "model.layers.0.self_attn.q_norm.weight" in hf_state
    assert "model.layers.0.pre_attention_norm.weight" in hf_state
    assert "model.embedding_norm.weight" in hf_state
    assert torch.equal(
        hf_state["model.layers.0.self_attn.q_proj.weight"],
        model.state_dict()["blocks.0.attention.w_q.weight"],
    )

    shutil.rmtree(output_dir)
