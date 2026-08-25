import pytest
from transformers import Olmo2Config

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

try:
    from transformers import FlexOlmoConfig  # type: ignore
except ImportError:
    FlexOlmoConfig = None


def test_get_hf_config():
    vocab_size = 4096
    model_config = TransformerConfig.olmo2_190M(vocab_size)
    model = model_config.build()

    hf_config = get_hf_config(model)
    assert isinstance(hf_config, Olmo2Config)
    assert hf_config.hidden_size == model_config.d_model
    assert hf_config.intermediate_size == 3072
    assert hf_config.num_hidden_layers == model_config.n_layers


def test_get_hf_config_default_block():
    vocab_size = 4096
    model_config = TransformerConfig.llama2_271M(vocab_size)
    model = model_config.build()

    with pytest.raises(NotImplementedError):
        get_hf_config(model)


def test_get_hf_config_moe():
    vocab_size = 4096
    model_config = TransformerConfig.smallmoe(vocab_size)
    model = model_config.build()

    if FlexOlmoConfig is None:
        pytest.skip("The installed transformers version does not support FlexOlmo")

    hf_config = get_hf_config(model)
    assert isinstance(hf_config, FlexOlmoConfig)
    assert hf_config.hidden_size == model_config.d_model
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    assert hf_config.intermediate_size == model_config.block.feed_forward_moe.hidden_size
    assert hf_config.num_hidden_layers == model_config.n_layers


def test_get_hf_config_legacy_ddp_peri_norm_with_shared_dense_layer():
    d_model = 16
    dtype = DType.float32
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)

    def attention() -> AttentionConfig:
        return AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            bias=False,
            rope=RoPEConfig(),
            qk_norm=norm,
            use_head_qk_norm=True,
            use_flash=False,
            dtype=dtype,
        )

    sparse_block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=attention(),
        layer_norm=norm,
        routed_experts=RoutedExpertsConfig(
            d_model=d_model, hidden_size=8, num_experts=4, bias=False, dtype=dtype
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=d_model, num_experts=4, top_k=2, dtype=dtype
        ),
        shared_experts=SharedExpertsConfig(
            d_model=d_model, hidden_size=4, num_experts=1, bias=False, dtype=dtype
        ),
        use_peri_norm=True,
    )
    dense_block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=attention(),
        layer_norm=norm,
        shared_experts=SharedExpertsConfig(
            d_model=d_model, hidden_size=32, num_experts=1, bias=False, dtype=dtype
        ),
        use_peri_norm=True,
    )
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=d_model,
        vocab_size=64,
        n_layers=2,
        block=sparse_block,
        block_overrides={0: dense_block},
        lm_head=LMHeadConfig(layer_norm=norm, bias=False, dtype=dtype),
        embedding_norm=norm,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    ).build(init_device="cpu")

    hf_config = get_hf_config(model)

    assert hf_config.dense_layers_indices == [0]
    assert hf_config.dense_layers_use_shared_expert is True
    assert hf_config.dense_mlp_intermediate_size == 32
    assert hf_config.use_peri_ln is True
