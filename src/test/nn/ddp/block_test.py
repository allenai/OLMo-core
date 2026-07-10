"""Tests for ``OLMoDDPTransformerBlock`` config accounting and construction."""

import pytest

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import TransformerBlockType

D_MODEL = 64


def _block_config(*, use_peri_norm: bool = False) -> OLMoDDPTransformerBlockConfig:
    dtype = DType.float32
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        attention=AttentionConfig(
            name=AttentionType.default, n_heads=4, bias=False, use_flash=False, dtype=dtype
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=D_MODEL, hidden_size=128, num_experts=4, bias=False, dtype=dtype
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL, num_experts=4, top_k=2, dtype=dtype
        ),
        shared_experts=None,
        layer_norm=layer_norm,
        use_peri_norm=use_peri_norm,
    )


def _build_block(config: OLMoDDPTransformerBlockConfig) -> OLMoDDPTransformerBlock:
    block = config.build(d_model=D_MODEL, block_idx=0, n_layers=1, init_device="cpu")
    assert isinstance(block, OLMoDDPTransformerBlock)
    return block


@pytest.mark.parametrize("use_peri_norm", [False, True])
def test_block_num_params_matches_built_block(use_peri_norm: bool):
    config = _block_config(use_peri_norm=use_peri_norm)
    block = _build_block(config)
    actual = sum(p.numel() for p in block.parameters())
    assert config.num_params(D_MODEL) == actual


def test_block_num_active_params_below_total():
    # Only top_k of num_experts routed experts are active per token, so active < total.
    config = _block_config()
    assert 0 < config.num_active_params(D_MODEL) < config.num_params(D_MODEL)


def test_block_flops_positive():
    config = _block_config()
    assert config.flops_per_seq(D_MODEL, seqlen=512) > 0
    assert _build_block(config).num_flops_per_token(seq_len=512) > 0


def test_block_parallelism_disabled_by_default():
    block = _build_block(_block_config())
    assert block.is_moe
    assert not block.ep_enabled
    assert not block.tp_enabled
