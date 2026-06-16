import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig, TransformerBlockType, TransformerType


def _layer_norm_config() -> LayerNormConfig:
    return LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )


def _attention_config() -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=2,
        n_kv_heads=2,
        bias=False,
        use_flash=False,
        dtype=DType.float32,
    )


def _shared_only_block_config() -> OLMoDDPTransformerBlockConfig:
    layer_norm = _layer_norm_config()
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=_attention_config(),
        attention_norm=layer_norm,
        feed_forward_norm=layer_norm,
        routed_experts=None,
        routed_experts_router=None,
        shared_experts=SharedExpertsConfig(
            d_model=16,
            hidden_size=32,
            num_experts=1,
            bias=False,
            dtype=DType.float32,
        ),
        shared_experts_router=None,
    )


def _routed_block_config() -> OLMoDDPTransformerBlockConfig:
    layer_norm = _layer_norm_config()
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=_attention_config(),
        attention_norm=layer_norm,
        feed_forward_norm=layer_norm,
        routed_experts=RoutedExpertsConfig(
            d_model=16,
            hidden_size=32,
            num_experts=4,
            bias=False,
            dtype=DType.float32,
        ),
        routed_experts_router=MoERouterConfigV2(
            d_model=16,
            num_experts=4,
            top_k=1,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=False,
            lb_loss_weight=None,
            z_loss_weight=None,
            dtype=DType.float32,
        ),
        shared_experts=None,
        shared_experts_router=None,
        ep_no_sync=True,
    )


def test_shared_only_ddp_block_is_dense_stack_block():
    config = _shared_only_block_config()

    block = config.build(d_model=16, block_idx=0, n_layers=1, init_device="cpu")

    assert not block.is_moe
    assert block.is_shared_only
    assert not block.has_routed_experts
    assert block.has_shared_experts

    x = torch.randn(2, 4, 16)
    out = block(x)
    assert out.shape == x.shape


def test_ddp_model_topology_iterators_split_shared_and_routed_blocks():
    model_config = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=16,
        vocab_size=64,
        n_layers=2,
        lm_head=LMHeadConfig(bias=False, dtype=DType.float32),
        block=_routed_block_config(),
        block_overrides={0: _shared_only_block_config()},
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
    )
    model = model_config.build(init_device="cpu")

    assert [key for key, _ in model.named_ddp_blocks()] == ["0", "1"]
    assert [key for key, _ in model.named_shared_blocks()] == ["0"]
    assert [key for key, _ in model.named_routed_blocks()] == ["1"]
    assert [key for key, _ in model.named_ep_no_sync_blocks()] == ["1"]
    assert model.count_ep_no_sync_blocks() == 1
    assert model.count_non_rowwise_ep_no_sync_blocks() == 1
    assert model.has_routed_blocks
