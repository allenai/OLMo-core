import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import TransformerBlockType


def test_shared_only_ddp_block_is_dense_stack_block():
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    config = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            use_flash=False,
            dtype=DType.float32,
        ),
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

    block = config.build(d_model=16, block_idx=0, n_layers=1, init_device="cpu")

    assert not block.is_moe
    assert block.is_shared_only
    assert not block.has_routed_experts
    assert block.has_shared_experts

    x = torch.randn(2, 4, 16)
    out = block(x)
    assert out.shape == x.shape
