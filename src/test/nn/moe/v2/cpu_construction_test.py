"""
CPU construction test for the fused MoE-v2 model.

The EP/TBO forward path is GPU-only (grouped-GEMM / symm-mem / NCCL-RMA kernels), but *construction*
should work on CPU: the per-block CUDA comm-overlap events are allocated lazily (in ``apply_ep``,
which only runs on the GPU EP path) rather than in ``__init__``. This lets configs be built,
inspected, and FLOP-accounted without a GPU.
"""

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.block import (
    MoEFusedV2TransformerBlock,
    MoEFusedV2TransformerBlockConfig,
)
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockType,
    TransformerType,
)


def _build_model_config(*, d_model: int = 64, n_layers: int = 2) -> MoEFusedV2TransformerConfig:
    dtype = DType.float32
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return MoEFusedV2TransformerConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=n_layers,
        name=TransformerType.moe_fused_v2,
        block=MoEFusedV2TransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default, n_heads=4, bias=False, use_flash=False, dtype=dtype
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model, num_experts=4, top_k=2, dtype=dtype
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def test_moe_v2_model_constructs_on_cpu():
    model = _build_model_config(n_layers=2).build(init_device="cpu")

    assert len(model.blocks) == 2
    assert any(p.numel() > 0 for p in model.parameters())

    # On a CUDA-less machine the comm-overlap events stay None (allocating torch.cuda.Event() would
    # raise) — this is what lets construction work on CPU. On a GPU host they're allocated up front.
    if not torch.cuda.is_available():
        for block in model.blocks.values():
            assert isinstance(block, MoEFusedV2TransformerBlock)
            assert block._dtoh_event is None

    # Model-level FLOP accounting works without a GPU / without building the optimizer.
    assert model.num_flops_per_token(seq_len=512) > 0
