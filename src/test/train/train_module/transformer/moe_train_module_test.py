"""Tests for :class:`MoEV2TransformerTrainModule` config and construction."""

import torch

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlockConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import MoEFusedV2OptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module import MoEV2TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
)


def test_moe_v2_train_module_config_roundtrips():
    config = MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=MoEFusedV2OptimizerConfig(lr=1e-3),
    )
    restored = MoEV2TransformerTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.optim.lr == 1e-3


def test_moe_v2_train_module_config_roundtrips_with_parallelism():
    config = MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=MoEFusedV2OptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, reduce_grads_in_fp32=False
        ),
        pp_config=TransformerPipelineParallelConfig(degree=2),
    )
    restored = MoEV2TransformerTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.dp_config is not None and restored.dp_config.reduce_grads_in_fp32 is False
    assert restored.pp_config is not None and restored.pp_config.degree == 2


def _tiny_model_config(*, d_model: int = 64, n_layers: int = 2) -> MoEFusedV2TransformerConfig:
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


def _run_construct_no_ep():
    model = _tiny_model_config().build(init_device="cpu")
    config = MoEV2TransformerTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=MoEFusedV2OptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    # eval_only=True skips the optimizer build (its fp32-master-param setup is exercised on GPU);
    # this covers the world-mesh build + data-parallel wrapping with no expert parallelism.
    train_module = config.build(model, device=torch.device("cpu"), eval_only=True)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.dp_world_size == 2
    assert train_module.world_mesh["dense"] is not None
    assert train_module.moe_mesh is None  # no expert parallelism


def test_moe_v2_train_module_construction_no_ep():
    run_distributed_test(
        _run_construct_no_ep,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )
