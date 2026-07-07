"""
Tests for :class:`MoEV2TransformerTrainModule`.

The fused MoE-v2 model is GPU-only by construction (its blocks allocate CUDA events for the
EP/TBO comm paths), so end-to-end training coverage lives in the GPU test job. What's CPU-testable
here is the config surface: construction and serialization round-trip.
"""

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.optim import MoEFusedV2OptimizerConfig
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
