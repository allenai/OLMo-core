"""Compatibility aliases and serialized configuration paths for OLMoDDP models."""

import subprocess
import sys

from olmo_core.config import Config, DType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.ddp import OLMoDDPModel as OLMoDDPModelFromDDP
from olmo_core.nn.ddp import OLMoDDPTransformerBlock as OLMoDDPTransformerBlockFromDDP
from olmo_core.nn.ddp import (
    OLMoDDPTransformerBlockConfig as OLMoDDPTransformerBlockConfigFromDDP,
)
from olmo_core.nn.ddp.block import (
    OLMoDDPTransformerBlock,
    OLMoDDPTransformerBlockConfig,
)
from olmo_core.nn.ddp.model import OLMoDDPModel as OLMoDDPModelFromCanonicalModule
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.block import (
    MoEFusedV2TransformerBlock,
    MoEFusedV2TransformerBlockConfig,
)
from olmo_core.nn.moe.v2.block import MoERouterConfigV2 as MoERouterConfigV2FromOldBlock
from olmo_core.nn.moe.v2.block import (
    RoutedExpertsConfig as RoutedExpertsConfigFromOldBlock,
)
from olmo_core.nn.moe.v2.block import (
    SharedExpertsConfig as SharedExpertsConfigFromOldBlock,
)
from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer, OLMoDDPModel
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    MoEFusedV2TransformerConfig,
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import (
    MoEFusedV2Optimizer,
    MoEFusedV2OptimizerConfig,
    OLMoDDPOptimizer,
    OLMoDDPOptimizerConfig,
)
from olmo_core.train.train_module.transformer import (
    MoEV2TransformerTrainModule,
    MoEV2TransformerTrainModuleConfig,
    OLMoDDPTrainModule,
    OLMoDDPTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.ddp_train_module import (
    OLMoDDPTrainModule as OLMoDDPTrainModuleFromBridge,
)
from olmo_core.train.train_module.transformer.moe_train_module import (
    MoEV2TransformerTrainModule as MoEV2TransformerTrainModuleFromOldModule,
)


def test_olmo_ddp_promoted_names_keep_moe_v2_compatibility():
    assert MoEFusedV2Transformer is OLMoDDPModel
    assert OLMoDDPModelFromCanonicalModule is OLMoDDPModel
    assert OLMoDDPModelFromDDP is OLMoDDPModel
    assert MoEFusedV2TransformerConfig is OLMoDDPModelConfig
    assert MoEFusedV2TransformerBlock is OLMoDDPTransformerBlock
    assert MoEFusedV2TransformerBlockConfig is OLMoDDPTransformerBlockConfig
    assert OLMoDDPTransformerBlockFromDDP is OLMoDDPTransformerBlock
    assert OLMoDDPTransformerBlockConfigFromDDP is OLMoDDPTransformerBlockConfig
    assert MoERouterConfigV2FromOldBlock is MoERouterConfigV2
    assert RoutedExpertsConfigFromOldBlock is RoutedExpertsConfig
    assert SharedExpertsConfigFromOldBlock is SharedExpertsConfig
    assert MoEV2TransformerTrainModule is OLMoDDPTrainModule
    assert MoEV2TransformerTrainModuleFromOldModule is OLMoDDPTrainModule
    assert MoEV2TransformerTrainModuleConfig is OLMoDDPTrainModuleConfig
    assert OLMoDDPTrainModuleFromBridge is OLMoDDPTrainModule
    assert MoEFusedV2OptimizerConfig is OLMoDDPOptimizerConfig
    assert MoEFusedV2Optimizer is OLMoDDPOptimizer


def test_olmo_ddp_promoted_config_names_round_trip():
    block_config = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=AttentionConfig(n_heads=2),
        layer_norm=LayerNormConfig(),
        routed_experts=RoutedExpertsConfig(
            d_model=16, hidden_size=32, num_experts=4, bias=False, dtype=DType.float32
        ),
        routed_experts_router=MoERouterConfigV2(d_model=16, num_experts=4, top_k=2),
    )
    model_config = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=16,
        vocab_size=128,
        n_layers=1,
        block=block_config,
        lm_head=LMHeadConfig(layer_norm=LayerNormConfig()),
    )
    model_config_dict = model_config.as_config_dict()
    assert model_config_dict["_CLASS_"] == "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    assert isinstance(Config.from_dict(model_config_dict), OLMoDDPModelConfig)

    old_model_config_dict = dict(model_config_dict)
    old_model_config_dict["_CLASS_"] = "olmo_core.nn.transformer.config.MoEFusedV2TransformerConfig"
    assert isinstance(Config.from_dict(old_model_config_dict), OLMoDDPModelConfig)

    block_config_dict = block_config.as_config_dict()
    assert block_config_dict["_CLASS_"] == "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    assert isinstance(Config.from_dict(block_config_dict), OLMoDDPTransformerBlockConfig)

    old_path_block_config_dict = dict(block_config_dict)
    old_path_block_config_dict[
        "_CLASS_"
    ] = "olmo_core.nn.moe.v2.block.OLMoDDPTransformerBlockConfig"
    assert isinstance(Config.from_dict(old_path_block_config_dict), OLMoDDPTransformerBlockConfig)

    old_block_config_dict = dict(block_config_dict)
    old_block_config_dict["_CLASS_"] = "olmo_core.nn.moe.v2.block.MoEFusedV2TransformerBlockConfig"
    assert isinstance(Config.from_dict(old_block_config_dict), OLMoDDPTransformerBlockConfig)

    optim_config = OLMoDDPOptimizerConfig()
    optim_config_dict = optim_config.as_config_dict()
    assert optim_config_dict["_CLASS_"] == "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"
    assert isinstance(Config.from_dict(optim_config_dict), OLMoDDPOptimizerConfig)

    old_optim_config_dict = dict(optim_config_dict)
    old_optim_config_dict["_CLASS_"] = "olmo_core.optim.moe_optimizer.MoEFusedV2OptimizerConfig"
    assert isinstance(Config.from_dict(old_optim_config_dict), OLMoDDPOptimizerConfig)


def test_olmo_ddp_import_does_not_load_optional_features():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import olmo_core.nn.ddp; "
            "assert not {"
            "'olmo_core.nn.moe.v2.ep_no_sync_tbo_rowwise',"
            "'olmo_core.nn.moe.v2.qwen',"
            "'olmo_core.nn.moe.v2.te.cpu_offload'"
            "}.intersection(sys.modules)",
        ],
        check=True,
    )
