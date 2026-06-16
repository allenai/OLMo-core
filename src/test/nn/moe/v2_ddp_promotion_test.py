from olmo_core.config import Config
from olmo_core.nn import OLMoDDPModel as OLMoDDPModelFromNN
from olmo_core.nn import OLMoDDPModelConfig as OLMoDDPModelConfigFromNN
from olmo_core.nn.ddp import OLMoDDPModel as OLMoDDPModelFromDDP
from olmo_core.nn.ddp import OLMoDDPModelConfig as OLMoDDPModelConfigFromDDP
from olmo_core.nn.moe.v2.qwen import build_debug_qwen3_moe_config
from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer, OLMoDDPModel
from olmo_core.nn.transformer import MoEFusedV2TransformerConfig, OLMoDDPModelConfig
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
    assert OLMoDDPModelFromDDP is OLMoDDPModel
    assert OLMoDDPModelFromNN is OLMoDDPModel
    assert MoEFusedV2TransformerConfig is OLMoDDPModelConfig
    assert OLMoDDPModelConfigFromDDP is OLMoDDPModelConfig
    assert OLMoDDPModelConfigFromNN is OLMoDDPModelConfig
    assert MoEV2TransformerTrainModule is OLMoDDPTrainModule
    assert MoEV2TransformerTrainModuleFromOldModule is OLMoDDPTrainModule
    assert MoEV2TransformerTrainModuleConfig is OLMoDDPTrainModuleConfig
    assert OLMoDDPTrainModuleFromBridge is OLMoDDPTrainModule
    assert MoEFusedV2OptimizerConfig is OLMoDDPOptimizerConfig
    assert MoEFusedV2Optimizer is OLMoDDPOptimizer


def test_olmo_ddp_promoted_config_names_round_trip():
    model_config = build_debug_qwen3_moe_config(vocab_size=128)
    model_config_dict = model_config.as_config_dict()
    assert model_config_dict["_CLASS_"] == "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    assert isinstance(Config.from_dict(model_config_dict), OLMoDDPModelConfig)

    old_model_config_dict = dict(model_config_dict)
    old_model_config_dict["_CLASS_"] = "olmo_core.nn.transformer.config.MoEFusedV2TransformerConfig"
    assert isinstance(Config.from_dict(old_model_config_dict), OLMoDDPModelConfig)

    optim_config = OLMoDDPOptimizerConfig()
    optim_config_dict = optim_config.as_config_dict()
    assert optim_config_dict["_CLASS_"] == "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"
    assert isinstance(Config.from_dict(optim_config_dict), OLMoDDPOptimizerConfig)

    old_optim_config_dict = dict(optim_config_dict)
    old_optim_config_dict["_CLASS_"] = "olmo_core.optim.moe_optimizer.MoEFusedV2OptimizerConfig"
    assert isinstance(Config.from_dict(old_optim_config_dict), OLMoDDPOptimizerConfig)
