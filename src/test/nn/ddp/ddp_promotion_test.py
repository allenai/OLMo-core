"""
The MoE-v2 stack was promoted/renamed to the canonical ``OLMoDDP*`` names. These tests pin that the
canonical names resolve to the same objects across their import paths and that configs serialized
under the former ``olmo_core.nn.moe.v2.*`` module paths still deserialize.
"""

from olmo_core.config import Config
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
from olmo_core.nn.moe.v2.block import MoERouterConfigV2 as MoERouterConfigV2FromOldBlock
from olmo_core.nn.moe.v2.block import (
    OLMoDDPTransformerBlock as OLMoDDPTransformerBlockFromOldBlock,
)
from olmo_core.nn.moe.v2.block import (
    RoutedExpertsConfig as RoutedExpertsConfigFromOldBlock,
)
from olmo_core.nn.moe.v2.block import (
    SharedExpertsConfig as SharedExpertsConfigFromOldBlock,
)
from olmo_core.nn.moe.v2.model import OLMoDDPModel as OLMoDDPModelFromOldModel
from olmo_core.nn.moe.v2.qwen import build_debug_qwen3_moe_config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.optim import OLMoDDPOptimizer, OLMoDDPOptimizerConfig
from olmo_core.train.train_module.transformer import (
    OLMoDDPTrainModule,
    OLMoDDPTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.ddp_train_module import (
    OLMoDDPTrainModule as OLMoDDPTrainModuleFromBridge,
)
from olmo_core.train.train_module.transformer.moe_train_module import (
    OLMoDDPTrainModule as OLMoDDPTrainModuleFromOldModule,
)

# Reference the optimizer symbols so they're covered by the import checks below.
_ = (OLMoDDPOptimizer, OLMoDDPTrainModule, OLMoDDPTrainModuleConfig)


def test_olmo_ddp_canonical_names_resolve_across_import_paths():
    assert OLMoDDPModelFromCanonicalModule is OLMoDDPModelFromDDP
    assert OLMoDDPModelFromOldModel is OLMoDDPModelFromDDP
    assert OLMoDDPTransformerBlockFromDDP is OLMoDDPTransformerBlock
    assert OLMoDDPTransformerBlockFromOldBlock is OLMoDDPTransformerBlock
    assert OLMoDDPTransformerBlockConfigFromDDP is OLMoDDPTransformerBlockConfig
    assert MoERouterConfigV2FromOldBlock is MoERouterConfigV2
    assert RoutedExpertsConfigFromOldBlock is RoutedExpertsConfig
    assert SharedExpertsConfigFromOldBlock is SharedExpertsConfig
    assert OLMoDDPTrainModuleFromBridge is OLMoDDPTrainModule
    assert OLMoDDPTrainModuleFromOldModule is OLMoDDPTrainModule


def test_olmo_ddp_config_names_round_trip():
    model_config = build_debug_qwen3_moe_config(vocab_size=128)
    model_config_dict = model_config.as_config_dict()
    assert model_config_dict["_CLASS_"] == "olmo_core.nn.transformer.config.OLMoDDPModelConfig"
    assert isinstance(Config.from_dict(model_config_dict), OLMoDDPModelConfig)

    block_config = build_debug_qwen3_moe_config(vocab_size=128, n_layers=1).block
    assert isinstance(block_config, OLMoDDPTransformerBlockConfig)
    block_config_dict = block_config.as_config_dict()
    assert block_config_dict["_CLASS_"] == "olmo_core.nn.ddp.block.OLMoDDPTransformerBlockConfig"
    assert isinstance(Config.from_dict(block_config_dict), OLMoDDPTransformerBlockConfig)

    # Configs serialized under the former ``olmo_core.nn.moe.v2.block`` module path still resolve.
    old_path_block_config_dict = dict(block_config_dict)
    old_path_block_config_dict[
        "_CLASS_"
    ] = "olmo_core.nn.moe.v2.block.OLMoDDPTransformerBlockConfig"
    assert isinstance(Config.from_dict(old_path_block_config_dict), OLMoDDPTransformerBlockConfig)

    optim_config = OLMoDDPOptimizerConfig()
    optim_config_dict = optim_config.as_config_dict()
    assert optim_config_dict["_CLASS_"] == "olmo_core.optim.moe_optimizer.OLMoDDPOptimizerConfig"
    assert isinstance(Config.from_dict(optim_config_dict), OLMoDDPOptimizerConfig)
