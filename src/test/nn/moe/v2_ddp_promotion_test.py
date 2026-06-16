from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer, OLMoDDPModel
from olmo_core.train.train_module.transformer import (
    MoEV2TransformerTrainModule,
    MoEV2TransformerTrainModuleConfig,
    OLMoDDPTrainModule,
    OLMoDDPTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.ddp_train_module import (
    OLMoDDPTrainModule as OLMoDDPTrainModuleFromBridge,
)


def test_olmo_ddp_promoted_names_keep_moe_v2_compatibility():
    assert issubclass(OLMoDDPModel, MoEFusedV2Transformer)
    assert issubclass(OLMoDDPTrainModule, MoEV2TransformerTrainModule)
    assert issubclass(OLMoDDPTrainModuleConfig, MoEV2TransformerTrainModuleConfig)
    assert OLMoDDPTrainModuleFromBridge is OLMoDDPTrainModule
