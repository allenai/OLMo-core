from olmo_core.nn import OLMoDDPModel as OLMoDDPModelFromNN
from olmo_core.nn.ddp import OLMoDDPModel as OLMoDDPModelFromDDP
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
    assert MoEFusedV2Transformer is OLMoDDPModel
    assert OLMoDDPModelFromDDP is OLMoDDPModel
    assert OLMoDDPModelFromNN is OLMoDDPModel
    assert MoEV2TransformerTrainModule is OLMoDDPTrainModule
    assert MoEV2TransformerTrainModuleConfig is OLMoDDPTrainModuleConfig
    assert OLMoDDPTrainModuleFromBridge is OLMoDDPTrainModule
