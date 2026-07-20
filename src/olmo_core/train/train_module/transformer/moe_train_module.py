"""
Compatibility imports for the old MoE V2 train-module path.
"""

from .ddp_train_module import OLMoDDPTrainModule

MoEV2TransformerTrainModule = OLMoDDPTrainModule

__all__ = ["OLMoDDPTrainModule", "MoEV2TransformerTrainModule"]
