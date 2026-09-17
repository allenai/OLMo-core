"""Compatibility imports for serialized MoE V2 train-module configurations."""

from .ddp_train_module import OLMoDDPTrainModule

MoEV2TransformerTrainModule = OLMoDDPTrainModule

__all__ = ["OLMoDDPTrainModule", "MoEV2TransformerTrainModule"]
