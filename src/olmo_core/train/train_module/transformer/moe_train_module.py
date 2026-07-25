"""
Compatibility re-export: :class:`~olmo_core.train.train_module.transformer.ddp_train_module.OLMoDDPTrainModule`
remains importable from this former module path.
"""

from .ddp_train_module import OLMoDDPTrainModule

__all__ = ["OLMoDDPTrainModule"]
