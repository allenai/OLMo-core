"""
MoE V2 building blocks. The fused block/model/train-module were promoted to the canonical
``OLMoDDP*`` names under :mod:`olmo_core.nn.ddp` and :mod:`olmo_core.optim`; import those directly.
"""

from ...output_discard_checkpoint import OutputDiscardCheckpoint

__all__ = ["OutputDiscardCheckpoint"]
