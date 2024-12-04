from dataclasses import dataclass
from typing import Any, Dict

import torch

from olmo_core.nn.moe import MoEHandler

from .callback import Callback


@dataclass
class MoEHandlerCallback(Callback):
    """
    A callback to be used in conjunction with :class:`~olmo_core.nn.moe.MoE` based models for
    including the MoE's internal losses in the training loss.
    """

    _handler = None

    @property
    def handler(self) -> MoEHandler:
        assert self._handler is not None
        return self._handler

    def pre_train(self):
        self._handler = MoEHandler(model=self.trainer.model)

    def pre_step(self, batch: Dict[str, Any]):
        del batch
        self.handler.clear_loss_buffers()

    def post_eval_batch(self):
        self.handler.clear_loss_buffers()

    def pre_backward(
        self,
        *,
        batch: Dict[str, Any],
        micro_batch: Dict[str, Any],
        loss: torch.Tensor,
    ):
        moe_loss = self.handler.get_combined_loss(batch=batch, micro_batch=micro_batch)
        if moe_loss is not None:
            loss += moe_loss

    def post_train_batch(self):
        if (moe_lb_loss := self.handler.get_lb_loss()) is not None:
            self.trainer.record_metric("train/load balancing loss", moe_lb_loss)
        if (moe_z_loss := self.handler.get_z_loss()) is not None:
            self.trainer.record_metric("train/router Z loss", moe_z_loss)
