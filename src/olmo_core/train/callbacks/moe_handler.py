from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from olmo_core.distributed.utils import get_local_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe import MoE
from olmo_core.utils import move_to_device

from .callback import Callback


@dataclass
class MoEHandlerCallback(Callback):
    """
    A callback to be used in conjunction with :class:`~olmo_core.nn.moe.MoE` based models for
    including the MoE's internal losses in the training loss.
    """

    _batch_lb_loss = None
    _batch_z_loss = None
    _moe_layer = None

    def clear_loss_buffers(self):
        assert self._moe_layer is not None
        self._moe_layer.clear_losses()
        if self._batch_lb_loss is not None:
            self._batch_lb_loss.zero_()
        if self._batch_z_loss is not None:
            self._batch_z_loss.zero_()

    def pre_train(self):
        for module in self.trainer.model.modules():
            if isinstance(module, MoE):
                self._moe_layer = module  # only need one
                break
        else:
            raise OLMoConfigurationError(
                f"No MoE layer found in model, required by {self.__class__.__name__}"
            )

    def pre_step(self, batch: Dict[str, Any]):
        del batch
        self.clear_loss_buffers()

    def post_eval_batch(self):
        self.clear_loss_buffers()

    def pre_backward(
        self,
        *,
        batch: Dict[str, Any],
        micro_batch: Dict[str, Any],
        loss: torch.Tensor,
    ):
        assert self._moe_layer is not None

        scale_factor = micro_batch["input_ids"].shape[0] / batch["input_ids"].shape[0]

        moe_loss: Optional[torch.Tensor] = None
        if (lb_loss := self._moe_layer.get_load_balancing_loss()) is not None:
            lb_loss.mul_(scale_factor)
            moe_loss = lb_loss
            if self._batch_lb_loss is None:
                self._batch_lb_loss = move_to_device(torch.tensor(0.0), lb_loss.device)
            self._batch_lb_loss += get_local_tensor(lb_loss)

        if (rz_loss := self._moe_layer.get_router_z_loss()) is not None:
            rz_loss.mul_(scale_factor)
            if moe_loss is not None:
                moe_loss += rz_loss
            else:
                moe_loss = rz_loss
            if self._batch_z_loss is None:
                self._batch_z_loss = move_to_device(torch.tensor(0.0), rz_loss.device)
            self._batch_z_loss += get_local_tensor(rz_loss)

        if moe_loss is not None:
            loss += moe_loss

    def post_train_batch(self):
        if self._batch_lb_loss is not None:
            self.trainer.record_metric("train/load balancing loss", self._batch_lb_loss)
        if self._batch_z_loss is not None:
            self.trainer.record_metric("train/router Z loss", self._batch_z_loss)
