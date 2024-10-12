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

    def pre_train(self):
        for module in self.trainer.model.modules():
            if isinstance(module, MoE):
                self._moe_layer = module  # only need one
                break
        else:
            raise OLMoConfigurationError(
                f"No MoE layer found in model, required by {self.__class__.__name__}"
            )

    def post_model_forward(
        self,
        *,
        batch: Dict[str, Any],
        micro_batch: Dict[str, Any],
        num_micro_batches: int,
        batch_num_tokens_for_loss: torch.Tensor,
        loss: torch.Tensor,
        ce_loss: torch.Tensor,
        z_loss: Optional[torch.Tensor] = None,
    ):
        del batch, micro_batch, batch_num_tokens_for_loss, ce_loss, z_loss
        assert self._moe_layer is not None

        moe_loss: Optional[torch.Tensor] = None
        if (lb_loss := self._moe_layer.get_load_balancing_loss()) is not None:
            lb_loss.div_(num_micro_batches)
            moe_loss = lb_loss
            if self._batch_lb_loss is None:
                self._batch_lb_loss = move_to_device(torch.tensor(0.0), lb_loss.device)
            self._batch_lb_loss += get_local_tensor(lb_loss)

        if (rz_loss := self._moe_layer.get_router_z_loss()) is not None:
            rz_loss.div_(num_micro_batches)
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
            self._batch_lb_loss = None
        if self._batch_z_loss is not None:
            self.trainer.record_metric("train/router Z loss", self._batch_z_loss)
            self._batch_z_loss = None
