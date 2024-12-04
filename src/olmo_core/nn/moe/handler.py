from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from olmo_core.distributed.utils import get_local_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import move_to_device

from .layers import MoE


@dataclass
class MoEHandler:
    """
    A handler for :class:`~olmo_core.nn.moe.MoE` based models to collect the internal load-balancing
    loss and Z-loss.
    """

    model: nn.Module

    _batch_lb_loss = None
    _batch_z_loss = None
    _moe_layer = None

    @staticmethod
    def has_moe(model: nn.Module) -> bool:
        for module in model.modules():
            if isinstance(module, MoE):
                return True
        return False

    def __post_init__(self):
        for module in self.model.modules():
            if isinstance(module, MoE):
                self._moe_layer = module  # only need one
                break
        else:
            raise OLMoConfigurationError(
                f"No MoE layer found in model, required by {self.__class__.__name__}"
            )

    def clear_loss_buffers(self):
        assert self._moe_layer is not None
        self._moe_layer.clear_losses()
        if self._batch_lb_loss is not None:
            self._batch_lb_loss.zero_()
        if self._batch_z_loss is not None:
            self._batch_z_loss.zero_()

    def get_combined_loss(
        self, *, batch: Dict[str, Any], micro_batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
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

        return moe_loss

    def get_lb_loss(self) -> Optional[torch.Tensor]:
        return self._batch_lb_loss

    def get_z_loss(self) -> Optional[torch.Tensor]:
        return self._batch_z_loss
