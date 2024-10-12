from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe import MoE

from .callback import Callback


@dataclass
class MoEHandlerCallback(Callback):
    """
    A callback to be used in conjunction with :class:`~olmo_core.nn.moe.MoE` based models for
    including the MoE's internal losses in the training loss.
    """

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
        del batch_num_tokens_for_loss, ce_loss, z_loss
        assert self._moe_layer is not None

        if (moe_loss := self._moe_layer.get_loss()) is not None:
            num_micro_batches = batch["input_ids"].shape[0] // micro_batch["input_ids"].shape[0]
            loss += moe_loss / num_micro_batches
