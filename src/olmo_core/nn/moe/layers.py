from typing import Optional

import torch
import torch.nn as nn

from olmo_core.doc_utils import beta_feature


@beta_feature
class MoE(nn.Module):
    """
    A thin wrapper around `megablocks <https://github.com/databricks/megablocks>`_ MoE layers.

    .. tip::
        Use :class:`MoEConfig` to build instances of this module.

    .. important::
        This should always be used in conjunction with the
        :class:`~olmo_core.train.callbacks.MoEHandlerCallback` for training.
    """

    def __init__(self, args, inner):
        super().__init__()
        self.args = args
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the MoE on the input.

        :param x: A tensor of shape ``(batch_size, sequence_length, d_model)``.
        """
        return self.inner(x)

    def get_load_balancing_loss(self) -> Optional[torch.Tensor]:
        """
        Get the batched load-balancing loss from the internal buffers.

        .. important::
            This method will clear the internal buffers so can only be called once per forward pass.
        """
        from megablocks.layers.moe import (  # type: ignore
            batched_load_balancing_loss,
            clear_load_balancing_loss,
        )

        if isinstance(lb_loss := batched_load_balancing_loss(self.args), torch.Tensor):
            clear_load_balancing_loss()
            return lb_loss
        else:
            return None

    def get_router_z_loss(self) -> Optional[torch.Tensor]:
        """
        Get the batched router Z-loss from the internal buffers.

        .. important::
            This method will clear the internal buffers so can only be called once per forward pass.
        """
        from megablocks.layers.router import (  # type: ignore
            batched_router_zloss,
            clear_router_zloss,
        )

        if self.args.moe_zloss_weight != 0 and isinstance(
            (z_loss_per_layer := batched_router_zloss(self.args)), torch.Tensor
        ):
            z_loss = z_loss_per_layer.sum() / self.args.num_layers
            clear_router_zloss()
            return z_loss
        else:
            return None

    def get_loss(self) -> Optional[torch.Tensor]:
        """
        Get the batched combined load-balancing loss and router Z-loss from the internal buffers.

        .. important::
            This method will clear the internal buffers so can only be called once per forward pass.
        """
        loss: Optional[torch.Tensor] = None
        if (lb_loss := self.get_load_balancing_loss()) is not None:
            loss = lb_loss

        if (rz_loss := self.get_router_z_loss()) is not None:
            if loss is not None:
                loss += rz_loss
            else:
                loss = rz_loss

        return loss

    def clear_losses(self):
        """
        Clear internal loss buffers.
        """
        from megablocks.layers.moe import clear_load_balancing_loss  # type: ignore
        from megablocks.layers.router import clear_router_zloss  # type: ignore

        clear_load_balancing_loss()
        clear_router_zloss()
