import torch
import torch.nn as nn


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

    def get_loss(self) -> torch.Tensor:
        """
        Get the batched load-balancing and router Z-loss from the internal buffers.

        .. important::
            This method will clear the internal buffers so can only be called once per forward pass.
        """
        from megablocks.layers.moe import (  # type: ignore
            batched_load_balancing_loss,
            clear_load_balancing_loss,
            get_load_balancing_loss,
        )
        from megablocks.layers.router import (  # type: ignore
            batched_router_zloss,
            clear_router_zloss,
        )

        _, expert_scores, _ = zip(*get_load_balancing_loss())
        tokens = expert_scores[0].shape[0]

        loss = batched_load_balancing_loss(self.args)
        assert loss is not None

        if self.args.moe_zloss_weight != 0:
            unscaled_z_loss = batched_router_zloss(self.args)
            assert isinstance(unscaled_z_loss, torch.Tensor)
            z_loss = unscaled_z_loss.sum() / (self.args.num_layers * tokens * self.args.moe_top_k)
            loss += z_loss
            clear_router_zloss()

        clear_load_balancing_loss()

        return loss  # type: ignore
