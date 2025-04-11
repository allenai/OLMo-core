import torch


class AutoAuxiliaryLoss(torch.autograd.Function):
    """
    An autograd function that triggers the backward pass for an auxiliary loss.
    """

    @staticmethod
    def forward(ctx, activation: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return activation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        aux_loss_grad = torch.ones_like(aux_loss)
        return grad_output, aux_loss_grad


def attach_auxiliary_loss(activation: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
    """
    Attach an auxiliary loss to an activation with an autograd function in order to trigger
    gradients for the aux loss in the backwards pass.

    :returns: The input activation unchanged.
    """
    return AutoAuxiliaryLoss.apply(activation, aux_loss)  # type: ignore[return-value]
