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
        scaled_aux_loss_grad = torch.ones_like(aux_loss)
        return grad_output, scaled_aux_loss_grad
