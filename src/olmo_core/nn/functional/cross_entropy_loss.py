import logging
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["cross_entropy_loss", "fused_linear_cross_entropy_loss", "cute_cross_entropy_loss"]

log = logging.getLogger(__name__)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss that optionally computes the softmax auxiliary loss (z-loss) as well.

    :param logits: Predicted unnormalized logits with shape ``(N, vocab_size)``.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param ignore_index: Specifies a target value that is ignored and does not contribute to
        the input gradient.
    :param reduction: Specifies the reduction to apply to the output.
        Can be "none", "mean", or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.

    :returns: The cross entropy loss and optionally the z-loss.
    """
    logits = logits.float()
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    mask = labels != ignore_index
    if reduction == "mean":
        z_squared = (z_squared * mask).sum() / mask.sum()
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


_fused_linear_cross_entropy_loss: Optional[Callable] = None

try:
    from liger_kernel.ops.fused_linear_cross_entropy import (  # type: ignore
        LigerFusedLinearCrossEntropyFunction,
    )

    _fused_linear_cross_entropy_loss = LigerFusedLinearCrossEntropyFunction.apply
except ImportError:
    pass
except Exception:
    log.exception("Error importing liger-kernel")


@torch._dynamo.disable()
def fused_linear_cross_entropy_loss(
    _input: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    ce_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    softcap: Optional[float] = None,
    accum_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss fused with the linear layer that computes the logits, which avoids materialization
    of the large logits tensor. Additionally, this function computes gradients during the forward pass,
    (valid when CrossEntropyLoss comes last), so _input and labels do not need to be stored for the backwards pass.

    :param _input: The inputs to pass through the linear layer to produce the logits ``(N, D)``.
    :param weight: The weight of the linear layer.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param bias: Optional bias for the linear layer.
    :param ignore_index: Specifies a target value that is ignored and does not contribute to
        the input gradient.
    :param reduction: Specifies the reduction to apply to the output.
        Can be "none", "mean", or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.
    :param accum_dtype: The dtype of intermediate result buffers for weight and bias gradient
        accumulations. Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`,
        if the training is unstable with original dtype. Default to performing accumulations in original dtype.

    :returns: The cross entropy loss and optionally the z-loss.
    """
    if _fused_linear_cross_entropy_loss is None:
        raise RuntimeError("'fused_linear_cross_entropy_loss' requires liger-kernel")
    ce_loss, z_loss, per_token_acc = _fused_linear_cross_entropy_loss(
        _input,
        weight,
        labels,
        bias,
        ce_weight,
        ignore_index,
        z_loss_multiplier,
        label_smoothing,
        reduction,
        softcap,
        compute_z_loss,
        accum_dtype,
    )
    del per_token_acc
    if compute_z_loss:
        return ce_loss, z_loss
    else:
        return ce_loss, None


_cute_cross_entropy_fwd: Optional[Callable] = None
_cute_cross_entropy_bwd: Optional[Callable] = None
try:
    from quack.cross_entropy import cross_entropy_bwd, cross_entropy_fwd  # type: ignore

    _cute_cross_entropy_fwd = cross_entropy_fwd
    _cute_cross_entropy_bwd = cross_entropy_bwd
except ImportError:
    pass


class _CuTeCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, lse_partial=None, ignore_index=-100, inplace_backward=False):
        assert _cute_cross_entropy_fwd is not None
        if lse_partial is None:
            loss, lse = _cute_cross_entropy_fwd(
                x, target, ignore_index=ignore_index, return_lse=True
            )
        else:
            # if we already compute partial lse, then to compute the final lse we treat
            # @lse_partial as @x and @x as @target_logit
            loss, lse = _cute_cross_entropy_fwd(
                lse_partial, target, target_logit=x, ignore_index=ignore_index, return_lse=True
            )
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        ctx.inplace_backward = inplace_backward
        return loss

    @staticmethod
    def backward(ctx, dloss):  # type: ignore
        assert _cute_cross_entropy_bwd is not None
        x, target, lse = ctx.saved_tensors
        dx = _cute_cross_entropy_bwd(
            x,
            target,
            dloss.contiguous(),
            lse,
            ctx.ignore_index,
            inplace_backward=ctx.inplace_backward,
        )
        return dx, None, None, None, None


def cute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Like :func:`cross_entropy_loss`, but uses an efficient CuTe-based implementation from the QuACK
    library.
    """
    lse_partial: Optional[torch.Tensor] = None
    if compute_z_loss:
        M, N = logits.shape
        assert N % 128 == 0, "CuTe cross entropy loss requires vocab size to be multiple of 128"
        lse_partial = logits.view(M, N // 128, 128).float().logsumexp(dim=-1)

    loss: torch.Tensor = _CuTeCrossEntropyFunction.apply(logits, labels, lse_partial, ignore_index)  # type: ignore[assignment]
    mask = labels != ignore_index
    mask_sum: Optional[torch.Tensor] = None
    if reduction == "mean":
        mask_sum = mask.sum().float()
        loss = loss.sum() / mask_sum
    elif reduction == "sum":
        loss = loss.sum()

    if not compute_z_loss:
        return loss, None

    assert lse_partial is not None
    z_squared = lse_partial.logsumexp(-1).pow(2)
    if reduction == "mean":
        assert mask_sum is not None
        z_squared = (z_squared * mask).sum() / mask_sum
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss
