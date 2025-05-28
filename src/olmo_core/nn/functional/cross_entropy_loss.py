import logging
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["cross_entropy_loss", "fused_linear_cross_entropy_loss"]

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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss fused with the linear layer that computes the logits.

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

    :returns: The cross entropy loss and optionally the z-loss.
    """
    if _fused_linear_cross_entropy_loss is None:
        raise RuntimeError("'fused_linear_cross_entropy_loss' requires liger-kernel")
    ce_loss, z_loss = _fused_linear_cross_entropy_loss(
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
    )
    if compute_z_loss:
        return ce_loss, z_loss
    else:
        return ce_loss, None
