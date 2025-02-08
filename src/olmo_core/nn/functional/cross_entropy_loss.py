from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["cross_entropy_loss", "fused_cross_entropy_loss"]


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

    .. seealso::
        :func:`fused_cross_entropy_loss()`.

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


_fused_cross_entropy_loss: Optional[Callable] = None

try:
    import olmo_core.triton.cross_entropy_loss as triton_ce_loss

    #  import flash_attn.ops.triton.cross_entropy as flash_attn_ce  # type: ignore

    _fused_cross_entropy_loss = triton_ce_loss.cross_entropy_loss
except ModuleNotFoundError:
    pass


def fused_cross_entropy_loss(
    logits,
    labels,
    *,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    A "fused" triton-based implementation of :func:`cross_entropy_loss`.

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
    if _fused_cross_entropy_loss is None:
        raise RuntimeError("triton is required for fused_cross_entropy_loss")

    logits = logits.float()

    loss, z_loss = _fused_cross_entropy_loss(
        logits,
        labels,
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=z_loss_multiplier,
        inplace_backward=False,
        process_group=None,
        ignore_index=ignore_index,
    )

    mask = labels != ignore_index

    if reduction == "mean":
        loss = loss.sum() / mask.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        loss = loss

    if not compute_z_loss:
        return loss, None

    if reduction == "mean":
        z_loss = z_loss.sum() / mask.sum()
    elif reduction == "sum":
        z_loss = z_loss.sum()
    else:
        z_loss = z_loss

    return loss, z_loss
