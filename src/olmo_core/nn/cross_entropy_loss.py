import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .functional import cross_entropy_loss, fused_cross_entropy_loss

log = logging.getLogger(__name__)


class LMCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for language modeling.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        compile: bool = False,
        fused: bool = False,
    ):
        super().__init__()

        if compile and fused:
            log.warning(f"{self.__class__.__name__} with fused+compile is experimental")

        self.ignore_index = ignore_index
        self.reduction: Literal["mean", "sum", "none"] = reduction
        self.z_loss_multiplier = z_loss_multiplier
        self.base_loss_fn = fused_cross_entropy_loss if fused else cross_entropy_loss
        if compile:
            if torch.cuda.is_available():
                log.info("Compiling loss function...")
                self.compile()
            else:
                log.warning("Skipping loss compilation since CUDA is not available")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the CE loss and optionally Z-loss.

        :param logits: The logits of shape ``(B, S, V)``.
            The final logits in the sequence dimension will be discarded to match the shape of
            the labels.
        :param labels: The target labels of shape ``(B, S-1)``.
        """
        if len(logits.shape) != 3:
            raise RuntimeError(
                f"expected logits to have shape (B, S, V) but found {tuple(logits.shape)} instead"
            )

        B, S, V = logits.shape
        if labels.shape != (B, S - 1):
            raise RuntimeError(
                f"expected labels to have shape (B, S-1) = {(B, S-1)}, but found {tuple(labels.shape)} instead"
            )

        # shape: (B, S - 1, V)
        logits_for_loss = logits[..., :-1, :].contiguous()

        # shape: (B * (S - 1), V)
        logits_for_loss = logits_for_loss.view(-1, V)

        # shape: (B, S - 1) -> (B * (S - 1),)
        labels_for_loss = labels.view(-1)

        ce_loss, z_loss = self.base_loss_fn(
            logits_for_loss,
            labels_for_loss,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            compute_z_loss=self.z_loss_multiplier is not None,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if self.reduction == "none":
            ce_loss = ce_loss.view(labels.shape)
            if z_loss is not None:
                z_loss = z_loss.view(labels.shape)

        return ce_loss, z_loss
