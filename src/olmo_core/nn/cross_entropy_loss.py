import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)

from olmo_core.distributed.utils import get_local_tensor

from .functional import cross_entropy_loss, fused_cross_entropy_loss

log = logging.getLogger(__name__)


class _InnerCELoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        fused: bool = False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction: Literal["mean", "sum", "none"] = reduction
        self.z_loss_multiplier = z_loss_multiplier
        self.base_loss_fn = fused_cross_entropy_loss if fused else cross_entropy_loss
        self.tp_enabled = False

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if logits.shape[:-1] != labels.shape:
            raise RuntimeError(
                f"expected labels to have shape {logits.shape[:-1]}, but found {tuple(labels.shape)} instead"
            )

        # shape: (B * S, V)
        logits_for_loss = logits.view(-1, logits.shape[-1])

        # shape: (B, S) -> (B * S,)
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
        elif self.tp_enabled:
            ce_loss = ce_loss.unsqueeze(0)
            if z_loss is not None:
                z_loss = z_loss.unsqueeze(0)

        return ce_loss, z_loss


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss.
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

        self._ce_loss = _InnerCELoss(
            ignore_index=ignore_index,
            reduction=reduction,
            z_loss_multiplier=z_loss_multiplier,
            fused=fused,
        )
        self._tp_enabled = False

        if compile:
            if torch.cuda.is_available():
                log.info("Compiling loss function...")
                self.compile()
            else:
                log.warning("Skipping loss compilation since CUDA is not available")

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def z_loss_enabled(self) -> bool:
        return self._ce_loss.z_loss_multiplier is not None

    @property
    def reduction(self) -> Literal["sum", "mean", "none"]:
        return self._ce_loss.reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the CE loss and optionally Z-loss.

        :param logits: The logits of shape ``(B, S, V)``.
        :param labels: The target labels of shape ``(B, S)``.
        """
        ce_loss, z_loss = self._ce_loss(get_local_tensor(logits), get_local_tensor(labels))

        if self.reduction != "none" and ce_loss.numel() > 0:
            # This will be the same case with tensor/sequence parallel loss.
            if self.reduction == "sum":
                ce_loss = ce_loss.sum()
                if z_loss is not None:
                    z_loss = z_loss.sum()
            elif self.reduction == "mean":
                ce_loss = ce_loss.mean()
                if z_loss is not None:
                    z_loss = z_loss.mean()
            else:
                raise NotImplementedError(self.reduction)

        return ce_loss, z_loss

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layouts: Optional[Tuple[Placement, Placement]] = None,
        shard_dimension: int = 1,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = False,
    ):
        if self.reduction == "none":
            raise NotImplementedError(self.reduction)

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=input_layouts,  # type: ignore
                desired_input_layouts=(Shard(shard_dimension), Shard(shard_dimension)),  # type: ignore
                use_local_output=False,
            ),
        )

        expected_output_layout = Shard(shard_dimension) if self.reduction == "none" else Shard(0)
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(  # type: ignore
                    expected_output_layout,
                    None if self.z_loss_enabled is None else expected_output_layout,
                ),
                use_local_output=False,
            ),
        )

        desired_output_layout = output_layout or Replicate()
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                desired_output_layouts=(  # type: ignore
                    desired_output_layout,
                    None if self.z_loss_enabled is None else desired_output_layout,
                ),
                use_local_output=use_local_output,
            ),
        )

        self._tp_enabled = True
        self._ce_loss.tp_enabled = True
