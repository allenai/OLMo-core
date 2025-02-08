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

from .functional import cross_entropy_loss, fused_cross_entropy_loss

log = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
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
        self.ignore_index = ignore_index
        self.reduction: Literal["mean", "sum", "none"] = reduction
        self.z_loss_multiplier = z_loss_multiplier
        self.base_loss_fn = fused_cross_entropy_loss if fused else cross_entropy_loss
        self._tp_enabled: bool = False
        if compile:
            if torch.cuda.is_available():
                log.info("Compiling loss function...")
                self.base_loss_fn = torch.compile(self.base_loss_fn)
            else:
                log.warning("Skipping loss compilation since CUDA is not available")

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the CE loss and optionally Z-loss.

        :param logits: The logits of shape ``(*, num_classes)``.
        :param labels: The target labels of shape ``(*, )``.
        """
        # Flatten inputs for loss function.
        logits_for_loss = logits.reshape(-1, logits.size(-1))
        labels_for_loss = labels.reshape(-1)

        ce_loss, z_loss = self.base_loss_fn(
            logits_for_loss,
            labels_for_loss,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            compute_z_loss=self.z_loss_multiplier is not None,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if self.reduction == "none":
            ce_loss = ce_loss.view(logits.shape[:-1])
            if z_loss is not None:
                z_loss = z_loss.view(logits.shape[:-1])
        elif self.tp_enabled:
            ce_loss = ce_loss.unsqueeze(0)
            if z_loss is not None:
                z_loss = z_loss.unsqueeze(0)

        return ce_loss, z_loss

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
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
                input_layouts=None if input_layout is None else (input_layout, input_layout),  # type: ignore
                desired_input_layouts=(Shard(shard_dimension),),
                use_local_output=True,
            ),
        )

        expected_output_layout = Shard(shard_dimension) if self.reduction == "none" else Shard(0)
        desired_output_layout = output_layout or Replicate()
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(  # type: ignore
                    expected_output_layout,
                    None if self.z_loss_multiplier is None else expected_output_layout,
                ),
                desired_output_layouts=(  # type: ignore
                    desired_output_layout,
                    None if self.z_loss_multiplier is None else desired_output_layout,
                ),
                use_local_output=use_local_output,
            ),
        )

        self._tp_enabled = True
