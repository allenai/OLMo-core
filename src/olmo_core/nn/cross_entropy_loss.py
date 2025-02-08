import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

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
        if compile:
            if torch.cuda.is_available():
                log.info("Compiling loss function...")
                self.base_loss_fn = torch.compile(self.base_loss_fn)
            else:
                log.warning("Skipping loss compilation since CUDA is not available")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # shape: (batch_size, seq_len - 1, vocab_size)
        logits_for_loss = logits[..., :-1, :].contiguous()

        # shape: (batch_size * (seq_len - 1), vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))

        # shape: (batch_size, seq_len - 1) -> (batch_size * (seq_len - 1),)
        labels_for_loss = labels.view(-1)

        # shape: depends on reduction
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

        return ce_loss, z_loss

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        shard_dimension: int = 1,
        #  output_layout: Optional[Placement] = None,
        #  use_local_output: bool = True,
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

        #  output_layout = output_layout or Replicate()
        #  parallelize_module(
        #      self,
        #      device_mesh=tp_mesh,
        #      parallelize_plan=PrepareModuleOutput(
        #          output_layouts=(  # type: ignore
        #              Shard(0),
        #              None if self.z_loss_multiplier is None else Shard(0),
        #          ),
        #          desired_output_layouts=(  # type: ignore
        #              output_layout,
        #              None if self.z_loss_multiplier is None else output_layout,
        #          ),
        #          use_local_output=use_local_output,
        #      ),
        #  )
