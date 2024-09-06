from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from olmo_core.optim import SkipStepOptimizer

from .callback import Callback


@dataclass
class GradClipperCallback(Callback):
    """
    Enables gradient clipping during training.
    """

    max_grad_norm: float = 1.0

    def pre_optim_step(self):
        grad_norm: torch.Tensor
        if isinstance(self.trainer.model, FSDP):
            grad_norm = self.trainer.model.clip_grad_norm_(self.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                self.trainer.model.parameters(), self.max_grad_norm
            )

        # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
        self.trainer.record_metric("optim/total grad norm", grad_norm, reduce_type=None)
        if isinstance(self.trainer.optim, SkipStepOptimizer):
            self.trainer.optim.latest_grad_norm = grad_norm
