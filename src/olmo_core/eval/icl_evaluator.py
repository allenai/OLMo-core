from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist

from .evaluator import Evaluator


class ICLEvaluator(Evaluator):
    """
    Evaluator for a variety of downstream in-context learning (ICL) tasks.
    """

    def __init__(
        self,
        *,
        name: str,
        batches: Iterable[Dict[str, Any]],
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(
            name=name, batches=batches, device=device, dp_process_group=dp_process_group
        )

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: torch.Tensor, logits: torch.Tensor
    ) -> None:
        pass

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        pass

    def reset_metrics(self) -> None:
        pass
