import logging
from dataclasses import dataclass, field
from typing import List

import torch

from olmo_core.eval import Evaluator
from olmo_core.utils import format_float, move_to_device

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class EvaluatorCallback(Callback):
    """
    Runs in-loop evaluations periodically during training.
    """

    evaluators: List[Evaluator] = field(default_factory=list)
    """
    The evaluators to run.
    """

    eval_interval: int = 1000
    """
    The interval (in steps) with which to run the evaluators.
    """

    def post_step(self):
        # Put model in eval train mode.
        self.trainer.optim.zero_grad(set_to_none=True)
        self.trainer.model.eval()

        for evaluator in self.evaluators:
            log.info(f"Running {evaluator.name} evals...")
            evaluator.reset_metrics()
            for eval_step, batch in enumerate(evaluator):
                batch = move_to_device(batch, self.trainer.device)
                with torch.no_grad():
                    ce_loss, _, logits = self.trainer._model_forward(
                        batch, loss_reduction="none", compute_z_loss=False
                    )
                evaluator.update_metrics(batch, ce_loss, logits)
                log.info(f"[eval={evaluator.name},step={eval_step+1}]")

            for name, value in evaluator.compute_metrics().items():
                name = f"eval/{evaluator.name}/{name}"
                self.trainer.record_metric(name, value)
                # NOTE: going to have a host-device sync here but that's okay. It's only once
                # per evaluator.
                log.info(f"{name}={format_float(value.item())}")

        # Restore model to train mode.
        self.trainer.model.train()
