import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from olmo_core.data import NumpyDatasetConfig, NumpyPaddedFSLDataset
from olmo_core.distributed.utils import get_world_size
from olmo_core.eval import Evaluator
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import format_float, move_to_device

from ..utils import Duration
from .callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from ..trainer import Trainer

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

    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """
    The duration to run each evaluator for.
    """

    log_interval: int = 5
    """
    How often to log eval progress to the console during an eval loop.
    """

    def post_step(self):
        if self.step <= 1 or self.step % self.eval_interval != 0:
            return

        # Put model in eval train mode.
        self.trainer.optim.zero_grad(set_to_none=True)
        self.trainer.model.eval()
        dp_world_size = get_world_size(self.trainer.dp_process_group)

        for evaluator in self.evaluators:
            log.info(f"Running {evaluator.name} evals...")
            evaluator.reset_metrics()
            eval_tokens = 0
            for eval_step, batch in enumerate(evaluator):
                eval_tokens += batch["input_ids"].numel() * dp_world_size
                batch = move_to_device(batch, self.trainer.device)
                with torch.no_grad():
                    ce_loss, _, logits = self.trainer._model_forward(
                        batch, loss_reduction="none", compute_z_loss=False
                    )
                evaluator.update_metrics(batch, ce_loss, logits)
                if (eval_step + 1) % self.log_interval == 0:
                    if evaluator.total_batches is not None:
                        log.info(
                            f"[eval={evaluator.name},step={eval_step+1}/{evaluator.total_batches}]"
                        )
                    else:
                        log.info(f"[eval={evaluator.name},step={eval_step+1}]")

                if (eval_step + 1) % self.trainer.cancel_check_interval == 0:
                    self.trainer.check_if_canceled()

                if self.trainer.is_canceled:
                    self.trainer.model.train()
                    return

                if self.eval_duration.due(step=eval_step + 1, tokens=eval_tokens, epoch=1):
                    break

            for name, value in evaluator.compute_metrics().items():
                name = f"eval/{evaluator.name}/{name}"
                self.trainer.record_metric(name, value)
                # NOTE: going to have a host-device sync here but that's okay. It's only once
                # per evaluator.
                log.info(f"{name}={format_float(value.item())}")

        # Restore model to train mode.
        self.trainer.model.train()


@dataclass
class LMEvaluatorCallbackConfig(CallbackConfig):
    eval_dataset: NumpyDatasetConfig
    eval_interval: int = 1000
    eval_batch_size: Optional[int] = None
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5

    def build(self, trainer: "Trainer") -> Callback:
        eval_batch_size = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else trainer.rank_microbatch_size * get_world_size(trainer.dp_process_group)
        )
        dataset = self.eval_dataset.build()
        if not isinstance(dataset, NumpyPaddedFSLDataset):
            raise OLMoConfigurationError(
                f"Expected a padded FSL dataset, got '{dataset.__class__.__name__}' instead"
            )
        evaluator = LMEvaluator.from_numpy_dataset(
            dataset,
            name="lm",
            global_batch_size=eval_batch_size,
            collator=trainer.collator,
            device=trainer.device,
        )
        return EvaluatorCallback(
            evaluators=[evaluator],
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )
