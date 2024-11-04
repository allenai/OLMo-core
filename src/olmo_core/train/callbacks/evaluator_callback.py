import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from olmo_core.data import NumpyDatasetConfig, NumpyPaddedFSLDataset, TokenizerConfig
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval import Evaluator
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import (
    cuda_sync_debug_mode,
    format_float,
    get_default_device,
    move_to_device,
)

from ..common import Duration
from .callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from olmo_eval import HFTokenizer

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
            eval_step = 0
            eval_tokens = 0
            for batch in evaluator:
                eval_step += 1
                eval_tokens += batch["input_ids"].numel() * dp_world_size
                batch = move_to_device(batch, self.trainer.device)
                logits, ce_loss, _ = self.trainer.eval_batch(
                    batch, loss_reduction="none", compute_z_loss=False
                )

                # NOTE: might have host-device syncs here but that's okay.
                with cuda_sync_debug_mode(0):
                    evaluator.update_metrics(batch, ce_loss, logits)

                if eval_step % self.trainer.cancel_check_interval == 0:
                    self.trainer.check_if_canceled()

                if self.trainer.is_canceled:
                    self._log_progress(evaluator, eval_step)
                    self.trainer.model.train()
                    return
                elif self.eval_duration.due(step=eval_step, tokens=eval_tokens, epoch=1):
                    self._log_progress(evaluator, eval_step)
                    break
                elif eval_step % self.log_interval == 0 or eval_step == evaluator.total_batches:
                    self._log_progress(evaluator, eval_step)

            # NOTE: going to have a host-device sync here but that's okay. It's only once
            # per evaluator.
            metrics = []
            with cuda_sync_debug_mode(0):
                for name, value in evaluator.compute_metrics().items():
                    value = value.item()
                    metrics.append(f"    {name}={format_float(value)}")
                    self.trainer.record_metric(f"eval/{evaluator.name}/{name}", value)
            log.info("Eval metrics:\n" + "\n".join(metrics))

        # Restore model to train mode.
        self.trainer.model.train()

    def _log_progress(self, evaluator: Evaluator, eval_step: int):
        if evaluator.total_batches is not None:
            log.info(f"[eval={evaluator.name},step={eval_step}/{evaluator.total_batches}]")
        else:
            log.info(f"[eval={evaluator.name},step={eval_step}]")


@dataclass
class LMEvaluatorCallbackConfig(CallbackConfig):
    eval_dataset: NumpyDatasetConfig
    eval_interval: int = 1000
    eval_batch_size: Optional[int] = None
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

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
            collator=trainer.data_loader.collator,
            device=trainer.device,
            dp_process_group=trainer.dp_process_group,
        )
        return EvaluatorCallback(
            evaluators=[evaluator],
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )


class DownstreamEvaluator(Evaluator):
    metric_type_to_label = {
        "f1": "F1 score",
        "acc": "accuracy",
        "len_norm": "length-normalized accuracy",
        "pmi_dc": "PMI-DC accuracy",
        "ce_loss": "CE loss",
        "bpb": "BPB",
    }

    def __init__(
        self,
        *,
        name: str,
        task: str,
        rank_batch_size: int,
        tokenizer: "HFTokenizer",
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        from olmo_eval import ICLMetric, build_task

        self.label = task
        self.task = build_task(task, tokenizer)
        self.metric = ICLMetric(metric_type=self.task.metric_type).to(
            device or get_default_device()
        )
        sampler: Optional[DistributedSampler] = None
        if is_distributed():
            sampler = DistributedSampler(
                self.task,  # type: ignore
                drop_last=True,
                shuffle=False,
                num_replicas=get_world_size(dp_process_group),
                rank=get_rank(dp_process_group),
            )

        rank_batch_size_instances = max(0, rank_batch_size // self.task.max_sequence_length)
        log.info(
            f"Using per-rank batch size of {rank_batch_size_instances} instances "
            f"for downstream eval task '{task}' with max sequence length {self.task.max_sequence_length:,d} tokens"
        )

        data_loader = DataLoader(
            self.task,  # type: ignore
            batch_size=rank_batch_size_instances,
            collate_fn=self.task.collate_fn,
            num_workers=0,
            sampler=sampler,
        )

        super().__init__(
            name=name, batches=data_loader, device=device, dp_process_group=dp_process_group
        )

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: torch.Tensor, logits: torch.Tensor
    ) -> None:
        del ce_loss
        self.metric.update(batch, logits)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        value = self.metric.compute()
        label = f"{self.label} ({self.metric_type_to_label[self.task.metric_type]})"
        return {label: value}

    def reset_metrics(self) -> None:
        self.metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    tasks: List[str]
    tokenizer: TokenizerConfig
    eval_batch_size: Optional[int] = None
    eval_interval: int = 1000
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        from olmo_eval import HFTokenizer

        global_eval_batch_size = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else trainer.rank_microbatch_size * get_world_size(trainer.dp_process_group)
        )
        rank_eval_batch_size = global_eval_batch_size // get_world_size(trainer.dp_process_group)
        if rank_eval_batch_size == 0:
            raise OLMoConfigurationError(
                f"'eval_batch_size' of {global_eval_batch_size:,d} tokens is too small for the given world size"
            )

        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Tokenizer 'identifier' required to build a concrete tokenizer"
            )

        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        evaluators: List[Evaluator] = []
        for task in self.tasks:
            evaluators.append(
                DownstreamEvaluator(
                    name="downstream",
                    task=task,
                    rank_batch_size=rank_eval_batch_size,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                )
            )

        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )
