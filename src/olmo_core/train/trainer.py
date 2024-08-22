import contextlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..aliases import PathOrStr
from ..data import DataCollator, IterableDataset, MemMapDataset
from ..distributed.utils import (
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
)
from ..exceptions import OLMoConfigurationError
from ..io import normalize_path
from ..nn.functional.cross_entropy_loss import (
    cross_entropy_loss,
    fused_cross_entropy_loss,
)
from ..utils import gc_cuda, move_to_device
from .callbacks import (
    Callback,
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
)
from .checkpoint import Checkpointer
from .utils import Duration, DurationUnit, EnvRngStates, ReduceType, reduce_metrics

log = logging.getLogger(__name__)

TRAIN_CE_LOSS_METRIC = "train/CE loss"
TRAIN_PPL_METRIC = "train/PPL"
TRAIN_Z_LOSS_METRIC = "train/Z loss"


@dataclass
class Trainer:
    """
    A language model trainer.
    """

    work_dir: PathOrStr
    """
    A local directory to use for temporary files needed during training.
    """

    model: nn.Module
    """
    The model to fit.
    """

    optim: Optimizer
    """
    The optimizer to use.
    """

    dataset: MemMapDataset
    """
    The training dataset.
    """

    collator: DataCollator
    """
    The data collator.
    """

    device: torch.device
    """
    The default device to use. Should match the device the model is on and be appropriate for the
    distributed backend, if there is one.
    """

    save_folder: str
    """
    The folder to save all checkpoints to. Could be a local directory (if using a shared filesytem)
    or a URL.
    """

    checkpointer: Checkpointer
    """
    The checkpointer.
    """

    callbacks: List[Callback]
    """
    Trainer callbacks.
    """

    max_duration: Duration
    """
    The duration to train for.
    """

    train_sequence_length: int
    """
    Training sequence length.
    """

    global_batch_size: int
    """
    Global training batch size (in terms of instances, not tokens).
    """

    microbatch_size: int
    """
    Microbatch size per rank, i.e. the number of instances to process at a time from each rank.
    """

    metrics_log_interval: int = 1
    """
    How often (in steps) to collect and log metrics.
    Increasing this can improve throughput.
    """

    fused_loss: bool = False
    """
    Used the fused cross-entropy loss function.
    """

    z_loss_multiplier: Optional[float] = None
    """
    Use Z-loss with this multiplier.
    """

    autocast_precision: Optional[torch.dtype] = None
    """
    Enable AMP with this data type.
    """

    dp_process_group: Optional[dist.ProcessGroup] = None
    """
    The distributed process group for all data parallel ranks.
    """

    data_seed: int = 0
    """
    The seed to use to shuffle the dataset.
    """

    data_loader_workers: int = 0
    """
    The number of data loading workers to use.
    """

    data_loader_prefetch_factor: Optional[int] = None
    """
    The number of batches to prefetch.
    """

    # Bookkeeping

    global_step: int = 0
    """
    The current step (1-based), though it's initialized to 0 before the first step.
    """

    global_train_tokens_seen: int = 0
    """
    The total number of training tokens seen.
    """

    global_train_tokens_seen_this_epoch: int = 0
    """
    The total number of training tokens seen in the current epoch.
    """

    global_train_examples_seen_this_epoch: int = 0
    """
    The total number of examples seen in the current epoch.
    """

    epoch: int = 1
    """
    The current epoch (1-based).
    """

    _metrics: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=lambda: defaultdict(dict))
    _metrics_reduce_type: Dict[str, Optional[ReduceType]] = field(default_factory=dict)
    _canceled: bool = False
    _rank_batch_size: Optional[int] = None

    def __post_init__(self):
        self.save_folder = normalize_path(self.save_folder)

        # Configure working directory.
        self.work_dir = Path(self.work_dir)

        # Ensure we have necessary callbacks.
        has_console_logger_callback = False
        has_checkpointer_callback = False
        has_speed_monitor_callback = False
        has_gc_collector_callback = False
        for callback in self.callbacks:
            if isinstance(callback, ConsoleLoggerCallback):
                has_console_logger_callback = True
            elif isinstance(callback, CheckpointerCallback):
                has_checkpointer_callback = True
            elif isinstance(callback, SpeedMonitorCallback):
                has_speed_monitor_callback = True
            elif isinstance(callback, GarbageCollectorCallback):
                has_gc_collector_callback = True
        if not has_console_logger_callback:
            self.callbacks.append(ConsoleLoggerCallback(log_interval=self.metrics_log_interval))
        if not has_checkpointer_callback:
            self.callbacks.append(CheckpointerCallback())
        if not has_speed_monitor_callback:
            self.callbacks.append(SpeedMonitorCallback())
        if is_distributed() and not has_gc_collector_callback:
            self.callbacks.append(GarbageCollectorCallback())

        # Set pointer to self in all callbacks.
        for callback in self.callbacks:
            callback.trainer = self

        # Other validation.
        if isinstance(self.dataset, MemMapDataset):
            if self.dataset.sequence_length != self.train_sequence_length:
                raise OLMoConfigurationError("trainer and dataset sequence length does not match")

    @property
    def rank_batch_size(self) -> int:
        if self._rank_batch_size is None:
            assert self.global_batch_size % get_world_size(self.dp_process_group) == 0
            self._rank_batch_size = self.global_batch_size // get_world_size(self.dp_process_group)
        return self._rank_batch_size

    @property
    def training_complete(self) -> bool:
        if self._canceled:
            return True
        elif self._duration_due(self.max_duration):
            return True
        else:
            return False

    @property
    def tokens_per_batch(self) -> int:
        return self.global_batch_size * self.train_sequence_length

    @property
    def steps_per_epoch(self) -> int:
        return self.dataset_total_size // self.global_batch_size

    @property
    def tokens_per_epoch(self) -> int:
        return self.dataset_total_size * self.train_sequence_length

    @property
    def dataset_total_size(self) -> int:
        dp_world_size = get_world_size(self.dp_process_group)
        if len(self.dataset) % dp_world_size == 0:
            return len(self.dataset) // dp_world_size
        else:
            return math.ceil((len(self.dataset) - dp_world_size) / dp_world_size)

    @property
    def max_steps(self) -> int:
        if self.max_duration.unit == DurationUnit.steps:
            return self.max_duration.value
        elif self.max_duration.unit == DurationUnit.epochs:
            # Need to account for a change in batch size.
            max_epochs = self.max_duration.value
            complete_epochs_remaining = max_epochs - self.epoch
            steps_remaining = complete_epochs_remaining * self.steps_per_epoch
            if self.global_train_tokens_seen_this_epoch > 0:
                tokens_remaining_this_epoch = max(
                    self.tokens_per_epoch - self.global_train_tokens_seen_this_epoch, 0
                )
                steps_remaining += math.ceil(tokens_remaining_this_epoch / self.tokens_per_batch)
            return self.global_step + steps_remaining
        elif self.max_duration.unit == DurationUnit.tokens:
            # Need to account for a change in batch size.
            max_tokens = self.max_duration.value
            tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
            steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
            return self.global_step + steps_remaining
        else:
            raise NotImplementedError

    def fit(self):
        """
        Fit the model.
        """
        self._canceled = False
        self.model.train()

        for callback in self.callbacks:
            callback.pre_train()

        try:
            while not self.training_complete:
                self._fit_epoch()
        except BaseException as exc:
            for callback in self.callbacks:
                callback.on_error(exc)
            raise

        for callback in self.callbacks:
            callback.post_train()

        log.info("Training complete")

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the trainer state to save.
        """
        return dict(
            global_step=self.global_step,
            global_train_tokens_seen=self.global_train_tokens_seen,
            global_train_tokens_seen_this_epoch=self.global_train_tokens_seen_this_epoch,
            global_train_examples_seen_this_epoch=self.global_train_examples_seen_this_epoch,
            epoch=self.epoch,
            world_size=get_world_size(),  # global world size here on purpose
            train_sequence_length=self.train_sequence_length,
            rng=EnvRngStates.current_state().as_dict(),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load trainer state (not model or optimizer state).
        """
        if state_dict["train_sequence_length"] != self.train_sequence_length:
            raise NotImplementedError(
                "Restoring training state with a different sequence length is not supported"
            )

        self.train_sequence_length = state_dict["train_sequence_length"]
        self.global_step = state_dict["global_step"]
        self.global_train_tokens_seen = state_dict["global_train_tokens_seen"]
        self.global_train_tokens_seen_this_epoch = state_dict["global_train_tokens_seen_this_epoch"]
        self.global_train_examples_seen_this_epoch = state_dict[
            "global_train_examples_seen_this_epoch"
        ]
        self.epoch = state_dict["epoch"]

        if state_dict["world_size"] == get_world_size():  # global world size here on purpose
            rng_state = EnvRngStates.from_dict(state_dict["rng"])
            if not rng_state.restore():
                log.warning(
                    "Some RNG states were not restored due to differences in library versions"
                )
        else:
            log.warning(
                "Trainer will not restore rank RNG states since the RNG states in the checkpoint "
                "were saved with a different world size."
            )

    def load_checkpoint(
        self, dir: PathOrStr, load_optimizer_state: bool = True, load_trainer_state: bool = True
    ):
        """
        Load a checkpoint.

        :param dir: The path/URL to the checkpoint.
        :param load_optimizer_state: Load optimizer state.
        :param load_trainer_state: Load trainer state.
        """
        log.info(f"Loading checkpoint from '{dir}'...")
        trainer_state = self.checkpointer.load(
            dir,
            self.model,
            self.optim,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
        )
        if load_trainer_state:
            assert trainer_state is not None
            self.load_state_dict(trainer_state)
        log.info("Checkpoint successfully loaded")

    def record_metric(
        self, name: str, value: Union[float, torch.Tensor], reduce_type: Optional[ReduceType] = None
    ):
        """
        Record a new metric for the current step.

        :param name: The name of the metric.
        :param value: The value of the metric.
        :param reduce_type: Specifies how to reduce the metric across the distributed process group.
            ``None`` means no reduction.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self._metrics[self.global_step][name] = value
        self._metrics_reduce_type[name] = reduce_type

    def _duration_due(self, duration: Duration) -> bool:
        if duration.unit == DurationUnit.steps:
            return self.global_step > duration.value
        elif duration.unit == DurationUnit.epochs:
            return self.epoch > duration.value
        elif duration.unit == DurationUnit.tokens:
            return self.global_train_tokens_seen >= duration.value
        else:
            raise NotImplementedError

    def _log_metrics(self):
        if not self._metrics:
            return

        # Check for NaN loss. We can afford the host-device sync here.
        for step in sorted(self._metrics.keys()):
            if TRAIN_CE_LOSS_METRIC in self._metrics[step]:
                if self._metrics[step][TRAIN_CE_LOSS_METRIC].isnan():
                    raise RuntimeError(f"NaN loss encountered at step {step}")

        metrics: Dict[int, Dict[str, float]] = reduce_metrics(
            self._metrics,
            self._metrics_reduce_type,
            self.device,
            process_group=self.dp_process_group,
        )
        self._metrics.clear()
        self._metrics_reduce_type.clear()
        gc_cuda()

        for step in sorted(metrics.keys()):
            # Add perplexity.
            if TRAIN_CE_LOSS_METRIC in metrics[step]:
                metrics[step][TRAIN_PPL_METRIC] = math.exp(metrics[step][TRAIN_CE_LOSS_METRIC])
            for callback in self.callbacks:
                callback.log_metrics(step, metrics[step])

    def _get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def _model_forward(
        self,
        batch: Dict[str, Any],
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        compute_z_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.model(
            input_ids=batch["input_ids"],
            #  attention_mask=batch.get("attention_mask"),
            #  attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        )

        # shape: (batch_size, seq_len - 1, vocab_size)
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * (seq_len - 1), vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len - 1)
        labels = batch.get("labels", self._get_labels(batch))
        # shape: (batch_size * (seq_len - 1),)
        labels = labels.view(-1)

        loss_fn = cross_entropy_loss if not self.fused_loss else fused_cross_entropy_loss
        ce_loss, z_loss = loss_fn(
            logits_for_loss,
            labels,
            ignore_index=-100,
            reduction=loss_reduction,
            compute_z_loss=compute_z_loss,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if loss_reduction == "none":
            # Reshape (batch_size * (seq_len - 1),) -> (batch_size, seq_len - 1)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)

        return ce_loss, z_loss, logits

    def _get_micro_batch_loss(
        self, micro_batch: Dict[str, Any], batch_num_tokens_for_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
        # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
        # to avoid biasing the loss in the case where micro-batches might not be the same size.
        ce_loss, z_loss, logits = self._model_forward(
            micro_batch, compute_z_loss=self.z_loss_multiplier is not None, loss_reduction="sum"
        )
        ce_loss = ce_loss / batch_num_tokens_for_loss

        # In case this helps with memory utilization.
        del micro_batch

        # Get loss to optimize for.
        if self.z_loss_multiplier is not None:
            assert z_loss is not None
            z_loss = z_loss / batch_num_tokens_for_loss
            loss = ce_loss + z_loss
        else:
            loss = ce_loss

        del logits

        return loss, ce_loss, z_loss

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, DDP) and micro_batch_idx != num_micro_batches - 1:
                # For DDP, only sync gradients on the final micro batch.
                stack.enter_context(self.model.no_sync())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _model_forward_backward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Generate labels, calculate how many tokens are going to be use in the loss.
        if "labels" not in batch:
            batch["labels"] = self._get_labels(batch)
        batch_num_tokens_for_loss = (batch["labels"] != -100).sum()

        # Split into micro-batches.
        micro_batches = self._split_batch(batch)
        num_micro_batches = len(micro_batches)

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = (
            None if self.z_loss_multiplier is None else torch.tensor(0.0, device=self.device)
        )

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                with self._model_forward_context():
                    # Run forward pass.
                    loss, ce_loss, z_loss = self._get_micro_batch_loss(
                        micro_batch, batch_num_tokens_for_loss
                    )

                    # Update overall CE batch loss.
                    ce_batch_loss += ce_loss.detach()

                    # Update overall Z batch loss.
                    if z_loss is not None:
                        assert z_batch_loss is not None
                        z_batch_loss += z_loss.detach()

                # Run backward pass.
                loss.backward()

        return ce_batch_loss, z_batch_loss

    def _train_batch(self, batch: Dict[str, Any]):
        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self._model_forward_backward(batch)
        self.record_metric(TRAIN_CE_LOSS_METRIC, ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric(TRAIN_Z_LOSS_METRIC, z_batch_loss, ReduceType.mean)

        # Run through callbacks.
        for callback in self.callbacks:
            callback.pre_optim_step()

        # Optimizer steps.
        self.optim.step()

    def _get_dataloader(self) -> DataLoader:
        iterable_dataset = IterableDataset(
            self.dataset,
            seed=self.data_seed,
            epoch=self.epoch,
            start_index=self.global_train_examples_seen_this_epoch,
            rank_batch_size=self.rank_batch_size,
            dp_world_size=get_world_size(self.dp_process_group),
            dp_rank=get_rank(self.dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            drop_last=True,
            work_dir=self.work_dir,
        )
        iterable_dataset.build_and_save_global_indices()
        return DataLoader(
            iterable_dataset,
            batch_size=self.rank_batch_size,
            drop_last=True,
            collate_fn=self.collator,
            num_workers=self.data_loader_workers,
            pin_memory=True,
            prefetch_factor=self.data_loader_prefetch_factor,
            persistent_workers=False,
            timeout=0,
        )

    def _fit_epoch(self):
        for callback in self.callbacks:
            callback.pre_epoch()

        for batch in self._get_dataloader():
            # Bookkeeping.
            # NOTE: To track the global batch size / number of tokens per batch we make the
            # assumption that all batches see the same number of tokens, which should always be
            # the case except for potentially the last batch in an epoch if drop_last=False.
            # Alternatively we'd have to use a distributed collective which isn't worth it.
            batch_size, seq_len = batch["input_ids"].shape
            assert seq_len == self.train_sequence_length
            assert batch_size == self.rank_batch_size
            self.global_step += 1
            self.global_train_tokens_seen_this_epoch += self.global_batch_size * seq_len
            self.global_train_examples_seen_this_epoch += self.global_batch_size
            self.global_train_tokens_seen += self.global_batch_size * seq_len

            for callback in self.callbacks:
                callback.pre_step(batch)

            self._train_batch(batch)

            for callback in self.callbacks:
                callback.post_train_batch()

            # TODO: evals

            for callback in self.callbacks:
                callback.post_step()

            if self.global_step % self.metrics_log_interval == 0:
                self._log_metrics()

        # Log any remaining metrics.
        if self._metrics:
            self._log_metrics()

        for callback in self.callbacks:
            callback.post_epoch()

        # Bookkeeping
        self.epoch += 1
        self.global_train_tokens_seen_this_epoch = 0
        self.global_train_examples_seen_this_epoch = 0

    def _split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= self.microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(self.microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[
                            self.microbatch_size * i : self.microbatch_size * i
                            + self.microbatch_size
                        ]
                        for i in range(math.ceil(batch_size / self.microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]
