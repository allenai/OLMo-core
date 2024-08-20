import logging
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..aliases import PathOrStr
from ..data.iterable_dataset import IterableDataset
from ..distributed.utils import get_fs_local_rank, get_world_size
from ..io import normalize_path
from ..nn.functional.cross_entropy_loss import (
    cross_entropy_loss,
    fused_cross_entropy_loss,
)
from ..utils import move_to_device
from .callbacks import Callback
from .checkpoint import Checkpointer
from .duration import Duration, DurationUnit
from .utils import EnvRngStates, ReduceType, reduce_metrics

log = logging.getLogger(__name__)


@dataclass
class Trainer:
    work_dir: PathOrStr
    model: nn.Module
    optim: Optimizer
    train_loader: DataLoader[IterableDataset]
    device: torch.device

    save_folder: str
    checkpointer: Checkpointer
    callbacks: List[Callback]
    max_duration: Duration

    train_sequence_length: int
    global_batch_size: int
    microbatch_size: int

    metrics_log_interval: int = 1
    metrics: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=lambda: defaultdict(dict))
    metrics_reduce_type: Dict[str, Optional[ReduceType]] = field(default_factory=dict)

    fused_loss: bool = False
    z_loss_multiplier: Optional[float] = None

    autocast_precision: Optional[torch.dtype] = None

    # Bookkeeping

    global_step: int = 0
    global_train_tokens_seen: int = 0
    global_train_tokens_seen_this_epoch: int = 0
    global_train_examples_seen_this_epoch: int = 0
    epoch: int = 0
    _canceled: bool = False
    _rank_batch_size: Optional[int] = None

    def __post_init__(self):
        self.save_folder = normalize_path(self.save_folder)

        self.work_dir = Path(self.work_dir)
        if get_fs_local_rank() == 0:
            self.work_dir.mkdir(parents=True, exist_ok=True)

        for callback in self.callbacks:
            callback.trainer = self

    @property
    def dataset(self) -> IterableDataset:
        assert isinstance(self.train_loader.dataset, IterableDataset)
        return self.train_loader.dataset

    @property
    def rank_batch_size(self) -> int:
        if self._rank_batch_size is None:
            assert self.global_batch_size % get_world_size() == 0
            self._rank_batch_size = self.global_batch_size // get_world_size()
        return self._rank_batch_size

    @property
    def training_complete(self) -> bool:
        if self._canceled:
            return True
        elif self.duration_due(self.max_duration):
            return True
        else:
            return False

    @property
    def tokens_per_batch(self) -> int:
        return self.global_batch_size * self.train_sequence_length

    @property
    def steps_per_epoch(self) -> int:
        return self.dataset.total_size // self.global_batch_size

    @property
    def tokens_per_epoch(self) -> int:
        return self.dataset.total_size * self.train_sequence_length

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

    def duration_due(self, duration: Duration) -> bool:
        if duration.unit == DurationUnit.steps:
            return self.global_step >= duration.value
        elif duration.unit == DurationUnit.epochs:
            return self.epoch >= duration.value
        elif duration.unit == DurationUnit.tokens:
            return self.global_train_tokens_seen >= duration.value
        else:
            raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            global_step=self.global_step,
            global_train_tokens_seen=self.global_train_tokens_seen,
            global_train_tokens_seen_this_epoch=self.global_train_tokens_seen_this_epoch,
            global_train_examples_seen_this_epoch=self.global_train_examples_seen_this_epoch,
            epoch=self.epoch,
            world_size=get_world_size(),
            train_sequence_length=self.train_sequence_length,
            rng=EnvRngStates.current_state().as_dict(),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
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

        if state_dict["world_size"] == get_world_size():
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
        self.checkpointer.load(
            dir,
            self.model,
            self.optim,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
        )

    def record_metric(
        self, name: str, value: torch.Tensor, reduce_type: Optional[ReduceType] = None
    ):
        self.metrics[self.global_step][name] = value
        self.metrics_reduce_type[name] = reduce_type

    def log_metrics(self):
        metrics: Dict[int, Dict[str, float]] = reduce_metrics(
            self.metrics, self.metrics_reduce_type, self.device
        )
        self.metrics.clear()
        self.metrics_reduce_type.clear()

        for step in sorted(metrics.keys()):
            for callback in self.callbacks:
                callback.log_metrics(step, metrics[step])

    def fit(self):
        self._canceled = False
        self.model.train()

        for callback in self.callbacks:
            callback.pre_train()

        while not self.training_complete:
            self._fit_epoch()

        for callback in self.callbacks:
            callback.post_train()

        log.info("Training complete")

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
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

    def model_forward(
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
        labels = self.get_labels(batch)
        # shape: (batch_size * (seq_len - 1),)
        labels = labels.view(-1)

        if self.fused_loss:
            ce_loss, z_loss = fused_cross_entropy_loss(
                logits,
                labels,
                ignore_index=-100,
                reduction=loss_reduction,
                compute_z_loss=compute_z_loss,
                z_loss_multiplier=self.z_loss_multiplier or 1e-4,
            )
        else:
            ce_loss, z_loss = cross_entropy_loss(
                logits,
                labels,
                ignore_index=-100,
                reduction=loss_reduction,
                compute_z_loss=compute_z_loss,
                z_loss_multiplier=self.z_loss_multiplier or 1e-4,
            )

        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)

        return ce_loss, z_loss, logits

    def train_micro_batch(
        self, micro_batch: Dict[str, Any], batch_size_in_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        ce_loss, z_loss, logits = self.model_forward(
            micro_batch, compute_z_loss=self.z_loss_multiplier is not None, loss_reduction="sum"
        )
        ce_loss = ce_loss / batch_size_in_tokens

        # In case this helps with memory utilization.
        del micro_batch

        # Get loss to optimize for.
        if self.z_loss_multiplier is not None:
            assert z_loss is not None
            z_loss = z_loss / batch_size_in_tokens
            loss = ce_loss + z_loss
        else:
            loss = ce_loss

        del logits

        return loss, ce_loss, z_loss

    def forward_backward(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = batch["input_ids"].numel()

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = (
            None if self.z_loss_multiplier is None else torch.tensor(0.0, device=self.device)
        )
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            # setup sync context for DDP for all micro-batches except the last
            grad_sync_context = nullcontext
            if isinstance(self.model, DDP) and micro_batch_idx != num_micro_batches - 1:
                grad_sync_context = self.model.no_sync

            with grad_sync_context():
                with torch.autocast(
                    self.device.type,
                    enabled=self.autocast_precision is not None,
                    dtype=self.autocast_precision,
                ):
                    # Run forward pass.
                    loss, ce_loss, z_loss = self.train_micro_batch(
                        micro_batch, batch_size_in_tokens
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

    def train_batch(self, batch: Dict[str, Any]):
        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.forward_backward(batch)
        self.record_metric("train/CrossEntropyLoss", ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric("train/ZLoss", z_batch_loss, ReduceType.mean)

        # Run through callbacks.
        for callback in self.callbacks:
            callback.pre_optim_step()

        # Optimizer steps.
        self.optim.step()

    def _fit_epoch(self):
        # Prepare dataset.
        self.dataset.reshuffle(self.epoch)
        self.dataset.start_index = self.global_train_examples_seen_this_epoch
        self.dataset.rank_batch_size = self.rank_batch_size

        for callback in self.callbacks:
            callback.pre_epoch()

        for batch in self.train_loader:
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
                callback.pre_step()

            self.train_batch(batch)

            for callback in self.callbacks:
                callback.post_train_batch()

            # TODO: evals

            for callback in self.callbacks:
                callback.post_step()

            if self.global_step % self.metrics_log_interval == 0:
                self.log_metrics()

        for callback in self.callbacks:
            callback.post_epoch()

        # Bookkeeping
        self.epoch += 1
        self.global_train_tokens_seen_this_epoch = 0
        self.global_train_examples_seen_this_epoch = 0
        self.dataset.start_index = 0

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
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
