import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.config import Config, DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.functional.cross_entropy_loss import (
    cross_entropy_loss,
    fused_cross_entropy_loss,
)
from olmo_core.nn.moe import MoEHandler
from olmo_core.nn.transformer import NormalizedTransformer, Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, mark_dynamic, move_to_device

from ..common import ReduceType, get_inputs_for_loss
from .train_module import TrainModule


@dataclass
class TransformerTrainModuleConfig(Config):
    """
    A configuration class for building :class:`TransformerTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` documentation for a description of the fields.
    """

    rank_microbatch_size: int
    fused_loss: bool = False
    compile_loss: bool = False
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    def build(self, model: Transformer, optim: Optimizer) -> "TransformerTrainModule":
        """
        Build the corresponding :class:`TransformerTrainModule`.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param optim: The corresponding optimizer.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        autocast_precision: Optional[DType] = kwargs.pop("autocast_precision", None)
        return TransformerTrainModule(
            model=model,
            optim=optim,
            autocast_precision=None if autocast_precision is None else autocast_precision.as_pt(),
            **kwargs,
        )


class TransformerTrainModule(TrainModule):
    """
    A :class:`TrainModule` for any :class:`~olmo_core.nn.transformer.Transformer` model
    implementation provided by this library.

    .. tip::
        Use the :class:`TransformerTrainModuleConfig` to easily configure and build
        :class:`TransformerTrainModule` instances.

    :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
    :param optim: The corresponding optimizer.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param fused_loss: Use the fused cross-entropy loss function (:func:`~olmo_core.nn.functional.fused_cross_entropy_loss`)
        instead the PyTorch built-in. This can help reduce GPU memory usage. Relative performance will
        depend on the input sizes.

        .. seealso::
            Alternatively you could compile the loss function (``compile_loss=True``).
    :param compile_loss: Compile the loss function. This works best when ``fused_loss=False``.
    :param z_loss_multiplier: Use Z-loss with this multiplier.
    :param autocast_precision: Enable AMP with this data type.
    :param max_grad_norm: Clip gradient norms to this value.
    :param scheduler: Optional learning rate scheduler for the optimizer.
    """

    def __init__(
        self,
        model: Transformer,
        optim: Optimizer,
        rank_microbatch_size: int,
        fused_loss: bool = False,
        compile_loss: bool = False,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
    ):
        super().__init__()
        self.model = model
        self.optim = optim
        self.rank_microbatch_size = rank_microbatch_size
        self.loss_fn = fused_cross_entropy_loss if fused_loss else cross_entropy_loss
        if compile_loss:
            self.loss_fn = torch.compile(self.loss_fn)
        self.z_loss_multiplier = z_loss_multiplier
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.moe_handler: Optional[MoEHandler] = None
        if MoEHandler.has_moe(self.model):
            self.moe_handler = MoEHandler(model=self.model)

    def on_attach(self):
        # Validate batch size.
        if (
            self.trainer.global_batch_size
            % (self.rank_microbatch_size * (ws := get_world_size(self.trainer.dp_process_group)))
            != 0
        ):
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({ws})"
            )

    def state_dict(self) -> Dict[str, Any]:
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=True)
        return {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
            "optim": dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            ),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dist_cp_sd.set_model_state_dict(
            self.model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=True)
        )
        gc_cuda()
        dist_cp_sd.set_optimizer_state_dict(
            self.model,
            self.optim,
            state_dict["optim"],
            options=dist_cp_sd.StateDictOptions(strict=True),
        )
        gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self.model.train()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.trainer.device)

        # Generate labels, calculate how many tokens are going to be use in the loss.
        if "labels" not in batch:
            batch["labels"] = get_labels(
                batch, label_ignore_index=self.trainer.data_loader.collator.label_ignore_index
            )
        batch_num_tokens_for_loss = (
            batch["labels"] != self.trainer.data_loader.collator.label_ignore_index
        ).sum()

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.trainer.device)
        z_batch_loss = (
            None
            if self.z_loss_multiplier is None
            else move_to_device(torch.tensor(0.0), self.trainer.device)
        )

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                # Run forward pass.
                logits = self.model_forward(micro_batch)

                # shape: (batch_size * (seq_len - 1), vocab_size), (batch_size * (seq_len - 1),)
                logits_for_loss, labels_for_loss = get_inputs_for_loss(
                    micro_batch,
                    logits,
                    label_ignore_index=self.trainer.data_loader.collator.label_ignore_index,
                )

                # Calculate loss.
                # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
                # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
                # to avoid biasing the loss in the case where micro-batches might not be the same size.
                ce_loss, z_loss = self.loss_fn(
                    logits_for_loss,
                    labels_for_loss,
                    ignore_index=self.trainer.data_loader.collator.label_ignore_index,
                    reduction="sum",
                    compute_z_loss=self.z_loss_multiplier is not None,
                    z_loss_multiplier=self.z_loss_multiplier or 1e-4,
                )
                ce_loss.div_(batch_num_tokens_for_loss)
                if z_loss is not None:
                    z_loss.div_(batch_num_tokens_for_loss)

                # Get loss to optimize for.
                loss = ce_loss
                if z_loss is not None:
                    loss += z_loss
                if self.moe_handler is not None:
                    moe_loss = self.moe_handler.get_combined_loss(
                        batch=batch, micro_batch=micro_batch
                    )
                    if moe_loss is not None:
                        loss += moe_loss

                # Update overall CE batch loss.
                ce_batch_loss += get_local_tensor(ce_loss.detach())

                # Update overall Z batch loss.
                if z_loss is not None:
                    assert z_batch_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())

                # Run backward pass.
                loss.backward()

        # In case this helps with memory utilization.
        del batch

        if dry_run:
            return

        # Record loss metrics.
        self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        if self.moe_handler is not None:
            if (moe_lb_loss := self.moe_handler.get_lb_loss()) is not None:
                self.record_metric("load balancing loss", moe_lb_loss, namespace="train")
            if (moe_z_loss := self.moe_handler.get_z_loss()) is not None:
                self.record_metric("router Z loss", moe_z_loss, namespace="train")
            self.moe_handler.clear_loss_buffers()

        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss

    def eval_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        self.model.eval()
        batch = move_to_device(batch, self.trainer.device)
        with torch.no_grad():
            logits = self.model_forward(batch)
        if self.moe_handler is not None:
            self.moe_handler.clear_loss_buffers()
        return logits

    def optim_step(self):
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            if isinstance(self.model, FSDP):
                grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm

        # Sync Float8 AMAXs (argmax of abs(max)) and scales.
        self.model.sync_float8_amax_and_scale_history()

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                if (lr_field := self.scheduler.lr_field) not in group and (
                    initial_lr_field := self.scheduler.initial_lr_field
                ) not in group:
                    raise RuntimeError(
                        f"learning rate field '{lr_field}' and initial learning rate field "
                        f"'{initial_lr_field}' not found in optimizer param group"
                    )

                # Ensure 'initial_lr' is set.
                if group.get(self.scheduler.initial_lr_field) is None:
                    group[self.scheduler.initial_lr_field] = group["lr"]

                # Set new LR.
                new_lr = self.scheduler.get_lr(
                    group[self.scheduler.initial_lr_field],
                    self.trainer.global_step,
                    self.trainer.max_steps,
                )

                if isinstance(current_lr := group.get(self.scheduler.lr_field), torch.Tensor):
                    current_lr.fill_(new_lr)
                else:
                    group[self.scheduler.lr_field] = new_lr

                self.trainer.record_metric(
                    f"LR (group {group_idx})", group[self.scheduler.lr_field], namespace="optim"
                )

        # Step optimizer.
        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        # NOTE: sometimes 'isinstance' checks fail when the model is wrapped in some way.
        if isinstance(self.model, NormalizedTransformer) or hasattr(
            self.model, "normalize_matrices"
        ):
            self.model.normalize_matrices()

        # Calculate Float8 dynamic AMAX/scale for all parameters.
        # For FSDP2 this issues a single all-reduce for all parameters at once.
        self.model.precompute_float8_dynamic_scale_for_fsdp()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(self, micro_batch: Dict[str, Any]) -> torch.Tensor:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            # NOTE: Input sizes might be dynamic, e.g. when training with variable sequence lengths
            # or during an eval loop, so we mark them as dynamic for torch.compile up-front to avoid
            # recompiling later.
            # In theory this could harm performance a bit when input sizes are actually static
            # but so far I haven't noticed any dip in throughput with the models I've tested.
            mark_dynamic(micro_batch["input_ids"], (0, 1))
            if "doc_lens" in micro_batch:
                mark_dynamic(micro_batch["doc_lens"], (0, 1))

            # shape: (batch_size, seq_len, vocab_size)
            logits = self.model(
                input_ids=micro_batch["input_ids"],
                #  attention_mask=micro_batch.get("attention_mask"),
                #  attention_bias=micro_batch.get("attention_bias"),
                doc_lens=micro_batch.get("doc_lens"),
                max_doc_lens=micro_batch.get("max_doc_lens"),
            )
        return logits

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.model.num_flops_per_token(seq_len)

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
                stack.enter_context(
                    torch.autocast(self.trainer.device.type, dtype=self.autocast_precision)
                )
            yield
