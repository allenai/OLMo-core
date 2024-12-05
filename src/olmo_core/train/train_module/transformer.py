import contextlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Union, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.config import Config, DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.parallel import PipelineSchedule
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

from ..common import ReduceType, reshape_inputs_for_loss
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

    def build(
        self,
        model: Union[Transformer, List[Transformer]],
        optim: Union[Optimizer, List[Optimizer]],
        *,
        pp_schedule: Optional[PipelineSchedule] = None,
    ) -> "TransformerTrainModule":
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
            pp_schedule=pp_schedule,
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
        model: Union[Transformer, List[Transformer]],
        optim: Union[Optimizer, List[Optimizer]],
        rank_microbatch_size: int,
        fused_loss: bool = False,
        compile_loss: bool = False,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        pp_schedule: Optional[PipelineSchedule] = None,
    ):
        super().__init__()
        self.model_parts = model if isinstance(model, list) else [model]
        self.optimizers = optim if isinstance(optim, list) else [optim]
        if len(self.model_parts) != len(self.optimizers):
            raise OLMoConfigurationError("There must be one optimizer per model part")
        if len(self.model_parts) > 1 and pp_schedule is None:
            raise OLMoConfigurationError("Expected a single model without a pipeline schedule")
        if len(self.model_parts) > 1 and isinstance(self.model_parts[0], FSDP):
            raise OLMoConfigurationError(
                "FSDP(1) is not supported with pipeline parallelism, please use FSDP2"
            )

        self.rank_microbatch_size = rank_microbatch_size
        self.z_loss_multiplier = z_loss_multiplier
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.pp_schedule = pp_schedule

        self.base_loss_fn = fused_cross_entropy_loss if fused_loss else cross_entropy_loss
        if compile_loss:
            self.base_loss_fn = torch.compile(self.base_loss_fn)
        if self.pp_schedule is not None:
            self.pp_schedule.loss_fn = self.loss_fn

        self.moe_handler: Optional[MoEHandler] = None
        for model in self.model_parts:
            if MoEHandler.has_moe(model):
                self.moe_handler = MoEHandler(model=model)
                if pp_schedule is not None:
                    # TODO (epwalsh): need to figure out how to handle the internal MoE losses correctly.
                    raise NotImplementedError(
                        "Pipeline parallelism with MoE's is currently not supported"
                    )
                break

        self._batch_num_tokens_for_loss: Optional[torch.Tensor] = None
        self._ce_batch_loss = move_to_device(torch.tensor(0.0), self.trainer.device)
        self._z_batch_loss = (
            None
            if self.z_loss_multiplier is None
            else move_to_device(torch.tensor(0.0), self.trainer.device)
        )

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_num_tokens_for_loss = self._batch_num_tokens_for_loss
        assert batch_num_tokens_for_loss is not None

        logits_for_loss, labels_for_loss = reshape_inputs_for_loss(logits, labels)

        # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
        # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
        # to avoid biasing the loss in the case where micro-batches might not be the same size.
        ce_loss, z_loss = self.base_loss_fn(
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

        # Update overall CE batch loss.
        if self._ce_batch_loss is None:
            self._ce_batch_loss = move_to_device(torch.tensor(0.0), self.trainer.device)
        self._ce_batch_loss += get_local_tensor(ce_loss.detach())

        # Update overall Z batch loss.
        if z_loss is not None:
            if self._z_batch_loss is None:
                self._z_batch_loss = move_to_device(torch.tensor(0.0), self.trainer.device)
            self._z_batch_loss += get_local_tensor(z_loss.detach())

        return loss

    def get_ce_batch_loss(self) -> torch.Tensor:
        assert self._ce_batch_loss is not None
        return self._ce_batch_loss

    def get_z_batch_loss(self) -> Optional[torch.Tensor]:
        return self._z_batch_loss

    def clear_loss_buffers(self):
        self._batch_num_tokens_for_loss = None
        self._ce_batch_loss = None
        self._z_batch_loss = None
        if self.moe_handler is not None:
            self.moe_handler.clear_loss_buffers()

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
            "model": {
                k: v
                for sd in map(
                    partial(dist_cp_sd.get_model_state_dict, options=sd_options), self.model_parts
                )
                for k, v in sd.items()
            },
            "optim": {
                k: v
                for sd in map(
                    partial(dist_cp_sd.get_optimizer_state_dict, options=sd_options),
                    self.model_parts,
                    self.optimizers,
                )
                for k, v in sd.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for model, optim in zip(self.model_parts, self.optimizers):
            dist_cp_sd.set_model_state_dict(
                model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=False)
            )
            gc_cuda()
            dist_cp_sd.set_optimizer_state_dict(
                model,
                optim,
                state_dict["optim"],
                options=dist_cp_sd.StateDictOptions(strict=False),
            )
            gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        for model in self.model_parts:
            model.train()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.trainer.device)

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(
                batch, label_ignore_index=self.trainer.data_loader.collator.label_ignore_index
            )

        # Calculate how many tokens are going to be used in the loss.
        self._batch_num_tokens_for_loss = (
            batch["labels"] != self.trainer.data_loader.collator.label_ignore_index
        ).sum()

        if self.pp_schedule is None:
            # Split into micro-batches.
            if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
                raise RuntimeError(
                    f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
                )
            micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
            num_micro_batches = len(micro_batches)

            # Train one micro-batch at a time.
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                    # Run forward pass.
                    logits = self.model_forward(micro_batch)
                    assert logits is not None
                    loss = self.loss_fn(logits, micro_batch["labels"])

                    # Maybe add MoE losses.
                    if self.moe_handler is not None:
                        moe_loss = self.moe_handler.get_combined_loss(
                            batch=batch, micro_batch=micro_batch
                        )
                        if moe_loss is not None:
                            loss += moe_loss

                    # Run backward pass.
                    loss.backward()
        else:
            # Run pipeline schedule.
            self.model_forward(batch, labels=batch["labels"])

        del batch  # In case this helps with memory utilization.

        if dry_run:
            self.clear_loss_buffers()
            return

        # Record loss metrics.
        self.record_ce_loss(self.get_ce_batch_loss(), ReduceType.mean)
        if (z_batch_loss := self.get_z_batch_loss()) is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        if self.moe_handler is not None:
            if (moe_lb_loss := self.moe_handler.get_lb_loss()) is not None:
                self.record_metric("load balancing loss", moe_lb_loss, namespace="train")
            if (moe_z_loss := self.moe_handler.get_z_loss()) is not None:
                self.record_metric("router Z loss", moe_z_loss, namespace="train")

        for optim in self.optimizers:
            if isinstance(optim, SkipStepOptimizer):
                optim.latest_loss = self.get_ce_batch_loss()

        # Lastly, clear internal loss buffers.
        self.clear_loss_buffers()

    def eval_batch(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        batch = move_to_device(batch, self.trainer.device)

        for model in self.model_parts:
            model.eval()

        with torch.no_grad():
            logits = self.model_forward(batch)

        self.clear_loss_buffers()

        return logits

    def optim_step(self):
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm: torch.Tensor
            if self.pp_schedule is None:
                assert len(self.model_parts) == 1
                model = self.model_parts[0]
                if isinstance(model, FSDP):
                    grad_norm = model.clip_grad_norm_(self.max_grad_norm)
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            else:
                grad_norm = self.pp_schedule.clip_grad_norm_(self.max_grad_norm, foreach=True)

            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            for optim in self.optimizers:
                if isinstance(optim, SkipStepOptimizer):
                    optim.latest_grad_norm = grad_norm

        # Sync Float8 AMAXs (argmax of abs(max)) and scales.
        for model in self.model_parts:
            model.sync_float8_amax_and_scale_history()

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for optim in self.optimizers:
                for group_idx, group in enumerate(optim.param_groups):
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
        for optim in self.optimizers:
            optim.step()
            if isinstance(optim, SkipStepOptimizer):
                self.record_metric("step skipped", optim.step_skipped, namespace="optim")

        # Maybe re-normalize matrices for nGPT-type models.
        # NOTE: sometimes 'isinstance' checks fail when the model is wrapped in some way.
        for model in self.model_parts:
            if isinstance(model, NormalizedTransformer) or hasattr(model, "normalize_matrices"):
                cast(NormalizedTransformer, model).normalize_matrices()

        # Calculate Float8 dynamic AMAX/scale for all parameters.
        # For FSDP2 this issues a single all-reduce for all parameters at once.
        for model in self.model_parts:
            model.precompute_float8_dynamic_scale_for_fsdp()

    def zero_grads(self):
        for optim in self.optimizers:
            optim.zero_grad(set_to_none=True)

    def model_forward(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            # NOTE: Input sizes might be dynamic, e.g. when training with variable sequence lengths
            # or during an eval loop, so we mark them as dynamic for torch.compile up-front to avoid
            # recompiling later.
            # In theory this could harm performance a bit when input sizes are actually static
            # but so far I haven't noticed any dip in throughput with the models I've tested.
            mark_dynamic(batch["input_ids"], (0, 1))
            if "doc_lens" in batch:
                mark_dynamic(batch["doc_lens"], (0, 1))

            if self.pp_schedule is None:
                # shape: (batch_size, seq_len, vocab_size)
                return self.model_parts[0](
                    input_ids=batch["input_ids"],
                    #  attention_mask=micro_batch.get("attention_mask"),
                    #  attention_bias=micro_batch.get("attention_bias"),
                    doc_lens=batch.get("doc_lens"),
                    max_doc_lens=batch.get("max_doc_lens"),
                )
            else:
                # shape: (batch_size, seq_len, vocab_size), (num_micro_batches,)
                logits, losses = self.pp_schedule.step(
                    input_ids=batch["input_ids"],
                    #  attention_mask=micro_batch.get("attention_mask"),
                    #  attention_bias=micro_batch.get("attention_bias"),
                    target=labels,
                    doc_lens=batch.get("doc_lens"),
                    max_doc_lens=batch.get("max_doc_lens"),
                )
                del losses  # probably don't need this since we already track the loss in 'self.loss_fn'
                return logits

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.model_parts[0].num_flops_per_token(seq_len)

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            for model in self.model_parts:
                if isinstance(model, DDP) and micro_batch_idx != num_micro_batches - 1:
                    # For DDP, only sync gradients on the final micro batch.
                    stack.enter_context(model.no_sync())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(
                    torch.autocast(self.trainer.device.type, dtype=self.autocast_precision)
                )
            yield
