import contextlib
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Generator, List, Optional, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.config import Config, DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    PipelineParallelConfig,
    PipelineSchedule,
    TensorParallelConfig,
    build_device_mesh,
    get_dp_mesh,
    get_dp_process_group,
    get_tp_mesh,
)
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config, Float8Handler
from olmo_core.nn.functional.cross_entropy_loss import (
    cross_entropy_loss,
    fused_cross_entropy_loss,
)
from olmo_core.nn.moe import MoEHandler
from olmo_core.nn.transformer import (
    NormalizedTransformer,
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, mark_dynamic, move_to_device

from ..common import ReduceType, reshape_inputs_for_loss
from .train_module import TrainModule

log = logging.getLogger(__name__)


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    """
    Transformer-specific data parallel config.
    """

    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    The wrapping strategy.
    """


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


@dataclass
class TransformerPipelineParallelConfig(PipelineParallelConfig):
    """
    Transformer-specific pipeline parallel config.
    """


@beta_feature
@dataclass
class TransformerActivationCheckpointingConfig(Config):
    """
    Defines the activation checkpointing strategy for a transformer model.
    """

    mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.full
    """
    The activation checkpointing mode.
    """

    block_interval: Optional[int] = None
    """
    Required when :data:`mode` is "selected_blocks". Determines which blocks are wrapped.
    """

    modules: Optional[List[str]] = None
    """
    Required when :data:`mode` is "selected_modules". A list of modules names to wrap for
    activation checkpointing. Globs are supported.
    """

    def __post_init__(self):
        if (
            self.mode == TransformerActivationCheckpointingMode.selected_blocks
            and self.block_interval is None
        ):
            raise OLMoConfigurationError(
                "'block_interval' is required for 'selected_blocks' activation checkpointing"
            )
        elif (
            self.mode == TransformerActivationCheckpointingMode.selected_modules
            and self.modules is None
        ):
            raise OLMoConfigurationError(
                "'modules' is required for 'selected_modules' activation checkpointing"
            )


@dataclass
class TransformerTrainModuleConfig(Config):
    """
    A configuration class for building :class:`TransformerTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` documentation for a description of the fields.
    """

    rank_microbatch_size: int

    # Optimizer settings.

    optim: OptimConfig
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

    fused_loss: bool = False
    compile_loss: bool = False
    z_loss_multiplier: Optional[float] = None

    # Other train settings.

    autocast_precision: Optional[DType] = None

    def build(
        self,
        model: Transformer,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "TransformerTrainModule":
        """
        Build the corresponding :class:`TransformerTrainModule`.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param max_seq_len: The maximum sequence length expected.
        :param device: The device to train on.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        autocast_precision: Optional[DType] = kwargs.pop("autocast_precision", None)
        return TransformerTrainModule(
            model=model,
            autocast_precision=None if autocast_precision is None else autocast_precision.as_pt(),
            max_seq_len=max_seq_len,
            device=device,
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
    :param optim: The corresponding optimizer config.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param compile_model: Whether to compile to the model.
    :param float8_config: Float8 configuration for the model.
    :param dp_config: Data parallel configuration for the model.
    :param tp_config: Tensor parallel configuration for the model.
    :param pp_config: Pipeline parallel configuration for the model.
    :param ac_config: Activation checkpointing configuration for the model.
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
    :param max_seq_len: The maximum sequence length expected.
    :param device: The device to train on.
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        fused_loss: bool = False,
        compile_loss: bool = False,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_default_device()
        self.world_mesh = build_device_mesh(
            dp=dp_config, tp=tp_config, pp=pp_config, device_type=self.device.type
        )

        self.base_loss_fn = fused_cross_entropy_loss if fused_loss else cross_entropy_loss
        if compile_loss:
            self.base_loss_fn = torch.compile(self.base_loss_fn)

        self.float8_handler: Optional[Float8Handler] = None
        float8_enabled = False
        if float8_config is not None:
            float8_enabled = float8_config.enabled
            float8_config.compile = compile_model
            self.float8_handler = float8_config.build()

        # TODO: build pp schedule
        self.model_parts: List[Transformer] = []
        self.pp_schedule: Optional[PipelineSchedule] = None
        if pp_config is not None:
            #  self.pp_schedule = PipelineSchedule(..., pp_mesh=..., loss_fn=self.loss_fn)
            raise NotImplementedError
        else:
            self.model_parts = [model]

        # Maybe convert linear layers to FP8 linear.
        if self.float8_handler is not None:
            for model in self.model_parts:
                self.float8_handler.convert_to_float8_training(
                    model, modules_to_ignore={"lm_head.w_out"}
                )
            log.info("Swapped linear layers to Float8 linear layers")

        # Maybe apply tensor parallelism.
        if tp_config is not None:
            tp_mesh = get_tp_mesh(self.world_mesh)
            assert tp_mesh is not None
            for model in self.model_parts:
                model.apply_tp(
                    tp_mesh,
                    float8_enabled=float8_enabled,
                    loss_parallel=False,  # TODO (epwalsh): figure out if this will work w/ z-loss
                )
            tp_config.maybe_enable_async_tp(tp_mesh)
            log.info(
                f"Applied {'Float8 ' if float8_enabled else ''}tensor parallelism to the model"
            )

        # Maybe apply activation checkpointing.
        if ac_config is not None:
            for model in self.model_parts:
                model.apply_activation_checkpointing(
                    ac_config.mode,
                    block_interval=ac_config.block_interval,
                    modules=ac_config.modules,
                )
            log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

        # Maybe compile.
        if compile_model:
            for model in self.model_parts:
                model.apply_compile()
            log.info("Applied torch.compile() to the model")

        # Maybe shard/replicate according to data parallel config.
        if dp_config is not None:
            dp_mesh = get_dp_mesh(self.world_mesh)
            if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
                for model in self.model_parts:
                    model.apply_fsdp(
                        dp_mesh=dp_mesh,
                        param_dtype=dp_config.param_dtype.as_pt()
                        if dp_config.param_dtype is not None
                        else None,
                        reduce_dtype=dp_config.reduce_dtype.as_pt(),
                        wrapping_strategy=dp_config.wrapping_strategy,
                        pp_enabled=self.pp_schedule is not None,
                    )
                log.info("Applied FSDP to the model")
            elif dp_config.name == DataParallelType.ddp:
                for model in self.model_parts:
                    model.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model)
                log.info("Applied DDP to the model")
            else:
                raise NotImplementedError(dp_config.name)

        # Materialize and init parameters.
        for model in self.model_parts:
            log.info("Initializing model weights...")
            model.init_weights(max_seq_len=max_seq_len, device=self.device)

        # Build optimizer(s).
        log.info("Building optimizer(s)...")
        self.optimizers: List[Optimizer] = [optim.build(model) for model in self.model_parts]

        # Validate.
        if len(self.model_parts) != len(self.optimizers):
            raise OLMoConfigurationError("There must be one optimizer per model part")
        if len(self.model_parts) > 1 and self.pp_schedule is None:
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

        self.moe_handler: Optional[MoEHandler] = None
        for model in self.model_parts:
            if MoEHandler.has_moe(model):
                self.moe_handler = MoEHandler(model=model)
                if self.pp_schedule is not None:
                    # TODO (epwalsh): need to figure out how to handle the internal MoE losses correctly.
                    raise NotImplementedError(
                        "Pipeline parallelism with MoE's is currently not supported"
                    )
                break

        self._batch_num_tokens_for_loss: Optional[torch.Tensor] = None
        self._ce_batch_loss: Optional[torch.Tensor] = None
        self._z_batch_loss: Optional[torch.Tensor] = None

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return get_dp_process_group(self.world_mesh)

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
            self._ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        self._ce_batch_loss += get_local_tensor(ce_loss.detach())

        # Update overall Z batch loss.
        if z_loss is not None:
            if self._z_batch_loss is None:
                self._z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
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
                model,
                state_dict["model"],
                options=dist_cp_sd.StateDictOptions(strict=self.pp_schedule is None),
            )
            gc_cuda()
            dist_cp_sd.set_optimizer_state_dict(
                model,
                optim,
                state_dict["optim"],
                options=dist_cp_sd.StateDictOptions(strict=self.pp_schedule is None),
            )
            gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        for model in self.model_parts:
            model.train()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

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
        batch = move_to_device(batch, self.device)

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
        if self.float8_handler is not None:
            for model in self.model_parts:
                self.float8_handler.sync_float8_amax_and_scale_history(model)

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
        if self.float8_handler is not None:
            for model in self.model_parts:
                self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)

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
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield
