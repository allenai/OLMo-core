import contextlib
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.config import Config, DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import _swap_param_keys
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    ExpertParallelConfig,
    TensorParallelConfig,
    build_device_mesh,
    get_dp_mesh,
    get_dp_process_group,
    get_ep_mesh,
    get_tp_mesh,
)
from olmo_core.distributed.utils import (
    get_full_tensor,
    get_local_tensor,
    get_world_size,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config, Float8Handler
from olmo_core.nn.cross_entropy_loss import CrossEntropyLoss
from olmo_core.nn.transformer import (
    MoETransformer,
    NormalizedTransformer,
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, mark_dynamic, move_to_device

from ..common import ReduceType
from .train_module import EvalBatchSpec, TrainModule

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

    loss_parallel: bool = True


@dataclass
class TransformerExpertParallelConfig(ExpertParallelConfig):
    """
    Transformer-specific expert parallel config.
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
    max_sequence_length: int

    # Optimizer settings.

    optim: OptimConfig
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

    fused_loss: bool = False
    compile_loss: bool = False
    z_loss_multiplier: Optional[float] = None

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None

    # Other train settings.

    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> "TransformerTrainModule":
        """
        Build the corresponding :class:`TransformerTrainModule`.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return TransformerTrainModule(
            model=model,
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
    :param max_sequence_length: The maximum expected sequence length during training and evaluation.
    :param compile_model: Whether to compile to the model.
    :param float8_config: Float8 configuration for the model.
    :param dp_config: Data parallel configuration for the model.
    :param tp_config: Tensor parallel configuration for the model.
    :param ac_config: Activation checkpointing configuration for the model.
    :param compile_loss: Compile the loss function. This can provide a small speedup while also
        reducing GPU memory usage, especially when using Z-loss.

        .. important::
            This is incompatible with ``fused_loss=True``.
    :param fused_loss: Use the fused cross-entropy loss function (:func:`~olmo_core.nn.functional.fused_cross_entropy_loss`)
        instead the PyTorch built-in. This can help reduce GPU memory usage when ``compile_loss=False``.
        Relative performance will depend on the input sizes.

        .. important::
            This is incompatible with ``compile_loss=True``.
    :param z_loss_multiplier: Use Z-loss with this multiplier.
    :param autocast_precision: Enable AMP with this data type.
    :param max_grad_norm: Clip gradient norms to this value.
    :param scheduler: Optional learning rate scheduler for the optimizer.
    :param device: The device to train on.
    :param state_dict_save_opts: Can be used to override the state dict options used
        when saving a checkpoint.
    :param state_dict_load_opts: Can be used to override the state dict options used
        when loading a checkpoint.
    :param load_key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        compile_loss: bool = False,
        fused_loss: bool = False,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
    ):
        super().__init__()

        # Validate some options.
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        self.device = device or get_default_device()
        self.world_mesh = build_device_mesh(
            dp=dp_config, tp=tp_config, ep=ep_config, device_type=self.device.type
        )
        log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")

        self.label_ignore_index = label_ignore_index
        self._train_loss_fn = CrossEntropyLoss(
            ignore_index=label_ignore_index,
            reduction="sum",
            z_loss_multiplier=z_loss_multiplier,
            compile=compile_loss,
            fused=fused_loss,
        )
        self._eval_loss_fn = CrossEntropyLoss(
            ignore_index=label_ignore_index,
            reduction="none",
            compile=compile_loss,
            fused=fused_loss,
        )

        self.float8_handler: Optional[Float8Handler] = None
        float8_enabled = False
        if float8_config is not None:
            float8_enabled = float8_config.enabled
            float8_config.compile = compile_model
            self.float8_handler = float8_config.build()

        self.model = model

        # Maybe convert linear layers to FP8 linear.
        if self.float8_handler is not None and self.float8_handler.enabled:
            self.float8_handler.convert_to_float8_training(
                self.model, modules_to_ignore={"lm_head.w_out"}
            )
            log.info("Swapped linear layers to Float8 linear layers")

        # Maybe apply tensor/expert parallelism.
        self._tp_enabled = False
        if tp_config is not None and ep_config is not None:
            raise NotImplementedError("TP + EP is not implemented yet")
        if tp_config is not None:
            tp_mesh = get_tp_mesh(self.world_mesh)
            self.model.apply_tp(
                tp_mesh,
                float8_enabled=float8_enabled,
                loss_parallel=tp_config.loss_parallel,
            )
            if tp_config.loss_parallel:
                self._train_loss_fn.apply_tp(
                    tp_mesh,
                    input_layouts=(Shard(1), Replicate(), Replicate()),
                    use_local_output=True,
                )
                self._eval_loss_fn.apply_tp(
                    tp_mesh,
                    input_layouts=(Shard(1), Replicate()),
                    use_local_output=True,
                )
            tp_config.maybe_enable_async_tp(tp_mesh)
            log.info(
                f"Applied {'Float8 ' if float8_enabled else ''}tensor parallelism to the model"
            )
            self._tp_enabled = True

        self._ep_enabled = False
        if ep_config is not None:
            if not self.model.is_moe:
                raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
            ep_mesh = get_ep_mesh(self.world_mesh)
            cast(MoETransformer, self.model).apply_ep(ep_mesh)
            log.info("Applied expert parallelism to the model")
            self._ep_enabled = True

        # Maybe apply activation checkpointing.
        if ac_config is not None:
            self.model.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
            )
            log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

        # Maybe compile.
        if compile_model:
            if torch.cuda.is_available():
                self.model.apply_compile()
                log.info("Applied torch.compile() to the model")
            else:
                log.warning("Skipping model compilation since CUDA is not available")

        # Maybe shard/replicate according to data parallel config.
        self._dp_config = dp_config
        if dp_config is not None:
            dp_mesh = get_dp_mesh(self.world_mesh)
            if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
                self.model.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=dp_config.param_dtype.as_pt()
                    if dp_config.param_dtype is not None
                    else None,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=False,
                )
                log.info("Applied FSDP to the model")
            elif dp_config.name == DataParallelType.ddp:
                self.model.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model)
                log.info("Applied DDP to the model")
            else:
                raise NotImplementedError(dp_config.name)

        # Materialize and init parameters.
        log.info("Initializing model weights...")
        self.model.init_weights(
            max_seq_len=max_sequence_length,
            max_local_microbatch_size=rank_microbatch_size,
            device=self.device,
        )

        # Build optimizer(s).
        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)

        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.rank_microbatch_size,
            max_sequence_length=self.max_sequence_length,
            #  fixed_sequence_length=self.tp_enabled,
        )

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    def loss_fn(
        self, logits: torch.Tensor, labels: torch.Tensor, batch_num_tokens_for_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
        # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
        # to avoid biasing the loss in the case where micro-batches might not be the same size.
        ce_loss, z_loss = self._train_loss_fn(logits, labels, batch_num_tokens_for_loss)

        # Get loss to optimize for.
        loss = ce_loss
        if z_loss is not None:
            loss += z_loss

        return (
            loss,
            get_local_tensor(ce_loss.detach()),
            None if z_loss is None else get_local_tensor(z_loss.detach()),
        )

    def eval_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss, _ = self._eval_loss_fn(logits, labels)
        return ce_loss.view(logits.shape[0], -1)

    def on_attach(self):
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

    def state_dict(self) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts)

    def state_dict_to_load(self, metadata: Metadata) -> Dict[str, Any]:
        load_opts = self.state_dict_load_opts

        if "optim.param_groups.0.params" in metadata.state_dict_metadata:
            # unflattened optimizer state
            if load_opts.flatten_optimizer_state_dict:
                log.warning(
                    "Loading checkpoint with an unflattened optimizer state even though "
                    "'flatten_optimizer_state_dict=True' in train module's 'state_dict_load_opts', "
                    "automatically switching to 'flatten_optimizer_state_dict=False'."
                )
                load_opts = replace(load_opts, flatten_optimizer_state_dict=False)
        else:
            # flattened optimizer state
            if not load_opts.flatten_optimizer_state_dict:
                log.warning(
                    "Loading checkpoint with a flattened optimizer state even though "
                    "'flatten_optimizer_state_dict=False' in train module's 'state_dict_load_opts', "
                    "automatically switching to 'flatten_optimizer_state_dict=True'."
                )
                load_opts = replace(load_opts, flatten_optimizer_state_dict=True)

        state_dict = self._get_state_dict(load_opts)
        if self.load_key_mapping is not None:
            _swap_param_keys(state_dict, self.load_key_mapping, metadata=metadata)

        has_optim_state: bool = False
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("optim."):
                has_optim_state = True
                break

        if not has_optim_state:
            del state_dict["optim"]
            log.warning("No optimizer state found in checkpoint")

        return state_dict

    def state_dict_to_save(self) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.load_key_mapping is not None:
            _swap_param_keys(state_dict, self.load_key_mapping, reverse=True, quiet=True)
        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if "optim" in state_dict:
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self.optim,
                state_dict["optim"],
                options=self.state_dict_load_opts,
            )
            gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self.model.train()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = (batch["labels"] != self.label_ignore_index).sum()

        # Batch losses to record.
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self._train_loss_fn.z_loss_enabled:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        auxiliary_batch_losses: Dict[str, torch.Tensor] = {}

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

                # Get loss to optimize for, and the separate detached CE and Z loss values.
                loss, ce_loss, z_loss = self.loss_fn(
                    logits, micro_batch["labels"], batch_num_tokens_for_loss
                )
                del logits

                # Update total batch CE and Z loss.
                ce_batch_loss += ce_loss
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += z_loss
                    del z_loss

                # Optionally get model auxiliary losses and update the total batch auxiliary losses.
                auxiliary_losses = self.model.compute_auxiliary_losses(
                    batch_num_tokens_for_loss, reset=True
                )
                for loss_name, loss_val in auxiliary_losses.items():
                    loss += loss_val
                    loss_val = get_local_tensor(loss_val.detach())
                    if loss_name in auxiliary_batch_losses:
                        auxiliary_batch_losses[loss_name] += loss_val
                    else:
                        auxiliary_batch_losses[loss_name] = loss_val
                del auxiliary_losses

                # Run backward pass.
                loss.backward()

        del batch  # In case this helps with memory utilization.

        if dry_run:
            self.model.reset_auxiliary_losses()
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
        for loss_name, loss_val in auxiliary_batch_losses.items():
            self.record_metric(
                loss_name,
                loss_val,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            batch_num_tokens_for_loss
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )

        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = move_to_device(batch, self.device)

        self.model.eval()

        with torch.no_grad():
            logits = self.model_forward(batch)
            loss: Optional[torch.Tensor] = None
            if labels is not None:
                loss = self.eval_loss_fn(logits, labels)
                loss = get_full_tensor(loss)
            logits = get_full_tensor(logits)

        return logits, loss

    def optim_step(self):
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm

        # Sync Float8 AMAXs (argmax of abs(max)) and scales.
        if self.float8_handler is not None:
            self.float8_handler.sync_float8_amax_and_scale_history(self.model)

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                if (lr_field := self.scheduler.lr_field) not in group and (
                    initial_lr_field := self.scheduler.initial_lr_field
                ) not in group:
                    group_fields_list = "\n - ".join(
                        [f"{k}: {v}" for k, v in group.items() if k != "params"]
                    )
                    raise RuntimeError(
                        f"learning rate field '{lr_field}' and initial learning rate field "
                        f"'{initial_lr_field}' not found in optimizer param group {group_idx} "
                        f"with {len(group['params'])} parameter(s):\n"
                        f" - {group_fields_list}"
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

        # Maybe re-normalize matrices for nGPT-type models.
        # NOTE: sometimes 'isinstance' checks fail when the model is wrapped in some way.
        if isinstance(self.model, NormalizedTransformer) or hasattr(
            self.model, "normalize_matrices"
        ):
            cast(NormalizedTransformer, self.model).normalize_matrices()

        # Calculate Float8 dynamic AMAX/scale for all parameters.
        # For FSDP2 this issues a single all-reduce for all parameters at once.
        if self.float8_handler is not None:
            self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
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

            # Run model forward, get logits.
            # shape: (batch_size, seq_len, vocab_size)
            logits = self.model(
                input_ids=batch["input_ids"],
                #  attention_mask=micro_batch.get("attention_mask"),
                #  attention_bias=micro_batch.get("attention_bias"),
                doc_lens=batch.get("doc_lens"),
                max_doc_lens=batch.get("max_doc_lens"),
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
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _get_state_dict(self, sd_options: dist_cp_sd.StateDictOptions) -> Dict[str, Any]:
        return {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
            "optim": dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            ),
        }

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)

        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm
