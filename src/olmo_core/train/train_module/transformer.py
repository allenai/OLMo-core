import contextlib
import copy
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.pipelining import PipelineStage
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
    get_pp_mesh,
    get_tp_mesh,
)
from olmo_core.distributed.utils import (
    get_global_rank,
    get_local_tensor,
    get_world_size,
)
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

    split_points: Optional[List[int]] = None
    """
    A list of unique, increasing block indices that define how to split the model into stages.

    For example, ``split_points = [0, 2]`` with a 4-layer model means the model will be split into
    3 stages, with the first containing just the embedding, the second containing blocks 0 and 1,
    and the third containing blocks 2 and 3 and the language modeling head.

    If not specified the split points are determined automatically based on the schedule type.
    """

    def get_split_points(self, n_layers: int) -> List[int]:
        if self.split_points is not None:
            return self.split_points

        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified.
        num_stages_per_rank = 1 if self.schedule.is_single_stage else 2
        total_stages = self.degree * num_stages_per_rank
        num_layers = n_layers
        if total_stages > num_layers:
            raise OLMoConfigurationError("Total stages cannot be greater than the number of layers")

        base_interval = num_layers // total_stages
        extra_layers = num_layers % total_stages

        splits: List[int] = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                # Middle stages get an extra layer if there are any remaining
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append(current_layer)
        log.info(f"Auto generated pipeline split points will be {splits}")
        return splits

    def split_model(
        self, model: Transformer, *, pp_mesh: DeviceMesh, device: torch.device
    ) -> Tuple[List[PipelineStage], List[Transformer]]:
        split_points = self.get_split_points(model.n_layers)
        pp_rank = pp_mesh.get_local_rank()

        def build_stage(
            stage_idx: int,
            start_layer: Optional[int],
            stop_layer: Optional[int],
            is_first: bool = False,
            is_last: bool = False,
        ) -> Tuple[PipelineStage, Transformer]:
            model_chunk = copy.deepcopy(model)
            if not is_first:
                model_chunk.embeddings = None  # type: ignore

            drop_layers = start_layer is not None
            for block_idx in range(model.n_layers):
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if block_idx == start_layer:
                    drop_layers = False
                if block_idx == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model_chunk.blocks[str(block_idx)]

            if not is_last:
                model_chunk.lm_head = None  # type: ignore

            stage = PipelineStage(
                model_chunk,
                stage_idx,
                num_stages,
                device,
                group=pp_mesh.get_group("pp"),
            )
            return stage, model_chunk

        num_stages = len(split_points) + 1
        stage_idx = pp_rank

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(pp_rank, num_stages, style="loop"):
            start_layer = split_points[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = split_points[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = build_stage(
                stage_idx,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            log.info(
                f"PP rank {pp_rank} is building stage {stage_idx} with start layer "
                f"{start_layer}, stop layer {stop_layer}: {model_chunk}"
            )
            stages.append(stage)
            models.append(model_chunk)

        return stages, models


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

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None

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
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return TransformerTrainModule(
            model=model,
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
    :param max_seq_len: The maximum sequence length expected.
    :param device: The device to train on.
    :param state_dict_save_opts: Can be used to override the state dict options used
        when saving a checkpoint.
    :param state_dict_load_opts: Can be used to override the state dict options used
        when loading a checkpoint.
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
        compile_loss: bool = False,
        fused_loss: bool = False,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
    ):
        super().__init__()

        # Validate some options.
        if fused_loss and compile_loss:
            raise OLMoConfigurationError("'fused_loss' is not compatible with 'compile_loss'")

        self.device = device or get_default_device()
        self.world_mesh = build_device_mesh(
            dp=dp_config, tp=tp_config, pp=pp_config, device_type=self.device.type
        )
        log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")

        self.base_loss_fn = fused_cross_entropy_loss if fused_loss else cross_entropy_loss
        if compile_loss:
            self.base_loss_fn = torch.compile(self.base_loss_fn)

        self.float8_handler: Optional[Float8Handler] = None
        float8_enabled = False
        if float8_config is not None:
            float8_enabled = float8_config.enabled
            float8_config.compile = compile_model
            self.float8_handler = float8_config.build()

        self._pp_config = pp_config
        # We'll initialize this lazily when the trainer is attached, since we need to know
        # the global batch size in order to determine the number of pipeline micro-batches.
        self._pp_schedule: Optional[PipelineSchedule] = None
        self._pp_stages: Optional[List[PipelineStage]] = None

        self.model_parts: List[Transformer] = []
        if pp_config is not None:
            pp_mesh = get_pp_mesh(self.world_mesh)
            assert pp_mesh is not None
            stages, model_parts = pp_config.split_model(model, pp_mesh=pp_mesh, device=self.device)
            self._pp_stages = stages
            self.model_parts = model_parts
        else:
            self.model_parts = [model]

        # Maybe convert linear layers to FP8 linear.
        if self.float8_handler is not None and self.float8_handler.enabled:
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
                        pp_enabled=self.pp_enabled,
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
        self.optimizers: List[Optimizer] = [
            optim.build(model, strict=not self.pp_enabled) for model in self.model_parts
        ]

        self.rank_microbatch_size = rank_microbatch_size
        self.z_loss_multiplier = z_loss_multiplier
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=not self.pp_enabled
        )

        self.moe_handler: Optional[MoEHandler] = None
        for model in self.model_parts:
            if MoEHandler.has_moe(model):
                self.moe_handler = MoEHandler(model=model)
                if self.pp_enabled is not None:
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

    @property
    def pp_enabled(self) -> bool:
        return self._pp_config is not None

    @property
    def pp_schedule(self) -> Optional[PipelineSchedule]:
        self.trainer  # make sure trainer has been attached before trying to access this
        return self._pp_schedule

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert self._batch_num_tokens_for_loss is not None

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

        ce_loss.div_(self._batch_num_tokens_for_loss)
        if z_loss is not None:
            z_loss.div_(self._batch_num_tokens_for_loss)

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

    def on_attach(self):
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

        # Maybe initialize pipeline schedule.
        if self._pp_config is not None:
            assert self._pp_schedule is None  # make sure we don't initialize this twice
            assert self._pp_stages is not None
            pp_mesh = get_pp_mesh(self.world_mesh)
            assert pp_mesh is not None

            # Determine the number of micro-batches.
            rank_batch_size = self.trainer.global_batch_size // dp_ws
            num_micro_batches = rank_batch_size // self.rank_microbatch_size

            self._pp_schedule = PipelineSchedule(
                model_parts=self.model_parts,  # type: ignore[arg-type]
                stages=self._pp_stages,
                pp_mesh=pp_mesh,
                loss_fn=self.loss_fn,
                schedule_name=self._pp_config.schedule,
                n_microbatches=num_micro_batches,
            )

    def state_dict(self) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts)

    def state_dict_to_load(self) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_load_opts)

    def state_dict_to_save(self) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for model, optim in zip(self.model_parts, self.optimizers):
            dist_cp_sd.set_model_state_dict(
                model,
                state_dict["model"],
                options=dist_cp_sd.StateDictOptions(
                    strict=not self.pp_enabled, flatten_optimizer_state_dict=True
                ),
            )
            gc_cuda()
            dist_cp_sd.set_optimizer_state_dict(
                model,
                optim,
                state_dict["optim"],
                options=dist_cp_sd.StateDictOptions(
                    strict=not self.pp_enabled, flatten_optimizer_state_dict=True
                ),
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

        if not self.pp_enabled:
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

            # Broadcast loss from final pipeline stage to other stages.
            self._broadcast_pipeline_losses()

        del batch  # In case this helps with memory utilization.

        if dry_run:
            self._clear_loss_buffers()
            return

        # Record loss metrics.
        assert self._ce_batch_loss is not None
        self.record_ce_loss(self._ce_batch_loss, ReduceType.mean)
        if self._z_batch_loss is not None:
            self.record_metric("Z loss", self._z_batch_loss, ReduceType.mean, namespace="train")
        if self.moe_handler is not None:
            if (moe_lb_loss := self.moe_handler.get_lb_loss()) is not None:
                self.record_metric("load balancing loss", moe_lb_loss, namespace="train")
            if (moe_z_loss := self.moe_handler.get_z_loss()) is not None:
                self.record_metric("router Z loss", moe_z_loss, namespace="train")

        for optim in self.optimizers:
            if isinstance(optim, SkipStepOptimizer):
                optim.latest_loss = self._ce_batch_loss

        # Lastly, clear internal loss buffers.
        self._clear_loss_buffers()

    def eval_batch(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        batch = move_to_device(batch, self.device)

        for model in self.model_parts:
            model.eval()

        with torch.no_grad():
            logits = self.model_forward(batch)

        self._clear_loss_buffers()

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

    def _broadcast_pipeline_losses(self):
        assert self.pp_schedule is not None
        pp_mesh = self.pp_schedule.pp_mesh
        pp_group = pp_mesh.get_group()

        loss: torch.Tensor
        if self.pp_schedule.is_last_stage:
            assert self._ce_batch_loss is not None
            if self.z_loss_multiplier is None:
                loss = self._ce_batch_loss
            else:
                assert self._z_batch_loss is not None
                loss = torch.stack([self._ce_batch_loss, self._z_batch_loss])
        else:
            if self.z_loss_multiplier is None:
                loss = move_to_device(torch.tensor(0.0), self.device)
            else:
                loss = move_to_device(torch.tensor([0.0, 0.0]), self.device)

        src_rank = get_global_rank(pp_mesh.size() - 1, pp_group)
        dist.broadcast(loss, src_rank, group=pp_group)

        if not self.pp_schedule.is_last_stage:
            if self.z_loss_multiplier is None:
                self._ce_batch_loss = loss
            else:
                self._ce_batch_loss = loss[0]
                self._z_batch_loss = loss[1]

    def _clear_loss_buffers(self):
        self._batch_num_tokens_for_loss = None
        self._ce_batch_loss = None
        self._z_batch_loss = None
        if self.moe_handler is not None:
            self.moe_handler.clear_loss_buffers()

    def _get_state_dict(self, sd_options: dist_cp_sd.StateDictOptions) -> Dict[str, Any]:
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
