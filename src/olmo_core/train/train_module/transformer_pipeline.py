import contextlib
import copy
import logging
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from olmo_core.config import Config, DType
from olmo_core.data.utils import get_labels
from olmo_core.distributed.checkpoint import _swap_param_keys
from olmo_core.distributed.parallel import (
    DataParallelType,
    PipelineParallelConfig,
    PipelineSchedule,
    build_world_mesh,
    get_cp_mesh,
    get_device_mesh_info,
    get_dp_model_mesh,
    get_dp_process_group,
    get_pp_mesh,
    get_tp_mesh,
)
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config, Float8Handler
from olmo_core.nn.attention import RingAttentionLoadBalancer
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import MoETransformer, NormalizedTransformer, Transformer
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, log_once, move_to_device

from ..common import ReduceType
from .train_module import EvalBatchSizeUnit, EvalBatchSpec, TrainModule
from .transformer import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


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


@dataclass
class TransformerPipelineTrainModuleConfig(Config):
    """
    A configuration class for building :class:`TransformerTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` documentation for a description of the fields.
    """

    rank_microbatch_size: int
    max_sequence_length: int
    pp_config: TransformerPipelineParallelConfig

    # Optimizer settings.

    optim: OptimConfig
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

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
    ) -> "TransformerPipelineTrainModule":
        """
        Build the corresponding :class:`TransformerPipelineTrainModule`.

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
        return TransformerPipelineTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class TransformerPipelineTrainModule(TrainModule):
    """
    A pipeline-parallel :class:`TrainModule` for most :class:`~olmo_core.nn.transformer.Transformer` model
    implementation provided by this library.

    .. tip::
        Use the :class:`TransformerPipelineTrainModuleConfig` to easily configure and build
        :class:`TransformerPipelineTrainModule` instances.

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
    :param cp_config: Context parallel configuration for the model.
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
        pp_config: TransformerPipelineParallelConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
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
        self.world_mesh = build_world_mesh(
            dp=dp_config, tp=tp_config, cp=cp_config, pp=pp_config, device_type=self.device.type
        )
        log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")

        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier

        self.float8_handler: Optional[Float8Handler] = None
        float8_enabled = False
        if float8_config is not None:
            float8_enabled = float8_config.enabled
            float8_config.compile = compile_model
            self.float8_handler = float8_config.build()

        self._pp_config = pp_config
        # We'll initialize this lazily when the trainer is attached, since we need to know
        # the global batch size in order to determine the number of pipeline micro-batches.
        self._train_pp_schedule: Optional[PipelineSchedule] = None
        self._pp_stages: Optional[List[PipelineStage]] = None

        self.model_parts: List[Transformer] = []
        pp_mesh = get_pp_mesh(self.world_mesh)
        stages, model_parts = pp_config.split_model(model, pp_mesh=pp_mesh, device=self.device)
        self._pp_stages = stages
        self.model_parts = model_parts
        log.info(f"Applied pipeline parallelism to the model with {get_device_mesh_info(pp_mesh)}")

        # Maybe convert linear layers to FP8 linear.
        if self.float8_handler is not None and self.float8_handler.enabled:
            for model in self.model_parts:
                self.float8_handler.convert_to_float8_training(
                    model, modules_to_ignore={"lm_head.w_out"}
                )
            log.info("Swapped linear layers to Float8 linear layers")

        # Maybe apply context parallelism.
        self._cp_config = cp_config
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None
        if cp_config is not None:
            cp_mesh = get_cp_mesh(self.world_mesh)
            self._cp_load_balancer = cp_config.load_balancer.build(cp_mesh)
            for model in self.model_parts:
                model.apply_cp(cp_mesh, load_balancer=cp_config.load_balancer)
            log.info(
                f"Applied context parallelism to the model with {get_device_mesh_info(cp_mesh)}"
            )

        # Maybe apply tensor parallelism.
        self._tp_enabled = False
        if tp_config is not None:
            tp_mesh = get_tp_mesh(self.world_mesh)
            for model in self.model_parts:
                model.apply_tp(
                    tp_mesh,
                    float8_enabled=float8_enabled,
                )
            tp_config.maybe_enable_async_tp(tp_mesh)
            log.info(
                f"Applied {'Float8 ' if float8_enabled else ''}tensor parallelism to the model "
                f"with {get_device_mesh_info(tp_mesh)}"
            )
            self._tp_enabled = True

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
            if torch.cuda.is_available():
                for model in self.model_parts:
                    model.apply_compile()
                log.info("Applied torch.compile() to the model")
            else:
                log.warning("Skipping model compilation since CUDA is not available")

        # Maybe shard/replicate according to data parallel config.
        self._dp_config = dp_config
        if dp_config is not None:
            dp_mesh = get_dp_model_mesh(self.world_mesh)
            if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
                param_dtype = (
                    dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
                )
                for model in self.model_parts:
                    if model.is_moe:
                        cast(MoETransformer, model).prepare_experts_for_fsdp(
                            self.world_mesh,
                            param_dtype=param_dtype,
                            reduce_dtype=dp_config.reduce_dtype.as_pt(),
                            pp_enabled=True,
                        )
                    model.apply_fsdp(
                        dp_mesh=dp_mesh,
                        param_dtype=param_dtype,
                        reduce_dtype=dp_config.reduce_dtype.as_pt(),
                        wrapping_strategy=dp_config.wrapping_strategy,
                        pp_enabled=True,
                    )
                log.info(f"Applied FSDP to the model with {get_device_mesh_info(dp_mesh)}")
            elif dp_config.name == DataParallelType.ddp:
                for model in self.model_parts:
                    if model.is_moe:
                        cast(MoETransformer, model).prepare_experts_for_ddp(self.world_mesh)
                    model.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model)
                log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")
            else:
                raise NotImplementedError(dp_config.name)

        # Materialize and init parameters.
        for model in self.model_parts:
            log.info("Initializing model weights...")
            model.init_weights(max_seq_len=max_sequence_length, device=self.device)

        # Build optimizer(s).
        log.info("Building optimizer(s)...")
        self.optimizers: List[Optimizer] = [
            optim.build(model, strict=False) for model in self.model_parts
        ]
        if isinstance(self.optimizers[0], SkipStepOptimizer):
            raise NotImplementedError(
                "Pipeline parallelism with a SkipStepOptimizer is currently not supported"
            )

        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=False
        )
        self.load_key_mapping = load_key_mapping

        for model in self.model_parts:
            if model.is_moe:
                # TODO (epwalsh): need to handle the MoE auxiliary losses correctly.
                raise NotImplementedError(
                    "Pipeline parallelism with MoE's is currently not supported"
                )

        self._ce_batch_loss: Optional[torch.Tensor] = None
        self._z_batch_loss: Optional[torch.Tensor] = None

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        # Determine the number of micro-batches.
        rank_batch_size = self.trainer.global_batch_size // get_world_size(
            self.trainer.dp_process_group
        )
        rank_batch_size_instances = rank_batch_size // self.max_sequence_length
        return EvalBatchSpec(
            rank_batch_size=rank_batch_size_instances,
            batch_size_unit=EvalBatchSizeUnit.instances,
            max_sequence_length=self.max_sequence_length,
            fixed_sequence_length=True,
        )

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def train_pp_schedule(self) -> PipelineSchedule:
        self.trainer  # make sure trainer has been attached before trying to access this
        assert self._train_pp_schedule is not None
        return self._train_pp_schedule

    def loss_fn(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        return output

    def on_attach(self):
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

        # Initialize pipeline schedule.
        assert self._train_pp_schedule is None  # make sure we don't initialize this twice
        assert self._pp_stages is not None
        pp_mesh = get_pp_mesh(self.world_mesh)
        assert pp_mesh is not None

        # Determine the number of micro-batches.
        rank_batch_size = self.trainer.global_batch_size // dp_ws
        num_micro_batches = rank_batch_size // self.rank_microbatch_size

        self._train_pp_schedule = PipelineSchedule(
            model_parts=self.model_parts,  # type: ignore[arg-type]
            stages=self._pp_stages,
            pp_mesh=pp_mesh,
            schedule_name=self._pp_config.schedule,
            loss_fn=self.loss_fn,
            n_microbatches=num_micro_batches,
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
        for model, optim in zip(self.model_parts, self.optimizers):
            dist_cp_sd.set_model_state_dict(
                model,
                state_dict["model"],
                options=self.state_dict_load_opts,
            )
            gc_cuda()
            if "optim" in state_dict:
                dist_cp_sd.set_optimizer_state_dict(
                    model,
                    optim,
                    state_dict["optim"],
                    options=self.state_dict_load_opts,
                )
                gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        for model in self.model_parts:
            model.train()

        # Generate labels.
        labels = batch.get("labels", get_labels(batch, label_ignore_index=self.label_ignore_index))

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = (labels != self.label_ignore_index).sum().item()
        if self.cp_enabled:
            assert self._cp_config is not None
            batch_num_tokens_for_loss /= self._cp_config.degree

        # Run pipeline schedule.
        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)
        assert labels is not None
        self.model_forward(
            input_ids,
            labels,
            ignore_index=self.label_ignore_index,
            loss_reduction="sum",
            z_loss_multiplier=self.z_loss_multiplier,
            loss_div_factor=batch_num_tokens_for_loss,
            return_logits=False,
            **model_kwargs,
        )

        if dry_run:
            self._clear_loss_buffers()
            return

        # Record loss metrics.
        # NOTE: losses could be none for pipeline parallelism if rank doesn't have the final stage.
        if self._ce_batch_loss is None:
            self.record_ce_loss(0.0, ReduceType.sum)
        else:
            self.record_ce_loss(
                self._ce_batch_loss / get_world_size(self.dp_process_group), ReduceType.sum
            )
        if self.z_loss_multiplier is not None:
            if self._z_batch_loss is None:
                self.record_metric("Z loss", 0.0, ReduceType.sum, namespace="train")
            else:
                self.record_metric(
                    "Z loss",
                    self._z_batch_loss / get_world_size(self.dp_process_group),
                    ReduceType.sum,
                    namespace="train",
                )
        # TODO: handle model auxiliary losses, like with MoE.

        for optim in self.optimizers:
            if isinstance(optim, SkipStepOptimizer):
                assert self._ce_batch_loss is not None
                optim.latest_loss = self._ce_batch_loss

        # Lastly, clear internal loss buffers.
        self._clear_loss_buffers()

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None) -> Any:
        del batch, labels
        raise RuntimeError(f"{self.__class__.__name__} does not support inference")

    def optim_step(self):
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
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
        self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Run a forward pass on a micro-batch, returning the logits and potentially the loss.
        """

        # Don't want the pipeline schedule to mess with kwargs, so we inject them with a hook.
        def kwargs_saver(_, args_, kwargs_):
            kwargs_.update(kwargs)
            return (args_, kwargs_)

        # We also don't want the pipeline schedule to save the logits from every stage because
        # that would blow up our memory usage, so we capture the output from the final stage
        # and only return the loss.
        def capture_losses(_, args_, output) -> torch.Tensor:
            del args_

            assert isinstance(output, LMOutputWithLoss)
            _, ce_loss, z_loss = output

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

        handles = []
        for model in self.model_parts:
            handle = model.register_forward_pre_hook(kwargs_saver, with_kwargs=True)
            handles.append(handle)
            if self.train_pp_schedule.is_first_stage and model.lm_head is not None:
                handle = model.register_forward_hook(capture_losses)
                handles.append(handle)

        with self._model_forward_context():
            output, _ = self.train_pp_schedule.step(
                input_ids,
                labels,
                target=labels,
            )

            if self.train_pp_schedule.is_last_stage:
                assert output is not None

        for handle in handles:
            handle.remove()

        return output

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.model_parts[0].num_flops_per_token(seq_len)

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _clear_loss_buffers(self):
        self._ce_batch_loss = None
        self._z_batch_loss = None
        for model in self.model_parts:
            model.reset_auxiliary_losses()
            model.reset_auxiliary_metrics()

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

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

        parameters = [p for m in self.model_parts for p in m.parameters()]
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

        pp_mesh = self.train_pp_schedule.pp_mesh
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels, batch
