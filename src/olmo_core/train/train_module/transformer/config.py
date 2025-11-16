import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage

from olmo_core.config import Config, DType
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_full_tensor, get_local_tensor
from olmo_core.distributed.parallel import (
    ContextParallelConfig,
    DataParallelConfig,
    ExpertParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import RingAttentionLoadBalancerType
from olmo_core.nn.transformer import (
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.utils import move_to_device

from torch.distributed.tensor import DTensor, distribute_tensor

if TYPE_CHECKING:
    from .pipeline_train_module import TransformerPipelineTrainModule
    from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)


@beta_feature
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
        for stage_idx in self.stage_ids_this_rank(pp_rank, num_stages):
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

    prefetch_factor: int = 0


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


@dataclass
class TransformerContextParallelConfig(ContextParallelConfig):
    """
    Transformer-specific context parallel config.
    """

    load_balancer: RingAttentionLoadBalancerType = RingAttentionLoadBalancerType.zig_zag
    """
    The type of load balancer to use for ring attention.
    """

    head_stride: int = 1
    """
    The stride of the head dimension to process for each iteration of ring attention. A value of 1
    means each iteration will process one k and one v head. A value of 2 will process two k and two
    v heads, etc. A larger stride will reduce the number of communication ops.
    """

    @classmethod
    def zig_zag(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            load_balancer=RingAttentionLoadBalancerType.zig_zag,
            head_stride=head_stride,
        )

    @classmethod
    def llama3(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            load_balancer=RingAttentionLoadBalancerType.llama3,
            head_stride=head_stride,
        )


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

    activation_memory_budget: Optional[float] = None
    """
    Required when :data:`mode` is "budget". Memory budget for activation checkpointing in range [0, 1].
    0 = recompute all activations, 1 = recompute none (default). Requires compilation to be enabled.

    See https://pytorch.org/blog/activation-checkpointing-techniques/ for more details.
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
    A configuration class for building :class:`TransformerTrainModule` or
    :class:`TransformerPipelineTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` and :class:`TransformerPipelineTrainModule`
        documentation for a description of the fields.
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
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
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
    ) -> Union["TransformerTrainModule", "TransformerPipelineTrainModule"]:
        """
        Build the corresponding :class:`TransformerTrainModule` or :class:`TransformerPipelineTrainModule.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        """
        from .pipeline_train_module import TransformerPipelineTrainModule
        from .train_module import TransformerTrainModule

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        if self.pp_config is not None:
            return TransformerPipelineTrainModule(
                model=model,
                device=device,
                **kwargs,
            )
        else:
            return TransformerTrainModule(
                model=model,
                device=device,
                **kwargs,
            )


@beta_feature
@dataclass
class TransformerPipelineTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Kept for backwards compatibility, but please use :class:`TransformerTrainModuleConfig` instead.
    """

    def __post_init__(self):
        if self.pp_config is None:
            raise OLMoConfigurationError("'pp_config' is required")
        
@dataclass
class FreezeTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    def __init__(self, *args, **kwargs):
        self.freeze_experts = kwargs.pop("freeze_experts", "first_half")
        super().__init__(*args, **kwargs)

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> "FreezeTransformerTrainModule":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # bp()
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)
        return FreezeTransformerTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


def distribute_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not isinstance(source, DTensor):
        return get_full_tensor(target)
    if isinstance(target, DTensor):
        if target.device_mesh == source.device_mesh and target.placements == source.placements:
            return target
        else:
            return target.redistribute(device_mesh=source.device_mesh, placements=source.placements)
    return distribute_tensor(target, device_mesh=source.device_mesh, placements=source.placements)


# Copied from FlexOlmo for SFT (testing as of 11/15)
@dataclass
class FreezeTransformerTrainModule(TransformerTrainModule):
    """
    Custom transformer train module that zeros out gradients for the first half of expert parameters.
    Inherits from the original TransformerTrainModule.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomTransformerTrainModule by calling the parent class's __init__ method.
        #"""
        # swj change
        # extract freeze_experts from kwargs
        self.freeze_experts = kwargs.pop("freeze_experts", "first_half")
        super().__init__(*args, **kwargs)

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self.model.train()

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate how many tokens are going to be used in the loss.
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        if self.cp_enabled:
            assert self._cp_config is not None
            batch_num_tokens_for_loss = batch_num_tokens_for_loss / self._cp_config.degree

        # Batch losses to record.
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
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
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Run forward pass, get losses.
                _, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )

                # Get loss to optimize for.
                loss = ce_loss
                if z_loss is not None:
                    loss += z_loss

                # Update total batch CE and Z loss.
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                # print("self.trainer.global_step: ", self.trainer.global_step)
                # Optionally get model auxiliary losses and update the total batch auxiliary losses. , step=self.trainer.global_step
                # step=self.trainer.global_step
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

        # for name, param in self.model.named_parameters():
        # if "expert2_bias" in name:
        # from ipdb import set_trace as bp
        # bp()
        # print(f"{name}: {get_full_tensor(param)}")
        # if param.numel() > /0:  # Only record non-empty tensors
        #     self.record_metric(
        #         name,
        #         get_full_tensor(param),
        #         ReduceType.mean,
        #         namespace="train",
        #     )
        # log.info("swj")
        # for name, param in self.model.named_parameters():
        # log.info(f"{name}: requires_grad={param.requires_grad}")
        # bp()
        # swj change
        # fsdp summon
        for name, param in self.model.named_parameters():
            if "experts" in name or "router" in name:
                # bp()
                if self.freeze_experts == "first_half":
                    # print("name: ", name, "shape: ", param.shape)
                    full_grad = get_full_tensor(param.grad)
                    # check whether the param is frozen
                    # print("param.grad: ", param.grad)
                    if param.grad is None:
                        # print(f"{name} grad is None")
                        continue
                    if "experts" in name:
                        # get_full_tensor(param.grad)[
                        #     : get_full_tensor(param.grad).shape[0] // 2, :
                        # ] = 0
                        mask = torch.zeros_like(full_grad, dtype=torch.bool)
                        mask[: full_grad.shape[0] // 2, :] = True
                        local_mask = get_local_tensor(distribute_like(param, mask))
                        # print("target.device_mesh: ", param.device_mesh)
                        get_local_tensor(param.grad).masked_fill_(local_mask, 0.0)
                        # print("mask: ", mask)
                    elif "router" in name:
                        # get_full_tensor(param.grad)[
                        #     : get_full_tensor(param.grad).shape[0] // 2
                        # ] = 0
                        mask = torch.zeros_like(full_grad, dtype=torch.bool)
                        # swj change, free the entire router
                        mask[: full_grad.shape[0] // 2] = 1
                        # mask = torch.ones_like(full_grad, dtype=torch.bool)
                        local_mask = get_local_tensor(distribute_like(param, mask))
                        get_local_tensor(param.grad).masked_fill_(local_mask, 0.0)
                        # print("mask: ", mask)
                        # get_local_tensor(param.grad) = get_local_tensor(param.grad).mul(local_mask)
                # elif self.freeze_experts == "last_half":
                #     if "experts" in name:
                #         get_full_tensor(param.grad)[
                #             get_full_tensor(param.grad).shape[0] // 2 :, :
                #         ] = 0
                #     elif "router" in name:
                #         get_full_tensor(param.grad)[
                #             get_full_tensor(param.grad).shape[0] // 2 :
                #         ] = 0
                else:
                    raise ValueError(f"Invalid freeze_experts value: {self.freeze_experts}")

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
            batch_num_tokens_for_loss,
            reset=True,
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )
        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss