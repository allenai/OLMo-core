import contextlib
import logging
from collections import OrderedDict
from dataclasses import replace
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import nvtx
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    MeshDimName,
    get_device_mesh_info,
)
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
    DPMeshDimName,
)
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig
from olmo_core.distributed.parallel.pipeline_parallel import (
    PipelineParallelConfig,
    PipelineSchedule,
    PipelineScheduleType,
    PipelineSplitStyle,
)
from olmo_core.distributed.parallel.tensor_parallel import TensorParallelConfig
from olmo_core.distributed.utils import (
    get_global_rank,
    get_local_tensor,
    get_rank,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer
from olmo_core.nn.transformer import MoETransformer, Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import MoEFusedV2OptimizerConfig, OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, log_once, move_to_device

from ...common import MetricMergeStrategy, ReduceType
from ..train_module import EvalBatchSpec, TrainModule
from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)

M = TypeVar("M", bound=List[MoEFusedV2Transformer])


class MoEV2TransformerTrainModule(TrainModule):
    def __init__(
        self,
        model: Transformer,
        optim: MoEFusedV2OptimizerConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
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
        reduce_scatter_grads: bool = True,
    ):
        super().__init__()
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer

        assert isinstance(
            model, MoEFusedV2Transformer
        ), "MoEV2TransformerTrainModule only supports MoEFusedV2Transformer model"

        ######################### Validate arguments. [BEGIN] #########################
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Dict[str, Optional[DeviceMesh]] = {}
        self.pp_group = None
        self.dp_group = None
        self.tp_group = None
        self.ep_dp_group = None
        self.ep_mp_group = None

        # PP related state.
        self._train_pp_schedule: Optional[PipelineSchedule] = None
        self._pp_stages: Optional[List[PipelineStage]] = None
        self.pp_group_rank = 0  # default 0
        self.pp_group_size = 1  # default 1
        self.pp_prev_rank = -1  # no previous stage
        self.pp_next_rank = -1  # no next stage
        self.pp_final_stage_rank = 0  # default 0

        # If True, the DDP will not all-reduce grad for the last microbatch
        # instead it will call optim.step() which will reduce_scatter
        # directly into the owner main grad
        self.reduce_scatter_grads = reduce_scatter_grads

        if tp_config is not None:
            assert (
                tp_config.degree > 1
            ), "Tensor parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Tensor parallelism is not implemented")
        if pp_config is not None:
            assert (
                pp_config.degree > 1
            ), "Pipeline parallelism requires a degree > 1, otherwise use None"
            # raise NotImplementedError("Pipeline parallelism is not implemented")
        if cp_config is not None:
            assert (
                cp_config.degree > 1
            ), "Context parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Context parallelism is not implemented")
        # if ac_config is not None:
        #     raise OLMoConfigurationError("In MoEV2TransformerTrainModule, activation checkpointing is controlled by the model, not the train module.")
        if float8_config is not None:
            raise NotImplementedError("Float8 quantization is not implemented")

        assert (
            dp_config is not None
        ), "Data parallel config is required for MoEV2TransformerTrainModule"
        assert dp_config.name == "ddp", "Data parallel config must be 'ddp'"

        ########################## Validate arguments. [END] ##########################

        if is_distributed():
            self._build_world_mesh(
                dp=dp_config,
                tp=tp_config,
                cp=cp_config,
                ep=ep_config,
                pp=pp_config,
                device_type=self.device.type,
            )
            self.dp_world_size = get_world_size(self.dp_process_group)
            log.info(f"Data parallel world size = {self.dp_world_size:,d}")
            assert self.world_mesh["dense"] is not None
        else:
            raise OLMoConfigurationError(
                "Training parallelism is required for MoEV2TransformerTrainModule"
            )

        # Parallelize model.
        self.model_parts = self.parallelize_model(
            model,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
            pp_config=pp_config,
        )

        self.init_model_weights(
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
        )

        self._cast_to_fwd_bwd_precision(self.model_parts)

        import torch._dynamo.config as dynamo_cfg

        dynamo_cfg.recompile_limit = 64  # or any higher number you want

        if compile_model:
            self.compile_model()

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self._pp_config = pp_config

        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length

        self.max_grad_norm = max_grad_norm  # TODO: remove, use optim.max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping

        # Build optimizer(s).
        log.info("Building optimizer...")

        from olmo_core.optim.moe_optimizer import (
            MoEFusedV2Optimizer,
            MoEFusedV2OptimizerConfig,
        )

        assert isinstance(optim, MoEFusedV2OptimizerConfig)
        optim = cast(MoEFusedV2OptimizerConfig, optim)
        self.optim: MoEFusedV2Optimizer = optim.build(
            self.model_parts,
            self,
            strict=False,  # group_overrides might only be matched in one group, strict=False allows it to not match in one group (could match in some other group),
        )

        self.optim.set_reduce_scatter_grads(self.reduce_scatter_grads)

    def _cast_to_fwd_bwd_precision(
        self, model: Union[MoEFusedV2Transformer, List[MoEFusedV2Transformer]]
    ) -> None:
        """
        Cast the model to the forward and backward precision.
        """
        if isinstance(model, list):
            for m in model:
                self._cast_to_fwd_bwd_precision(m)
        else:
            # if the model has implemented `cast_to_fwd_bwd_precision` method, call it.
            if hasattr(model, "_cast_to_fwd_bwd_precision"):
                assert callable(
                    model._cast_to_fwd_bwd_precision
                ), "model._cast_to_fwd_bwd_precision must be callable"
                model._cast_to_fwd_bwd_precision()
            else:
                # if the model doesn't have `cast_to_fwd_bwd_precision`, we need to cast the parameters directly.
                for p in model.parameters():
                    p.data = p.data.to(torch.bfloat16)

    def _build_world_mesh(
        self,
        *,
        dp: Optional[DataParallelConfig] = None,
        tp: Optional[TensorParallelConfig] = None,
        cp: Optional[ContextParallelConfig] = None,
        pp: Optional[PipelineParallelConfig] = None,
        ep: Optional[ExpertParallelConfig] = None,
        device_type: Optional[str] = None,
    ) -> None:
        """
        Build a :class:`~torch.distributed.device_mesh.DeviceMesh` suitable for the given parallel strategies.

        .. seealso::
            Pass the mesh created by this function to any of the ``get_*_mesh()`` functions in
            this module to get the right sub-mesh for a any given parallel strategy.

            - :func:`get_dp_model_mesh()` gives you the 1 or 2D sub-mesh suitable for data parallel *model*
            wrappers like FSDP(2) or DDP.
            - :func:`get_dp_mesh()` gives you the 1D sub-mesh suitable for configuring *data loaders*.
            - :func:`get_tp_mesh()` gives you the 1D sub-mesh for tensor parallelism.
            - :func:`get_cp_mesh()` gives you the 1D sub-mesh for context parallelism.
            - :func:`get_pp_mesh()` gives you the 1D sub-mesh for pipeline parallelism.
            - :func:`get_ep_mesh()` gives you the 1D sub-mesh for expert parallelism.

        .. important::
            A data parallel config is required if any other parallel config is set.

        .. important::
            Not all parallel strategies are compatible with each other.

        :param dp: Data parallel config.
        :param tp: Tensor parallel config.
        :param cp: Context parallel config.
        :param pp: Pipeline parallel config.
        :param ep: Expert parallel config.
        :param device_type: The device type.

        :returns: The world mesh with a shape compatible with the given parallel configs.
        """

        if len(self.world_mesh) > 0:
            raise RuntimeError(
                "world mesh already exists! You can only call 'build_world_mesh' once!"
            )

        device_type = device_type or get_default_device().type
        dp_world_size = get_world_size()

        # no config, Pure DP
        # if pp is None and tp is None and cp is None and dp is None and ep is None:
        #     self.world_mesh = init_device_mesh(device_type, (dp_world_size,), mesh_dim_names=(MeshDimName.dp,))
        #     return

        if dp is None:
            raise OLMoConfigurationError(
                "Data parallel config is required in addition to expert/tensor/context/pipeline parallel configs"
            )

        # Validate parallelism degrees while adjust the DP degree.
        if pp is not None:
            if pp.degree <= 1 or dp_world_size % pp.degree != 0:
                raise OLMoConfigurationError(
                    f"{pp.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )
            dp_world_size //= pp.degree
        if cp is not None:
            if cp.degree <= 1 or dp_world_size % cp.degree != 0:
                raise OLMoConfigurationError(
                    f"{cp.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )
            dp_world_size //= cp.degree
        if tp is not None:
            if tp.degree <= 1 or dp_world_size % tp.degree != 0:
                raise OLMoConfigurationError(
                    f"{tp.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )
            dp_world_size //= tp.degree
        if ep is not None:
            if ep.degree <= 1 or dp_world_size % ep.degree != 0:
                raise OLMoConfigurationError(
                    f"{ep.__class__.__name__}.degree must be at least 1 and divide into the world size"
                )

        # Build up dense mesh dimensions. (PP , DP)
        names: List[str] = []
        dims: List[int] = []

        # Pipeline parallel first.
        if pp is not None:
            names.append(MeshDimName.pp)
            dims.append(pp.degree)

        # Then data parallel.
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)

        self.dense_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
        log.info(f"Built dense_mesh {get_device_mesh_info(self.dense_mesh)}")

        if ep is None:  # EP not used
            self.moe_mesh = None
            log.info(f"Built moe_mesh None")

        else:
            # Build up moe mesh dimensions. (PP , EP_DP, EP_MP)
            names: List[str] = []
            dims: List[int] = []

            # Pipeline parallel first.
            if pp is not None:
                names.append(MeshDimName.pp)
                dims.append(pp.degree)

            # Then EP data parallel.
            ep_dp_world_size = dp_world_size // ep.degree
            names.append(MeshDimName.ep_dp)
            dims.append(ep_dp_world_size)

            # Then EP model parallel.
            ep_mp_world_size = ep.degree
            names.append(MeshDimName.ep_mp)
            dims.append(ep_mp_world_size)

            self.moe_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))

            log.info(f"Built moe_mesh{get_device_mesh_info(self.moe_mesh)}")

        # build process group

        if pp is not None:
            # self.dense_mesh['pp'] and self.moe_mesh['pp'] should be the same
            self.pp_group = self.dense_mesh["pp"].get_group()  #

        self.dp_group = self.dense_mesh["dp"].get_group()
        if self.moe_mesh is not None:
            self.ep_dp_group = self.moe_mesh["ep_dp"].get_group()
            self.ep_mp_group = self.moe_mesh["ep_mp"].get_group()

        self.world_mesh = {"dense": self.dense_mesh, "moe": self.moe_mesh}

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return self.dp_group

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            # max(self.rank_microbatch_size // 2, 1 * self.max_sequence_length),
            1 * self.max_sequence_length,
            max_sequence_length=self.max_sequence_length,
            #  fixed_sequence_length=self.tp_enabled,
        )

    @property
    def dp_config(self) -> TransformerDataParallelConfig:
        return self._dp_config

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def ep_enabled(self) -> bool:
        return self._ep_config is not None

    @property
    def pp_enabled(self) -> bool:
        return self._pp_config is not None

    @property
    def train_pp_schedule(self) -> PipelineSchedule:
        self.trainer  # make sure trainer has been attached before trying to access this
        assert self._train_pp_schedule is not None
        return self._train_pp_schedule

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(self.world_size)

    def on_attach(self):
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            # raise OLMoConfigurationError(
            #     f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
            #     f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            # )
            pass  # BUG: when batch size warmup + load checkpoint

        if self.pp_enabled:
            # Initialize pipeline schedule.
            assert self._train_pp_schedule is None  # make sure we don't initialize this twice
            assert self._pp_stages is not None
            assert self._pp_config is not None
            assert self.world_mesh["dense"] is not None
            pp_mesh = self.world_mesh["dense"]["pp"]
            assert pp_mesh is not None

            # Determine the number of micro-batches.
            rank_batch_size = self.trainer.global_batch_size // dp_ws
            num_microbatches = rank_batch_size // self.rank_microbatch_size

            self._train_pp_schedule = PipelineSchedule(
                model_parts=self.model_parts,  # type: ignore[arg-type]
                stages=self._pp_stages,
                pp_mesh=pp_mesh,
                schedule_name=self._pp_config.schedule,
                loss_fn=self.loss_fn,
                num_microbatches=num_microbatches,
            )

    def loss_fn(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # NOTE: the output is the loss.
        del labels
        return output

    def state_dict(self, *, optim: bool = True) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def state_dict_to_load(self, metadata: Metadata, *, optim: bool = True) -> Dict[str, Any]:
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

        has_optim_state: bool = False
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("optim."):
                has_optim_state = True
                break

        if optim and not has_optim_state:
            log.warning("No optimizer state found in checkpoint")
            optim = False

        state_dict = self._get_state_dict(load_opts, optim=optim)
        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, metadata=metadata)

        if not load_opts.strict:
            # Remove any keys in the 'state_dict' that are not present in the checkpoint.
            pruned_keys = prune_state_dict(state_dict, set(metadata.state_dict_metadata.keys()))
            if pruned_keys:
                log.warning(f"Checkpoint is missing the following keys: {pruned_keys}")

        return state_dict

    def state_dict_to_save(self, *, optim: bool = True) -> Dict[str, Any]:
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        load_optim = "optim" in state_dict

        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, reverse=True, quiet=True)

        # NOTE: `dist_cp_sd.set_(model|optimizer)_state_dict()` doesn't respect `strict=False`
        # option with missing keys, so we have to handle that on our own.
        if not self.state_dict_load_opts.strict:
            flatten_optimizer_state_dict = (
                False if not load_optim else ("state" not in state_dict["optim"])
            )
            load_opts = replace(
                self.state_dict_load_opts, flatten_optimizer_state_dict=flatten_optimizer_state_dict
            )
            full_state_dict = self._get_state_dict(load_opts, optim=load_optim)
            merge_state_dicts(state_dict, full_state_dict)

        plain_model_state_dict = OrderedDict()
        for key, value in state_dict["model"].items():
            if isinstance(value, DTensor):
                plain_model_state_dict[key] = value.to_local()
            else:
                plain_model_state_dict[key] = value

        dist_cp_sd.set_model_state_dict(
            self.model,
            plain_model_state_dict,
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if load_optim:
            self.optim.load_state_dict(state_dict["optim"])

            # debug_model1 = torch.load(f'tmp.model.{dist.get_rank()}.pt')
            # debug_model2 = self.model.state_dict()

            # # compare
            # for key in debug_model1.keys():
            #     if not torch.equal(debug_model1[key], debug_model2[key]):
            #         print(f"Difference found in key: {key}")

            # debug_optim1 = torch.load(f'tmp.optim.{dist.get_rank()}.pt')

            # debug_optim2 = torch.optim.Optimizer.state_dict(self.optim)
            # debug_optim2['_flat_main_dp'] = self.optim._flat_main_dp
            # debug_optim2['_flat_main_ep_dp '] = self.optim._flat_main_ep_dp
            # debug_optim2['_flat_exp_avg_dp '] = self.optim._flat_exp_avg_dp
            # debug_optim2['_flat_exp_avg_ep_dp'] = self.optim._flat_exp_avg_ep_dp
            # debug_optim2['_flat_exp_avg_sq_dp '] = self.optim._flat_exp_avg_sq_dp
            # debug_optim2['_flat_exp_avg_sq_ep_dp  '] = self.optim._flat_exp_avg_sq_ep_dp
            # # compare
            # for key in debug_optim1.keys():
            #     if key.startswith('_') and not torch.equal(debug_optim1[key], debug_optim2[key]):
            #         print(f"Difference found in key: {key}")

            gc_cuda()

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        for m in self.model_parts:
            m.train()

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        #############################
        if not self.pp_enabled:
            # Calculate how many tokens are going to be used in the loss.
            batch_num_tokens_for_loss = move_to_device(
                (batch["labels"] != self.label_ignore_index).sum(), self.device
            )

            # Batch losses to record.
            ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
            z_batch_loss: Optional[torch.Tensor] = None
            if self.z_loss_multiplier is not None:
                z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

            # Split into micro-batches.
            if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
                raise RuntimeError(
                    f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
                )
            micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
            num_micro_batches = len(micro_batches)

            dbg_mem_before_fwd0 = torch.cuda.memory_allocated() / 1024**3

            dbg_mem_activation_usage_all = []
            dbg_mem_activation_freed_all = []
            # Train one micro-batch at a time.
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                    with nvtx.annotate(f"fwd_mb{micro_batch_idx}", color="blue"):
                        input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
                        dbg_mem_before_fwd = torch.cuda.memory_allocated() / 1024**3
                        # Run forward pass, get losses.
                        _, loss, ce_loss, z_loss = self.model_forward_no_pipeline(
                            input_ids,
                            labels=labels,
                            ignore_index=self.label_ignore_index,
                            loss_reduction="sum",
                            z_loss_multiplier=self.z_loss_multiplier,
                            loss_div_factor=batch_num_tokens_for_loss,
                            return_logits=False,
                            **model_kwargs,
                        )
                        dbg_mem_after_fwd = torch.cuda.memory_allocated() / 1024**3
                        dbg_mem_activation_usage = dbg_mem_after_fwd - dbg_mem_before_fwd
                        dbg_mem_activation_usage_all.append(dbg_mem_activation_usage)
                        # Update total batch CE and Z loss.
                        ce_batch_loss += get_local_tensor(ce_loss.detach())
                        del ce_loss
                        if z_batch_loss is not None:
                            assert z_loss is not None
                            z_batch_loss += get_local_tensor(z_loss.detach())
                            del z_loss

                    with nvtx.annotate(f"bwd_mb{micro_batch_idx}", color="red"):
                        # Run backward pass.
                        dbg_mem_before_bwd = torch.cuda.memory_allocated() / 1024**3
                        loss.backward()
                        # self.model.reset_offload_handler()

                        dbg_mem_after_bwd = torch.cuda.memory_allocated() / 1024**3
                        dbg_mem_activation_freed = dbg_mem_before_bwd - dbg_mem_after_bwd
                        dbg_mem_activation_freed_all.append(dbg_mem_activation_freed)
                        pass

            del batch  # In case this helps with memory utilization.
            if dry_run:
                print("activation: ", dbg_mem_activation_usage_all)
                print("freed:      ", dbg_mem_activation_freed_all)
                for tag, mem in self.model_parts[0]._debug_alloc_mem_layer_logs:
                    print(f"Alloc - {tag}: {mem:.2f} GB")
                for tag, mem in self.model_parts[0]._debug_max_alloc_mem_layer_logs:
                    print(f"Max - {tag}: {mem:.2f} GB")
        else:
            # pipeline parallel forward / backward
            # Calculate how many tokens are going to be used in the loss.
            batch_num_tokens_for_loss = (batch["labels"] != self.label_ignore_index).sum().item()

            # Run pipeline schedule.
            input_ids, labels, model_kwargs = self._prepare_batch(batch, batch["labels"])
            assert labels is not None
            ce_batch_loss, z_batch_loss = self.run_pipeline(
                input_ids,
                labels,
                batch_num_tokens_for_loss,
                ignore_index=self.label_ignore_index,
                loss_reduction="sum",
                z_loss_multiplier=self.z_loss_multiplier,
                return_logits=False,
                **model_kwargs,
            )
        #############################
        debug_norm = [
            torch.nn.utils.get_total_norm([p.grad for p in model_part.parameters()])
            for model_part in self.model_parts
        ]

        for model in self.model_parts:
            model.post_batch(dry_run=dry_run)

        if dry_run:
            for model in self.model_parts:
                model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        from olmo_core.optim.moe_optimizer import MoEFusedV2Optimizer

        if self.pp_enabled:
            if isinstance(self.optim, MoEFusedV2Optimizer):
                ce_batch_loss = self.reduce_send_recv(ce_batch_loss)
                self.record_ce_loss(ce_batch_loss)
                self.optim.latest_loss = ce_batch_loss
            elif ce_batch_loss is not None:
                self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        else:
            assert ce_batch_loss is not None, "CE loss should not be None"
            if isinstance(self.optim, MoEFusedV2Optimizer):
                # Need to reduce the loss right away for the SkipStepOptimizer.
                if is_distributed():
                    ce_batch_loss.div_(self._reduce_divide_factor)
                    dist.all_reduce(ce_batch_loss)
                    ce_batch_loss.div_(self.world_size)
                    ce_batch_loss.mul_(self._reduce_divide_factor)
                self.record_ce_loss(ce_batch_loss)
                self.optim.latest_loss = ce_batch_loss
            else:
                self.record_ce_loss(ce_batch_loss, ReduceType.mean)

        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for m in self.model_parts:
            for metric_name, (metric_val, reduction) in m.compute_auxiliary_metrics(
                reset=True
            ).items():
                merge_strategy = MetricMergeStrategy.warn
                if reduction in (ReduceType.sum, ReduceType.mean):
                    merge_strategy = MetricMergeStrategy.sum
                elif reduction == ReduceType.max:
                    merge_strategy = MetricMergeStrategy.max
                self.record_metric(
                    metric_name,
                    metric_val,
                    reduction,
                    namespace="train",
                    merge_strategy=merge_strategy,
                )

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        if self.pp_enabled:
            raise NotImplementedError("Pipeline parallelism is not implemented")

        # TODO: (epwalsh) Currently all of our evaluators require the full logits locally,
        # but when we're using CP/TP we usually can't materialize the full logits locally (due to OOMs).
        # However we could at least support in-loop PPL evals with a little work in the evaluator
        # code to handle the sharded logits.
        if self.cp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support context parallelism yet, "
                "please disable in-loop evals"
            )
        if self.tp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support tensor parallelism yet, "
                "please disable in-loop evals"
            )

        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        for m in self.model_parts:
            m.eval()

        with self._eval_batch_context():
            return self.model_forward_no_pipeline(
                input_ids,
                labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="none",
                **model_kwargs,
            )

    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_num_tokens_for_loss: Union[int, float],
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run the pipeline, returning the losses captured.
        """

        ce_batch_loss: Optional[torch.Tensor] = None
        z_batch_loss: Optional[torch.Tensor] = None

        def capture_losses(
            model: Transformer, args: Tuple[torch.Tensor, ...], output: Any
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            del args
            nonlocal ce_batch_loss
            nonlocal z_batch_loss

            if model.lm_head is not None:
                assert isinstance(output, LMOutputWithLoss)
                _, loss, ce_loss, z_loss = output
                if ce_batch_loss is None:
                    ce_batch_loss = get_local_tensor(ce_loss.detach())
                else:
                    ce_batch_loss += get_local_tensor(ce_loss.detach())

                if self.z_loss_multiplier is not None:
                    assert z_loss is not None
                    if z_batch_loss is None:
                        z_batch_loss = get_local_tensor(z_loss.detach())
                    else:
                        z_batch_loss += get_local_tensor(z_loss.detach())

                return loss.unsqueeze(0)
            else:
                assert isinstance(output, torch.Tensor)
                return output

        handles = []
        for model in self.model_parts:
            handles.append(model.register_forward_hook(capture_losses))

        with self._model_forward_context():
            self.train_pp_schedule.step(
                input_ids,
                target=labels,
                loss_div_factor=batch_num_tokens_for_loss,
                labels=labels,
                **kwargs,
            )

        for handle in handles:
            handle.remove()

        return ce_batch_loss, z_batch_loss

    def optim_step(self):
        from olmo_core.optim.moe_optimizer import MoEFusedV2Optimizer

        if self.reduce_scatter_grads:
            pass
        else:
            assert (
                self.pp_enabled == False
            ), "Pipeline parallelism with all-reduce grad is not supported"
            model = self.model_parts[0]
            # Maybe clip gradients.
            if self.max_grad_norm is not None:
                grad_norm = self._clip_grad_norm(self.max_grad_norm)
                # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
                self.trainer.record_metric(
                    "total grad norm", grad_norm, reduce_type=None, namespace="optim"
                )
                if isinstance(self.optim, MoEFusedV2Optimizer):
                    self.optim.latest_grad_norm = grad_norm

            # calculate per layer grad norm
            per_layer_norms = []
            for layer_idx, layer in enumerate(model.blocks.values()):
                layer_grads = [p.grad for p in layer.parameters() if p.grad is not None]
                if layer_grads:
                    per_layer_norm = nn.utils.get_total_norm(
                        layer_grads, norm_type=2.0, error_if_nonfinite=False, foreach=None
                    )
                    if isinstance(per_layer_norm, DTensor):
                        # If per_layer_norm is a DTensor, we need to reduce it to get the correct value.
                        per_layer_norm = per_layer_norm.full_tensor()
                    per_layer_norms.append(per_layer_norm)
                else:
                    per_layer_norms.append(torch.tensor(0.0, device=self.device))

                self.trainer.record_metric(
                    f"clipped grad norm (layer {layer_idx})",
                    per_layer_norms[layer_idx],
                    reduce_type=None,
                    namespace="optim",
                )

                del layer_grads

            del per_layer_norms

            # embedding layer grad norm
            embedding_grads = [p.grad for p in model.embeddings.parameters() if p.grad is not None]
            if embedding_grads:
                embedding_grad_norm = nn.utils.get_total_norm(
                    embedding_grads, norm_type=2.0, error_if_nonfinite=False, foreach=None
                )
                if isinstance(embedding_grad_norm, DTensor):
                    # If embedding_grad_norm is a DTensor, we need to reduce it to get the correct value.
                    embedding_grad_norm = embedding_grad_norm.full_tensor()
            else:
                embedding_grad_norm = torch.tensor(0.0, device=self.device)
            self.trainer.record_metric(
                "clipped grad norm (embedding)",
                embedding_grad_norm,
                reduce_type=None,
                namespace="optim",
            )

            del embedding_grads

            # lm head grad norm
            lm_head_grads = [p.grad for p in model.lm_head.parameters() if p.grad is not None]
            if lm_head_grads:
                lm_head_grad_norm = nn.utils.get_total_norm(
                    lm_head_grads, norm_type=2.0, error_if_nonfinite=False, foreach=None
                )
                if isinstance(lm_head_grad_norm, DTensor):
                    # If lm_head_grad_norm is a DTensor, we need to reduce it to get the correct value.
                    lm_head_grad_norm = lm_head_grad_norm.full_tensor()
            else:
                lm_head_grad_norm = torch.tensor(0.0, device=self.device)
            self.trainer.record_metric(
                "clipped grad norm (lm head)",
                lm_head_grad_norm,
                reduce_type=None,
                namespace="optim",
            )
            del lm_head_grads

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        # Step optimizer.
        self.optim.step()
        total_grad_norm = self.optim.latest_grad_norm
        if self.reduce_scatter_grads:
            if total_grad_norm is not None:
                self.trainer.record_metric(
                    "total grad norm", total_grad_norm, reduce_type=None, namespace="optim"
                )
        if isinstance(self.optim, MoEFusedV2Optimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        for model in self.model_parts:
            model.post_optim_step()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)  # clear main grad
        for m in self.model_parts:
            m.zero_grad(set_to_none=True)  # clear model grad

    def model_forward_no_pipeline(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            return self.model_parts[0](input_ids, labels=labels, **kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        # NOTE: it uses the config to calculate the FLOPs for the whole model, so it should be fine to just use the first model part.
        return self.model_parts[0].num_flops_per_token(seq_len)

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model_parts[0], FSDPModule):
                raise OLMoConfigurationError("FSDP not supported. Use replicate()")
            elif isinstance(self.model_parts[0], DDP):
                raise OLMoConfigurationError("torch DDP not supported. Use replicate()")
            elif self.dp_config.name == DataParallelType.ddp:  # temp fix
                if (
                    self.reduce_scatter_grads  # if use RS, always no sync
                    or (  # if not, fall back to all-reduce
                        # if specified, only AR at the last microbatch
                        not is_last_mb
                        and self.dp_config.only_allreduce_last_microbatch
                    )
                ):
                    stack.enter_context(
                        self.ddp_no_sync(self.model_parts)
                    )  # only DDP has no_sync(), can only call set_requires_gradient_sync()
            yield

    @contextlib.contextmanager
    def ddp_no_sync(self, model_parts: List[MoEFusedV2Transformer]):
        for module in model_parts:
            assert callable(getattr(module, "set_requires_gradient_sync", None)), (
                f"{type(module).__name__} must implement set_requires_gradient_sync(flag: bool). "
                "This is automatically managed if the model is returned by torch.distributed._composable.replicate"
            )
            # non-EP modules
            module.set_requires_gradient_sync(False)  # type: ignore

            # EP managed modules
            ep_modules = [m for m in module.modules() if getattr(m, "_ep_sharded", False)]
            for m in ep_modules:
                m.set_requires_gradient_sync(False)  # type: ignore

        try:
            yield
        finally:
            for module in model_parts:
                module.set_requires_gradient_sync(True)  # type: ignore
                ep_modules = [m for m in module.modules() if getattr(m, "_ep_sharded", False)]
                for m in ep_modules:
                    m.set_requires_gradient_sync(True)  # type: ignore

    @contextlib.contextmanager
    def _eval_batch_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            # if self.autocast_precision is not None:
            #     stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            # NOTE: autocast_precision is deleted
            yield

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions, optim: bool = True
    ) -> Dict[str, Any]:
        plain_model_sd = self.model.state_dict()  # bf16 version

        # debug
        # torch.save(plain_model_sd, f'tmp.model.{dist.get_rank()}.pt')
        # og_optim_sd = torch.optim.Optimizer.state_dict(self.optim)
        # og_optim_sd['_flat_main_dp'] = self.optim._flat_main_dp
        # og_optim_sd['_flat_main_ep_dp '] = self.optim._flat_main_ep_dp
        # og_optim_sd['_flat_exp_avg_dp '] = self.optim._flat_exp_avg_dp
        # og_optim_sd['_flat_exp_avg_ep_dp'] = self.optim._flat_exp_avg_ep_dp
        # og_optim_sd['_flat_exp_avg_sq_dp '] = self.optim._flat_exp_avg_sq_dp
        # og_optim_sd['_flat_exp_avg_sq_ep_dp  '] = self.optim._flat_exp_avg_sq_ep_dp
        # torch.save(og_optim_sd, f'tmp.optim.{dist.get_rank()}.pt')

        # wrap it in dtensor so that it works with checkpointer
        wrapped_model_sd = OrderedDict()
        for k, v in plain_model_sd.items():
            if self.ep_enabled and isinstance(v, torch.Tensor) and "routed_experts." in k:
                assert self.world_mesh is not None
                assert self.world_mesh["moe"] is not None
                wrapped_model_sd[k] = DTensor.from_local(
                    v,
                    device_mesh=self.world_mesh["moe"]["ep_dp", "ep_mp"],
                    placements=(Replicate(), Shard(0)),
                )
            elif isinstance(v, torch.Tensor):
                assert self.world_mesh is not None
                assert self.world_mesh["dense"] is not None
                wrapped_model_sd[k] = DTensor.from_local(
                    v, device_mesh=self.world_mesh["dense"]["dp"], placements=(Replicate(),)
                )
            else:
                wrapped_model_sd[k] = v

        state_dict: Dict[str, Any] = {
            "model": wrapped_model_sd,
        }

        if optim:
            optim_sd = self.optim.state_dict()
            state_dict["optim"] = optim_sd

        return state_dict

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        assert (
            self.pp_enabled == False
        ), "Pipeline parallelism with grad clipping inside train module is not supported yet. Use optimizer's grad clipping instead."
        parameters = [p for p in self.model_parts[0].parameters()]
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
            raise NotImplementedError("WHY?")
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if self.pp_enabled:
            input_ids = batch.pop("input_ids")
            labels = labels if labels is not None else batch.pop("labels", None)
            kwargs: Dict[str, Any] = {}
            if "doc_lens" in batch and "max_doc_lens" in batch:
                log_once(log, "intra-document masking enabled")
                kwargs["doc_lens"] = batch["doc_lens"]
                kwargs["max_doc_lens"] = batch["max_doc_lens"]
            return input_ids, labels, kwargs
        else:
            input_ids = batch.pop("input_ids")
            labels = labels if labels is not None else batch.pop("labels", None)
            if "doc_lens" in batch and "max_doc_lens" in batch:
                log_once(log, "intra-document masking enabled")
            return input_ids, labels, batch

    def parallelize_model(
        self,
        model: MoEFusedV2Transformer,  # the full model before sharding, the same on all models
        *,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
    ) -> List["MoEFusedV2Transformer"]:
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer

        assert isinstance(model, MoEFusedV2Transformer), "model must be an instance of Transformer"

        if tp_config is not None:
            raise NotImplementedError("TP not supported yet")
        if cp_config is not None:
            raise NotImplementedError("CP not supported yet")

        if pp_config is not None:
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying pipeline parallelism"
            self.pp_mesh = self.world_mesh["dense"]["pp"]
            self.pp_group = self.pp_mesh.get_group()
            self.pp_group_rank = get_rank(self.pp_group)
            self.pp_group_size = get_world_size(self.pp_group)
            self.pp_prev_rank = (self.pp_group_rank - 1) % self.pp_group_size
            self.pp_next_rank = (self.pp_group_rank + 1) % self.pp_group_size
            self.pp_final_stage_rank = pp_config.final_stage_rank()

            # Split model into pipeline stages.
            model.purge_cuda_events()  # set event to None so that can be deepcopied

            stages_and_model_parts = pp_config.split_model(
                model,
                pp_mesh=self.pp_mesh,
                device=self.device,
                use_ddp=self.world_mesh["dense"]["dp"].size() > 1,
            )
            stages = stages_and_model_parts[0]

            model_parts: List[MoEFusedV2Transformer] = cast(
                List[MoEFusedV2Transformer], stages_and_model_parts[1]
            )

            for model_part in model_parts:
                assert isinstance(model_part, MoEFusedV2Transformer)
                model_part.install_cuda_events()

            self._pp_stages = stages
            log.info(
                f"Applied pipeline parallelism to the model with {get_device_mesh_info(self.pp_mesh)}"
            )

            # TODO: chunk layers into stages
            assert (
                self.world_mesh is not None
            ), "World mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying expert parallelism"

            for m in model_parts:
                m.apply_pp(self.pp_mesh)

        else:
            model_parts: List[MoEFusedV2Transformer] = [model]  # no PP, single part

        # Maybe apply FP8 training.
        if float8_config is not None and float8_config.enabled:
            for m in model_parts:
                m.apply_fp8(float8_config)
                log.info("Swapped linear layers to Float8 linear layers\n%s", m)

        assert dp_config is not None

        if ep_config is not None:
            # EP-DP combined
            # for the dense part, replicate over DP pg
            # for the moe part, replicate over EP-DP pg
            assert (
                self.world_mesh is not None
            ), "World mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["moe"] is not None
            ), "MoE mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying expert parallelism"
            ep_mesh = self.world_mesh["moe"]
            dp_mesh = self.world_mesh["dense"]["dp"]
            for m in model_parts:
                if not m.is_moe:
                    raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
                cast(MoEFusedV2Transformer, m).apply_epdp(
                    dp_mesh=dp_mesh,
                    ep_mesh=ep_mesh,
                    param_dtype=None,
                    compile_enabled=compile_model,
                )
            log.info(
                f"Applied expert parallelism + DDP to the model with {get_device_mesh_info(ep_mesh)}"
            )
        else:
            # Pure DP (no EP)
            assert (
                self.world_mesh is not None
            ), "World mesh must be built before applying expert parallelism"
            assert (
                self.world_mesh["dense"] is not None
            ), "Dense mesh must be built before applying expert parallelism"
            param_dtype = (
                dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            )
            dp_mesh = self.world_mesh["dense"]["dp"]
            for m in model_parts:
                cast(MoEFusedV2Transformer, m).apply_ddp(
                    dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype
                )
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")

            # Maybe apply activation checkpointing.
        if ac_config is not None:
            for m in model_parts:
                m.apply_activation_checkpointing(
                    ac_config.mode,
                    block_interval=ac_config.block_interval,
                    modules=ac_config.modules,
                    activation_memory_budget=ac_config.activation_memory_budget,
                )
            log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

        return model_parts

    def memory_usage_estimation(self):
        pass

    def init_model_weights(
        self,
        max_sequence_length: int,
        rank_microbatch_size: int,
    ):
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer

        # Materialize and init parameters.
        log.info("Initializing model weights...")
        for model_part_idx, m in enumerate(self.model_parts):
            m = cast(MoEFusedV2Transformer, m)
            m.init_weights(
                max_seq_len=max_sequence_length,
                max_local_microbatch_size=rank_microbatch_size,
                device=self.device,
                world_mesh=self.world_mesh,  # only PP mesh is used, should be fine
                model_part_idx=model_part_idx,
            )

    def compile_model(self):
        if torch.cuda.is_available():
            for m in self.model_parts:
                m.apply_compile()
            log.info("Applied torch.compile() to the model")
        else:
            log.warning("Skipping model compilation since CUDA is not available")

    def reduce_send_recv(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: review
        assert (
            self.pp_enabled and self._pp_config is not None
        ), "reduce_send_recv is only valid when PP is enabled"
        if self.pp_group_rank == self.pp_final_stage_rank:
            assert x is not None
            # Reduce across DP process group.
            x.div_(self._reduce_divide_factor)
            dist.all_reduce(x, group=self.dp_process_group)
            x.div_(self.dp_world_size)
            x.mul_(self._reduce_divide_factor)
        else:
            assert x is None
            x = move_to_device(torch.empty([]), self.device)

        # Asynchronously send to previous stage rank in the PP group.
        ordered_ranks = list(self._pp_config.rank_completion_order())
        local_index = ordered_ranks.index(self.pp_group_rank)
        src_rank = None if local_index == 0 else ordered_ranks[local_index - 1]
        dst_rank = (
            None if local_index == (len(ordered_ranks) - 1) else ordered_ranks[local_index + 1]
        )

        ops: List[dist.P2POp] = []
        if src_rank is not None:
            log.debug(
                f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) receiving from rank "
                f"{get_global_rank(src_rank, group=self.pp_group)} (pp group rank {src_rank})"
            )
            ops.append(dist.P2POp(dist.irecv, x, group=self.pp_group, group_peer=src_rank))
        if dst_rank is not None:
            log.debug(
                f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) sending to rank "
                f"{get_global_rank(dst_rank, group=self.pp_group)} (pp group rank {dst_rank})"
            )
            ops.append(dist.P2POp(dist.isend, x, group=self.pp_group, group_peer=dst_rank))

        reqs = dist.batch_isend_irecv(ops)
        if self.pp_group_rank != self.pp_final_stage_rank:
            for req in reqs:
                req.wait()

        return x
