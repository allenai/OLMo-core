import contextlib
import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, Generator, Optional, Tuple, Union, Iterable, Sequence

from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from typing import List, Optional, Tuple
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.pipelining import PipelineStage
from ...common import MetricMergeStrategy, ReduceType
import math
from olmo_core.aliases import PathOrStr
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from olmo_core.distributed.checkpoint import _prepare_env_for_save, RemoteFileSystemWriter, RemoteFileSystemReader
from torch.distributed import ProcessGroup
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
)
from olmo_core.distributed.utils import (
    get_local_tensor,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
    get_global_rank,
    get_rank,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import OptimConfig, SkipStepOptimizer, MoEFusedV2OptimizerConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, log_once, move_to_device
from typing import List, Optional, TypeVar, cast
from olmo_core.nn.transformer import MoETransformer, Transformer
from collections import OrderedDict
from ...common import ReduceType
from ..train_module import EvalBatchSpec, TrainModule

from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
    TransformerPipelineParallelConfig
)
from olmo_core.distributed.parallel.data_parallel import DataParallelConfig, DataParallelType, DPMeshDimName
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig
from olmo_core.distributed.parallel.pipeline_parallel import (
    PipelineParallelConfig,
    PipelineSchedule,
    PipelineScheduleType,
    PipelineSplitStyle,
)
from olmo_core.distributed.parallel.tensor_parallel import TensorParallelConfig
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel import MeshDimName, get_device_mesh_info
from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel
import nvtx

log = logging.getLogger(__name__)

M = TypeVar("M", bound=List[MoEFusedV2Transformer])

def cpu_mesh_like(gpu_mesh: DeviceMesh) -> DeviceMesh:
    # gpu_mesh.mesh is a CPU int tensor of ranks with the mesh's shape
    ranks = gpu_mesh.mesh.clone()          # e.g., tensor([[0,1],[2,3]]) or nested shape
    return DeviceMesh(
        "cpu",
        ranks,                             # keep the exact shape
        mesh_dim_names=gpu_mesh.mesh_dim_names,
    )


class FlatSavePlanner(DefaultSavePlanner):
    pass

from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.default_planner import (
    create_default_global_load_plan,
    create_default_local_load_plan,
    create_default_global_save_plan,
    create_default_local_save_plan,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    StorageMeta,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_default_metadata_only_plan,
    _create_read_items,
    _create_write_items,
    _init_state_dict,
)

def debug_check_grad(name, tag, tensor, input_ids, micro_batch_idx):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        # save input_ids for debugging
        # if input_ids is not None:
        #     torch.save(input_ids, f"input_ids_rank{dist.get_rank()}_mbx{micro_batch_idx}.pt")
        # raise RuntimeError(f"rank={dist.get_rank()} mbx={micro_batch_idx} NaN or Inf detected in {name} {tag}")
        print(f"rank={dist.get_rank()} mbx={micro_batch_idx} NaN or Inf detected in {name} {tag}")
        return True
    return False

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
        reduce_scatter_grads: bool = False,
        eval_only: bool = False,
    ):
        super().__init__()
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer
        assert isinstance(model, MoEFusedV2Transformer), "MoEV2TransformerTrainModule only supports MoEFusedV2Transformer model"
        
        ######################### Validate arguments. [BEGIN] #########################
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )
        self.max_sequence_length = max_sequence_length
        self.rank_microbatch_size = rank_microbatch_size
        self.eval_only = eval_only
        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Dict[str, Optional[DeviceMesh]] = {}
        self.pp_group = None
        self.dp_group = None
        self.tp_group = None
        self.ep_dp_group = None
        self.ep_mp_group = None
        self.ep_mp_high_priority_group = None
        self.ep_mp_high_priority_groups = None

        # compatibility
        if autocast_precision is not None:
            assert False, "Autocast precision is not supported in MoEV2TransformerTrainModule"
        self.autocast_precision = None

        # PP related state.
        self._train_pp_schedule: Optional[PipelineSchedule] = None
        self._pp_stages: Optional[List[PipelineStage]] = None
        self.pp_group_rank = 0 # default 0
        self.pp_group_size = 1 # default 1
        self.pp_prev_rank = -1 # no previous stage
        self.pp_next_rank = -1 # no next stage
        self.pp_final_stage_rank = 0 # default 0

        # If True, the DDP will not all-reduce grad for the last microbatch
        # instead it will call optim.step() which will reduce_scatter 
        # directly into the owner main grad 
        self.reduce_scatter_grads = reduce_scatter_grads

        if tp_config is not None:
            assert tp_config.degree > 1, "Tensor parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Tensor parallelism is not implemented")
        if pp_config is not None:
            assert pp_config.degree > 1, "Pipeline parallelism requires a degree > 1, otherwise use None"
            # raise NotImplementedError("Pipeline parallelism is not implemented")
        if cp_config is not None:
            assert cp_config.degree > 1, "Context parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Context parallelism is not implemented")
        # if ac_config is not None:
        #     raise OLMoConfigurationError("In MoEV2TransformerTrainModule, activation checkpointing is controlled by the model, not the train module.")
        if float8_config is not None:
            raise NotImplementedError("Float8 quantization is not implemented")

        assert dp_config is not None, "Data parallel config is required for MoEV2TransformerTrainModule"
        assert dp_config.name == "ddp", "Data parallel config must be 'ddp'"

        ########################## Validate arguments. [END] ##########################

        if is_distributed():
            self._build_world_mesh(
                dp=dp_config, tp=tp_config, cp=cp_config, ep=ep_config, pp=pp_config, device_type=self.device.type
            )
            self.dp_world_size = get_world_size(self.dp_process_group)
            log.info(f"Data parallel world size = {self.dp_world_size:,d}")
            assert self.world_mesh['dense'] is not None
        else:
            raise OLMoConfigurationError(
                "Training parallelism is required for MoEV2TransformerTrainModule"
            )


        # Parallelize model.
        self.model_parts = self.parallelize_and_init_model(
                    model,
                    compile_model=compile_model,
                    float8_config=float8_config,
                    dp_config=dp_config,
                    tp_config=tp_config,
                    cp_config=cp_config,
                    ep_config=ep_config,
                    ac_config=ac_config,
                    pp_config=pp_config,
                    eval_only=eval_only,
                )

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

        self.max_grad_norm = max_grad_norm # TODO: remove, use optim.max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping

        self.optim = None
        if not self.eval_only:
            # Build optimizer(s).
            log.info("Building optimizer...")

            from olmo_core.optim.moe_optimizer import MoEFusedV2Optimizer, MoEFusedV2OptimizerConfig
            assert isinstance(optim, MoEFusedV2OptimizerConfig)
            optim = cast(MoEFusedV2OptimizerConfig, optim)
            self.optim = optim.build(
                self.model_parts, 
                self, 
                strict=False # group_overrides might only be matched in one group, strict=False allows it to not match in one group (could match in some other group),
            )

            if self.reduce_scatter_grads and isinstance(self.model_parts[0], MultiGroupDistributedDataParallel):
                raise NotImplementedError(
                    "reduce_scatter_grads=True is incompatible with MultiGroupDistributedDataParallel. "
                    "Disable DDP all-reduce path first or use reduce_scatter_grads=False."
                )

            self.optim.set_reduce_scatter_grads(self.reduce_scatter_grads)
        else:
            log.info("Skipping optimizer build because eval_only=True")

    def _require_optimizer(self):
        if self.optim is None:
            raise RuntimeError(
                f"{self.__class__.__name__} was built with eval_only=True and has no optimizer"
            )
        return self.optim

    def _cast_to_fwd_bwd_precision(self, model: Union[MoEFusedV2Transformer, List[MoEFusedV2Transformer]]) -> None:
        """
        Cast the model to the forward and backward precision.
        """
        if isinstance(model, list):
            for m in model:
                self._cast_to_fwd_bwd_precision(m)
        else:
            # if the model has implemented `cast_to_fwd_bwd_precision` method, call it.
            if hasattr(model, "_cast_to_fwd_bwd_precision"):
                assert callable(model._cast_to_fwd_bwd_precision), \
                    "model._cast_to_fwd_bwd_precision must be callable"
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
            raise RuntimeError("world mesh already exists! You can only call 'build_world_mesh' once!")

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
        use_paired_pp = False # TODO: move to config
        # TODO 2: implement paired pp properly
        if pp is not None:
            names.append(MeshDimName.pp)
            if use_paired_pp:
                dims.append(pp.degree // 2)
            else:
                dims.append(pp.degree)

        # Then data parallel.
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)

        if pp is not None and use_paired_pp:
            names.append('pp_paired')
            dims.append(2)

        with torch.device("cpu"):
            mesh = torch.arange(math.prod(tuple(dims)), dtype=torch.int).view(tuple(dims))

        if pp is not None and use_paired_pp:
            r = mesh.permute(0, 2, 1).contiguous()
            pp_dim, pp_paired, dp_dim = r.shape
            r_merged = r.reshape(pp_dim * pp_paired, dp_dim)      # (pp*pp_paired, dp)
            mesh = r_merged
            names.remove('pp_paired')

        self.dense_mesh  = DeviceMesh(
            device_type=device_type,
            mesh=mesh,
            mesh_dim_names=tuple(names),
        )
        # self.dense_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
        log.info(f"Built dense_mesh {get_device_mesh_info(self.dense_mesh)}")

        if ep is None: # EP not used
            self.moe_mesh = None
            log.info(f"Built moe_mesh None")

        else:
            # Build up moe mesh dimensions. (PP , EP_DP, EP_MP)
            names: List[str] = []
            dims: List[int] = []

            # Pipeline parallel first.
            if pp is not None:
                    
                names.append(MeshDimName.pp)
                if use_paired_pp:
                    dims.append(pp.degree // 2)
                else:
                    dims.append(pp.degree)

            # Then EP data parallel.
            ep_dp_world_size = dp_world_size // ep.degree
            names.append(MeshDimName.ep_dp)
            dims.append(ep_dp_world_size)

            # Then EP model parallel.
            ep_mp_world_size = ep.degree
            names.append(MeshDimName.ep_mp)
            dims.append(ep_mp_world_size)

            if pp is not None and use_paired_pp:
                names.append('pp_paired')
                dims.append(2)

            with torch.device("cpu"):
                mesh = torch.arange(math.prod(tuple(dims)), dtype=torch.int).view(tuple(dims))

            if pp is not None and use_paired_pp:
                r = mesh.permute(0, 3, 1, 2).contiguous() # (pp, ep_dp, ep_mp, pp_paired) -> (pp, pp_paired, ep_dp, ep_mp)
                pp_dim, pp_paired, ep_dp_dim, ep_mp_dim = r.shape
                r_merged = r.reshape(pp_dim * pp_paired, ep_dp_dim, ep_mp_dim)      # (pp*pp_paired, ep_dp, ep_mp)
                mesh = r_merged
                names.remove('pp_paired')
            
            device_mesh = DeviceMesh(
                device_type=device_type,
                mesh=mesh,
                mesh_dim_names=tuple(names),
            )
            self.moe_mesh = device_mesh
            # self.moe_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))

            log.info(f"Built moe_mesh{get_device_mesh_info(self.moe_mesh)}")

        # build process group

        if pp is not None:
            # self.dense_mesh['pp'] and self.moe_mesh['pp'] should be the same
            self.pp_group = self.dense_mesh['pp'].get_group() # 
            
        self.dp_group = self.dense_mesh['dp'].get_group()
        if self.moe_mesh is not None:
            self.ep_dp_group = self.moe_mesh['ep_dp'].get_group()
            self.ep_mp_group = self.moe_mesh['ep_mp'].get_group()

        self.world_mesh = {
            "dense": self.dense_mesh,
            "moe": self.moe_mesh,
            "dense_cpu": cpu_mesh_like(self.dense_mesh),
            "moe_cpu": None if self.moe_mesh is None else cpu_mesh_like(self.moe_mesh),
        }

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
            pass # BUG: when batch size warmup + load checkpoint

        if self.pp_enabled:
            # Initialize pipeline schedule.
            assert self._train_pp_schedule is None  # make sure we don't initialize this twice
            assert self._pp_stages is not None
            assert self._pp_config is not None
            assert self.world_mesh['dense'] is not None
            pp_mesh = self.world_mesh['dense']['pp']
            assert pp_mesh is not None

            # Determine the number of micro-batches.
            rank_batch_size = self.trainer.global_batch_size // dp_ws
            num_microbatches = rank_batch_size // self.rank_microbatch_size

            self._train_pp_schedule = PipelineSchedule(
                model_parts=self.model_parts,  # type: ignore[arg-type]
                stages=self._pp_stages,
                pp_mesh=pp_mesh,
                schedule_name=self._pp_config.schedule,

                num_microbatches=num_microbatches,
            )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError("Use load_state_dict_direct instead")

    def state_dict_to_load(self, metadata: Metadata, *, optim: Optional[bool] = True) -> Dict[str, Any]:
        raise NotImplementedError("Use load_state_dict_direct instead")
    
    def state_dict(self, *, optim: Optional[bool] = True) -> Dict[str, Any]:
        raise NotImplementedError("Use save_state_dict_direct or load_state_dict_direct instead")

    def save_state_dict_direct(
        self,
        dir: PathOrStr,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        save_overwrite: bool = False,
        thread_count: Optional[int] = None,
        throttle_uploads: bool = False,
    ):
        optim = self._require_optimizer()
        state_dict = optim.state_dict() # this will free optim states, need to load back after save

        # this is count the param size of the global dtensor, not the local shard
        main_param_sz = 0 
        for key, value in state_dict.items():
            if key.endswith('.main'):
                main_param_sz += value.numel()

        # this is the theretical model param size calculated from config before PP split
        # model_param_sz = self.model_parts[0].num_params

        # assert main_param_sz == model_param_sz, f"Main param size {main_param_sz} != model param size {model_param_sz}"

        dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
        planner = FlatSavePlanner(dedup_save_to_lowest_rank=True)
        dist_cp.state_dict_saver.save(
            state_dict,
            storage_writer=RemoteFileSystemWriter(
                dir,
                thread_count=thread_count,
                process_group=process_group,
                throttle_uploads=throttle_uploads,
            ),
            process_group=process_group,
            planner=planner,
        )

        optim.load_state_dict(
            state_dict,
            reset_optimizer_moments_on_load=False,
        )  # load back the optim state after save

        return

    def load_state_dict_direct(
        self,
        dir: PathOrStr,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        pre_download: bool = False,
        work_dir: Optional[PathOrStr] = None,
        thread_count: Optional[int] = None,
    ):
        from olmo_core.io import normalize_path

        dir = normalize_path(dir)
        reader = RemoteFileSystemReader(
            dir, 
            thread_count=thread_count, 
            pre_download=pre_download, work_dir=work_dir
        )

        metadata = reader.read_metadata()

        if self.eval_only:
            sd_to_load = self._get_model_state_dict_for_eval_load(metadata)
            dist_cp.state_dict_loader.load(
                sd_to_load,
                checkpoint_id=dir,
                storage_reader=reader,
                process_group=process_group,
            )
        else:
            optim = self._require_optimizer()
            sd_to_load = optim.state_dict()
            checkpoint_keys = set(metadata.state_dict_metadata.keys())
            reset_optimizer_moments_on_load = getattr(
                optim, "reset_optimizer_moments_on_load", False
            )

            # Backward compatibility: old checkpoints won't have rolling skip-step stats.
            for optional_key in (
                optim.LOSSES_STATE_DICT_KEY,
                optim.GRAD_NORMS_STATE_DICT_KEY,
            ):
                if optional_key not in checkpoint_keys:
                    sd_to_load.pop(optional_key, None)

            if reset_optimizer_moments_on_load:
                log.info("Resetting optimizer exp_avg and exp_avg_sq buffers during checkpoint load")
                for key in list(sd_to_load.keys()):
                    if key.endswith(".exp_avg") or key.endswith(".exp_avg_sq"):
                        sd_to_load.pop(key)

            dist_cp.state_dict_loader.load(
                sd_to_load,
                checkpoint_id=dir,
                storage_reader=reader,
                process_group=process_group,
                # planner=FlatLoadPlanner(),
            )

            optim.load_state_dict(sd_to_load)

            # load into model params
            optim._copy_main_params_to_model_params()

        torch.cuda.empty_cache()

        return

    def _get_model_state_dict_for_eval_load(self, metadata: Metadata) -> Dict[str, Any]:
        model_state: Dict[str, Any] = {}
        checkpoint_keys = set(metadata.state_dict_metadata.keys())

        for model_part in self.model_parts:
            for name, param in model_part.named_parameters():
                checkpoint_key = self._resolve_model_checkpoint_key(name, checkpoint_keys)
                if checkpoint_key is None:
                    continue

                tensor_meta = metadata.state_dict_metadata[checkpoint_key]
                assert isinstance(tensor_meta, TensorStorageMetadata)
                global_numel = tensor_meta.size.numel()
                local_flat = param.data.view(-1)

                if local_flat.numel() == global_numel:
                    model_state[checkpoint_key] = local_flat
                else:
                    if self.world_mesh["moe"] is None:
                        raise RuntimeError(
                            f"Expected MoE mesh when loading sharded param '{name}' in eval mode"
                        )

                    model_state[checkpoint_key] = DTensor.from_local(
                        local_flat,
                        device_mesh=self.world_mesh["moe"]["ep_dp", "ep_mp"],
                        placements=[Replicate(), Shard(0)],
                        shape=(global_numel,),
                        stride=(1,),
                        run_check=False,
                    )

        if not model_state:
            raise RuntimeError("Did not find any model weights to load in eval mode")

        return model_state

    def _resolve_model_checkpoint_key(
        self, param_name: str, checkpoint_keys: Sequence[str]
    ) -> Optional[str]:
        candidates = (
            f"{param_name}.main",
            f"module.{param_name}.main",
        )
        for key in candidates:
            if key in checkpoint_keys:
                return key
        return None



    def _dump_debug_info(self, step=None, tag=None):
        # dump model param stats
        debug_info = {}
        step = self.trainer.global_step if step is None else step
        rank = dist.get_rank()
        for m_idx, model_part in enumerate(self.model_parts):
            debug_info[m_idx] = {}
            for name, param in model_part.named_parameters():
                grad = param._main_grad_fp32
                debug_info[m_idx][f'{name}'] = {
                    'param': {
                        'mean': param.data.mean().item(),
                        'std': param.data.std().item(),
                        'max': param.data.max().item(),
                        'min': param.data.min().item(),
                        'norm': param.data.norm().item(),
                    },
                    'grad': {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),   
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'norm': grad.norm().item(),
                    }
                }
        base_dir = './debug_info'
        filename = f'{base_dir}/rank{rank}_step{step}'
        if tag is not None:
            filename += f'_{tag}'
        filename += '.pt'
        import os
        os.makedirs(base_dir, exist_ok=True)
        torch.save(debug_info, filename)


    @nvtx.annotate("train_batch")
    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._require_optimizer()
        DEBUG_MODE = False

        # Set model to train mode if it isn't already.
        for m in self.model_parts:
            m.train()

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        instance_mask = batch.get("instance_mask")
        if (instance_mask) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

            if (~instance_mask).all():
                print(f'[Warning] rank {dist.get_rank()} All instances ({instance_mask.shape}) in the micro-batch are masked out')

        #############################

        if not self.pp_enabled:
            # Calculate how many tokens are going to be used in the loss.
            batch_num_tokens_for_loss = (batch["labels"] != self.label_ignore_index).sum()

            if instance_mask is not None:
                # WARN: When we mask out instances with the instance filter, we count those tokens
                # for the loss anyways. They will count as tokens with a zero loss. This means we
                # get an artificially *low* loss for these batches. But it is really hard (and slow)
                # to do this properly in a distributed setup. We add back in the full number of tokens
                # for the loss so that each rank contributes to the loss calculation fairly.
                batch_num_tokens_for_loss += (~instance_mask).sum() * (batch["labels"].shape[1] - 1) # shifted labels does not count last token
            
            if batch_num_tokens_for_loss.item() == 0:
                print(f'[Warning] rank {dist.get_rank()} batch_num_tokens_for_loss == 0')

            batch_num_tokens_for_loss = move_to_device(batch_num_tokens_for_loss, self.device)

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

            dbg_mem_before_fwd0 = torch.cuda.memory_allocated()/1024**3

            dbg_mem_activation_usage_all = []
            dbg_mem_activation_freed_all = []

            # for name, param in self.model_parts[0].named_parameters():
            #     debug_check_grad(name, "param", param.data, input_ids=None, micro_batch_idx=-1)

            # Train one micro-batch at a time.
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                    with nvtx.annotate(f"fwd_mb{micro_batch_idx}", color='blue'):
                        
                        input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
                        dbg_mem_before_fwd = torch.cuda.memory_allocated()/1024**3
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
                        dbg_mem_after_fwd = torch.cuda.memory_allocated()/1024**3
                        dbg_mem_activation_usage = dbg_mem_after_fwd - dbg_mem_before_fwd
                        dbg_mem_activation_usage_all.append(dbg_mem_activation_usage)
                        # Update total batch CE and Z loss.
                        ce_batch_loss += get_local_tensor(ce_loss.detach())
                        del ce_loss
                        if z_batch_loss is not None:
                            assert z_loss is not None
                            z_batch_loss += get_local_tensor(z_loss.detach())
                            del z_loss

                        # if dry_run:
                        #     torch.cuda.empty_cache()
                        #     print(f'[Dry Run {dist.get_rank()}] after fwd mb{micro_batch_idx} {torch.cuda.memory_allocated()/1024**3:.2f} GB, activation used {dbg_mem_activation_usage:.2f} GB')

                    with nvtx.annotate(f"bwd_mb{micro_batch_idx}", color='red'):
                        # Run backward pass.
                        dbg_mem_before_bwd = torch.cuda.memory_allocated()/1024**3


                        loss.backward()

                        dbg_mem_after_bwd = torch.cuda.memory_allocated()/1024**3
                        dbg_mem_activation_freed = dbg_mem_before_bwd - dbg_mem_after_bwd
                        dbg_mem_activation_freed_all.append(dbg_mem_activation_freed)

                        # if DEBUG_MODE and not dry_run and self.trainer.global_step == 23525:
                        #     if micro_batch_idx == 17 and dist.get_rank() == 17:
                        #         torch.save(input_ids.cpu(), f'input_ids_rank{dist.get_rank()}_step{self.trainer.global_step}_mb{micro_batch_idx}.pt')
                            # self._dump_debug_info(step=micro_batch_idx)
                        pass


            del batch  # In case this helps with memory utilization.

            if dry_run:
                symm_total_bytes = 0
                symm_block_count = 0
                symm_unique_storage_count = 0
                seen_module_ids = set()
                seen_storages = set()

                def account_tensor_storage(tensor: torch.Tensor):
                    nonlocal symm_total_bytes, symm_unique_storage_count
                    storage = tensor.untyped_storage()
                    key = (storage.data_ptr(), storage.nbytes(), str(tensor.device))
                    if key in seen_storages:
                        return
                    seen_storages.add(key)
                    symm_total_bytes += storage.nbytes()
                    symm_unique_storage_count += 1

                for model_part in self.model_parts:
                    for module in model_part.modules():
                        module_id = id(module)
                        if module_id in seen_module_ids:
                            continue
                        seen_module_ids.add(module_id)
                        symm_cache = getattr(module, "_ep_no_sync_symm_cache", None)
                        shared_pool = getattr(module, "_ep_no_sync_shared_pool", None)
                        has_local = bool(symm_cache)
                        has_shared = shared_pool is not None
                        if not has_local and not has_shared:
                            continue
                        symm_block_count += 1
                        if symm_cache:
                            for tensor in symm_cache.values():
                                if isinstance(tensor, torch.Tensor):
                                    account_tensor_storage(tensor)
                        if shared_pool is not None:
                            for tensor in shared_pool.iter_tensors():
                                if isinstance(tensor, torch.Tensor):
                                    account_tensor_storage(tensor)
                print(
                    f"[symm_mem] step={self.trainer.global_step} rank={dist.get_rank()} "
                    f"blocks={symm_block_count} unique_storages={symm_unique_storage_count} "
                    f"total={symm_total_bytes / (1024 ** 3):.3f} GiB "
                    f"({symm_total_bytes} bytes)"
                )
                print("activation: ", dbg_mem_activation_usage_all)
                print("freed:      ", dbg_mem_activation_freed_all)
                for (tag, mem) in self.model_parts[0]._debug_alloc_mem_layer_logs:
                    print(f"Alloc - {tag}: {mem:.2f} GB")
                for (tag, mem) in self.model_parts[0]._debug_max_alloc_mem_layer_logs:
                    print(f"Max - {tag}: {mem:.2f} GB")



            if DEBUG_MODE and not dry_run:
                self._dump_debug_info()

            for model in self.model_parts:
                model.finalize_grad_reduce()
            
        else:
            # pipeline parallel forward / backward
            # Calculate how many tokens are going to be used in the loss.
            batch_num_tokens_for_loss = (batch['labels'] != self.label_ignore_index).sum().item()



            if instance_mask is not None:

                # WARN: When we mask out instances with the instance filter, we count those tokens
                # for the loss anyways. They will count as tokens with a zero loss. This means we
                # get an artificially *low* loss for these batches. But it is really hard (and slow)
                # to do this properly in a distributed setup. We add back in the full number of tokens
                # for the loss so that each rank contributes to the loss calculation fairly.
                batch_num_tokens_for_loss += (~instance_mask).sum() * (batch["labels"].shape[1] - 1) # shifted labels does not count last token
            
            if batch_num_tokens_for_loss == 0:
                print(f'[Warning] rank {dist.get_rank()} batch_num_tokens_for_loss == 0')

            # Run pipeline schedule.
            input_ids, labels, model_kwargs = self._prepare_batch(batch, batch['labels'])
            assert labels is not None
            pp_outputs = self.run_pipeline(
                input_ids,
                labels,
                batch_num_tokens_for_loss,
                ignore_index=self.label_ignore_index,
                loss_reduction="sum",
                z_loss_multiplier=self.z_loss_multiplier,
                return_logits=False,
                **model_kwargs,
            )   
            # Collect losses from all micro-batches and all stages.
            ce_batch_loss = None
            z_batch_loss = None
            for stage_outputs in pp_outputs:
                for mb_output in stage_outputs:
                    if mb_output is None: # non-last stage
                        continue
                    else:
                        assert isinstance(mb_output, LMOutputWithLoss)
                        _, loss, ce_loss, z_loss = mb_output

                        # ce_loss should always be not None
                        ce_batch_loss = (ce_batch_loss + ce_loss.detach()) if ce_batch_loss is not None else ce_loss.detach()

                        # z loss is optional
                        if z_loss is not None: 
                            z_batch_loss = (z_batch_loss + z_loss.detach()) if z_batch_loss is not None else z_loss.detach()

            if DEBUG_MODE and not dry_run:
                self._dump_debug_info()


            for model in self.model_parts:
                model.finalize_grad_reduce()
            
        #############################


        for model in self.model_parts:
            model.post_batch(dry_run=dry_run)

        if dry_run:
            for model in self.model_parts:
                model.reset_auxiliary_metrics()

            torch.cuda.empty_cache()
            from olmo_core.train.globals import set_global_arg
            set_global_arg("dry_run_done", True)
            return

        # Record loss metrics.
        from olmo_core.optim.moe_optimizer import MoEFusedV2Optimizer
        optim = self._require_optimizer()
        with nvtx.annotate("record_metrics"):
            if self.pp_enabled:
                ce_batch_loss = self.reduce_send_recv(ce_batch_loss)
                # if ce_batch_loss > 20 or ce_batch_loss < 0.05:
                #     log.warning(f"Irregular CE loss detected: {ce_batch_loss.item()}")
                self.record_ce_loss(ce_batch_loss)
                optim.latest_loss = ce_batch_loss
            else:
                assert ce_batch_loss is not None, "CE loss should not be None"
                # Need to reduce the loss right away for the SkipStepOptimizer. NOTE: WHY?
                if is_distributed():
                    ce_batch_loss.div_(self._reduce_divide_factor)
                    dist.all_reduce(ce_batch_loss)
                    ce_batch_loss.div_(self.world_size)
                    ce_batch_loss.mul_(self._reduce_divide_factor)
                self.record_ce_loss(ce_batch_loss)
                optim.latest_loss = ce_batch_loss

    
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

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print("-------train batch end--------")

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Optional[LMOutputWithLoss]]:

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
            if not self.pp_enabled:
                lm_output = self.model_forward_no_pipeline(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    **model_kwargs,
                )
                assert isinstance(lm_output, LMOutputWithLoss), "Expected LMOutputWithLoss"
                return lm_output

            else:
                assert labels is not None
                lm_output = self.run_pipeline_eval(
                    input_ids,
                    labels,
                    batch_num_tokens_for_loss=None,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="none",
                    **model_kwargs,
                ) 
                # merge lm_output from all stages and all micro-batches
                # List[List[Optional[LMOutputWithLoss]]] --> Optional[LMOutputWithLoss]
                final_lm_output: Optional[LMOutputWithLoss] = None
                logits_list = []
                loss_list = []
                ce_loss_list = []
                z_loss_list = []

                # NOTE: here we combine the multiple stages assuming they are different stages
                # with the same DP rank (e.g., stage 0 and stage 1 in X-stage PP, working on the same batch data).
                # so at max one of the stage_outputs will have non-None output.
                # TODO: this assumption might not hold in dualpipe and other schedules.
                for stage_outputs in lm_output:
                    for mb_output in stage_outputs:
                        if mb_output is None:
                            continue
                        else:
                            assert isinstance(mb_output, LMOutputWithLoss)
                            (
                                logits, # [mb, seq_len, vocab_size]
                                loss,  # [mb, seq_len] because loss_reduction="none"
                                ce_loss, # [mb, seq_len]
                                z_loss,  # ??
                             ) = mb_output
                            
                            # always not None
                            loss_list.append(loss)
                            ce_loss_list.append(ce_loss)

                            # optional
                            if logits is not None:
                                logits_list.append(logits)
                            if z_loss is not None:
                                z_loss_list.append(z_loss)

                # Concatenate all logits and losses
                merged_logits = torch.cat(logits_list, dim=0) if len(logits_list) > 0 else None
                merged_loss = torch.cat(loss_list, dim=0) if len(loss_list) > 0 else None
                merged_ce_loss = torch.cat(ce_loss_list, dim=0) if len(ce_loss_list) > 0 else None
                merged_z_loss = torch.cat(z_loss_list, dim=0) if len(z_loss_list) > 0 else None

                if merged_loss is not None:
                    # has last stage output
                    assert merged_ce_loss is not None
                    final_lm_output = LMOutputWithLoss(
                        merged_logits,
                        merged_loss,
                        merged_ce_loss,
                        merged_z_loss,
                    )
                else:
                    # no last stage output
                    final_lm_output = None

                return final_lm_output

    def run_pipeline_eval(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> List[List[Optional[LMOutputWithLoss]]]:

        # the micro-batch size should be a multiple of pp degree
        def pad_dim_0_to_multiple_of(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
            bsz = tensor.size(0)
            if bsz % multiple == 0:
                return tensor
            pad_size = multiple - (bsz % multiple)
            padding_tensor = tensor[-1].unsqueeze(0).expand(pad_size, *tensor.size()[1:]) # repeat last element
            padded_tensor = torch.cat([tensor, padding_tensor], dim=0).contiguous()
            return padded_tensor

        original_batch_size = input_ids.size(0)
        padded_input_ids = pad_dim_0_to_multiple_of(input_ids, self.train_pp_schedule.pp_mesh.size())
        padded_labels = pad_dim_0_to_multiple_of(labels, self.train_pp_schedule.pp_mesh.size())
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.size(0) == original_batch_size:
                kwargs[k] = pad_dim_0_to_multiple_of(v, self.train_pp_schedule.pp_mesh.size())

        with self._model_forward_context():
            schedule_outputs = self.train_pp_schedule.step(
                padded_input_ids,
                target=padded_labels,
                forward_only=True, # <<<--- eval mode
                # kwargs from now ---
                # loss_div_factor=batch_num_tokens_for_loss,
                labels=padded_labels,
                **kwargs,
            )

        # remove padding results from outputs
        for stage_idx in range(len(schedule_outputs)):
            schedule_outputs[stage_idx] = schedule_outputs[stage_idx][: original_batch_size]

        return schedule_outputs
    
    def _point_to_low_precision_params(self):
        for model in self.model_parts:
            for name, param in model.named_parameters():
                if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                    continue
                param.data = param._mp_param

    def _point_to_full_precision_params(self):
        for model in self.model_parts:
            for name, param in model.named_parameters():
                if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                    continue
                param.data = param._mp_param

    def _copy_full_precision_to_low_precision_params(self):
        from torch.distributed.utils import _alloc_storage
         
        for model in self.model_parts:
            for name, param in model.named_parameters():
                if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                    continue
                _alloc_storage(param._mp_param, param.size())
                with torch.no_grad():
                    # assert param.data == param._fp_param, "param.data should point to param._fp_param before copy"
                    assert param.data.dtype == torch.float32, "param.data should be float32 before copy"
                    param._mp_param.copy_(param.data)


    def run_pipeline(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_num_tokens_for_loss: Union[int, float],
        **kwargs,
    ) -> List[List[Optional[LMOutputWithLoss]]]:
        """
        Run the pipeline, returning the losses captured.
        Returns:
            ce_batch_loss: Cross-entropy loss for the batch.
            z_batch_loss: Z loss for the batch (if applicable).
        """
        with self._model_forward_context():
            schedule_outputs = self.train_pp_schedule.step(
                input_ids,
                target=labels,
                loss_div_factor=batch_num_tokens_for_loss,
                labels=labels,
                **kwargs,
            )
        return schedule_outputs



    def optim_step(self):
        from olmo_core.optim.moe_optimizer import MoEFusedV2Optimizer
        optim = self._require_optimizer()

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print("-------optim_step start--------")

        if self.reduce_scatter_grads:
            pass
        else:
            pass
            # raise RuntimeError("Deprecated code path, only reduce-scatter grads is supported now")
            
        

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        # Step optimizer.
        optim.step()

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print("-------optim step done--------")

        # self._copy_full_precision_to_low_precision_params()

        total_grad_norm = optim.latest_grad_norm

        if total_grad_norm is not None:
            self.trainer.record_metric(
                "total grad norm", total_grad_norm, reduce_type=None, namespace="optim"
            )

        if isinstance(optim, MoEFusedV2Optimizer):
            self.record_metric("step skipped", optim.step_skipped, namespace="optim")

        for model in self.model_parts:
            model.post_optim_step()
        

    def zero_grads(self):
        self._require_optimizer()
        # Contract: optimizer consumes model grads but does not clear model grad buffers.
        # Keep grad-buffer lifecycle in the model wrapper so bucket views stay stable.
        for m in self.model_parts:
            m.zero_grad(set_to_none=False)

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
            elif isinstance(self.model_parts[0], MultiGroupDistributedDataParallel):
                if not is_last_mb and self.dp_config.only_allreduce_last_microbatch:
                    stack.enter_context(self.model_parts[0].no_sync())
            elif self.dp_config.name == DataParallelType.ddp: # temp fix
                if (
                    self.reduce_scatter_grads # if use RS, always no sync
                    or  # if not, fall back to all-reduce
                    ( 
                        # if specified, only AR at the last microbatch
                        not is_last_mb and self.dp_config.only_allreduce_last_microbatch
                    )
                ):
                    stack.enter_context(self.ddp_no_sync(self.model_parts)) # only DDP has no_sync(), can only call set_requires_gradient_sync()
            yield


    @contextlib.contextmanager
    def ddp_no_sync(self, model_parts: List[MoEFusedV2Transformer]):
        for module in model_parts:
            assert callable(getattr(module, "set_requires_gradient_sync", None)), \
        f"{type(module).__name__} must implement set_requires_gradient_sync(flag: bool). " \
        "This is automatically managed if the model is returned by torch.distributed._composable.replicate"
            # non-EP modules
            module.set_requires_gradient_sync(False) # type: ignore

            # EP managed modules
            ep_modules = [m for m in module.modules() if getattr(m, '_ep_sharded', False) ]
            for m in ep_modules:
                m.set_requires_gradient_sync(False) # type: ignore

        try:
            yield
        finally:
            for module in model_parts:
                module.set_requires_gradient_sync(True) # type: ignore
                ep_modules = [m for m in module.modules() if getattr(m, '_ep_sharded', False) ]
                for m in ep_modules:
                    m.set_requires_gradient_sync(True) # type: ignore

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



    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        raise RuntimeError("Deprecated. Use optimizer's grad clipping instead.")


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


    def parallelize_and_init_model(
        self,
        model: MoEFusedV2Transformer, # the full model before sharding, the same on all models
        *,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
        eval_only: bool = False,
    ) -> List["MoEFusedV2Transformer"]:
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer
        assert isinstance(model, MoEFusedV2Transformer), "model must be an instance of Transformer"

        if tp_config is not None:
            raise NotImplementedError("TP not supported yet")
        if cp_config is not None:
            raise NotImplementedError("CP not supported yet")


        if pp_config is not None:

            assert self.world_mesh['dense'] is not None, "Dense mesh must be built before applying pipeline parallelism"
            self.pp_mesh = self.world_mesh['dense']['pp']
            self.pp_group = self.pp_mesh.get_group()
            self.pp_group_rank = get_rank(self.pp_group)
            self.pp_group_size = get_world_size(self.pp_group)
            self.pp_prev_rank = (self.pp_group_rank - 1) % self.pp_group_size
            self.pp_next_rank = (self.pp_group_rank + 1) % self.pp_group_size
            self.pp_final_stage_rank = pp_config.final_stage_rank()

            # Split model into pipeline stages.
            model.purge_cuda_events() # set event to None so that can be deepcopied
            
            stages_and_model_parts = pp_config.split_model(
                model, pp_mesh=self.pp_mesh, device=self.device, 
                use_ddp=self.world_mesh['dense']['dp'].size() > 1
            )
            stages = stages_and_model_parts[0]

            model_parts: List[MoEFusedV2Transformer] = cast(List[MoEFusedV2Transformer], stages_and_model_parts[1])

            for model_part in model_parts:
                assert isinstance(model_part, MoEFusedV2Transformer)
                model_part.install_cuda_events()

            self._pp_stages = stages
            log.info(
                f"Applied pipeline parallelism to the model with {get_device_mesh_info(self.pp_mesh)}"
            )
            
            # TODO: chunk layers into stages
            assert self.world_mesh is not None, "World mesh must be built before applying expert parallelism"
            assert self.world_mesh['dense'] is not None, "Dense mesh must be built before applying expert parallelism"

            for m in model_parts:
                m.apply_pp(self.pp_mesh)

        else:
            model_parts: List[MoEFusedV2Transformer] = [model] # no PP, single part

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
            assert self.world_mesh is not None, "World mesh must be built before applying expert parallelism"
            assert self.world_mesh["moe"] is not None, "MoE mesh must be built before applying expert parallelism"
            assert self.world_mesh["dense"] is not None, "Dense mesh must be built before applying expert parallelism"
            ep_mesh = self.world_mesh["moe"]
            dp_mesh = self.world_mesh["dense"]["dp"]
            ep_mp_group_override: Optional[ProcessGroup] = None

            if dist.get_backend() == "nccl":
                ep_mp_dim = ep_mesh.mesh_dim_names.index(MeshDimName.ep_mp)
                ep_rank_grid = ep_mesh.mesh
                if ep_mp_dim != ep_rank_grid.ndim - 1:
                    ep_rank_grid = torch.movedim(ep_rank_grid, ep_mp_dim, -1).contiguous()
                ep_mp_rank_groups = ep_rank_grid.view(-1, ep_rank_grid.shape[-1]).tolist()
                nccl_opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
                ep_mp_group, ep_mp_groups = dist.new_subgroups_by_enumeration(
                    ranks_per_subgroup_list=ep_mp_rank_groups,
                    backend="nccl",
                    pg_options=nccl_opts,
                    group_desc="ep_mp_high_priority",
                )
                if ep_mp_group == dist.GroupMember.NON_GROUP_MEMBER:
                    raise RuntimeError("Current rank is not in any EP-MP high-priority subgroup")
                ep_mp_group_override = cast(ProcessGroup, ep_mp_group)
                self.ep_mp_high_priority_group = ep_mp_group_override
                self.ep_mp_high_priority_groups = ep_mp_groups
                log.info("Created high-priority NCCL EP-MP process groups")
            else:
                log.warning(
                    "Skipping EP-MP high-priority group creation because backend is '%s'",
                    dist.get_backend(),
                )

            ddp_model_parts = []
            for m in model_parts:
                if not m.is_moe:
                    raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
                cast(MoEFusedV2Transformer, m).apply_ep(
                    dp_mesh=dp_mesh,
                    ep_mesh=ep_mesh,
                    ep_mp_group=ep_mp_group_override,
                    param_dtype=None,
                    compile_enabled=compile_model
                )

            log.info(
                "Applied expert parallelism to the model with %s",
                get_device_mesh_info(ep_mesh),
            )
        else:
            # Pure DP (no EP)
            pass
            assert self.world_mesh is not None, "World mesh must be built before applying expert parallelism"
            assert self.world_mesh["dense"] is not None, "Dense mesh must be built before applying expert parallelism"
            # param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            dp_mesh = self.world_mesh["dense"]["dp"]
            ep_mesh = None
            # for m in model_parts:
            #     cast(MoEFusedV2Transformer, m).apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            # log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")

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

        # ac_budget = 0.05
        # torch._functorch.config.activation_memory_budget = ac_budget

        self.init_model_weights(
            model_parts=model_parts,
            max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )

        # now wrap with DDP (requires initialized params)
        if eval_only:
            # apply_dp() currently performs the bf16 cast used for MoE forward kernels.
            # Keep that cast in eval-only mode even though we intentionally skip DDP wrapping.
            self._cast_to_fwd_bwd_precision(model_parts)
            log.info("Skipping DDP wrapping because eval_only=True")
            return model_parts

        ddp_model_parts = []
        for idx, m in enumerate(model_parts):
            ddp_m = m.apply_dp(
                dp_mesh=dp_mesh,
                ep_mesh=ep_mesh,
                accumulate_grads_in_fp32=dp_config.accumulate_grads_in_fp32,
                reduce_grads_in_fp32=dp_config.reduce_grads_in_fp32,
            )
            ddp_model_parts.append(ddp_m)

            if pp_config is not None:
                # update stage reference to point to the ddp wrapped model part
                assert self._pp_stages is not None
                assert self._pp_stages[idx].submod is m
                self._pp_stages[idx].submod = ddp_m

        return ddp_model_parts


    def memory_usage_estimation(self):
        pass

    def init_model_weights(
        self,
        model_parts,
        max_sequence_length: int,
        rank_microbatch_size: int,
    ):
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer
        
        # Materialize and init parameters.
        log.info("Initializing model weights...")
        for model_part_idx, m in enumerate(model_parts):
            m = cast(MoEFusedV2Transformer, m)
            m.init_weights(
                max_seq_len=max_sequence_length,
                max_local_microbatch_size=rank_microbatch_size,
                device=self.device,
                world_mesh=self.world_mesh, # only PP mesh is used, should be fine
                model_part_idx=model_part_idx
            )
            # for n, p in m.named_parameters():
            #     print(f'{n} {p.shape}: mean={p.data.mean().item()}, std={p.data.std().item()}')

        return

    def compile_model(self):
        if torch.cuda.is_available():
            for m in self.model_parts:
                m.apply_compile()
            log.info("Applied torch.compile() to the model")
        else:
            log.warning("Skipping model compilation since CUDA is not available")


    def reduce_send_recv(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # assert self.pp_enabled and self._pp_config is not None

        # if self.pp_group_rank == self.pp_final_stage_rank:
        #     assert x is not None
        #     # DP reduce (mean)
        #     x = x / self._reduce_divide_factor
        #     dist.all_reduce(x, group=self.dp_process_group)
        #     x = x / self.dp_world_size * self._reduce_divide_factor
        # else:
        #     assert x is None
        #     x = torch.empty([], device=self.device, dtype=torch.float32)

        # # Broadcast from final stage to all PP ranks.
        # handle = dist.broadcast(x, src=get_global_rank(self.pp_final_stage_rank, group=self.pp_group), group=self.pp_group, async_op=True)
        # handle.wait()
        # # print(f'{get_rank()} (pp group rank {self.pp_group_rank}) got {x.item()}')
        # return x


        assert self.pp_enabled and self._pp_config is not None, "reduce_send_recv is only valid when PP is enabled"
        if self.pp_group_rank == self.pp_final_stage_rank:
            assert x is not None
            # Reduce across DP process group.
            dist.all_reduce(x, group=self.dp_process_group)
            x.div_(self.dp_world_size)
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

        send_ops: List[dist.P2POp] = []
        recv_ops: List[dist.P2POp] = []
        if src_rank is not None:
            # print(
            #     f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) receiving from rank "
            #     f"{get_global_rank(src_rank, group=self.pp_group)} (pp group rank {src_rank})"
            # )
            recv_ops.append(dist.P2POp(dist.irecv, x, group=self.pp_group, group_peer=src_rank))
        if dst_rank is not None:
            # print(
            #     f"Rank {get_rank()} (pp group rank {self.pp_group_rank}) sending to rank "
            #     f"{get_global_rank(dst_rank, group=self.pp_group)} (pp group rank {dst_rank})"
            # )
            send_ops.append(dist.P2POp(dist.isend, x, group=self.pp_group, group_peer=dst_rank))
        
        if len(recv_ops) > 0:
            recv_reqs = dist.batch_isend_irecv(recv_ops)
            for req in recv_reqs:
                req.wait()

        if len(send_ops) > 0:
            send_reqs = dist.batch_isend_irecv(send_ops)
            for req in send_reqs:
                req.wait()

        # print(f'{get_rank()} (pp group rank {self.pp_group_rank}) got {x.item()}')
        return x
