import contextlib
import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, Generator, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from typing import List, Optional, Tuple
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed import ProcessGroup
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import (
    get_local_tensor,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, log_once, move_to_device
from typing import List, Optional, TypeVar, cast
from olmo_core.nn.transformer import MoETransformer, Transformer
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
import nvtx

log = logging.getLogger(__name__)

M = TypeVar("M", Transformer, List[Transformer])

class MoEV2TransformerTrainModule(TrainModule):
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
    ):
        super().__init__()

        ######################### Validate arguments. [BEGIN] #########################
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Optional[Dict[str, Optional[DeviceMesh]]] = None
        self.pp_group = None
        self.dp_group = None
        self.tp_group = None
        self.ep_dp_group = None
        self.ep_mp_group = None


        if tp_config is not None:
            assert tp_config.degree > 1, "Tensor parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Tensor parallelism is not implemented")
        if pp_config is not None:
            assert pp_config.degree > 1, "Pipeline parallelism requires a degree > 1, otherwise use None"
            # raise NotImplementedError("Pipeline parallelism is not implemented")
        if cp_config is not None:
            assert cp_config.degree > 1, "Context parallelism requires a degree > 1, otherwise use None"
            raise NotImplementedError("Context parallelism is not implemented")
        if ac_config is not None:
            raise OLMoConfigurationError("In MoEV2TransformerTrainModule, activation checkpointing is controlled by the model, not the train module.")
        if float8_config is not None:
            raise NotImplementedError("Float8 quantization is not implemented")

        assert dp_config is not None, "Data parallel config is required for MoEV2TransformerTrainModule"
        assert dp_config.name == "ddp", "Data parallel config must be 'ddp'"

        ########################## Validate arguments. [END] ##########################

        if is_distributed():
            self.build_world_mesh(
                dp=dp_config, tp=tp_config, cp=cp_config, ep=ep_config, pp=pp_config, device_type=self.device.type
            )
            log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")
        else:
            raise OLMoConfigurationError(
                "Training parallelism is required for MoEV2TransformerTrainModule"
            )

        # Parallelize model.
        self.model = self.parallelize_model(
            model,
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
            pp_config=pp_config
        )

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config

        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        
        self.max_grad_norm = max_grad_norm
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
        # dense_filter = lambda x: not getattr(x, "_ep_sharded", False)
        # moe_filter = lambda x:  getattr(x, "_ep_sharded", False)
        self.optim: Optimizer = optim.build(self.model, self, strict=True)
        # self.dense_optim: Optimizer = optim.build(self.model, self, strict=True, 
        #                                         #   reduce_group=self.dp_group, 
        #                                           param_filter=dense_filter)
        # self.moe_optim: Optimizer = optim.build(self.model, self, strict=True, 
        #                                         # reduce_group=self.ep_dp_group, 
        #                                         param_filter=moe_filter)


    def build_world_mesh(
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

        if self.world_mesh is not None:
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
        if pp is not None:
            names.append(MeshDimName.pp)
            dims.append(pp.degree)

        # Then data parallel.
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)



        self.dense_mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
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
            self.pp_group = self.dense_mesh['pp'].get_group() # 
            
        self.dp_group = self.dense_mesh['dp'].get_group()
        if self.moe_mesh is not None:
            self.ep_dp_group = self.moe_mesh['ep_dp'].get_group()
            self.ep_mp_group = self.moe_mesh['ep_mp'].get_group()

        self.world_mesh = {
            "dense": self.dense_mesh,
            "moe": self.moe_mesh
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
    def dp_config(self) -> Optional[TransformerDataParallelConfig]:
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

        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if load_optim:
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

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                with nvtx.annotate(f"fwd_mb{micro_batch_idx}", color='blue'):
                    
                    input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                    # Run forward pass, get losses.
                    _, loss, ce_loss, z_loss = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )

                    # Update total batch CE and Z loss.
                    ce_batch_loss += get_local_tensor(ce_loss.detach())
                    del ce_loss
                    if z_batch_loss is not None:
                        assert z_loss is not None
                        z_batch_loss += get_local_tensor(z_loss.detach())
                        del z_loss

                with nvtx.annotate(f"bwd_mb{micro_batch_idx}", color='red'):
                    # Run backward pass.
                    loss.backward()

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
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
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
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

        self.model.eval()

        with self._eval_batch_context():
            return self.model_forward(
                input_ids,
                labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="none",
                **model_kwargs,
            )

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

        # calculate per layer grad norm
        per_layer_norms = []
        for layer_idx, layer in enumerate(self.model.blocks.values()):
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
                f"clipped grad norm (layer {layer_idx})", per_layer_norms[layer_idx], reduce_type=None, namespace="optim"
            )
            
            del layer_grads
            
        del per_layer_norms

        # embedding layer grad norm
        embedding_grads = [p.grad for p in self.model.embeddings.parameters() if p.grad is not None]
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
        lm_head_grads = [p.grad for p in self.model.lm_head.parameters() if p.grad is not None]
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
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        self.model.post_optim_step()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            return self.model(input_ids, labels=labels, **kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.model.num_flops_per_token(seq_len)

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, FSDPModule):
                assert self.dp_config is not None
                # On the last backward FSDP waits on pending gradient reduction and clears internal data
                # data structures for backward prefetching.
                self.model.set_is_last_backward(is_last_mb)
                # For HSDP we can delay the gradients all-reduce until the final micro-batch.
                if self.dp_config.name == DataParallelType.hsdp:
                    self.model.set_requires_all_reduce(is_last_mb)
            elif isinstance(self.model, DDP):  # BUG: always false.  --> the model is returned by replicate(), so it's a torch.distributed._composable.replicate.DDP not torch.nn.parallel.DistributedDataParallel
                # see: debug message below
                # print(type(self.model).__mro__)
                # (<class 'torch.distributed._composable.replicate.DDPMoETransformer'>, <class 'torch.distributed._composable.replicate.DDP'>, <class 'olmo_core.nn.transformer.model.MoETransformer'>, <class 'olmo_core.nn.transformer.model.Transformer'>, <class 'torch.nn.modules.module.Module'>, <class 'object'>)
                
                # For DDP, only sync gradients on the final micro-batch.
                if not is_last_mb:
                    stack.enter_context(self.model.no_sync())
            elif self.dp_config.name == DataParallelType.ddp: # temp fix
                if not is_last_mb and self.dp_config.only_allreduce_last_microbatch:
                    stack.enter_context(self.ddp_no_sync(self.model)) # only DDP has no_sync(), can only call set_requires_gradient_sync()
            yield


    @contextlib.contextmanager
    def ddp_no_sync(self, module: torch.nn.Module):
        module.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            module.set_requires_gradient_sync(True)

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
        state_dict: Dict[str, Any] = {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
        }
        if optim:
            state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            )
        return state_dict

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

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels, batch


    def parallelize_model(
        self,
        model: M, # the full model before sharding, the same on all models
        *,
        max_sequence_length: int,
        rank_microbatch_size: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        pp_config: Optional[TransformerPipelineParallelConfig] = None,
    ) -> M:
        from olmo_core.nn.moe.v2.model import MoEFusedV2Transformer
        model_parts: List[Transformer] = [model] if isinstance(model, Transformer) else model

        pp_mesh: Optional[DeviceMesh] = None
        if pp_config is not None:
            # TODO: chunk layers into stages
            assert self.world_mesh is not None, "World mesh must be built before applying expert parallelism"
            assert self.world_mesh['dense'] is not None, "Dense mesh must be built before applying expert parallelism"
            pp_mesh = self.world_mesh['dense']['pp']
            for m in model_parts:
                m.apply_pp(pp_mesh)

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
            for m in model_parts:
                if not m.is_moe:
                    raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
                cast(MoEFusedV2Transformer, m).apply_epdp(
                    dp_mesh=dp_mesh,
                    ep_mesh=ep_mesh,
                    param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None,
                    compile_enabled=compile_model
                )
            log.info(f"Applied expert parallelism + DDP to the model with {get_device_mesh_info(ep_mesh)}")
        else:
            # Pure DP (no EP)
            assert self.world_mesh is not None, "World mesh must be built before applying expert parallelism"
            assert self.world_mesh["dense"] is not None, "Dense mesh must be built before applying expert parallelism"
            param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
            dp_mesh = self.world_mesh["dense"]["dp"]
            for m in model_parts:
                m.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")


        # Maybe compile.
        if compile_model:
            if torch.cuda.is_available():
                for m in model_parts:
                    m.apply_compile()
                log.info("Applied torch.compile() to the model")
            else:
                log.warning("Skipping model compilation since CUDA is not available")




        # Materialize and init parameters.
        log.info("Initializing model weights...")
        for model_part_idx, m in enumerate(model_parts):
            m.init_weights(
                max_seq_len=max_sequence_length,
                max_local_microbatch_size=rank_microbatch_size,
                device=self.device,
                world_mesh=self.world_mesh['moe'], # only PP mesh is used, should be fine
                model_part_idx=model_part_idx
            )

        return model

    def memory_usage_estimation(self):
        pass