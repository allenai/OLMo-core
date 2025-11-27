from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    Callable,
    overload,
    cast,
)
from ..config import DType
from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer
import logging
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import NamedTuple
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import (
    hook_with_zero_step,
    hook_with_zero_step_interleaved,
)
from ..train.train_module import TrainModule
from olmo_core.utils import get_default_device, move_to_device
from collections import OrderedDict
from fnmatch import fnmatch

from .adamw import foreach_adamw_step, adamw_step
from typing import Any, Dict, Iterable, List, Optional, Union
from torch.optim.optimizer import Optimizer
from typing import cast
from ..config import Config
from .config import OptimGroupOverride, INITIAL_LR_FIELD, LR_FIELD
from ..exceptions import OLMoConfigurationError
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard, DTensor, distribute_tensor
import nvtx

log = logging.getLogger(__name__)

# Opt = TypeVar("Opt", bound=torch.optim.Optimizer)

### DEBUG PRINT ###
def _str_paramt(paramt):
    rets = ''
    total_numel = 0
    for i, pgrp in enumerate(paramt):
        rets += f'{i}: {pgrp["__pg_tag__"]}\n'
        rets += f'    num params: {len(pgrp["params"])}\n'
        rets += f'    num ele: {sum(p.numel() for p in pgrp["params"]):,}\n'
        total_numel += sum(p.numel() for p in pgrp["params"])
    rets += f'Total num ele: {total_numel:,}\n'
    return rets

@dataclass
class MoEFusedV2OptimizerConfig(Config): 

    group_overrides: Optional[List[OptimGroupOverride]] = None
    """
    Use this to pull out groups parameters into a separate param groups with their own options.
    """

    compile: bool = False
    """
    Compile the optimizer step.

    .. warning::
        Optimizer step compilation is still in beta and may not work with some optimizers.
        You could also see unexpected behavior and very poor performance when turning this feature
        on in the middle of a run that was previously trained without compiling the optimizer
        due to the LR being restored to a float instead of a tensor.
    """

    fixed_fields: Tuple[str, ...] = (INITIAL_LR_FIELD,)
    """
    These are fields that should not be overridden by the value in a checkpoint after
    loading optimizer state.
    """


    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    dtype: Optional[DType] = None

    # foreach: bool = True
    """
    Whether to use multi-tensor (*foreach*) kernels for the AdamW update.
    Faster than the non-foreach version.
    """

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation of the loss.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss to skip a step.
    """

    max_grad_norm: float = 1.0

    @property
    def device(self) -> torch.device:
        return get_default_device()

    def _expand_param_globs(
        self,
        go: OptimGroupOverride,
        all_params: Dict[str, Any],
        frozen_param_names: Set[str],
        g_idx: int,
        strict: bool = True,
    ) -> OptimGroupOverride:
        param_names: List[str] = []
        for pattern in go.params:
            matches = 0
            for name in list(all_params.keys()):
                if fnmatch(name, pattern):
                    param_names.append(name)
                    matches += 1

            if matches == 0:
                for name in frozen_param_names:
                    if fnmatch(name, pattern):
                        log.warning(
                            f"optim group {g_idx} override pattern '{pattern}' matches a frozen parameter and will be ignored"
                        )
                        break
                else:
                    msg = f"optim group {g_idx} override pattern '{pattern}' does not match any parameters"
                    if strict:
                        raise OLMoConfigurationError(msg)
                    else:
                        log.warning(msg)

        return OptimGroupOverride(param_names, go.opts.copy())

    def build_groups(
        self, model_parts: List[nn.Module], strict: bool = True, param_filter=None
    ) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: Dict[str, torch.Tensor] = OrderedDict()
        frozen_params: set = set()
        for part in model_parts:
            for n, p in part.named_parameters():
                if p.requires_grad:
                    if param_filter is None: # No filter applied
                        all_params[n] = p
                    else:
                        # Apply the parameter filter
                        if param_filter(p):
                            all_params[n] = p

                else:
                    frozen_params.add(n)

        group_overrides = [
            self._expand_param_globs(go, all_params, frozen_params, g_idx, strict=strict)
            for g_idx, go in enumerate(self.group_overrides or [])
        ]

        # Treat no overrides as its own override group
        overriden_param_names = {name for go in group_overrides for name in go.params}
        default_override = OptimGroupOverride(
            [name for name in all_params.keys() if name not in overriden_param_names], {}
        )
        # group_overrides.append(default_override)
        group_overrides.insert(0, default_override) # to ensure default is first

        param_groups = []
        for go in group_overrides:
            if len(go.params) > 0:
                param_groups.append(
                    {
                        "named_params": {param_name: all_params[param_name] for param_name in go.params},
                        **go.opts, # 
                    }
                )

        return param_groups



    @classmethod
    def optimizer(cls):
        return MoEFusedV2Optimizer


    def _collect_ep_param_ids(self, model_parts: List[nn.Module]) -> Set[int]:
        """
        Collect ids() of parameters that belong to modules marked as EP-managed
        (i.e., modules having attribute `_ep_sharded` set to True).
        """
        ep_param_ids: Set[int] = set()
        for part in model_parts:
            for m in part.modules():
                if getattr(m, "_ep_sharded", False):
                    for _, p in m.named_parameters(recurse=True):
                        ep_param_ids.add(id(p))
        return ep_param_ids

    def build(self, model_parts: List, train_module: TrainModule, strict: bool = True, param_filter=None) -> "MoEFusedV2Optimizer":
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        from ..train.train_module.transformer.moe_train_module import MoEV2TransformerTrainModule
        from ..nn.moe.v2.model import MoEFusedV2Transformer
        model_parts = cast(List[MoEFusedV2Transformer], model_parts)
        train_module = cast(MoEV2TransformerTrainModule, train_module)

        # not used: train_module (was); now used to pass process groups
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")

        # Stable parameter order (by name) for each partition, used by all ranks for packing/broadcast.
        ep_param_ids = self._collect_ep_param_ids(model_parts)

        # Build param groups for the two PGs by filtering.
        dp_groups = self.build_groups(model_parts, strict=strict, param_filter=lambda p: id(p) not in ep_param_ids)
        for g in dp_groups:
            g["pg"] = 'dp' # type: ignore

        ep_groups = self.build_groups(model_parts, strict=strict, param_filter=lambda p: id(p) in ep_param_ids)
        for g in ep_groups:
            g["pg"] = 'ep_dp' # type: ignore

        # Concatenate, ensuring the "default" groups remain first in each partition (already ensured by build_groups()).
        all_groups: List[Dict[str, Any]] = list(dp_groups) + list(ep_groups) # type: ignore

        has_grad_accum_fp32_buffer = [part.has_grad_accum_fp32_buffer for part in model_parts]
        # shuold all have the same value
        if not all(x == has_grad_accum_fp32_buffer[0] for x in has_grad_accum_fp32_buffer):
            raise ValueError("Inconsistent `has_grad_accum_fp32_buffer` among model parts")
        
        has_grad_accum_fp32_buffer = has_grad_accum_fp32_buffer[0]

        optim = self.optimizer()(
            all_groups,
            dp_group=getattr(train_module, "dp_group", None),
            ep_dp_group=getattr(train_module, "ep_dp_group", None),
            world_mesh=train_module.world_mesh,
            device=train_module.device,
            model_has_grad_accum_fp32_buffer=has_grad_accum_fp32_buffer,
            **kwargs,
        )

        # Set 'lr' and 'initial_lr' in each group if needed.
        fixed_fields_per_group: List[Dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: Optional[float] = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            elif hasattr(self, LR_FIELD):
                lr = getattr(self, LR_FIELD)

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "named_params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['named_params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx}, {len(group['named_params'])} parameter(s)")

        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        # Register hook to reset fixed fields after loading a checkpoint.
        # def reset_fixed_fields(opt: torch.optim.Optimizer):
        #     for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
        #         group.update(fixed_fields)

        # optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return optim


def assign_full_tensor_to_dtensor(dst: DTensor, src: torch.Tensor) -> None:
    assert dst.shape == src.shape  # global shape

    src_dt = distribute_tensor(src, dst.device_mesh, placements=dst.placements)
    dst.copy_(src_dt)

class MoEFusedV2Optimizer:

    def __init__(
        self,
        param_groups: Iterable[Dict[str, Any]],
        world_mesh: Dict[str, Optional[DeviceMesh]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        max_grad_norm: float = 1.0,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        device: Optional[torch.device] = None,
        model_has_grad_accum_fp32_buffer: bool = False, # whether the optimizer should expect the model to have fp32 grad accum buffers
        # foreach: bool = False,
        # --- new args for sharding across multiple PGs ---
        dp_group: Optional[ProcessGroup] = None,
        ep_dp_group: Optional[ProcessGroup] = None,
        broadcast_bucket_mb: int = 32,
        do_not_shard_tensor_smaller_than: int = 1024,
        use_distributed: bool = False,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.model_has_grad_accum_fp32_buffer = model_has_grad_accum_fp32_buffer
        self.use_distributed = use_distributed

        def _add_defaults_to_param_group(pg: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in defaults.items():
                pg.setdefault(k, v)
            return pg
        # add defaults to each param group
        param_groups = [_add_defaults_to_param_group(pg) for pg in param_groups]

        # for print info
        self._model_param_sz = 0
        for param_group in param_groups:
            self._model_param_sz += sum(p.numel() * p.element_size() for (n, p) in param_group['named_params'].items())

        # ---- Sharding context (DP and EP-DP) ----
        self._dp_group: Optional[ProcessGroup] = dp_group
        self._ep_dp_group: Optional[ProcessGroup] = ep_dp_group

        self._broadcast_bucket_bytes: int = int(broadcast_bucket_mb * 1024 * 1024)

        assert world_mesh['dense'] is not None, "DP mesh must be provided"

        self.dense_mesh: DeviceMesh = world_mesh['dense'] # ('pp', 'dp')
        self.moe_mesh: Optional[DeviceMesh] = world_mesh['moe'] # ('pp', 'ep_dp', 'ep_mp')

        self.dp_mesh = self.dense_mesh['dp']
        self.ep_dp_mesh = self.moe_mesh['ep_dp'] if self.moe_mesh else None
        self.ep_mp_mesh = self.moe_mesh['ep_mp'] if self.moe_mesh else None

        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = device
        self.max_grad_norm = max_grad_norm
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        # self.foreach = foreach
        self._step_skipped: Optional[torch.Tensor] = None
        self.do_not_shard_tensor_smaller_than = do_not_shard_tensor_smaller_than
        self._use_reduce_scatter_grads = True
        self.main_grad: Dict[str, DTensor] = {} # trasient storage for main grads in step()

        # check
        device = None
        for param_group in param_groups:
            for i, (name, param) in enumerate(param_group['named_params'].items()):
                if not param.requires_grad:
                    continue
                if device is None:
                    device = param.device
                else:
                    assert device == param.device, "Inconsistent device found"
                # float16 params:
                assert param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor'], 'Only support 16 bit params. Received {}'.format(param.type())




        self.states: Dict[str, DTensor] = OrderedDict()


        for param_group in param_groups:
            # configure the device mesh to shard the group
            device_mesh = self._get_dp_device_mesh_for_tag(param_group['pg'])
            assert device_mesh is not None, f"Device mesh for pg tag {param_group['pg']} is None"

            # wrap each param with DTensor
            for name, param in param_group['named_params'].items():
                old_shape = param.shape
                # flat in fp32
                num_elements = param.numel()

                # main param
                main_param = torch.zeros(num_elements, dtype=torch.float32, device=device) 
                main_param = self._distribute_tensor(main_param, device_mesh)
                self.states[f'{name}.main'] = main_param

                # exp avg
                exp_avg = torch.zeros(num_elements, dtype=torch.float32, device=device)
                exp_avg = self._distribute_tensor(exp_avg, device_mesh)
                self.states[f'{name}.exp_avg'] = exp_avg

                # exp avg sq
                exp_avg_sq = torch.zeros(num_elements, dtype=torch.float32, device=device)
                exp_avg_sq = self._distribute_tensor(exp_avg_sq, device_mesh)
                self.states[f'{name}.exp_avg_sq'] = exp_avg_sq

                # step
                step_tensor = torch.zeros((), dtype=torch.float32, device=device)
                step_tensor = distribute_tensor(
                    step_tensor,
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
                self.states[f'{name}.step'] = step_tensor

        # copy model params to main params
        for param_group in param_groups:
            for name, param in param_group['named_params'].items():
                main_param = self.states[f'{name}.main']
                
                assign_full_tensor_to_dtensor(dst=main_param, src=param.data.float().reshape(-1))

        self.param_groups = param_groups

        self._check_model_param_main_param_the_same()

        self.print_memory_summary()

        return

    def print_memory_summary(self):
        total_params = 0
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                total_params += param.numel()
        print(f'[MoEFusedV2Optimizer] Total model params: {total_params:,}')

        # main
        def count_numel(tag: str):
            global_state_numel = 0
            local_state_numel = 0
            num_tensors_sharded = 0
            num_tensors_replicated = 0
            sharded_state_numel = 0
            replicated_state_numel = 0
            for state_key, state_val in self.states.items():
                if state_key.endswith(f'.{tag}'):
                    global_state_numel += state_val.numel()
                    local_state_numel += state_val.to_local().numel()
                    if any(isinstance(p, Shard) for p in state_val.placements):
                        num_tensors_sharded += 1
                        sharded_state_numel += state_val.to_local().numel()
                    else:
                        num_tensors_replicated += 1
                        replicated_state_numel += state_val.to_local().numel()
            return global_state_numel, local_state_numel, num_tensors_sharded, num_tensors_replicated, sharded_state_numel, replicated_state_numel

        def to_str_N_B_GB(num):
            return f'{num:,} | {num/1000**3:.4} Billion | {num * 4 /1024**3:.4} GB'

        def info_str(tag: str, stat: Tuple[int, int, int, int, int, int]):
            info_str = ''
            info_str += f'[MoEFusedV2Optimizer] {tag} - Global params: {to_str_N_B_GB(stat[0])}, Local params: {to_str_N_B_GB(stat[1])}\n'
            info_str += f'    Sharded tensors: {stat[2]}, total local sharded params: {to_str_N_B_GB(stat[4])}\n'
            info_str += f'    Replicated tensors: {stat[3]}, total local replicated params: {to_str_N_B_GB(stat[5])}\n'
            return info_str

        main_stat = count_numel('main')
        exp_avg_stat = count_numel('exp_avg')
        exp_avg_sq_stat = count_numel('exp_avg_sq')

        print_str = ''

        print_str += info_str('Main param', main_stat)
        print_str += info_str('Exp avg', exp_avg_stat)
        print_str += info_str('Exp avg sq', exp_avg_sq_stat)

        total_global_optim_gb = (main_stat[0] + exp_avg_stat[0] + exp_avg_sq_stat[0]) * 4 / 1024**3
        total_local_optim_gb = (main_stat[1] + exp_avg_stat[1] + exp_avg_sq_stat[1]) * 4 / 1024**3
        total_model_gb = self._model_param_sz / 1024**3
        print_str += f'[MoEFusedV2Optimizer] Total optimizer states size: {total_global_optim_gb:.4f} GB global, {total_local_optim_gb:.4f} GB local\n'
         
        if self.model_has_grad_accum_fp32_buffer:
            total_model_grad_gb = 2 * total_model_gb # extra fp32 grad buffer
        else:
            total_model_grad_gb = total_model_gb # bf16 grad only
        print_str += f'[MoEFusedV2Optimizer] Model params size (GB): {total_model_gb:.4f} GB, model grads size (GB): {total_model_grad_gb:.4f} GB\n'
        total_static = total_local_optim_gb + total_model_gb + total_model_grad_gb

        print_str += f'[MoEFusedV2Optimizer] Total estimated static memory (GB): {total_static:.4f} GB\n'

        print(print_str)

    def _check_model_param_main_param_the_same(self):
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                main_param = self.states[f'{name}.main']
                # get global tensor from DTensor
                main_param_full = main_param.full_tensor().reshape(-1)
                model_param = param.data.float().reshape(-1)
                if not torch.allclose(model_param, main_param_full, atol=1e-5):
                    raise ValueError(f"{name}: Model param {param} and main param {main_param} are not close")


    def _distribute_tensor(self, tensor, device_mesh):
        num_elements = tensor.numel()
        if self.use_distributed:
            if num_elements >= self.do_not_shard_tensor_smaller_than and num_elements % device_mesh.size(0) == 0:
                # this is distributed optimizer, so each rank holds one shard of the data
                placements=[Shard(0)]
            else:
                # small tensor, do not shard
                placements=[Replicate()]
        else:
            # always no shard
            placements=[Replicate()]

        tensor_dt = distribute_tensor(
            tensor,
            device_mesh=device_mesh,
            placements=placements,
        )

        return tensor_dt

    def offload_optimizer_states(self):
        raise NotImplementedError()
        # Offload optimizer states to CPU to save GPU memory
        

    def reload_optimizer_states_to_device(self,):
        raise NotImplementedError()
        # Reload optimizer states to the given device



    @property
    def device(self) -> torch.device:
        if self._device is None:
            for group in self.param_groups:
                for n, p in group["named_params"].items():
                    if p.numel() > 0:
                        self._device = p.device
                        break
            if self._device is None:
                self._device = get_default_device()
        return self._device
    
    @property
    def latest_loss(self) -> Optional[torch.Tensor]:
        if not self._losses:
            return None
        else:
            return self._losses[-1]

    @latest_loss.setter
    def latest_loss(self, loss: torch.Tensor):
        self._losses.append(loss)
        while len(self._losses) > self.rolling_interval_length + 1:
            self._losses.pop(0)

    @property
    def latest_grad_norm(self) -> Optional[torch.Tensor]:
        if not self._grad_norms:
            return None
        else:
            return self._grad_norms[-1]

    @latest_grad_norm.setter
    def latest_grad_norm(self, grad_norm: torch.Tensor):
        self._grad_norms.append(grad_norm)
        while len(self._grad_norms) > self.rolling_interval_length + 1:
            self._grad_norms.pop(0)

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @overload # make pylance happy
    def step(self, closure: None = ...) -> None:
        ...

    @overload # make pylance happy
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def set_reduce_scatter_grads(self, enabled: bool = True):
        self._use_reduce_scatter_grads = enabled

    def _clip_grad(self) -> torch.Tensor:
        """
        We need to first compute the grad norm for the FULL model.
        The optimizer sees the model that's sharded across PP and EP_MP when initialized. 
        Then the optimizer further shards the model across DP or EP_DP.
        At this point, ranks in the same DP/EP_DP already have the same grads because we've done grad-reduce.

        We need to consider:
        1. PP: compute for each PP rank, then reduce across PP ranks. Apply to all grads.
        2. EP_MP: compute for each EP_MP rank, then reduce across EP_MP ranks. Apply to EP grads.
        3. DP: compute for each DP rank, don't need to reduce across DP ranks, they should be the same already. Apply to DP grads.
        4. Watch out for small tensors that are replicated instead of sharded.

        """

        # separate DP and EP_DP grads
        dp_grads = []
        ep_dp_grads = []

        # debug_grads = {}
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                if not param.requires_grad:
                    continue
                main_grad = self.main_grad[name]
                
                if param_group['pg'] == 'dp':
                    dp_grads.append(main_grad)
                    # debug_grads[name] = main_grad.full_tensor()
                elif param_group['pg'] == 'ep_dp':
                    ep_dp_grads.append(main_grad)
                    # debug_ep_g = main_grad.full_tensor()
                    # debug_ep_g = DTensor.from_local(debug_ep_g, device_mesh=self.ep_mp_mesh, placements=[Shard(0)])
                    # debug_grads[name] = debug_ep_g.full_tensor()

        # ref_total_norm = nn.utils.get_total_norm(list(debug_grads.values()), norm_type=2.0, error_if_nonfinite=False)

        dp_grad_norm = nn.utils.get_total_norm(dp_grads, norm_type=2.0, error_if_nonfinite=False)
        dp_grad_norm = cast(DTensor, dp_grad_norm).full_tensor()

        if self.moe_mesh is not None:
            ep_dp_grad_norm = nn.utils.get_total_norm(ep_dp_grads, norm_type=2.0, error_if_nonfinite=False)
            ep_dp_grad_norm = cast(DTensor, ep_dp_grad_norm).full_tensor()

            # reduce EP_MP
            assert self.ep_mp_mesh is not None
            ep_dp_grad_norm = ep_dp_grad_norm.square()
            dist.all_reduce(ep_dp_grad_norm, op=dist.ReduceOp.SUM, group=self.ep_mp_mesh.get_group())
            ep_dp_grad_norm = ep_dp_grad_norm.sqrt()

            # combine DP and EP_DP grad norms
            total_grad_norm = torch.sqrt(dp_grad_norm.square() + ep_dp_grad_norm.square())
        else:
            assert len(ep_dp_grads) == 0, "No EP_DP grads should exist if no MOE mesh"
            total_grad_norm = dp_grad_norm

        # reduce PP
        assert self.dense_mesh.mesh_dim_names is not None
        if 'pp' in self.dense_mesh.mesh_dim_names:
            total_grad_norm = total_grad_norm.square()
            dist.all_reduce(total_grad_norm, op=dist.ReduceOp.SUM, group=self.dense_mesh['pp'].get_group())
            total_grad_norm = total_grad_norm.sqrt()


        clip_coef = self.max_grad_norm / (total_grad_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0).to(total_grad_norm.device)

        torch._foreach_mul_(dp_grads, clip_coef_clamped)
        if len(ep_dp_grads) > 0:
            torch._foreach_mul_(ep_dp_grads, clip_coef_clamped)


        return total_grad_norm

    @torch.no_grad()
    @nvtx.annotate("MoEFusedV2Optimizer.step")
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        
        dbg_mem_before_cp1 = torch.cuda.memory_allocated()/1024**3
        if getattr(self, "_use_reduce_scatter_grads", True):
            # Precondition: DDP model did not all-reduce grads, grads on dp ranks different now
            # the optimizer has sharded main param + states in fp32
            # now call reduce scatter to collect averaged grads from dp ranks
            # directly into the owned main param views
            self._reduce_scatter_model_grads()
        else:
            # Precondition: DDP model called all-reduce grads, bf16 model grads on dp ranks are the same
            self._copy_model_grads_to_main_grads()

        total_grad_norm = self._clip_grad()
        self.latest_grad_norm = total_grad_norm
        dbg_mem_before_step = torch.cuda.memory_allocated()/1024**3
        self._step_foreach(closure)
        dbg_mem_after_step = torch.cuda.memory_allocated()/1024**3
        self._dealloc_main_grad()
        dbg_mem_before_cp2 = torch.cuda.memory_allocated()/1024**3

        self._copy_main_params_to_model_params()

        return None

    def _get_process_group_for_tag(self, tag: str):
        if tag == "dp":
            return self._dp_group
        elif tag == "ep_dp":
            return self._ep_dp_group
        else:
            raise RuntimeError(f"Unknown pg tag: {tag}")
        
    def _get_dp_device_mesh_for_tag(self, tag: str):
        if tag == "dp":
            return self.dp_mesh
        elif tag == "ep_dp":
            return self.ep_dp_mesh
        else:
            raise RuntimeError(f"Unknown pg tag: {tag}")


    def _world_and_rank(self, pg: Optional[ProcessGroup]) -> Tuple[int, int]:
        if pg is None:
            return 1, 0
        return dist.get_world_size(pg), dist.get_rank(pg)

    @nvtx.annotate("MoEFusedV2Optimizer._reduce_scatter_model_grads")
    def _reduce_scatter_model_grads(self) -> None:
        
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                if self.model_has_grad_accum_fp32_buffer:
                    # the model already has a fp32 grad buffer, so the grad is already in fp32
                    # and model's bf16 grad should be None
                    if param.grad is not None:
                        raise RuntimeError("Expected model param grad to be None. Use _main_grad_fp32 to store the grad.")

                    if param._main_grad_fp32 is None:
                        print(f"Warning: model param {name} has None for _main_grad_fp32")
                        param._main_grad_fp32 = torch.zeros_like(param.data, dtype=torch.float32)
                        # continue

                    model_grad_fp32 = param._main_grad_fp32.detach().view(-1) # unsharded local shape, FP32
                else:
                    if param.grad is None:
                        print(f"Warning: model param {name} has None for grad")
                        param.grad = torch.zeros_like(param.data)
                        # continue

                    # model's grad is in bf16, need to convert to fp32 for reduce-scatter
                    model_grad_fp32 = param.grad.detach().view(-1).float() # unsharded local shape, FP32

                # prepare main param grad view
                main_param = self.states[f'{name}.main'] # DTensor, full shape unsharded
                # depending on whether the tensor is sharded or replicated, use reduce_scatter or all-reduce
                dp_world_process_group = self._get_process_group_for_tag('dp')
                dp_world_size = 1 if dp_world_process_group is None else dist.get_world_size(dp_world_process_group)
                if all(isinstance(p, Shard) for p in main_param.placements): # actually main_param is always 1D flat, so it's sharded along dim 0 always
                    # reduce scatter from model grad to main param grad local
                    main_grad_local = torch.empty_like(main_param.to_local()) # local shard shape
                    dist.reduce_scatter_tensor(
                        main_grad_local,
                        model_grad_fp32,
                        group=self._get_process_group_for_tag(param_group['pg']),
                        op=dist.ReduceOp.SUM
                    )
                else:
                    # the tensor is replicated, use all-reduce so that all ranks have the same grad
                    # all-reduce model grad to main param grad local
                    dist.all_reduce(
                        model_grad_fp32,
                        op=dist.ReduceOp.SUM,
                        group=self._get_process_group_for_tag(param_group['pg'])
                    )
                    main_grad_local = model_grad_fp32 # now all ranks have the same grad

                # NOTE: no matter the sum is over dp ranks or ep_dp ranks, ALWAYS divide by dp world size.
                # Explain for ep_dp grads: 
                # if the EP_MP world size is X, then each EP_MP rank is already seeing X times the
                # data, hence each rank's grad is already equivalent to summing over X ranks. The above
                # reduce scatter further sums over the EP_DP ranks, which is equivalent to summing over
                # the full DP world size.
                main_grad_local.div_(dp_world_size) 

                # release model grad to save memory
                if self.model_has_grad_accum_fp32_buffer:
                    param._main_grad_fp32 = None
                else:
                    param.grad = None

                # save main param grad
                self.main_grad[name] = DTensor.from_local(main_grad_local, device_mesh=main_param.device_mesh, placements=main_param.placements)

        return

    @nvtx.annotate("MoEFusedV2Optimizer._copy_model_grads_to_main_grads")
    def _copy_model_grads_to_main_grads(self):
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                if self.model_has_grad_accum_fp32_buffer:
                    # the model already has a fp32 grad buffer, so the grad is already in fp32
                    # and model's bf16 grad should be None
                    if param.grad is not None:
                        raise RuntimeError("Expected model param grad to be None. Use _main_grad_fp32 to store the grad.")

                    if param._main_grad_fp32 is None:
                        print(f"Warning: model param {name} has None for _main_grad_fp32")
                        param._main_grad_fp32 = torch.zeros_like(param.data, dtype=torch.float32)
                        # continue

                    model_grad_fp32 = param._main_grad_fp32.detach().view(-1) # unsharded local shape, FP32
                else:
                    if param.grad is None:
                        print(f"Warning: model param {name} has None for grad")
                        param.grad = torch.zeros_like(param.data)
                        # continue

                    # model's grad is in bf16, need to convert to fp32 for reduce-scatter
                    model_grad_fp32 = param.grad.detach().view(-1).float() # unsharded local shape, FP32

                # release model grad to save memory
                if self.model_has_grad_accum_fp32_buffer:
                    param._main_grad_fp32 = None
                    param.grad = None
                else:
                    param.grad = None

                # prepare main param grad view
                main_param = self.states[f'{name}.main'] # DTensor, full shape unsharded

                self.main_grad[name] = distribute_tensor(model_grad_fp32, device_mesh=main_param.device_mesh, placements=main_param.placements, src_data_rank=None)
                del model_grad_fp32
                
                # further divide by ep_mp world size if it's ep_mp sharded
                if self.moe_mesh is not None and param_group['pg'] == 'ep_dp':
                    ep_mp_world_process_group = self.ep_mp_mesh.get_group()
                    ep_mp_world_size = dist.get_world_size(ep_mp_world_process_group)
                    self.main_grad[name].div_(ep_mp_world_size)



    @nvtx.annotate("MoEFusedV2Optimizer._copy_main_params_to_model_param`s")
    def _copy_main_params_to_model_params(self):
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                main_param = self.states[f'{name}.main']
                # get global tensor from DTensor
                main_param_full = main_param.full_tensor().reshape(param.data.shape) # NOTE: collective ops
                param.data.copy_(main_param_full.to(param.data.dtype))

    def _dealloc_main_grad(self):
        self.main_grad.clear()


    @torch._dynamo.disable()
    def get_step_factor(self) -> torch.Tensor:
        """
        Returns a float tensor which will be `1.0` if the optimizer should proceed with the step
        and `0.0` if the optimizer should skip the step.

        The tensor can be used within the optimizer's step computation to essentially skip a step
        without a host-device sync.
        """
        if len(self._losses) < max(2, self.rolling_interval_length // 2):
            return move_to_device(torch.tensor(1.0), self.device)

        loss_std, loss_mean = torch.std_mean(torch.stack(self._losses[:-1]))
        assert self.latest_loss is not None
        if self._grad_norms:
            assert self.latest_grad_norm is not None
            grad_norm_std, grad_norm_mean = torch.std_mean(torch.stack(self._grad_norms[:-1]))
            step_factor = torch.logical_and(
                (self.latest_loss - loss_mean) <= self.sigma_factor * loss_std,
                (self.latest_grad_norm - grad_norm_mean) <= self.sigma_factor * grad_norm_std,
            )
        else:
            step_factor = (self.latest_loss - loss_mean) <= self.sigma_factor * loss_std

        return step_factor.float()

    @nvtx.annotate("MoEFusedV2Optimizer._step_foreach")
    def _step_foreach(self, closure=None) -> None:
        """Performs adamw step using foreach impl, limiting chunk size to reduce memory usage."""


        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()  # type: ignore
        step_factor = cast(torch.Tensor, step_factor)
        self._step_skipped = 1 - step_factor

        # Allow overriding via attribute; default to 500M elements.
        CHUNK_ELEMS = getattr(self, "_foreach_chunk_threshold", 500_000_000)

        for group in self.param_groups:
            # Per-chunk accumulators
            main_params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps_list: list[torch.Tensor] = []
            running_elems: int = 0

            def flush_chunk():
                nonlocal main_params, grads, exp_avgs, exp_avg_sqs, steps_list, running_elems
                if not main_params:
                    return
                foreach_adamw_step(
                    main_params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    steps_list,
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    step_factor=step_factor,
                    step_increment_bugfix=True,
                )
                # reset for next chunk
                main_params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                steps_list = []
                running_elems = 0

            for name, model_p in group["named_params"].items():
                if not model_p.requires_grad:
                    continue

                main_params.append(self.states[f"{name}.main"].to_local())
                grads.append(self.main_grad[name].to_local())
                exp_avgs.append(self.states[f"{name}.exp_avg"].to_local())
                exp_avg_sqs.append(self.states[f"{name}.exp_avg_sq"].to_local())
                steps_list.append(self.states[f"{name}.step"].to_local())

                running_elems += self.states[f"{name}.main"].to_local().numel()
                # Flush when we reach/exceed the threshold. It's OK to overshoot with the last add.
                if running_elems >= CHUNK_ELEMS:
                    flush_chunk()

            # Flush any tail chunk
            flush_chunk()


    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for n, p in group["named_params"].items():
                # clear bf16 grad
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None

                    else:
                        p.grad.detach_()
                        p.grad.zero_()
                        
                # clear fp32 grad buffer if exists
                if self.model_has_grad_accum_fp32_buffer:
                    if getattr(p, "_main_grad_fp32", None) is not None:
                        p._main_grad_fp32  = None

    
    def unsharded_state_dict(self) -> dict:
        raise NotImplementedError("Removed function")
        
    def _install_optim_from_cpu_dtensor(self, main_sd, state1_sd, state2_sd, distribute_ep_func):
        raise NotImplementedError("Removed function")

        
        

    def state_dict(self) -> dict:
        sd = {}
        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                if not param.requires_grad:
                    continue
                all_suffixes = ['main', 'exp_avg', 'exp_avg_sq', 'step']
                if param_group['pg'] == 'ep_dp':
                    for suffix in all_suffixes:
                        state_dt = self.states[f'{name}.{suffix}']
                        if suffix in ['main', 'exp_avg', 'exp_avg_sq']:
                            # need to convert to dtensor sharded over ep_dp and ep_mp
                            assert self.moe_mesh is not None
                            state_local = state_dt.to_local()
                            state_dt = DTensor.from_local(
                                state_local.unsqueeze(0), # (N,) -> (1, N)
                                device_mesh=self.moe_mesh['ep_dp','ep_mp'],
                                placements=[Shard(1), Shard(0)], # first dim sharded by mp, second dim sharded by dp
                            )
                            state_dt = state_dt.full_tensor().reshape(-1) # NOTE: additional memory usage
                            state_dt = self._distribute_tensor(state_dt, self.dp_mesh)
                            sd[f'{name}.{suffix}'] = state_dt
                        else: # "step"
                            sd[f'{name}.{suffix}'] = state_dt
                else: # DP tensor already in the right dtensor
                    for suffix in all_suffixes:
                        state_dt = self.states[f'{name}.{suffix}']
                        sd[f'{name}.{suffix}'] = state_dt

        assert len(sd) == len(self.states), f"State dict length {len(sd)} does not match live states length {len(self.states)}"
        main_param_count = sum(1 for k in sd.keys() if k.endswith('.main'))
        exp_avg_count = sum(1 for k in sd.keys() if k.endswith('.exp_avg'))
        exp_avg_sq_count = sum(1 for k in sd.keys() if k.endswith('.exp_avg_sq'))
        step_count = sum(1 for k in sd.keys() if k.endswith('.step'))
        assert main_param_count == exp_avg_count == exp_avg_sq_count == step_count, f"State dict counts do not match: main {main_param_count}, exp_avg {exp_avg_count}, exp_avg_sq {exp_avg_sq_count}, step {step_count}"

        return sd       
    


    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        # the loaded state dict is already distributed over the DP mesh,
        # here we need to convert the DP sharded tensors to EP_MP + EP_DP sharded

        for param_group in self.param_groups:
            for name, param in param_group['named_params'].items():
                if not param.requires_grad:
                    continue
                all_suffixes = ['main', 'exp_avg', 'exp_avg_sq', 'step']
                if param_group['pg'] == 'ep_dp':
                    for suffix in all_suffixes:
                        state_dt = self.states[f'{name}.{suffix}']
                        if suffix in ['main', 'exp_avg', 'exp_avg_sq']:
                            # need to convert to dtensor sharded over ep_dp and ep_mp
                            assert self.moe_mesh is not None
                            ckpt_state = state_dict.pop(f'{name}.{suffix}')
                            ckpt_state = ckpt_state.full_tensor() # global full tensor
                            ckpt_state = distribute_tensor(
                                ckpt_state,
                                device_mesh=self.moe_mesh['ep_mp'], # first shard over ep_mp
                                placements=[Shard(0)],
                            ).to_local()
                            ckpt_state = distribute_tensor(
                                ckpt_state,
                                device_mesh=self.moe_mesh['ep_dp'], # then shard over ep_dp
                                placements=[Shard(0)],
                            )
                            # shape checks
                            assert ckpt_state.shape == state_dt.shape, \
                                f"Global shape mismatch: {ckpt_state.shape} vs {state_dt.shape}"
                            assert ckpt_state.to_local().shape == state_dt.to_local().shape, \
                                f"Local shape mismatch: {ckpt_state.to_local().shape} vs {state_dt.to_local().shape}"
                            
                            # now the sharded local tensor should match the local tensor shape of the live state
                            state_dt.to_local().copy_(ckpt_state.to_local())


                        else: # "step"
                            assert "step" == suffix
                            ckpt_state = state_dict.pop(f'{name}.{suffix}').full_tensor()
                            state_dt.copy_(ckpt_state)  # step is a scalar, so no need to convert to local



        return

    def _global_numel(self, tag: str) -> int:
        raise NotImplementedError()


