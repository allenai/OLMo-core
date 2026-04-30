import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)

import nvtx
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Placement,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)

from olmo_core.utils import get_default_device, move_to_device

from ..config import Config, DType
from ..exceptions import OLMoConfigurationError
from ..train.train_module import TrainModule
from .adamw import foreach_adamw_step
from .config import INITIAL_LR_FIELD, LR_FIELD, OptimGroupOverride

log = logging.getLogger(__name__)

MUON_DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_DEFAULT_EPS = 1e-7
MUON_DEFAULT_NS_STEPS = 5

# Opt = TypeVar("Opt", bound=torch.optim.Optimizer)


### DEBUG PRINT ###
def _str_paramt(paramt):
    rets = ""
    total_numel = 0
    for i, pgrp in enumerate(paramt):
        rets += f'{i}: {pgrp["__pg_tag__"]}\n'
        rets += f'    num params: {len(pgrp["params"])}\n'
        rets += f'    num ele: {sum(p.numel() for p in pgrp["params"]):,}\n'
        total_numel += sum(p.numel() for p in pgrp["params"])
    rets += f"Total num ele: {total_numel:,}\n"
    return rets


def _zeropower_via_newtonschulz_2d(
    grad: torch.Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if ns_steps >= 100:
        raise ValueError("Muon ns_steps must be less than 100 for computational efficiency")
    if grad.ndim != 2:
        raise ValueError(f"Muon 2D kernel requires a 2D tensor, got shape {tuple(grad.shape)}")

    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T

    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


_compiled_zeropower_via_newtonschulz_2d = torch.compile(
    _zeropower_via_newtonschulz_2d,
    dynamic=False,
)


def _zeropower_via_newtonschulz_nd(
    grad: torch.Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grad.ndim < 3:
        raise ValueError(
            f"Muon ND kernel requires a tensor with at least 3 dims, got shape {tuple(grad.shape)}"
        )

    a, b, c = ns_coefficients
    ortho_grad = grad.reshape(-1, grad.size(-2), grad.size(-1)).bfloat16()

    if grad.size(-2) > grad.size(-1):
        ortho_grad = ortho_grad.transpose(1, 2)

    ortho_grad = ortho_grad / ortho_grad.norm(dim=(1, 2), keepdim=True).clamp(min=eps)
    for _ in range(ns_steps):
        gram_matrix = torch.bmm(ortho_grad, ortho_grad.transpose(1, 2))
        gram_update = torch.baddbmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad_rhs = (
            ortho_grad.clone()
        )  # copy rhs to safely use baddbmm_ which is faster than baddbmm
        ortho_grad.baddbmm_(gram_update, ortho_grad_rhs, beta=a, alpha=1.0)

    if grad.size(-2) > grad.size(-1):
        ortho_grad = ortho_grad.transpose(1, 2)
    return ortho_grad.reshape_as(grad)


_compiled_zeropower_via_newtonschulz_nd = _zeropower_via_newtonschulz_nd


@nvtx.annotate("_zeropower_via_newtonschulz")
def _zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grad.ndim < 2:
        raise ValueError(
            f"Muon requires a tensor with at least 2 dims, got shape {tuple(grad.shape)}"
        )
    if grad.ndim == 2:
        return _compiled_zeropower_via_newtonschulz_2d(grad, ns_coefficients, ns_steps, eps)
    return _compiled_zeropower_via_newtonschulz_nd(grad, ns_coefficients, ns_steps, eps)


def _adjust_muon_lr(
    lr: Union[float, torch.Tensor],
    adjust_lr_fn: Optional[str],
    param_shape: torch.Size,
) -> Union[float, torch.Tensor]:
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1.0, rows / cols))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(rows, cols))
    else:
        raise ValueError(f"Unsupported Muon lr adjustment function: {adjust_lr_fn}")

    return lr * adjusted_ratio


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


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
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_coefficients: Tuple[float, float, float] = MUON_DEFAULT_NS_COEFFICIENTS
    muon_eps: float = MUON_DEFAULT_EPS
    muon_ns_steps: int = MUON_DEFAULT_NS_STEPS
    muon_adjust_lr_fn: Optional[str] = None

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

    use_distributed: bool = True
    reset_optimizer_moments_on_load: bool = False
    """
    When ``True``, ignore checkpointed ``exp_avg`` and ``exp_avg_sq`` values and
    reset them to zero when restoring optimizer state.
    """

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
                    if param_filter is None:  # No filter applied
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
        group_overrides.insert(0, default_override)  # to ensure default is first

        param_groups = []
        for go in group_overrides:
            if len(go.params) > 0:
                param_groups.append(
                    {
                        "named_params": {
                            param_name: all_params[param_name] for param_name in go.params
                        },
                        **go.opts,  #
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

    def build(
        self, model_parts: List, train_module: TrainModule, strict: bool = True, param_filter=None
    ) -> "MoEFusedV2Optimizer":
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        from ..nn.moe.v2.model import MoEFusedV2Transformer
        from ..train.train_module.transformer.moe_train_module import (
            MoEV2TransformerTrainModule,
        )

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
        dp_groups = self.build_groups(
            model_parts, strict=strict, param_filter=lambda p: id(p) not in ep_param_ids
        )
        for g in dp_groups:
            g["pg"] = "dp"  # type: ignore

        ep_groups = self.build_groups(
            model_parts, strict=strict, param_filter=lambda p: id(p) in ep_param_ids
        )
        for g in ep_groups:
            g["pg"] = "ep_dp"  # type: ignore

        # Concatenate, ensuring the "default" groups remain first in each partition (already ensured by build_groups()).
        all_groups: List[Dict[str, Any]] = list(dp_groups) + list(ep_groups)  # type: ignore

        from olmo_core.nn.parallel.distributed import MultiGroupDistributedDataParallel

        if isinstance(model_parts[0], MultiGroupDistributedDataParallel):
            has_grad_accum_fp32_buffer = model_parts[0]._accumulate_grads_in_fp32

        else:
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
            group_param_names = "\n - ".join(group["named_params"].keys())
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "named_params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['named_params'])} parameter(s):\n - {group_fields_list}\n - params:\n - {group_param_names}"
                )
            else:
                log.info(
                    f"Group {g_idx}, {len(group['named_params'])} parameter(s):\n - params:\n - {group_param_names}"
                )

        has_muon_groups = any(bool(group.get("use_muon", False)) for group in optim.param_groups)
        if self.compile and not has_muon_groups:
            # Compile only the math-heavy update path. Keeping comm/copy-back eager
            # avoids Dynamo/Inductor capturing giant all_gather graphs that can OOM.
            log.info("Compiling optimizer update path (_step_foreach)...")
            optim._step_foreach = torch.compile(optim._step_foreach)
        elif self.compile and has_muon_groups:
            log.info("Skipping optimizer step compilation because Muon groups are enabled.")

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


@dataclass
class _FlatModelParamSyncEntry:
    state_key: str
    param: torch.nn.Parameter
    flat_slice: torch.Tensor
    sharded_target: Optional[torch.Tensor]
    numel: int
    is_sharded: bool
    local_numel: int
    local_offset: int


@dataclass
class _FlatModelParamSyncGroup:
    tag: str
    dtype: torch.dtype
    flat_buffer: torch.Tensor
    sharded_entries: List[_FlatModelParamSyncEntry]
    replicated_entries: List[_FlatModelParamSyncEntry]
    total_sharded_local_numel: int
    process_group: Optional[ProcessGroup]
    world_size: int


class MoEFusedV2Optimizer:
    LOSSES_STATE_DICT_KEY = "__moe_skip_step_losses"
    GRAD_NORMS_STATE_DICT_KEY = "__moe_skip_step_grad_norms"
    ADAM_MOMENT_STATE_SUFFIXES = ("exp_avg", "exp_avg_sq")
    MUON_MOMENT_STATE_SUFFIXES = ("muon_momentum",)
    MOMENT_STATE_SUFFIXES = ADAM_MOMENT_STATE_SUFFIXES + MUON_MOMENT_STATE_SUFFIXES

    def __init__(
        self,
        param_groups: Iterable[Dict[str, Any]],
        world_mesh: Dict[str, Optional[DeviceMesh]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_coefficients: Tuple[float, float, float] = MUON_DEFAULT_NS_COEFFICIENTS,
        muon_eps: float = MUON_DEFAULT_EPS,
        muon_ns_steps: int = MUON_DEFAULT_NS_STEPS,
        muon_adjust_lr_fn: Optional[str] = None,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        max_grad_norm: float = 1.0,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        device: Optional[torch.device] = None,
        model_has_grad_accum_fp32_buffer: bool = False,  # whether the optimizer should expect the model to have fp32 grad accum buffers
        # foreach: bool = False,
        # --- new args for sharding across multiple PGs ---
        dp_group: Optional[ProcessGroup] = None,
        ep_dp_group: Optional[ProcessGroup] = None,
        broadcast_bucket_mb: int = 32,
        do_not_shard_tensor_smaller_than: int = 4096,
        use_distributed: bool = True,
        check_nan_inf_grad: bool = False,
        reset_optimizer_moments_on_load: bool = False,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        assert muon_momentum >= 0.0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        defaults.update(
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            muon_ns_coefficients=muon_ns_coefficients,
            muon_eps=muon_eps,
            muon_ns_steps=muon_ns_steps,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
        )

        self.model_has_grad_accum_fp32_buffer = model_has_grad_accum_fp32_buffer
        self.use_distributed = use_distributed
        self.reset_optimizer_moments_on_load = reset_optimizer_moments_on_load

        def _add_defaults_to_param_group(pg: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in defaults.items():
                pg.setdefault(k, v)
            pg.setdefault("use_muon", False)
            return pg

        # add defaults to each param group
        param_groups = [_add_defaults_to_param_group(pg) for pg in param_groups]

        # for print info
        self._model_param_sz = 0
        for param_group in param_groups:
            self._model_param_sz += sum(
                p.numel() * p.element_size() for (n, p) in param_group["named_params"].items()
            )

        # ---- Sharding context (DP and EP-DP) ----
        self._dp_group: Optional[ProcessGroup] = dp_group
        self._ep_dp_group: Optional[ProcessGroup] = ep_dp_group

        # self._broadcast_bucket_bytes: int = int(broadcast_bucket_mb * 1024 * 1024)

        assert world_mesh["dense"] is not None, "DP mesh must be provided"

        self.dense_mesh: DeviceMesh = world_mesh["dense"]  # ('pp', 'dp')
        self.moe_mesh: Optional[DeviceMesh] = world_mesh["moe"]  # ('pp', 'ep_dp', 'ep_mp')

        self.dp_mesh = self.dense_mesh["dp"]
        self.ep_dp_mesh = self.moe_mesh["ep_dp"] if self.moe_mesh else None
        self.ep_mp_mesh = self.moe_mesh["ep_mp"] if self.moe_mesh else None

        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = device
        self.max_grad_norm = max_grad_norm
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        self.check_nan_inf_grad = check_nan_inf_grad
        self.states_dtype = torch.float32
        self.main_grad_dtype = torch.float32

        # self.foreach = foreach
        self._step_skipped: Optional[torch.Tensor] = None
        self.do_not_shard_tensor_smaller_than = do_not_shard_tensor_smaller_than
        self._use_reduce_scatter_grads = True
        self.main_grad: Dict[str, torch.Tensor] = {}
        self._param_uses_muon: Dict[str, bool] = {}
        self._flat_model_sync_groups: "OrderedDict[str, _FlatModelParamSyncGroup]" = OrderedDict()

        # check
        device = None
        has_bf16_param = False
        has_fp32_param = False
        for param_group in param_groups:
            for i, (name, param) in enumerate(param_group["named_params"].items()):
                if not param.requires_grad:
                    continue
                if device is None:
                    device = param.device
                else:
                    assert device == param.device, "Inconsistent device found"
                # float16 params:
                if param.type() in ["torch.cuda.HalfTensor", "torch.cuda.BFloat16Tensor"]:
                    has_bf16_param = True
                elif param.type() in ["torch.cuda.FloatTensor"]:
                    has_fp32_param = True

        if has_bf16_param and has_fp32_param:
            raise ValueError(
                "Mixed bf16 and fp32 parameters are not supported in MoEFusedV2Optimizer"
            )

        if has_bf16_param:
            # The model only has bf16 params
            # The optimizer has to decide whether to maintain fp32 main params
            self.should_maintain_fp32_main_param = True
        else:
            # The model has its own copy of fp32 main params
            self.should_maintain_fp32_main_param = False

        self.states: Dict[str, DTensor] = OrderedDict()

        muon_fallback_params: List[str] = []
        for param_group in param_groups:
            group_requests_muon = bool(param_group.get("use_muon", False))
            for name, param in param_group["named_params"].items():
                use_muon_for_param = group_requests_muon and param.ndim >= 2
                self._param_uses_muon[name] = use_muon_for_param
                if group_requests_muon and param.ndim < 2:
                    muon_fallback_params.append(name)

        if muon_fallback_params:
            log.warning(
                "Falling back to AdamW for params with rank < 2 in Muon groups: %s",
                ", ".join(muon_fallback_params),
            )

        for param_group in param_groups:
            # configure the device mesh to shard the group
            device_mesh = self._get_dp_device_mesh_for_tag(param_group["pg"])
            assert device_mesh is not None, f"Device mesh for pg tag {param_group['pg']} is None"

            # wrap each param with DTensor
            for name, param in param_group["named_params"].items():
                # flat in fp32
                num_elements = param.numel()
                use_muon_for_param = self._param_uses_muon[name]

                # main param
                if self.should_maintain_fp32_main_param:
                    main_param = torch.zeros(num_elements, dtype=torch.float32, device=device)
                    main_param = self._distribute_tensor(main_param, device_mesh)
                    self.states[f"{name}.main"] = main_param
                else:
                    assert (
                        param.dtype == torch.float32
                    ), "Expect fp32 param when should_maintain_fp32_main_param is False"
                    # wrap in DTensor so it works with rest of the code
                    self.states[f"{name}.main"] = DTensor.from_local(
                        param.data.view(-1), device_mesh=device_mesh, placements=[Replicate()]
                    )

                if use_muon_for_param:
                    muon_momentum_buffer = torch.zeros(
                        num_elements, dtype=self.states_dtype, device=device
                    )
                    muon_momentum_buffer = self._distribute_tensor(
                        muon_momentum_buffer, device_mesh
                    )
                    self.states[f"{name}.muon_momentum"] = muon_momentum_buffer
                else:
                    # exp avg
                    exp_avg = torch.zeros(num_elements, dtype=self.states_dtype, device=device)
                    exp_avg = self._distribute_tensor(exp_avg, device_mesh)
                    self.states[f"{name}.exp_avg"] = exp_avg

                    # exp avg sq
                    exp_avg_sq = torch.zeros(num_elements, dtype=self.states_dtype, device=device)
                    exp_avg_sq = self._distribute_tensor(exp_avg_sq, device_mesh)
                    self.states[f"{name}.exp_avg_sq"] = exp_avg_sq

                # step
                step_tensor = torch.zeros((), dtype=torch.float32, device=device)
                step_tensor = distribute_tensor(
                    step_tensor,
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )
                self.states[f"{name}.step"] = step_tensor

        self.param_groups = param_groups
        if self.should_maintain_fp32_main_param:
            self._init_flat_model_param_buffers()

        # copy model params to main params
        if self.should_maintain_fp32_main_param:
            for param_group in param_groups:
                for name, param in param_group["named_params"].items():
                    main_param = self.states[f"{name}.main"]

                    assign_full_tensor_to_dtensor(
                        dst=main_param, src=param.data.float().reshape(-1)
                    )

        if self.should_maintain_fp32_main_param:
            self._check_model_param_main_param_the_same()

        self.print_memory_summary()

        return

    def print_memory_summary(self):
        total_params = 0
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                total_params += param.numel()
        log.info(f"[MoEFusedV2Optimizer] Total model params: {total_params:,}")

        # main
        def count_numel(tag: str):
            global_state_numel = 0
            local_state_numel = 0
            num_tensors_sharded = 0
            num_tensors_replicated = 0
            sharded_state_numel = 0
            replicated_state_numel = 0
            for state_key, state_val in self.states.items():
                if state_key.endswith(f".{tag}"):
                    global_state_numel += state_val.numel()
                    local_state_numel += state_val.to_local().numel()
                    if any(isinstance(p, Shard) for p in state_val.placements):
                        num_tensors_sharded += 1
                        sharded_state_numel += state_val.to_local().numel()
                    else:
                        num_tensors_replicated += 1
                        replicated_state_numel += state_val.to_local().numel()
            return (
                global_state_numel,
                local_state_numel,
                num_tensors_sharded,
                num_tensors_replicated,
                sharded_state_numel,
                replicated_state_numel,
            )

        def to_str_N_B_GB(num):
            return f"{num:,} | {num/1000**3:.4} Billion | {num * 4 /1024**3:.4} GB"

        def info_str(tag: str, stat: Tuple[int, int, int, int, int, int]):
            info_str = ""
            info_str += f"[MoEFusedV2Optimizer] {tag} - Global params: {to_str_N_B_GB(stat[0])}, Local params: {to_str_N_B_GB(stat[1])}\n"
            info_str += f"    Sharded tensors: {stat[2]}, total local sharded params: {to_str_N_B_GB(stat[4])}\n"
            info_str += f"    Replicated tensors: {stat[3]}, total local replicated params: {to_str_N_B_GB(stat[5])}\n"
            return info_str

        main_stat = count_numel("main")
        exp_avg_stat = count_numel("exp_avg")
        exp_avg_sq_stat = count_numel("exp_avg_sq")
        muon_momentum_stat = count_numel("muon_momentum")

        print_str = ""

        print_str += info_str("Main param", main_stat)
        if exp_avg_stat[0] > 0:
            print_str += info_str("Exp avg", exp_avg_stat)
        if exp_avg_sq_stat[0] > 0:
            print_str += info_str("Exp avg sq", exp_avg_sq_stat)
        if muon_momentum_stat[0] > 0:
            print_str += info_str("Muon momentum", muon_momentum_stat)

        BYTES_IN_GB = 1024**3

        total_global_optim_gb = main_stat[0] * self.main_grad_dtype.itemsize / BYTES_IN_GB
        total_global_optim_gb += (
            (exp_avg_stat[0] + exp_avg_sq_stat[0] + muon_momentum_stat[0])
            * self.states_dtype.itemsize
            / BYTES_IN_GB
        )

        total_local_optim_gb = main_stat[1] * self.main_grad_dtype.itemsize / BYTES_IN_GB
        total_local_optim_gb += (
            (exp_avg_stat[1] + exp_avg_sq_stat[1] + muon_momentum_stat[1])
            * self.states_dtype.itemsize
            / BYTES_IN_GB
        )

        total_model_gb = self._model_param_sz / BYTES_IN_GB
        print_str += f"[MoEFusedV2Optimizer] Total optimizer states size: {total_global_optim_gb:.4f} GB global, {total_local_optim_gb:.4f} GB local\n"

        if self.model_has_grad_accum_fp32_buffer:
            total_model_grad_gb = 2 * total_model_gb  # extra fp32 grad buffer
        else:
            total_model_grad_gb = total_model_gb  # bf16 grad only
        print_str += f"[MoEFusedV2Optimizer] Model params size (GB): {total_model_gb:.4f} GB, model grads size (GB): {total_model_grad_gb:.4f} GB\n"
        total_static = total_local_optim_gb + total_model_gb + total_model_grad_gb

        print_str += (
            f"[MoEFusedV2Optimizer] Total estimated static memory (GB): {total_static:.4f} GB\n"
        )

        log.info(print_str)

    def _init_flat_model_param_buffers(self) -> None:
        groups_by_tag: "OrderedDict[str, List[Tuple[str, torch.nn.Parameter]]]" = OrderedDict()
        seen_param_ids: Set[int] = set()

        for param_group in self.param_groups:
            tag = param_group["pg"]
            entries = groups_by_tag.setdefault(tag, [])
            for name, param in param_group["named_params"].items():
                param_id = id(param)
                if param_id in seen_param_ids:
                    raise RuntimeError(
                        f"Parameter '{name}' appears multiple times in optimizer groups"
                    )
                seen_param_ids.add(param_id)
                entries.append((name, param))

        self._flat_model_sync_groups = OrderedDict()

        for tag, named_params in groups_by_tag.items():
            if not named_params:
                continue

            group_dtype = named_params[0][1].dtype
            total_numel = 0
            total_sharded_local_numel = 0
            for name, param in named_params:
                if not param.data.is_contiguous():
                    raise RuntimeError(
                        f"Flat model param buffers require contiguous parameter storage, got '{name}'"
                    )
                if param.dtype != group_dtype:
                    raise RuntimeError(
                        f"Mixed dtypes are not supported in flat model buffer group '{tag}'"
                    )
                total_numel += param.numel()

                main_param = self.states[f"{name}.main"]
                if any(isinstance(p, Shard) for p in main_param.placements):
                    total_sharded_local_numel += main_param.to_local().numel()

            flat_buffer = torch.empty(total_numel, device=self.device, dtype=group_dtype)
            process_group = self._get_process_group_for_tag(tag)
            world_size = 1 if process_group is None else dist.get_world_size(process_group)

            sharded_entries: List[_FlatModelParamSyncEntry] = []
            replicated_entries: List[_FlatModelParamSyncEntry] = []

            global_offset = 0
            local_offset = 0
            for name, param in named_params:
                numel = param.numel()
                old_param_data = param.data
                flat_slice = flat_buffer.narrow(0, global_offset, numel)
                flat_param_view = flat_slice.view_as(old_param_data)
                flat_param_view.copy_(old_param_data)
                param.data = flat_param_view

                main_param = self.states[f"{name}.main"]
                is_sharded = any(isinstance(p, Shard) for p in main_param.placements)
                local_numel = main_param.to_local().numel() if is_sharded else 0
                sharded_target = flat_slice.view(world_size, local_numel) if is_sharded else None

                entry = _FlatModelParamSyncEntry(
                    state_key=f"{name}.main",
                    param=param,
                    flat_slice=flat_slice,
                    sharded_target=sharded_target,
                    numel=numel,
                    is_sharded=is_sharded,
                    local_numel=local_numel,
                    local_offset=local_offset,
                )
                if is_sharded:
                    sharded_entries.append(entry)
                    local_offset += local_numel
                else:
                    replicated_entries.append(entry)

                global_offset += numel

            self._flat_model_sync_groups[tag] = _FlatModelParamSyncGroup(
                tag=tag,
                dtype=group_dtype,
                flat_buffer=flat_buffer,
                sharded_entries=sharded_entries,
                replicated_entries=replicated_entries,
                total_sharded_local_numel=total_sharded_local_numel,
                process_group=process_group,
                world_size=world_size,
            )

        self._refresh_rowwise_fp8_caches_from_model_params()

    def _check_model_param_main_param_the_same(self):
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                main_param = self.states[f"{name}.main"]
                # get global tensor from DTensor
                main_param_full = main_param.full_tensor().reshape(-1)
                model_param = param.data.float().reshape(-1)
                if not torch.allclose(model_param, main_param_full, atol=1e-5):
                    raise ValueError(
                        f"{name}: Model param {param} and main param {main_param} are not close"
                    )

    def _distribute_tensor(
        self,
        tensor,
        device_mesh,
        force_shard: bool = False,
        force_replicate: bool = False,
    ) -> DTensor:
        num_elements = tensor.numel()
        if force_shard and force_replicate:
            raise ValueError("A tensor cannot be both force-sharded and force-replicated")
        if force_shard:
            # always shard, useful for saving checkpoint
            placements = [Shard(0)]
        elif force_replicate:
            placements = [Replicate()]
        elif self.use_distributed:
            if (
                num_elements >= self.do_not_shard_tensor_smaller_than
                and num_elements % device_mesh.size(0) == 0
            ):
                # this is distributed optimizer, so each rank holds one shard of the data
                placements = [Shard(0)]
            else:
                # small tensor, do not shard
                placements = [Replicate()]
                log.info(f"[MoEFusedV2Optimizer] A tensor of size {num_elements} is replicated.")
        else:
            # always no shard
            placements = [Replicate()]

        tensor_dt = distribute_tensor(
            tensor,
            device_mesh=device_mesh,
            placements=placements,
        )

        return tensor_dt

    def offload_optimizer_states(self):
        raise NotImplementedError()
        # Offload optimizer states to CPU to save GPU memory

    def reload_optimizer_states_to_device(
        self,
    ):
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
        dp_grads_replicated = []
        dp_grads_sharded = []
        ep_dp_grads_replicated = []
        ep_dp_grads_sharded = []

        # debug_grads = {}
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                placements = self.states[f"{name}.main"].placements
                assert len(placements) == 1, "Expect only one placement per tensor"
                main_grad = self.main_grad[name]

                if param_group["pg"] == "dp":
                    if placements[0].is_shard():
                        dp_grads_sharded.append(main_grad)
                    else:
                        dp_grads_replicated.append(main_grad)
                elif param_group["pg"] == "ep_dp":
                    if placements[0].is_shard():
                        ep_dp_grads_sharded.append(main_grad)
                    else:
                        ep_dp_grads_replicated.append(main_grad)

        dp_grads_norm_sharded = nn.utils.get_total_norm(
            dp_grads_sharded, norm_type=2.0, error_if_nonfinite=False
        )
        dp_grads_norm_replicated = nn.utils.get_total_norm(
            dp_grads_replicated, norm_type=2.0, error_if_nonfinite=False
        )

        dp_grads_norm_sharded_reduced = self._reduce_norm(
            dp_grads_norm_sharded, self.dp_mesh.get_group()
        )  # reduce across DP
        dp_grad_norm = self._combine_norm(dp_grads_norm_replicated, dp_grads_norm_sharded_reduced)

        if self.moe_mesh is not None:
            ep_dp_grads_norm_sharded = nn.utils.get_total_norm(
                ep_dp_grads_sharded, norm_type=2.0, error_if_nonfinite=False
            )
            ep_dp_grads_norm_replicated = nn.utils.get_total_norm(
                ep_dp_grads_replicated, norm_type=2.0, error_if_nonfinite=False
            )

            ep_dp_grads_norm_sharded_reduced = self._reduce_norm(
                ep_dp_grads_norm_sharded, self.ep_dp_mesh.get_group()
            )  # reduce across EP_DP
            ep_dp_grad_norm = self._combine_norm(
                ep_dp_grads_norm_replicated, ep_dp_grads_norm_sharded_reduced
            )

            ep_grad_norm = self._reduce_norm(
                ep_dp_grad_norm, self.ep_mp_mesh.get_group()
            )  # reduce across EP_MP

            total_grad_norm = self._combine_norm(dp_grad_norm, ep_grad_norm)
        else:
            total_grad_norm = dp_grad_norm

        ################

        # dp_grad_norm = nn.utils.get_total_norm(dp_grads, norm_type=2.0, error_if_nonfinite=False)
        # dp_grad_norm = cast(DTensor, dp_grad_norm).full_tensor()

        # if self.moe_mesh is not None:
        #     ep_dp_grad_norm = nn.utils.get_total_norm(ep_dp_grads, norm_type=2.0, error_if_nonfinite=False)
        #     ep_dp_grad_norm = cast(DTensor, ep_dp_grad_norm).full_tensor()

        #     # reduce EP_MP
        #     assert self.ep_mp_mesh is not None
        #     ep_dp_grad_norm = ep_dp_grad_norm.square()
        #     dist.all_reduce(ep_dp_grad_norm, op=dist.ReduceOp.SUM, group=self.ep_mp_mesh.get_group())
        #     ep_dp_grad_norm = ep_dp_grad_norm.sqrt()

        #     # combine DP and EP_DP grad norms
        #     total_grad_norm = torch.sqrt(dp_grad_norm.square() + ep_dp_grad_norm.square())
        # else:
        #     assert len(ep_dp_grads) == 0, "No EP_DP grads should exist if no MOE mesh"
        #     total_grad_norm = dp_grad_norm

        # reduce PP
        assert self.dense_mesh.mesh_dim_names is not None
        if "pp" in self.dense_mesh.mesh_dim_names:
            total_grad_norm = self._reduce_norm(total_grad_norm, self.dense_mesh["pp"].get_group())
            # total_grad_norm = total_grad_norm.square()
            # dist.all_reduce(total_grad_norm, op=dist.ReduceOp.SUM, group=self.dense_mesh['pp'].get_group())
            # total_grad_norm = total_grad_norm.sqrt()

        clip_coef = self.max_grad_norm / (total_grad_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0).to(total_grad_norm.device)

        all_grads = list(self.main_grad.values())
        torch._foreach_mul_(all_grads, clip_coef_clamped)

        # torch._foreach_mul_(dp_grads, clip_coef_clamped)
        # if len(ep_dp_grads) > 0:
        #     torch._foreach_mul_(ep_dp_grads, clip_coef_clamped)

        return total_grad_norm

    def _combine_norm(self, n1, n2) -> torch.Tensor:
        return torch.sqrt(n1.square() + n2.square())

    def _reduce_norm(self, norm: torch.Tensor, pg: ProcessGroup) -> torch.Tensor:
        norm = norm.square()
        dist.all_reduce(norm, op=dist.ReduceOp.SUM, group=pg)
        norm = norm.sqrt()
        return norm

    @overload  # make pylance happy
    def step(self, closure: None = ...) -> None:
        ...

    @overload  # make pylance happy
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @torch.no_grad()
    @nvtx.annotate("MoEFusedV2Optimizer.step")
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # dbg_mem_before_cp1 = torch.cuda.memory_allocated() / 1024**3
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

        if self.check_nan_inf_grad and (total_grad_norm.isnan() or total_grad_norm.isinf()):
            assert (
                False
            ), f"[Error] rank={dist.get_rank()} grad norm is {total_grad_norm}, skipping step"

        self.latest_grad_norm = total_grad_norm

        self._step_foreach(closure)

        self._dealloc_main_grad()

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

    def _state_suffixes_for_param(self, name: str) -> Tuple[str, ...]:
        if self._param_uses_muon[name]:
            return ("main", "muon_momentum", "step")
        return ("main", "exp_avg", "exp_avg_sq", "step")

    def _gather_sharded_flat_tensor(
        self, local_tensor: torch.Tensor, state_dt: DTensor
    ) -> torch.Tensor:
        if not any(isinstance(p, Shard) for p in state_dt.placements):
            return local_tensor

        gathered = coalesced_all_gather([local_tensor], state_dt.device_mesh.get_group())[0]
        return gathered.reshape(-1)

    def _load_moment_state_or_zero(
        self, state_dict: Dict[str, Any], state_key: str
    ) -> Optional[Any]:
        if state_key in state_dict:
            return state_dict.pop(state_key)

        self._ensure_local_state_storage(state_key).to_local().zero_()
        return None

    def _ep_dp_state_to_checkpoint(self, live_state_dt: DTensor) -> DTensor:
        assert self.moe_mesh is not None
        assert len(live_state_dt.placements) == 1
        ep_dp_placement = live_state_dt.placements[0]
        if ep_dp_placement.is_shard():
            combined_ep_dp_placement: Placement = Shard(1)
        else:
            combined_ep_dp_placement = Replicate()

        state_local = live_state_dt.to_local()
        state_dt_for_save = DTensor.from_local(
            state_local.unsqueeze(0),
            device_mesh=self.moe_mesh["ep_dp", "ep_mp"],
            placements=[combined_ep_dp_placement, Shard(0)],
        )
        state_dt_for_save = state_dt_for_save.full_tensor().reshape(-1)
        return self._distribute_tensor(state_dt_for_save, self.dp_mesh, force_shard=True)

    def _load_ep_dp_state_from_checkpoint(self, state_key: str, ckpt_state: DTensor) -> None:
        assert self.moe_mesh is not None
        state_dt = self._ensure_local_state_storage(state_key)
        ckpt_state = ckpt_state.full_tensor()
        ckpt_state = distribute_tensor(
            ckpt_state,
            device_mesh=self.moe_mesh["ep_mp"],
            placements=[Shard(0)],
        ).to_local()

        if state_dt.placements[0].is_shard():
            ckpt_state = distribute_tensor(
                ckpt_state,
                device_mesh=self.moe_mesh["ep_dp"],
                placements=[Shard(0)],
            )
        else:
            ckpt_state = distribute_tensor(
                ckpt_state,
                device_mesh=self.moe_mesh["ep_dp"],
                placements=[Replicate()],
            )

        ckpt_local = ckpt_state.to_local()
        assert (
            ckpt_state.shape == state_dt.shape
        ), f"Global shape mismatch for {state_key}: {ckpt_state.shape} vs {state_dt.shape}"
        assert (
            ckpt_local.shape == state_dt.to_local().shape
        ), f"Local shape mismatch for {state_key}: {ckpt_local.shape} vs {state_dt.to_local().shape}"
        state_dt.to_local().copy_(ckpt_local)

    @nvtx.annotate("MoEFusedV2Optimizer._reduce_scatter_model_grads")
    def _reduce_scatter_model_grads(self) -> None:
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if self.model_has_grad_accum_fp32_buffer:
                    # the model already has a fp32 grad buffer, so the grad is already in fp32
                    # and model's bf16 grad should be None
                    if param.grad is not None:
                        raise RuntimeError(
                            "Expected model param grad to be None. Use _main_grad_fp32 to store the grad."
                        )

                    if param._main_grad_fp32 is None:
                        raise RuntimeError(
                            f"Missing _main_grad_fp32 for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    model_grad_fp32 = param._main_grad_fp32.detach().view(
                        -1
                    )  # unsharded local shape, FP32
                else:
                    if param.grad is None:
                        raise RuntimeError(
                            f"Missing .grad for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    # model's grad is in bf16, need to convert to fp32 for reduce-scatter
                    model_grad_fp32 = (
                        param.grad.detach().view(-1).float()
                    )  # unsharded local shape, FP32

                # prepare main param grad view
                main_param = self.states[f"{name}.main"]  # DTensor, full shape unsharded
                # depending on whether the tensor is sharded or replicated, use reduce_scatter or all-reduce
                dp_world_process_group = self._get_process_group_for_tag("dp")
                dp_world_size = (
                    1
                    if dp_world_process_group is None
                    else dist.get_world_size(dp_world_process_group)
                )
                if all(
                    isinstance(p, Shard) for p in main_param.placements
                ):  # actually main_param is always 1D flat, so it's sharded along dim 0 always
                    # reduce scatter from model grad to main param grad local
                    main_grad_local = torch.empty_like(main_param.to_local())  # local shard shape
                    dist.reduce_scatter_tensor(
                        main_grad_local,
                        model_grad_fp32,
                        group=self._get_process_group_for_tag(param_group["pg"]),
                        op=dist.ReduceOp.SUM,
                    )
                else:
                    # the tensor is replicated, use all-reduce so that all ranks have the same grad
                    # all-reduce model grad to main param grad local
                    dist.all_reduce(
                        model_grad_fp32,
                        op=dist.ReduceOp.SUM,
                        group=self._get_process_group_for_tag(param_group["pg"]),
                    )
                    main_grad_local = model_grad_fp32  # now all ranks have the same grad

                # NOTE: no matter the sum is over dp ranks or ep_dp ranks, ALWAYS divide by dp world size.
                # Explain for ep_dp grads:
                # if the EP_MP world size is X, then each EP_MP rank is already seeing X times the
                # data, hence each rank's grad is already equivalent to summing over X ranks. The above
                # reduce scatter further sums over the EP_DP ranks, which is equivalent to summing over
                # the full DP world size.
                main_grad_local.div_(dp_world_size)

                # save main param grad
                self.main_grad[name] = DTensor.from_local(
                    main_grad_local,
                    device_mesh=main_param.device_mesh,
                    placements=main_param.placements,
                )

        return

    @nvtx.annotate("MoEFusedV2Optimizer._copy_model_grads_to_main_grads")
    def _copy_model_grads_to_main_grads(self):
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if self.model_has_grad_accum_fp32_buffer:
                    # the model already has a fp32 grad buffer, so the grad is already in fp32
                    # and model's bf16 grad should be None
                    if param.grad is not None:
                        raise RuntimeError(
                            "Expected model param grad to be None. Use _main_grad_fp32 to store the grad."
                        )

                    if param._main_grad_fp32 is None:
                        raise RuntimeError(
                            f"Missing _main_grad_fp32 for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    model_grad_fp32 = param._main_grad_fp32.detach().view(
                        -1
                    )  # unsharded local shape, FP32
                else:
                    if param.grad is None:
                        raise RuntimeError(
                            f"Missing .grad for param '{name}'. "
                            "Grad buffers must stay bound to DDP bucket views."
                        )

                    # model's grad is in bf16, need to convert to fp32 for reduce-scatter
                    # model_grad_fp32 = param.grad.detach().view(-1).float() # unsharded local shape, FP32
                    model_grad_fp32 = param.grad.detach().view(
                        -1
                    )  # unsharded local shape, BF16. It should be a view of the reducer bucket

                # prepare main param grad view
                main_param = self.states[f"{name}.main"]  # DTensor, full shape unsharded

                # self.main_grad[name] = distribute_tensor(model_grad_fp32, device_mesh=main_param.device_mesh, placements=main_param.placements, src_data_rank=None)

                # it turns out distribute_tensor is too slow on cpu
                # here is a more direct way
                self.main_grad[name] = self.narrow_tensor(
                    model_grad_fp32, main_param.device_mesh, main_param.placements
                )

                del model_grad_fp32

                # further divide by ep_mp world size if it's ep_mp sharded
                if self.moe_mesh is not None and param_group["pg"] == "ep_dp":
                    ep_mp_world_process_group = self.ep_mp_mesh.get_group()
                    ep_mp_world_size = dist.get_world_size(ep_mp_world_process_group)
                    self.main_grad[name].div_(ep_mp_world_size)

    def narrow_tensor(
        self, orignal: torch.Tensor, device_mesh: DeviceMesh, placements: List[Placement]
    ):
        assert len(placements) == 1, "Only support 1D sharding"
        assert device_mesh.ndim == 1, "Only support 1D device mesh"
        assert orignal.dim() == 1, "Only support 1D tensor"
        placement = placements[0]

        if placement.is_replicate():
            return orignal
            # return DTensor.from_local(orignal, device_mesh=device_mesh, placements=placements)

        assert placement.is_shard(), "Only support shard or replicate placements"
        coord = device_mesh.get_coordinate()[0]
        ws = device_mesh.size(0)
        shard_size = orignal.numel() // ws
        start = coord * shard_size
        local_shard = orignal.narrow(0, start, shard_size)

        return local_shard
        # return DTensor.from_local(local_shard, device_mesh=device_mesh, placements=placements)

    @torch._dynamo.disable()
    @nvtx.annotate("MoEFusedV2Optimizer._copy_main_params_to_model_params")
    def _copy_main_params_to_model_params(self):
        if self._flat_model_sync_groups:
            self._copy_main_params_to_flat_model_buffers()
            self._refresh_rowwise_fp8_caches_from_model_params()
            return

        LAUNCH_AG_THRESHOLD = 500_000_000  # X elements
        for param_group in self.param_groups:
            # initialize for coalesced all_gather
            input_dtensors: List[DTensor] = []
            output_params: List[torch.Tensor] = []
            input_numel = 0

            def flush_all_gather():
                nonlocal input_dtensors, output_params, input_numel
                if len(input_dtensors) == 0:
                    return
                pg = input_dtensors[0].device_mesh.get_group()
                gather_dtype = output_params[0].dtype
                input_locals = [t.to_local() for t in input_dtensors]
                flat_global, sizes, offsets = coalesced_all_gather_flat(
                    input_locals,
                    pg,
                    output_dtype=gather_dtype,
                )

                world_size = flat_global.shape[0]
                for size, off, out_param in zip(sizes, offsets, output_params):
                    out_param.data.view(world_size, size).copy_(flat_global[:, off : off + size])

                output_params.clear()
                input_dtensors.clear()
                input_numel = 0

            for name, param in param_group["named_params"].items():
                main_param = self.states[f"{name}.main"]
                if not any(isinstance(p, Shard) for p in main_param.placements):
                    # replicated tensor, directly get full tensor
                    main_param_local = main_param.to_local().reshape(param.data.shape)
                    # param.data.copy_(main_param_local.to(param.data.dtype))
                    param.data.copy_(main_param_local)
                    continue

                # check for process group
                if len(input_dtensors) > 0:
                    assert main_param.device_mesh == input_dtensors[0].device_mesh
                    assert param.dtype == output_params[0].dtype

                input_dtensors.append(main_param)
                output_params.append(param)
                input_numel += main_param.numel()

                if input_numel >= LAUNCH_AG_THRESHOLD:
                    flush_all_gather()

            # final gather
            if len(input_dtensors) > 0:
                flush_all_gather()

        self._refresh_rowwise_fp8_caches_from_model_params()
        return

    def _copy_main_params_to_flat_model_buffers(self) -> None:
        for sync_group in self._flat_model_sync_groups.values():
            for entry in sync_group.replicated_entries:
                main_param = self.states[entry.state_key]
                # entry.flat_slice.copy_(main_param.to_local().reshape(-1).to(sync_group.dtype))
                entry.flat_slice.copy_(main_param.to_local().reshape(-1))

            if not sync_group.sharded_entries:
                continue

            if sync_group.world_size == 1:
                for entry in sync_group.sharded_entries:
                    main_param = self.states[entry.state_key]
                    assert entry.sharded_target is not None
                    # entry.sharded_target.copy_(
                    #     main_param.to_local().reshape(1, entry.local_numel).to(sync_group.dtype)
                    # )
                    entry.sharded_target.copy_(main_param.to_local().reshape(1, entry.local_numel))
                continue

            assert sync_group.process_group is not None
            pack_buffer = torch.empty(
                sync_group.total_sharded_local_numel,
                device=self.device,
                dtype=sync_group.dtype,
            )
            gathered_buffer = torch.empty(
                sync_group.world_size * sync_group.total_sharded_local_numel,
                device=self.device,
                dtype=sync_group.dtype,
            )

            for entry in sync_group.sharded_entries:
                main_param = self.states[entry.state_key]
                pack_buffer[entry.local_offset : entry.local_offset + entry.local_numel].copy_(
                    main_param.to_local().reshape(-1)
                )

            dist.all_gather_into_tensor(
                gathered_buffer,
                pack_buffer,
                group=sync_group.process_group,
            )

            gathered_matrix = gathered_buffer.view(
                sync_group.world_size,
                sync_group.total_sharded_local_numel,
            )
            for entry in sync_group.sharded_entries:
                assert entry.sharded_target is not None
                entry.sharded_target.copy_(
                    gathered_matrix[:, entry.local_offset : entry.local_offset + entry.local_numel]
                )

    @nvtx.annotate("MoEFusedV2Optimizer._refresh_rowwise_fp8_caches_from_model_params")
    def _refresh_rowwise_fp8_caches_from_model_params(self) -> None:
        owners: List[Any] = []
        seen: Set[int] = set()
        for group in self.param_groups:
            for _, param in group["named_params"].items():
                owner_ref = getattr(param, "_moe_rowwise_fp8_cache_owner", None)
                if owner_ref is None:
                    continue
                owner = owner_ref() if callable(owner_ref) else owner_ref
                if owner is None:
                    continue
                owner_id = id(owner)
                if owner_id in seen:
                    continue
                if not hasattr(owner, "refresh_rowwise_fp8_cache"):
                    continue
                seen.add(owner_id)
                owners.append(owner)

        for owner in owners:
            owner.refresh_rowwise_fp8_cache()

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

        # Allow overriding via attribute; default to X elements.
        CHUNK_ELEMS = getattr(self, "_foreach_chunk_threshold", 600_000_000)

        for group in self.param_groups:
            # Per-chunk accumulators
            main_params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []

            exp_avgs: list[torch.Tensor] = []  # always fp32
            exp_avg_sqs: list[torch.Tensor] = []  # always fp32

            exp_avgs_original: list[
                torch.Tensor
            ] = (
                []
            )  # if states_dtype is bf16, we need to keep a reference to the original bf16 tensors
            exp_avg_sqs_original: list[torch.Tensor] = []

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

            def reset_chunk_buffers():
                nonlocal main_params, grads, exp_avgs, exp_avg_sqs, steps_list, running_elems, exp_avgs_original, exp_avg_sqs_original
                # reset for next chunk
                main_params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                exp_avgs_original = []
                exp_avg_sqs_original = []
                steps_list = []
                running_elems = 0

            def maybe_copy_back_16bit_states():
                # foreach_adamw_step makes in place updates to exp_avgs and exp_avg_sqs which are in fp32
                # so if the fp32 states are copies of bf16 states, we need to copy them back
                # otherwies, they are the original fp32 states, no need to copy back
                nonlocal exp_avgs_original, exp_avg_sqs_original, exp_avgs, exp_avg_sqs
                if self.states_dtype == torch.bfloat16:
                    for i in range(len(exp_avgs)):
                        # copy back fp32 to original bf16 tensors
                        # exp_avgs_original[i].copy_(exp_avgs[i].to(torch.bfloat16))
                        # exp_avg_sqs_original[i].copy_(exp_avg_sqs[i].to(torch.bfloat16))
                        exp_avgs_original[i].copy_(exp_avgs[i])
                        exp_avg_sqs_original[i].copy_(exp_avg_sqs[i])

            for name, model_p in group["named_params"].items():
                if not model_p.requires_grad:
                    continue

                if self._param_uses_muon[name]:
                    flush_chunk()
                    maybe_copy_back_16bit_states()
                    reset_chunk_buffers()

                    main_state = self.states[f"{name}.main"]
                    momentum_state = self.states[f"{name}.muon_momentum"]

                    main_param_local = main_state.to_local()
                    grad_local = _to_local_tensor(self.main_grad[name]).float()
                    momentum_local = momentum_state.to_local()
                    step = self.states[f"{name}.step"].to_local()

                    momentum_weight = step_factor * (1.0 - group["muon_momentum"])
                    momentum_local.mul_(1.0 - momentum_weight)
                    momentum_local.add_(grad_local * momentum_weight)

                    if group["muon_nesterov"]:
                        local_pre_update = (
                            grad_local * (1.0 - group["muon_momentum"])
                            + momentum_local * group["muon_momentum"]
                        )
                    else:
                        local_pre_update = momentum_local

                    full_pre_update = self._gather_sharded_flat_tensor(local_pre_update, main_state)
                    full_pre_update = full_pre_update.reshape(model_p.shape)
                    full_ortho_update = _zeropower_via_newtonschulz(
                        full_pre_update,
                        group["muon_ns_coefficients"],
                        group["muon_ns_steps"],
                        group["muon_eps"],
                    )

                    local_ortho_update = self.narrow_tensor(
                        full_ortho_update.reshape(-1),
                        main_state.device_mesh,
                        main_state.placements,
                    )
                    adjusted_lr = _adjust_muon_lr(
                        group["lr"], group["muon_adjust_lr_fn"], model_p.shape
                    )

                    main_param_local.mul_(1.0 - step_factor * group["lr"] * group["weight_decay"])
                    main_param_local.add_(
                        local_ortho_update * (adjusted_lr * step_factor), alpha=-1.0
                    )
                    step.add_(step_factor)
                    continue

                # in adam step(), make everything local and fp32
                main_params.append(self.states[f"{name}.main"].to_local())
                grads.append(_to_local_tensor(self.main_grad[name]).float())
                if self.states_dtype == torch.bfloat16:
                    # new fp32 copy
                    exp_avgs.append(self.states[f"{name}.exp_avg"].to_local().to(torch.float32))
                    exp_avg_sqs.append(
                        self.states[f"{name}.exp_avg_sq"].to_local().to(torch.float32)
                    )

                    exp_avgs_original.append(self.states[f"{name}.exp_avg"].to_local())
                    exp_avg_sqs_original.append(self.states[f"{name}.exp_avg_sq"].to_local())
                else:
                    # original fp32
                    exp_avgs.append(self.states[f"{name}.exp_avg"].to_local())
                    exp_avg_sqs.append(self.states[f"{name}.exp_avg_sq"].to_local())
                steps_list.append(self.states[f"{name}.step"].to_local())

                running_elems += self.states[f"{name}.main"].to_local().numel()
                # Flush when we reach/exceed the threshold. It's OK to overshoot with the last add.
                if running_elems >= CHUNK_ELEMS:
                    flush_chunk()
                    maybe_copy_back_16bit_states()
                    reset_chunk_buffers()

            # Flush any tail chunk
            flush_chunk()
            maybe_copy_back_16bit_states()
            reset_chunk_buffers()

    def zero_grad(self, set_to_none=True):
        raise RuntimeError(
            "zero_grad should be called by the MoE TrainModule on the models directly now."
        )
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
                        p._main_grad_fp32 = None

    def unsharded_state_dict(self) -> dict:
        raise NotImplementedError("Removed function")

    def _install_optim_from_cpu_dtensor(self, main_sd, state1_sd, state2_sd, distribute_ep_func):
        raise NotImplementedError("Removed function")

    def _restore_rolling_stats(self, values: Any) -> List[torch.Tensor]:
        if values is None:
            log.info("No rolling stats found in checkpoint, skipping restore.")
            return []

        log.info("Restoring rolling stats from checkpoint ...")

        if isinstance(values, torch.Tensor):
            raw_values: List[Any]
            if values.ndim == 0:
                raw_values = [values]
            else:
                raw_values = [v for v in values.reshape(-1).unbind()]
        elif isinstance(values, (list, tuple)):
            raw_values = list(values)
        else:
            raw_values = [values]

        restored: List[torch.Tensor] = []
        for value in raw_values:
            if isinstance(value, torch.Tensor):
                tensor_value = value.detach().to(device=self.device, dtype=torch.float32)
                tensor_value = tensor_value.reshape(-1)[0]
            else:
                tensor_value = move_to_device(
                    torch.tensor(float(value), dtype=torch.float32), self.device
                )
            restored.append(tensor_value)

        return restored[-(self.rolling_interval_length + 1) :]

    def state_dict(self) -> dict:
        sd = {}
        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                all_suffixes = self._state_suffixes_for_param(name)
                if param_group["pg"] == "ep_dp":
                    for suffix in all_suffixes:
                        state_key = f"{name}.{suffix}"
                        live_state_dt = self.states[state_key]
                        if suffix != "step":
                            sd[state_key] = self._ep_dp_state_to_checkpoint(live_state_dt)

                            # Free the local shard storage while keeping DTensor metadata.
                            state_local = live_state_dt.to_local()
                            empty_local = torch.empty(
                                0, dtype=state_local.dtype, device=state_local.device
                            )
                            self.states[state_key] = DTensor.from_local(
                                empty_local,
                                device_mesh=live_state_dt.device_mesh,
                                placements=live_state_dt.placements,
                                shape=live_state_dt.shape,
                                stride=live_state_dt.stride(),
                                run_check=False,
                            )
                        else:  # "step"
                            sd[state_key] = live_state_dt

                else:  # DP tensor already in the right dtensor
                    for suffix in all_suffixes:
                        state_dt = self.states[f"{name}.{suffix}"]
                        sd[f"{name}.{suffix}"] = state_dt

        assert set(sd.keys()) == set(
            self.states.keys()
        ), f"State dict keys do not match live states: {set(sd.keys()) ^ set(self.states.keys())}"

        # Store rolling skip-step statistics as plain lists so they can be checkpointed as a single BYTE_IO entry.
        sd[self.LOSSES_STATE_DICT_KEY] = [float(v.detach().cpu().item()) for v in self._losses]
        sd[self.GRAD_NORMS_STATE_DICT_KEY] = [
            float(v.detach().cpu().item()) for v in self._grad_norms
        ]

        return sd

    def _ensure_local_state_storage(self, state_key: str) -> DTensor:
        state_dt = self.states[state_key]
        if state_dt.to_local().numel() != 0:
            return state_dt

        local_shape, _ = compute_local_shape_and_global_offset(
            state_dt.shape,
            state_dt.device_mesh,
            state_dt.placements,
        )
        local_stride = compute_local_stride(
            state_dt.stride(),
            state_dt.device_mesh,
            state_dt.placements,
        )
        new_local = torch.empty_strided(
            tuple(local_shape),
            tuple(local_stride),
            dtype=state_dt.dtype,
            device=self.device,
        )
        state_dt = DTensor.from_local(
            new_local,
            device_mesh=state_dt.device_mesh,
            placements=state_dt.placements,
            shape=state_dt.shape,
            stride=state_dt.stride(),
            run_check=False,
        )
        self.states[state_key] = state_dt
        return state_dt

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
        reset_optimizer_moments_on_load: Optional[bool] = None,
    ) -> None:
        # the loaded state dict is already distributed over the DP mesh,
        # here we need to convert the DP sharded tensors to EP_MP + EP_DP sharded
        if reset_optimizer_moments_on_load is None:
            reset_optimizer_moments_on_load = self.reset_optimizer_moments_on_load

        loaded_losses = state_dict.pop(self.LOSSES_STATE_DICT_KEY, None)
        loaded_grad_norms = state_dict.pop(self.GRAD_NORMS_STATE_DICT_KEY, None)

        for param_group in self.param_groups:
            for name, param in param_group["named_params"].items():
                if not param.requires_grad:
                    continue
                all_suffixes = self._state_suffixes_for_param(name)
                if reset_optimizer_moments_on_load:
                    for suffix in self.MOMENT_STATE_SUFFIXES:
                        state_key = f"{name}.{suffix}"
                        if state_key in self.states:
                            state_dict.pop(state_key, None)
                            self._ensure_local_state_storage(state_key).to_local().zero_()

                if param_group["pg"] == "ep_dp":
                    for suffix in all_suffixes:
                        if reset_optimizer_moments_on_load and suffix in self.MOMENT_STATE_SUFFIXES:
                            continue
                        state_key = f"{name}.{suffix}"
                        state_dt = self.states[state_key]
                        if suffix != "step":
                            ckpt_state = (
                                self._load_moment_state_or_zero(state_dict, state_key)
                                if suffix in self.MOMENT_STATE_SUFFIXES
                                else state_dict.pop(state_key)
                            )
                            if ckpt_state is not None:
                                self._load_ep_dp_state_from_checkpoint(state_key, ckpt_state)
                        else:
                            ckpt_state = state_dict.pop(state_key, None)
                            if ckpt_state is not None:
                                state_dt.copy_(ckpt_state.full_tensor())
                else:
                    for suffix in all_suffixes:
                        if reset_optimizer_moments_on_load and suffix in self.MOMENT_STATE_SUFFIXES:
                            continue
                        state_key = f"{name}.{suffix}"
                        live_state = self.states[state_key]
                        if suffix in self.MOMENT_STATE_SUFFIXES:
                            ckpt_state = self._load_moment_state_or_zero(state_dict, state_key)
                            if ckpt_state is None:
                                continue
                        else:
                            ckpt_state = state_dict.pop(state_key, None)
                            if ckpt_state is None:
                                continue

                        if suffix == "step":
                            live_state.copy_(ckpt_state.full_tensor())
                        else:
                            ckpt_local = ckpt_state.to_local()
                            live_state = self._ensure_local_state_storage(state_key)
                            assert (
                                ckpt_state.shape == live_state.shape
                            ), f"Global shape mismatch {name}.{suffix}: {ckpt_state.shape} vs {live_state.shape}"
                            assert (
                                ckpt_local.shape == live_state.to_local().shape
                            ), f"Local shape mismatch {name}.{suffix}: {ckpt_local.shape} vs {live_state.to_local().shape}"
                            live_state.to_local().copy_(ckpt_local)

        self._losses = self._restore_rolling_stats(loaded_losses)
        self._grad_norms = self._restore_rolling_stats(loaded_grad_norms)

        return

    def _global_numel(self, tag: str) -> int:
        raise NotImplementedError()


@torch._dynamo.disable()
def coalesced_all_gather_flat(
    input_tensors: List[torch.Tensor],
    process_group: dist.ProcessGroup,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Coalesced all_gather for a list of 1-D tensors.

    Returns the gathered flat buffer as a `[world_size, total_elems]` tensor
    together with the per-input sizes and offsets inside the packed buffer.
    """
    if not input_tensors:
        raise ValueError("input_tensors must be non-empty")

    device = input_tensors[0].device
    input_dtype = input_tensors[0].dtype
    output_dtype = input_dtype if output_dtype is None else output_dtype
    for t in input_tensors:
        assert t.dim() == 1, "All input_tensors must be 1-D"
        assert t.device == device, "All input_tensors must be on the same device"
        assert t.dtype == input_dtype, "All input_tensors must have the same dtype"

    world_size = dist.get_world_size(process_group)
    sizes = [t.numel() for t in input_tensors]
    offsets: List[int] = []
    running = 0
    for size in sizes:
        offsets.append(running)
        running += size
    total_elems = running

    flat_local = torch.empty(total_elems, device=device, dtype=output_dtype)
    for t, off in zip(input_tensors, offsets):
        flat_local[off : off + t.numel()].copy_(t.view(-1))

    flat_global = torch.empty(world_size * total_elems, device=device, dtype=output_dtype)
    dist.all_gather_into_tensor(flat_global, flat_local, group=process_group)
    return flat_global.view(world_size, total_elems), sizes, offsets


@torch._dynamo.disable()
def coalesced_all_gather(
    input_tensors: List[torch.Tensor],
    process_group: dist.ProcessGroup,
) -> List[torch.Tensor]:
    """
    Coalesced all_gather for a list of 1-D tensors.

    Args:
        input_tensors: List of 1-D tensors. For a given index i, all ranks must
            have input_tensors[i] with the same numel, dtype, and device.
        process_group: The process group to use for all_gather (default: world group).

    Returns:
        A list of tensors, one per input tensor.
        For input_tensors[i] of shape [N_i], the output[i] has shape [world_size, N_i],
        where output[i][r] is the data from rank r.
    """
    if not input_tensors:
        return []

    flat_global, sizes, offsets = coalesced_all_gather_flat(input_tensors, process_group)

    # 3) Unpack into per-tensor gathered outputs
    gathered_outputs: List[torch.Tensor] = []
    for size, off in zip(sizes, offsets):
        # [world_size, size]
        gathered = flat_global[:, off : off + size].contiguous()
        gathered_outputs.append(gathered)

    return gathered_outputs
