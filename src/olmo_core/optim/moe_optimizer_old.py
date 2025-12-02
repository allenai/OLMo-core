import logging
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import (
    hook_with_zero_step,
    hook_with_zero_step_interleaved,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.tensor import DTensor, Shard
from torch.optim.optimizer import Optimizer

from olmo_core.utils import get_default_device, move_to_device

from ..config import Config, DType
from ..exceptions import OLMoConfigurationError
from ..train.train_module import TrainModule
from .adamw import adamw_step, foreach_adamw_step
from .config import INITIAL_LR_FIELD, LR_FIELD, OptimConfig, OptimGroupOverride
from .skip_step_optimizer import SkipStepOptimizer

log = logging.getLogger(__name__)

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

        return [
            {"params": [all_params[param_name] for param_name in go.params], **go.opts}
            for go in group_overrides
            if len(go.params) > 0
        ]

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
        dp_param_order: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
        ep_param_order: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
        for part in model_parts:
            for name, p in part.named_parameters():
                if not p.requires_grad:
                    continue
                if id(p) in ep_param_ids:
                    ep_param_order[name] = p
                else:
                    dp_param_order[name] = p

        # Build param groups for the two PGs by filtering.
        dp_groups = self.build_groups(
            model_parts, strict=strict, param_filter=lambda p: id(p) not in ep_param_ids
        )
        for g in dp_groups:
            g["__pg_tag__"] = "dp"  # type: ignore

        ep_groups = self.build_groups(
            model_parts, strict=strict, param_filter=lambda p: id(p) in ep_param_ids
        )
        for g in ep_groups:
            g["__pg_tag__"] = "ep_dp"  # type: ignore

        # Concatenate, ensuring the "default" groups remain first in each partition (already ensured by build_groups()).
        all_groups: List[Dict[str, Any]] = list(dp_groups) + list(ep_groups)  # type: ignore

        optim: torch.optim.Optimizer = self.optimizer()(
            all_groups,
            param_order_dp=dp_param_order,
            param_order_ep=ep_param_order,
            dp_group=getattr(train_module, "dp_group", None),
            ep_dp_group=getattr(train_module, "ep_dp_group", None),
            world_mesh=train_module.world_mesh,
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
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx}, {len(group['params'])} parameter(s)")

        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        # Register hook to reset fixed fields after loading a checkpoint.
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(fixed_fields)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return optim


class MoEFusedV2Optimizer(Optimizer):
    def __init__(
        self,
        params,
        world_mesh: Dict[str, Optional[DeviceMesh]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        max_grad_norm: float = 1.0,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        # foreach: bool = False,
        # --- new args for sharding across multiple PGs ---
        dp_group: Optional[ProcessGroup] = None,
        ep_dp_group: Optional[ProcessGroup] = None,
        param_order_dp: Optional[OrderedDict[str, torch.nn.Parameter]] = None,
        param_order_ep: Optional[OrderedDict[str, torch.nn.Parameter]] = None,
        broadcast_bucket_mb: int = 32,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

        # for print info
        self._model_param_sz = 0
        for param_group in params:
            self._model_param_sz += sum(p.numel() * p.element_size() for p in param_group["params"])

        # ---- Sharding context (DP and EP-DP) ----
        self._dp_group: Optional[ProcessGroup] = dp_group
        self._ep_dp_group: Optional[ProcessGroup] = ep_dp_group
        assert param_order_dp is not None, "param_order_dp must be provided"
        assert param_order_ep is not None, "param_order_ep must be provided"
        self._param_order_dp: OrderedDict[str, torch.nn.Parameter] = param_order_dp
        self._param_order_ep: OrderedDict[str, torch.nn.Parameter] = param_order_ep
        self._broadcast_bucket_bytes: int = int(broadcast_bucket_mb * 1024 * 1024)
        self._enable_sharding: bool = (
            self._dp_group is not None and len(self._param_order_dp) > 0
        ) or (self._ep_dp_group is not None and len(self._param_order_ep) > 0)
        assert world_mesh["dense"] is not None, "DP mesh must be provided"
        self.dp_mesh: DeviceMesh = world_mesh["dense"]  # ('pp', 'dp')
        self.moe_mesh = world_mesh["moe"]  # ('pp', 'ep_dp', 'ep_mp')
        self.ep_dp_mesh = self.moe_mesh["ep_dp"] if self.moe_mesh else None
        # Tag each param_group with its PG tag if present (default to 'dp' for backwards compat).
        for g in self.param_groups:
            tag = g.get("__pg_tag__")
            if tag not in ("dp", "ep_dp"):
                g["__pg_tag__"] = "dp"

        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = None
        self.max_grad_norm = max_grad_norm
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        # self.foreach = foreach
        self._step_skipped: Optional[torch.Tensor] = None

        params = cast(Iterable[Dict[str, Any]], params)

        device = None
        for param_group in params:
            for i, param in enumerate(param_group["params"]):
                if not param.requires_grad:
                    continue
                if device is None:
                    device = param.device
                else:
                    assert device == param.device, "Inconsistent device found"
                # float16 params:
                assert param.type() in [
                    "torch.cuda.HalfTensor",
                    "torch.cuda.BFloat16Tensor",
                ], "Only support 16 bit params. Received {}".format(param.type())

        self._owned_views: List[MoEFusedV2Optimizer._OwnedView] = []

        self._segments_by_rank: Dict[str, List[List[Tuple[str, torch.nn.Parameter, int, int]]]] = {}

        # A mapping from PG tag to param ranges (start, end) in global concatenation
        # eg,
        # self.param_ranges_by_tag = {
        #   'dp': OrderedDict{param_name1: (start, end), param_name2: (start, end)},
        #   'ep_dp': OrderedDict{param_name3: (start, end), param_name4: (start, end)},
        # }
        self.param_ranges_by_tag: Dict[str, OrderedDict[str, Tuple[int, int]]] = {}

        new_param_groups: List[Dict[str, Any]] = []
        self._flat_main_grad_buf: Dict[str, Optional[torch.Tensor]] = {}
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            ws, r = self._world_and_rank(pg)

            # Build per-rank segments list (deterministic across ranks)
            # order, offsets, total = self._build_concat_layout(tag)
            segs_by_rank = self._build_segments_by_rank(tag)
            self._segments_by_rank[tag] = segs_by_rank

            my_segs = segs_by_rank[r]
            if not my_segs:
                continue

            # Allocate flat main/state shards (fp32) for the sum of my segment lengths
            owned_numel = sum(seg_len for (_, _, _, seg_len) in my_segs)
            # device = order[0].device
            flat_main = torch.empty(owned_numel, dtype=torch.float32, device=device)
            flat_exp_avg = torch.zeros_like(flat_main, dtype=torch.float32)
            flat_exp_avg_sq = torch.zeros_like(flat_main, dtype=torch.float32)

            # Create 1-D views per segment
            offset = 0
            for model_param_name, model_param, seg_start, seg_len in my_segs:
                main_param_view = flat_main.narrow(0, offset, seg_len)
                exp_avg_view = flat_exp_avg.narrow(0, offset, seg_len)
                exp_avg_sq_view = flat_exp_avg_sq.narrow(0, offset, seg_len)

                # Initialize main from model param slice
                main_param_view.data.copy_(
                    model_param.data.view(-1)[seg_start : seg_start + seg_len].float()
                )

                self._owned_views.append(
                    MoEFusedV2Optimizer._OwnedView(
                        model_param_name=model_param_name,
                        model_param=model_param,
                        param_start=seg_start,
                        length=seg_len,
                        main_param_view=main_param_view,
                        exp_avg_view=exp_avg_view,
                        exp_avg_sq_view=exp_avg_sq_view,
                        pg_tag=tag,
                    )
                )
                offset += seg_len

            # Hold references by tag for broadcasts
            setattr(self, f"_flat_main_{tag}", flat_main)
            setattr(self, f"_flat_exp_avg_{tag}", flat_exp_avg)
            setattr(self, f"_flat_exp_avg_sq_{tag}", flat_exp_avg_sq)
            self._flat_main_grad_buf[tag] = None

        def _param_in_list(param, lst):
            return id(param) in (id(p) for p in lst)

        def _filter_owned_main_views_whose_model_params_are_in(params):
            # Iterate all owned views and return those whose model_param is in params
            rst = []
            for ov in self._owned_views:
                if _param_in_list(ov.model_param, params):
                    rst.append(ov.main_param_view)
            return rst

        for old_param_group in self.param_groups:
            owned_params = _filter_owned_main_views_whose_model_params_are_in(
                old_param_group["params"]
            )
            new_param_group = {
                "params": owned_params,
                "__pg_tag__": old_param_group["__pg_tag__"],
                "lr": old_param_group["lr"],
                "betas": old_param_group["betas"],
                "eps": old_param_group["eps"],
                "weight_decay": old_param_group["weight_decay"],
            }
            if owned_params:
                new_param_groups.append(new_param_group)

        log_str = "\n"
        log_str += f"Old param group:\n"
        log_str += _str_paramt(self.param_groups)

        log_str += f"New param group:\n"
        log_str += _str_paramt(new_param_groups)

        print(log_str)
        ##########

        # update param groups with only OWNED ones
        self.param_groups = new_param_groups

        # init optimizer per-parameter states
        # the states are a view into the flat buffers
        for group in self.param_groups:
            tag = group["__pg_tag__"]
            # flat_exp_avg = getattr(self, f"_flat_exp_avg_{tag}")
            # flat_exp_avg_sq = getattr(self, f"_flat_exp_avg_sq_{tag}")

            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    # find the owned view for this main param
                    ov = next(ov for ov in self._owned_views if ov.main_param_view is p)
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = ov.exp_avg_view
                    state["exp_avg_sq"] = ov.exp_avg_sq_view

        self.print_memory_summary()
        self._check_model_param_main_param_the_same()

    def offload_optimizer_states(self):
        raise NotImplementedError()
        # Offload optimizer states to CPU to save GPU memory
        dbg_mem1 = torch.cuda.memory_allocated() / (1024**3)
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            flat_main = getattr(self, f"_flat_main_{tag}")
            flat_exp_avg = getattr(self, f"_flat_exp_avg_{tag}")
            flat_exp_avg_sq = getattr(self, f"_flat_exp_avg_sq_{tag}")
            flat_main.data = flat_main.data.to("cpu")
            flat_exp_avg.data = flat_exp_avg.data.to("cpu")
            flat_exp_avg_sq.data = flat_exp_avg_sq.data.to("cpu")

        dbg_mem2 = torch.cuda.memory_allocated() / (1024**3)
        pass

    def reload_optimizer_states_to_device(
        self,
    ):
        raise NotImplementedError()
        # Reload optimizer states to the given device
        for ov in self._owned_views:
            ov.exp_avg_view.data = ov.exp_avg_view.data.to(self.device)
            ov.exp_avg_sq_view.data = ov.exp_avg_sq_view.data.to(self.device)
            ov.main_param_view.data = ov.main_param_view.data.to(self.device)

    def print_memory_summary(self):
        main_param_bytes = 0
        states_bytes = 0
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None:
                continue
            flat_main = getattr(self, f"_flat_main_{tag}")
            main_param_bytes += flat_main.numel() * flat_main.element_size()

            flat_exp_avg = getattr(self, f"_flat_exp_avg_{tag}")
            states_bytes += flat_exp_avg.numel() * flat_exp_avg.element_size()

            flat_exp_avg_sq = getattr(self, f"_flat_exp_avg_sq_{tag}")
            states_bytes += flat_exp_avg_sq.numel() * flat_exp_avg_sq.element_size()

        main_grad_bytes = main_param_bytes
        print(f"Model param memory usage: {self._model_param_sz / 1024**3:.2f} GB - 1/(EP)")
        print(f"Model grad memory usage: {self._model_param_sz / 1024**3:.2f} GB - 1/(EP)")
        print(f"Main param memory usage: {main_param_bytes / 1024**3:.2f} GB - 1/(DP)")
        print(f"Main grad memory usage: {main_grad_bytes / 1024**3:.2f} GB - 1/(DP)")
        print(f"States memory usage: {states_bytes / 1024**3:.2f} GB - 1/(DP)")

        total_mem = self._model_param_sz + self._model_param_sz + main_param_bytes + states_bytes
        peak_mem = total_mem + main_grad_bytes
        print(
            f"----\nTotal estimated static memory usage: {total_mem / 1024**3:.2f} GB (peak {peak_mem / 1024**3:.2f} GB)"
        )

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for group in self.param_groups:
                for p in group["params"]:
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

    @overload  # make pylance happy
    def step(self, closure: None = ...) -> None:
        ...

    @overload  # make pylance happy
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def set_reduce_scatter_grads(self, enabled: bool = True):
        self._use_reduce_scatter_grads = enabled

    def _clip_grad(self) -> torch.Tensor:
        global_grads = []

        # dp grad
        flat_main_grad_buf_dp = self._flat_main_grad_buf["dp"]
        if flat_main_grad_buf_dp is not None:
            mesh_shape = self.dp_mesh.shape
            norm_dp = flat_main_grad_buf_dp.norm(p=2.0).unsqueeze(0)
            flat_main_grad_buf_dp_dt = DTensor.from_local(
                # flat_main_grad_buf_dp,
                norm_dp,
                device_mesh=self.dp_mesh["dp"],
                placements=[Shard(0)],
            )
            global_grads.append(flat_main_grad_buf_dp_dt)

        flat_main_grad_buf_ep_dp = None
        if self.moe_mesh is not None:
            flat_main_grad_buf_ep_dp = self._flat_main_grad_buf["ep_dp"]
            if flat_main_grad_buf_ep_dp is not None:
                mesh_shape = self.dp_mesh.shape
                norm_ep_dp = flat_main_grad_buf_ep_dp.norm(p=2.0).unsqueeze(0)
                flat_main_grad_buf_ep_dp_dt = DTensor.from_local(
                    # flat_main_grad_buf_ep_dp.unsqueeze(0),
                    norm_ep_dp,
                    # device_mesh=self.moe_mesh['ep_dp', 'ep_mp'],
                    # placements=[Shard(1), Shard(0)],
                    device_mesh=self.dp_mesh["dp"],  # NOTE: EP_DP+EP_MP = DP, let's just reuse
                    placements=[Shard(0)],
                )
                global_grads.append(flat_main_grad_buf_ep_dp_dt)

        # NOTE: aten._foreach_norm.Scalar: DTensor does not support cross-mesh operation yet!
        # TODO: early PP stages have large grad norm, need to investigate
        total_grad_norm = nn.utils.get_total_norm(
            global_grads, norm_type=2.0, error_if_nonfinite=False
        )
        total_grad_norm = cast(DTensor, total_grad_norm).full_tensor()
        # alternative
        # n0 = torch.norm(global_grads[0], p=2.0)
        # n0 = cast(DTensor, n0).full_tensor()
        # n1 = torch.norm(global_grads[1], p=2.0)
        # n1 = cast(DTensor, n1).full_tensor()
        # total_grad_norm = (n0.square() + n1.square()).sqrt()

        ###
        # global_grads = []
        # flat_main_grad_buf_dp = self._flat_main_grad_buf['dp']
        # flat_main_grad_buf_ep_dp = self._flat_main_grad_buf['ep_dp']
        # flat_main_grad_buf_dp_dt = DTensor.from_local(
        #     flat_main_grad_buf_dp,
        #     device_mesh=self.dp_mesh,
        #     placements=[Shard(0)],
        # )
        # global_grads.append(flat_main_grad_buf_dp_dt)

        # flat_main_grad_buf_ep_dp = self._flat_main_grad_buf['ep_dp']
        # flat_main_grad_buf_ep_dp_dt = DTensor.from_local(
        #     flat_main_grad_buf_ep_dp,
        #     device_mesh=self.dp_mesh,
        #     placements=[Shard(0)],
        # )
        # global_grads.append(flat_main_grad_buf_ep_dp_dt)
        # total_grad_norm2 = nn.utils.get_total_norm(global_grads, norm_type=2.0, error_if_nonfinite=False)
        ###

        # dbg0 = global_grads[0].full_tensor()
        # dbg1 = global_grads[1].full_tensor()
        # total_grad_norm1 = nn.utils.get_total_norm([dbg0, dbg1], norm_type=2.0, error_if_nonfinite=False)

        # If pipeline parallelism
        assert self.dp_mesh.mesh_dim_names is not None
        if "pp" in self.dp_mesh.mesh_dim_names:
            total_grad_norm = total_grad_norm.square()
            dist.all_reduce(
                total_grad_norm, op=dist.ReduceOp.SUM, group=self.dp_mesh["pp"].get_group()
            )
            total_grad_norm = total_grad_norm.sqrt()

        clip_coef = self.max_grad_norm / (total_grad_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0).to(global_grads[0].device)
        # torch._foreach_mul_(global_grads, clip_coef_clamped)
        if flat_main_grad_buf_dp is not None:
            flat_main_grad_buf_dp.mul_(clip_coef_clamped)
        if flat_main_grad_buf_ep_dp is not None:
            flat_main_grad_buf_ep_dp.mul_(clip_coef_clamped)

        return total_grad_norm

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        dbg_mem_before_cp1 = torch.cuda.memory_allocated() / 1024**3
        if getattr(self, "_use_reduce_scatter_grads", True):
            # Precondition: DDP model did not all-reduce grads, grads on dp ranks different now
            # the optimizer has sharded main param + states in fp32
            # now call reduce scatter to collect averaged grads from dp ranks
            # directly into the owned main param views
            dbg_mem_before_rs = torch.cuda.memory_allocated() / 1024**3
            dbg_mem_peak_before_rs = torch.cuda.max_memory_allocated() / 1024**3
            self._reduce_scatter_model_grads_chunked()
            dbg_mem_after_rs = torch.cuda.memory_allocated() / 1024**3
            dbg_mem_peak_after_rs = torch.cuda.max_memory_allocated() / 1024**3

        else:
            # Precondition: DDP model called all-reduce grads, bf16 model grads on dp ranks are the same
            # only copy the model grads to OWNED main grads
            self._copy_owned_model_grads_to_main_grads()

        total_grad_norm = self._clip_grad()
        self.latest_grad_norm = total_grad_norm
        dbg_mem_before_step = torch.cuda.memory_allocated() / 1024**3
        self._step_foreach(closure)
        dbg_mem_after_step = torch.cuda.memory_allocated() / 1024**3
        self._dealloc_main_grad()
        dbg_mem_before_cp2 = torch.cuda.memory_allocated() / 1024**3

        # 1) owners write back to local model params
        self._copy_owned_main_to_model_bf16()
        # 2) broadcast owned shards to replicas within each PG
        # self._broadcast_updated_model_params()
        # or 2b) use all-gather
        self._allgather_updated_model_params()

        return None

    class _OwnedView(NamedTuple):
        # A contiguous slice (in flattened space) of a model parameter owned by this rank.
        model_param_name: str
        model_param: torch.nn.Parameter  # replicated bf16/fp16 param
        param_start: int  # start index into model_param.view(-1)
        length: int  # number of elements in this slice
        main_param_view: torch.Tensor  # 1-D fp32 view into our flat main shard
        exp_avg_view: torch.Tensor  # 1-D fp32 view into our flat exp_avg shard
        exp_avg_sq_view: torch.Tensor  # 1-D fp32 view into our flat exp_avg_sq shard
        pg_tag: str  # "dp" or "ep_dp"

        def __repr__(self) -> str:
            return (
                f"_OwnedView(pg_tag={self.pg_tag}, model_param={self.model_param.shape}, param_start={self.param_start}, "
                f"length={self.length}, main_param_view={self.main_param_view.shape}, "
                f"exp_avg_view={self.exp_avg_view.shape}, exp_avg_sq_view={self.exp_avg_sq_view.shape})"
            )

    def _pg_and_order_for_tag(self, tag: str):
        if tag == "dp":
            return self._dp_group, self._param_order_dp
        elif tag == "ep_dp":
            return self._ep_dp_group, self._param_order_ep
        else:
            raise RuntimeError(f"Unknown pg tag: {tag}")

    def _world_and_rank(self, pg: Optional[ProcessGroup]) -> Tuple[int, int]:
        if pg is None:
            return 1, 0
        return dist.get_world_size(pg), dist.get_rank(pg)

    def _owner_for_index(self, index: int, world_size: int) -> int:
        # Simple round-robin owner assignment by (global) param index.
        return index % max(1, world_size)

    # def _iter_group_params_with_tag(self, tag: str) -> List[torch.nn.Parameter]:
    #     out = OrderedDict()
    #     # Respect the stable global order (by name) computed in Config.build()
    #     _, order = self._pg_and_order_for_tag(tag)
    #     if order:
    #         out.update(order)
    #     else:
    #         raise RuntimeError(f"No parameter order found for tag '{tag}'")
    #         # Fallback: walk current param_groups tagged with 'tag'
    #         for g in self.param_groups:
    #             if g.get("__pg_tag__") == tag:
    #                 for p in g["params"]:
    #                     out.append(p if isinstance(p, torch.nn.Parameter) else p)
    #     return out

    def _build_concat_layout(self, tag: str):
        """
        Build a stable concatenation layout for all parameters in `tag`.
        Returns (order, offsets, total_numel) where:
        - order: OrderedDict[str, param] in stable order
        - offsets: List[int] prefix starts for each param in flattened concatenation
        - total_numel: int total elements over all params
        """
        pg, order = self._pg_and_order_for_tag(tag)
        if order is None or len(order) == 0:
            return OrderedDict(), [], 0
        offsets: List[int] = []
        total = 0
        for n, p in order.items():
            offsets.append(total)
            total += int(p.numel())
        return order, offsets, total

    @staticmethod
    def _ceil_to_multiple(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    def _compute_shard_ranges(self, total_numel: int, ws: int) -> List[Tuple[int, int]]:
        """
        Split [0, padded_total) into `ws` equal shards. Return list of (start, end).
        """
        if ws <= 1:
            return [(0, total_numel)]
        padded = self._ceil_to_multiple(total_numel, ws)
        shard = padded // ws
        return [(r * shard, (r + 1) * shard) for r in range(ws)]

    def _intersect(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        return (s, e) if e > s else (0, 0)

    def _build_segments_by_rank(self, tag: str):
        """
        For the given tag ('dp' or 'ep_dp'), compute, for every rank in that PG,
        the list of contiguous segments (param, start_in_param, length) that fall
        inside that rank's equal-size shard of the globally concatenated parameter space.

        Returns:
        segments_by_rank: List[List[Tuple[param, start, length]]], length = ws
        """
        pg, order = self._pg_and_order_for_tag(tag)
        if pg is None or not order:
            return []
        ws, _ = self._world_and_rank(pg)
        order, offsets, total = self._build_concat_layout(tag)
        if total == 0:
            return [[] for _ in range(ws)]

        shard_ranges = self._compute_shard_ranges(total, ws)

        # Param ranges in global (concatenated) coordinates.
        param_ranges: List[Tuple[int, int]] = []
        for i, (n, p) in enumerate(order.items()):
            start = offsets[i]
            end = start + int(p.numel())
            param_ranges.append((start, end))
        self.param_ranges_by_tag[tag] = OrderedDict(zip(order.keys(), param_ranges))

        segments_by_rank: List[List[Tuple[str, torch.nn.Parameter, int, int]]] = [
            [] for _ in range(ws)
        ]
        # segments_by_rank[rank]     = List of (param_name, param, start_in_param, length)
        for rank in range(ws):
            s_start, s_end = shard_ranges[
                rank
            ]  # marks the start/end of this rank's shard in global coords
            for i, (n, p) in enumerate(order.items()):
                p_start, p_end = param_ranges[i]
                inter = self._intersect((s_start, s_end), (p_start, p_end))
                if inter == (0, 0):
                    continue
                seg_start_in_param = inter[0] - p_start
                seg_len = inter[1] - inter[0]
                if seg_len > 0:
                    segments_by_rank[rank].append((n, p, seg_start_in_param, seg_len))
        return segments_by_rank

    def _reduce_scatter_model_grads(self) -> None:
        """
        Pack bf16/fp16 model grads in the stable concat order per PG, pad to ws multiple,
        and perform reduce-scatter (SUM) to obtain the local shard of grads. Then scale by 1/ws
        and place into each owned main view's .grad (fp32).
        """
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            ws, _ = self._world_and_rank(pg)
            # if ws == 1:
            #     # No communication needed; fall back to owned-copy.
            #     self._copy_owned_model_grads_to_main_grads()
            #     continue

            order, offsets, total = self._build_concat_layout(tag)
            if total == 0:
                continue
            device = next(iter(order.values())).device

            grad_dtype = torch.float32  # force fp32 RS

            padded_total = self._ceil_to_multiple(total, ws)
            shard = padded_total // ws

            # Pack grads into flat buffer
            send_buf = torch.zeros(padded_total, dtype=grad_dtype, device=device)
            for i, (n, p) in enumerate(order.items()):
                if p.grad is None:
                    continue

                dst = send_buf.narrow(0, offsets[i], p.numel())
                src = p.grad.detach().view(-1)
                dst.copy_(src)  # may auto-cast on some builds
                # dst.copy_(src.to(dst.dtype)) # fallback: small, per-param temp only

                p.grad = None  # free bf16/fp16 model grad

            # Reduce-scatter -> local shard
            # TODO: this send buf is huge, consider using a few small buckets
            flat_main_grad_buf = torch.empty(shard, dtype=grad_dtype, device=device)
            self._flat_main_grad_buf[tag] = flat_main_grad_buf  # record for grad clip
            if ws == 1:
                dist.reduce_scatter_tensor(
                    flat_main_grad_buf, send_buf, group=pg, op=dist.ReduceOp.AVG
                )
            else:
                dist.reduce_scatter_tensor(
                    flat_main_grad_buf, send_buf, group=pg, op=dist.ReduceOp.AVG
                )

            # Point owned fp32 main.grad as views into the buffer
            off = 0
            for ov in self._owned_views:
                if ov.pg_tag != tag:
                    continue
                n = ov.length
                main_param_grad = flat_main_grad_buf.narrow(0, off, n)

                ov.main_param_view.grad = main_param_grad

                off += n

            assert off == flat_main_grad_buf.size(0), "Size mismatch: part of the buffer not used"

    def _reduce_scatter_model_grads_chunked(self) -> None:
        """
        Same semantics as _reduce_scatter_model_grads, but stream the send buffer
        in chunks. Each flush packs data rank-major:
            [r0_block | r1_block | ... | r(ws-1)_block]
        so that reduce_scatter's even split delivers the correct next K elements
        of each rank's shard.
        """

        for tag in ("ep_dp", "dp"):
            pg, order_list = self._pg_and_order_for_tag(tag)
            if pg is None or not order_list:
                continue
            ws, my_rank = self._world_and_rank(pg)

            # Layout info / sizes
            order_list, offsets, total = self._build_concat_layout(tag)
            if total == 0:
                continue
            device = next(iter(order_list.values())).device
            grad_dtype = torch.float32  # match the non-chunked path
            padded_total = self._ceil_to_multiple(total, ws)
            shard = padded_total // ws

            # Output shard buffer for this rank
            flat_main_grad_buf = torch.empty(shard, dtype=grad_dtype, device=device)
            self._flat_main_grad_buf[tag] = flat_main_grad_buf  # used later (e.g., grad clipping)

            # Build (or reuse) per-rank segment lists that define each rank's shard
            segs_by_rank = self._segments_by_rank.get(tag)
            # if not segs_by_rank:
            #     segs_by_rank = self._build_segments_by_rank(tag)
            #     self._segments_by_rank[tag] = segs_by_rank

            # Cursors over the per-rank segment lists
            cursor_idx = [0 for _ in range(ws)]  # which segment tuple we are on
            cursor_off = [0 for _ in range(ws)]  # how much consumed inside that segment

            # Helper: copy up to `need` elems destined for rank r into dst at dst_off
            def _fill_rank_piece(r: int, need: int, dst: torch.Tensor, dst_off: int) -> None:
                wrote = 0
                assert segs_by_rank is not None

                segs = segs_by_rank[r]
                while wrote < need:
                    if cursor_idx[r] >= len(segs):
                        # No more real elements in this rank's shard: pad with zeros
                        remain = need - wrote
                        dst.narrow(0, dst_off + wrote, remain).zero_()
                        wrote += remain
                        break
                    mp_name, p, s0, seg_len = segs[cursor_idx[r]]
                    pos = cursor_off[r]
                    take = min(seg_len - pos, need - wrote)
                    if take > 0:
                        assert (
                            p.grad is not None
                        ), f"Param grad is None for param {p} on rank {my_rank} for tag {tag}"
                        src = p.grad.detach().view(-1).narrow(0, s0 + pos, take)
                        dst.narrow(0, dst_off + wrote, take).copy_(src)
                        wrote += take
                        pos += take
                    # advance or move to next segment
                    if pos == seg_len:
                        cursor_idx[r] += 1
                        cursor_off[r] = 0
                    else:
                        cursor_off[r] = pos

            # Choose chunk capacity (multiple of ws) and buffers
            chunk_budget_bytes = getattr(self, "_rs_chunk_bytes", 1 << 30)  # ~1 GiB by default
            elem_bytes = torch.tensor([], dtype=grad_dtype).element_size()
            chunk_capacity = max(
                ws, (chunk_budget_bytes // elem_bytes) // ws * ws
            )  # divisible by ws
            per_rank_cap = chunk_capacity // ws
            send_chunk = torch.empty(chunk_capacity, dtype=grad_dtype, device=device)

            # Stream: each flush advances this rank's shard by out_elems
            shard_cursor = 0
            while shard_cursor < shard:
                out_elems = min(per_rank_cap, shard - shard_cursor)

                # rank-major packing
                for r in range(ws):
                    _fill_rank_piece(r, out_elems, send_chunk, r * out_elems)

                # reduce_scatter on the packed chunk
                out_view = flat_main_grad_buf.narrow(0, shard_cursor, out_elems)
                dist.reduce_scatter_tensor(
                    out_view,
                    send_chunk.narrow(0, 0, out_elems * ws),
                    group=pg,
                    op=dist.ReduceOp.AVG,
                )
                shard_cursor += out_elems

            # Now it's safe to clear the low-precision model grads
            for n, p in order_list.items():
                p.grad = None

            # Point owned fp32 main.grad as views into the shard buffer
            off = 0
            for ov in self._owned_views:
                if ov.pg_tag != tag:
                    continue
                n = ov.length
                ov.main_param_view.grad = flat_main_grad_buf.narrow(0, off, n)
                off += n
            assert off == flat_main_grad_buf.numel(), "Size mismatch: part of the buffer not used"

    def _copy_owned_model_grads_to_main_grads(self):
        """
        Copy bf16/fp16 model grads into fp32 main.grad for OWNED segments only.
        Assumes DDP has already allreduced grads in-place on model params.
        """

        for ov in self._owned_views:
            mp = ov.model_param
            if mp.grad is None:
                continue
            src = mp.grad.detach().view(-1).narrow(0, ov.param_start, ov.length)
            if ov.main_param_view.grad is None:
                ov.main_param_view.grad = torch.empty_like(ov.main_param_view, dtype=torch.float32)
            ov.main_param_view.grad.copy_(src.float())

    def _copy_owned_main_to_model_bf16(self):
        """
        Copy updated fp32 main shard views back into local bf16/fp16 model params for OWNED segments only.
        """
        for ov in self._owned_views:
            dst = ov.model_param.data.view(-1).narrow(0, ov.param_start, ov.length)
            dst.copy_(ov.main_param_view.data.to(dtype=ov.model_param.dtype))

    def _allgather_updated_model_params(self):
        """
        param sync: each rank packs its UPDATED bf16/fp16 model param shard
        (equal size) and calls all_gather_into_tensor to materialize the full params,
        then unpacks into the live tensors. Padding is ignored.
        """

        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            ws, r = self._world_and_rank(pg)
            if ws == 1:
                continue

            order_list, offsets, total = self._build_concat_layout(tag)
            if total == 0:
                continue
            padded_total = self._ceil_to_multiple(total, ws)
            shard = padded_total // ws

            segs_by_rank = self._segments_by_rank.get(tag, [])
            if not segs_by_rank:
                assert False, "Allgather called before segments_by_rank built?"
                # segs_by_rank = self._build_segments_by_rank(tag)
                # self._segments_by_rank[tag] = segs_by_rank

            my_segs = segs_by_rank[r]
            device = next(iter(order_list.values())).device
            dtype = next(iter(order_list.values())).dtype

            # Pack my shard
            local_shard = torch.empty(shard, dtype=dtype, device=device)
            off = 0
            for mp_name, mp, start, length in my_segs:
                view = mp.data.view(-1).narrow(0, start, length)
                local_shard.narrow(0, off, length).copy_(view)
                off += length
            if off < shard:
                local_shard.narrow(0, off, shard - off).zero_()

            # All-gather all shards
            full_buf = torch.empty(ws * shard, dtype=dtype, device=device)  # TODO: big buffer
            if ws == 1:
                dist.all_gather_into_tensor(full_buf, local_shard, group=pg)
            else:
                dist.all_gather_into_tensor(full_buf, local_shard, group=pg)

            # Unpack into live params for each rank's shard
            for src in range(ws):
                segs = segs_by_rank[src]
                if not segs:
                    continue
                chunk = full_buf.narrow(0, src * shard, shard)
                off = 0
                for mp_name, mp, start, length in segs:
                    dst = mp.data.view(-1).narrow(0, start, length)
                    dst.copy_(chunk.narrow(0, off, length))
                    off += length

    def _broadcast_updated_model_params(self):
        """
        After owners update local bf16/fp16 model params from main shard views,
        broadcast each rank's OWNED segments (in bf16/fp16) so all replicas end
        with identical model params. We pack segments into buckets up to
        `self._broadcast_bucket_bytes`.
        """

        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            ws, r = self._world_and_rank(pg)
            if ws == 1:
                continue

            segs_by_rank = self._segments_by_rank.get(tag, [])
            if not segs_by_rank:
                continue

            # Iterate over each rank as source
            for src in range(ws):
                segs = segs_by_rank[src]
                if not segs:
                    continue

                # Bucketize by bytes
                current: List[Tuple[str, torch.nn.Parameter, int, int]] = []
                current_bytes = 0
                buckets: List[List[Tuple[str, torch.nn.Parameter, int, int]]] = []
                for mp_name, mp, start, length in segs:
                    nbytes = length * mp.element_size()
                    if current and current_bytes + nbytes > self._broadcast_bucket_bytes:
                        buckets.append(current)
                        current = []
                        current_bytes = 0
                    current.append((mp_name, mp, start, length))
                    current_bytes += nbytes
                if current:
                    buckets.append(current)

                # Send / recv each bucket
                for bucket in buckets:
                    numel = sum(length for (_, _, _, length) in bucket)
                    device = bucket[0][1].device
                    dtype = bucket[0][1].dtype

                    if r == src:
                        # Pack from our local model params (already updated)
                        send_buf = torch.empty(numel, dtype=dtype, device=device)
                        off = 0
                        for mp_name, mp, start, length in bucket:
                            view = mp.data.view(-1).narrow(0, start, length)
                            send_buf.narrow(0, off, length).copy_(view)
                            off += length
                        src_global = dist.get_global_rank(pg, src)
                        dist.broadcast(send_buf, src=src_global, group=pg)
                        del send_buf
                    else:
                        recv_buf = torch.empty(numel, dtype=dtype, device=device)
                        src_global = dist.get_global_rank(pg, src)
                        dist.broadcast(recv_buf, src=src_global, group=pg)
                        # Unpack into our model params
                        off = 0
                        for mp_name, mp, start, length in bucket:
                            dst = mp.data.view(-1).narrow(0, start, length)
                            dst.copy_(recv_buf.narrow(0, off, length))
                            off += length
                        del recv_buf

    def _dealloc_main_grad(self):
        for main_param_group in self.param_groups:
            for main_param in main_param_group["params"]:
                main_param.grad = None

        for tag in self._flat_main_grad_buf:
            self._flat_main_grad_buf[tag] = None

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

    def _step_foreach(self, closure=None) -> None:
        """Performs adamw step using foreach impl, limiting chunk size to reduce memory usage."""

        # TODO: step can be performed directly on the flat buffers instead of the views.

        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()  # type: ignore
        step_factor = cast(torch.Tensor, step_factor)
        self._step_skipped = 1 - step_factor

        # Allow overriding via attribute; default to X elements.
        CHUNK_ELEMS = getattr(self, "_foreach_chunk_threshold", 1000_000_000)

        for group in self.param_groups:
            # Per-chunk accumulators
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps_list: list[torch.Tensor] = []
            running_elems: int = 0

            def flush_chunk():
                nonlocal params_with_grad, grads, exp_avgs, exp_avg_sqs, steps_list, running_elems
                if not params_with_grad:
                    return
                foreach_adamw_step(
                    params_with_grad,
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
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                steps_list = []
                running_elems = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    raise RuntimeError("Optimizer state not initialized")

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps_list.append(state["step"])
                running_elems += p.numel()

                # Flush when we reach/exceed the threshold. It's OK to overshoot with the last add.
                if running_elems >= CHUNK_ELEMS:
                    flush_chunk()

            # Flush any tail chunk
            flush_chunk()

    # def _step_foreach(self, closure=None) -> None:
    #     if closure is not None:
    #         with torch.enable_grad():
    #             closure()

    #     step_factor = self.get_step_factor() # type: ignore
    #     step_factor = cast(torch.Tensor, step_factor)
    #     self._step_skipped = 1 - step_factor
    #     for group in self.param_groups:
    #         params_with_grad: list[torch.Tensor] = []
    #         grads: list[torch.Tensor] = []
    #         exp_avgs: list[torch.Tensor] = []
    #         exp_avg_sqs: list[torch.Tensor] = []
    #         steps_list = []  # create list outside loops

    #         for p in group["params"]:
    #             if p.grad is None:
    #                 continue

    #             state = self.state[p]
    #             if len(state) == 0:
    #                 raise RuntimeError("Optimizer state not initialized")
    #                 # state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
    #                 # state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
    #                 # state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

    #             params_with_grad.append(p)
    #             grads.append(p.grad)
    #             exp_avgs.append(state["exp_avg"])
    #             exp_avg_sqs.append(state["exp_avg_sq"])
    #             steps_list.append(state["step"])

    #         if not params_with_grad:
    #             continue  # nothing to update in this group

    #         foreach_adamw_step(
    #             params_with_grad,
    #             grads,
    #             exp_avgs,
    #             exp_avg_sqs,
    #             steps_list,
    #             lr=group["lr"],
    #             betas=group["betas"],
    #             eps=group["eps"],
    #             weight_decay=group["weight_decay"],
    #             step_factor=step_factor,
    #             step_increment_bugfix=True,
    #         )
    #         # grads_size = sum([g.numel() * g.element_size() for g in grads])/1024**3
    #         # exp_avgs_size = sum([ea.numel() * ea.element_size() for ea in exp_avgs])/1024**3
    #         # exp_avg_sqs_size = sum([eas.numel() * eas.element_size() for eas in exp_avg_sqs])/1024**3

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

    def _check_model_param_main_param_the_same(self):
        for ov in self._owned_views:
            param_start_offset = ov.param_start
            param_end_offset = param_start_offset + ov.length
            model_param = ov.model_param.data.view(-1)[param_start_offset:param_end_offset]  # bf16
            main_param_bf16 = ov.main_param_view.bfloat16()  # fp32 -> bf16
            if not torch.allclose(model_param, main_param_bf16, atol=1e-5):
                raise ValueError(
                    f"Model param {ov.model_param} and main param {ov.main_param_view} are not close"
                )

    def unsharded_state_dict(self) -> dict:
        # xxx.main_param, xxx.exp_avg, xxx.exp_avg_sq
        # all in fp32, unsharded
        unsharded_state_dict = OrderedDict()
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            N = self._global_numel(tag)
            ws, r = self._world_and_rank(pg)
            for buf_name in ("main", "exp_avg", "exp_avg_sq"):
                sharded_buf = getattr(self, f"_flat_{buf_name}_{tag}")  # Sharded over DP or EP_DP
                all_global_bufs_cpu = []
                global_buf = torch.empty_like(
                    sharded_buf
                )  # 1 rank's local shard size, receive other ranks' shards
                for src_rank in range(ws):
                    print(
                        f"Broadcasting/Receiving {buf_name} shard for tag {tag} from src_rank {src_rank} (total {ws})"
                    )
                    # all ranks gather the buffer owned by src_rank, and use it to construct state dict
                    if r == src_rank:
                        # this rank owns the shard, send its local shard
                        global_buf.copy_(sharded_buf)
                        dist.broadcast(
                            global_buf, src=dist.get_global_rank(pg, src_rank), group=pg
                        )  # send
                    else:
                        dist.broadcast(
                            global_buf, src=dist.get_global_rank(pg, src_rank), group=pg
                        )  # recv

                    all_global_bufs_cpu.append(
                        global_buf.cpu()
                    )  # put each rank's shard to cpu, and later concat them

                unsharded_buf_cpu = torch.cat(
                    all_global_bufs_cpu, dim=0
                )  # a unsharded buffer on cpu, all ranks in the pg have the same copy
                assert (
                    unsharded_buf_cpu.numel() == N
                ), f"Unsharded buffer size {unsharded_buf_cpu.numel()}, expected {N}"
                del all_global_bufs_cpu  # release cpu memory
                del global_buf  # release gpu memory
                for param_name, param in order.items():  # for each param in this tag
                    p_start, p_end = self.param_ranges_by_tag[tag][
                        param_name
                    ]  # get start/end in the global unsharded buffer
                    per_param_buf = unsharded_buf_cpu[p_start:p_end]
                    unsharded_state_dict[f"{param_name}.{buf_name}"] = per_param_buf.reshape(
                        param.shape
                    )

        print("Finished building unsharded optimizer state dict")
        return unsharded_state_dict

    def _install_optim_from_cpu_dtensor(self, main_sd, state1_sd, state2_sd, distribute_ep_func):
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue

            for param_name, param in order.items():  # for each param in this tag
                # p_start, p_end = self.param_ranges_by_tag[tag][param_name] # get start/end in the global unsharded buffer, but sharded over pp and ep_mp

                # first get the full unsharded tensor on CPU
                main = main_sd[param_name + ".main"].full_tensor()
                exp_avg = state1_sd[param_name + ".exp_avg"].full_tensor()
                exp_avg_sq = state2_sd[param_name + ".exp_avg_sq"].full_tensor()

                if self.ep_dp_mesh is not None and "routed_experts." in param_name:
                    # then shard them to each rank based on EP_MP
                    main = distribute_ep_func(main)
                    exp_avg = distribute_ep_func(exp_avg)
                    exp_avg_sq = distribute_ep_func(exp_avg_sq)

                    # get local shard for this rank
                    main = main.to_local()
                    exp_avg = exp_avg.to_local()
                    exp_avg_sq = exp_avg_sq.to_local()
                else:
                    main = main.to(self.device)
                    exp_avg = exp_avg.to(self.device)
                    exp_avg_sq = exp_avg_sq.to(self.device)

                # based on whether it's owned by this rank, copy to the live flat buffers
                for ov in self._owned_views:
                    if ov.pg_tag != tag:
                        continue
                    if ov.model_param is param:
                        # owned by this rank
                        owned_length = ov.length
                        param_start_offset = ov.param_start
                        # copy to the live flat buffers
                        ov.main_param_view.copy_(
                            main[param_start_offset : param_start_offset + owned_length]
                        )
                        ov.exp_avg_view.copy_(
                            exp_avg[param_start_offset : param_start_offset + owned_length]
                        )
                        ov.exp_avg_sq_view.copy_(
                            exp_avg_sq[param_start_offset : param_start_offset + owned_length]
                        )

        return

    def state_dict(self) -> dict:
        # ori_sd = super().state_dict()  # validate unsharded state
        sd = {"meta": {}}
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None or not order:
                continue
            N = self._global_numel(tag)
            exp_avg = getattr(self, f"_flat_exp_avg_{tag}")
            exp_avg_sq = getattr(self, f"_flat_exp_avg_sq_{tag}")
            main_param = getattr(self, f"_flat_main_{tag}")
            # Wrap local shards as DTensors with the *global* size
            sd[tag] = {
                "exp_avg": self._as_dtensor(exp_avg, tag, N),
                "exp_avg_sq": self._as_dtensor(exp_avg_sq, tag, N),
                "main_param": self._as_dtensor(main_param, tag, N),
            }
            # sd["meta"][f"order_{tag}_names"] = [n for n, _ in self._iter_named_params_for_tag(tag)]
        # Single global step (identical across ranks)
        tmp_state = list(self.state.values())[0]
        sd["step"] = tmp_state["step"].clone()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Load optimizer state saved by our custom `state_dict()`.

        Supported formats:
        1) **Custom flat format** (recommended):
           {
             "dp":    {"exp_avg": DTensor/Tensor, "exp_avg_sq": DTensor/Tensor},
             "ep_dp": {"exp_avg": DTensor/Tensor, "exp_avg_sq": DTensor/Tensor},
             "step":  scalar tensor or int
             "meta":  {... optional auxiliary info ...}
           }
           - If values are DTensors with Shard(0) on the correct mesh, we take `.to_local()`
             and copy directly into our live flat buffers.
           - If values are regular Tensors with *global* length, we slice out this rank's
             owned segments and copy into the live flat buffers.

        2) **PyTorch-style** dict with "state"/"param_groups":
           Delegated to `super().load_state_dict(...)`.

        Main fp32 weights are *not* saved/loaded; they are rebuilt from the model each run.
        """

        # TODO
        # Otherwise expect our custom per-tag flat representation.
        for tag in ("dp", "ep_dp"):
            pg, order = self._pg_and_order_for_tag(tag)
            if pg is None:
                continue
            if tag not in state_dict:
                # Not present: skip gracefully unless strict and this tag is active.
                flat_exp_avg = getattr(self, f"_flat_exp_avg_{tag}", None)
                flat_exp_avg_sq = getattr(self, f"_flat_exp_avg_sq_{tag}", None)
                if strict and (flat_exp_avg is not None or flat_exp_avg_sq is not None):
                    raise KeyError(f"Missing '{tag}' in optimizer state_dict")
                continue

            entry = state_dict[tag]

            exp_avg_src = entry["exp_avg"].to_local().squeeze(0)
            exp_avg_sq_src = entry["exp_avg_sq"].to_local().squeeze(0)
            main_param = entry["main_param"].to_local().squeeze(0)
            # Destination live flat buffers (must exist if tag is active)
            flat_exp_avg = getattr(self, f"_flat_exp_avg_{tag}")
            flat_exp_avg_sq = getattr(self, f"_flat_exp_avg_sq_{tag}")
            flat_main_param = getattr(self, f"_flat_main_{tag}")

            # flat_exp_avg.copy_(exp_avg_src)
            # flat_exp_avg_sq.copy_(exp_avg_sq_src)
            # flat_main_param.copy_(main_param)
            assert torch.equal(flat_exp_avg, exp_avg_src)
            assert torch.equal(flat_exp_avg_sq, exp_avg_sq_src)
            assert torch.equal(flat_main_param, main_param)

        loaded_step = state_dict.get("step", None)
        for state in self.state.values():
            state["step"].fill_(loaded_step)

        self._check_model_param_main_param_the_same()

        return None

    def _as_dtensor(self, local: torch.Tensor, tag: str, global_numel: int) -> DTensor:
        if tag == "dp":
            # optimizer is sharded over dp ranks
            return DTensor.from_local(
                local,
                self.dp_mesh["dp"],
                placements=[Shard(0)],
            )
        elif tag == "ep_dp":
            # optimizer is sharded over ep_dp ranks, which is previously already sharded over ep_mp ranks
            # (N, ) -> (1, N)
            assert self.moe_mesh is not None
            local = local.unsqueeze(0)
            return DTensor.from_local(
                local,
                self.moe_mesh["ep_dp", "ep_mp"],
                placements=[Shard(1), Shard(0)],
            )  # mp shards first dimension (0)
        else:
            raise ValueError(f"Unknown tag: {tag}")

    def _global_numel(self, tag: str) -> int:
        _, order = self._pg_and_order_for_tag(tag)
        return sum(int(p.numel()) for n, p in order.items())


def _multi_tensor_cp(src: List[torch.Tensor], dst: List[torch.Tensor]):
    """
    Copy data from one list of tensors to another.
    """
    for s, d in zip(src, dst):
        d.copy_(s)
