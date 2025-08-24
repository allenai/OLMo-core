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

log = logging.getLogger(__name__)

# Opt = TypeVar("Opt", bound=torch.optim.Optimizer)

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
        self, model: nn.Module, strict: bool = True, param_filter=None
    ) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: Dict[str, torch.Tensor] = OrderedDict()
        frozen_params: set = set()
        for n, p in model.named_parameters():
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

        return [
            {"params": [all_params[param_name] for param_name in go.params], **go.opts}
            for go in group_overrides
            if len(go.params) > 0
        ]


    @classmethod
    def optimizer(cls):
        return MoEFusedV2Optimizer


    def build(self, model: nn.Module, train_module: TrainModule, strict: bool = True, param_filter=None) -> torch.optim.Optimizer:
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        
        # not used: train_module
        
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")

        optim: torch.optim.Optimizer = self.optimizer()(
            self.build_groups(model, strict=strict, param_filter=param_filter), **kwargs
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

        return cast(torch.optim.Optimizer, optim)


class MoEFusedV2Optimizer(Optimizer):


    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        # foreach: bool = False,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = None

        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        # self.foreach = foreach
        self._step_skipped: Optional[torch.Tensor] = None


        # keep the main copy in fp32
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []
        
        for param_group in params:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()

                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param

                        # Store handle to main_param.
                        param.main_param = main_param

                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.state:
                            self.state[main_param] = self.state.pop(param)
                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError(
                            'Wrapped parameters must be one of '
                            'torch.cuda.FloatTensor,  '
                            'torch.cuda.HalfTensor, or '
                            'torch.cuda.BFloat16Tensor. '
                            'Received {}'.format(param.type())
                        )

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

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

    @overload # make pylance happy
    def step(self, closure: None = ...) -> None:
        ...

    @overload # make pylance happy
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        dbg_mem_before_cp1 = torch.cuda.memory_allocated()/1024**3
        self._copy_model_grads_to_main_grads()
        dbg_mem_before_step = torch.cuda.memory_allocated()/1024**3
        self._step_foreach(closure)
        dbg_mem_after_step = torch.cuda.memory_allocated()/1024**3
        self._dealloc_main_grad()
        dbg_mem_before_cp2 = torch.cuda.memory_allocated()/1024**3
        self._copy_main_params_to_model_params()
        return None

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_cp(
            src=main_data, dst=model_data,
        )

    def _dealloc_main_grad(self):
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                assert model_param.main_param is main_param, 'main param pointer mismatch'
                if hasattr(main_param, 'grad'):
                    main_param.grad = None

        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                assert model_param.main_param is main_param, 'main param pointer mismatch'


    def _copy_model_params_to_main_params(self, state_dict=None):
        assert state_dict is None, "Initialize main params from state dict is not supported"
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_cp(
            src=model_data, dst=main_data,
        )

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data
    
    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                assert model_param.main_param is main_param, 'main param pointer mismatch'
                if hasattr(model_param, 'main_grad'):
                    assert False, "Use main_param.grad, not model_param.main_grad"
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                # model_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                model_param.grad = model_param.main_grad

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
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor() # type: ignore
        step_factor = cast(torch.Tensor, step_factor)
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps_list = []  # create list outside loops

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps_list.append(state["step"])

            if not params_with_grad:
                continue  # nothing to update in this group

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
            # grads_size = sum([g.numel() * g.element_size() for g in grads])/1024**3
            # exp_avgs_size = sum([ea.numel() * ea.element_size() for ea in exp_avgs])/1024**3
            # exp_avg_sqs_size = sum([eas.numel() * eas.element_size() for eas in exp_avg_sqs])/1024**3

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

def _multi_tensor_cp(src: List[torch.Tensor], dst: List[torch.Tensor]):
    """
    Copy data from one list of tensors to another.
    """
    for s, d in zip(src, dst):
        d.copy_(s)


def _zero_grad_group_helper(
    group: List[torch.nn.Parameter], set_to_none: bool, use_decoupled_grad: bool = False
):
    """
    Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer.
    """
    for param in group:
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        if hasattr(param, grad_attr) and getattr(param, grad_attr) is not None:
            if set_to_none:
                setattr(param, grad_attr, None)
            else:
                grad_obj = getattr(param, grad_attr)
                if grad_obj.grad_fn is not None:
                    grad_obj.detach_()
                else:
                    grad_obj.requires_grad_(False)
                grad_obj.zero_()

