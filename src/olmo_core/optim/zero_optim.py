import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import (
    hook_with_zero_step,
    hook_with_zero_step_interleaved,
)
from torch.distributed.optim import ZeroRedundancyOptimizer

from ..config import DType
from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer

log = logging.getLogger(__name__)

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
    cast,
)

# --------------------------------------------------------------------------- #
# ZeRO-1 sharded Skip-Step AdamW
# --------------------------------------------------------------------------- #
from torch.distributed.optim import ZeroRedundancyOptimizer

from ..utils import get_default_device, move_to_device
from .config import INITIAL_LR_FIELD, LR_FIELD

Opt = TypeVar("Opt", bound=torch.optim.Optimizer)

from ..train.train_module import TrainModule
from .config import OptimConfig, OptimGroupOverride


@dataclass
class ZeroOptimConfig(OptimConfig):
    inner_optimizer: OptimConfig

    @classmethod
    def optimizer(cls):
        return ZeroRedundancyOptimizer

    def build(
        self, model: nn.Module, train_module: TrainModule, strict: bool = True, param_filter=None
    ):
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")

        inner_optimizer = self.inner_optimizer.build(
            model, train_module, strict=strict, param_filter=param_filter
        )

        optim: torch.optim.Optimizer = self.optimizer()(
            self.build_groups(model, strict=strict),
            optimizer_class=torch.optim.AdamW,
            process_group=train_module.dp_process_group,
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

        return cast(Opt, optim)
