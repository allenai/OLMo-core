import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn

from ..config import Config
from ..exceptions import OLMoConfigurationError
from ..utils import get_default_device, move_to_device

__all__ = [
    "OptimConfig",
    "OptimGroupOverride",
]

log = logging.getLogger(__name__)

Opt = TypeVar("Opt", bound=torch.optim.Optimizer)


@dataclass
class OptimGroupOverride(Config):
    params: List[str]
    """
    A list of fully qualified parameter names (FQNs) or wild card to match FQNs.
    """

    opts: Dict[str, Any]
    """
    Options to set in the corresponding param group.
    """


@dataclass
class OptimConfig(Config, Generic[Opt], metaclass=ABCMeta):
    """
    Base class for :class:`~torch.optim.Optimizer` configs.
    """

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

    @property
    def device(self) -> torch.device:
        return get_default_device()

    def build_groups(
        self, model: nn.Module, strict: bool = True
    ) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        if self.group_overrides is None:
            log.info(f"Building {self.optimizer().__name__} optimizer with 1 param group...")
            return model.parameters()

        all_params: Dict[str, torch.Tensor] = OrderedDict()
        for n, p in model.named_parameters():
            all_params[n] = p

        # Build groups.
        param_groups: List[Dict[str, Any]] = []
        for g_idx, go in enumerate(self.group_overrides):
            g = {"params": [], **go.opts}
            for pattern in go.params:
                matches = 0
                for name in list(all_params.keys()):
                    if fnmatch(name, pattern):
                        g["params"].append(all_params.pop(name))
                        matches += 1

                if matches == 0:
                    msg = f"optim group {g_idx} override pattern '{pattern}' does not match any parameters"
                    if strict:
                        raise OLMoConfigurationError(msg)
                    else:
                        log.warning(msg)

            if len(g["params"]) > 0:
                param_groups.append(g)

        # Put any left-over params into a default group.
        if all_params:
            param_groups.append({"params": list(all_params.values())})

        return param_groups

    @classmethod
    @abstractmethod
    def optimizer(cls) -> Type[Opt]:
        """
        Get the optimizer class associated with this config.
        """
        raise NotImplementedError

    def build(self, model: nn.Module, strict: bool = True) -> Opt:
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")

        optim = self.optimizer()(self.build_groups(model, strict=strict), **kwargs)

        # Set 'lr' and 'initial_lr' in each group if needed.
        for group in optim.param_groups:
            lr: Optional[float] = None
            if "lr" in group:
                lr = group["lr"]
            elif hasattr(self, "lr"):
                lr = getattr(self, "lr")

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group["lr"] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group["lr"] = lr
                group.setdefault("initial_lr", lr)

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param groups..."
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

        return optim
