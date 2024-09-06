from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union

import torch
import torch.nn as nn

from ..config import Config
from ..exceptions import OLMoConfigurationError

__all__ = [
    "OptimConfig",
    "OptimGroupOverride",
]


Opt = TypeVar("Opt", bound=torch.optim.Optimizer)


@dataclass
class OptimGroupOverride(Config):
    params: List[str]
    """
    A list of fully qualified parameter names.
    """

    opts: Dict[str, Any]
    """
    Options to set in the corresponding param group.
    """


@dataclass
class OptimConfig(Config, Generic[Opt]):
    """
    Base class for :class:`~torch.optim.Optimizer` configs.
    """

    group_overrides: Optional[List[OptimGroupOverride]] = None
    """
    Use this to pull out groups parameters into a separate param groups with their own options.
    """

    def build_groups(self, model: nn.Module) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameters groups.
        """
        if self.group_overrides is None:
            return model.parameters()

        all_params: Dict[str, torch.Tensor] = OrderedDict()
        for n, p in model.named_parameters():
            all_params[n] = p

        # Build groups.
        param_groups: List[Dict[str, Any]] = [
            {"params": [], **go.opts} for go in self.group_overrides
        ]
        for g_idx, (g, go) in enumerate(zip(param_groups, self.group_overrides)):
            for n in go.params:
                if n not in all_params:
                    raise OLMoConfigurationError(
                        f"optim group {g_idx} override param name '{n}' does not match any parameters"
                    )
                g["params"].append(all_params.pop(n))

        # Put any left-over params into a default group.
        if all_params:
            param_groups.append({"params": list(all_params.values())})

        return param_groups

    @abstractmethod
    def build(self, model: nn.Module) -> Opt:
        """
        Build the optimizer.
        """
        raise NotImplementedError
