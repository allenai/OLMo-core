import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Type,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

from .config import OptimConfig

log = logging.getLogger(__name__)


@dataclass
class MuonWithAuxAdamConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`MuonWithAuxAdam` optimizer.
    See https://github.com/KellerJordan/Muon for more details.
    """

    @classmethod
    def optimizer(cls) -> Type[MuonWithAuxAdam]:
        try:
            dist.get_world_size()
            return MuonWithAuxAdam
        except ValueError:
            log.warning(
                "MuonWithAuxAdam is not available in single-device mode, using SingleDeviceMuonWithAuxAdam instead."
            )
            return SingleDeviceMuonWithAuxAdam

    def build_groups(
        self, model: nn.Module, strict: bool = True
    ) -> Union[Iterable[torch.Tensor], List[Dict[str, Any]]]:
        """
        Build parameter groups for MuonWithAuxAdam.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        param_groups = super().build_groups(model, strict)

        # Verify that Muon is only used for appropriate parameters
        for group in param_groups:
            if isinstance(group, dict) and "params" in group:
                try:
                    use_muon = group["use_muon"]
                except KeyError as ex:
                    # Log the parameters in the group before raising the error
                    param_names = []
                    for param in group["params"]:
                        param_name = next((name for name, p in model.named_parameters() if p is param), None)
                        param_names.append(param_name or "unknown")
                    log.error(f"Parameters in group without 'use_muon' specified: {param_names}")

                    raise ValueError(
                        "MuonWithAuxAdam requires 'use_muon' to be specified in the optimizer group options."
                    ) from ex

                for param in group["params"]:
                    param_name = next((name for name, p in model.named_parameters() if p is param), None)

                    if use_muon and param.ndim < 2:
                        raise ValueError(
                            f"Parameter '{param_name}' with ndim={param.ndim} should not have use_muon=True. "
                            "Only parameters with ndim >= 2 should use Muon."
                        )

                    if (
                        use_muon
                        and param_name
                        and ("lm_head" in param_name.lower() or "embed" in param_name.lower())
                    ):
                        raise ValueError(
                            f"Parameter '{param_name}' appears to be a head or embedding parameter "
                            "and should not have use_muon=True."
                        )

        return param_groups
