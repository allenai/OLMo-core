import logging
from dataclasses import dataclass
from typing import Type

import torch.distributed as dist
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
