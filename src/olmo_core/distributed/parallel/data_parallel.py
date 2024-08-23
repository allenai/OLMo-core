from dataclasses import dataclass
from typing import Optional

import torch

from olmo_core.config import Config, StrEnum


class DataParallelType(StrEnum):
    fsdp = "fsdp"
    ddp = "ddp"


@dataclass
class DataParallelConfig(Config):
    name: DataParallelType
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: torch.dtype = torch.float32
