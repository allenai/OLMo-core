from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config, DType, StrEnum


class DataParallelType(StrEnum):
    fsdp = "fsdp"
    ddp = "ddp"


@dataclass
class DataParallelConfig(Config):
    name: DataParallelType
    param_dtype: Optional[DType] = None
    reduce_dtype: DType = DType.float32
