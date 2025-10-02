import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import get_num_nodes
from olmo_core.exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)


class DPMeshDimName(StrEnum):
    """
    ``DeviceMesh`` dimension names for data parallelism.
    """

    replicate = "dp_replicate"
    """
    The device mesh dimension over which the model is replicated.
    """
    shard = "dp_shard"
    """
    The device mesh dimension over which the model is sharded.
    """


class DataParallelType(StrEnum):
    fsdp = "fsdp"
    hsdp = "hsdp"
    ddp = "ddp"


@dataclass
class DataParallelConfig(Config):
    name: DataParallelType
    param_dtype: Optional[DType] = None
    reduce_dtype: DType = DType.float32
    num_replicas: Optional[int] = None
    shard_degree: Optional[int] = None

    def get_replicate_and_shard_degree(self, dp_world_size: int) -> Tuple[int, int]:
        """
        Defaults to one replica per node, with the shard degree set to the number of gpus per node.

        :param dp_world_size: The data parallel world size.
        :return: A tuple of (num_replicas, shard_degree)
        """
        if self.num_replicas is None and self.shard_degree is None:
            return get_num_nodes(), dp_world_size // get_num_nodes()
        elif self.num_replicas is not None and self.shard_degree is not None:
            return _check_num_replicas(self.num_replicas, dp_world_size), _check_shard_degree(
                self.shard_degree, dp_world_size
            )
        elif self.num_replicas is not None:
            return (
                _check_num_replicas(self.num_replicas, dp_world_size),
                dp_world_size // self.num_replicas,
            )
        else:
            assert self.shard_degree is not None
            return dp_world_size // self.shard_degree, _check_shard_degree(
                self.shard_degree, dp_world_size
            )


def _check_num_replicas(num_replicas: int, dp_world_size: int) -> int:
    if dp_world_size % num_replicas != 0:
        raise OLMoConfigurationError(
            f"data parallel world size ({dp_world_size}) must be "
            f"divisible by 'num_replicas' ({num_replicas})"
        )
    return num_replicas


def _check_shard_degree(shard_degree: int, dp_world_size: int) -> int:
    if dp_world_size % shard_degree != 0:
        raise OLMoConfigurationError(
            f"data parallel world size ({dp_world_size}) must be "
            f"divisible by 'shard_degree' ({shard_degree})"
        )
    return shard_degree
