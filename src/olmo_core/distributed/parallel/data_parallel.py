import logging
from dataclasses import dataclass
from typing import Optional

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import get_num_nodes, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

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

    def build_device_mesh(self, device_type: Optional[str] = None) -> Optional[DeviceMesh]:
        """
        Build the optional device mesh needed for this config.
        """
        if self.name == DataParallelType.hsdp:
            num_replicas = self.num_replicas or get_num_nodes()
            device_type = device_type or get_default_device().type
            if get_world_size() % num_replicas != 0:
                raise OLMoConfigurationError(
                    "HSDP requires world size to be divisible by 'num_replicas'"
                )

            log.info(f"Building device mesh for HSDP with {num_replicas} replicas...")
            return init_device_mesh(
                device_type,
                (num_replicas, get_world_size() // num_replicas),
                mesh_dim_names=(DPMeshDimName.replicate, DPMeshDimName.shard),
            )
        else:
            return None
