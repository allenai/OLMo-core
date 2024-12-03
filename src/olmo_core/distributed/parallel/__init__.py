import logging
from typing import List, Optional

from torch.distributed import DeviceMesh, ProcessGroup, init_device_mesh

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_num_nodes, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from .data_parallel import DataParallelConfig, DataParallelType, DPMeshDimName
from .tensor_parallel import TensorParallelConfig

__all__ = [
    "build_device_mesh",
    "MeshDimName",
    "get_dp_mesh",
    "get_tp_mesh",
    "get_dp_process_group",
    "DataParallelType",
    "DataParallelConfig",
    "DPMeshDimName",
    "TensorParallelConfig",
]

log = logging.getLogger(__name__)


class MeshDimName(StrEnum):
    """
    ``DeviceMesh`` dimensions names for different forms of parallelism.
    """

    dp = "dp"
    """
    Data parallel (DP).
    """

    dp_replicate = DPMeshDimName.replicate
    """
    The DP dimension over which the model is replicated.
    """

    dp_shard = DPMeshDimName.shard
    """
    The DP dimension over which the model is sharded.
    """

    tp = "tp"
    """
    Tensor parallel (TP).
    """


def build_device_mesh(
    *,
    dp: Optional[DataParallelConfig] = None,
    tp: Optional[TensorParallelConfig] = None,
    device_type: Optional[str] = None,
) -> Optional[DeviceMesh]:
    """
    Build a ``DeviceMesh`` suitable for the given parallel strategies.
    The resulting dimension names will be defined in :class:`MeshDimName`.
    """
    device_type = device_type or get_default_device().type

    if tp is None and dp is None:
        return None
    elif tp is None:
        assert dp is not None
        return dp.build_device_mesh(device_type=device_type)
    else:
        assert dp is not None
        assert tp is not None

        if get_world_size() % tp.degree != 0:
            raise OLMoConfigurationError(
                f"World size {get_world_size()} must be divisible by TP degree ({tp.degree})"
            )

        dp_world_size = get_world_size() // tp.degree

        dims: List[int] = []
        names: List[str] = []

        if dp.name == DataParallelType.hsdp:
            num_replicas = dp.num_replicas or get_num_nodes()
            if dp_world_size % num_replicas != 0:
                raise OLMoConfigurationError(
                    f"HSDP requires DP world size ({dp_world_size}) to be divisible by 'num_replicas' ({num_replicas})"
                )
            dims.append(num_replicas)
            dims.append(dp_world_size // num_replicas)
            names.append(MeshDimName.dp_replicate)
            names.append(MeshDimName.dp_shard)
        else:
            dims.append(dp_world_size)
            names.append(MeshDimName.dp)

        dims.append(tp.degree)
        names.append(MeshDimName.tp)

        log.info(f"Building {len(dims)}-D device mesh with {names}, {dims}...")

        return init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))


def get_dp_mesh(device_mesh: Optional[DeviceMesh] = None) -> Optional[DeviceMesh]:
    """
    Get the data parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if device_mesh is None:
        return None

    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine data parallel sub-mesh without dimension names")

    if MeshDimName.dp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.dp]
    elif (
        MeshDimName.dp_replicate in device_mesh.mesh_dim_names
        and MeshDimName.dp_shard in device_mesh.mesh_dim_names
    ):
        return device_mesh[MeshDimName.dp_replicate, MeshDimName.dp_shard]
    else:
        raise RuntimeError(
            f"could not determine data parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_dp_process_group(device_mesh: Optional[DeviceMesh] = None) -> Optional[ProcessGroup]:
    """
    Get the data parallel process group associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    dp_mesh = get_dp_mesh(device_mesh)
    if dp_mesh is None:
        return None
    else:
        if len(dp_mesh.shape) > 1:
            return dp_mesh._flatten(mesh_dim_name=MeshDimName.dp).get_group()
        else:
            return dp_mesh.get_group()


def get_tp_mesh(device_mesh: Optional[DeviceMesh] = None) -> Optional[DeviceMesh]:
    """
    Get the tensor parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if device_mesh is None:
        return None

    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine tensor parallel sub-mesh without dimension names")

    if MeshDimName.tp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.tp]
    else:
        return None
