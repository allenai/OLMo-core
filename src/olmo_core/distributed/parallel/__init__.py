import logging
from typing import List, Optional, Tuple

from torch.distributed import DeviceMesh, ProcessGroup, init_device_mesh

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_num_nodes, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from .context_parallel import (
    ContextParallelConfig,
    ContextParallelLoadBalancer,
    ContextParallelLoadBalancerType,
    ContextParallelZigZagLoadBalancer,
)
from .data_parallel import DataParallelConfig, DataParallelType, DPMeshDimName
from .expert_parallel import ExpertParallelConfig
from .pipeline_parallel import (
    PipelineParallelConfig,
    PipelineSchedule,
    PipelineScheduleType,
)
from .tensor_parallel import TensorParallelConfig

__all__ = [
    "build_device_mesh",
    "build_expert_parallel_mesh",
    "MeshDimName",
    "get_dp_mesh",
    "get_dp_model_mesh",
    "get_tp_mesh",
    "get_pp_mesh",
    "get_ep_mesh",
    "get_cp_mesh",
    "get_dp_process_group",
    "DataParallelType",
    "DataParallelConfig",
    "DPMeshDimName",
    "TensorParallelConfig",
    "ExpertParallelConfig",
    "PipelineParallelConfig",
    "PipelineScheduleType",
    "PipelineSchedule",
    "ContextParallelConfig",
    "ContextParallelLoadBalancerType",
    "ContextParallelLoadBalancer",
    "ContextParallelZigZagLoadBalancer",
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

    cp = "cp"
    """
    Context parallel (CP).
    """

    pp = "pp"
    """
    Pipeline parallel (PP).
    """

    ep_replicate = "ep_replicate"
    ep_shard = "ep_shard"

    dp_cp = "dp_cp"
    dp_shard_cp = "dp_shard_cp"


def build_device_mesh(
    *,
    dp: Optional[DataParallelConfig] = None,
    tp: Optional[TensorParallelConfig] = None,
    cp: Optional[ContextParallelConfig] = None,
    pp: Optional[PipelineParallelConfig] = None,
    ep: Optional[ExpertParallelConfig] = None,
    device_type: Optional[str] = None,
) -> DeviceMesh:
    """
    Build a ``DeviceMesh`` suitable for the given parallel strategies.
    The resulting dimension names will be defined in :class:`MeshDimName`.

    .. important::
        A data parallel config is required if any other parallel config is set.
    """
    device_type = device_type or get_default_device().type
    dp_world_size = get_world_size()

    if pp is None and tp is None and cp is None and dp is None and ep is None:
        return init_device_mesh(device_type, (dp_world_size,), mesh_dim_names=(MeshDimName.dp,))

    if dp is None:
        raise OLMoConfigurationError(
            "Data parallel config is required in addition to expert/tensor/context/pipeline parallel configs"
        )

    if pp is not None:
        if pp.degree < 1 or dp_world_size % pp.degree != 0:
            raise OLMoConfigurationError(
                f"{pp.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        dp_world_size //= pp.degree
    if cp is not None:
        if cp.degree < 1 or dp_world_size % cp.degree != 0:
            raise OLMoConfigurationError(
                f"{cp.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        dp_world_size //= cp.degree
    if tp is not None:
        if tp.degree < 1 or dp_world_size % tp.degree != 0:
            raise OLMoConfigurationError(
                f"{tp.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        dp_world_size //= tp.degree
    if ep is not None:
        if ep.degree == 0 or dp_world_size % ep.degree != 0:
            raise OLMoConfigurationError(
                f"{ep.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        if dp.name != DataParallelType.hsdp:
            raise OLMoConfigurationError(
                "expert parallelism can currently only be used with HSDP data parallelism"
            )
        if tp is not None:
            raise OLMoConfigurationError(
                "expert parallelism is mutually exclusive with tensor parallism"
            )
        if pp is not None:
            raise NotImplementedError(
                "expert parallelism + pipeline parallelism is not implemented yet"
            )

    # Build up mesh dimensions.
    names: List[str] = []
    dims: List[int] = []

    # Pipeline parallel first.
    if pp is not None:
        names.append(MeshDimName.pp)
        dims.append(pp.degree)

    # Then data parallel.
    if dp.name == DataParallelType.hsdp:
        num_replicas = dp.num_replicas or get_num_nodes()
        if dp_world_size % num_replicas != 0:
            raise OLMoConfigurationError(
                f"HSDP requires DP world size ({dp_world_size}) to be divisible by 'num_replicas' ({num_replicas})"
            )
        shard_degree = dp_world_size // num_replicas
        if ep is not None:
            if ep.degree >= 0 and ep.degree != shard_degree:
                raise OLMoConfigurationError(
                    "expert parallelism + HSDP requires the same sharding degree"
                )

        names.append(MeshDimName.dp_replicate)
        dims.append(num_replicas)
        names.append(MeshDimName.dp_shard)
        dims.append(shard_degree)
    else:
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)

    # Context parallel.
    if cp is not None:
        names.append(MeshDimName.cp)
        dims.append(cp.degree)

    # And lastly tensor parallel.
    if tp is not None:
        names.append(MeshDimName.tp)
        dims.append(tp.degree)

    log.info(f"Building {len(dims)}-D device mesh with dimensions:")
    for i, (name, dim) in enumerate(zip(names, dims)):
        log.info(f" > dimension {i}, size={dim}, name={name}")

    mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    # Ensure data parallel process group is created here.
    get_dp_process_group(mesh)
    return mesh


def build_expert_parallel_mesh(
    ep_config: ExpertParallelConfig, device_type: Optional[str] = None
) -> DeviceMesh:
    """
    Build a device mesh for expert parallelism.
    """
    device_type = device_type or get_default_device().type
    world_size = get_world_size()

    # Build up mesh dimensions.
    names: List[str] = []
    dims: List[int] = []

    ep_degree = ep_config.degree
    if ep_degree < 0:
        ep_degree = world_size

    if world_size % ep_degree != 0:
        raise OLMoConfigurationError(
            f"Expert parallelism requires world size ({world_size}) to "
            f"be divisible by 'degree' ({ep_degree})"
        )
    names.append(MeshDimName.ep_replicate)
    dims.append(world_size // ep_degree)

    names.append(MeshDimName.ep_shard)
    dims.append(ep_degree)

    log.info(f"Building {len(dims)}-D device mesh with dimensions:")
    for i, (name, dim) in enumerate(zip(names, dims)):
        log.info(f" > dimension {i}, size={dim}, name={name}")

    return init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the data parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if (dim_names := device_mesh.mesh_dim_names) is None:
        raise RuntimeError("could not determine DP sub-mesh without dimension names")

    # Expert parallel dims get flattened into DP dimension.
    if MeshDimName.ep_replicate in dim_names and MeshDimName.ep_shard in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.ep_replicate,
            MeshDimName.ep_shard,
            name=MeshDimName.dp,
        )

    # Flattened context parallel dimensions should not be in this mesh since ranks within the
    # same CP group should receive the same data instances.
    if MeshDimName.dp_cp in dim_names or MeshDimName.dp_shard_cp in dim_names:
        raise RuntimeError("'get_dp_mesh' should be called on the original world mesh")

    dp_dim_names = tuple(name for name in dim_names if name.startswith("dp"))
    return device_mesh[dp_dim_names]


def _flatten_dims(
    device_mesh: DeviceMesh, *dims: str, name: str
) -> Tuple[DeviceMesh, Tuple[str, ...]]:
    log.info(f"Flattening mesh dimensions {dims} into {name}")
    assert (names := device_mesh.mesh_dim_names) is not None
    out_names = []
    for n in names:
        if n in dims:
            if name not in out_names:
                out_names.append(name)
        else:
            out_names.append(n)
    device_mesh[dims]._flatten(mesh_dim_name=name)
    new_names = tuple(out_names)
    return device_mesh[new_names], new_names


def get_dp_model_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the right sub-mesh for a data parallel model wrapper like FSDP or DDP.
    In most cases this returns the same things as :func:`get_dp_mesh()`.
    """
    if (dim_names := device_mesh.mesh_dim_names) is None:
        raise RuntimeError("could not determine DP model sub-mesh without dimension names")

    # Expert parallel dims get flattened into DP dimension.
    if MeshDimName.ep_replicate in dim_names and MeshDimName.ep_shard in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh, MeshDimName.ep_replicate, MeshDimName.ep_shard, name=MeshDimName.dp
        )

    # Context parallel dimension gets flattened into the adjacent DP dimension.
    if MeshDimName.dp in dim_names and MeshDimName.cp in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh, MeshDimName.dp, MeshDimName.cp, name=MeshDimName.dp_cp
        )
    elif (
        MeshDimName.dp_replicate in dim_names
        and MeshDimName.dp_shard in dim_names
        and MeshDimName.cp in dim_names
    ):
        device_mesh, dim_names = _flatten_dims(
            device_mesh, MeshDimName.dp_shard, MeshDimName.cp, name=MeshDimName.dp_shard_cp
        )

    dp_dim_names = tuple(name for name in dim_names if name.startswith("dp"))
    return device_mesh[dp_dim_names]


def get_dp_process_group(device_mesh: DeviceMesh) -> ProcessGroup:
    """
    Get the data parallel process group associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    dp_mesh = get_dp_mesh(device_mesh)
    if len(dp_mesh.shape) > 1:
        return dp_mesh._flatten(mesh_dim_name=MeshDimName.dp).get_group()
    else:
        return dp_mesh.get_group()


def get_ep_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the expert parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine expert parallel sub-mesh without dimension names")

    if (
        MeshDimName.ep_replicate in device_mesh.mesh_dim_names
        and MeshDimName.ep_shard in device_mesh.mesh_dim_names
    ):
        return device_mesh[MeshDimName.ep_replicate, MeshDimName.ep_shard]
    elif (
        MeshDimName.dp_replicate in device_mesh.mesh_dim_names
        and MeshDimName.dp_shard in device_mesh.mesh_dim_names
    ):
        return device_mesh[MeshDimName.dp_replicate, MeshDimName.dp_shard]
    else:
        raise RuntimeError(
            f"could not determine expert parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_tp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the tensor parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine tensor parallel sub-mesh without dimension names")

    if MeshDimName.tp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.tp]
    else:
        raise RuntimeError(
            f"could not determine tensor parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_cp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the context parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine context parallel sub-mesh without dimension names")

    if MeshDimName.cp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.cp]
    else:
        raise RuntimeError(
            f"could not determine context parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_pp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the tensor parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_device_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine pipeline parallel sub-mesh without dimension names")

    if MeshDimName.pp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.pp]
    else:
        raise RuntimeError(
            f"could not determine pipeline parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )
