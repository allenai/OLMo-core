import logging
from typing import List, Optional, Tuple

from torch.distributed import DeviceMesh, ProcessGroup, init_device_mesh

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from .context_parallel import ContextParallelConfig
from .data_parallel import DataParallelConfig, DataParallelType, DPMeshDimName
from .expert_parallel import ExpertParallelConfig
from .pipeline_parallel import (
    PipelineParallelConfig,
    PipelineSchedule,
    PipelineScheduleType,
    PipelineSplitStyle,
)
from .tensor_parallel import TensorParallelConfig

__all__ = [
    "build_world_mesh",
    "get_world_mesh",
    "build_expert_parallel_mesh",
    "MeshDimName",
    "get_dp_model_mesh",
    "get_dp_mesh",
    "get_tp_mesh",
    "get_cp_mesh",
    "get_pp_mesh",
    "get_pp_stage_mesh",
    "get_ep_mesh",
    "get_dp_process_group",
    "get_device_mesh_info",
    "flatten_mesh",
    "DataParallelType",
    "DataParallelConfig",
    "DPMeshDimName",
    "TensorParallelConfig",
    "ExpertParallelConfig",
    "PipelineParallelConfig",
    "PipelineScheduleType",
    "PipelineSplitStyle",
    "PipelineSchedule",
    "ContextParallelConfig",
]

log = logging.getLogger(__name__)


class MeshDimName(StrEnum):
    """
    ``DeviceMesh`` dimensions names for different forms of parallelism.
    This are the dimension names that you will find in the mesh created by :func:`build_world_mesh()`.
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

    ep = "ep"
    """
    Expert parallel (EP).
    """

    ep_replicate = "ep_replicate"
    ep_shard = "ep_shard"

    dp_ep = "dp_ep"
    dp_cp = "dp_cp"


_WORLD_MESH: Optional[DeviceMesh] = None


def get_world_mesh() -> Optional[DeviceMesh]:
    """
    Get the global world mesh built with :meth:`build_world_mesh()`.
    """
    global _WORLD_MESH
    return _WORLD_MESH


def build_world_mesh(
    *,
    dp: Optional[DataParallelConfig] = None,
    tp: Optional[TensorParallelConfig] = None,
    cp: Optional[ContextParallelConfig] = None,
    pp: Optional[PipelineParallelConfig] = None,
    ep: Optional[ExpertParallelConfig] = None,
    device_type: Optional[str] = None,
) -> DeviceMesh:
    """
    Build a :class:`~torch.distributed.device_mesh.DeviceMesh` suitable for the given parallel strategies.

    .. seealso::
        Pass the mesh created by this function to any of the ``get_*_mesh()`` functions in
        this module to get the right sub-mesh for a any given parallel strategy.

        - :func:`get_dp_model_mesh()` gives you the 1 or 2D sub-mesh suitable for data parallel *model*
          wrappers like FSDP(2) or DDP.
        - :func:`get_dp_mesh()` gives you the 1D sub-mesh suitable for configuring *data loaders*.
        - :func:`get_tp_mesh()` gives you the 1D sub-mesh for tensor parallelism.
        - :func:`get_cp_mesh()` gives you the 1D sub-mesh for context parallelism.
        - :func:`get_pp_mesh()` gives you the 1D sub-mesh for pipeline parallelism.
        - :func:`get_ep_mesh()` gives you the 1D sub-mesh for expert parallelism.

    .. important::
        A data parallel config is required if any other parallel config is set.

    .. important::
        Not all parallel strategies are compatible with each other.

    :param dp: Data parallel config.
    :param tp: Tensor parallel config.
    :param cp: Context parallel config.
    :param pp: Pipeline parallel config.
    :param ep: Expert parallel config.
    :param device_type: The device type.

    :returns: The world mesh with a shape compatible with the given parallel configs.
    """
    global _WORLD_MESH

    if _WORLD_MESH is not None:
        raise RuntimeError("world mesh already exists! You can only call 'build_world_mesh' once!")

    device_type = device_type or get_default_device().type
    dp_world_size = get_world_size()

    if pp is None and tp is None and cp is None and dp is None and ep is None:
        return init_device_mesh(device_type, (dp_world_size,), mesh_dim_names=(MeshDimName.dp,))

    if dp is None:
        raise OLMoConfigurationError(
            "Data parallel config is required in addition to expert/tensor/context/pipeline parallel configs"
        )

    # Validate parallelism degrees while adjust the DP degree.
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
        if tp is not None:
            raise OLMoConfigurationError(
                "expert parallelism is mutually exclusive with tensor parallism"
            )
        # With HSDP we just reuse the 'dp_shard' dimension for expert sharding.
        if dp.name != DataParallelType.hsdp:
            dp_world_size //= ep.degree

            # TODO: remove this restriction once DTensor supports cross-mesh operations.
            raise OLMoConfigurationError(
                "expert parallelism can currently only be used with HSDP data parallelism"
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
        num_replicas, shard_degree = dp.get_replicate_and_shard_degree(dp_world_size)
        names.append(MeshDimName.dp_replicate)
        dims.append(num_replicas)
        names.append(MeshDimName.dp_shard)
        dims.append(shard_degree)

        # Expert parallel.
        if ep is not None:
            # We just reuse the 'dp_shard' dimension for expert sharding.
            if ep.degree >= 0 and ep.degree != shard_degree:
                raise OLMoConfigurationError(
                    "expert parallelism + HSDP requires the same sharding degree"
                )
    else:
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)

        # Expert parallel.
        if ep is not None:
            names.append(MeshDimName.ep)
            dims.append(ep.degree)

    # Context parallel.
    if cp is not None:
        names.append(MeshDimName.cp)
        dims.append(cp.degree)

    # And lastly tensor parallel.
    if tp is not None:
        names.append(MeshDimName.tp)
        dims.append(tp.degree)

    mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    log.info(f"Built {get_device_mesh_info(mesh)}")

    # Ensure data parallel process group is created here.
    get_dp_process_group(mesh)

    _WORLD_MESH = mesh

    return mesh


def get_device_mesh_info(device_mesh: DeviceMesh) -> str:
    """
    Get a human-readable string representation of a ``DeviceMesh``.

    :param device_mesh: The device mesh to get info for.
    """
    shape: str
    if device_mesh.mesh_dim_names is not None:
        shape = ", ".join(
            f"{dim_name}={d}" for dim_name, d in zip(device_mesh.mesh_dim_names, device_mesh.shape)
        )
    else:
        shape = ", ".join(f"{d}" for d in device_mesh.shape)
    if device_mesh.ndim == 1:
        shape += ","
    return f"{device_mesh.ndim}D device mesh with shape ({shape})"


def build_expert_parallel_mesh(
    ep_config: ExpertParallelConfig, device_type: Optional[str] = None
) -> DeviceMesh:
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

    mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    log.info(f"Built {get_device_mesh_info(mesh)}")

    return mesh


def _get_model_mesh(device_mesh: DeviceMesh) -> Tuple[DeviceMesh, Tuple[str, ...]]:
    if (dim_names := device_mesh.mesh_dim_names) is None:
        raise RuntimeError("could not determine DP model sub-mesh without dimension names")

    # Expert parallel dims get flattened into a DP dimension.
    if MeshDimName.dp in dim_names and MeshDimName.ep in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.dp,
            MeshDimName.ep,
            name=MeshDimName.dp_ep,
            dim_names=dim_names,
        )
    elif MeshDimName.ep_replicate in dim_names and MeshDimName.ep_shard in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.ep_replicate,
            MeshDimName.ep_shard,
            name=MeshDimName.dp,
            dim_names=dim_names,
        )

    # Context parallel dimension gets flattened into the adjacent DP dimension.
    # NOTE: We do this because for param-synchronization purposes a CP group behaves like an extra
    # DP replica set. CP splits the context across ranks but every CP rank still holds a copy of
    # the model parameters. Gradients need to be reduced across the union of DP ranks and CP ranks.
    if MeshDimName.cp in dim_names:
        last_dp_dim = dim_names[dim_names.index(MeshDimName.cp) - 1]
        assert last_dp_dim.startswith("dp")
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            last_dp_dim,
            MeshDimName.cp,
            name=MeshDimName.dp_cp,
            dim_names=dim_names,
        )

    return device_mesh, dim_names


def get_dp_model_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the right sub-mesh for a data parallel model wrapper like FSDP or DDP from a ``DeviceMesh``
    created by :func:`build_world_mesh()`.

    .. important::
        You should use :func:`get_dp_mesh()` instead for getting the sub-mesh to assign ranks
        to data loading workers. In many cases these two functions will return the same result,
        but there are cases where they could be different.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    device_mesh, dim_names = _get_model_mesh(device_mesh)
    dp_dim_names = tuple(name for name in dim_names if name.startswith("dp"))
    return device_mesh[dp_dim_names]


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the data parallel sub-mesh associated from a ``DeviceMesh`` created by :func:`build_world_mesh()`.

    .. important::
        This is the mesh that should be used to assign ranks to data loading workers,
        however you should use :func:`get_dp_model_mesh()` to get the mesh for DDP/FSDP.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    if (dim_names := device_mesh.mesh_dim_names) is None:
        raise RuntimeError("could not determine DP sub-mesh without dimension names")

    # Expert parallel dims get flattened into DP dimension since ranks within each EP group
    # should receive different data instances.
    if MeshDimName.dp in dim_names and MeshDimName.ep in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.dp,
            MeshDimName.ep,
            name=MeshDimName.dp_ep,
            dim_names=dim_names,
        )
    elif MeshDimName.ep_replicate in dim_names and MeshDimName.ep_shard in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.ep_replicate,
            MeshDimName.ep_shard,
            name=MeshDimName.dp,
            dim_names=dim_names,
        )

    # Flattened context parallel dimensions should not be in this mesh since ranks within the
    # same CP group should receive the same data instances.
    if MeshDimName.dp_cp in dim_names:
        raise RuntimeError("'get_dp_mesh' should be called on the original world mesh")

    dp_dim_names = tuple(name for name in dim_names if name.startswith("dp"))
    return device_mesh[dp_dim_names]


def get_dp_process_group(device_mesh: DeviceMesh) -> ProcessGroup:
    """
    Get the data parallel process group associated with a ``DeviceMesh``
    created from :func:`build_world_mesh()`.

    Like :func:`get_dp_mesh()`, this should be used for data loading, but not necessarily for
    data parallel model wrappers.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    dp_mesh = get_dp_mesh(device_mesh)
    if len(dp_mesh.shape) > 1:
        return dp_mesh._flatten(mesh_dim_name=MeshDimName.dp).get_group()
    else:
        return dp_mesh.get_group()


def get_ep_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the expert parallel sub-mesh associated with a ``DeviceMesh`` that was potentially
    created from :func:`build_world_mesh()`.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine expert parallel sub-mesh without dimension names")

    if MeshDimName.ep in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.ep]
    elif MeshDimName.ep_shard in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.ep_shard]
    elif MeshDimName.dp_shard in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.dp_shard]
    else:
        raise RuntimeError(
            f"could not determine expert parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_tp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the tensor parallel sub-mesh associated with a ``DeviceMesh``
    created from :func:`build_world_mesh()`.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    device_mesh, dim_names = _get_model_mesh(device_mesh)

    if MeshDimName.tp in dim_names:
        return device_mesh[MeshDimName.tp]
    else:
        raise RuntimeError(
            f"could not determine tensor parallel sub-mesh from mesh with dimensions {dim_names}"
        )


def get_cp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the context parallel sub-mesh associated with a ``DeviceMesh``
    created from :func:`build_world_mesh()`.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
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
    Get the pipeline parallel sub-mesh associated with a ``DeviceMesh``
    created from :func:`build_world_mesh()`.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine pipeline parallel sub-mesh without dimension names")

    if MeshDimName.pp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.pp]
    else:
        raise RuntimeError(
            f"could not determine pipeline parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_pp_stage_mesh(device_mesh: DeviceMesh, pp_mesh: Optional[DeviceMesh] = None) -> DeviceMesh:
    """
    Get the sub-mesh for a single pipeline stage.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    :param pp_mesh: Optional pipeline parallel mesh. If not provided, it will be
        extracted from the device_mesh using :func:`get_pp_mesh()`.
    """
    if pp_mesh is None:
        pp_mesh = get_pp_mesh(device_mesh)

    if device_mesh.mesh_dim_names is None or pp_mesh.mesh_dim_names is None:
        raise RuntimeError(
            "could not determine pipeline parallel stage sub-mesh without dimension names"
        )

    target_dims = tuple(n for n in device_mesh.mesh_dim_names if n not in pp_mesh.mesh_dim_names)
    return device_mesh[target_dims]


def _flatten_dims(
    device_mesh: DeviceMesh,
    *dims: str,
    name: Optional[str] = None,
    dim_names: Optional[Tuple[str, ...]] = None,
) -> Tuple[DeviceMesh, Tuple[str, ...]]:
    """
    Flatten *dims* into a single dimension called *name*.

    :param device_mesh: The world-mesh object. Only views of *device_mesh* are actually mutated.
    :param dims: The existing dimension names to merge.
    :param name: New dimension name. If ``None`` we join *dims* with "_".
    :param dim_names: Optional cached list of current dimension names. Supplying this avoids
        relying on ``device_mesh.mesh_dim_names`` (which is stale after a prior
        flatten) and therefore allows chaining multiple flatten operations.

    :returns: The root mesh (now indexable by the new dimension names
        as well as the original names) and the new dimension names.
    """
    if name is None:
        name = "_".join(dims)

    curr_names = list(dim_names or device_mesh.mesh_dim_names or [])
    if not curr_names:
        raise RuntimeError("Could not determine current dimension names for flattening")

    log.info(f"Flattening mesh dimensions {dims} into {name}")

    out_names: list[str] = []
    for n in curr_names:
        if n in dims:
            if name not in out_names:
                out_names.append(name)
        else:
            out_names.append(n)

    flatten_mesh(device_mesh[dims], name)  # in-place flatten on sub-mesh
    new_names = tuple(out_names)

    try:
        # NOTE: device_mesh.mesh_dim_names is not updated based on the flatten operation.
        # We need to check that the root mesh is indexable by the new dimension names.
        _ = device_mesh[new_names]
    except KeyError as exc:
        raise RuntimeError(
            "Flattening failed: root device mesh does not recognize the new "
            f"dimension names {new_names}. Original dims: {dims}."
        ) from exc

    return device_mesh, new_names


def flatten_mesh(device_mesh: DeviceMesh, name: Optional[str] = None) -> DeviceMesh:
    """
    Flatten a multi-dimensional ``DeviceMesh`` into a 1D ``DeviceMesh``.

    :param device_mesh: The multi-dimensional ``DeviceMesh`` to flatten.
    :param name: Optional name for the flattened dimension.

    .. important::
        The ``device_mesh`` is modified in-place.
    """
    return device_mesh._flatten(mesh_dim_name=name)
