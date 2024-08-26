"""
Distributed helpers, most of which work in a non-distributed context as well for API unity.
"""

import os
from datetime import timedelta
from typing import List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..config import StrEnum
from ..exceptions import OLMoConfigurationError, OLMoEnvironmentError
from ..utils import get_default_device

OLMO_SHARED_FS_ENV_VAR = "OLMO_SHARED_FS"
OLMO_FS_LOCAL_RANK_ENV_VAR = "FS_LOCAL_RANK"
OLMO_LOCAL_RANK_ENV_VAR = "LOCAL_RANK"
OLMO_NUM_NODES_ENV_VAR = "NUM_NODES"
OLMO_LOCAL_WORLD_SIZE_ENV_VAR = "LOCAL_WORLD_SIZE"
BEAKER_HOSTNAME_ENV_VAR = "BEAKER_NODE_HOSTNAME"


def validate_env_vars():
    if not is_distributed():
        return

    if OLMO_LOCAL_RANK_ENV_VAR not in os.environ:
        raise OLMoEnvironmentError(f"Missing env var '{OLMO_LOCAL_RANK_ENV_VAR}'")

    if (
        os.environ.get(OLMO_SHARED_FS_ENV_VAR) != "1"
        and os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) is None
    ):
        raise OLMoEnvironmentError(
            f"Missing env var '{OLMO_FS_LOCAL_RANK_ENV_VAR}' for non-shared filesystem. "
            f"If this is a shared filesystem you can set '{OLMO_SHARED_FS_ENV_VAR}=1' instead."
        )

    if OLMO_NUM_NODES_ENV_VAR not in os.environ and OLMO_LOCAL_WORLD_SIZE_ENV_VAR not in os.environ:
        raise OLMoEnvironmentError(
            f"Missing either '{OLMO_NUM_NODES_ENV_VAR}' or '{OLMO_LOCAL_WORLD_SIZE_ENV_VAR}' env vars"
        )


def init_distributed(backend: str = "nccl", timeout: timedelta = timedelta(minutes=30)):
    """
    Initialize the distributed backend(s).
    """
    # to mitigate the memory issue that collectives using async_op=True hold memory longer
    # than they should such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # Force processes to synchronize at init process group.
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # Host-specific env vars.
    # See https://beaker-docs.apps.allenai.org/experiments/distributed-training.html#ai2pluto-cirrascale.
    if "jupiter" in get_node_hostname():
        os.environ["NCCL_SOCKET_IFNAME"] = "ib"
        os.environ["NCCL_IB_HCA"] = "^=mlx5_bond_0"
    elif "pluto" in get_node_hostname():
        os.environ["NCCL_SOCKET_IFNAME"] = "ib"
        os.environ["NCCL_IB_HCA"] = "^=mlx5_1,mlx5_2"

    validate_env_vars()

    dist.init_process_group(backend, timeout=timeout)

    if "nccl" in backend:
        # Set CUDA device.
        torch.cuda.set_device(f"cuda:{get_local_rank()}")


def get_node_hostname() -> str:
    """
    Get the hostname of the node.
    """
    if BEAKER_HOSTNAME_ENV_VAR in os.environ:
        return os.environ[BEAKER_HOSTNAME_ENV_VAR]
    else:
        import socket

        return socket.gethostname()


def is_distributed() -> bool:
    """
    Check if in a distributed context.
    """
    return dist.is_available() and dist.is_initialized()


def barrier(group: Optional[dist.ProcessGroup] = None) -> None:
    """
    Wait for all ranks in the group.
    """
    if is_distributed():
        dist.barrier(group)


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the rank within the process group.
    """
    if is_distributed():
        return dist.get_rank(group)
    else:
        return 0


def get_local_rank() -> int:
    """
    Get the local rank within the current node.

    .. warning::
        This relies on the environment variable ``LOCAL_RANK`` being set correctly.

    :returns: The rank.
    """
    if is_distributed():
        return int(os.environ.get(OLMO_LOCAL_RANK_ENV_VAR) or 0)
    else:
        return 0


def get_fs_local_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then :func:`get_fs_local_rank()` will be equivalent
    to :func:`get_rank()`, but if nodes do not share the same filesystem then
    :func:`get_fs_local_rank()` will be equivalent to :func:`get_local_rank()`.

    .. warning::
        This relies on some environment variables to determine the correct rank.
        If you are using a shared filesystem across nodes, you can simply set the environment
        variable ``OLMO_SHARED_FS=1``. Otherwise you can set ``FS_LOCAL_RANK`` for each process.

    :returns: The rank.
    """
    if not is_distributed():
        return 0
    elif os.environ.get(OLMO_SHARED_FS_ENV_VAR) == "1":
        return int(os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) or get_rank(group))
    else:
        return int(os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) or get_local_rank())


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the world size of the distributed process group.
    """
    if is_distributed():
        return dist.get_world_size(group)
    else:
        return 0


def get_local_world_size() -> int:
    """
    Get the local world size.

    .. warning::
        This relies on the 'LOCAL_WORLD_SIZE' env var.

    :returns: The local world size.
    """
    if not is_distributed():
        return 1
    else:
        return int(os.environ[OLMO_LOCAL_WORLD_SIZE_ENV_VAR])


def get_num_nodes() -> int:
    """
    Get the number of nodes.

    .. warning::
        This relies on either the 'NUM_NODES'or 'LOCAL_WORLD_SIZE' env var.

    :returns: The number of nodes.
    """
    if not is_distributed():
        return 1
    elif OLMO_NUM_NODES_ENV_VAR in os.environ:
        return int(os.environ[OLMO_NUM_NODES_ENV_VAR])
    else:
        return get_world_size() // get_local_world_size()


V = TypeVar("V", bool, int, float, torch.Tensor)


def synchronize_value(
    value: V, device: torch.device, src: int = 0, group: Optional[dist.ProcessGroup] = None
) -> V:
    """
    Synchronize a value across the distributed process group.
    """
    if dist.is_available() and dist.is_initialized():
        is_tensor = isinstance(value, torch.Tensor)
        value_tensor = value.to(device) if is_tensor else torch.tensor(value, device=device)  # type: ignore
        dist.broadcast(value_tensor, src, group=group)
        return value_tensor if is_tensor else value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(
    flag: bool, device: torch.device, group: Optional[dist.ProcessGroup] = None
) -> bool:
    """
    Synchronize a boolean across the distributed process group.
    """
    return synchronize_value(flag, device, group=group)


def all_reduce_value(
    value: V, device: torch.device, op=dist.ReduceOp.SUM, group: Optional[dist.ProcessGroup] = None
) -> V:
    """
    All reduce a value across the distributed process group.
    """
    if dist.is_available() and dist.is_initialized():
        is_tensor = isinstance(value, torch.Tensor)
        value_tensor = value.to(device) if is_tensor else torch.tensor(value, device=device)  # type: ignore
        dist.all_reduce(value_tensor, op=op, group=group)
        return value_tensor if is_tensor else value_tensor.item()  # type: ignore
    else:
        return value


T = TypeVar("T")


def scatter_object(obj: T, src: int = 0, group: Optional[dist.ProcessGroup] = None) -> T:
    """
    Scatter an object using pickle to all ranks in the process group.
    """
    if not is_distributed():
        return obj

    output_list: List[T] = [obj]
    input_list = [obj] * get_world_size(group) if get_rank(group) == src else None
    dist.scatter_object_list(output_list, input_list, src=src, group=group)
    return output_list[0]


def all_gather_object(obj: T, group: Optional[dist.ProcessGroup] = None) -> List[T]:
    """
    All-gather an object using pickle to all ranks in a process group.
    """
    if not is_distributed():
        return [obj]

    output_list = [obj] * get_world_size(group)
    dist.all_gather_object(output_list, obj, group=group)
    return output_list


def get_mesh_coordinates(mesh: "DeviceMesh", rank: Optional[int] = None) -> Optional[List[int]]:
    """
    Calculate the coordinates of a global rank on a device mesh.

    :param mesh: The device mesh.
    :param rank: The global rank. If ``None``, the current global rank is used.

    :return: The coordinates or ``None`` if the rank is not part of the mesh.
    """
    rank = rank if rank is not None else get_rank()
    rank_coords = (mesh.mesh == rank).nonzero()
    assert rank_coords.size(0) in (0, 1)
    return rank_coords[0].tolist() if rank_coords.size(0) > 0 else None


def backend_supports_cpu():
    if not is_distributed():
        return True

    backend = dist.get_backend()
    if "gloo" in backend or "mpi" in backend:
        return True
    else:
        return False


class HybridShardMeshDimName(StrEnum):
    replicas = "replicas"
    shards = "shards"


def init_hybrid_shard_mesh(
    num_replicas: Optional[int] = None, device_type: Optional[str] = None
) -> DeviceMesh:
    """
    Initialize a device mesh for FSDP hybrid sharding.
    """
    num_replicas = num_replicas or get_num_nodes()
    device_type = device_type or get_default_device().type

    if get_world_size() % num_replicas != 0:
        raise OLMoConfigurationError(
            "hybrid mesh requires world size to be divisible by 'num_replicas'"
        )

    return init_device_mesh(
        device_type,
        (num_replicas, get_world_size() // num_replicas),
        mesh_dim_names=(HybridShardMeshDimName.replicas, HybridShardMeshDimName.shards),
    )
