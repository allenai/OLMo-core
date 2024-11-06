"""
Distributed helpers, most of which work in a non-distributed context as well for API unity.
"""

import logging
import os
from datetime import timedelta
from typing import List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..config import StrEnum
from ..exceptions import OLMoConfigurationError, OLMoEnvironmentError
from ..utils import get_default_device, logging_configured, move_to_device, set_env_var

OLMO_SHARED_FS_ENV_VAR = "OLMO_SHARED_FS"
OLMO_FS_LOCAL_RANK_ENV_VAR = "FS_LOCAL_RANK"
OLMO_LOCAL_RANK_ENV_VAR = "LOCAL_RANK"
OLMO_NUM_NODES_ENV_VAR = "NUM_NODES"
OLMO_LOCAL_WORLD_SIZE_ENV_VAR = "LOCAL_WORLD_SIZE"
BEAKER_HOSTNAME_ENV_VAR = "BEAKER_NODE_HOSTNAME"


log = logging.getLogger(__name__)


def init_distributed(backend: str = "nccl", timeout: timedelta = timedelta(minutes=30)):
    """
    Initialize the distributed process group with the given backend(s) and check/set the
    relevant environment variables.
    This also calls :func:`torch.cuda.set_device()` for backends that support CUDA.
    """
    # To mitigate the memory issue that collectives using async_op=True hold memory longer
    # than they should such as those in tensor parallelism.
    set_env_var("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")

    # Force processes to synchronize at init process group.
    set_env_var("TORCH_DIST_INIT_BARRIER", "1")

    # Set host-specific env var defaults.
    if _running_in_beaker():
        multi_node = int(os.environ.get(OLMO_NUM_NODES_ENV_VAR, "1")) > 1
        # See https://beaker-docs.apps.allenai.org/experiments/distributed-training.html
        if "jupiter" in get_node_hostname():
            set_env_var("NCCL_IB_HCA", "^=mlx5_bond_0")
            if multi_node:
                # Only for multi-node
                set_env_var("NCCL_SOCKET_IFNAME", "ib")
        elif "pluto" in get_node_hostname():
            set_env_var("NCCL_IB_HCA", "^=mlx5_1,mlx5_2")
        elif "augusta" in get_node_hostname():
            set_env_var("NCCL_CROSS_NIC", "0")
            set_env_var("NCCL_ALGO", "Ring,Tree")
            set_env_var("NCCL_PROTO", "Simple")
            set_env_var("NCCL_MIN_NCHANNELS", "4")
            set_env_var("NCCL_P2P_NET_CHUNKSIZE", "524288")
            set_env_var("NCCL_P2P_PCI_CHUNKSIZE", "524288")
            set_env_var("NCCL_P2P_NVL_CHUNKSIZE", "1048576")
            set_env_var("NCCL_FASTRAK_NUM_FLOWS", "2")
            set_env_var("NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL", "0")
            set_env_var("NCCL_BUFFSIZE", "8388608")
            set_env_var("NCCL_FASTRAK_USE_SNAP", "1")
            set_env_var("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
            set_env_var("NCCL_NET_GDR_LEVEL", "PIX")
            set_env_var("NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING", "0")
            set_env_var("NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS", "600000")
            set_env_var("NCCL_NVLS_ENABLE", "0")
            set_env_var("NCCL_FASTRAK_CTRL_DEV", "enp0s12")
            set_env_var(
                "NCCL_FASTRAK_IFNAME",
                "enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
            )
            set_env_var("NCCL_USE_SNAP", "1")
            set_env_var("NCCL_FASTRAK_USE_LLCM", "1")
            set_env_var("NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY", "/dev/aperture_devices")
            # NOTE: This path var must be set prior to launching Python
            #  set_env_var(
            #      "LD_LIBRARY_PATH",
            #      "/var/lib/tcpxo/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
            #      override=True,
            #  )
            set_env_var("NCCL_TUNER_PLUGIN", "libnccl-tuner.so")
            set_env_var(
                "NCCL_TUNER_CONFIG_PATH", "/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto"
            )
            set_env_var(
                "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE",
                "/var/lib/tcpxo/lib64/a3plus_guest_config.textproto",
            )
            if multi_node:
                set_env_var("NCCL_SOCKET_IFNAME", "enp0s12")

    if backend_supports_cuda(backend):
        # Set CUDA device.
        # NOTE: important to do this *before* initializing the process group to avoid
        # other ranks initializing CUDA on GPU 0.
        device = torch.device(f"cuda:{int(os.environ[OLMO_LOCAL_RANK_ENV_VAR])}")
        torch.cuda.set_device(device)

    dist.init_process_group(backend, timeout=timeout)

    validate_env_vars()

    msg = (
        f"Global rank {get_rank()} "
        f"= local rank {get_local_rank()} "
        f"= file system local rank {get_fs_local_rank()}"
    )
    if logging_configured():
        log.warning(msg)
    else:
        print(msg)


def validate_env_vars():
    """
    Validate distributed environment variables. This is called internally by :func:`init_distributed()`.
    """
    if not is_distributed():
        return

    if OLMO_LOCAL_RANK_ENV_VAR not in os.environ:
        raise OLMoEnvironmentError(f"Missing env var '{OLMO_LOCAL_RANK_ENV_VAR}'")

    if os.environ.get(OLMO_SHARED_FS_ENV_VAR) != "1" and (
        os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) is None
        and os.environ.get(OLMO_LOCAL_RANK_ENV_VAR) is None
    ):
        raise OLMoEnvironmentError(
            f"Missing env var '{OLMO_FS_LOCAL_RANK_ENV_VAR}'/'{OLMO_LOCAL_RANK_ENV_VAR}' for non-shared filesystem. "
            f"If this is a shared filesystem you can set '{OLMO_SHARED_FS_ENV_VAR}=1' instead."
        )

    if OLMO_NUM_NODES_ENV_VAR not in os.environ and OLMO_LOCAL_WORLD_SIZE_ENV_VAR not in os.environ:
        raise OLMoEnvironmentError(
            f"Missing either '{OLMO_NUM_NODES_ENV_VAR}' or '{OLMO_LOCAL_WORLD_SIZE_ENV_VAR}' env vars"
        )


def _running_in_beaker() -> bool:
    return BEAKER_HOSTNAME_ENV_VAR in os.environ


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
        This relies on the environment variable ``LOCAL_RANK`` being set correctly, but
        will always return 0 if a distributed process group has not been initialized.

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

        This will always return 0 if a distributed process group has not been initialized.

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
    Get the world size of the default distributed process group.

    .. warning::
        This will always return 1 if a distributed group has not been initialized.
    """
    if is_distributed():
        return dist.get_world_size(group)
    else:
        return 0


def get_local_world_size() -> int:
    """
    Get the local world size within the default distributed process group.

    .. warning::
        This relies on the 'LOCAL_WORLD_SIZE' env var but will always return 1 if a distributed
        process group has not been initialized.

    :returns: The local world size.
    """
    if not is_distributed():
        return 1
    else:
        return int(os.environ[OLMO_LOCAL_WORLD_SIZE_ENV_VAR])


def get_num_nodes() -> int:
    """
    Get the number of nodes in the default distributed process group.

    .. warning::
        This relies on either the 'NUM_NODES' or 'LOCAL_WORLD_SIZE' env var, but will always
        return 1 if a distributed process group has not been initialized.

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
        value_tensor = move_to_device(value, device) if is_tensor else move_to_device(torch.tensor(value), device)  # type: ignore
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
        value_tensor = move_to_device(value, device) if is_tensor else move_to_device(torch.tensor(value), device)  # type: ignore
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


def all_gather(
    tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None
) -> List[torch.Tensor]:
    """
    All-gather tensors from the whole group into a list.
    """
    if not is_distributed():
        return [tensor]

    shapes = all_gather_object(tensor.shape, group=group)
    output_list = [
        move_to_device(torch.zeros(shape, dtype=tensor.dtype), tensor.device) for shape in shapes
    ]
    dist.all_gather(output_list, tensor, group=group)
    return output_list


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


def backend_supports_cuda(backend: Optional[str] = None) -> bool:
    """
    Check if a distributed backend supports CUDA tensors.
    """
    if backend is None and not is_distributed():
        return torch.cuda.is_available()

    backend = backend or dist.get_backend()
    if "nccl" in backend:
        return True
    else:
        return False


def backend_supports_cpu(backend: Optional[str] = None) -> bool:
    """
    Check if a distributed backend supports CPU tensors.
    """
    if backend is None and not is_distributed():
        return True

    backend = backend or dist.get_backend()
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

    :param num_replicas: The number of model replicas. Defaults to the number of nodes in the main
        process group.
    :param device_type: The device type for the mesh.
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


def get_reduce_divide_factor(world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
        factor *= 2
    return float(factor)


def get_local_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        return x.to_local()
    else:
        return x
