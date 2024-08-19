"""
Distributed helpers, most of which work in a non-distributed context as well for API unity.
"""

import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, TypeVar

import torch
import torch.distributed as dist

from olmo_core.exceptions import OLMoEnvironmentError
from olmo_core.io import PathOrStr, clear_directory

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

OLMO_SHARED_FS_ENV_VAR = "OLMO_SHARED_FS"
OLMO_FS_LOCAL_RANK_ENV_VAR = "FS_LOCAL_RANK"
OLMO_LOCAL_RANK_ENV_VAR = "LOCAL_RANK"


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
    Get the local rank within the current node. Relies on the environment variable "LOCAL_RANK".
    """
    if is_distributed():
        return int(os.environ.get(OLMO_LOCAL_RANK_ENV_VAR) or 0)
    else:
        return 0


def get_fs_local_rank(
    *, dir: Optional[PathOrStr] = None, group: Optional[dist.ProcessGroup] = None
) -> int:
    """
    Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then :func:`get_fs_local_rank()` will be equivalent
    to :func:`get_rank()`, but if nodes do not share the same filesystem then
    :func:`get_fs_local_rank()` will be equivalent to :func:`get_local_rank()`.
    """
    if os.environ.get(OLMO_SHARED_FS_ENV_VAR) == "1":
        return int(os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) or get_rank(group))
    elif dir is not None:
        global _fs_local_rank_check_results

        if not Path(dir).is_dir():
            raise FileNotFoundError(dir)

        rank_check_dir = Path(dir) / ".rank_check"
        rank_check_dir.mkdir(exist_ok=True)
        rank_check_file = rank_check_dir / f"rank{get_rank()}.tmp"
        rank_check_file.touch()

        # Wait for all ranks to write their file.
        barrier()

        # Find FS local rank.
        local_ranks = [
            int(p.name.replace("rank", "").replace(".tmp", "")) for p in rank_check_dir.iterdir()
        ]
        fs_local_rank = get_rank() - min(local_ranks)

        # Clean up.
        barrier()
        if fs_local_rank == 0:
            clear_directory(rank_check_dir)

        return fs_local_rank
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


def validate_env_vars():
    if (
        os.environ.get(OLMO_SHARED_FS_ENV_VAR) != "1"
        and os.environ.get(OLMO_FS_LOCAL_RANK_ENV_VAR) is None
        and os.environ.get(OLMO_LOCAL_RANK_ENV_VAR) is None
    ):
        raise OLMoEnvironmentError(
            f"Missing env var '{OLMO_FS_LOCAL_RANK_ENV_VAR}' or '{OLMO_LOCAL_RANK_ENV_VAR}' for non-shared filesystem"
        )


V = TypeVar("V", bool, int, float)


def synchronize_value(
    value: V, device: torch.device, group: Optional[dist.ProcessGroup] = None
) -> V:
    """
    Synchronize an object across the distributed process group. The object can be anything that's
    serializable.
    """
    if dist.is_available() and dist.is_initialized():
        value_tensor = torch.tensor(value, device=device)
        dist.broadcast(value_tensor, 0, group=group)
        return value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(
    flag: bool, device: torch.device, group: Optional[dist.ProcessGroup] = None
) -> bool:
    """
    Synchronize a boolean across the distributed process group.
    """
    return synchronize_value(flag, device, group=group)


def gc_cuda():
    """
    Run garbage collection, including CUDA.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
