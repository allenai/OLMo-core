import gc
from typing import TYPE_CHECKING, List, Optional, TypeVar

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    if is_distributed():
        return dist.get_rank(group)
    else:
        return 0


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    if is_distributed():
        return dist.get_world_size(group)
    else:
        return 0


V = TypeVar("V", bool, int, float)


def synchronize_value(value: V, device: torch.device) -> V:
    if dist.is_available() and dist.is_initialized():
        value_tensor = torch.tensor(value, device=device)
        dist.broadcast(value_tensor, 0)
        return value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(flag: bool, device: torch.device) -> bool:
    return synchronize_value(flag, device)


def gc_cuda():
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


def get_gradient_divide_factor(world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
        factor *= 2
    return float(factor)


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
