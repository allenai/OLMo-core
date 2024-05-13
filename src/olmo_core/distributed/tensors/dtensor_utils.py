"""
Helper functions for dealing with PyTorch's :class:`DTensor`.
"""

from typing import Optional, Sequence, Tuple

from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Placement, Shard
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.utils import ShapeType

from ..utils import get_mesh_coordinates


def get_local_shape_and_global_offset(
    dtensor: DTensor, rank: Optional[int] = None
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Like :func:`compute_local_shape_and_global_offset`, but acts directly on a :class:`DTensor`
    instance.

    :param dtensor: A DTensor instance.
    :param rank: The global rank to compute the local shape and global offsets for. If ``None``,
        defaults to the current rank.

    :returns: The local shape and global offset.
    """
    global_shape = dtensor.shape
    mesh = dtensor.device_mesh
    placements = dtensor.placements
    local_shape, global_offset = compute_local_shape_and_global_offset(global_shape, mesh, placements, rank=rank)
    return local_shape, global_offset


# Adapted from `torch.distributed._tensor._utils.py`.
def compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    rank: Optional[int] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    :param global_shape: The shape of the global unsharded tensor.
    :param mesh: The device mesh.
    :param placements: The placements of the :class:`DTensor`.
    :param rank: The global rank to compute the local shape and global offsets for. If ``None``,
        defaults to the current rank.

    :returns: The local shape and global offset.

    Example (2 host with 4GPUs each)::

        # Below is a DeviceMesh with mesh_shape of (2, 4)
        mesh = DeviceMesh(device_type="cuda", mesh=[
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ])

    Let's say we distribute a global_tensor of shape ``(8,4)`` over the above DeviceMesh
    with a placements of ``[Shard(0), Shard(0)]``.

    The local shape and global offset will be as follows:

    - ``rank0 -- local_shape:[1, 4], global_offset:[0, 0]``
    - ``rank1 -- local_shape:[1, 4], global_offset:[1, 0]``
    - ``rank2 -- local_shape:[1, 4], global_offset:[2, 0]``
    - ``rank5 -- local_shape:[1, 4], global_offset:[5, 0]``
    - ``rank3 -- local_shape:[1, 4], global_offset:[3, 0]``
    - ``rank4 -- local_shape:[1, 4], global_offset:[4, 0]``
    - ``rank6 -- local_shape:[1, 4], global_offset:[6, 0]``
    - ``rank7 -- local_shape:[1, 4], global_offset:[7, 0]``

    Let's say we distribute a global_tensor of shape ``(2,)`` over the above DeviceMesh with
    a placements of ``[Shard(0)]``. We will not have non-empty local tensor for all the ranks.

    The local shape and global offset will be as follows:

    - ``rank0 -- local_shape:[1,], global_offset:[0,]``
    - ``rank1 -- local_shape:[1,], global_offset:[1,]``
    - ``rank2 -- local_shape:[0,], global_offset:[2,]``
    - ``rank5 -- local_shape:[0,], global_offset:[2,]``
    - ``rank3 -- local_shape:[0,], global_offset:[2,]``
    - ``rank4 -- local_shape:[0,], global_offset:[2,]``
    - ``rank6 -- local_shape:[0,], global_offset:[2,]``
    - ``rank7 -- local_shape:[0,], global_offset:[2,]``
    """
    my_coordinate = mesh.get_coordinate() if rank is None else get_mesh_coordinates(mesh, rank)

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )

                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                # it means that this dimension has been already sharded in previous placement.
                # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

        return tuple(local_shape), tuple(global_offset)
