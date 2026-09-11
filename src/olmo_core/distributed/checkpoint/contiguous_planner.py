"""Constant-memory metadata for ordinary contiguous DTensor shards.

Some PyTorch versions materialize an index vector per shard to find its first offset.
These helpers avoid that allocation for concrete Shard/Replicate layouts only; all other
layouts retain the default planner. No tensor data or global PyTorch state is changed.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch.distributed.checkpoint.default_planner import create_default_local_save_plan
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    SavePlan,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement


def contiguous_shard_metadata(
    shape: Sequence[int],
    mesh_shape: Sequence[int],
    coordinate: Sequence[int],
    placements: Sequence[Placement],
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return exact chunk sizes/offsets, or ``None`` for unsupported/symbolic layouts."""
    if (
        len(mesh_shape) != len(coordinate)
        or len(mesh_shape) != len(placements)
        or any(type(p) not in (Shard, Replicate) for p in placements)
        or any(type(n) is not int or n < 0 for n in shape)
        or any(type(n) is not int or n <= 0 for n in mesh_shape)
        or any(type(c) is not int or not 0 <= c < n for c, n in zip(coordinate, mesh_shape))
    ):
        return None
    sizes, offsets = list(shape), [0] * len(shape)
    for chunks, rank, placement in zip(mesh_shape, coordinate, placements):
        if type(placement) is Replicate:
            continue
        assert isinstance(placement, Shard)
        dim = placement.dim
        if type(dim) is not int or not 0 <= dim < len(shape):
            return None
        width = (sizes[dim] + chunks - 1) // chunks
        offset = min(rank * width, sizes[dim])
        sizes[dim] = min(width, sizes[dim] - offset)
        # PyTorch uses the global dimension size as the offset for an empty shard,
        # including when sharding the same dimension again on a second mesh axis.
        offsets[dim] = offsets[dim] + offset if sizes[dim] else shape[dim]
    return tuple(sizes), tuple(offsets)


def create_contiguous_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool) -> SavePlan:
    """Create an ordinary DCP plan, using bounded metadata arithmetic where applicable."""
    items = []
    for name, value in state_dict.items():
        if isinstance(value, DTensor):
            coordinate = value.device_mesh.get_coordinate()
            if coordinate is None:
                continue
            metadata = contiguous_shard_metadata(
                value.shape, value.device_mesh.shape, coordinate, value.placements
            )
            local = value.to_local()
            # Nested/custom local tensors may define their own DCP metadata hooks.
            if metadata is not None and type(local) is torch.Tensor:
                sizes, offsets = map(torch.Size, metadata)
                if sizes != local.size():
                    raise ValueError(
                        f"Checkpoint shard shape mismatch for {name}: {sizes} != {local.size()}"
                    )
                items.append(
                    WriteItem(
                        index=MetadataIndex(name, offsets),
                        type=WriteItemType.SHARD,
                        tensor_data=TensorWriteData(
                            chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
                            properties=TensorProperties.create_from_tensor(local),
                            size=value.size(),
                        ),
                    )
                )
                continue
        items.extend(create_default_local_save_plan({name: value}, is_coordinator).items)
    return SavePlan(items)
