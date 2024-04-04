from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter
from olmo_core.distributed.sharded_flat_tensor import ShardedFlatTensor, ShardingSpec
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.utils import get_default_device


@dataclass
class FlatParamHandle:
    """
    Manages the data for a group of sharded flat parameters in order to use a single all-reduce
    to unshard all of the parameters at once.
    """

    params: List[ShardedFlatParameter]
    """
    The params managed by this handle.
    """

    params_data: ShardedFlatTensor
    """
    Consolidated data for all of the local sharded data of the parameters.
    """

    params_offsets_per_rank: List[Dict[int, Tuple[int, int]]]
    """
    For each parameter, provides a mapping of rank to the offsets into the rank's local `params_data`
    for that parameter.
    """

    process_group: Optional[dist.ProcessGroup] = None

    @classmethod
    def collate_flat_params(
        cls,
        params: Iterable[ShardedFlatParameter],
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> FlatParamHandle:
        """
        Collate the data from a group of sharded flat parameters into a single flat param handle.
        """
        device = device or get_default_device()
        params = list(params)
        world_size = get_world_size(process_group)

        # Find max numel of all sharded flat params across the process group to determine padding.
        # All ranks will have the same sized `params_data` at the end to avoid needed padding at runtime.
        numel_total_per_rank: List[int] = [0] * world_size
        for param in params:
            if not param.is_sharded:
                raise ValueError("All sharded flat params should be sharded at this point!")
            if param.dtype != torch.float32:
                raise NotImplementedError("Only float32 params are supported at this time")
            for rank, n in enumerate(param.sharding_spec.sharded_numels):
                numel_total_per_rank[rank] += n
        max_numel = max(numel_total_per_rank)

        # Initialize local data for all params.
        params_data = ShardedFlatTensor(torch.empty(max_numel, device=device))
        params_data.mark_as_sharded(
            ShardingSpec(
                unsharded_shape=(world_size, max_numel),
                unsharded_flattened_offsets=tuple(
                    [
                        (start_idx, end_idx)
                        for start_idx, end_idx in zip(
                            range(0, max_numel * world_size, max_numel),
                            range(max_numel, max_numel * world_size + 1, max_numel),
                        )
                    ]
                ),
            ),
            process_group=process_group,
        )

        # Consolidate the sharded data from each param into `params_data` and collect offsets.
        params_offsets_per_rank: List[Dict[int, Tuple[int, int]]] = []
        offset_start_per_rank = {rank: 0 for rank in range(world_size)}
        for param in params:
            param_offsets: Dict[int, Tuple[int, int]] = {}
            for rank in range(world_size):
                offset_start = offset_start_per_rank[rank]
                offset_end = offset_start + param.sharding_spec.sharded_numels[rank]
                param_offsets[rank] = (offset_start, offset_end)
                offset_start_per_rank[rank] = offset_end
            params_offsets_per_rank.append(param_offsets)

            # Set data for param as a view into `params_data`.
            offset_start, offset_end = param_offsets[get_rank(process_group)]
            params_data.data[offset_start:offset_end].copy_(param.data)
            param.data = params_data.data[offset_start:offset_end]

        return cls(
            params=params,
            params_data=params_data,
            params_offsets_per_rank=params_offsets_per_rank,
            process_group=process_group,
        )

    def unshard_(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False):
        """
        Unshard the handle's managed flat parameters in-place.
        """
        local_rank = get_rank(self.process_group)
        world_size = get_world_size(self.process_group)
        all_params_unsharded_data = self.params_data.gather(dtype=dtype, rank0_only=rank0_only)
        for param, param_offsets in zip(self.params, self.params_offsets_per_rank):
            if rank0_only and local_rank != 0:
                param.unshard_(
                    unsharded_data=torch.empty_like(all_params_unsharded_data), dtype=dtype, rank0_only=rank0_only
                )
            else:
                unsharded_data = torch.empty(
                    param.sharding_spec.unsharded_flattened_shape, dtype=all_params_unsharded_data.dtype
                )
                for rank in range(world_size):
                    rank_local_data = all_params_unsharded_data[rank][
                        param_offsets[rank][0] : param_offsets[rank][1]
                    ]
                    unsharded_data[
                        param.sharding_spec.unsharded_flattened_offsets[rank][
                            0
                        ] : param.sharding_spec.unsharded_flattened_offsets[rank][1]
                    ] = rank_local_data
                param.unshard_(unsharded_data=unsharded_data, dtype=dtype)
        del all_params_unsharded_data

    def reshard_(self, writeback: bool = False):
        """
        Reshard the handle's managed flat parameters in-place.
        """
        for param in self.params:
            param.reshard_(writeback=writeback)
