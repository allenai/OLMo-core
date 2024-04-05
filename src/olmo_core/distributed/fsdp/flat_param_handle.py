from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from olmo_core.distributed.tensors import (
    ShardedFlatParameter,
    ShardedFlatTensor,
    ShardingSpec,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.utils import get_default_device


@dataclass
class FlatParamHandle:
    """
    Manages the data for a group of sharded flat parameters in order to use a single all-reduce
    to unshard all of the parameters at once.
    """

    params: List[ShardedFlatParameter] = field(default_factory=list)
    """
    The params managed by this handle.
    """

    param_fqns: List[str] = field(default_factory=list)
    """
    The FQNs of the managed params.
    """

    grads: List[Optional[torch.Tensor]] = field(default_factory=list)
    """
    Used for caching gradients during gradient accumulation.
    """

    params_data: ShardedFlatTensor = field(default_factory=lambda: ShardedFlatTensor(torch.empty(0)))
    """
    Consolidated data for all of the local sharded data of the parameters.
    """

    params_offsets_per_rank: List[Dict[int, Tuple[int, int]]] = field(default_factory=list)
    """
    For each parameter, provides a mapping of rank to the offsets into the rank's local `params_data`
    for that parameter.
    """

    process_group: Optional[dist.ProcessGroup] = None

    device: Optional[torch.device] = None

    @classmethod
    def collate_flat_params(
        cls,
        params: Iterable[ShardedFlatParameter],
        param_fqns: Iterable[str],
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> FlatParamHandle:
        """
        Collate the data from a group of sharded flat parameters into a single flat param handle.
        """
        device = device or get_default_device()
        params = list(params)
        world_size = get_world_size(process_group)
        local_rank = get_rank(process_group)

        if not params:
            return cls(process_group=process_group)

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
            offset_start, offset_end = param_offsets[local_rank]
            params_data.data[offset_start:offset_end].copy_(param.data)
            param.data = params_data.data[offset_start:offset_end]

        return cls(
            params=params,
            param_fqns=list(param_fqns),
            grads=[None] * len(params),
            params_data=params_data,
            params_offsets_per_rank=params_offsets_per_rank,
            process_group=process_group,
            device=device,
        )

    def unshard_(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False, cache_grads: bool = False):
        """
        Unshard the handle's managed flat parameters in-place.
        """
        if not self.params:
            return
        local_rank = get_rank(self.process_group)
        world_size = get_world_size(self.process_group)
        all_params_unsharded_data = self.params_data.gather(dtype=dtype, rank0_only=rank0_only)
        for i, (param, param_offsets) in enumerate(zip(self.params, self.params_offsets_per_rank)):
            if rank0_only and local_rank != 0:
                param.unshard_(
                    unsharded_data=torch.empty_like(all_params_unsharded_data), dtype=dtype, rank0_only=rank0_only
                )
            else:
                unsharded_data = torch.empty(
                    param.sharding_spec.unsharded_flattened_shape,
                    dtype=all_params_unsharded_data.dtype,
                    device=self.device,
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

            if cache_grads and param.grad is not None:
                # We should only be caching these between the pre-backward and post-backward
                # hooks. The post-backward hook will remove the cached grad as it accumulates
                # it into the persistent sharded grad.
                assert self.grads[i] is None
                self.grads[i] = param.grad.data
                param.grad = None

        del all_params_unsharded_data

    def reshard_(self, writeback: bool = False):
        """
        Reshard the handle's managed flat parameters in-place.
        """
        local_rank = get_rank(self.process_group)
        for param, param_offsets in zip(self.params, self.params_offsets_per_rank):
            param.reshard_(writeback=writeback)
            if writeback:
                offset_start, offset_end = param_offsets[local_rank]
                self.params_data.data[offset_start:offset_end].copy_(param.data)
                param.data = self.params_data.data[offset_start:offset_end]

    def reduce_scatter_grads(
        self, grad_dtype: Optional[torch.dtype] = None, grad_reduce_dtype: Optional[torch.dtype] = None
    ):
        for i, param in enumerate(self.params):
            if (unsharded_grad := param.grad) is None:
                continue

            if grad_reduce_dtype is not None:
                unsharded_grad = unsharded_grad.to(dtype=grad_reduce_dtype)

            if grad_dtype is None:
                grad_dtype = param.dtype

            # TODO: batch reductions together

            # Only NCCL supports 'reduce_scatter'. So with other backends we use 'all_reduce'.
            if dist.get_backend() == dist.Backend.NCCL:
                # Get chunks corresponding to each rank.
                grad_chunks = param.chunk_unsharded(unsharded_grad, pad=True)
                new_sharded_grad = torch.empty_like(grad_chunks[0])
                dist.reduce_scatter(new_sharded_grad, grad_chunks, group=self.process_group)
                param.grad = new_sharded_grad[: param.unsharded_flattened_offsets[1]].to(dtype=grad_dtype)
            else:
                dist.all_reduce(unsharded_grad, group=self.process_group)
                param.grad = param.sharded_chunk(unsharded_grad).detach().to(dtype=grad_dtype)

            del unsharded_grad

            if (cached_grad := self.grads[i]) is not None:
                param.grad.add_(cached_grad)
                self.grads[i] = None
                del cached_grad
