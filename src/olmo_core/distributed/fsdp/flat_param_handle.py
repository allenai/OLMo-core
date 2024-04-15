from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.distributed.tensors import (
    ShardedFlatParameter,
    ShardedFlatTensor,
    ShardingSpec,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.utils import get_default_device

log = logging.getLogger(__name__)


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

    grads_cache: List[Optional[torch.Tensor]] = field(default_factory=list)
    """
    Used for caching gradients during gradient accumulation.
    """

    params_data: ShardedFlatTensor = field(default_factory=lambda: ShardedFlatTensor(torch.empty(0)))
    """
    Consolidated data for all of the local sharded data of the parameters including padding.
    """

    process_group: Optional[dist.ProcessGroup] = None

    device: Optional[torch.device] = None

    @classmethod
    def shard_params(
        cls,
        params: Iterable[nn.Parameter],
        param_fqns: Iterable[str],
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> FlatParamHandle:
        """
        Shard and flatten parameters and collect into a :class:`FlatParamHandle`.
        """
        device = device or get_default_device()
        world_size = get_world_size(process_group)
        local_rank = get_rank(process_group)
        params = list(params)

        if not params:
            params_data = ShardedFlatTensor(torch.empty(0, device=device))
            params_data.mark_as_sharded(
                ShardingSpec(unsharded_shape=(0,), unsharded_flattened_offsets=tuple([(0, 0)] * world_size))
            )
            return FlatParamHandle(process_group=process_group, device=device)

        total_numel = sum(p.numel() for p in params)
        padded_sharded_numel = math.ceil(total_numel / world_size)
        padded_unsharded_numel = padded_sharded_numel * world_size

        # The idea is to flatten and concatenate all (unsharded) original parameters together to form
        # the unsharded, unpadded flat parameter.
        #
        # For example, suppose we have 3 parameters that, when flattened, look like this:
        #
        #  x = (x1, x2, x3, x4)
        #  y = (y1, y2, y3)
        #  z = (z1, z2, z3, z4)
        #
        # And suppose we have a world size of 4. Then the padded unsharded flat parameter would look
        # like this:
        #
        #   |x1 x2 x3|x4 y1 y2|y3 z1 z2|z3 z4 0 |
        #   | rank 0 | rank 1 | rank 2 | rank 3 |
        #    1  2  3  4  5  6  7  8  9  10 11 12
        #
        # Now first we need to initialize a flat parameter to take the place of each regular parameter.
        flat_params: List[ShardedFlatParameter] = []
        numel_running_total = 0
        for param in params:
            flat_param_global_offsets = (numel_running_total, numel_running_total + param.numel())

            # First we need to determine which ranks will have a slice of the data.
            unsharded_flattened_offsets: List[Tuple[int, int]] = []
            for rank in range(world_size):
                rank_global_start = rank * padded_sharded_numel
                rank_global_end = rank_global_start + padded_sharded_numel
                if (rank_global_end <= flat_param_global_offsets[0]) or (
                    flat_param_global_offsets[1] <= rank_global_start
                ):
                    # No overlap with this rank.
                    unsharded_flattened_offsets.append((0, 0))
                elif (
                    rank_global_start <= flat_param_global_offsets[0]
                    and flat_param_global_offsets[1] <= rank_global_end
                ):
                    # Param is completely contained by this rank.
                    unsharded_flattened_offsets.append((0, param.numel()))
                elif (
                    rank_global_start <= flat_param_global_offsets[0]
                    and rank_global_end < flat_param_global_offsets[1]
                ):
                    # Param starts in this rank and ends in a subsequent rank.
                    unsharded_flattened_offsets.append((0, rank_global_end - flat_param_global_offsets[0]))
                elif (
                    flat_param_global_offsets[0] < rank_global_start
                    and flat_param_global_offsets[1] <= rank_global_end
                ):
                    # Param starts in a previous rank and ends in this one.
                    unsharded_flattened_offsets.append(
                        (rank_global_start - flat_param_global_offsets[0], param.numel())
                    )
                elif (
                    flat_param_global_offsets[0] < rank_global_start
                    and rank_global_end < flat_param_global_offsets[1]
                ):
                    # Param spans this rank and overflows into other ranks.
                    unsharded_flattened_offsets.append(
                        (
                            rank_global_start - flat_param_global_offsets[0],
                            rank_global_end - flat_param_global_offsets[0],
                        )
                    )

            sharding_spec = ShardingSpec(
                unsharded_shape=tuple(param.shape), unsharded_flattened_offsets=tuple(unsharded_flattened_offsets)
            )
            flat_param: ShardedFlatParameter
            if (local_rank_numel := sharding_spec.sharded_numels[local_rank]) > 0:
                if param.device == torch.device("meta"):
                    flat_param = ShardedFlatParameter(torch.empty(local_rank_numel, device=device))
                else:
                    flat_param = ShardedFlatParameter(
                        param.data.flatten()[
                            unsharded_flattened_offsets[local_rank][0] : unsharded_flattened_offsets[local_rank][1]
                        ].to(device)
                    )
            else:
                flat_param = ShardedFlatParameter(torch.empty(0, device=device))
            flat_param.mark_as_sharded(sharding_spec, process_group=process_group)

            flat_params.append(flat_param)

            numel_running_total += param.numel()
            param.data = torch.empty(0, device=param.device)

        # Now that we have all of the flat parameters we need to collate all of their data into a single
        # sharded flat tensor, then set the data for each flat parameter as a view into that flat tensor.
        local_flat_sharded_data = torch.cat([flat_param.data for flat_param in flat_params])
        params_data = ShardedFlatTensor(
            F.pad(local_flat_sharded_data, (0, padded_sharded_numel - local_flat_sharded_data.numel()))
        )
        del local_flat_sharded_data
        params_data.mark_as_sharded(
            ShardingSpec(
                unsharded_shape=(padded_unsharded_numel,),
                unsharded_flattened_offsets=tuple(
                    [
                        (start_idx, end_idx)
                        for start_idx, end_idx in zip(
                            range(0, padded_unsharded_numel, padded_sharded_numel),
                            range(padded_sharded_numel, padded_unsharded_numel + 1, padded_sharded_numel),
                        )
                    ]
                ),
            ),
            process_group=process_group,
        )
        offset = 0
        for flat_param in flat_params:
            flat_param.data = params_data[offset : offset + flat_param.numel()]
            offset += flat_param.numel()

        return cls(
            params=flat_params,
            param_fqns=list(param_fqns),
            grads_cache=[None] * len(flat_params),
            params_data=params_data,
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

        # Gather full, padded, unsharded data for all params.
        all_params_unsharded_data: torch.Tensor
        if rank0_only or dist.get_backend() == dist.Backend.GLOO:
            all_params_unsharded_data = self.params_data.gather(dtype=dtype, rank0_only=rank0_only)
        else:
            # We prefer to use `all_gather_into_tensor()` directly when possible as it involves
            # fewer allocations.
            all_params_unsharded_data = torch.empty(
                self.params_data.unsharded_shape, dtype=dtype or self.params_data.dtype, device=self.device
            )
            dist.all_gather_into_tensor(
                all_params_unsharded_data,
                self.params_data.data.to(dtype or self.params_data.dtype),
                group=self.process_group,
            )
        self.params_data.unshard_(unsharded_data=all_params_unsharded_data, dtype=dtype, rank0_only=rank0_only)

        # Set the data for each param as a view into `all_params_unsharded_data`.
        numel_running_total = 0
        for i, param in enumerate(self.params):
            if rank0_only and local_rank != 0:
                param.unshard_(
                    unsharded_data=torch.empty_like(all_params_unsharded_data), dtype=dtype, rank0_only=rank0_only
                )
            else:
                unsharded_data = all_params_unsharded_data[
                    numel_running_total : numel_running_total + param.unsharded_numel
                ]
                param.unshard_(unsharded_data, dtype=dtype, rank0_only=rank0_only)
            numel_running_total += param.unsharded_numel

            if cache_grads and param.grad is not None:
                # We should only be caching these between the pre-backward and post-backward
                # hooks. The post-backward hook will remove the cached grad as it accumulates
                # it into the persistent sharded grad.
                assert self.grads_cache[i] is None
                self.grads_cache[i] = param.grad.data
                param.grad = None

        del all_params_unsharded_data

    def reshard_(self, writeback: bool = False):
        """
        Reshard the handle's managed flat parameters in-place.
        """
        if not self.params:
            return

        self.params_data.reshard_(writeback=writeback)
        offset = 0
        for flat_param in self.params:
            flat_param.reshard_(writeback=False)
            if writeback:
                # Reset the view into the new `params_data`.
                flat_param.data = self.params_data[offset : offset + flat_param.numel()]
            offset += flat_param.numel()

    def reduce_scatter_grads(
        self, grad_dtype: Optional[torch.dtype] = None, grad_reduce_dtype: Optional[torch.dtype] = None
    ):
        local_rank = get_rank(self.process_group)
        for i, param in enumerate(self.params):
            if (unsharded_grad := param.grad) is None:
                continue

            if grad_reduce_dtype is not None:
                unsharded_grad = unsharded_grad.to(dtype=grad_reduce_dtype)

            if grad_dtype is None:
                grad_dtype = param.dtype

            # TODO: batch reductions together?

            # Only NCCL supports 'reduce_scatter'. So with other backends we use 'all_reduce'.
            if dist.get_backend() == dist.Backend.NCCL:
                # Get chunks corresponding to each rank.
                grad_chunks = param.chunk_unsharded(unsharded_grad)
                new_sharded_grad = torch.empty_like(grad_chunks[local_rank])
                dist.reduce_scatter(new_sharded_grad, grad_chunks, group=self.process_group)
                param.grad = new_sharded_grad.to(dtype=grad_dtype)
            else:
                dist.all_reduce(unsharded_grad, group=self.process_group)
                param.grad = param.sharded_chunk(unsharded_grad).detach().to(dtype=grad_dtype)

            del unsharded_grad

            if (cached_grad := self.grads_cache[i]) is not None:
                param.grad.add_(cached_grad)
                self.grads_cache[i] = None
                del cached_grad
