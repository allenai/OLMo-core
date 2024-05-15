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
from olmo_core.distributed.utils import (
    get_gradient_divide_factor,
    get_rank,
    get_world_size,
)
from olmo_core.stream import Stream
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
    The params managed by this handle. The data for each of these params will be a view into ``params_data``.
    """

    param_fqns: List[str] = field(default_factory=list)
    """
    The FQNs of the managed params.
    """

    params_data: ShardedFlatTensor = field(default_factory=lambda: ShardedFlatTensor(torch.empty(0)))
    """
    Consolidated data for all of the local sharded data of the parameters including padding.
    """

    params_sharded_data_lp: Optional[torch.Tensor] = None
    """
    Low-precision version of sharded ``params_data``.
    """

    params_unsharded_grad: Optional[torch.Tensor] = None
    """
    Consolidated unsharded grads for all of the local sharded flat parameters. When initialized this will have
    the same shape as the unsharded version of ``params_data``.
    """

    params_sharded_grad: Optional[torch.Tensor] = None
    """
    Consolidated sharded grads for all of the local sharded flat parameters. When initialized this will have
    the same shape as the sharded version of ``params_data``.
    """

    process_group: Optional[dist.ProcessGroup] = None
    """
    Process group containing all shards.
    """

    inter_group_process_group: Optional[dist.ProcessGroup] = None
    """
    Process group for between-group reductions with hybrid sharding.
    """

    device: Optional[torch.device] = None

    requires_grad: bool = True

    pre_reduce_grad_divide_factor: float = 1.0

    post_reduce_grad_divide_factor: float = 1.0

    _ran_pre_unshard: bool = False

    _ran_pre_reduce_scatter_grads: bool = False

    def __post_init__(self):
        data_parallel_world_size = get_world_size(self.process_group)
        if self.inter_group_process_group is not None:
            data_parallel_world_size *= self.inter_group_process_group.size()
        self.pre_reduce_grad_divide_factor = get_gradient_divide_factor(data_parallel_world_size)
        self.post_reduce_grad_divide_factor = data_parallel_world_size / self.pre_reduce_grad_divide_factor

    @classmethod
    def shard_params(
        cls,
        params: Iterable[nn.Parameter],
        param_fqns: Iterable[str],
        process_group: Optional[dist.ProcessGroup] = None,
        inter_group_process_group: Optional[dist.ProcessGroup] = None,
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
                ShardingSpec(unsharded_shape=(0,), unsharded_flattened_offsets=tuple([((0, 0),)] * world_size))
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
        requires_grad = None
        numel_running_total = 0
        for param in params:
            if requires_grad is None:
                requires_grad = param.requires_grad
            elif requires_grad != param.requires_grad:
                raise ValueError("FlatParamHandle requires all params to have the same value of '.requires_grad'")

            flat_param_global_offsets = (numel_running_total, numel_running_total + param.numel())

            # First we need to determine which ranks will have a slice of the data.
            unsharded_flattened_offsets: List[Tuple[Tuple[int, int], ...]] = []
            for rank in range(world_size):
                rank_global_start = rank * padded_sharded_numel
                rank_global_end = rank_global_start + padded_sharded_numel
                if (rank_global_end <= flat_param_global_offsets[0]) or (
                    flat_param_global_offsets[1] <= rank_global_start
                ):
                    # No overlap with this rank.
                    unsharded_flattened_offsets.append(((0, 0),))
                elif (
                    rank_global_start <= flat_param_global_offsets[0]
                    and flat_param_global_offsets[1] <= rank_global_end
                ):
                    # Param is completely contained by this rank.
                    unsharded_flattened_offsets.append(((0, param.numel()),))
                elif (
                    rank_global_start <= flat_param_global_offsets[0]
                    and rank_global_end < flat_param_global_offsets[1]
                ):
                    # Param starts in this rank and ends in a subsequent rank.
                    unsharded_flattened_offsets.append(((0, rank_global_end - flat_param_global_offsets[0]),))
                elif (
                    flat_param_global_offsets[0] < rank_global_start
                    and flat_param_global_offsets[1] <= rank_global_end
                ):
                    # Param starts in a previous rank and ends in this one.
                    unsharded_flattened_offsets.append(
                        ((rank_global_start - flat_param_global_offsets[0], param.numel()),)
                    )
                elif (
                    flat_param_global_offsets[0] < rank_global_start
                    and rank_global_end < flat_param_global_offsets[1]
                ):
                    # Param spans this rank and overflows into other ranks.
                    unsharded_flattened_offsets.append(
                        (
                            (
                                rank_global_start - flat_param_global_offsets[0],
                                rank_global_end - flat_param_global_offsets[0],
                            ),
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
                            unsharded_flattened_offsets[local_rank][0][0] : unsharded_flattened_offsets[
                                local_rank
                            ][0][1]
                        ].to(device)
                    )
            else:
                flat_param = ShardedFlatParameter(torch.empty(0, device=device))
            flat_param.requires_grad = param.requires_grad
            flat_param.mark_as_sharded(sharding_spec, process_group=process_group)

            flat_params.append(flat_param)

            numel_running_total += param.numel()
            param.data = torch.empty(0, device=param.device)

        assert requires_grad is not None

        # Now that we have all of the flat parameters we need to collate all of their data into a single
        # sharded flat tensor, then set the data for each flat parameter as a view into that flat tensor.
        local_flat_sharded_data = torch.cat([flat_param.sharded_data for flat_param in flat_params])
        params_data = ShardedFlatTensor(
            F.pad(local_flat_sharded_data, (0, padded_sharded_numel - local_flat_sharded_data.numel()))
        )
        del local_flat_sharded_data
        params_data.mark_as_sharded(
            ShardingSpec(
                unsharded_shape=(padded_unsharded_numel,),
                unsharded_flattened_offsets=tuple(
                    [
                        ((start_idx, end_idx),)
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
            flat_param.sharded_data = params_data[offset : offset + flat_param.numel()]
            offset += flat_param.numel()

        return cls(
            params=flat_params,
            param_fqns=list(param_fqns),
            params_data=params_data,
            process_group=process_group,
            inter_group_process_group=inter_group_process_group,
            device=device,
            requires_grad=requires_grad,
        )

    def pre_unshard_(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False, set_grads: bool = False):
        """
        Allocate the unsharded, padded data prior to the all-gather. Ideally this should be called
        in a separate stream from :meth:`.unshard_()` for better throughput.
        """
        self._ran_pre_unshard = True

        if not self.params:
            return

        if rank0_only or dist.get_backend() == dist.Backend.GLOO:
            return

        # Initialize unsharded, padded ``params_data`` without the all-gather.
        all_params_unsharded_data = torch.empty(
            self.params_data.unsharded_shape, dtype=dtype or self.params_data.dtype, device=self.device
        )
        self.params_data.unshard_(unsharded_data=all_params_unsharded_data, dtype=dtype, rank0_only=rank0_only)

        # Cast sharded ``params_data`` to ``dtype``.
        if dtype is not None:
            self.params_sharded_data_lp = self.params_data.sharded_chunk(all_params_unsharded_data)
            self.params_sharded_data_lp.copy_(self.params_data.sharded_data)

        # Initialize unsharded, padded gradient.
        if set_grads and self.requires_grad and self.params_unsharded_grad is None:
            self.params_unsharded_grad = torch.zeros_like(all_params_unsharded_data)

    def unshard_(
        self,
        dtype: Optional[torch.dtype] = None,
        rank0_only: bool = False,
        set_grads: bool = False,
    ):
        """
        Unshard the handle's managed flat parameters in-place.
        """
        if not self.params:
            return

        if rank0_only:
            assert not set_grads

        local_rank = get_rank(self.process_group)

        if not self._ran_pre_unshard:
            self.pre_unshard_(dtype=dtype, rank0_only=rank0_only, set_grads=set_grads)
        else:
            # The following tensors were potentially created in a different stream, so we need
            # to make sure they're not deallocated prematurely.
            Stream.current(self.device).record_for(self.params_data.data)
            if self.params_sharded_data_lp is not None:
                Stream.current(self.device).record_for(self.params_sharded_data_lp)

        self._ran_pre_unshard = False

        # Gather full, padded, unsharded data for all params.
        if rank0_only or dist.get_backend() == dist.Backend.GLOO:
            assert self.params_data.is_sharded
            self.params_data.unshard_(dtype=dtype, rank0_only=rank0_only)
            if set_grads and self.requires_grad:
                self.params_unsharded_grad = torch.zeros_like(self.params_data.data)
        else:
            assert not self.params_data.is_sharded
            # We prefer to use `all_gather_into_tensor()` directly when possible as it involves
            # fewer allocations.
            local_shard: torch.Tensor
            if dtype is not None:
                assert self.params_sharded_data_lp is not None
                local_shard = self.params_sharded_data_lp
            else:
                local_shard = self.params_data.sharded_data
            dist.all_gather_into_tensor(
                self.params_data.data,
                local_shard,
                group=self.process_group,
            )
            self.params_sharded_data_lp = None
            del local_shard

        # Set the data for each param as a view into `self.params_data`.
        offset = 0
        for param in self.params:
            if rank0_only and local_rank != 0:
                unsharded_data = torch.empty_like(self.params_data.data)
            else:
                unsharded_data = self.params_data.data[offset : offset + param.unsharded_numel]

            param.unshard_(unsharded_data, dtype=dtype, rank0_only=rank0_only)

            if set_grads and self.requires_grad:
                if param.grad is None and self.params_sharded_grad is not None:
                    self.params_sharded_grad = None
                assert self.params_unsharded_grad is not None
                param.grad = self.params_unsharded_grad[offset : offset + param.unsharded_numel].view(param.shape)

            offset += param.unsharded_numel

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
                flat_param.sharded_data = self.params_data[offset : offset + flat_param.sharded_numel]
            offset += flat_param.sharded_numel

    def pre_reduce_scatter_grads_(
        self, grad_dtype: Optional[torch.dtype] = None, grad_reduce_dtype: Optional[torch.dtype] = None
    ):
        """
        Cast the unsharded, padded gradient to the reduce dtype if needed.
        """
        self._ran_pre_reduce_scatter_grads = True

        if not self.requires_grad or self.params_unsharded_grad is None:
            return

        grad_dtype = grad_dtype or self.params_data.dtype
        grad_reduce_dtype = grad_reduce_dtype or grad_dtype

        if grad_reduce_dtype != self.params_unsharded_grad.dtype:
            Stream.current(self.device).record_for(self.params_unsharded_grad)
            self.params_unsharded_grad = self.params_unsharded_grad.to(dtype=grad_reduce_dtype)

        if self.pre_reduce_grad_divide_factor > 1.0:
            self.params_unsharded_grad.div_(self.pre_reduce_grad_divide_factor)

    def reduce_scatter_grads_(
        self, grad_dtype: Optional[torch.dtype] = None, grad_reduce_dtype: Optional[torch.dtype] = None
    ):
        """
        Reduce-scatter the unsharded, padded gradient, and set the ``.grad`` attribute of each
        parameter as a view into the new sharded grad.
        """
        if not self.requires_grad or self.params_unsharded_grad is None:
            self._ran_pre_reduce_scatter_grads = False
            return

        if not self._ran_pre_reduce_scatter_grads:
            self.pre_reduce_scatter_grads_(grad_dtype=grad_dtype, grad_reduce_dtype=grad_reduce_dtype)
        else:
            Stream.current(self.device).record_for(self.params_unsharded_grad)

        self._ran_pre_reduce_scatter_grads = False

        grad_dtype = grad_dtype or self.params_data.dtype
        grad_reduce_dtype = grad_reduce_dtype or grad_dtype
        assert self.params_unsharded_grad.dtype == grad_reduce_dtype

        # Reduce the unsharded padded grad for all params.
        # NOTE: Only NCCL supports reduce-scatter. So with other backends we use all-reduce.
        if dist.get_backend() == dist.Backend.NCCL:
            # Get chunks corresponding to each rank.
            grad_chunks = self.params_data.chunk_unsharded(self.params_unsharded_grad)
            dist.reduce_scatter(
                grad_chunks[get_rank(group=self.process_group)], grad_chunks, group=self.process_group
            )
        else:
            dist.all_reduce(self.params_unsharded_grad, group=self.process_group)

    def post_reduce_scatter_grads_(
        self, grad_dtype: Optional[torch.dtype] = None, grad_reduce_dtype: Optional[torch.dtype] = None
    ):
        """
        Finalize sharded gradients after the reduce-scatter.
        """
        if not self.requires_grad or self.params_unsharded_grad is None:
            return

        grad_dtype = grad_dtype or self.params_data.dtype
        grad_reduce_dtype = grad_reduce_dtype or grad_dtype

        new_sharded_grad = self.params_data.sharded_chunk(self.params_unsharded_grad)

        if self.inter_group_process_group is not None:
            dist.all_reduce(new_sharded_grad, group=self.inter_group_process_group)

        if self.post_reduce_grad_divide_factor > 1.0:
            new_sharded_grad.div_(self.post_reduce_grad_divide_factor)

        # Cast the new sharded gradient to the right dtype, potentially accumulating it into
        # the existing sharded gradient.
        if self.params_sharded_grad is None:
            if new_sharded_grad.dtype == grad_dtype:
                self.params_sharded_grad = new_sharded_grad.clone()
            else:
                self.params_sharded_grad = new_sharded_grad.to(grad_dtype)
        else:
            self.params_sharded_grad.add_(new_sharded_grad)

        # Deallocate the unsharded padded grad.
        # NOTE: Since we're potentially using a separate stream here, we need to make
        # sure `params_unsharded_grad` is not deallocated before this finishes.
        Stream.current(self.device).record_for(self.params_unsharded_grad)
        self.params_unsharded_grad = None
        del new_sharded_grad

        # At this point each param will be sharded again, and we set the grad for each param as a view
        # into the sharded grad.
        offset = 0
        for param in self.params:
            param.grad = self.params_sharded_grad[offset : offset + param.sharded_numel]
            offset += param.sharded_numel
