from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple, Type, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging import version

try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]

from ..utils import get_rank, get_world_size

__all__ = ["ShardedFlatTensor", "ShardingSpec"]


T = TypeVar("T", bound="ShardedFlatTensor")


@dataclass
class ShardingSpec:
    unsharded_shape: Tuple[int, ...]
    """
    The shape of the full unsharded (unflattened) parameter.
    """

    unsharded_flattened_offsets: Tuple[Tuple[Tuple[int, int], ...], ...]
    """
    The offsets (``(start_idx, end_idx)``) within the full unsharded flattened parameter that each
    local shard within the process group corresponds to.

    This tuple is indexed by rank within the process group.
    For example, the offsets within the full unsharded flattened parameter for the
    local shard of the current rank is given by ``unsharded_flattened_offsets[dist.get_rank(process_group)]``.
    """

    def __post_init__(self):
        numel_accounted_for = 0
        for rank_offsets in self.unsharded_flattened_offsets:
            for start_idx, end_idx in rank_offsets:
                assert start_idx <= end_idx
                numel_accounted_for += end_idx - start_idx
        if numel_accounted_for != self.unsharded_numel:
            raise ValueError(f"invalid sharding spec {self}")

    @property
    def unsharded_numel(self) -> int:
        """
        The number of elements in the full unsharded tensor.
        """
        return reduce(lambda x, y: x * y, self.unsharded_shape, 1)

    @property
    def sharded_numels(self) -> Tuple[int, ...]:
        """
        The number of elements in each shard.
        """
        return tuple(
            (
                sum(end_idx - start_idx for start_idx, end_idx in offsets)
                for offsets in self.unsharded_flattened_offsets
            )
        )

    @property
    def unsharded_flattened_shape(self) -> Tuple[int, ...]:
        """
        The shape of the unsharded flattened tensor.
        """
        return (self.unsharded_numel,)


class ShardedFlatTensor(torch.Tensor):
    """
    :class:`ShardedFlatTensor` represents a sharded tensor with the assumption that every shard is
    a contiguous slice into the flattened unsharded tensor.
    """

    __slots__ = ["_local_tensor", "_global_tensor", "_sharding_spec", "_process_group"]

    @staticmethod
    def __new__(cls, data: torch.Tensor, requires_grad: bool = False) -> ShardedFlatTensor:
        if data.ndim != 1:
            raise ValueError(f"{cls.__name__} requires flat data! Got {data.shape}")

        sharding_spec: Optional[ShardingSpec] = None
        process_group: Optional[dist.ProcessGroup] = None

        tensor: ShardedFlatTensor
        if isinstance(data, ShardedFlatTensor):
            sharding_spec = data._sharding_spec
            process_group = data._process_group
            data = data._local_tensor

        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        tensor._local_tensor = data
        tensor._global_tensor = None
        tensor._sharding_spec = sharding_spec  # type: ignore[assignment]
        tensor._process_group = process_group

        return tensor

    if version.parse(torch.__version__) >= version.parse("2.3.0"):
        # There are some bugs with __torch_dispatch__ in earlier versions.

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            kwargs = kwargs or {}

            def unwrap(x):
                if isinstance(x, ShardedFlatTensor):
                    return x._global_tensor if x._global_tensor is not None else x._local_tensor
                else:
                    return x

            def wrap(x):
                if isinstance(x, torch.Tensor):
                    if x.shape == self.shape:
                        return self.wrap(x, requires_grad=x.requires_grad)
                return x

            out = func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs))

            if func in {torch.ops.aten.empty_like.default, torch.ops.aten.zeros_like.default, torch.ops.aten.ones_like.default}:  # type: ignore
                out = pytree.tree_map(wrap, out)

            return out

    def __repr__(self) -> str:
        if not self.metadata_set:
            return super().__repr__()

        if self._global_tensor is not None:
            return f"ShardedFlatTensor(local_tensor={self._local_tensor}, global_tensor={self._global_tensor})"
        else:
            return f"ShardedFlatTensor(local_tensor={self._local_tensor})"

    def _gather_data(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False) -> torch.Tensor:
        # NOTE: ``all_gather_into_tensor`` is not supported on Gloo.
        local_rank = get_rank(group=self.process_group)
        sharded_numels = self.sharding_spec.sharded_numels
        max_numel = max(sharded_numels)
        local_padding = (0, max_numel - sharded_numels[local_rank])

        flat_sharded_tensor_list: Optional[List[torch.Tensor]] = None
        local_flat_padded_tensor = F.pad(self._local_tensor.to(dtype or self.dtype), local_padding)

        # Pad sharded tensors to the same size.
        if not rank0_only or local_rank == 0:
            flat_sharded_tensor_list = [
                torch.empty(max_numel, device=self.device, dtype=dtype or self.dtype)
                for _ in range(len(self.sharding_spec.sharded_numels) - 1)
            ]
            flat_sharded_tensor_list.insert(local_rank, local_flat_padded_tensor)

        if not rank0_only:
            # Gather padded sharded tensors across all ranks.
            assert flat_sharded_tensor_list is not None
            dist.all_gather(
                flat_sharded_tensor_list,
                local_flat_padded_tensor,
                group=self.process_group,
            )
        else:
            # Gather padded sharded tensors to rank 0.
            dist.gather(local_flat_padded_tensor, gather_list=flat_sharded_tensor_list, group=self.process_group)

        if flat_sharded_tensor_list is None:
            # rank0_only=True and this is not rank 0.
            return torch.empty(0, dtype=dtype or self.dtype, device=self.device)

        # Unpad and pull out contiguous sharded chunks from each ranks sharded flat tensor.
        contiguous_flat_sharded_tensors = []
        contiguous_offsets = []
        for rank, rank_sharded_tensor in enumerate(flat_sharded_tensor_list):
            rank_sharded_tensor = rank_sharded_tensor[: sharded_numels[rank]]
            local_offset = 0
            for start_idx, end_idx in self.sharding_spec.unsharded_flattened_offsets[rank]:
                chunk_numel = end_idx - start_idx
                contiguous_flat_sharded_tensors.append(
                    rank_sharded_tensor[local_offset : local_offset + chunk_numel]
                )
                contiguous_offsets.append((start_idx, end_idx))
                local_offset += chunk_numel

        # Now sort by starting offset and concatenate together.
        flat_tensor = torch.cat(
            [
                contiguous_flat_sharded_tensors[idx]
                for idx in sorted(
                    range(len(contiguous_offsets)),
                    key=lambda idx: contiguous_offsets[idx][0],
                )
            ]
        )

        # Reshape and return.
        return flat_tensor.reshape(self.sharding_spec.unsharded_shape)

    @classmethod
    def shard(
        cls: Type[T],
        tensor: torch.Tensor,
        sharding_spec: Optional[ShardingSpec] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        synchronize: bool = True,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = None,
    ) -> T:
        """
        Shard a tensor across a process group.
        """
        if sharding_spec is not None and tensor.shape != sharding_spec.unsharded_shape:
            raise ValueError(
                f"Shape of tensor to shard ({tensor.shape}) should match unsharded shape in sharding spec ({sharding_spec.unsharded_shape})"
            )

        tensor_is_initialized = tensor.device != torch.device("meta")
        if device is None and tensor_is_initialized:
            device = tensor.device

        if synchronize and tensor_is_initialized:
            if device is not None:
                tensor = tensor.to(device)
            dist.broadcast(
                tensor,
                0 if process_group is None else dist.get_global_rank(process_group, 0),
                group=process_group,
            )

        if sharding_spec is None:
            # Shard tensor as evenly as possible across all ranks.
            world_size = get_world_size(group=process_group)
            shard_max_numel = math.ceil(tensor.numel() / world_size)
            all_offsets = tuple(
                ((rank * shard_max_numel, min((rank + 1) * shard_max_numel, tensor.numel())),)
                for rank in range(world_size)
            )
            sharding_spec = ShardingSpec(
                unsharded_shape=tuple(tensor.shape), unsharded_flattened_offsets=all_offsets
            )

        sharded_tensor = torch.empty(
            sharding_spec.sharded_numels[get_rank(group=process_group)], device=device, dtype=tensor.dtype
        )
        if tensor_is_initialized:
            flat_tensor = tensor.flatten()
            start_offset = 0
            for start_idx, end_idx in sharding_spec.unsharded_flattened_offsets[get_rank(group=process_group)]:
                chunk_numel = end_idx - start_idx
                sharded_tensor[start_offset : start_offset + chunk_numel].copy_(flat_tensor[start_idx:end_idx])
                start_offset += chunk_numel

        sharded_tensor = cls(  # type: ignore
            sharded_tensor, requires_grad=requires_grad if requires_grad is not None else tensor.requires_grad
        )
        sharded_tensor.mark_as_sharded(sharding_spec, process_group=process_group)
        return sharded_tensor

    def gather(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False) -> torch.Tensor:
        """
        Gather the sharded flat parameter across a process group into a full unsharded parameter.
        """
        if self._global_tensor is not None:
            unsharded_data = self._global_tensor
        else:
            unsharded_data = self._gather_data(dtype=dtype, rank0_only=rank0_only)
            unsharded_data.requires_grad = self.requires_grad
        return unsharded_data

    def unshard_(
        self,
        unsharded_data: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        rank0_only: bool = False,
    ):
        """
        Unshard this parameter's data in-place. You should generally call :meth:`reshard_()` afterwards.

        If ``rank0_only=True``, non rank 0 processes will have an empty tensor in their data.
        """
        if unsharded_data is None:
            unsharded_data = (
                self._global_tensor
                if self._global_tensor is not None
                else self._gather_data(dtype=dtype, rank0_only=rank0_only)
            )
        elif not rank0_only or get_rank(self.process_group) == 0:
            unsharded_data = unsharded_data.view(*self.unsharded_shape)
        self._global_tensor = unsharded_data
        # NOTE: despite `__torch_dispatch__`, we still need to set `self.data` to the unsharded
        # data in order for `self.shape()`, `self.numel()`, and other methods to return the
        # right values corresponding to the unsharded data.
        self.data = unsharded_data  # type: ignore[misc]

    def reshard_(self, writeback: bool = False):
        """
        Reshard this parameter's data in-place. Should only be called after :meth:`unshard_()`.
        This does *not* do anything with the parameter's gradient, if it has one. That should
        be handled separately by the calling code.
        """
        if (unsharded_data := self._global_tensor) is None:
            return

        if writeback:
            if unsharded_data.shape != self.unsharded_shape:
                # unsharded data could be empty if `.unshard_` was called with `rank0_only=True`.
                if unsharded_data.numel() > 0:
                    raise ValueError(
                        f"Unexpected shape found for {self.__class__.__name__}, "
                        f"expected {self.unsharded_shape}, found {tuple(unsharded_data.shape)}."
                    )
                unsharded_data = torch.empty(
                    self.unsharded_shape, dtype=unsharded_data.dtype, device=unsharded_data.device
                )
            dist.broadcast(
                unsharded_data,
                0 if self.process_group is None else dist.get_global_rank(self.process_group, 0),
                group=self.process_group,
            )
            self._local_tensor = self.sharded_chunk(unsharded_data).to(dtype=self._local_tensor.dtype).clone()

        self._global_tensor = None

        # NOTE: despite `__torch_dispatch__`, we still need to set `self.data` back to the sharded
        # data in order for `self.shape()`, `self.numel()`, and other methods to return the
        # right values corresponding to the sharded data.
        self.data = self._local_tensor  # type: ignore[misc]

    def mark_as_sharded(self, sharding_spec: ShardingSpec, process_group: Optional[dist.ProcessGroup] = None):
        if self.numel() != (shard_numel := sharding_spec.sharded_numels[get_rank(group=process_group)]):
            raise ValueError(
                f"invalid sharding spec, numel in spec ({shard_numel}) does not match numel in shard ({self.numel()})"
            )
        self._sharding_spec = sharding_spec
        self._process_group = process_group

    def wrap(self, tensor: torch.Tensor, requires_grad: Optional[bool] = None) -> ShardedFlatTensor:
        """
        Wrap another tensor and mark as sharded with the same sharding spec.
        ``tensor`` should have the same shape.
        """
        if self.is_sharded and tensor.shape != self.sharded_shape:
            raise ValueError(f"shape mismatched, expected {self.sharded_shape}, got {tensor.shape}")
        elif not self.is_sharded and tensor.shape != self.unsharded_shape:
            raise ValueError(f"shape mismatched, expected {self.unsharded_shape}, got {tensor.shape}")
        requires_grad = requires_grad if requires_grad is not None else tensor.requires_grad
        if self.is_sharded:
            wrapped = ShardedFlatTensor(tensor.data, requires_grad=requires_grad)  # type: ignore
            wrapped.mark_as_sharded(self.sharding_spec, process_group=self.process_group)
        else:
            wrapped = ShardedFlatTensor(self.sharded_chunk(tensor), requires_grad=requires_grad)  # type: ignore
            wrapped.mark_as_sharded(self.sharding_spec, process_group=self.process_group)
            wrapped.unshard_(tensor)
        return wrapped

    def chunk_unsharded(self, tensor: torch.Tensor, pad: bool = False) -> List[torch.Tensor]:
        """
        Chunk an unsharded tensor with the same shape as ``self.unsharded_shape`` and split it
        into flat chunks where each chunk has the shape of sharded data corresponding to that rank.

        :param pad: Whether or not to add right padding to the chunks to ensure they're all the same size.
        """
        if tensor.shape != self.unsharded_shape:
            raise ValueError(f"shape mismatched, expected {self.unsharded_shape}, got {tensor.shape}")
        chunks = []
        flat_tensor = tensor.flatten()
        max_size = max(self.sharding_spec.sharded_numels)
        for rank_offsets in self.sharding_spec.unsharded_flattened_offsets:
            rank_chunks = []
            for start_idx, end_idx in rank_offsets:
                rank_chunks.append(flat_tensor[start_idx:end_idx])
            chunk = rank_chunks[0] if len(rank_chunks) == 1 else torch.cat(rank_chunks)
            if pad:
                chunk = F.pad(chunk, (0, max_size - chunk.numel()))
            chunks.append(chunk)
        return chunks

    def sharded_chunk(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Get this rank's sharded chunk of an unsharded tensor with the same shape as ``self.unsharded_shape``.
        """
        if tensor.shape != self.unsharded_shape:
            raise ValueError(f"shape mismatched, expected {self.unsharded_shape}, got {tensor.shape}")
        flat_tensor = tensor.flatten()
        rank_chunks = []
        for start_idx, end_idx in self.unsharded_flattened_offsets:
            rank_chunks.append(flat_tensor[start_idx:end_idx])
        return rank_chunks[0] if len(rank_chunks) == 1 else torch.cat(rank_chunks)

    @property
    def metadata_set(self) -> bool:
        for slot in self.__slots__:
            if not hasattr(self, slot):
                return False
        return True

    @property
    def is_sharded(self) -> bool:
        return self._global_tensor is None

    @property
    def sharding_spec(self) -> ShardingSpec:
        if self._sharding_spec is None:
            raise ValueError(
                f"{self.__class__.__name__} has not been marked as sharded yet, "
                "did you forget to class '.mark_as_sharded()'?"
            )
        return self._sharding_spec

    @property
    def process_group(self) -> Optional[dist.ProcessGroup]:
        return self._process_group

    @property
    def unsharded_flattened_offsets(self) -> Tuple[Tuple[int, int], ...]:
        return self.sharding_spec.unsharded_flattened_offsets[get_rank(group=self.process_group)]

    @property
    def unsharded_numel(self) -> int:
        return self.sharding_spec.unsharded_numel

    @property
    def unsharded_shape(self) -> Tuple[int, ...]:
        return self.sharding_spec.unsharded_shape

    @property
    def sharded_numel(self) -> int:
        return self.sharding_spec.sharded_numels[get_rank(self.process_group)]

    @property
    def sharded_shape(self) -> Tuple[int, ...]:
        return (self.sharding_spec.sharded_numels[get_rank(self.process_group)],)

    @property
    def sharded_data(self) -> torch.Tensor:
        return self._local_tensor

    @sharded_data.setter
    def sharded_data(self, sharded_data: torch.Tensor):
        self._local_tensor = sharded_data
        self.data = sharded_data

    @property
    def unsharded_data(self) -> Optional[torch.Tensor]:
        return self._global_tensor
