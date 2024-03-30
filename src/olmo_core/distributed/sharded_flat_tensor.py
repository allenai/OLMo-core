from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_rank, get_world_size

__all__ = ["ShardedFlatTensor", "ShardingSpec"]


@dataclass
class ShardingSpec:
    unsharded_shape: Tuple[int, ...]
    """
    The shape of the full unsharded (unflattened) parameter.
    """

    unsharded_flattened_offsets: Tuple[Tuple[int, int], ...]
    """
    The ``(start_idx, end_idx)`` within the full unsharded flattened parameter that each local shard
    within the process group corresponds to.

    This tuple is indexed by rank. For example, the ``(start_idx, end_idx)`` within the full unsharded flattened
    parameter for the local shard of the current rank is given by ``unsharded_flattened_offsets[dist.get_rank(process_group)]``.
    """

    @property
    def unsharded_flattened_shape(self) -> int:
        return reduce(lambda x, y: x * y, self.unsharded_shape, 1)

    @property
    def sharded_numels(self) -> Tuple[int, ...]:
        return tuple((offsets[1] - offsets[0] for offsets in self.unsharded_flattened_offsets))


class ShardedFlatTensor(torch.Tensor):
    SHARDED_FLAT_TENSOR_METADATA_NAME = "__sharded_metadata__"
    SHARDED_FLAT_TENSOR_SHARDING_SPEC_KEY = "sharding_spec"
    SHARDED_FLAT_TENSOR_PROCESS_GROUP_KEY = "process_group"
    SHARDED_FLAT_TENSOR_CACHED_SHARDED_DATA_KEY = "sharded_data"

    def __new__(cls, data: torch.Tensor, requires_grad: bool = False) -> ShardedFlatTensor:
        if data.ndim != 1:
            raise ValueError(f"{cls.__name__} requires flat data! Got {data.shape}")

        tensor: ShardedFlatTensor
        if isinstance(data, ShardedFlatTensor):
            tensor = data
            setattr(
                tensor,
                cls.SHARDED_FLAT_TENSOR_METADATA_NAME,
                getattr(data, cls.SHARDED_FLAT_TENSOR_METADATA_NAME).copy(),
            )
        elif type(data) is torch.Tensor or type(data) is nn.Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
            setattr(tensor, cls.SHARDED_FLAT_TENSOR_METADATA_NAME, {})
        else:
            raise TypeError(f"found unexpected type for {cls.__name__} data: {type(data)}")

        return tensor

    def __repr__(self) -> str:
        return torch.Tensor.__repr__(self)

    def _set_metadata(self, key: str, value: Any, force: bool = False):
        metadata = getattr(self, self.SHARDED_FLAT_TENSOR_METADATA_NAME)
        if not force and key in metadata:
            raise ValueError(f"Metadata key '{key}' already exists in {self.__class__.__name__}")
        else:
            metadata[key] = value

    def _gather_data(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False) -> torch.Tensor:
        # NOTE: ``all_gather_into_tensor`` is not supported on Gloo.
        local_rank = get_rank(group=self.process_group)
        sharded_numels = self.sharding_spec.sharded_numels
        max_numel = max(sharded_numels)
        local_padding = (0, max_numel - sharded_numels[local_rank])

        flat_sharded_tensor_list: Optional[List[torch.Tensor]] = None
        local_flat_padded_tensor = F.pad(self.data.to(dtype or self.dtype), local_padding)

        # Pad sharded tensors to the same size.
        if not rank0_only or get_rank(group=self.process_group) == 0:
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

        # Unpad, sort by starting offset, and concatenate.
        if flat_sharded_tensor_list is not None:
            flat_tensor = torch.cat(
                [
                    flat_sharded_tensor_list[idx][: sharded_numels[idx]]
                    for idx in sorted(
                        range(len(sharded_numels)),
                        key=lambda idx: self.sharding_spec.unsharded_flattened_offsets[idx][0],
                    )
                ]
            )
            return flat_tensor.reshape(self.sharding_spec.unsharded_shape)
        else:
            return torch.empty(0, dtype=dtype or self.dtype, device=self.device)

    @classmethod
    def shard(
        cls,
        tensor: torch.Tensor,
        sharding_spec: Optional[ShardingSpec] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        synchronize: bool = True,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = None,
    ) -> ShardedFlatTensor:
        """
        Shard a tensor across a process group.
        """
        if sharding_spec is not None and tensor.shape != sharding_spec.unsharded_shape:
            raise ValueError(
                f"Shape of tensor to shard ({tensor.shape}) should match unsharded shape in sharding spec ({sharding_spec.unsharded_shape})"
            )

        tensor_is_initialized = tensor.device != torch.device("meta")

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
                (rank * shard_max_numel, min((rank + 1) * shard_max_numel, tensor.numel()))
                for rank in range(world_size)
            )
            sharding_spec = ShardingSpec(
                unsharded_shape=tuple(tensor.shape), unsharded_flattened_offsets=all_offsets
            )

        offsets = sharding_spec.unsharded_flattened_offsets[get_rank(group=process_group)]

        if tensor_is_initialized:
            sharded_tensor = tensor.flatten()[offsets[0] : offsets[1]].clone().to(device=device)
        else:
            sharded_tensor = torch.empty(offsets[1] - offsets[0], device=device, dtype=tensor.dtype)

        sharded_param = cls(  # type: ignore
            sharded_tensor, requires_grad=requires_grad if requires_grad is not None else tensor.requires_grad
        )
        sharded_param.mark_as_sharded(sharding_spec, process_group=process_group)
        return sharded_param

    def gather(self, dtype: Optional[torch.dtype] = None) -> nn.Parameter:
        """
        Gather the sharded flat parameter across a process group into a full unsharded parameter.
        """
        unsharded_data = self._gather_data(dtype=dtype)
        return nn.Parameter(unsharded_data, requires_grad=self.requires_grad)

    def unshard_(self, dtype: Optional[torch.dtype] = None, rank0_only: bool = False):
        """
        Unshard this parameter's data in-place. You should generally call :meth:`reshard_()` afterwards.

        If ``rank0_only=True``, non rank 0 processes will have an empty tensor in their data.
        """
        unsharded_data = self._gather_data(dtype=dtype, rank0_only=rank0_only)
        self._set_metadata(self.SHARDED_FLAT_TENSOR_CACHED_SHARDED_DATA_KEY, self.data)
        self.data = unsharded_data

    def reshard_(self, writeback: bool = False):
        """
        Reshard this parameter's data in-place. Should only be called after :meth:`unshard_()`.
        This does *not* do anything with the parameter's gradient, if it has one. That should
        be handled separately by the calling code.
        """
        if self.is_sharded:
            return

        metadata = getattr(self, self.SHARDED_FLAT_TENSOR_METADATA_NAME)
        try:
            sharded_data = metadata[self.SHARDED_FLAT_TENSOR_CACHED_SHARDED_DATA_KEY]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has not been unsharded in place yet, "
                "did you forget to class '.unshard_()'?"
            )

        if writeback:
            unsharded_data = self.data
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
            self.data = self.sharded_chunk(unsharded_data).to(dtype=sharded_data.dtype)
        else:
            self.data = sharded_data
        del metadata[self.SHARDED_FLAT_TENSOR_CACHED_SHARDED_DATA_KEY]

    def mark_as_sharded(self, sharding_spec: ShardingSpec, process_group: Optional[dist.ProcessGroup] = None):
        if self.numel() != (shard_numel := sharding_spec.sharded_numels[get_rank(group=process_group)]):
            raise ValueError(
                f"invalid sharding spec, numel in spec ({shard_numel}) does not match numel in shard ({self.numel()})"
            )
        self._set_metadata(self.SHARDED_FLAT_TENSOR_SHARDING_SPEC_KEY, sharding_spec)
        self._set_metadata(self.SHARDED_FLAT_TENSOR_PROCESS_GROUP_KEY, process_group)

    def wrap(self, tensor: torch.Tensor, requires_grad: bool = True) -> ShardedFlatTensor:
        """
        Wrap another tensor and mark as sharded with the same sharding spec.
        ``tensor`` should have the same shape as ``self.data``, the sharded data.
        """
        if tensor.shape != self.data.shape:
            raise ValueError(f"shape mismatched, expected {self.data.shape}, got {tensor.shape}")
        wrapped = ShardedFlatTensor(tensor.data, requires_grad=requires_grad)  # type: ignore
        wrapped.mark_as_sharded(self.sharding_spec, process_group=self.process_group)
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
        for offsets in self.sharding_spec.unsharded_flattened_offsets:
            chunk = flat_tensor[offsets[0] : offsets[1]]
            if pad:
                chunk = F.pad(chunk, (0, max_size - (offsets[1] - offsets[0])))
            chunks.append(chunk)
        return chunks

    def sharded_chunk(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Get this rank's sharded chunk of an unsharded tensor with the same shape as ``self.unsharded_shape``.
        """
        if tensor.shape != self.unsharded_shape:
            raise ValueError(f"shape mismatched, expected {self.unsharded_shape}, got {tensor.shape}")
        offset_start, offset_end = self.unsharded_flattened_offsets
        return tensor.flatten()[offset_start:offset_end]

    @property
    def is_sharded(self) -> bool:
        metadata = getattr(self, self.SHARDED_FLAT_TENSOR_METADATA_NAME)
        return (
            self.SHARDED_FLAT_TENSOR_SHARDING_SPEC_KEY in metadata
            and self.SHARDED_FLAT_TENSOR_CACHED_SHARDED_DATA_KEY not in metadata
        )

    @property
    def sharding_spec(self) -> ShardingSpec:
        metadata = getattr(self, self.SHARDED_FLAT_TENSOR_METADATA_NAME)
        try:
            return metadata[self.SHARDED_FLAT_TENSOR_SHARDING_SPEC_KEY]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has not been marked as sharded yet, "
                "did you forget to class '.mark_as_sharded()'?"
            )

    @property
    def process_group(self) -> Optional[dist.ProcessGroup]:
        try:
            return getattr(self, self.SHARDED_FLAT_TENSOR_METADATA_NAME)[
                self.SHARDED_FLAT_TENSOR_PROCESS_GROUP_KEY
            ]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has not been marked as sharded yet, "
                "did you forget to class '.mark_as_sharded()'?"
            )

    @property
    def unsharded_flattened_offsets(self) -> Tuple[int, int]:
        # mypy is really bad some times
        offsets: Tuple[int, int] = self.sharding_spec.unsharded_flattened_offsets[  # type: ignore[assignment]
            get_rank(group=self.process_group)
        ]
        return offsets

    @property
    def unsharded_shape(self) -> Tuple[int, ...]:
        return self.sharding_spec.unsharded_shape

    @property
    def sharded_shape(self) -> Tuple[int, ...]:
        return (self.sharding_spec.sharded_numels[get_rank(self.process_group)],)
