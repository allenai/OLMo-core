from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_rank, get_world_size


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


class ShardedFlatParameter(nn.Parameter):
    SHARDED_FLAT_PARAMETER_METADATA_NAME = "__sharded_metadata__"
    SHARDED_FLAT_PARAMETER_SHARDING_SPEC_KEY = "sharding_spec"
    SHARDED_FLAT_PARAMETER_PROCESS_GROUP_KEY = "process_group"

    def __new__(cls, data: Optional[torch.Tensor] = None, requires_grad: bool = True) -> ShardedFlatParameter:
        if data is not None and data.ndim != 1:
            raise ValueError(f"{cls.__name__} requires flat data! Got {data.shape}")

        param: ShardedFlatParameter = nn.Parameter.__new__(  # type: ignore
            cls,
            data=None if data is None else data.data.detach(),
            requires_grad=requires_grad,
        )

        if isinstance(data, ShardedFlatParameter):
            setattr(
                param,
                cls.SHARDED_FLAT_PARAMETER_METADATA_NAME,
                getattr(data, cls.SHARDED_FLAT_PARAMETER_METADATA_NAME).copy(),
            )
        else:
            setattr(param, cls.SHARDED_FLAT_PARAMETER_METADATA_NAME, {})

        return param

    def __repr__(self) -> str:
        r = torch.Tensor.__repr__(self)
        if r.startswith("Parameter("):  # )  -- the open parenthesis confuses treesitter sometimes
            r = r.replace("Parameter(", "", 1)  # )  -- the open parenthesis confuses treesitter sometimes
            r = r[:-1]
        return r

    def _set_metadata(self, key: str, value: Any):
        metadata = getattr(self, self.SHARDED_FLAT_PARAMETER_METADATA_NAME)
        if key in metadata:
            raise ValueError(f"Metadata key '{key}' already exists in {self.__class__.__name__}")
        else:
            metadata[key] = value

    @classmethod
    def shard(
        cls,
        tensor: torch.Tensor,
        sharding_spec: Optional[ShardingSpec] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        synchronize: bool = True,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = None,
    ) -> ShardedFlatParameter:
        """
        Shard a tensor across a process group.
        """
        if sharding_spec is not None and tensor.shape != sharding_spec.unsharded_shape:
            raise ValueError(
                f"Shape of tensor to shard ({tensor.shape}) should match unsharded shape in sharding spec ({sharding_spec.unsharded_shape})"
            )

        tensor_is_initialized = tensor.device != torch.device("meta")

        if synchronize and tensor_is_initialized:
            dist.broadcast(tensor, 0, group=process_group)

        if sharding_spec is None:
            # Shard tensor as evenly as possible across all ranks.
            world_size = get_world_size(group=process_group)
            shard_max_numel = tensor.numel() // world_size
            offsets = tuple(
                (rank * shard_max_numel, min((rank + 1) * shard_max_numel, tensor.numel()))
                for rank in range(world_size)
            )
            sharding_spec = ShardingSpec(unsharded_shape=tuple(tensor.shape), unsharded_flattened_offsets=offsets)

        offsets = sharding_spec.unsharded_flattened_offsets[get_rank(group=process_group)]

        if tensor_is_initialized:
            sharded_tensor = tensor.flatten()[offsets[0] : offsets[1]].clone().to(device=device)
        else:
            sharded_tensor = torch.empty(offsets[1] - offsets[0], device=device, dtype=tensor.dtype)

        sharded_param = cls(
            sharded_tensor, requires_grad=requires_grad if requires_grad is not None else tensor.requires_grad
        )
        sharded_param.mark_as_sharded(sharding_spec, process_group=process_group)
        return sharded_param

    def gather(self) -> nn.Parameter:
        """
        Gather the sharded flat parameter across a process group into the full unsharded parameter.
        """
        # NOTE: ``all_gather_into_tensor`` is not supported on Gloo.
        sharded_numels = self.sharding_spec.sharded_numels
        max_numel = max(sharded_numels)

        # Pad sharded tensors to the same size.
        flat_sharded_tensor_list = [
            torch.empty(max_numel, device=self.device, dtype=self.dtype) for _ in self.sharding_spec.sharded_numels
        ]

        # Gather padding sharded tensors across all ranks.
        dist.all_gather(
            flat_sharded_tensor_list,
            F.pad(self.data, (0, max_numel - sharded_numels[get_rank(group=self.process_group)])),
            group=self.process_group,
        )

        # Unpad, sort by starting offset, and concatenate.
        flat_tensor = torch.cat(
            [
                flat_sharded_tensor_list[idx][: sharded_numels[idx]]
                for idx in sorted(
                    range(len(sharded_numels)),
                    key=lambda idx: self.sharding_spec.unsharded_flattened_offsets[idx][0],
                )
            ]
        )

        return nn.Parameter(
            flat_tensor.reshape(self.sharding_spec.unsharded_shape), requires_grad=self.requires_grad
        )

    def mark_as_sharded(self, sharding_spec: ShardingSpec, process_group: Optional[dist.ProcessGroup] = None):
        self._set_metadata(self.SHARDED_FLAT_PARAMETER_SHARDING_SPEC_KEY, sharding_spec)
        self._set_metadata(self.SHARDED_FLAT_PARAMETER_PROCESS_GROUP_KEY, process_group)

    def wrap(self, tensor: torch.Tensor, requires_grad: bool = True) -> ShardedFlatParameter:
        """
        Wrap another tensor and mark as sharded with the same sharding spec.
        ``tensor`` should have the same shape as ``self.data``, the sharded data.
        """
        if tensor.shape != self.data.shape:
            raise ValueError(f"shape mismatched, expected {self.data.shape}, got {tensor.shape}")
        wrapped = ShardedFlatParameter(tensor.data, requires_grad=requires_grad)
        wrapped.mark_as_sharded(self.sharding_spec, process_group=self.process_group)
        return wrapped

    @property
    def is_sharded(self) -> bool:
        return self.SHARDED_FLAT_PARAMETER_SHARDING_SPEC_KEY in getattr(
            self, self.SHARDED_FLAT_PARAMETER_METADATA_NAME
        )

    @property
    def sharding_spec(self) -> ShardingSpec:
        try:
            return getattr(self, self.SHARDED_FLAT_PARAMETER_METADATA_NAME)[
                self.SHARDED_FLAT_PARAMETER_SHARDING_SPEC_KEY
            ]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has not been marked as sharded yet, "
                "did you forget to class '.mark_as_sharded()'?"
            )

    @property
    def process_group(self) -> Optional[dist.ProcessGroup]:
        try:
            return getattr(self, self.SHARDED_FLAT_PARAMETER_METADATA_NAME)[
                self.SHARDED_FLAT_PARAMETER_PROCESS_GROUP_KEY
            ]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} has not been marked as sharded yet, "
                "did you forget to class '.mark_as_sharded()'?"
            )

    @property
    def unsharded_flattened_offsets(self) -> Tuple[int, int]:
        return self.sharding_spec.unsharded_flattened_offsets[get_rank(group=self.process_group)]

    @property
    def unsharded_shape(self) -> Tuple[int, ...]:
        return self.sharding_spec.unsharded_shape
