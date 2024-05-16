from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .sharded_flat_tensor import ShardedFlatTensor, ShardingSpec

__all__ = ["ShardedFlatParameter", "ShardingSpec"]


class ShardedFlatParameter(ShardedFlatTensor, nn.Parameter):
    """
    A :class:`~torch.nn.parameter.Parameter` version of :class:`ShardedFlatTensor`.
    """

    @staticmethod
    def __new__(cls, data: Optional[torch.Tensor] = None, requires_grad: bool = True) -> ShardedFlatParameter:
        if data is not None and data.ndim != 1:
            raise ValueError(f"{cls.__name__} requires flat data! Got {data.shape}")

        param: ShardedFlatParameter = nn.Parameter.__new__(  # type: ignore
            cls,
            data=None if data is None else data.data.detach(),
            requires_grad=requires_grad,
        )

        if isinstance(data, ShardedFlatTensor):
            param._local_tensor = data._local_tensor
            param._sharding_spec = data._sharding_spec
            param._process_group = data._process_group
        else:
            param._local_tensor = None if data is None else data.data.detach()
            param._sharding_spec = None  # type: ignore[assignment]
            param._process_group = None

        param._global_tensor = None

        return param

    def __repr__(self) -> str:
        if self._global_tensor is not None:
            return f"ShardedFlatParameter(local_tensor={self._local_tensor}, global_tensor={self._global_tensor}, requires_grad={self.requires_grad})"
        else:
            return f"ShardedFlatParameter(local_tensor={self._local_tensor}, requires_grad={self.requires_grad})"
