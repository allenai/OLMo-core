from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .sharded_flat_tensor import ShardedFlatTensor, ShardingSpec

__all__ = ["ShardedFlatParameter", "ShardingSpec"]


class ShardedFlatParameter(ShardedFlatTensor, nn.Parameter):
    def __new__(cls, data: Optional[torch.Tensor] = None, requires_grad: bool = True) -> ShardedFlatParameter:
        if data is not None and data.ndim != 1:
            raise ValueError(f"{cls.__name__} requires flat data! Got {data.shape}")

        param: ShardedFlatParameter = nn.Parameter.__new__(  # type: ignore
            cls,
            data=None if data is None else data.data.detach(),
            requires_grad=requires_grad,
        )

        if isinstance(data, ShardedFlatTensor):
            setattr(
                param,
                cls.SHARDED_FLAT_TENSOR_METADATA_NAME,
                getattr(data, cls.SHARDED_FLAT_TENSOR_METADATA_NAME).copy(),
            )
        else:
            setattr(param, cls.SHARDED_FLAT_TENSOR_METADATA_NAME, {})

        return param

    def __repr__(self) -> str:
        r = torch.Tensor.__repr__(self)
        if r.startswith("Parameter("):  # )  -- the open parenthesis confuses treesitter sometimes
            r = r.replace("Parameter(", "", 1)  # )  -- the open parenthesis confuses treesitter sometimes
            r = r[:-1]
        return r
