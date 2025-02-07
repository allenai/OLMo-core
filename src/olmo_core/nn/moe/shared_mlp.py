from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import PrepareModuleOutput, parallelize_module

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.exceptions import OLMoConfigurationError

from ..feed_forward import FeedForward

__all__ = ["SharedMLP", "SharedMLPConfig", "SharedMLPType"]


class SharedMLPType(StrEnum):
    """
    An enumeration of the different shared MLP implementations.
    """

    default = "default"
    """
    ➡️ :class:`SharedMLP`
    """


@dataclass
class SharedMLPConfig(Config):
    """
    A config for building :class:`SharedMLP` modules.
    """

    name: SharedMLPType = SharedMLPType.default
    """
    The name of the implementation.
    """
    weighted_sum: bool = True
    hidden_size: Optional[int] = None
    bias: bool = True
    dtype: Optional[DType] = None

    def num_params(self, d_model: int, hidden_size: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        params = 0

        hidden_size = self.hidden_size or hidden_size
        params += 3 * d_model * hidden_size
        if self.bias:
            params += 2 * hidden_size + d_model

        return params

    def build(
        self,
        d_model: int,
        hidden_size: int,
        *,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "SharedMLP":
        """
        Build the corresponding shared MLP module.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            init_device=init_device,
        )
        kwargs.setdefault("hidden_size", hidden_size)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            if self.name == SharedMLPType.default:
                return SharedMLP(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class SharedMLP(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        weighted_sum: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.mlp = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            bias=bias,
            dtype=dtype,
            init_device=init_device,
        )
        self.weighted_sum = weighted_sum

    def forward(self, x: torch.Tensor, experts_out: torch.Tensor, top_k: int) -> torch.Tensor:
        shared_out = self.mlp(x)
        if self.weighted_sum:
            # Weighted by number of experts used
            n_active_experts = top_k + 1
            shared_out = shared_out / n_active_experts
            return shared_out.add(experts_out, alpha=top_k / n_active_experts)
        else:
            return shared_out + experts_out

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        # Alternatively could do colwise->rowwise->colwise parallelism
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "mlp.w1": SequenceParallel(),
                "mlp.w2": SequenceParallel(),
                "mlp.w3": SequenceParallel(),
                "mlp": PrepareModuleOutput(
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Shard(1),),
                    use_local_output=True,
                ),
            },
        )
