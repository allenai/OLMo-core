from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributed import DeviceMesh

from olmo_core.config import Config, StrEnum

from ..utils import get_rank, get_world_size


class ContextParallelLoadBalancerType(StrEnum):
    """
    An enumeration of the different :class:`ContextParallelLoadBalancer` implementations.
    """

    zig_zag = "zig_zag"
    """
    ➡️ :class:`ContextParallelZigZagLoadBalancer`
    """

    def build(self, cp_mesh: DeviceMesh) -> "ContextParallelLoadBalancer":
        """
        Build the load balancer.
        """
        pg = cp_mesh.get_group()
        cp_rank = get_rank(pg)
        cp_world_size = get_world_size(pg)
        if self == self.zig_zag:
            return ContextParallelZigZagLoadBalancer(cp_rank=cp_rank, cp_world_size=cp_world_size)
        else:
            raise NotImplementedError(self)


@dataclass
class ContextParallelConfig(Config):
    """
    Configuration class for context parallelism (CP).
    """

    degree: int
    """
    The CP degree.
    """

    load_balancer: ContextParallelLoadBalancerType = ContextParallelLoadBalancerType.zig_zag
    """
    The type of load balancer to use.
    """


class ContextParallelLoadBalancer(metaclass=ABCMeta):
    """
    A class that handles the logic of sharding inputs on the sequence dimension
    for context parallelism.
    """

    def __init__(self, *, cp_rank: int, cp_world_size: int):
        self.cp_rank = cp_rank
        self.cp_world_size = cp_world_size

    @abstractmethod
    def shard(
        self, x: torch.Tensor, seq_dim: int, cu_doc_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Shard an input on its sequence dimension.
        """
        raise NotImplementedError


class ContextParallelZigZagLoadBalancer(ContextParallelLoadBalancer):
    """
    Implements the zig-zag load-balancing strategy.
    """

    def shard(
        self, x: torch.Tensor, seq_dim: int, cu_doc_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if cu_doc_lens is not None:
            if cu_doc_lens.device.type != "cpu":
                raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
            if cu_doc_lens.ndim != 1:
                raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
            if cu_doc_lens[0] != 0:
                raise RuntimeError("expected 'cu_doc_lens' to start with a 0")
            if not torch.all(cu_doc_lens % (2 * self.cp_world_size) == 0):
                raise RuntimeError(
                    f"document lengths must all be divisible by 2 x CP degree ({2 * self.cp_world_size})"
                )

            local_values = []
            for i in range(len(cu_doc_lens) - 1):
                start, end = cu_doc_lens[i], cu_doc_lens[i + 1]
                # NOTE: Since 'torch.slice' is not available from the Python API, we just call
                # the JIT op directly.
                x_doc_slice = torch.ops.aten.slice(x, dim=seq_dim, start=start, end=end)  # type: ignore
                x_chunks = x_doc_slice.chunk(2 * self.cp_world_size, dim=seq_dim)
                local_values.extend(
                    [
                        x_chunks[self.cp_rank],
                        x_chunks[2 * self.cp_world_size - 1 - self.cp_rank],
                    ]
                )
            return torch.cat(local_values, dim=seq_dim).contiguous()
        else:
            if x.shape[seq_dim] % self.cp_world_size != 0:
                raise RuntimeError(
                    f"sequence dimension size ({x.shape[seq_dim]}) must be divisible by "
                    f"the CP degree ({self.cp_world_size})"
                )

            x_chunks = x.chunk(2 * self.cp_world_size, dim=seq_dim)
            local_value = torch.cat(
                [x_chunks[self.cp_rank], x_chunks[2 * self.cp_world_size - self.cp_rank - 1]],
                dim=seq_dim,
            )
            return local_value.contiguous()
