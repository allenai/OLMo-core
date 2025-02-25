from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributed import DeviceMesh

from olmo_core.config import Config, StrEnum
from olmo_core.utils import ensure_multiple_of

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
    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Shard inputs on their sequence dimension, optionally adding padding if needed.

        :returns: The local shards of the inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Shard inputs by document on their sequence dimension, optionally adding padding if needed.

        :returns: The local shards of the inputs and the new cumulative document lengths after padding.
        """
        raise NotImplementedError


class ContextParallelZigZagLoadBalancer(ContextParallelLoadBalancer):
    """
    Implements the zig-zag load-balancing strategy.
    """

    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        assert len(inputs) == len(seq_dims)
        assert len(set(x.shape[seq_dim] for x, seq_dim in zip(inputs, seq_dims))) == 1
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if length_multiple is None:
            length_multiple = 2 * self.cp_world_size
        elif length_multiple % (2 * self.cp_world_size) != 0:
            raise RuntimeError(
                f"length multiple ({length_multiple}) must be divisible by "
                f"2 x CP degree ({2 * self.cp_world_size})"
            )

        out = []
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            if x.shape[seq_dim] % length_multiple != 0:
                if pad_value is None:
                    raise RuntimeError(
                        f"sequence dimension size ({x.shape[seq_dim]}) must be divisible by "
                        f"{length_multiple}, otherwise provide a padding value"
                    )
                else:
                    x, _ = self.pad(x, seq_dim, pad_value, length_multiple=length_multiple)

            x_chunks = x.chunk(2 * self.cp_world_size, dim=seq_dim)
            local_value = torch.cat(
                [x_chunks[self.cp_rank], x_chunks[2 * self.cp_world_size - self.cp_rank - 1]],
                dim=seq_dim,
            )
            out.append(local_value.contiguous())

        return out

    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        assert len(inputs) == len(seq_dims)
        assert len(set(x.shape[seq_dim] for x, seq_dim in zip(inputs, seq_dims))) == 1
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if cu_doc_lens.device.type != "cpu":
            raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
        if cu_doc_lens.ndim != 1:
            raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
        if cu_doc_lens[0] != 0:
            raise RuntimeError("expected 'cu_doc_lens' to start with a 0")

        out = []
        padding_added = [0 for _ in range(len(cu_doc_lens) - 1)]
        final_padding: Optional[int] = None if length_multiple is None else 0
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            local_values = []
            for i in range(len(cu_doc_lens) - 1):
                start, end = cu_doc_lens[i], cu_doc_lens[i + 1]
                # NOTE: Since 'torch.slice' is not available from the Python API we just call
                # the JIT op directly.
                x_doc_slice = torch.ops.aten.slice(x, dim=seq_dim, start=start, end=end)  # type: ignore
                if x_doc_slice.shape[seq_dim] % (2 * self.cp_world_size) != 0:
                    if pad_value is None:
                        raise RuntimeError(
                            f"document length ({x_doc_slice.shape[seq_dim]}) must be divisible by "
                            f"2 x CP degree ({2 * self.cp_world_size}), otherwise provide a padding value"
                        )
                    else:
                        x_doc_slice, padding = self.pad(x_doc_slice, seq_dim, pad_value)
                        padding_added[i] = padding

                x_chunks = x_doc_slice.chunk(2 * self.cp_world_size, dim=seq_dim)
                local_values.extend(
                    [
                        x_chunks[self.cp_rank],
                        x_chunks[2 * self.cp_world_size - 1 - self.cp_rank],
                    ]
                )
            local_value = torch.cat(local_values, dim=seq_dim).contiguous()
            if length_multiple is not None and local_value.shape[seq_dim] % length_multiple != 0:
                if pad_value is None:
                    raise RuntimeError(
                        "You must provide a 'pad_value' when 'length_multiple' is specified!"
                    )
                else:
                    local_value, final_padding = self.pad(
                        local_value, seq_dim, pad_value, length_multiple=length_multiple
                    )
            out.append(local_value)

        if pad_values is not None:
            cumulative_padding = torch.cat(
                [
                    torch.tensor([0], dtype=cu_doc_lens.dtype, device=cu_doc_lens.device),
                    torch.tensor(padding_added, device=cu_doc_lens.device).cumsum(
                        0, dtype=cu_doc_lens.dtype
                    ),
                ]
            )
            cu_doc_lens = cu_doc_lens + cumulative_padding

        local_cu_doc_lens = cu_doc_lens // self.cp_world_size
        if final_padding is not None:
            local_cu_doc_lens = torch.cat(
                [local_cu_doc_lens, (local_cu_doc_lens[-1] + final_padding).unsqueeze(0)]
            )

        return out, local_cu_doc_lens

    def pad(
        self,
        x: torch.Tensor,
        seq_dim: int,
        value: Union[int, float],
        length_multiple: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        if length_multiple is None:
            length_multiple = 2 * self.cp_world_size
        pad_to = ensure_multiple_of(x.shape[seq_dim], length_multiple)
        padding_to_add = pad_to - x.shape[seq_dim]
        padding = (0, 0) * (x.ndim - seq_dim - 1) + (0, padding_to_add)
        return F.pad(x, padding, value=value), padding_to_add
