from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributed import DeviceMesh

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.utils import ensure_multiple_of


class RingAttentionLoadBalancerType(StrEnum):
    """
    An enumeration of the different :class:`RingAttentionLoadBalancer` implementations.
    """

    zig_zag = "zig_zag"
    """
    ➡️ :class:`RingAttentionZigZagLoadBalancer`
    """

    llama3 = "llama3"
    """
    ➡️ :class:`RingAttentionLlama3LoadBalancer`
    """

    def build(self, cp_mesh: DeviceMesh) -> "RingAttentionLoadBalancer":
        """
        Build the load balancer.
        """
        pg = cp_mesh.get_group()
        cp_rank = get_rank(pg)
        cp_world_size = get_world_size(pg)
        if self == self.zig_zag:
            return RingAttentionZigZagLoadBalancer(cp_rank=cp_rank, cp_world_size=cp_world_size)
        elif self == self.llama3:
            return RingAttentionLlama3LoadBalancer(cp_rank=cp_rank, cp_world_size=cp_world_size)
        else:
            raise NotImplementedError(self)


class RingAttentionLoadBalancer(metaclass=ABCMeta):
    """
    A class that handles the logic of sharding inputs on the sequence dimension
    for ring attention (context parallelism).
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

        .. important::
            If using intra-document masking, use :meth:`batch_shard_by_document` instead.

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
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Same as :meth:`batch_shard` but for strategies that support intra-document masking.

        :returns: The local shards of the inputs and any other additional inputs required for the
            corresponding ring attention implementation.
        """
        raise NotImplementedError


class RingAttentionZigZagLoadBalancer(RingAttentionLoadBalancer):
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
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
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

        local_max_doc_len = (local_cu_doc_lens[1:] - local_cu_doc_lens[:-1]).max().item()

        return out, dict(cu_doc_lens=local_cu_doc_lens, max_doc_len=local_max_doc_len)

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


class RingAttentionLlama3LoadBalancer(RingAttentionLoadBalancer):
    """
    Implements Llama3's load-balancing strategy for context parallelism.

    The Llama3 strategy assigns each rank a contiguous slice of the full sequence.
    Rank ``i`` receives positions ``[i * local_len, (i + 1) * local_len)`` where
    ``local_len = total_seq_len // cp_world_size``.

    This strategy is designed specifically for **intra-document masking** with
    variable-length documents packed into a single sequence. It computes separate
    cumulative sequence lengths for queries (``cu_doc_lens_q``) and keys
    (``cu_doc_lens_k``) per rank, enabling proper causal masking across document
    boundaries within the ring attention loop. Padding is added as a synthetic
    document at the end when the total sequence length is not divisible by the
    context parallel world size.

    .. note::
        This strategy only supports :meth:`batch_shard_by_document` and will raise
        an error if :meth:`batch_shard` is called directly.
    """

    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        del inputs, seq_dims, pad_values, length_multiple
        raise NotImplementedError(
            f"{self.__class__.__name__} should only be used with intra-document masking. "
            "Please use the 'batch_shard_by_document()' instead."
        )

    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        try:
            from ring_flash_attn import llama3_flash_attn_prepare_cu_seqlens
        except ImportError as e:
            raise RuntimeError(f"ring-flash-attn is required for {self.__class__.__name__}") from e

        assert len(inputs) == len(seq_dims)
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if cu_doc_lens.device.type != "cpu":
            raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
        if cu_doc_lens.ndim != 1:
            raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
        if cu_doc_lens[0] != 0:
            raise RuntimeError("expected 'cu_doc_lens' to start with a 0")

        if length_multiple is None:
            length_multiple = self.cp_world_size
        else:
            length_multiple = length_multiple * self.cp_world_size

        total_length = int(cu_doc_lens[-1])
        padding_to_add = total_length - ensure_multiple_of(total_length, length_multiple)
        local_length = (total_length + padding_to_add) // self.cp_world_size

        if padding_to_add > 0:
            if pad_values is None:
                raise RuntimeError("'pad_values' is required since padding is needed")

            cu_doc_lens = torch.cat(
                [
                    cu_doc_lens,
                    torch.tensor(
                        [total_length + padding_to_add],
                        dtype=cu_doc_lens.dtype,
                        device=cu_doc_lens.device,
                    ),
                ]
            )

        out = []
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            if x.shape[seq_dim] != total_length:
                raise RuntimeError(
                    f"expected input to be have size {total_length} on the sequence dimension "
                    f"but got {x.shape[seq_dim]}"
                )

            if padding_to_add > 0:
                assert pad_value is not None
                x = self.pad(x, seq_dim, padding_to_add, pad_value)

            # NOTE: Since 'torch.slice' is not available from the Python API we just call
            # the JIT op directly.
            local_value = torch.ops.aten.slice(  # type: ignore
                x,
                dim=seq_dim,
                start=self.cp_rank * local_length,
                end=(self.cp_rank + 1) * local_length,
            ).contiguous()
            out.append(local_value)

        (
            cu_doc_lens_q,
            cu_doc_lens_k,
            max_doc_len_q,
            max_doc_len_k,
            local_k_slice,
        ) = llama3_flash_attn_prepare_cu_seqlens(
            cu_doc_lens,
            causal=True,
            rank=self.cp_rank,
            world_size=self.cp_world_size,
        )

        return out, dict(
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
        )

    def pad(
        self,
        x: torch.Tensor,
        seq_dim: int,
        padding_to_add: int,
        value: Union[int, float],
    ) -> Tuple[torch.Tensor, int]:
        padding = (0, 0) * (x.ndim - seq_dim - 1) + (0, padding_to_add)
        return F.pad(x, padding, value=value), padding_to_add


class UlyssesLoadBalancer(RingAttentionLoadBalancer):
    """
    Implements simple contiguous sequence sharding for Ulysses-style context parallelism.

    Unlike ring attention which uses zig-zag or other interleaving strategies,
    Ulysses just needs simple contiguous chunking since the all-to-all communication
    handles the sequence/head exchange.
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
            length_multiple = self.cp_world_size
        elif length_multiple % self.cp_world_size != 0:
            raise RuntimeError(
                f"length multiple ({length_multiple}) must be divisible by "
                f"CP degree ({self.cp_world_size})"
            )

        out = []
        for x, seq_dim, pad_value in zip(
            inputs,
            seq_dims,
            pad_values or [None for _ in range(len(inputs))],  # type: ignore
        ):
            seq_len = x.shape[seq_dim]

            # Pad if needed to make divisible by CP world size
            if seq_len % length_multiple != 0:
                if pad_value is None:
                    raise RuntimeError(
                        f"sequence dimension size ({seq_len}) must be divisible by "
                        f"{length_multiple}, otherwise provide a padding value"
                    )
                else:
                    x, _ = self.pad(x, seq_dim, pad_value, length_multiple=length_multiple)

            # Simple contiguous chunking - each rank gets seq_len / cp_world_size tokens
            local_seq_len = x.shape[seq_dim] // self.cp_world_size
            start = self.cp_rank * local_seq_len
            end = start + local_seq_len

            # Use torch.ops.aten.slice for efficiency
            local_value = torch.ops.aten.slice(x, dim=seq_dim, start=start, end=end).contiguous()
            out.append(local_value)

        return out

    def batch_shard_by_document(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        cu_doc_lens: torch.Tensor,
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        # Ulysses reconstructs full sequences via all-to-all, so we don't shard cu_doc_lens.
        # We just shard the inputs and pass through the original document boundaries.
        assert len(inputs) == len(seq_dims)
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        if cu_doc_lens.device.type != "cpu":
            raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
        if cu_doc_lens.ndim != 1:
            raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
        if cu_doc_lens[0] != 0:
            raise RuntimeError("expected 'cu_doc_lens' to start with a 0")

        if length_multiple is None:
            length_multiple = self.cp_world_size
        elif length_multiple % self.cp_world_size != 0:
            raise RuntimeError(
                f"length multiple ({length_multiple}) must be divisible by "
                f"CP degree ({self.cp_world_size})"
            )

        total_length = int(cu_doc_lens[-1])
        padded_total_length = ensure_multiple_of(total_length, length_multiple)
        padding_to_add = padded_total_length - total_length

        # Shard the inputs (handles padding to length_multiple internally)
        out = self.batch_shard(
            inputs=inputs,
            seq_dims=seq_dims,
            pad_values=pad_values,
            length_multiple=length_multiple,
        )

        # Compute max_doc_len from the (potentially padded) cu_doc_lens
        max_doc_len = (cu_doc_lens[1:] - cu_doc_lens[:-1]).max().item()
        if padding_to_add > 0:
            cu_doc_lens = torch.cat(
                [
                    cu_doc_lens,
                    (cu_doc_lens[-1] + padding_to_add).unsqueeze(0),
                ]
            )
            max_doc_len = max(max_doc_len, padding_to_add)

        # Pass through the (possibly padded) cu_doc_lens and max_doc_len
        # since Ulysses reconstructs full sequences before attention
        return out, dict(cu_doc_lens=cu_doc_lens, max_doc_len=max_doc_len)

    def pad(
        self,
        x: torch.Tensor,
        seq_dim: int,
        value: Union[int, float],
        length_multiple: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        if length_multiple is None:
            length_multiple = self.cp_world_size
        pad_to = ensure_multiple_of(x.shape[seq_dim], length_multiple)
        padding_to_add = pad_to - x.shape[seq_dim]
        padding = (0, 0) * (x.ndim - seq_dim - 1) + (0, padding_to_add)
        return F.pad(x, padding, value=value), padding_to_add
