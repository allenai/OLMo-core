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

    ulysses = "ulysses"
    """
    ➡️ :class:`UlyssesA2ALoadBalancer`
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
        elif self == self.ulysses:
            return UlyssesA2ALoadBalancer(cp_rank=cp_rank, cp_world_size=cp_world_size)
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
    Implements the zig-zag load-balancing strategy. Assumes 2xCP chunking and alternation to smooth ring waves.
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
    Implements Llama3's load-balancing strategy.
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


class UlyssesA2ALoadBalancer(RingAttentionLoadBalancer):
    """
    Implements *contiguous* sequence sharding for Ulysses-style (A2A) context parallelism.

    Notes
    -----
    - This is *not* a ring strategy; no zig-zag interleaving and no ring metadata.
    - For fixed-length batches, we simply slice the sequence evenly across CP ranks.
    - For intra-document masking, we return local 'cu_doc_lens' and 'max_doc_len'
      for the *local* slice on each CP rank.
    - Padding is applied at the *end* (global tail). If padding is required and
      no pad_value(s) are provided, we raise an error (matches existing style).
    """

    # ---------------------------
    # Public API
    # ---------------------------

    def batch_shard(
        self,
        *,
        inputs: List[torch.Tensor],
        seq_dims: List[int],
        pad_values: Optional[List[Union[int, float]]] = None,
        length_multiple: Optional[int] = None,
    ) -> List[torch.Tensor]:
        assert len(inputs) == len(seq_dims)
        # All inputs must share the same sequence length (but may place it on different dims)
        seq_lens = [x.shape[seq_dim] for x, seq_dim in zip(inputs, seq_dims)]
        assert len(set(seq_lens)) == 1, "all inputs must have the same sequence length"
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        # We want global length to be a multiple of both 'cp_world_size' and any requested multiple.
        required_multiple = self._required_multiple(length_multiple)
        target_len = self._ensure_multiple(seq_lens[0], required_multiple)

        # Pad (at the global tail) if needed.
        if target_len != seq_lens[0]:
            if pad_values is None:
                raise RuntimeError(
                    f"sequence dimension size ({seq_lens[0]}) must be divisible by "
                    f"{required_multiple}, otherwise provide a padding value"
                )

        out: List[torch.Tensor] = []
        for idx, (x, seq_dim) in enumerate(zip(inputs, seq_dims)):
            pad_value = None if pad_values is None else pad_values[idx]
            if target_len != x.shape[seq_dim]:
                assert pad_value is not None
                x, _ = self._pad_tail(x, seq_dim, pad_value, target_len)
            local = self._slice_contiguous_rank_chunk(x, seq_dim, target_len)
            out.append(local.contiguous())
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
        if pad_values is not None:
            assert len(inputs) == len(pad_values)

        # Validate cu_doc_lens
        if cu_doc_lens.device.type != "cpu":
            raise RuntimeError("expected 'cu_doc_lens' to be on CPU")
        if cu_doc_lens.ndim != 1:
            raise RuntimeError("expected 'cu_doc_lens' to be a 1D tensor")
        if cu_doc_lens[0] != 0:
            raise RuntimeError("expected 'cu_doc_lens' to start with a 0")

        total_len = int(cu_doc_lens[-1])
        # Make the *global* length a multiple of both 'cp_world_size' and any requested multiple.
        required_multiple = self._required_multiple(length_multiple)
        target_len = self._ensure_multiple(total_len, required_multiple)
        padding_to_add = target_len - total_len
        local_len = target_len // self.cp_world_size
        start = self.cp_rank * local_len
        end = (self.cp_rank + 1) * local_len

        # If we need global tail padding, extend cu_doc_lens by one terminal entry.
        if padding_to_add > 0:
            if pad_values is None:
                raise RuntimeError(
                    f"'pad_values' is required since global tail padding ({padding_to_add}) is needed"
                )
            cu_doc_lens = torch.cat(
                [
                    cu_doc_lens,
                    torch.tensor(
                        [total_len + padding_to_add],
                        dtype=cu_doc_lens.dtype,
                        device=cu_doc_lens.device,
                    ),
                ]
            )

        # Shard each input contiguously, padding at tail if needed.
        out: List[torch.Tensor] = []
        for idx, (x, seq_dim) in enumerate(zip(inputs, seq_dims)):
            if x.shape[seq_dim] != total_len:
                raise RuntimeError(
                    f"expected input to have size {total_len} on the sequence dimension, "
                    f"but got {x.shape[seq_dim]}"
                )
            if padding_to_add > 0:
                assert pad_values is not None
                x, _ = self._pad_tail(x, seq_dim, pad_values[idx], target_len)
            # NOTE: can use torch.ops.aten.slice for parity with other impls
            local = torch.ops.aten.slice(x, dim=seq_dim, start=start, end=end)  # type: ignore
            out.append(local.contiguous())

        # Build *local* cu_doc_lens for the [start, end) window.
        local_cu_doc_lens = self._clip_and_shift_cu_doc_lens(cu_doc_lens, start, end)
        local_max_doc_len = int((local_cu_doc_lens[1:] - local_cu_doc_lens[:-1]).max().item())

        return out, dict(cu_doc_lens=local_cu_doc_lens, max_doc_len=local_max_doc_len)

    # ---------------------------
    # Helpers
    # ---------------------------

    def _required_multiple(self, length_multiple: Optional[int]) -> int:
        """
        Return the global length multiple we must satisfy. This is LCM(cp_world_size, length_multiple or 1).
        """
        user_mult = 1 if length_multiple is None else length_multiple
        # Python 3.9+: math.lcm
        return math.lcm(self.cp_world_size, user_mult)

    @staticmethod
    def _ensure_multiple(length: int, multiple: int) -> int:
        """
        Return the smallest value >= length that is divisible by 'multiple'.
        """
        if length % multiple == 0:
            return length
        return ((length + multiple - 1) // multiple) * multiple

    @staticmethod
    def _pad_tail(
        x: torch.Tensor,
        seq_dim: int,
        pad_value: Union[int, float],
        pad_to: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Right-pad along 'seq_dim' to reach 'pad_to'. Returns (tensor, padding_added).
        """
        pad_add = pad_to - x.shape[seq_dim]
        if pad_add < 0:
            raise RuntimeError("pad_to must be >= current length")
        padding = (0, 0) * (x.ndim - seq_dim - 1) + (0, pad_add)
        return F.pad(x, padding, value=pad_value), pad_add

    def _slice_contiguous_rank_chunk(
        self, x: torch.Tensor, seq_dim: int, global_len: int
    ) -> torch.Tensor:
        """
        Extract this rank's contiguous chunk along 'seq_dim' from a (possibly padded) global tensor of length 'global_len'.
        """
        local_len = global_len // self.cp_world_size
        start = self.cp_rank * local_len
        end = (self.cp_rank + 1) * local_len
        # Use normal slicing for readability here.
        sl = [slice(None)] * x.ndim
        sl[seq_dim] = slice(start, end)
        return x[tuple(sl)]

    @staticmethod
    def _clip_and_shift_cu_doc_lens(
        cu_doc_lens: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """
        Intersect global cu_doc_lens with [start, end) and shift so that local start becomes 0.
        Ensures 0 and local_len are present as sentinels.
        """
        # Keep boundaries that intersect [start, end)
        # For each boundary t, clamp to [start, end] and shift by -start
        inside = []
        prev = start
        for t in cu_doc_lens.tolist():
            t_clamped = min(max(t, start), end)
            if t_clamped != prev:
                inside.append(t_clamped - start)
                prev = t_clamped
        # Ensure 0 at head
        if not inside or inside[0] != 0:
            inside = [0] + inside
        # Ensure local_len at tail
        local_len = end - start
        if inside[-1] != local_len:
            inside.append(local_len)
        return torch.tensor(inside, dtype=cu_doc_lens.dtype, device=cu_doc_lens.device)
