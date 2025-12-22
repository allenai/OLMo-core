# This code was originally adapted from 'https://github.com/databricks/megablocks/blob/main/megablocks/layers/moe.py'.
# It has since changed substantially.

from abc import abstractmethod
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh

from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.ops import moe as ops
from olmo_core.utils import (
    ensure_multiple_of,
    get_default_device,
    move_to_device,
    warn_once,
)

from ..buffer_cache import BufferCache
from .mlp import DroplessMoEMLP, MoEMLP, MoEMLPBase

__all__ = ["ParallelMLPBase", "ParallelMLP", "ParallelDroplessMLP"]


class PermutedAllToAllOutput(NamedTuple):
    parallel_x: torch.Tensor
    parallel_indices: torch.Tensor
    parallel_bin_ids: Optional[torch.Tensor]
    parallel_bins: torch.Tensor
    parallel_batch_size_per_expert: torch.Tensor
    recv_counts: Optional[List[int]]
    send_counts: Optional[List[int]]
    expert_capacity: int
    handle: dist.Work


class ParallelMLPBase(nn.Module):
    """
    Wraps an MoE MLP layer to coordinate the routing and expert parallelism.
    """

    def __init__(self, *, mlp: MoEMLPBase, top_k: int, cache: Optional[BufferCache] = None):
        super().__init__()
        self.mlp = mlp
        self.top_k = top_k
        self._cache = cache or BufferCache()
        self._expert_parallel_enabled: bool = False

    def warmup_cache(self, max_local_microbatch_size: int):
        del max_local_microbatch_size

    @property
    def d_model(self) -> int:
        return self.mlp.d_model

    @property
    def num_experts(self) -> int:
        return self.mlp.num_experts

    @property
    def num_local_experts(self) -> int:
        return self.mlp.num_local_experts

    @property
    def hidden_sharding_degree(self) -> int:
        return self.mlp.hidden_sharding_degree

    @property
    def ep_world_size(self) -> int:
        if self.ep_pg is not None:
            return get_world_size(self.ep_pg)
        else:
            return 1

    @property
    def ep_pg(self) -> Optional[dist.ProcessGroup]:
        return self.mlp.ep_pg

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        Apply expert parallelism.
        """
        self.mlp.apply_ep(ep_mesh, **kwargs)
        self._expert_parallel_enabled = True

    def apply_tp(self, tp_mesh: DeviceMesh, **kwargs):
        """
        Apply tensor parallelism.
        """
        self.mlp.apply_tp(tp_mesh, **kwargs)
        self._expert_parallel_enabled = True

    def prepare_experts_for_fsdp(self, **kwargs):
        """
        Should be called before wrapping this module with FSDP2.
        """
        self.mlp.prepare_experts_for_fsdp(**kwargs)

    def prepare_experts_for_ddp(self, **kwargs):
        """
        Should be called before wrapping this module with DDP2.
        """
        self.mlp.prepare_experts_for_ddp(**kwargs)

    def indices_and_bins(
        self,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param expert_indices: A 1D tensor.
        :param batch_size_per_expert: A 1D tensor.
        """
        expert_indices = expert_indices.int()

        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        # shape: (batch_size,), (batch_size,)
        # TODO: for non-dropless MoE, should do secondary sort by expert weight so we drop tokens
        # with the lowest expert weight.
        bin_ids, indices = torch.sort(expert_indices)

        # Calculate the bin bounds for the sorted items/tokens.
        # shape: (num_experts,)
        bins = torch.empty_like(batch_size_per_expert, dtype=torch.int32)
        torch.cumsum(batch_size_per_expert, 0, out=bins)

        return indices.int(), bin_ids, bins

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: The input of shape ``(N, d_model)``.
        :param expert_weights: Expert weights of shape ``(N, top_k)``.
        :param expert_indices: The indices of the top-k experts, shape ``(N, top_k)``.
        :param batch_size_per_expert: The number of items routed to each expert, shape ``(num_experts,)``.

        :returns: The output with the same shape as ``x``.
        """
        x, expert_weights, expert_indices, batch_size_per_expert = (
            get_local_tensor(x),
            get_local_tensor(expert_weights),
            get_local_tensor(expert_indices),
            get_local_tensor(batch_size_per_expert),
        )

        in_shape = x.size()

        # shape: (N, d_model)
        x = x.view(-1, x.shape[-1])
        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.indices_and_bins(expert_indices, batch_size_per_expert)

        # Compute the experts.
        if not self._expert_parallel_enabled:
            x = self.forward_once(
                x,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                indices=indices,
                bin_ids=bin_ids,
                bins=bins,
                batch_size_per_expert=batch_size_per_expert,
            )
        else:
            x = self.parallel_forward_once(
                x,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                indices=indices,
                bin_ids=bin_ids,
                bins=bins,
                batch_size_per_expert=batch_size_per_expert,
            )

        return x.view(in_shape)

    @abstractmethod
    def forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: The input of shape ``(*, d_model)``, typically ``(num_docs, seq_len, d_model)``
            such that ``num_docs x seq_len = batch_size``.
        :param expert_weights: Expert weights of shape ``(batch_size, top_k)``, where ``batch_size``
            typically equals ``num_docs x seq_len``.
        :param expert_indices: The indices of the top-k experts, shape ``(batch_size, top_k)``.
        """
        raise NotImplementedError

    def parallel_forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: The input of shape ``(*, d_model)``.
        :param expert_weights: Expert weights of shape ``(N, top_k)``, where ``N``
            typically equals ``batch_size x seq_len``.
        :param expert_indices: The indices of the top-k experts, shape ``(N, top_k)``.
        """
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignment. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            parallel_x_handle,
        ) = self.permute_and_all_to_all(
            x,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        parallel_x_handle.wait()
        parallel_x = self.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x, x_handle = self.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        x_handle.wait()

        x = self.unpermute(
            x,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        )
        return x

    @abstractmethod
    def permute_and_all_to_all(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> PermutedAllToAllOutput:
        raise NotImplementedError

    @abstractmethod
    def compute_local_experts(
        self,
        parallel_x,
        *,
        parallel_indices: torch.Tensor,
        parallel_bin_ids: Optional[torch.Tensor],
        parallel_bins: torch.Tensor,
        parallel_batch_size_per_expert: torch.Tensor,
        expert_capacity: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reverse_all_to_all(
        self,
        parallel_x: torch.Tensor,
        *,
        send_counts: Optional[List[int]],
        recv_counts: Optional[List[int]],
    ) -> Tuple[torch.Tensor, dist.Work]:
        raise NotImplementedError

    @abstractmethod
    def unpermute(
        self,
        x,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        raise NotImplementedError

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        # Each token activates top_k experts.
        # The expert MLP parameters are typically stored as a single batched tensor with a leading
        # expert dimension (not shared weights). On average, each token "touches" top_k experts,
        # i.e. a fraction (top_k / num_experts) of the total expert parameters.
        expert_params = sum(p.numel() for p in self.mlp.parameters())
        return 6 * int(expert_params * self.top_k / self.num_experts)


class ParallelMLP(ParallelMLPBase):
    def num_flops_per_token(self, seq_len: int) -> int:
        warn_once(
            f"{self.__class__.__name__}: approximating extra FLOPs from padding experts to a fixed capacity using "
            "capacity_factor. The true overhead depends on batch size and rounding to alignment constraints.",
            UserWarning,
        )
        dropless_flops = super().num_flops_per_token(seq_len)
        return int(dropless_flops * self.capacity_factor)

    def __init__(
        self,
        *,
        mlp: MoEMLP,
        top_k: int,
        capacity_factor: float,
        cache: Optional[BufferCache] = None,
        max_local_microbatch_size: Optional[int] = None,
    ):
        super().__init__(mlp=mlp, top_k=top_k, cache=cache)
        self.capacity_factor = capacity_factor
        self.tp_degree: int = 1
        self.max_local_microbatch_size = max_local_microbatch_size
        if self.max_local_microbatch_size is not None:
            self.warmup_cache(self.max_local_microbatch_size)

    def warmup_cache(self, max_local_microbatch_size: int):
        self.max_local_microbatch_size = max_local_microbatch_size
        expert_capacity = self.expert_capacity(self.max_local_microbatch_size // self.tp_degree)
        local_expert_capacity = expert_capacity // self.ep_world_size
        self._get_parallel_indices_and_bins(
            expert_capacity=expert_capacity,
            local_expert_capacity=local_expert_capacity,
            device=get_default_device(),
        )

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        super().apply_ep(ep_mesh, **kwargs)
        if self.max_local_microbatch_size is not None:
            self.warmup_cache(self.max_local_microbatch_size)

    def apply_tp(self, tp_mesh: DeviceMesh, **kwargs):
        super().apply_tp(tp_mesh, **kwargs)
        self.tp_degree = tp_mesh.size()
        if self.max_local_microbatch_size is not None:
            self.warmup_cache(self.max_local_microbatch_size)

    def expert_capacity(self, local_batch_size: int) -> int:
        # NOTE: need to ensure this is the same across the process group.
        # If local batch sizes are different then these will be different, and `parallel_forward_once`
        # will break. This shouldn't be a problem with our trainer, but would be an issue for inference.
        # To avoid that you could set `self.max_local_microbatch_size` up-front.
        if self.max_local_microbatch_size is not None:
            max_local_microbatch_size = self.max_local_microbatch_size // self.tp_degree
            if local_batch_size > max_local_microbatch_size:
                raise RuntimeError(
                    f"Local batch size ({local_batch_size:d}) bigger than "
                    f"configured max local batch size ({max_local_microbatch_size:d})"
                )
            else:
                local_batch_size = max_local_microbatch_size

        ideal_local_inputs_per_expert = self.top_k * local_batch_size / self.num_experts
        allowed_local_inputs_per_expert = ensure_multiple_of(
            int(self.capacity_factor * ideal_local_inputs_per_expert), 8
        )
        return self.ep_world_size * allowed_local_inputs_per_expert

    @torch.no_grad()
    def _get_parallel_indices_and_bins(
        self, *, expert_capacity: int, local_expert_capacity: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices_cache_key = f"moe_par_expert_indices_{expert_capacity}_{local_expert_capacity}"
        bins_cache_key = f"moe_par_expert_bins_{expert_capacity}_{local_expert_capacity}"

        if (
            parallel_indices := self._cache.get_for_device(indices_cache_key, device)
        ) is not None and (
            parallel_bins := self._cache.get_for_device(bins_cache_key, device)
        ) is not None:
            return parallel_indices, parallel_bins

        # Construct the expert indices for the permuted tokens.
        # shape: (num_experts,) = (num_local_experts * ep_world_size,)
        parallel_top_expert = torch.remainder(
            torch.arange(
                self.num_experts * self.hidden_sharding_degree,
                dtype=torch.int32,
                device=device,
            ),
            self.num_local_experts,
        )

        # shape: (num_local_experts * ep_world_size * local_expert_capacity,)
        #      = (num_local_experts * expert_capacity,)
        parallel_top_expert = torch.repeat_interleave(
            parallel_top_expert,
            local_expert_capacity,
            output_size=parallel_top_expert.numel() * local_expert_capacity,
        )

        # shape: (num_local_experts * expert_capacity,)
        _, parallel_indices = torch.sort(parallel_top_expert)
        parallel_indices = parallel_indices.int()

        # Calculate the bins boundaries from the token counts.
        # shape: (num_local_experts,)
        parallel_batch_size_per_expert = move_to_device(
            torch.tensor([expert_capacity] * self.num_local_experts),
            parallel_indices.device,
        )
        # shape: (num_local_experts,)
        parallel_bins = torch.empty_like(parallel_batch_size_per_expert, dtype=torch.int32)
        torch.cumsum(parallel_batch_size_per_expert, 0, out=parallel_bins)

        self._cache[indices_cache_key] = parallel_indices
        self._cache[bins_cache_key] = parallel_bins

        return parallel_indices, parallel_bins

    def forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        del bin_ids, batch_size_per_expert, expert_indices

        batch_size = expert_weights.numel() // self.top_k
        expert_capacity = self.expert_capacity(batch_size)

        x = self.permute_and_compute(
            x,
            indices=indices,
            expert_weights=expert_weights,
            bins=bins,
            expert_capacity=expert_capacity,
            top_k=self.top_k,
        )
        return x

    def permute_and_all_to_all(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> PermutedAllToAllOutput:
        del bin_ids

        expert_capacity = self.expert_capacity(x.shape[0])
        local_expert_capacity = expert_capacity // self.ep_world_size

        # Permute locally so that the tokens for each device are stored contiguously.
        # shape: (num_experts, local_expert_capacity, d_model)
        x = ops.binned_gather(x, indices, bins, local_expert_capacity, self.top_k)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        if self.hidden_sharding_degree > 1:
            # shape: (num_local_experts, ep_world_size // hidden_sharding_degree, local_expert_capacity, d_model)
            x = x.view(self.num_local_experts, -1, local_expert_capacity, self.d_model)
            # shape: (num_experts * hidden_sharding_degree, local_expert_capacity, d_model)
            x = x.repeat(1, self.hidden_sharding_degree, 1, 1).view(
                -1, local_expert_capacity, self.d_model
            )

        # After we do the cross-device permutation we have the tokens on the
        # correct device but not yet grouped by expert because we received
        # tokens from each device as contiguous chunks. To group the tokens
        # for expert computation we'll do one more local permutation.
        # shape (both): (num_local_experts,)
        parallel_indices, parallel_bins = self._get_parallel_indices_and_bins(
            expert_capacity=expert_capacity,
            local_expert_capacity=local_expert_capacity,
            device=x.device,
        )

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        # shape: (num_local_experts * ep_world_size, local_expert_capacity, d_model)
        #     ~= (num_local_experts, expert_capacity, d_model)
        parallel_x, handle = ops.all_to_all(x, group=self.ep_pg, async_op=True)

        return PermutedAllToAllOutput(
            parallel_x,
            parallel_indices,
            None,
            parallel_bins,
            batch_size_per_expert,
            None,
            None,
            expert_capacity,
            handle,
        )

    def compute_local_experts(
        self,
        parallel_x,
        *,
        parallel_indices: torch.Tensor,
        parallel_bin_ids: Optional[torch.Tensor],
        parallel_bins: torch.Tensor,
        parallel_batch_size_per_expert: torch.Tensor,
        expert_capacity: int,
    ) -> torch.Tensor:
        assert parallel_bin_ids is None
        del parallel_batch_size_per_expert

        # Locally permute the tokens and perform the expert computation.
        parallel_x = self.permute_and_compute(
            parallel_x,
            indices=parallel_indices,
            expert_weights=None,
            bins=parallel_bins,
            expert_capacity=expert_capacity,
            top_k=1,
        )

        return parallel_x

    def reverse_all_to_all(
        self,
        parallel_x: torch.Tensor,
        *,
        send_counts: Optional[List[int]],
        recv_counts: Optional[List[int]],
    ) -> Tuple[torch.Tensor, dist.Work]:
        assert send_counts is None
        assert recv_counts is None

        # Un-permute the tokens across the devices.
        x, handle = ops.all_to_all(parallel_x, group=self.ep_pg, async_op=True)
        return x, handle

    def unpermute(
        self,
        x,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        del expert_indices, bin_ids

        # Reduce along the hidden sharding to get the final outputs.
        if self.hidden_sharding_degree > 1:
            x = ops.sum_tensor(x.view(self.hidden_sharding_degree, -1, self.d_model), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.binned_scatter(
            x.view(self.num_experts, -1, self.d_model), indices, expert_weights, bins, self.top_k
        )

        return x

    def permute_and_compute(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        expert_weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        expert_capacity: int,
        top_k: int,
    ) -> torch.Tensor:
        x = x.view(-1, x.shape[-1])

        # Route the tokens for MoE computation.
        # shape: (num_experts, expert_capacity, d_model)
        x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation.
        # shape: (num_experts, expert_capacity, d_model)
        x = self.mlp(x)

        # Un-route the data for the MoE output. Items that were dropped will be zeroed out.
        # shape: (N, d_model)
        x = ops.binned_scatter(x, indices, expert_weights, bins, top_k)
        return x


class ParallelDroplessMLP(ParallelMLPBase):
    """
    A dropless implementation of a :class:`ParallelMLP`.

    .. warning::
        When expert parallelism is enabled the forward pass involves a host-device sync.
    """

    def __init__(self, *, mlp: DroplessMoEMLP, top_k: int, cache: Optional[BufferCache] = None):
        super().__init__(mlp=mlp, top_k=top_k, cache=cache)

    def forward_once(
        self,
        x: torch.Tensor,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        del expert_indices
        return self.permute_and_compute(
            x,
            batch_size_per_expert=batch_size_per_expert,
            indices=indices,
            bin_ids=bin_ids,
            expert_weights=expert_weights,
            bins=bins,
            top_k=self.top_k,
        )

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def permute_and_all_to_all(
        self,
        x: torch.Tensor,
        *,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
    ) -> PermutedAllToAllOutput:
        with torch.no_grad():
            # If we're sharding the experts along the hidden dimension
            # multiple devices own parts of the same sets of experts.
            # Replicate the token counts so every device gets the counts.
            repeated_batch_size_per_expert = ops.repeat(
                batch_size_per_expert,
                (self.hidden_sharding_degree,),
            )

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_batch_size_per_expert = torch.empty_like(
                repeated_batch_size_per_expert,
            )
            tpe_handle = dist.all_to_all_single(
                parallel_batch_size_per_expert,
                repeated_batch_size_per_expert,
                group=self.ep_pg,
                async_op=True,
            )
            assert tpe_handle is not None

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        x = ops.gather(x.view(-1, x.shape[-1]), indices, bin_ids, bins, self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()

            # Reshape to (ep_world_size, num_local_experts).
            repeated_batch_size_per_expert = repeated_batch_size_per_expert.view(
                self.ep_world_size, self.num_local_experts
            )
            parallel_batch_size_per_expert = parallel_batch_size_per_expert.view(
                self.ep_world_size, self.num_local_experts
            )

            # NOTE: host-device sync here.
            send_counts = repeated_batch_size_per_expert.sum(dim=-1).cpu().tolist()
            recv_counts = parallel_batch_size_per_expert.sum(dim=-1).cpu().tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        x = ops.repeat(x, (self.hidden_sharding_degree, 1))

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * self.hidden_sharding_degree,
                    dtype=torch.int32,
                    device=indices.device,
                ),
                self.num_local_experts,
            )

            parallel_top_expert = torch.repeat_interleave(
                parallel_top_expert,
                parallel_batch_size_per_expert.flatten(),
                output_size=tokens_received,
            )

            parallel_bin_ids, parallel_indices = torch.sort(parallel_top_expert)

            # Calculate the bins boundaries from the token counts.
            parallel_batch_size_per_expert = parallel_batch_size_per_expert.sum(
                dim=0,
                dtype=torch.long,
            )
            parallel_bins = torch.empty_like(parallel_batch_size_per_expert, dtype=torch.int32)
            torch.cumsum(parallel_batch_size_per_expert, 0, out=parallel_bins)

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = ops.all_to_all(
            x,
            recv_counts,
            send_counts,
            group=self.ep_pg,
            async_op=True,
        )

        return PermutedAllToAllOutput(
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            -1,
            parallel_x_handle,
        )

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def compute_local_experts(
        self,
        parallel_x,
        *,
        parallel_indices: torch.Tensor,
        parallel_bin_ids: Optional[torch.Tensor],
        parallel_bins: torch.Tensor,
        parallel_batch_size_per_expert: torch.Tensor,
        expert_capacity: int,
    ) -> torch.Tensor:
        assert parallel_bin_ids is not None
        del expert_capacity

        parallel_x = self.permute_and_compute(
            parallel_x,
            batch_size_per_expert=parallel_batch_size_per_expert,
            indices=parallel_indices.int(),
            bin_ids=parallel_bin_ids,
            expert_weights=None,
            bins=parallel_bins,
            top_k=1,
        )

        return parallel_x

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def reverse_all_to_all(
        self,
        parallel_x: torch.Tensor,
        *,
        send_counts: Optional[List[int]],
        recv_counts: Optional[List[int]],
    ) -> Tuple[torch.Tensor, dist.Work]:
        assert send_counts is not None
        assert recv_counts is not None

        # Un-permute the tokens across the devices.
        x, handle = ops.all_to_all(
            parallel_x,
            send_counts,
            recv_counts,
            group=self.ep_pg,
            async_op=True,
        )
        return x, handle

    @torch._dynamo.disable()  # TODO: might be able to relax this, or be more fine-grained
    def unpermute(
        self,
        x,
        *,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
    ):
        del expert_indices

        # Reduce along the hidden sharding to get the final outputs.
        x = ops.sum_tensor(x.view(self.hidden_sharding_degree, -1, self.d_model), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k)

        return x

    def permute_and_compute(
        self,
        x: torch.Tensor,
        *,
        batch_size_per_expert: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        expert_weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        x = x.view(-1, x.shape[-1])

        # Route the tokens for MoE computation.
        x = ops.gather(x, indices, bin_ids, bins, top_k)

        # Perform the expert computation.
        x = self.mlp(x, batch_size_per_expert)

        # Un-route the data for the MoE output.
        return ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)
