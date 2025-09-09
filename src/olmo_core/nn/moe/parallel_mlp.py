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
from olmo_core.utils import ensure_multiple_of, get_default_device, move_to_device, mark_dynamic

from ..buffer_cache import BufferCache
from .mlp import DroplessMoEMLP, MoEMLP, MoEMLPBase
import nvtx
from .utils import async_copy_to_cpu
from torch.utils.checkpoint import checkpoint, CheckpointFunction
from typing import Callable

__all__ = ["ParallelMLPBase", "ParallelMLP", "ParallelDroplessMLP"]

from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_sort_chunks_by_index,
    moe_unpermute,
)


# disable compile for permute
@torch.compiler.disable()
def moe_permute_no_compile(*args, **kwargs):
    return moe_permute(*args, **kwargs)
    
@torch.compiler.disable()
def moe_unpermute_no_compile(*args, **kwargs):
    return moe_unpermute(*args, **kwargs)    

@torch.compiler.disable()
def moe_sort_chunks_by_index_no_compile(*args, **kwargs):
    return moe_sort_chunks_by_index(*args, **kwargs)

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

    @nvtx.annotate("ParallelMLPBase.indices_and_bins", color='blue')
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

    @nvtx.annotate("ParallelMLPBase.forward", color='blue')
    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        batch_size_per_expert_cpu: Optional[torch.Tensor] = None,
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
                batch_size_per_expert=batch_size_per_expert_cpu if batch_size_per_expert_cpu is not None else batch_size_per_expert, # when cpu given, use cpu
            )
        else:
            # tp or ep
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


class ParallelMLP(ParallelMLPBase):
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

    @nvtx.annotate("ParallelDroplessMLP.forward", color='blue')
    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        batch_size_per_expert_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # assert False, "This function Not Used"
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
        
        ################
        use_te = False
        if use_te:
            # assert False
            bsz, n_token_per_batch, d_model = in_shape
            routing_map = expert_indices.view(-1, self.top_k).int()
            num_out_tokens = routing_map.size(0) * self.top_k # dropless
            hidden_shape_before_permute = x.shape
            
            

            def permute_and_compute(x, batch_size_per_expert_cpu, routing_map, num_out_tokens):
                permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(inp=x, routing_map=routing_map, num_out_tokens=num_out_tokens, map_type='index')
                # permutated_input_tokens_sorted =  moe_sort_chunks_by_index(
                #     permutated_input_tokens,
                #     split_sizes= batch_size_per_expert,
                #     sorted_index=torch.arange(batch_size_per_expert.size(0), device=batch_size_per_expert.device, dtype=torch.int32),
                # ) # only useful for ep?
                permutated_input_tokens_sorted = permutated_input_tokens
                x = self.mlp(permutated_input_tokens_sorted, batch_size_per_expert_cpu)
                # assert torch.isclose(permutated_input_tokens_sorted, x).all(), "Permuted input tokens do not match the gathered input tokens."
                            # unpermutated_input_tokens_sorted = moe_sort_chunks_by_index(
                    #     x,
                    #     batch_size_per_expert,
                    #     torch.arange(batch_size_per_expert.size(0), device=batch_size_per_expert.device, dtype=torch.int32),
                    # ) # only useful for ep?
                    
                return x, reversed_input_permutation_mapping
                    
            USE_RECOMPUTE=False
            if USE_RECOMPUTE:        
                # recompute the permute and compute
                x, reversed_input_permutation_mapping = checkpoint(
                    permute_and_compute,
                    x,
                    batch_size_per_expert_cpu=batch_size_per_expert_cpu,
                    routing_map=routing_map,
                    num_out_tokens=num_out_tokens,
                    use_reentrant=False,
                )
            else:
                x, reversed_input_permutation_mapping = permute_and_compute(
                    x,
                    batch_size_per_expert_cpu=batch_size_per_expert_cpu,
                    routing_map=routing_map,
                    num_out_tokens=num_out_tokens,
                )        
                    
            unpermutated_input_tokens_sorted = x
            unpermutated_input_tokens = moe_unpermute_no_compile(
                inp=unpermutated_input_tokens_sorted,
                row_id_map=reversed_input_permutation_mapping,
                restore_shape=hidden_shape_before_permute,
                map_type='index',
                merging_probs=expert_weights.view(-1, self.top_k)
            )
            # assert ((unpermutated_input_tokens - x).abs() < 1e-4).all(), "Unpermuted input tokens do not match the scattered output tokens."
            return unpermutated_input_tokens.view(in_shape)
        ################
                    
        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins_bounds = self.indices_and_bins(expert_indices, batch_size_per_expert)

        # Compute the experts.
        if not self._expert_parallel_enabled:

            # Route the tokens for MoE computation.
            x = ops.gather(x, indices, bin_ids, bins_bounds, self.top_k)
            
            # Perform the expert computation.
            x = self.mlp(x, batch_size_per_expert.cpu().tolist())
            
            # Un-route the data for the MoE output.
            x = ops.scatter(x, indices, bin_ids, expert_weights, bins_bounds, self.top_k)

        else:
            # tp or ep
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
        #### NOTE: after this point, the tokens have duplications (duplicated by TOP_K times)
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


    # @torch.compile
    @nvtx.annotate("ParallelDroplessMLP.global_permute_mlp_unpermute", color='blue')
    def global_permute_mlp_unpermute(
        self,
        local_x: torch.Tensor,
        local_x_global_expert_weights: torch.Tensor,
        local_x_global_expert_indices: torch.Tensor,
        local_batch_size_per_global_expert: torch.Tensor,
        overlap_callback: Optional[Callable] = None,
        overlap_callback_x=None,
        **overlap_callback_kwargs,
    ):
        assert self.hidden_sharding_degree == 1, "Global permutation is only supported when hidden sharding degree is 1."
        # mark_dynamic(local_batch_size_per_global_expert, (0,), strict=False)
        
        '''
        The global_permute_mlp_unpermute function performs the following steps:
        1. **Communicates the number of tokens that will be sent to each device**:
        2. **Permutes local tokens to be ready for all-to-all communication**:
        3. **Configures the sizes for grouped GEMM**:
        4. **Starts the all-to-all communication asynchronously**:
        5. **Permutes the global tokens to be ready for MLP computation**:
        6. **MLP forward**:
        7. **Unpermutates the tokens for reverse all-to-all communication**:
        8. **Reverse all-to-all communication**:
        9. **Unpermutates the tokens to restore the original order**:
        10. **Returns the unpermutated tokens**.
        '''
        
        
        ########### 1. Communicate the number of tokens that will be sent to each device ###########
        with nvtx.annotate("Token count all_to_all", color='green'):
            with torch.no_grad():
                # Pass token count information to the device on which the
                # target expert resides.
                global_batch_size_per_local_expert = torch.empty_like(
                    local_batch_size_per_global_expert,
                )
                global_batch_size_handle = dist.all_to_all_single(
                    global_batch_size_per_local_expert, # Gathered concatenated output tensor.
                    local_batch_size_per_global_expert, # Input tensor to scatter.
                    group=self.ep_pg,
                    async_op=True,
                )

        ############################################ end



        ###########  3. Configure the sizes for grouped GEMM ###########

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with nvtx.annotate("Sync token count", color='green'):
            with torch.no_grad():
                global_batch_size_handle.wait()

                # Reshape to (ep_world_size, num_local_experts).
                local_batch_size_per_global_expert = local_batch_size_per_global_expert.view(
                    self.ep_world_size, self.num_local_experts
                )
                global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
                    self.ep_world_size, self.num_local_experts
                )
                # Calculate the bins boundaries from the token counts. # [EP, num_local_experts] -> [num_local_experts,]
                parallel_batch_size_per_expert = global_batch_size_per_local_expert.sum(
                    dim=0,
                    dtype=torch.long,
                )
                
                # NOTE: host-device sync here.
                
                # send_counts, copy_stream, dtoh_event1 = async_copy_to_cpu(local_batch_size_per_global_expert.sum(dim=-1))
                # recv_counts, copy_stream, dtoh_event2 = async_copy_to_cpu(global_batch_size_per_local_expert.sum(dim=-1))
                
                # option 1
                # dtoh_event1.synchronize() # wait for the copy to CPU to finish
                # dtoh_event2.synchronize()
                
                # option 2
                # copy_stream.synchronize() # wait for the copy to CPU to finish

                
                # option 3 
                # NOTE: this is not going to work because only current stream can wait for events, but the all_to_all communication is done in a different stream.
                # torch.cuda.current_stream().wait_event(dtoh_event1) # wait for the copy to CPU to finish
                # torch.cuda.current_stream().wait_event(dtoh_event2)
                
                # option 4
                send_counts = local_batch_size_per_global_expert.sum(dim=-1).to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU
                recv_counts = global_batch_size_per_local_expert.sum(dim=-1).to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU
                
                parallel_batch_size_per_expert_cpu = parallel_batch_size_per_expert.to(torch.device("cpu"), non_blocking=True) # WARNING: tensor to CPU

            
            # re-indent to enable grad
            
            # put the dense branch here to overlap DtoH sync
            # TODO: one potential issue is that:
            # GPU side: permute in step 2 is finished
            # CPU side: the dense branch (attention + MLP) is still being submitted
            # result: GPU idle before the dense branch actually starts to run
            # potential fix: 
            # 1. find some else to run before overlap_callback
            # 2. make permute in step 2 run longer (increase batch size)
            # 3. break the dense branch into smaller pieces and blend with other operations
            if overlap_callback is not None:
                overlap_out = overlap_callback(
                    overlap_callback_x, 
                    **overlap_callback_kwargs,
                )
            else:
                overlap_out = None
                
            with torch.no_grad():
                torch.cuda.current_stream().synchronize() # wait for the copy to CPU to finish
                
                send_counts = send_counts.tolist() # tensor to list
                recv_counts = recv_counts.tolist() # tensor to list
                
                tokens_received = sum(recv_counts)

                # Construct the expert indices for the permuted tokens.
                global_x_local_expert_indices = torch.remainder(
                    torch.arange(
                        self.num_experts * self.hidden_sharding_degree,
                        dtype=torch.int32,
                        device=local_x.device,
                    ),
                    self.num_local_experts,
                ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts

                # this specifiyes for the received global tokens, which local expert they belong to
                global_x_local_expert_indices = torch.repeat_interleave(
                    global_x_local_expert_indices,
                    global_batch_size_per_local_expert.flatten(),
                    output_size=tokens_received,
                ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts
        


        
        EP_PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE = True
        if EP_PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE:
            local_x = checkpoint(
                self.forward_step_1_9,
                local_x,
                local_x_global_expert_weights,
                global_x_local_expert_indices,
                parallel_batch_size_per_expert_cpu,
                local_x_global_expert_indices,
                recv_counts,
                send_counts,
                use_reentrant=False,
            )
        else:
            local_x = self.forward_step_1_9(
                local_x,
                local_x_global_expert_weights,
                global_x_local_expert_indices,
                parallel_batch_size_per_expert_cpu,
                local_x_global_expert_indices,
                recv_counts,
                send_counts,
            )


        return local_x, overlap_out


    # @torch.compile
    def forward_step_1_9(
        self,
        local_x: torch.Tensor,
        local_x_global_expert_weights: torch.Tensor,
        global_x_local_expert_indices: torch.Tensor,
        parallel_batch_size_per_expert_cpu: torch.Tensor,
        local_x_global_expert_indices: torch.Tensor,
        recv_counts: List[int],
        send_counts: List[int]
    ):
        
        ########### 2. permute local tokens to be ready for all-to-all communication ###########
        with nvtx.annotate("Permute local tokens", color='green'):
            routing_map = local_x_global_expert_indices.view(-1, self.top_k).int()
            num_out_tokens = routing_map.size(0) * self.top_k # dropless
            hidden_shape_before_permute = local_x.shape
            permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
                inp=local_x, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            ) # type: ignore
            
            # now permutated_local_x tokens are grouped by expert, which means tokens will go to expert id:
            # [0 , 0 , ... 1, ... 2, ..., ..., 31, 31]  (if 32 experts)
            # if EP=8, each rank has 4 experts, then tokens of
            # [0, 0, ..., 3, 3] go to rank 0,
            # [4, 4, ..., 7, 7] go to rank 1, 
            # and so on.
        ############################################ end
        
        ###########  4. Start the all-to-all communication asynchronously ###########

        with nvtx.annotate("all2all", color='green'):
            global_x, global_x_handle = ops.all_to_all(
                permutated_local_x,
                recv_counts,
                send_counts,
                group=self.ep_pg,
                async_op=True,
            )
            
            global_x_handle.wait()
            del permutated_local_x
        ############################################ end
    
        ###########  5. Permute the global tokens to be ready for MLP computation ###########
        with nvtx.annotate("Permute global tokens for MLP", color='green'):
            # option 1: use moe_sort_chunks_by_index (by TE <- trition)
            # input_chunk_idxs = torch.arange(
            #     self.num_experts, device=local_x.device
            # )
            # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
            # sort_input_by_local_experts = input_chunk_idxs.reshape(
            #     -1, self.num_local_experts
            # ).T.ravel() 
            # split into 32 chunks (32 experts)
            # e.g., [ 
            # 0,  4,  8, 12, 16, 20, 24, 28,    --> these 8 chunks come from all 8 EP ranks, go to local expert 0
            # 1,  5,  9, 13, 17, 21, 25, 29,    --> these 8 chunks come from all 8 EP ranks, go to local expert 1
            # 2,  6, 10, 14, 18, 22, 26, 30,    --> these 8 chunks come from all 8 EP ranks, go to local expert 2
            # 3,  7, 11, 15, 19, 23, 27, 31     --> these 8 chunks come from all 8 EP ranks, go to local expert 3
            # ].  (1D tensor)

            
            ## chunk size is specified by `global_batch_size_per_local_expert`
            
            
            # e.g., global_batch_size_per_local_expert
            # local experts 0     1     2     3
            # ep0       [[3108, 5307, 5798, 4067],
            # ep1        [4642, 3836, 3488, 3477],
            # ep2        [5129, 3964, 2472, 4194],
            # ep3        [4266, 3191, 4511, 3841],
            # ep4        [5059, 5758, 4838, 3201],
            # ep5        [5388, 3531, 3419, 2860],
            # ep6        [3862, 3605, 2945, 3840],
            # ep7        [3960, 4624, 3414, 4406]]
            
            # so we want to put (3108+4642+5129+4266+5059+5388+3862+3960) tokens to local expert 0,
            # and so on
            
            # global_x = moe_sort_chunks_by_index_no_compile(
            #     inp=global_x,
            #     split_sizes=global_batch_size_per_local_expert.ravel(),
            #     sorted_index=sort_input_by_local_experts
            # ) # type: ignore

            # option 2: use moe_permute (by TE), and pretend topk is 1
            routing_map2 = global_x_local_expert_indices.view(-1, 1).int()
            num_out_tokens2 = routing_map2.size(0) * 1 # dropless
            hidden_shape_before_permute2 = global_x.shape
            global_x, reversed_global_x_permutation_mapping = moe_permute_no_compile(
                inp=global_x, 
                routing_map=routing_map2, 
                num_out_tokens=num_out_tokens2, 
                map_type='index'
            )    # type: ignore
                
                
        ############################################ end

        
        ########## 6. MLP forwrad ###########

        global_x = self.mlp(global_x, parallel_batch_size_per_expert_cpu.tolist())

        ############################################ end
        
        
        ############ 7. Unpermute the output tokens to be ready for all-to-all communication ##########
        with nvtx.annotate("Unpermute global tokens", color='green'):
            # option 1: use moe_sort_chunks_by_index (by TE <- trition)
            # restore_output_by_local_experts = input_chunk_idxs.reshape(
            #     self.num_local_experts, -1
            # ).T.ravel() # [ 0,  8, 16, 24,  1,  9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31]
            # global_x = moe_sort_chunks_by_index_no_compile(
            #     global_x, 
            #     split_sizes=global_batch_size_per_local_expert.T.ravel(),
            #     sorted_index=restore_output_by_local_experts
            # )
            
            # option 2: use moe_unpermute (by TE)
            global_x = moe_unpermute_no_compile(
                inp=global_x,
                row_id_map=reversed_global_x_permutation_mapping,
                merging_probs=None,
                restore_shape=hidden_shape_before_permute2,
                map_type='index',
            ) # type: ignore
        ############################################ end
    
            
    
        ########## 8. reverse_all_to_all ###########
        with nvtx.annotate("reverse_all_to_all", color='green'):
            local_x, local_x_handle = ops.all_to_all(
                global_x,
                send_counts,
                recv_counts,
                group=self.ep_pg,
                async_op=True,
            )
            
            local_x_handle.wait()
            
            del global_x # done with global tokens
        ############################################ end
        
        
        ############ 9. Unpermute the (local) tokens returned by all-to-all communication ##########
        with nvtx.annotate("Unpermute-Merge local tokens", color='green'):
            local_x = moe_unpermute_no_compile(
                inp=local_x,
                row_id_map=reversed_local_x_permutation_mapping,
                merging_probs=local_x_global_expert_weights.view(-1, self.top_k),
                restore_shape=hidden_shape_before_permute,
                map_type='index',
            ) # type: ignore
        ############################################ end
    
        return local_x


    # @torch.compile
    @nvtx.annotate("ParallelDroplessMLP.global_permute_mlp_unpermute_no_ep", color='blue')
    def global_permute_mlp_unpermute_no_ep(
        self,
        local_x: torch.Tensor,
        local_x_global_expert_weights: torch.Tensor,
        local_x_global_expert_indices: torch.Tensor,
        local_batch_size_per_global_expert: torch.Tensor,
        overlap_callback: Optional[Callable] = None,
        overlap_callback_x=None,
        **overlap_callback_kwargs,
    ):
        x, expert_weights, expert_indices, batch_size_per_expert = (
            get_local_tensor(local_x),
            get_local_tensor(local_x_global_expert_weights),
            get_local_tensor(local_x_global_expert_indices),
            get_local_tensor(local_batch_size_per_global_expert),
        )

        in_shape = x.size()

        # shape: (N, d_model)
        x = x.view(-1, x.shape[-1])

        # step 1A: DtoH token count communication
        # mark_dynamic(batch_size_per_expert, (0,), strict=False)
        batch_size_per_expert_cpu, copy_stream, dtoh_event1 = async_copy_to_cpu(batch_size_per_expert) 
        
        # overlap compute while waiting for the copy to CPU to finish
        with nvtx.annotate("overlap_callback", color='green'):
            if overlap_callback is not None:
                overlap_out = overlap_callback(overlap_callback_x, **overlap_callback_kwargs)
            else:
                overlap_out = None
                
        # step 1B: permute the input tokens
        copy_stream.synchronize() # wait for the copy to CPU to finish
        

        PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE = True
        if PERMUTE_GEMM_UNPERMUTE_USE_RECOMPUTE:
            x_moe = checkpoint(
                self._forward_step_rc,
                x,
                expert_indices,
                expert_weights,
                batch_size_per_expert_cpu,
                use_reentrant=False,
                in_shape=in_shape,
            )
        else:
            x_moe = self._forward_step_rc(
                x,
                expert_indices,
                expert_weights,
                batch_size_per_expert_cpu,
                in_shape=in_shape,
            )

        return x_moe, overlap_out

    # @torch.compile
    def _forward_step_rc(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        batch_size_per_expert_cpu: torch.Tensor,
        in_shape: torch.Size
    ):
        routing_map = expert_indices.view(-1, self.top_k).int()
        num_out_tokens = routing_map.size(0) * self.top_k # dropless
        hidden_shape_before_permute = x.shape

        # step 2: permute the input tokens
        with nvtx.annotate("Permute", color='green'):
            permutated_input_tokens, reversed_input_permutation_mapping = moe_permute_no_compile(
                inp=x, 
                routing_map=routing_map, 
                num_out_tokens=num_out_tokens, 
                map_type='index'
            ) # type: ignore

        # step 3: MLP
        x = self.mlp(permutated_input_tokens, batch_size_per_expert_cpu.tolist())

        # step 4: unpermutate the output tokens
        with nvtx.annotate("Unpermute", color='green'):
            unpermutated_x = moe_unpermute_no_compile(
                inp=x,
                row_id_map=reversed_input_permutation_mapping,
                restore_shape=hidden_shape_before_permute,
                map_type='index',
                merging_probs=expert_weights.view(-1, self.top_k)
            ) # type: ignore
            
        return unpermutated_x.view(in_shape)
    # OLD COPY (with comparison)

    # def global_permute_mlp_unpermute(
    #     self,
    #     local_x: torch.Tensor,
    #     local_x_global_expert_weights: torch.Tensor,
    #     local_x_global_expert_indices: torch.Tensor,
    #     local_batch_size_per_global_expert: torch.Tensor,
    # ):
    #     assert self.hidden_sharding_degree == 1, "Global permutation is only supported when hidden sharding degree is 1."
        
    #     # Communicate the number of tokens that will be sent to each device.
    #     with torch.no_grad():

    #         # Pass token count information to the device on which the
    #         # target expert resides.
    #         global_batch_size_per_local_expert = torch.empty_like(
    #             local_batch_size_per_global_expert,
    #         )
    #         tpe_handle = dist.all_to_all_single(
    #             global_batch_size_per_local_expert, # Gathered concatenated output tensor.
    #             local_batch_size_per_global_expert, # Input tensor to scatter.
    #             group=self.ep_pg,
    #             async_op=True,
    #         )
    #         assert tpe_handle is not None

    #     ########### OLD: end ######################
    #     with torch.no_grad():
    #         indices, bin_ids, bins_bounds = self.indices_and_bins(local_x_global_expert_indices, local_batch_size_per_global_expert)
        
    #     # Permute locally and without any padding so that tokens for each
    #     # parallel device are stored contiguously.
    #     #### NOTE: after this point, the tokens have duplications (duplicated by TOP_K times)
    #     local_x_old = ops.gather(local_x.view(-1, local_x.shape[-1]), indices, bin_ids, bins_bounds, self.top_k)
    #     ########### OLD: end ######################
        
        
        
    #     ########### NEW: ######################
    #     routing_map = local_x_global_expert_indices.view(-1, self.top_k).int()
    #     num_out_tokens = routing_map.size(0) * self.top_k # dropless
    #     hidden_shape_before_permute = local_x.shape
    #     permutated_local_x, reversed_local_x_permutation_mapping = moe_permute_no_compile(
    #         inp=local_x, 
    #         routing_map=routing_map, 
    #         num_out_tokens=num_out_tokens, 
    #         map_type='index'
    #     ) # type: ignore
        
    #     # now permutated_local_x tokens are grouped by expert, which means tokens will go to expert id:
    #     # [0 , 0 , ... 1, ... 2, ..., ..., 31, 31]  (if 32 experts)
    #     # if EP=8, each rank has 4 experts, then tokens of
    #     # [0, 0, ..., 3, 3] go to rank 0,
    #     # [4, 4, ..., 7, 7] go to rank 1, 
    #     # and so on.
        
    #             # permutated_input_tokens_sorted =  moe_sort_chunks_by_index(
    #             #     permutated_input_tokens,
    #             #     split_sizes= batch_size_per_expert,
    #             #     sorted_index=torch.arange(batch_size_per_expert.size(0), device=batch_size_per_expert.device, dtype=torch.int32),
    #             # ) # only useful for ep?
    #     permutated_input_tokens_sorted = permutated_local_x
        
    #     ########## NEW: end ######################


    #     # Compute the number of tokens that will be received from each
    #     # device and permute the input data across the devices.
    #     with torch.no_grad():
    #         tpe_handle.wait()

    #         # Reshape to (ep_world_size, num_local_experts).
    #         local_batch_size_per_global_expert = local_batch_size_per_global_expert.view(
    #             self.ep_world_size, self.num_local_experts
    #         )
    #         global_batch_size_per_local_expert = global_batch_size_per_local_expert.view(
    #             self.ep_world_size, self.num_local_experts
    #         )

    #         # NOTE: host-device sync here.
    #         send_counts = local_batch_size_per_global_expert.sum(dim=-1).cpu().tolist()
    #         recv_counts = global_batch_size_per_local_expert.sum(dim=-1).cpu().tolist()
    #         tokens_received = sum(recv_counts)


    #     with torch.no_grad():
    #         # After we do the cross-device permutation we have the tokens on the
    #         # correct device but not yet grouped by expert because we received
    #         # tokens from each device as contiguous chunks. To group the tokens
    #         # for expert computation we'll do one more local permutation. The
    #         # rest of this torch.no_grad() scope sets up the indices and bins
    #         # for this permutation.

    #         # Construct the expert indices for the permuted tokens.
    #         global_x_local_expert_indices = torch.remainder(
    #             torch.arange(
    #                 self.num_experts * self.hidden_sharding_degree,
    #                 dtype=torch.int32,
    #                 device=indices.device,
    #             ),
    #             self.num_local_experts,
    #         ) # e.g. [0, 1, 2, 3, 0, 1, 2, 3, ...] for 4 local experts

    #         # this specifiyes for the received global tokens, which local expert they belong to
    #         global_x_local_expert_indices = torch.repeat_interleave(
    #             global_x_local_expert_indices,
    #             global_batch_size_per_local_expert.flatten(),
    #             output_size=tokens_received,
    #         ) # e.g. [0, ...,  0, ... , 3, ..., 3, 0, ...] for 4 local experts

    #         parallel_bin_ids, parallel_indices = torch.sort(global_x_local_expert_indices)

    #         # Calculate the bins boundaries from the token counts. # [EP, num_local_experts] -> [num_local_experts,]
    #         parallel_batch_size_per_expert = global_batch_size_per_local_expert.sum(
    #             dim=0,
    #             dtype=torch.long,
    #         )
    #         parallel_bins = torch.empty_like(parallel_batch_size_per_expert, dtype=torch.int32)
    #         torch.cumsum(parallel_batch_size_per_expert, 0, out=parallel_bins)

    #     # Start the cross-device permutation asynchronously so we can
    #     # overlap communication with computation.
    #     parallel_x, parallel_x_handle = ops.all_to_all(
    #         permutated_local_x,
    #         recv_counts,
    #         send_counts,
    #         group=self.ep_pg,
    #         async_op=True,
    #     )
        
    #     parallel_x_handle.wait()
    #     ############
    #     input_chunk_idxs = torch.arange(
    #         self.num_experts, device=local_x.device
    #     )
    #     # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
    #     sort_input_by_local_experts = input_chunk_idxs.reshape(
    #         -1, self.num_local_experts
    #     ).T.ravel() 
    #     # split into 32 chunks (32 experts)
    #     # e.g., [ 
    #     # 0,  4,  8, 12, 16, 20, 24, 28,    --> these 8 chunks come from all 8 EP ranks, go to local expert 0
    #     # 1,  5,  9, 13, 17, 21, 25, 29,    --> these 8 chunks come from all 8 EP ranks, go to local expert 1
    #     # 2,  6, 10, 14, 18, 22, 26, 30,    --> these 8 chunks come from all 8 EP ranks, go to local expert 2
    #     # 3,  7, 11, 15, 19, 23, 27, 31     --> these 8 chunks come from all 8 EP ranks, go to local expert 3
    #     # ].  (1D tensor)

        
    #     ## chunk size is specified by `global_batch_size_per_local_expert`
        
        
    #     # e.g., global_batch_size_per_local_expert
    #     # local experts 0     1     2     3
    #     # ep0       [[3108, 5307, 5798, 4067],
    #     # ep1        [4642, 3836, 3488, 3477],
    #     # ep2        [5129, 3964, 2472, 4194],
    #     # ep3        [4266, 3191, 4511, 3841],
    #     # ep4        [5059, 5758, 4838, 3201],
    #     # ep5        [5388, 3531, 3419, 2860],
    #     # ep6        [3862, 3605, 2945, 3840],
    #     # ep7        [3960, 4624, 3414, 4406]]
        
    #     # so we want to put (3108+4642+5129+4266+5059+5388+3862+3960) tokens to local expert 0,
    #     # and so on
    #     tmp = moe_sort_chunks_by_index(
    #         inp=parallel_x,
    #         split_sizes=global_batch_size_per_local_expert.ravel(),
    #         sorted_index=sort_input_by_local_experts
    #     )
        
    #     ###########
        

    #     alltoall_tmp = PermutedAllToAllOutput(
    #         parallel_x,
    #         parallel_indices,
    #         parallel_bin_ids,
    #         parallel_bins,
    #         parallel_batch_size_per_expert,
    #         recv_counts,
    #         send_counts,
    #         -1,
    #         parallel_x_handle,
    #     )
        
    #     assert parallel_bin_ids is not None


    #     # parallel_x = self.permute_and_compute(
    #     #     parallel_x,
    #     #     batch_size_per_expert=parallel_batch_size_per_expert,
    #     #     indices=parallel_indices.int(),
    #     #     bin_ids=parallel_bin_ids,
    #     #     expert_weights=None,
    #     #     bins=parallel_bins,
    #     #     top_k=1,
    #     # )

        
    #     ########## permute_and_compute
        
        
    #     parallel_x = parallel_x.view(-1, parallel_x.shape[-1])

    #     # Route the tokens for MoE computation.
    #     parallel_x = ops.gather(parallel_x, parallel_indices.int(), parallel_bin_ids, parallel_bins, top_k=1)
    #     assert torch.isclose(parallel_x, tmp).all()
    #     # Perform the expert computation.
    #     mlp_out = self.mlp(parallel_x, parallel_batch_size_per_expert)

    #     # Un-route the data for the MoE output.
    #     parallel_x = ops.scatter(mlp_out, parallel_indices.int(), parallel_bin_ids, weights=None, bins=parallel_bins, top_k=1)
        
        
    #     ############
    #     restore_output_by_local_experts = input_chunk_idxs.reshape(
    #         self.num_local_experts, -1
    #     ).T.ravel() # [ 0,  8, 16, 24,  1,  9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31]
        
    #     tmp2 = moe_sort_chunks_by_index(
    #         mlp_out, 
    #         split_sizes=global_batch_size_per_local_expert.T.ravel(),
    #         sorted_index=restore_output_by_local_experts
    #     )
    #     assert torch.isclose(tmp2, parallel_x).all(), "The output of the permute_and_compute does not match the expected output."
    #     ##########
        
    #     ########## reverse_all_to_all
    #     x_moe, x_moe_handle = ops.all_to_all(
    #         parallel_x,
    #         send_counts,
    #         recv_counts,
    #         group=self.ep_pg,
    #         async_op=True,
    #     )
        
    #     x_moe_handle.wait()
        
        
    #     x_moe_1 = self.unpermute(
    #         x_moe.float(),
    #         expert_weights=local_x_global_expert_weights,
    #         expert_indices=local_x_global_expert_indices,
    #         indices=indices,
    #         bin_ids=bin_ids,
    #         bins=bins_bounds,
    #     ) # type: ignore
        
    #     x_moe_te_unpermute = moe_unpermute_no_compile(
    #         inp= x_moe.float(),
    #         row_id_map=reversed_local_x_permutation_mapping,
    #         merging_probs=local_x_global_expert_weights.view(-1, self.top_k),
    #         restore_shape=hidden_shape_before_permute,
    #         map_type='index',
    #     ) # type: ignore
    #     close_rate = torch.isclose(x_moe_1, x_moe_te_unpermute, rtol=0.05, atol=0).sum()/(16384*2048) # 96% are within 5% relative error under bf16, 100% under fp32
    #     return x_moe_1
