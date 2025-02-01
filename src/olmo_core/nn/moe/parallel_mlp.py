# Adapted from 'https://github.com/databricks/megablocks/blob/main/megablocks/layers/moe.py' and 'dmoe.py'

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh

from ...distributed.utils import get_world_size
from . import ops
from .mlp import MoEMLP

__all__ = ["ParallelMLP", "ParallelDroplessMLP"]


class ParallelMLP(nn.Module):
    """
    Wraps an MoE MLP layer to coordinate the routing and expert parallelism.
    """

    def __init__(self, *, mlp: MoEMLP):
        super().__init__()
        self.mlp = mlp
        self._expert_parallel_enabled: bool = False
        self._ep_mesh: Optional[DeviceMesh] = None
        self._ep_pg: Optional[dist.ProcessGroup] = None

    @property
    def d_model(self) -> int:
        return self.mlp.d_model

    @property
    def num_experts(self) -> int:
        return self.mlp.num_experts

    @property
    def experts_per_rank(self) -> int:
        return self.mlp.experts_per_rank

    @property
    def hidden_sharding_degree(self) -> int:
        return self.mlp.hidden_sharding_degree

    @property
    def ep_world_size(self) -> int:
        if self._ep_pg is not None:
            return get_world_size(self._ep_pg)
        else:
            return 1

    def apply_ep(self, ep_mesh: DeviceMesh):
        """
        Apply expert parallelism.
        """
        self.mlp.apply_ep(ep_mesh)
        self._expert_parallel_enabled = True
        self._ep_mesh = ep_mesh
        self._ep_pg = ep_mesh.get_group()

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: The input of shape ``(*, d_model)``.
        :param expert_weights: Expert weights of shape ``(N, top_k)``.
        :param expert_indices: The indices of the top-k experts, shape ``(N, top_k)``.

        :returns: The output with the same shape as ``x`` and a tensor with shape ``(num_experts,)``
            containing the number of items/tokens routed to each expert.
        """
        del x, expert_weights, expert_indices
        raise NotImplementedError


class ParallelDroplessMLP(ParallelMLP):
    """
    A dropless implementation of a :class:`ParallelMLP`.
    """

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_shape = x.size()

        # Compute the experts.
        if self._expert_parallel_enabled:
            x, batch_size_per_expert = self.parallel_forward_once(x, expert_weights, expert_indices)
        else:
            x, batch_size_per_expert = self.forward_once(x, expert_weights, expert_indices)

        return x.view(in_shape), batch_size_per_expert

    def forward_once(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: The input of shape ``(*, d_model)``.
        :param expert_weights: Expert weights of shape ``(N, top_k)``, where ``N``
            typically equals ``batch_size x seq_len``.
        :param expert_indices: The indices of the top-k experts, shape ``(N, top_k)``.
        """
        top_k = expert_weights.shape[-1]

        # shape: (N * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (N * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(expert_indices)

        out = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
        )

        return out, tokens_per_expert

    def parallel_forward_once(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.

        top_k = expert_weights.shape[-1]

        # shape: (N * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (N * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(expert_indices)

            # If we're sharding the experts along the hidden dimension
            # multiple devices own parts of the same sets of experts.
            # Replicate the token counts so every device gets the counts.
            repeated_tokens_per_expert = ops.repeat(
                tokens_per_expert,
                (self.hidden_sharding_degree,),
            )

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(
                repeated_tokens_per_expert,
            )
            tpe_handle = dist.all_to_all_single(
                parallel_tokens_per_expert,
                repeated_tokens_per_expert,
                group=self._ep_pg,
                async_op=True,
            )
            assert tpe_handle is not None

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        x = ops.gather(x.view(-1, x.shape[-1]), indices, bin_ids, bins, top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()

            # Reshape to (ep_world_size, experts_per_rank).
            repeated_tokens_per_expert = repeated_tokens_per_expert.view(
                self.ep_world_size, self.experts_per_rank
            )
            parallel_tokens_per_expert = parallel_tokens_per_expert.view(
                self.ep_world_size, self.experts_per_rank
            )

            # TODO: can we avoid the host-device sync?
            send_counts = repeated_tokens_per_expert.sum(dim=-1).cpu().tolist()
            recv_counts = parallel_tokens_per_expert.sum(dim=-1).cpu().tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        # TODO: Fuse this into the prior, local permutation?
        x = ops.repeat(x, (self.hidden_sharding_degree, 1))

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = ops.all_to_all(
            x,
            recv_counts,
            send_counts,
            group=self._ep_pg,
            async_op=True,
        )

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
                self.experts_per_rank,
            )

            parallel_top_expert = torch.repeat_interleave(
                parallel_top_expert,
                parallel_tokens_per_expert.flatten(),
                output_size=tokens_received,
            )
            #  replicate_bins = torch.cumsum(parallel_tokens_per_expert.flatten(), 0)
            #  parallel_top_expert = ops.replicate(
            #      parallel_top_expert.unsqueeze(dim=0),
            #      replicate_bins,
            #      tokens_received,
            #  ).flatten()

            parallel_bin_ids, parallel_indices = torch.sort(parallel_top_expert)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0,
                dtype=torch.int,
            )
            parallel_bins = torch.cumsum(parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1) if not len(parallel_bins.size()) else parallel_bins
            )

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            1,
        )

        # Un-permute the tokens across the devices.
        x, _ = ops.all_to_all(
            parallel_x,
            send_counts,
            recv_counts,
            group=self._ep_pg,
        )

        # Reduce along the hidden sharding to get the final outputs.
        # TODO: Fuse this into the following local permutation?
        x = ops.sum(x.view(self.hidden_sharding_degree, -1, self.d_model), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)

        return x, tokens_per_expert.flatten()

    def indices_and_bins(
        self, expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param expert_indices: A 1D tensor.
        """
        # shape: (N,)
        expert_indices = expert_indices.int()

        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        # shape: (N,), (N,)
        bin_ids, indices = torch.sort(expert_indices)

        # Histogram the expert ids to identify the number of
        # items/tokens routed to each expert.
        # shape: (num_experts,)
        batch_size_per_expert = torch.histc(
            expert_indices, bins=self.num_experts, min=0, max=self.num_experts - 1
        )

        # Calculate the bin bounds for the sorted items/tokens.
        # shape: (num_experts,)
        bins = torch.empty_like(batch_size_per_expert)
        torch.cumsum(batch_size_per_expert, 0, out=bins)

        return indices.int(), bin_ids, bins, batch_size_per_expert

    def permute_and_compute(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        expert_weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.gather(x, indices, bin_ids, bins, top_k)

        # Perform the expert computation.
        x = self.mlp(x, tokens_per_expert)

        # Un-route the data for the MoE output.
        return ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)
