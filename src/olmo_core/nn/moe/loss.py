from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

import olmo_core.distributed.utils as dist_utils
from olmo_core.config import StrEnum


class MoELoadBalancingLossGranularity(StrEnum):
    """
    Defines the granularity for the router's load balancing loss.
    """

    local_batch = "local_batch"
    """
    The loss is always computed over the rank-local shard of the batch, ignoring any
    parallelism strategies used. This is ideal for minimizing the number of dropped tokens for
    any parallel strategy.
    """

    instance = "instance"
    """
    The loss is computed over each instance, taking into account any parallelism strategies used.
    """


def load_balancing_loss(
    *,
    num_experts: int,
    top_k: int,
    expert_scores: torch.Tensor,
    batch_size_per_expert: torch.Tensor,
    batched_batch_size_per_expert: torch.Tensor,
    granularity: MoELoadBalancingLossGranularity,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    tp_mesh: Optional[dist.DeviceMesh] = None,
    cp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    expert_scores, batch_size_per_expert, batched_batch_size_per_expert = (
        dist_utils.get_local_tensor(expert_scores),
        dist_utils.get_local_tensor(batch_size_per_expert),
        dist_utils.get_local_tensor(batched_batch_size_per_expert),
    )

    B, S, _ = expert_scores.shape

    loss: torch.Tensor
    if granularity == MoELoadBalancingLossGranularity.instance:
        # shape: (B, num_experts)
        batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)

        # NOTE: for CP it suffices to reduce the 'batched_batch_size_per_expert' across the CP group
        # and do the rest of the computation locally.
        if cp_mesh is not None:
            dist_utils.all_reduce(batched_batch_size_per_expert, group=cp_mesh.get_group())

        # NOTE: for TP, the end result needs to be a DTensor over the TP mesh, so we handle this case
        # a little differently.
        if tp_mesh is not None:
            # NOTE: assumes sharded on sequence dimension and equal splits across TP group.
            dist_utils.all_reduce(batched_batch_size_per_expert, group=tp_mesh.get_group())
            batched_batch_size_per_expert = DTensor.from_local(
                batched_batch_size_per_expert, tp_mesh, (Replicate(),)
            )
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).mean(dim=1, keepdim=True)
            # shape: (B, 1, num_experts) -> (B, num_experts)
            expert_scores = DTensor.from_local(expert_scores, tp_mesh, (Shard(1),)).mean(dim=1)
        else:
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).mean(dim=1)

        # We compute this across the TP and CP groups, so the 'loss_div_factor' should represent
        # the total number of tokens across the TP and CP groups.
        if loss_div_factor is None:
            # this gives us total number of tokens across TP + CP groups.
            loss_div_factor = batched_batch_size_per_expert.sum() / top_k

        # shape: scalar
        loss = (expert_scores * batched_batch_size_per_expert).sum() / loss_div_factor
    elif granularity == MoELoadBalancingLossGranularity.local_batch:
        # NOTE: We essentially ignore CP for this granularity, and for TP we still compute the loss
        # locally, but wrap as a DTensor and reduce it at the end because the end result has to be
        # a DTensor over the TP mesh.
        # Due to that DTensor reduction, with TP the 'loss_div_factor' should be the total number
        # of tokens across the TP group, but not the CP group.
        if loss_div_factor is None:
            loss_div_factor = B * S
            if tp_mesh is not None:
                loss_div_factor = loss_div_factor * tp_mesh.size()
        elif cp_mesh is not None:
            loss_div_factor = loss_div_factor / cp_mesh.size()

        # shape: (num_experts,)
        batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
        # shape: (B, S, num_experts) -> (B * S, num_experts)
        expert_scores = expert_scores.view(-1, num_experts)
        # shape: (B * S, num_experts) -> (num_experts,)
        expert_scores = expert_scores.mean(dim=0)
        # shape: scalar
        loss = torch.dot(batch_size_per_expert, expert_scores) / loss_div_factor
        if tp_mesh is not None:
            loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()
    else:
        raise NotImplementedError(granularity)

    scale = num_experts / top_k

    return scale * loss


def router_z_loss(
    *,
    expert_logits: torch.Tensor,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    tp_mesh: Optional[dist.DeviceMesh] = None,
    cp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    expert_logits = dist_utils.get_local_tensor(expert_logits)
    B, S, _ = expert_logits.shape

    # NOTE: with TP, end result has to be a DTensor over the TP mesh, so we wrap as a DTensor
    # and reduce it. Due to this reduction, the 'loss_div_factor' should represent the total
    # number of tokens across the TP group (but not the CP group).
    if loss_div_factor is None:
        loss_div_factor = B * S
        if tp_mesh is not None:
            loss_div_factor = loss_div_factor * tp_mesh.size()
    elif cp_mesh is not None:
        loss_div_factor = loss_div_factor / cp_mesh.size()

    loss = torch.logsumexp(expert_logits, dim=-1).square().sum() / loss_div_factor
    if tp_mesh is not None:
        loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()

    return loss
