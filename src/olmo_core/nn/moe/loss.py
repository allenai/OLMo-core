from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_tensor


class MoELoadBalancingLossGranularity(StrEnum):
    """
    Defines the granularity for the router's load balancing loss.
    """

    local_batch = "local_batch"
    """
    The loss is always computed over the rank-local shard of the batch, regardless of the
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
    tp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    loss: torch.Tensor
    if granularity == MoELoadBalancingLossGranularity.instance:
        B = batched_batch_size_per_expert.shape[0]
        # shape: (B, num_experts)
        batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)
        if tp_mesh is not None:
            dist.all_reduce(batched_batch_size_per_expert, group=tp_mesh.get_group())
            batched_batch_size_per_expert = DTensor.from_local(
                batched_batch_size_per_expert, tp_mesh, (Replicate(),)
            )
            # NOTE: assumes equal sequence splits across group
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).mean(dim=1, keepdim=True)
            # shape: (B, 1, num_experts) -> (B, num_experts)
            expert_scores = DTensor.from_local(expert_scores, tp_mesh, (Shard(1),)).mean(dim=1)
        else:
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).mean(dim=1)
        loss = (expert_scores * batched_batch_size_per_expert).sum()
    elif granularity == MoELoadBalancingLossGranularity.local_batch:
        # shape: (num_experts,)
        batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
        # shape: (B, S, num_experts) -> (B * S, num_experts)
        expert_scores = expert_scores.view(-1, num_experts)
        # shape: (B * S, num_experts) -> (num_experts,)
        expert_scores = expert_scores.mean(dim=0)
        loss = torch.dot(batch_size_per_expert, expert_scores)
        if tp_mesh is not None:
            loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()
    else:
        raise NotImplementedError(granularity)

    scale = num_experts / top_k

    return scale * loss


def router_z_loss(
    *, expert_logits: torch.Tensor, tp_mesh: Optional[dist.DeviceMesh] = None
) -> torch.Tensor:
    loss = torch.logsumexp(get_local_tensor(expert_logits), dim=-1).square().sum()
    if tp_mesh is not None:
        loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()
    return loss
