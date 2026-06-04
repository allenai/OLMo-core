"""
The EMO MoE router: document-pool ("two-level") routing with per-document expert pools,
shared experts, and an optional data-parallel-global load-balancing loss.

This is the routing scheme used by the EMO / FlexMoE models. The high-level idea is "two-level"
routing: for each *document* in a sequence we first restrict routing to a pool of experts
(pool size sampled per document during training), then do the usual top-k routing *within* that pool. A fixed
number of experts are always-active "shared" experts. See :class:`EmoRouter` for details.

Document boundaries are derived upstream (in :meth:`olmo_core.nn.transformer.Transformer.forward`)
from the EOS token id and passed in as ``document_boundaries``; routers store
:attr:`~MoETwoLevelRouter.eos_token_id` so the model can detect them.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

import olmo_core.ops.moe as ops
from olmo_core.distributed.utils import is_distributed
from olmo_core.exceptions import OLMoConfigurationError

from .loss import load_balancing_loss, router_z_loss
from .router import MoELinearRouter, MoERouterGatingFunction

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = ["MoETwoLevelRouter", "EmoRouter"]


class MoETwoLevelRouter(MoELinearRouter):
    """
    Abstract base class for document-pool ("two-level") MoE routers.

    It extends :class:`~olmo_core.nn.moe.router.MoELinearRouter` with the bits shared by all
    document-aware routers: the EOS token id used (upstream) to derive document boundaries, the
    document expert-pool size, and a couple of document-level routing diagnostics. It does not
    define a usable ``forward`` — instantiate a concrete subclass such as :class:`EmoRouter`.

    :param document_expert_pool: The number of experts each document may route to (the rest are
        masked out for that document).
    :param eos_token_id: The EOS token id marking document boundaries. Stored so the model can
        detect document-aware routers and compute boundaries.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        document_expert_pool: int,
        eos_token_id: int,
        **kwargs,
    ):
        super().__init__(dtype=dtype, init_device=init_device, **kwargs)

        self.document_expert_pool = document_expert_pool
        if eos_token_id is None:
            raise OLMoConfigurationError("eos_token_id must be provided for a two-level MoE router")
        self.eos_token_id = eos_token_id

        # Document-level routing diagnostics, accumulated over a metrics-collect interval and
        # reset by `reset_metrics`. These are pure diagnostics and do not affect training.
        self._unique_experts_sum: float = 0.0
        self._reducedp_unique_experts_sum: float = 0.0
        self._num_batches_tracked: int = 0

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out = super().compute_metrics(reset=False)

        if self._num_batches_tracked > 0:
            avg_unique_experts = self._unique_experts_sum / self._num_batches_tracked
            out["unique experts used per batch"] = (
                torch.tensor(avg_unique_experts, device=self.device),
                ReduceType.mean,
            )
            out["fraction of experts used per batch"] = (
                torch.tensor(avg_unique_experts / self.num_experts, device=self.device),
                ReduceType.mean,
            )
            out["reducedp unique experts used per batch"] = (
                torch.tensor(
                    self._reducedp_unique_experts_sum / self._num_batches_tracked,
                    device=self.device,
                ),
                ReduceType.mean,
            )

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        super().reset_metrics()
        self._unique_experts_sum = 0.0
        self._reducedp_unique_experts_sum = 0.0
        self._num_batches_tracked = 0

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, document_expert_pool={self.document_expert_pool}, "
            f"eos_token_id={self.eos_token_id}"
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MoETwoLevelRouter is an abstract base; use a concrete subclass like EmoRouter."
        )


class EmoRouter(MoETwoLevelRouter):
    """
    The EMO router: two-level document routing with a *random* per-document expert pool and a fixed
    number of always-active shared experts.

    For each document (a contiguous span delimited by EOS tokens, supplied via
    ``document_boundaries``) we:

    1. Compute expert logits (excluding the last ``num_shared_experts`` experts, which are always
       active and handled separately).
    2. Sample a pool size uniformly from
       ``[min_document_expert_pool, max_document_expert_pool]`` during training (or use
       ``eval_document_expert_pool`` at eval time), and mask out all but the top
       ``pool_size`` experts for that document based on the document-summed expert probabilities.
    3. Route each token to its top ``top_k - num_shared_experts`` experts within the surviving pool,
       then append the shared experts (always active, weight ``1.0``).

    The load-balancing loss can optionally be made *data-parallel-global* via
    ``global_load_balancing``: the per-expert token counts (and the loss divisor) are summed across
    DP ranks before computing the loss, giving a batch-level objective over the whole DP group
    (within a single micro-batch). When disabled, the standard rank-local counts are used.

    .. note::
        Tensor and context parallelism are not supported by this router.

    :param min_document_expert_pool: Minimum per-document expert pool size (sampled during training).
    :param max_document_expert_pool: Maximum per-document expert pool size (sampled during training).
    :param eval_document_expert_pool: Fixed pool size used at eval time. Defaults to the midpoint of
        the min/max range.
    :param eos_token_id: The EOS token id marking document boundaries.
    :param num_shared_experts: Number of always-active shared experts (taken as the last experts).
    :param global_load_balancing: If ``True``, reduce the load-balancing statistics across DP ranks.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        min_document_expert_pool: int,
        max_document_expert_pool: int,
        eos_token_id: int,
        num_shared_experts: int = 0,
        eval_document_expert_pool: Optional[int] = None,
        global_load_balancing: bool = True,
        **kwargs,
    ):
        # The "document_expert_pool" of the base is the max possible pool size.
        super().__init__(
            dtype=dtype,
            init_device=init_device,
            document_expert_pool=max_document_expert_pool,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.min_document_expert_pool = min_document_expert_pool
        self.max_document_expert_pool = max_document_expert_pool
        self.eval_document_expert_pool = (
            eval_document_expert_pool
            if eval_document_expert_pool is not None
            else (min_document_expert_pool + max_document_expert_pool) // 2
        )

        if num_shared_experts > self.top_k:
            raise OLMoConfigurationError(
                f"num_shared_experts ({num_shared_experts}) must be <= top_k ({self.top_k})"
            )
        self.num_shared_experts = num_shared_experts
        # Number of experts each token actually chooses (the rest of top_k is shared experts).
        self.num_choose_experts = self.top_k - self.num_shared_experts

        self.global_load_balancing = global_load_balancing

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like the base router's ``get_top_k`` but selects ``num_choose_experts`` instead of
        ``top_k`` (the remaining slots are the always-active shared experts).
        """
        if self.uniform_expert_assignment:
            raise NotImplementedError("uniform_expert_assignment is not supported by EmoRouter.")

        if self.bias_gamma is None:
            if self.num_choose_experts == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.num_choose_experts, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.num_choose_experts, dim=-1  # type: ignore
                )
            expert_weights = scores.gather(-1, expert_indices)

        return expert_weights, expert_indices

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        document_boundaries: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.tp_mesh is not None:
            raise NotImplementedError("Tensor parallelism is not supported by EmoRouter.")
        if self.cp_mesh is not None:
            raise NotImplementedError("Context parallelism is not supported by EmoRouter.")
        if document_boundaries is None:
            raise RuntimeError(
                "EmoRouter requires `document_boundaries`; these are derived from the EOS token id "
                "in Transformer.forward. Make sure the model contains an EmoRouter so boundaries "
                "are computed and plumbed through."
            )

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # Drop the shared experts (the last `num_shared_experts`); they are always active and are
        # re-added at the end. Routing/load-balancing only happens over the non-shared experts.
        num_non_shared_experts = self.num_experts - self.num_shared_experts
        logits = logits[:, :, :num_non_shared_experts]
        logits_mask = torch.zeros_like(logits, dtype=torch.bool)

        # Normalize document boundaries to plain python lists of exclusive ends, ending at seq_len.
        document_boundaries_cpu = []
        for b in document_boundaries:
            bc = b.detach().cpu().tolist()
            if not bc or bc[-1] != x.size(1):
                bc.append(int(x.size(1)))
            document_boundaries_cpu.append(bc)

        # For each document, keep only a (random, per-document) pool of experts.
        for seq_idx in range(x.size(0)):
            start = 0
            for end in document_boundaries_cpu[seq_idx]:
                if end <= start:
                    start = end
                    continue
                # shape: (doc_len, num_non_shared_experts)
                sequence_logits = logits[seq_idx, start:end, :]
                expert_probs = F.softmax(sequence_logits, dim=-1)
                # Document-summed expert probability mass.
                document_expert_probs = expert_probs.sum(dim=0)

                # Sample the pool size for this document.
                if self.training:
                    document_expert_pool = int(
                        torch.randint(
                            self.min_document_expert_pool,
                            self.max_document_expert_pool + 1,
                            (1,),
                        ).item()
                    )
                else:
                    document_expert_pool = self.eval_document_expert_pool

                bot_document_expert_pool = num_non_shared_experts - document_expert_pool
                if bot_document_expert_pool <= 0:
                    # Pool covers all non-shared experts; nothing to mask.
                    start = end
                    continue

                # Discard the lowest-probability experts (outside the pool) for this document.
                experts_to_discard = torch.topk(
                    -document_expert_probs, bot_document_expert_pool
                ).indices
                logits_mask[seq_idx, start:end, experts_to_discard] = True
                start = end

        logits.masked_fill_(logits_mask, float("-inf"))

        # shape: (batch_size, seq_len, num_non_shared_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, num_choose_experts)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(expert_weights, p=self.normalize_expert_weights, dim=-1, keepdim=True)
            )

        with torch.no_grad():
            # Histogram expert ids to count tokens routed to each (non-shared) expert.
            # shape: (batch_size, seq_len, num_non_shared_experts)
            tot_batched_batch_size_per_expert = ops.batched_histc(
                expert_indices, num_non_shared_experts
            )
            # shape: (batch_size, num_non_shared_experts)
            tot_batched_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_non_shared_experts,)
            tot_batch_size_per_expert = tot_batched_batch_size_per_expert.sum(dim=0)

            if self.training:
                # Diagnostic: unique experts used this batch (shared experts always count).
                unique_experts = torch.unique(expert_indices.view(-1))
                self._unique_experts_sum += unique_experts.numel() + self.num_shared_experts
                self._num_batches_tracked += 1

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise the LB loss misbehaves.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    # Optionally make the load-balancing statistics data-parallel-global by summing
                    # the per-expert counts (and loss divisor) across the DP group.
                    lb_batch_size_per_expert = tot_batch_size_per_expert.clone()
                    lb_loss_div_factor = loss_div_factor
                    if self.global_load_balancing and is_distributed():
                        assert isinstance(
                            loss_div_factor, torch.Tensor
                        ), "global_load_balancing requires a tensor loss_div_factor"
                        lb_loss_div_factor = loss_div_factor.clone()
                        dist.all_reduce(lb_batch_size_per_expert, op=dist.ReduceOp.SUM)
                        dist.all_reduce(lb_loss_div_factor, op=dist.ReduceOp.SUM)

                    # Diagnostic: unique experts used across the (reduced) batch.
                    self._reducedp_unique_experts_sum += (
                        lb_batch_size_per_expert > 0
                    ).sum().item() + self.num_shared_experts

                    lb_loss = load_balancing_loss(
                        num_experts=num_non_shared_experts,
                        top_k=self.num_choose_experts,
                        expert_scores=scores,
                        batch_size_per_expert=lb_batch_size_per_expert,
                        batched_batch_size_per_expert=tot_batched_batch_size_per_expert,
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=lb_loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.load_balancing_loss += lb_loss.detach()

                    aux_loss = self.lb_loss_weight * lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            # The persistent per-expert counters only track the non-shared experts (the shared
            # experts are always active and carry no load-balancing signal). The base router sizes
            # this buffer to `num_experts`, so shrink it once to the non-shared size.
            if self.batch_size_per_expert.shape[-1] != tot_batch_size_per_expert.shape[-1]:
                extra = self.batch_size_per_expert[tot_batch_size_per_expert.shape[-1] :]
                assert torch.all(
                    extra == 0
                ), f"expected shared-expert counters to be zero, got {extra}"
                self.batch_size_per_expert = self.batch_size_per_expert[
                    : tot_batch_size_per_expert.shape[-1]
                ]
            self.batch_size_per_expert += tot_batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += tot_batch_size_per_expert

        # Re-add the always-active shared experts (last `num_shared_experts` indices, weight 1.0).
        if self.num_shared_experts > 0:
            expert_weights = F.pad(expert_weights, (0, self.num_shared_experts), value=1.0)
            shared_expert_indices = (
                torch.arange(num_non_shared_experts, self.num_experts, device=expert_indices.device)
                .view(1, 1, self.num_shared_experts)
                .expand(expert_indices.size(0), expert_indices.size(1), self.num_shared_experts)
            )
            expert_indices = torch.cat([expert_indices, shared_expert_indices], dim=-1)
            # Shared experts process every token.
            tot_batch_size_per_expert = F.pad(
                tot_batch_size_per_expert, (0, self.num_shared_experts), value=x.size(0) * x.size(1)
            )

        return expert_weights, expert_indices, tot_batch_size_per_expert, aux_loss

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, min_document_expert_pool={self.min_document_expert_pool}, "
            f"max_document_expert_pool={self.max_document_expert_pool}, "
            f"eval_document_expert_pool={self.eval_document_expert_pool}, "
            f"num_shared_experts={self.num_shared_experts}, "
            f"global_load_balancing={self.global_load_balancing}"
        )
