"""Expert-parallel backend interface and backend selection.

The backend owns the scheduling of routed and shared expert work. Concrete implementations may
split the work into dispatch / experts / combine phases or keep it fused when required by their
transport and autograd implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch

from .ep_config import ExpertParallelPath

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


@dataclass(frozen=True)
class PreparedMoEForward:
    """Model-space state shared by expert-parallel forward paths."""

    attention_residual: torch.Tensor
    model_input: torch.Tensor


@dataclass(frozen=True)
class RoutedTokens:
    """Router outputs consumed by an expert-parallel backend."""

    expert_weights: torch.Tensor
    expert_indices: torch.Tensor
    tokens_per_expert: torch.Tensor
    aux_loss_info: Optional[tuple[object, ...]]


def prepare_and_route_moe(
    block: "OLMoDDPTransformerBlock",
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]],
    forward_kwargs: dict[str, object],
) -> tuple[PreparedMoEForward, RoutedTokens]:
    """Run the common attention/norm prologue and routed-expert router."""

    assert block.routed_experts_router is not None
    attention_residual = block._checkpointed_res_norm_attn(x, **forward_kwargs)
    forward_kwargs.pop("max_doc_len", None)
    forward_kwargs.pop("cu_doc_lens", None)
    model_input = block._prepare_moe_input(attention_residual)
    expert_weights, expert_indices, tokens_per_expert, aux_loss_info = (
        block.routed_experts_router(
            model_input,
            False,
            loss_div_factor=loss_div_factor,
        )
    )
    return (
        PreparedMoEForward(attention_residual=attention_residual, model_input=model_input),
        RoutedTokens(
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            tokens_per_expert=tokens_per_expert,
            aux_loss_info=aux_loss_info,
        ),
    )


def finish_moe_forward(
    block: "OLMoDDPTransformerBlock",
    prepared: PreparedMoEForward,
    mlp_output: torch.Tensor,
    routing: RoutedTokens,
) -> torch.Tensor:
    """Apply the common residual/norm epilogue and routed auxiliary loss."""

    final_output = block._res_norm_mlp(prepared.attention_residual, mlp_output)
    return block._attach_routed_aux_loss(final_output, routing.aux_loss_info)


class ExpertParallelBackend(ABC):
    """Strategy interface for an expert-parallel MoE execution path."""

    path: ExpertParallelPath

    @abstractmethod
    def run_moe(
        self,
        block: "OLMoDDPTransformerBlock",
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the backend's complete MoE schedule and return the block output."""


class Sync1DBackend(ExpertParallelBackend):
    path = ExpertParallelPath.sync_1d

    def run_moe(self, block, x, *, loss_div_factor=None, **kwargs):
        return block.combined_forward_ep_1d(
            x, loss_div_factor=loss_div_factor, **kwargs
        )


class NoSync1DBackend(ExpertParallelBackend):
    path = ExpertParallelPath.no_sync_1d

    def run_moe(self, block, x, *, loss_div_factor=None, **kwargs):
        # Imported lazily to keep the backend contract independent from concrete communication
        # modules and to avoid a block/backend/module import cycle.
        from .ep_no_sync_1d import combined_forward_ep_no_sync_1d

        return combined_forward_ep_no_sync_1d(
            block,
            x, loss_div_factor=loss_div_factor, **kwargs
        )


class RowwiseNVSHMEMBackend(ExpertParallelBackend):
    path = ExpertParallelPath.rowwise_nvshmem

    def run_moe(self, block, x, *, loss_div_factor=None, **kwargs):
        return block.combined_forward_ep_no_sync_rowwise(
            x, loss_div_factor=loss_div_factor, **kwargs
        )


class RowwiseWaveBackend(ExpertParallelBackend):
    path = ExpertParallelPath.rowwise_wave

    def run_moe(self, block, x, *, loss_div_factor=None, **kwargs):
        return block.combined_forward_ep_no_sync_rowwise_wave(
            x, loss_div_factor=loss_div_factor, **kwargs
        )


class DeepEPV2Backend(ExpertParallelBackend):
    path = ExpertParallelPath.deepep_v2

    def run_moe(self, block, x, *, loss_div_factor=None, **kwargs):
        return block.combined_forward_ep_deepep_v2(
            x, loss_div_factor=loss_div_factor, **kwargs
        )


_BACKENDS: Dict[ExpertParallelPath, ExpertParallelBackend] = {
    backend.path: backend
    for backend in (
        Sync1DBackend(),
        NoSync1DBackend(),
        RowwiseNVSHMEMBackend(),
        RowwiseWaveBackend(),
        DeepEPV2Backend(),
    )
}


def get_expert_parallel_backend(path: ExpertParallelPath) -> ExpertParallelBackend:
    """Return the stateless backend strategy for ``path``."""

    try:
        return _BACKENDS[ExpertParallelPath(path)]
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Unsupported expert-parallel path {path!r}") from e
