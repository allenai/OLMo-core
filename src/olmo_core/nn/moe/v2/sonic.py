"""SonicMoE adapter for the non-expert-parallel MoE-v2 path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .routed_experts import RoutedExperts


def _prepare_sonic_inputs(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: RoutedExperts,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_tokens, top_k = expert_indices.shape
    token_indices = torch.arange(
        num_tokens,
        device=x.device,
        dtype=torch.int32,
    ).repeat_interleave(top_k)
    flat_expert_indices = expert_indices.reshape(-1).to(dtype=torch.int32).contiguous()
    router_scores = expert_weights.reshape(-1).to(dtype=torch.float32).contiguous()

    # Sonic's concatenated SwiGLU layout is [gate, up]. For fresh Sonic runs we deliberately
    # assign that meaning to the two identically initialized halves of OLMo's existing
    # ``w_up_gate`` parameter. No data is reordered: these permutes only expose Sonic's required
    # [2I, H, E] and [H, I, E] strided views.
    w1 = experts.w_up_gate.permute(1, 2, 0)
    w2 = experts.w_down.permute(2, 1, 0)
    return token_indices, flat_expert_indices, router_scores, w1, w2


@torch.compiler.disable
def sonic_moe_forward(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: RoutedExperts,
) -> torch.Tensor:
    """
    Execute routed experts with SonicMoE, including gather and weighted combine.

    This adapter is intentionally limited to fresh, BF16, bias-free, non-EP SwiGLU runs. The first
    half of :attr:`RoutedExperts.w_up_gate` is interpreted as the gate projection and the second as
    the up projection, which is the reverse of the ordinary OLMo grouped-MM backend. Checkpoints
    produced with this backend must therefore continue using this backend unless their two halves
    are converted explicitly.
    """

    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs

    token_indices, flat_expert_indices, router_scores, w1, w2 = _prepare_sonic_inputs(
        x,
        expert_indices,
        expert_weights,
        experts,
    )
    output, _expert_frequency = moe_general_routing_inputs(
        x=x,
        router_scores=router_scores,
        token_indices=token_indices,
        expert_indices=flat_expert_indices,
        w1=w1,
        b1=None,
        w2=w2,
        b2=None,
        E=experts.num_experts,
        stream_id=int(torch.cuda.current_stream(x.device).cuda_stream),
        activation_type=ActivationType.SWIGLU,
        is_inference_mode_enabled=not torch.is_grad_enabled(),
        concat_layout=True,
    )
    return output
