from typing import Literal

import torch

try:
    import fla
except ImportError:
    fla = None


def has_fla() -> bool:
    """Check if flash-linear-attention (fla) is installed."""
    return fla is not None


def dispatch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | torch.Tensor | None = None,
) -> torch.Tensor:
    assert has_fla()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule(  # type: ignore[reportCallIssue]
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


def dispatch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    backend: Literal["triton", "cuda"] = "triton",
    cu_seqlens: torch.LongTensor | torch.Tensor | None = None,
) -> torch.Tensor:
    assert has_fla()
    from fla.modules.convolution import causal_conv1d

    return causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        backend=backend,
        cu_seqlens=cu_seqlens,
    )
