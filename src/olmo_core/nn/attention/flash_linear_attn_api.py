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


def dispatch_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    cu_seqlens: torch.LongTensor | torch.Tensor | None = None,
    use_cute_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch Moonshot's pinned Triton KDA training kernel lazily.

    With ``use_cute_kernel=True``, calls the **experimental** CuTe/Triton kernels vendored
    in :mod:`olmo_core.nn.attention.kda_cute` instead. That entry point is a drop-in with
    this exact signature and forwards any call it does not support (packed documents,
    non-Blackwell, off-shape, graph capture) to FLA itself — so there is no predicate to
    check here, and the branch below is only about not importing the kernels at all when
    the flag is off. Those kernels are not numerically identical to FLA's, so opt in only
    when you are deliberately testing them.
    """
    assert has_fla()
    if use_cute_kernel:
        from olmo_core.nn.attention.kda_cute import cute_chunk_kda

        return cute_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            cu_seqlens=cu_seqlens,
        )

    from fla.ops.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
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
