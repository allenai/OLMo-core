from typing import Literal

import torch

try:
    import fla
except ImportError:
    fla = None

try:
    # Cheap on a machine with no GPU: the package imports nothing (no torch, triton, or the
    # CuTe DSL) until one of its op families is actually used.
    import kernel_fun
except ImportError:
    kernel_fun = None


def has_fla() -> bool:
    """Check if flash-linear-attention (fla) is installed."""
    return fla is not None


def has_kernel_fun() -> bool:
    """Check if ``kernel-fun`` is installed.

    ``kernel-fun`` ships CuTe/Triton drop-ins for FLA's KDA chunk kernel and short
    convolution. Install it with the ``kernel-fun`` extra:
    ``pip install 'ai2-olmo-core[kernel-fun]'``.
    """
    return kernel_fun is not None


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

    With ``use_cute_kernel=True``, calls the **experimental** CuTe/Triton kernels from the
    ``kernel-fun`` package (:func:`kernel_fun.kda.chunk_kda`) instead. That entry point is a
    drop-in with this exact signature and forwards any call it does not support (packed
    documents, non-Blackwell, off-shape, graph capture) to FLA itself — so there is no
    predicate to check here, and the branch below is only about not importing the kernels
    at all when the flag is off. Those kernels are not numerically identical to FLA's, so
    opt in only when you are deliberately testing them. ``KERNEL_FUN_DISABLE=1`` (or
    ``KERNEL_FUN_KDA_DISABLE=1``) forces FLA everywhere without a config change.
    """
    assert has_fla()
    if use_cute_kernel:
        if not has_kernel_fun():
            raise RuntimeError(
                "use_cute_kernel=True requires the kernel-fun package; "
                "install it with the 'kernel-fun' extra: pip install 'ai2-olmo-core[kernel-fun]'"
            )
        from kernel_fun.kda import chunk_kda as cute_chunk_kda

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
    use_cute_kernel: bool = False,
) -> torch.Tensor:
    """Dispatch FLA's short convolution lazily.

    With ``use_cute_kernel=True``, calls the **experimental** fused short-conv kernels from
    the ``kernel-fun`` package (:func:`kernel_fun.cconv.causal_conv1d`) instead. That entry
    point is a drop-in with this exact signature and return contract, and forwards any call
    it does not support (bias, packed-document ``cu_seqlens``, ``activation=None``,
    ``backend="cuda"``, an unsupported device) to FLA itself — so there is no predicate to
    check here, and the branch below is only about not importing the kernels at all when the
    flag is off. The package logs once per process whether its kernels engaged. This is the
    same flag that selects the CuTe KDA kernels in :func:`dispatch_chunk_kda`.
    ``KERNEL_FUN_DISABLE=1`` (or ``KERNEL_FUN_CCONV_DISABLE=1``) forces FLA everywhere
    without a config change.
    """
    assert has_fla()
    if use_cute_kernel:
        if not has_kernel_fun():
            raise RuntimeError(
                "use_cute_kernel=True requires the kernel-fun package; "
                "install it with the 'kernel-fun' extra: pip install 'ai2-olmo-core[kernel-fun]'"
            )
        from kernel_fun.cconv import causal_conv1d
    else:
        from fla.modules.convolution import causal_conv1d

    return causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        backend=backend,
        cu_seqlens=cu_seqlens,
    )
