from importlib.metadata import PackageNotFoundError, version
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


try:
    version("nvidia-cutlass-dsl-libs-cu13")
    _has_cuda13_cute = True
except PackageNotFoundError:
    _has_cuda13_cute = False


def require_kernel_fun() -> None:
    """Validate the optional KDA backend before launching kernels or tracing it."""
    if kernel_fun is None:
        raise RuntimeError(
            "use_experimental_kernels=True requires the kernel-fun package; "
            "install it with: pip install 'ai2-olmo-core[kernel-fun]'"
        )
    if not _has_cuda13_cute:
        raise RuntimeError(
            "Experimental KDA requires the CUDA 13 CuTe DSL libraries; the CUDA 12 "
            "compiler cannot lower its MMA backward. On a CUDA 13-compatible driver, "
            "install with: pip install 'ai2-olmo-core[kernel-fun]'"
        )


def has_fla() -> bool:
    """Check if flash-linear-attention (fla) is installed and usable.

    fla >= 0.5 imports ``triton`` at import time, so a CPU-only torch wheel that doesn't bundle
    triton can have fla installed yet unimportable. Treat a missing triton as fla being unavailable
    so callers skip rather than fail on ``import triton``.
    """
    if fla is None:
        return False
    from olmo_core.nn.moe.utils import has_triton

    return has_triton()


def has_kernel_fun() -> bool:
    """Check if ``kernel-fun`` is installed."""
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
    use_experimental_kernels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert has_fla()
    if use_experimental_kernels:
        require_kernel_fun()
        from kernel_fun.kda import chunk_kda as experimental_chunk_kda

        return experimental_chunk_kda(
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
    use_experimental_kernels: bool = False,
) -> torch.Tensor:
    assert has_fla()

    if use_experimental_kernels:
        if not has_kernel_fun():
            raise RuntimeError(
                "use_experimental_kernels=True requires the kernel-fun package; "
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
