import logging
from typing import Literal

import torch

log = logging.getLogger(__name__)

try:
    import fla
except Exception as e:  # noqa: BLE001 -- see below
    # Deliberately broader than `ImportError`. `fla` is optional (hence `has_fla()`), but
    # until now only an *absent* install degraded gracefully -- a *broken* one took down
    # the entire `olmo_core` package import, because `import fla` pulls in transformers
    # and thence `torchaudio`, whose `torch.ops.load_library` raises `OSError` when its
    # compiled extension does not match the installed torch:
    #
    #   OSError: .../libtorchaudio.abi3.so: undefined symbol: torch_dtype_float4_e2m1fn_x2
    #
    # That killed every job on an otherwise fine Beaker image, in `import olmo_core`, for
    # a dependency nothing in the Molmo2 stack uses. A model that genuinely needs fla
    # still fails loudly: `has_fla()` returns False and the dispatch helpers assert on it.
    log.warning(
        "flash-linear-attention is installed but failed to import (%s: %s)", type(e).__name__, e
    )
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
