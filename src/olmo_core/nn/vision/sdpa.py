"""Memory-efficient SDPA helpers for vision modules (ViT + connector)."""

from contextlib import contextmanager
from typing import Iterator, List, Optional

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None  # type: ignore
    sdpa_kernel = None  # type: ignore


def vision_sdpa_backends(*, cudnn_first: bool = False) -> List:
    """
    Backends matching mm_olmo ViT (flash + efficient; exclude cuDNN math pitfalls).

    :param cudnn_first: Put cuDNN's attention ahead of them (see
        :func:`vision_scaled_dot_product_attention`).
    """
    if SDPBackend is None:
        return []
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    return [SDPBackend.CUDNN_ATTENTION, *backends] if cudnn_first else backends


CUDNN_FIRST_MIN_VERSION = 91300
"""The oldest cuDNN that :func:`vision_scaled_dot_product_attention` puts first on SM100. cuDNN
9.10.2 builds no execution plan there for the ViT's attention (16 heads of 72, bfloat16, with or
without gradients), and PyTorch does not fall back; 9.13.0 (the training image's runtime) and 9.20.0
run all of it (B300, 2026-10-18)."""


def cudnn_runtime_version() -> Optional[int]:
    """
    The loaded cuDNN's version in :func:`torch.backends.cudnn.version`'s encoding (9.13.0 is
    91300), or ``None`` without cuDNN.

    Read from the library itself: :func:`torch.backends.cudnn.version` raises when the loaded cuDNN
    is older than the one PyTorch was built against, as in the training image (built against 9.20,
    loads 9.13).
    """
    if not torch.backends.cudnn.is_available():
        return None
    major, minor, patch = torch._C._cudnn.getRuntimeVersion()
    return major * 10_000 + minor * 100 + patch


def cudnn_first_supported(device: torch.device) -> bool:
    """Whether cuDNN's attention goes first on ``device``: an SM100 or newer GPU with a cuDNN of at
    least :data:`CUDNN_FIRST_MIN_VERSION`."""
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 10
        and (cudnn_runtime_version() or 0) >= CUDNN_FIRST_MIN_VERSION
    )


@contextmanager
def vision_sdpa_context(*, cudnn_first: bool = False) -> Iterator[None]:
    backends = vision_sdpa_backends(cudnn_first=cudnn_first)
    if backends and sdpa_kernel is not None:
        with sdpa_kernel(backends, set_priority=cudnn_first):
            yield
    else:
        yield


def vision_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    cudnn_first_on_sm100: bool = False,
) -> torch.Tensor:
    """
    :func:`torch.nn.functional.scaled_dot_product_attention` on the backends of
    :func:`vision_sdpa_backends`.

    :param cudnn_first_on_sm100: On Blackwell (SM100 and newer) GPUs, try cuDNN's attention
        first. PyTorch's flash attention is slow there for the ViT's head dimension (72); cuDNN
        runs the ViT's unmasked self-attention in about a third of the time with the same accuracy
        against a float32 reference. Older cuDNNs are skipped (:func:`cudnn_first_supported`).
    """
    cudnn_first = cudnn_first_on_sm100 and cudnn_first_supported(q.device)
    with vision_sdpa_context(cudnn_first=cudnn_first):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
