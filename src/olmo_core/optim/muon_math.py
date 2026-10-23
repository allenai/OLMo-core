"""Standard Muon math with AdamW-compatible update scaling.

https://arxiv.org/html/2502.16982v1#S2.SS2
Distributed integration lives in ``moe_optimizer``; these routines see complete
logical matrices, never flat optimizer shards.
"""

import math

import torch


def uses_muon(name: str, shape: tuple[int, ...]) -> bool:
    """Select hidden matrices; keep norms, embeddings, heads and KDA convs on AdamW."""
    parts = name.split(".")
    if any("norm" in part or "conv" in part for part in parts):
        return False
    if (
        "embeddings" in parts
        or "lm_head" in parts
        or name.endswith("bias")
        or parts[-1].startswith("b_")
    ):
        return False
    return len(shape) == 2 or (len(shape) == 3 and name.endswith(("w_up_gate", "w_down")))


def newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonalize independently over the last two axes with five iterations."""
    if matrix.ndim < 2:
        raise ValueError("Muon requires matrices")
    x = matrix.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    if x.is_cuda:
        x = x.bfloat16()
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.mT if transposed else x).float()


def muon_direction(name: str, direction: torch.Tensor, rms: float = 0.2) -> torch.Tensor:
    """Rescale each independent matrix; split packed SwiGLU up and gate projections."""
    if name.endswith("w_up_gate"):
        if direction.shape[-2] % 2:
            raise ValueError("Packed SwiGLU up/gate dimension must be even")
        up, gate = direction.chunk(2, dim=-2)
        return torch.cat((muon_direction("up", up, rms), muon_direction("gate", gate, rms)), dim=-2)
    scale = math.sqrt(max(direction.shape[-2:])) * rms
    # Bound Gram-matrix temporary memory for the many-expert case.
    if direction.ndim == 3:
        return torch.cat([newton_schulz(chunk) * scale for chunk in direction.split(16)], dim=0)
    return newton_schulz(direction) * scale
