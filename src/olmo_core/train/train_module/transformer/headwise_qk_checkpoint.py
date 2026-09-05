"""Strict, opt-in shared-to-per-head QK gain migration; never modifies the source checkpoint."""

import logging
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

log = logging.getLogger(__name__)


def prepare_qk_expansion(
    state: dict[str, Any], metadata: Metadata, gain_shapes: dict[str, tuple[int, ...]]
) -> dict[str, torch.Tensor]:
    """Replace only old shared QK gains/moments with tiny replicated load destinations.

    ``gain_shapes`` must come from the live model's two-dimensional Q/K RMSNorm weights.
    Optimizer checkpoint tensors are flattened. Every other tensor shape must match exactly.
    """
    expansions = {}
    for key, target in state.items():
        saved = metadata.state_dict_metadata.get(key)
        if not isinstance(target, torch.Tensor) or not isinstance(saved, TensorStorageMetadata):
            continue
        if tuple(saved.size) == tuple(target.shape):
            continue
        name, suffix = key.rsplit(".", 1)
        shape = gain_shapes.get(name)
        if (
            shape is None
            or len(shape) != 2
            or not name.endswith((".q_norm.weight", ".k_norm.weight"))
            or suffix not in ("main", "exp_avg", "exp_avg_sq")
            or tuple(saved.size) != (shape[1],)
            or tuple(target.shape) != (shape[0] * shape[1],)
        ):
            raise ValueError(
                f"Unsupported checkpoint shape change: {key}: {saved.size} -> {target.shape}"
            )
        source = torch.empty(tuple(saved.size), dtype=target.dtype, device=target.device)
        if isinstance(target, DTensor):
            source = DTensor.from_local(
                source,
                device_mesh=target.device_mesh,
                placements=[Replicate()] * target.device_mesh.ndim,
                run_check=False,
            )
        expansions[key] = target
        state[key] = source
        log.info("Expanding checkpoint %s: %s -> %s", key, saved.size, target.shape)
    return expansions


def finish_qk_expansion(state: dict[str, Any], expansions: dict[str, torch.Tensor]) -> None:
    """Repeat each loaded shared vector across heads and redistribute to the live state layout."""
    with torch.no_grad():
        for key, target in expansions.items():
            source = state[key]
            source = source.to_local() if isinstance(source, DTensor) else source
            expanded = source.repeat(target.numel() // source.numel()).reshape(target.shape)
            if isinstance(target, DTensor):
                expanded = distribute_tensor(
                    expanded, device_mesh=target.device_mesh, placements=target.placements
                )
            target.copy_(expanded)
            state[key] = target
