from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


_CUDA_EXTENSION = None
_CUDA_EXTENSION_ATTEMPTED = False
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


@dataclass
class _PermuteDropWorkspace:
    sorted_indices: torch.Tensor
    row_id: torch.Tensor
    sorted_row_id: torch.Tensor
    temp_storage: torch.Tensor
    max_expanded_rows: int


_WORKSPACE_BY_DEVICE: dict[Tuple[str, int], _PermuteDropWorkspace] = {}


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ATTEMPTED
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ATTEMPTED and _CUDA_EXTENSION is None:
        raise RuntimeError("CUDA moe_permute_drop extension is unavailable") from _CUDA_EXTENSION_ERROR

    _CUDA_EXTENSION_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load

        this_dir = Path(__file__).resolve().parent
        cpp_src = this_dir / "cuda" / "moe_permute_drop.cpp"
        cu_src = this_dir / "cuda" / "moe_permute_drop_kernel.cu"
        _CUDA_EXTENSION = load(
            name="olmo_moe_permute_drop_ext",
            sources=[str(cpp_src), str(cu_src)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(f"Failed to build/load CUDA moe_permute_drop extension: {e}") from e

    return _CUDA_EXTENSION


def _workspace_key(device: torch.device) -> Tuple[str, int]:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return (device.type, idx)


def _ensure_workspace(
    *,
    device: torch.device,
    expanded_rows: int,
) -> _PermuteDropWorkspace:
    ext = _load_cuda_extension()
    key = _workspace_key(device)
    ws = _WORKSPACE_BY_DEVICE.get(key)
    if ws is not None and ws.max_expanded_rows >= expanded_rows:
        return ws

    temp_storage_bytes = int(ext.moe_permute_drop_temp_storage_bytes(expanded_rows))
    sorted_indices = torch.empty((expanded_rows,), device=device, dtype=torch.int32)
    row_id = torch.arange(expanded_rows, device=device, dtype=torch.int32)
    sorted_row_id = torch.empty((expanded_rows,), device=device, dtype=torch.int32)
    temp_storage = torch.empty((temp_storage_bytes,), device=device, dtype=torch.uint8)
    ws = _PermuteDropWorkspace(
        sorted_indices=sorted_indices,
        row_id=row_id,
        sorted_row_id=sorted_row_id,
        temp_storage=temp_storage,
        max_expanded_rows=expanded_rows,
    )
    _WORKSPACE_BY_DEVICE[key] = ws
    return ws


def moe_permute_drop_fwd(
    *,
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    requested_offsets: torch.Tensor,
    keep_offsets: torch.Tensor,
    keep_splits: torch.Tensor,
    num_out_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inp.ndim != 2:
        raise ValueError(f"Expected rank-2 input, got shape={tuple(inp.shape)}")
    if routing_map.ndim != 2:
        raise ValueError(f"Expected rank-2 routing_map, got shape={tuple(routing_map.shape)}")
    if routing_map.shape[0] != inp.shape[0]:
        raise ValueError(
            f"routing_map/input rows mismatch: routing_map={routing_map.shape[0]} input={inp.shape[0]}"
        )
    if requested_offsets.ndim != 1:
        raise ValueError(f"Expected rank-1 requested_offsets, got shape={tuple(requested_offsets.shape)}")
    if keep_offsets.ndim != 1:
        raise ValueError(f"Expected rank-1 keep_offsets, got shape={tuple(keep_offsets.shape)}")
    if keep_splits.ndim != 1:
        raise ValueError(f"Expected rank-1 keep_splits, got shape={tuple(keep_splits.shape)}")
    if requested_offsets.numel() != keep_offsets.numel() or requested_offsets.numel() != keep_splits.numel():
        raise ValueError(
            "requested_offsets/keep_offsets/keep_splits size mismatch: "
            f"{requested_offsets.numel()} vs {keep_offsets.numel()} vs {keep_splits.numel()}"
        )
    if not inp.is_cuda:
        raise ValueError("moe_permute_drop_fwd expects CUDA input")

    expanded_rows = int(routing_map.numel())
    ws = _ensure_workspace(device=inp.device, expanded_rows=expanded_rows)
    ext = _load_cuda_extension()
    return ext.moe_permute_drop_fwd_cuda(
        inp,
        routing_map if routing_map.dtype == torch.int32 else routing_map.to(dtype=torch.int32),
        requested_offsets if requested_offsets.dtype == torch.long else requested_offsets.to(dtype=torch.long),
        keep_offsets if keep_offsets.dtype == torch.long else keep_offsets.to(dtype=torch.long),
        keep_splits if keep_splits.dtype == torch.long else keep_splits.to(dtype=torch.long),
        int(num_out_tokens),
        ws.sorted_indices,
        ws.row_id,
        ws.sorted_row_id,
        ws.temp_storage,
        out,
    )

