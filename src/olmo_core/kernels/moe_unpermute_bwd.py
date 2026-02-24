from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .cuda_extension_utils import load_cuda_extension


_CUDA_EXTENSION = None
_CUDA_EXTENSION_ATTEMPTED = False
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ATTEMPTED
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_ATTEMPTED and _CUDA_EXTENSION is None:
        raise RuntimeError("CUDA moe_unpermute_bwd extension is unavailable") from _CUDA_EXTENSION_ERROR

    _CUDA_EXTENSION_ATTEMPTED = True
    try:
        this_dir = Path(__file__).resolve().parent
        cpp_src = this_dir / "cuda" / "moe_unpermute_bwd.cpp"
        cu_src = this_dir / "cuda" / "moe_unpermute_bwd_kernel.cu"
        _CUDA_EXTENSION = load_cuda_extension(
            base_name="olmo_moe_unpermute_bwd_ext",
            sources=[cpp_src, cu_src],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose_env_names=("OLMO_MOE_UNPERMUTE_BWD_VERBOSE", "OLMO_MOE_CUDA_EXT_VERBOSE"),
            force_rebuild_env_names=(
                "OLMO_MOE_UNPERMUTE_BWD_FORCE_REBUILD",
                "OLMO_MOE_CUDA_EXT_FORCE_REBUILD",
            ),
            stale_lock_timeout_env_names=("OLMO_MOE_EXT_STALE_LOCK_TIMEOUT_SEC",),
            with_arch_suffix=True,
        )
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(f"Failed to build/load CUDA moe_unpermute_bwd extension: {e}") from e

    return _CUDA_EXTENSION


def _check_inputs(
    grad_output: torch.Tensor,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    keep_mask: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
) -> None:
    if grad_output.ndim != 2:
        raise ValueError(f"Expected rank-2 grad_output, got shape={tuple(grad_output.shape)}")
    if input_fwd.ndim != 2:
        raise ValueError(f"Expected rank-2 input_fwd, got shape={tuple(input_fwd.shape)}")
    if row_id_map.ndim != 1:
        raise ValueError(f"Expected rank-1 row_id_map, got shape={tuple(row_id_map.shape)}")
    if probs.ndim != 2:
        raise ValueError(f"Expected rank-2 probs, got shape={tuple(probs.shape)}")
    if not grad_output.is_cuda:
        raise ValueError("moe_unpermute_bwd expects CUDA grad_output")
    if not input_fwd.is_cuda or not row_id_map.is_cuda or not probs.is_cuda:
        raise ValueError("All moe_unpermute_bwd inputs must be CUDA tensors")
    if grad_output.device != input_fwd.device:
        raise ValueError(
            f"grad_output/input_fwd device mismatch: grad_output={grad_output.device} input_fwd={input_fwd.device}"
        )
    if row_id_map.device != grad_output.device:
        raise ValueError(
            f"row_id_map/grad_output device mismatch: row_id_map={row_id_map.device} grad_output={grad_output.device}"
        )
    if probs.device != grad_output.device:
        raise ValueError(
            f"probs/grad_output device mismatch: probs={probs.device} grad_output={grad_output.device}"
        )
    if grad_output.dtype != input_fwd.dtype:
        raise ValueError(
            f"grad_output/input_fwd dtype mismatch: grad_output={grad_output.dtype} input_fwd={input_fwd.dtype}"
        )
    if grad_output.shape[1] != input_fwd.shape[1]:
        raise ValueError(
            "grad_output/input_fwd hidden size mismatch: "
            f"grad_output={grad_output.shape[1]} input_fwd={input_fwd.shape[1]}"
        )

    num_tokens = probs.shape[0]
    topk = probs.shape[1]
    if grad_output.shape[0] != num_tokens:
        raise ValueError(
            f"grad_output/probs rows mismatch: grad_output={grad_output.shape[0]} probs={num_tokens}"
        )
    if row_id_map.numel() != num_tokens * topk:
        raise ValueError(
            "row_id_map/probs size mismatch: "
            f"row_id_map={row_id_map.numel()} probs={num_tokens}x{topk}"
        )
    if keep_mask is not None:
        if keep_mask.ndim != 1:
            raise ValueError(f"Expected rank-1 keep_mask, got shape={tuple(keep_mask.shape)}")
        if keep_mask.numel() != input_fwd.shape[0]:
            raise ValueError(
                f"keep_mask/input_fwd rows mismatch: keep_mask={keep_mask.numel()} input_fwd={input_fwd.shape[0]}"
            )
        if keep_mask.device != input_fwd.device:
            raise ValueError(
                f"keep_mask/input_fwd device mismatch: keep_mask={keep_mask.device} input_fwd={input_fwd.device}"
            )

    if out is not None:
        expected_shape = input_fwd.shape
        if tuple(out.shape) != tuple(expected_shape):
            raise ValueError(f"out shape mismatch: expected={tuple(expected_shape)} got={tuple(out.shape)}")
        if out.dtype != input_fwd.dtype:
            raise ValueError(f"out dtype mismatch: out={out.dtype} input_fwd={input_fwd.dtype}")
        if out.device != input_fwd.device:
            raise ValueError(f"out device mismatch: out={out.device} input_fwd={input_fwd.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")


@torch.compiler.disable
def moe_unpermute_bwd(
    *,
    grad_output: torch.Tensor,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    keep_mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(grad_output, input_fwd, row_id_map, probs, keep_mask, out)

    grad_output_in = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    input_fwd_in = input_fwd if input_fwd.is_contiguous() else input_fwd.contiguous()
    row_id_map_i32 = row_id_map if row_id_map.dtype == torch.int32 else row_id_map.to(dtype=torch.int32)
    probs_f32 = probs if probs.dtype == torch.float32 else probs.to(dtype=torch.float32)
    keep_mask_bool = None
    if keep_mask is not None:
        keep_mask_bool = keep_mask if keep_mask.dtype == torch.bool else keep_mask.to(dtype=torch.bool)
    row_id_map_i32 = row_id_map_i32 if row_id_map_i32.is_contiguous() else row_id_map_i32.contiguous()
    probs_f32 = probs_f32 if probs_f32.is_contiguous() else probs_f32.contiguous()
    if keep_mask_bool is not None and not keep_mask_bool.is_contiguous():
        keep_mask_bool = keep_mask_bool.contiguous()

    ext = _load_cuda_extension()
    out1, out2 = ext.moe_unpermute_bwd_cuda(
        grad_output_in,
        input_fwd_in,
        row_id_map_i32,
        probs_f32,
        keep_mask_bool,
        out,
    )
    return out1, out2
