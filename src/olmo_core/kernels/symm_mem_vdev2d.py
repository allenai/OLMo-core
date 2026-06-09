from __future__ import annotations

import importlib
import os
from typing import Optional

import torch

try:
    import nvtx
except ImportError:
    from olmo_core._nvtx import nvtx

from .mxfp8_utils import quantize_rows_to_mxfp8, reduce_gathered_rows_from_mxfp8

_EXTENSION_MODULE_NAME = "olmo_core.kernels._symm_mem_vdev2d_ext_gpu"
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR: Optional[Exception] = None


def _load_cuda_extension():
    global _CUDA_EXTENSION
    global _CUDA_EXTENSION_ERROR
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    auto_build = os.getenv("OLMO_SYMM_VDEV2D_AUTO_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    build_backend = os.getenv("OLMO_SYMM_VDEV2D_BUILD_BACKEND", "cmake")
    if auto_build:
        try:
            from .build_symm_mem_vdev2d_ext import build_extension

            build_extension(
                inplace=True,
                verbose=False,
                force=False,
                backend=build_backend,
            )
        except Exception as e:
            _CUDA_EXTENSION_ERROR = e
            raise RuntimeError(f"Failed to auto-build CUDA symm_mem_vdev2d extension: {e}") from e

    try:
        _CUDA_EXTENSION = importlib.import_module(_EXTENSION_MODULE_NAME)
    except Exception as e:
        _CUDA_EXTENSION_ERROR = e
        raise RuntimeError(
            "GPU-side symm_mem_vdev2d extension is unavailable. Build it first with:\n"
            "  python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace --backend cmake\n"
            "Or set OLMO_SYMM_VDEV2D_AUTO_BUILD=1 to build automatically at import time."
        ) from e

    return _CUDA_EXTENSION


@torch.compiler.disable
def nvshmem_world_barrier() -> None:
    """Enqueue an NVSHMEM_TEAM_WORLD barrier on the current CUDA stream."""
    ext = _load_cuda_extension()
    ext.olmo_symm_world_barrier()


@torch.compiler.disable
def all_to_all_vdev_2d_nblocks(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    *,
    major_align: int = 1,
    nblocks: int = 0,
) -> None:
    """
    Variable-length 2D all-to-all over NVSHMEM symmetric memory.

    Sends each rank's ``input`` rows to peers per ``in_splits`` and writes the
    received rows into ``out`` at the positions described by ``out_splits_offsets``.

    :param input: This rank's send buffer (symmetric memory).
    :param out: This rank's receive buffer (symmetric memory).
    :param in_splits: Per-destination send counts.
    :param out_splits_offsets: Per-source receive counts + offsets into ``out``.
    :param group_name: The registered process-group name.
    :param major_align: Alignment (in rows) for the major dimension of the layout.
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    """
    ext = _load_cuda_extension()
    ext.all_to_all_vdev_2d_nblocks(
        input,
        out,
        in_splits,
        out_splits_offsets,
        group_name,
        major_align,
        nblocks,
    )


@torch.compiler.disable
def all_to_all_vdev_2d_offset_nblocks(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
) -> None:
    """
    Like :func:`all_to_all_vdev_2d_nblocks` but with precomputed send offsets.

    Takes ``in_splits_offsets`` (per-destination counts + offsets into ``input``)
    instead of plain splits, so the caller controls the exact source layout.

    :param input: This rank's send buffer (symmetric memory).
    :param out: This rank's receive buffer (symmetric memory).
    :param in_splits_offsets: Per-destination send counts + offsets into ``input``.
    :param out_splits_offsets: Per-source receive counts + offsets into ``out``.
    :param group_name: The registered process-group name.
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    """
    ext = _load_cuda_extension()
    ext.all_to_all_vdev_2d_offset_nblocks(
        input,
        out,
        in_splits_offsets,
        out_splits_offsets,
        group_name,
        nblocks,
    )


@torch.compiler.disable
def rowwise_dispatch_put(
    input: torch.Tensor,
    out: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
) -> None:
    """
    One-sided per-row dispatch of tokens to experts over NVSHMEM.

    Puts each row of ``input`` to its destination ``(dst_ranks[i], dst_rows[i])``
    slot in the peer's ``out`` buffer; rows with a negative destination are skipped.

    :param input: Local rows to dispatch.
    :param out: Destination buffer on the peer (symmetric memory).
    :param dst_ranks: Per-row destination rank.
    :param dst_rows: Per-row destination row index within the peer buffer.
    :param group_name: The registered process-group name.
    :param probs: Optional per-row routing weights to carry alongside.
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    :param pre_barrier: Barrier before the puts.
    :param post_barrier: Barrier after the puts (so receivers see completion).
    """
    ext = _load_cuda_extension()
    ext.rowwise_dispatch_put(
        input,
        out,
        dst_ranks,
        dst_rows,
        probs,
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_dispatch_put_scaled(
    input_hp: torch.Tensor,
    out_q: torch.Tensor,
    out_scales: torch.Tensor,
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    group_name: str,
    *,
    block_size: int = 32,
    nblocks: int = 0,
    pre_barrier: bool = False,
    post_barrier: bool = True,
) -> None:
    """
    MXFP8 variant of :func:`rowwise_dispatch_put`.

    Quantizes ``input_hp`` to rowwise MXFP8 (qdata + e8m0 block scales), then
    dispatches the quantized data and the scales as two back-to-back rowwise puts
    (into ``out_q`` and ``out_scales``), barriering only after the second.

    :param input_hp: High-precision local rows to quantize and dispatch.
    :param out_q: Destination FP8 qdata buffer on the peer.
    :param out_scales: Destination scales buffer on the peer.
    :param dst_ranks: Per-row destination rank.
    :param dst_rows: Per-row destination row index.
    :param group_name: The registered process-group name.
    :param block_size: MXFP8 block size (default 32).
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    :param pre_barrier: Barrier before the first put.
    :param post_barrier: Barrier after the scales put.
    """
    # Optional debug safety init for stale-capacity issues; off by default to
    # avoid full-buffer memset overhead in the fp8 hot path.
    if os.getenv("OLMO_ROWWISE_FP8_DISPATCH_INIT_OUT", "0") == "1":
        out_q.zero_()
        out_scales.fill_(1.0)

    qdata, scales = quantize_rows_to_mxfp8(input_hp, block_size=block_size)
    rowwise_dispatch_put(
        qdata,
        out_q,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=pre_barrier,
        post_barrier=False,
    )
    rowwise_dispatch_put(
        scales,
        out_scales,
        dst_ranks,
        dst_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=False,
        post_barrier=post_barrier,
    )


@torch.compiler.disable
def rowwise_combine_get(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    """
    One-sided per-row combine: gather expert outputs back and reduce into ``out``.

    Pulls each token's expert-output rows from ``(src_ranks, src_rows)`` and combines
    them (optionally weighted by ``probs``) into ``out`` — the inverse of dispatch.

    :param expert_out: Local expert-output buffer peers read from (symmetric memory).
    :param out: Per-token combined output (this rank).
    :param src_ranks: Per-row source rank to gather from.
    :param src_rows: Per-row source row index.
    :param group_name: The registered process-group name.
    :param probs: Optional per-row combine weights.
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    :param gathered_out: Optional buffer to also receive the raw gathered rows.
    :param pre_barrier: Barrier before the gets (so senders are ready).
    :param post_barrier: Barrier after the gets.
    """
    ext = _load_cuda_extension()
    ext.rowwise_combine_get(
        expert_out,
        out,
        src_ranks,
        src_rows,
        probs,
        group_name,
        nblocks,
        gathered_out,
        pre_barrier,
        post_barrier,
    )


@nvtx.annotate("rowwise_combine_get_scaled")
def rowwise_combine_get_scaled(
    expert_out_q: torch.Tensor,
    expert_out_scales: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    block_size: int = 32,
    nblocks: int = 0,
    gathered_out: Optional[torch.Tensor] = None,
    gathered_q_out: Optional[torch.Tensor] = None,
    gathered_scales_out: Optional[torch.Tensor] = None,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    """
    MXFP8 variant of :func:`rowwise_combine_get` for top-k routing.

    For ``[N, K]`` routing, gathers each token's K expert-output rows in MXFP8
    (qdata via :func:`rowwise_gather_get`, then the scales), masking invalid/dropped
    ``(src_ranks, src_rows)`` entries, then dequantizes and reduces the K rows
    (weighted by ``probs``) into ``out``.

    :param expert_out_q: Local FP8 expert-output qdata peers read from.
    :param expert_out_scales: Local expert-output block scales peers read from.
    :param out: Per-token combined high-precision output (this rank).
    :param src_ranks: ``[N, K]`` per-(token, slot) source rank (negative = dropped).
    :param src_rows: ``[N, K]`` per-(token, slot) source row index.
    :param group_name: The registered process-group name.
    :param probs: Optional ``[N, K]`` combine weights.
    :param block_size: MXFP8 block size (default 32).
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    :param gathered_out: Optional buffer for the dequantized gathered rows.
    :param gathered_q_out: Optional pre-allocated ``[N, K, ·]`` buffer for gathered qdata.
    :param gathered_scales_out: Optional pre-allocated ``[N, K, ·]`` buffer for gathered scales.
    :param pre_barrier: Barrier before the first gather.
    :param post_barrier: Barrier after the scales gather.
    """
    if src_ranks.ndim != 2 or src_rows.ndim != 2:
        raise ValueError(
            f"src_ranks/src_rows must be [N,K], got {tuple(src_ranks.shape)} and {tuple(src_rows.shape)}"
        )
    n, k = src_ranks.shape
    if src_rows.shape != src_ranks.shape:
        raise ValueError("src_ranks/src_rows shape mismatch")

    valid_rows = (src_rows >= 0) & (src_rows < expert_out_q.shape[0])
    valid = (src_ranks >= 0) & valid_rows
    safe_ranks = torch.where(valid, src_ranks, torch.full_like(src_ranks, -1))
    safe_rows = torch.where(valid, src_rows, torch.full_like(src_rows, -1))

    flat_ranks = safe_ranks.reshape(-1, 1).contiguous()
    flat_rows = safe_rows.reshape(-1, 1).contiguous()

    expected_q_shape = (n, k, expert_out_q.shape[1])
    if gathered_q_out is not None:
        if tuple(gathered_q_out.shape) != expected_q_shape:
            raise ValueError(
                f"gathered_q_out shape mismatch: expected {expected_q_shape}, got {tuple(gathered_q_out.shape)}"
            )
        if gathered_q_out.dtype != expert_out_q.dtype:
            raise ValueError(
                f"gathered_q_out dtype mismatch: expected {expert_out_q.dtype}, got {gathered_q_out.dtype}"
            )
        if gathered_q_out.device != out.device:
            raise ValueError(
                f"gathered_q_out device mismatch: expected {out.device}, got {gathered_q_out.device}"
            )
        if not gathered_q_out.is_contiguous():
            raise ValueError("gathered_q_out must be contiguous")
        gathered_q_3d = gathered_q_out
        gathered_q = gathered_q_3d.view(n * k, -1)
    else:
        gathered_q = torch.empty(
            (n * k, expert_out_q.shape[1]),
            device=out.device,
            dtype=expert_out_q.dtype,
        )
        gathered_q_3d = gathered_q.view(n, k, -1)

    rowwise_gather_get(
        expert_out_q,
        gathered_q,
        flat_ranks,
        flat_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=pre_barrier,
        post_barrier=False,
    )

    expected_scales_shape = (n, k, expert_out_scales.shape[1])
    if gathered_scales_out is not None:
        if tuple(gathered_scales_out.shape) != expected_scales_shape:
            raise ValueError(
                "gathered_scales_out shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(gathered_scales_out.shape)}"
            )
        if gathered_scales_out.dtype != expert_out_scales.dtype:
            raise ValueError(
                "gathered_scales_out dtype mismatch: "
                f"expected {expert_out_scales.dtype}, got {gathered_scales_out.dtype}"
            )
        if gathered_scales_out.device != out.device:
            raise ValueError(
                "gathered_scales_out device mismatch: "
                f"expected {out.device}, got {gathered_scales_out.device}"
            )
        if not gathered_scales_out.is_contiguous():
            raise ValueError("gathered_scales_out must be contiguous")
        gathered_scales_3d = gathered_scales_out
        gathered_scales = gathered_scales_3d.view(n * k, -1)
    else:
        gathered_scales = torch.empty(
            (n * k, expert_out_scales.shape[1]),
            device=out.device,
            dtype=expert_out_scales.dtype,
        )
        gathered_scales_3d = gathered_scales.view(n, k, -1)

    rowwise_gather_get(
        expert_out_scales,
        gathered_scales,
        flat_ranks,
        flat_rows,
        group_name,
        nblocks=nblocks,
        pre_barrier=False,
        post_barrier=post_barrier,
    )

    reduce_gathered_rows_from_mxfp8(
        gathered_q_3d,
        gathered_scales_3d,
        out,
        probs=probs,
        valid_mask=valid,
        block_size=block_size,
        gathered_out=gathered_out,
    )


@torch.compiler.disable
def rowwise_combine_get_fused(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    probs: Optional[torch.Tensor] = None,
    nblocks: int = 0,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    """
    Fused single-kernel variant of :func:`rowwise_combine_get`.

    Performs the gather and the weighted combine in one kernel launch (rather than a
    gather followed by a separate reduce), for the bf16/high-precision path.

    :param expert_out: Local expert-output buffer peers read from (symmetric memory).
    :param out: Per-token combined output (this rank).
    :param src_ranks: Per-row source rank to gather from.
    :param src_rows: Per-row source row index.
    :param group_name: The registered process-group name.
    :param probs: Optional per-row combine weights.
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    :param pre_barrier: Barrier before the gets.
    :param post_barrier: Barrier after the gets.
    """
    ext = _load_cuda_extension()
    ext.rowwise_combine_get_fused(
        expert_out,
        out,
        src_ranks,
        src_rows,
        probs,
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


@torch.compiler.disable
def rowwise_gather_get(
    expert_out: torch.Tensor,
    out: torch.Tensor,
    src_ranks: torch.Tensor,
    src_rows: torch.Tensor,
    group_name: str,
    *,
    nblocks: int = 0,
    pre_barrier: bool = True,
    post_barrier: bool = False,
) -> None:
    """
    One-sided per-row gather (no combine): pull source rows into ``out``.

    Like :func:`rowwise_combine_get` but only gathers — writes each gathered row to
    its own slot in ``out`` without reducing across them. Used as the gather half of
    :func:`rowwise_combine_get_scaled`.

    :param expert_out: Local buffer peers read from (symmetric memory).
    :param out: Destination buffer for the gathered rows (this rank).
    :param src_ranks: Per-row source rank.
    :param src_rows: Per-row source row index.
    :param group_name: The registered process-group name.
    :param nblocks: Number of CTA blocks to launch (0 = kernel default).
    :param pre_barrier: Barrier before the gets.
    :param post_barrier: Barrier after the gets.
    """
    ext = _load_cuda_extension()
    ext.rowwise_gather_get(
        expert_out,
        out,
        src_ranks,
        src_rows,
        group_name,
        nblocks,
        pre_barrier,
        post_barrier,
    )


__all__ = [
    "nvshmem_world_barrier",
    "all_to_all_vdev_2d_nblocks",
    "all_to_all_vdev_2d_offset_nblocks",
    "rowwise_dispatch_put",
    "rowwise_dispatch_put_scaled",
    "rowwise_combine_get",
    "rowwise_combine_get_scaled",
    "rowwise_combine_get_fused",
    "rowwise_gather_get",
]
